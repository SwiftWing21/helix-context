"""
Replication — Distributed genome clones across drives.

Biology analogy:
    DNA replication creates identical copies of the genome for each
    daughter cell. Our replication creates read-only copies of genome.db
    across multiple drives so parallel agents can query without contention.

Architecture:
    - Master genome (F:/Projects/helix-context/genome.db) — receives all writes
    - Read replicas (C:/helix-cache/genome.db, E:/helix-cache/genome.db) — read-only
    - Delta-sync: WAL checkpoint + file copy every N inserts
    - Agents open replicas with ?mode=ro for zero write contention

Usage:
    from helix_context.replication import ReplicationManager

    mgr = ReplicationManager(
        master="F:/Projects/helix-context/genome.db",
        replicas=["C:/helix-cache/genome.db", "E:/helix-cache/genome.db"],
        sync_interval=100,  # sync every 100 inserts
    )

    # Called by genome.upsert_gene() after each write
    mgr.notify_write()  # increments counter, syncs when threshold reached

    # Manual sync
    mgr.sync_now()

    # Get a read-only connection to the best replica
    conn = mgr.get_reader()
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("helix.replication")


class ReplicationManager:
    """
    Manages read-only genome clones across multiple drives.

    The master genome receives all writes. After every `sync_interval`
    writes, a WAL checkpoint + file copy propagates changes to replicas.
    Replicas are opened read-only by query agents for zero contention.
    """

    def __init__(
        self,
        master: str,
        replicas: Optional[List[str]] = None,
        sync_interval: int = 100,
    ):
        self.master = os.path.abspath(master)
        self.replicas = [os.path.abspath(r) for r in (replicas or [])]
        self.sync_interval = sync_interval

        self._write_count = 0
        self._lock = threading.Lock()
        self._last_sync = 0.0
        self._sync_in_progress = False
        # Cached read-only connections per replica path (see get_reader/close).
        self._reader_cache: dict = {}
        self._reader_lock = threading.Lock()

        # Ensure replica directories exist
        for replica in self.replicas:
            replica_dir = os.path.dirname(replica)
            if replica_dir:
                os.makedirs(replica_dir, exist_ok=True)

        # Initial sync if replicas don't exist yet
        for replica in self.replicas:
            if not os.path.exists(replica):
                log.info("Initial replica sync: %s", replica)
                src = sqlite3.connect(self.master)
                self._backup_to(src, replica)
                src.close()

        if self.replicas:
            log.info(
                "ReplicationManager: master=%s, %d replicas, sync every %d writes",
                self.master, len(self.replicas), self.sync_interval,
            )

    def notify_write(self) -> None:
        """Called after each write to the master genome. Syncs when threshold reached."""
        with self._lock:
            self._write_count += 1
            if self._write_count >= self.sync_interval:
                self._write_count = 0
                # Sync in background thread to avoid blocking writes
                if not self._sync_in_progress:
                    self._sync_in_progress = True
                    t = threading.Thread(target=self._do_sync, daemon=True)
                    t.start()

    def sync_now(self) -> int:
        """
        Force an immediate sync to all replicas.
        Returns the number of replicas successfully updated.
        """
        with self._lock:
            self._write_count = 0
        return self._do_sync()

    def _do_sync(self) -> int:
        """Delta-sync master to all replicas using SQLite backup API.

        Uses sqlite3.Connection.backup() which copies only changed pages —
        true delta replication. For a 500MB genome with 100 new genes,
        this transfers ~1-5MB instead of the full file.
        """
        try:
            src = sqlite3.connect(self.master)

            synced = 0
            for replica in self.replicas:
                try:
                    self._backup_to(src, replica)
                    synced += 1
                except Exception:
                    log.warning("Failed to sync replica: %s", replica, exc_info=True)

            src.close()
            self._last_sync = time.time()
            log.info(
                "Genome replicated to %d/%d replicas (delta sync, %.1f MB master)",
                synced, len(self.replicas),
                os.path.getsize(self.master) / 1024 / 1024,
            )
            return synced

        except Exception:
            log.warning("Replication sync failed", exc_info=True)
            return 0
        finally:
            # Always clear the in-progress flag under the same lock used to
            # set it in notify_write() — otherwise a concurrent notify_write
            # could observe a stale value and skip a needed sync (or start a
            # duplicate thread).
            with self._lock:
                self._sync_in_progress = False

    def _backup_to(self, src: sqlite3.Connection, replica: str) -> None:
        """Delta-copy using SQLite backup API (only changed pages)."""
        t0 = time.time()
        dst = sqlite3.connect(replica)
        src.backup(dst, pages=256)  # Copy 256 pages per step (1MB chunks)
        dst.close()
        elapsed_ms = (time.time() - t0) * 1000
        log.debug("Replica sync to %s in %.0fms", replica, elapsed_ms)

    def _get_cached_reader(self, path: str) -> sqlite3.Connection:
        """Return a process-wide cached read-only connection for ``path``.

        Connections are opened with ``check_same_thread=False`` so the same
        cached handle is safe to share across reader threads (SQLite will
        serialize reads internally). Cached connections are closed via
        ``close()``.
        """
        with self._reader_lock:
            conn = self._reader_cache.get(path)
            if conn is not None:
                return conn
            conn = sqlite3.connect(
                f"file:{path}?mode=ro",
                uri=True,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            self._reader_cache[path] = conn
            return conn

    def get_reader(self) -> sqlite3.Connection:
        """
        Get a read-only connection to the best available replica.
        Falls back to master if no replicas exist.

        Connections are cached per path inside the ReplicationManager —
        callers MUST NOT close the returned connection. Use ``close()``
        on the manager to release all cached handles on shutdown.
        """
        # Prefer replicas (they don't contend with writes)
        for replica in self.replicas:
            if os.path.exists(replica):
                try:
                    return self._get_cached_reader(replica)
                except Exception:
                    log.warning(
                        "Failed to open replica reader: %s", replica, exc_info=True,
                    )
                    continue

        # Fall back to master (read-only mode)
        return self._get_cached_reader(self.master)

    def close(self) -> None:
        """Close all cached read-only connections."""
        with self._reader_lock:
            paths = list(self._reader_cache.keys())
            for path in paths:
                conn = self._reader_cache.pop(path, None)
                if conn is None:
                    continue
                try:
                    conn.close()
                except Exception:
                    log.warning(
                        "Failed to close cached reader: %s", path, exc_info=True,
                    )

    def status(self) -> dict:
        """Return replication status."""
        replica_status = []
        for replica in self.replicas:
            exists = os.path.exists(replica)
            size = os.path.getsize(replica) if exists else 0
            mtime = os.path.getmtime(replica) if exists else 0
            replica_status.append({
                "path": replica,
                "exists": exists,
                "size_mb": round(size / 1024 / 1024, 1),
                "age_s": round(time.time() - mtime, 1) if exists else -1,
            })

        master_size = os.path.getsize(self.master) if os.path.exists(self.master) else 0
        return {
            "master": self.master,
            "master_size_mb": round(master_size / 1024 / 1024, 1),
            "replicas": replica_status,
            "writes_since_sync": self._write_count,
            "sync_interval": self.sync_interval,
            "last_sync": self._last_sync,
        }
