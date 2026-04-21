"""
WriteQueue — Serialized write access to the genome.

Problem: Multiple agents (Claude, Gemini, CpuTagger, ingest scripts)
all want to write to genome.db simultaneously. SQLite WAL mode handles
concurrent reads but writes serialize with busy-wait timeouts.

Solution: A single writer thread processes a queue of write operations.
All callers submit writes to the queue and get futures back. The writer
thread holds the only connection that does INSERT/UPDATE operations.

Reads go through the ReplicationManager's read-only replicas — no
contention with writes at all.

Usage:
    from helix_context.write_queue import GenomeWriter

    writer = GenomeWriter("genome.db")

    # Submit a gene for writing (non-blocking)
    future = writer.submit_gene(gene)
    gene_id = future.result(timeout=30)  # blocks until written

    # Submit a batch
    futures = writer.submit_genes(genes)

    # Graceful shutdown
    writer.close()
"""

from __future__ import annotations

import logging
import queue
import sqlite3
import threading
import time
from concurrent.futures import Future
from typing import List, Optional

log = logging.getLogger("helix.write_queue")


class GenomeWriter:
    """
    Single-writer thread for genome.db.

    All write operations go through this queue. The writer thread
    is the ONLY thing that opens genome.db in read-write mode.
    Everything else uses read-only replicas.
    """

    def __init__(self, db_path: str, batch_size: int = 50, flush_interval: float = 1.0):
        self.db_path = db_path
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._running = True
        self._stats = {"total_writes": 0, "total_batches": 0, "errors": 0}
        self._stats_lock = threading.Lock()

        # Start writer thread
        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name="genome-writer",
        )
        self._thread.start()
        log.info("GenomeWriter started (batch=%d, flush=%.1fs)", batch_size, flush_interval)

    def submit_sql(self, sql: str, params: tuple = (), future: Optional[Future] = None) -> Future:
        """Submit a raw SQL write operation."""
        if future is None:
            future = Future()
        self._queue.put(("sql", sql, params, future))
        return future

    def submit_gene_row(
        self,
        values: tuple,
        promoter_tags: List[tuple],
        fts_values: Optional[tuple] = None,
    ) -> Future:
        """
        Submit a gene upsert (INSERT OR REPLACE + promoter index + FTS5).

        Args:
            values: tuple of column values for genes table
            promoter_tags: list of (gene_id, tag_type, tag_value) tuples
            fts_values: optional (gene_id, content, complement) for FTS5
        """
        future = Future()
        self._queue.put(("gene", values, promoter_tags, fts_values, future))
        return future

    def flush(self) -> None:
        """Block until all pending writes are processed."""
        sentinel = Future()
        self._queue.put(("flush", sentinel))
        sentinel.result(timeout=60)

    def close(self) -> None:
        """Graceful shutdown — flush pending writes and stop."""
        self._running = False
        try:
            self.flush()
        except Exception:
            log.warning(
                "write_queue flush during close failed", exc_info=True,
            )
        self._thread.join(timeout=10)
        log.info("GenomeWriter stopped: %s", self._stats)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    # ── Writer thread ─────────────────────────────────────────────

    def _writer_loop(self) -> None:
        """Main writer loop — processes queue in batches."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")

        while self._running or not self._queue.empty():
            batch = []
            deadline = time.time() + self.flush_interval

            # Collect items up to batch_size or flush_interval
            while len(batch) < self.batch_size and time.time() < deadline:
                try:
                    item = self._queue.get(timeout=0.1)
                    batch.append(item)

                    # Flush sentinel — process immediately
                    if item[0] == "flush":
                        break
                except queue.Empty:
                    if not self._running:
                        break
                    continue

            if not batch:
                continue

            # Process batch in a single transaction
            try:
                self._process_batch(conn, batch)
            except Exception:
                log.warning("Write batch failed", exc_info=True)
                with self._stats_lock:
                    self._stats["errors"] += 1
                # Fail all futures in the batch
                for item in batch:
                    future = item[-1]
                    if isinstance(future, Future) and not future.done():
                        future.set_exception(RuntimeError("Batch write failed"))

        conn.close()

    def _process_batch(self, conn: sqlite3.Connection, batch: list) -> None:
        """Process a batch of write operations in a single transaction."""
        cur = conn.cursor()
        cur.execute("BEGIN IMMEDIATE")

        try:
            for item in batch:
                op = item[0]

                if op == "sql":
                    _, sql, params, future = item
                    try:
                        cur.execute(sql, params)
                        future.set_result(cur.lastrowid)
                    except Exception as e:
                        future.set_exception(e)

                elif op == "gene":
                    _, values, promoter_tags, fts_values, future = item
                    try:
                        gene_id = values[0]

                        # INSERT OR REPLACE gene
                        placeholders = ",".join("?" * len(values))
                        cur.execute(
                            f"INSERT OR REPLACE INTO genes VALUES ({placeholders})",
                            values,
                        )

                        # Rebuild promoter index
                        cur.execute(
                            "DELETE FROM promoter_index WHERE gene_id = ?",
                            (gene_id,),
                        )
                        if promoter_tags:
                            cur.executemany(
                                "INSERT INTO promoter_index VALUES (?, ?, ?)",
                                promoter_tags,
                            )

                        # FTS5 sync
                        if fts_values:
                            cur.execute(
                                "INSERT OR REPLACE INTO genes_fts"
                                "(gene_id, content, complement) VALUES (?, ?, ?)",
                                fts_values,
                            )

                        future.set_result(gene_id)
                    except Exception as e:
                        future.set_exception(e)

                elif op == "flush":
                    future = item[1]
                    # Will be resolved after commit

            conn.commit()

            # Resolve flush sentinels
            for item in batch:
                if item[0] == "flush":
                    future = item[1]
                    if not future.done():
                        future.set_result(True)

            with self._stats_lock:
                gene_count = sum(1 for item in batch if item[0] == "gene")
                self._stats["total_writes"] += gene_count
                self._stats["total_batches"] += 1

        except Exception:
            conn.rollback()
            raise
