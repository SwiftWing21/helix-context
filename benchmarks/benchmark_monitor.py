"""
Benchmark state monitor — detects the three failure modes we hit in practice:

  1. Dual-load VRAM pressure (wrong model loaded alongside the benchmark target)
  2. Hung benchmark process (httpx stall, proxy deadlock, network drop)
  3. Silent background contamination (another process writing to the snapshot DB
     mid-run, or a new model appearing in VRAM)

Design notes:
  - Config-driven: reads helix.toml to find the genome path, Ollama URL, etc.
    No hardcoded paths. Follows raude's temporary A/B switches automatically.
  - Progress-gated response: restart early (<N-weighted %), warn late.
  - Tiered VRAM detection: 90% = pressure (3 consecutive passes = warn),
    95% = saturation (immediate, progress-gated).
  - Does NOT spawn the benchmark itself. Detects and exits; operator relaunches
    (or wraps in a while-loop for full auto-restart).

Usage (integrated mode — inside bench_needle_1000.py):

    from benchmark_monitor import BenchmarkMonitor
    monitor = BenchmarkMonitor(
        benchmark_model=MODEL,
        incremental_output_path=incremental_path,
        total_needles=N_TOTAL,
    )
    monitor.preflight()             # blocks, aborts on fatal pre-flight failures
    monitor.start()                 # launches background thread
    # ... run benchmark ...
    monitor.stop()
    print(monitor.final_report())

Usage (standalone pre-flight check):

    python benchmarks/benchmark_monitor.py --model qwen3:8b --check-only
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import httpx


# Severity levels
SEV_CRITICAL = "critical"  # data integrity breach — always abort
SEV_HIGH = "high"          # progress-gated: restart early, warn late
SEV_MEDIUM = "medium"      # warn only, never restart
SEV_LOW = "low"            # informational


@dataclass
class Incident:
    timestamp: str
    check_number: int
    severity: str
    type: str
    action: str  # "warn", "restart_requested", "abort", "log"
    progress: float
    details: dict
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MonitorConfig:
    """Tunable knobs — all have sensible defaults."""

    # Timing
    check_interval_s: float = 90.0
    stall_threshold_s: float = 180.0

    # Progress-gated restart threshold
    protection_budget_needles: int = 150
    restart_threshold_min_pct: float = 0.05
    restart_threshold_max_pct: float = 0.30

    # VRAM tracking
    vram_warn_pct: float = 0.90
    vram_abort_pct: float = 0.95
    vram_warn_passes: int = 3

    # Snapshot integrity
    strict_hash: bool = False  # False = mtime+size only; True = SHA256


class BenchmarkMonitor:
    """Background thread that watches benchmark health and fails fast on violations."""

    def __init__(
        self,
        benchmark_model: str,
        incremental_output_path: str,
        total_needles: int,
        allowed_models: Optional[list[str]] = None,
        genome_snapshot_path: Optional[str] = None,
        helix_config_path: Optional[str] = None,
        helix_url: str = "http://127.0.0.1:11437",
        ollama_url: str = "http://localhost:11434",
        monitor_log_path: Optional[str] = None,
        config: Optional[MonitorConfig] = None,
    ):
        self.benchmark_model = benchmark_model.lower()
        self.allowed_models = [m.lower() for m in (allowed_models or [benchmark_model])]
        self.incremental_path = Path(incremental_output_path)
        self.total_needles = total_needles
        self.helix_url = helix_url.rstrip("/")
        self.ollama_url = ollama_url.rstrip("/")
        self.cfg = config or MonitorConfig()

        # Config-driven genome path — resolved from helix.toml unless overridden
        if genome_snapshot_path:
            self.genome_path = Path(genome_snapshot_path)
        else:
            self.genome_path = Path(self._resolve_genome_path(helix_config_path))

        # Monitor log file (one JSON line per check or incident)
        if monitor_log_path:
            self.monitor_log = Path(monitor_log_path)
        else:
            # Default: sibling of the incremental file with .monitor.jsonl suffix
            base = self.incremental_path.with_suffix("")
            self.monitor_log = base.with_suffix(".monitor.jsonl")

        # State
        self.incidents: list[Incident] = []
        self.check_count = 0
        self.vram_counter = 0
        self.snapshot_fingerprint: Optional[dict] = None
        self.last_jsonl_lines = 0
        self.last_jsonl_change_ts = time.time()
        self.bench_pid = os.getpid()
        self.start_ts = time.time()

        self.abort_flag = threading.Event()
        self.stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._client = httpx.Client(timeout=10)

    # ────────────────────────────────────────────────────────────────
    # Config resolution (no hardcoded paths)
    # ────────────────────────────────────────────────────────────────

    def _resolve_genome_path(self, config_path: Optional[str]) -> str:
        """Read helix.toml via load_config() to find the current genome path."""
        try:
            # Lazy import — avoids pulling in ΣĒMA/transformers at module load time
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from helix_context.config import load_config
            cfg = load_config(config_path) if config_path else load_config()
            return cfg.genome.path
        except Exception as e:
            self._print(f"[!] Could not resolve genome path from config: {e}")
            self._print("  Falling back to genome.db in current directory")
            return "genome.db"

    # ────────────────────────────────────────────────────────────────
    # Snapshot integrity
    # ────────────────────────────────────────────────────────────────

    def _snapshot_fingerprint(self) -> dict:
        """Compute a cheap (or strict) fingerprint of the genome snapshot."""
        if not self.genome_path.exists():
            return {"exists": False}
        st = self.genome_path.stat()
        fp = {
            "exists": True,
            "size": st.st_size,
            "mtime": st.st_mtime,
        }
        if self.cfg.strict_hash:
            # Full SHA256 — slow on 500MB+ files, opt-in only
            h = hashlib.sha256()
            with open(self.genome_path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            fp["sha256"] = h.hexdigest()
        return fp

    def _snapshot_changed(self) -> tuple[bool, dict]:
        """Detect genome snapshot tampering. Returns (changed, new_fingerprint)."""
        if self.snapshot_fingerprint is None:
            return False, self._snapshot_fingerprint()
        current = self._snapshot_fingerprint()
        if current != self.snapshot_fingerprint:
            return True, current
        return False, current

    # ────────────────────────────────────────────────────────────────
    # Ollama model state
    # ────────────────────────────────────────────────────────────────

    def _loaded_models(self) -> list[dict]:
        """Enumerate models currently resident in Ollama's VRAM pool."""
        try:
            r = self._client.get(f"{self.ollama_url}/api/ps", timeout=5)
            if r.status_code == 200:
                return r.json().get("models", [])
        except Exception:
            pass
        return []

    def _unauthorized_models(self, loaded: list[dict]) -> list[str]:
        """Return names of loaded models that aren't in the whitelist."""
        offending = []
        for m in loaded:
            name = m.get("name", "").lower()
            if name and not any(name.startswith(allowed) for allowed in self.allowed_models):
                offending.append(m.get("name"))
        return offending

    def _benchmark_model_loaded(self, loaded: list[dict]) -> bool:
        """Check whether the target benchmark model is resident."""
        for m in loaded:
            if m.get("name", "").lower().startswith(self.benchmark_model):
                return True
        return False

    # ────────────────────────────────────────────────────────────────
    # VRAM pressure tracking
    # ────────────────────────────────────────────────────────────────

    def _vram_state(self) -> Optional[dict]:
        """Read GPU memory usage via nvidia-smi. None if nvidia-smi not available."""
        try:
            import subprocess
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if r.returncode != 0:
                return None
            parts = [p.strip() for p in r.stdout.strip().split(",")]
            if len(parts) < 2:
                return None
            used_mb = float(parts[0])
            total_mb = float(parts[1])
            util_pct = float(parts[2]) if len(parts) > 2 else 0.0
            temp_c = float(parts[3]) if len(parts) > 3 else 0.0
            return {
                "used_mb": used_mb,
                "total_mb": total_mb,
                "used_pct": used_mb / max(total_mb, 1),
                "util_pct": util_pct,
                "temp_c": temp_c,
            }
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────────
    # Progress + stall detection
    # ────────────────────────────────────────────────────────────────

    def _completed_needles(self) -> int:
        """Count lines in the incremental JSONL."""
        if not self.incremental_path.exists():
            return 0
        try:
            with open(self.incremental_path, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def _progress(self) -> float:
        """Return completion ratio 0.0 – 1.0."""
        done = self._completed_needles()
        return done / max(self.total_needles, 1)

    def _restart_threshold_pct(self) -> float:
        """N-weighted threshold: max(5%, min(30%, 150/N))."""
        raw = self.cfg.protection_budget_needles / max(self.total_needles, 1)
        return max(
            self.cfg.restart_threshold_min_pct,
            min(self.cfg.restart_threshold_max_pct, raw),
        )

    def _is_early_run(self) -> bool:
        """True if progress is below the restart threshold."""
        return self._progress() < self._restart_threshold_pct()

    # ────────────────────────────────────────────────────────────────
    # Incident logging
    # ────────────────────────────────────────────────────────────────

    def _log_incident(
        self,
        severity: str,
        incident_type: str,
        action: str,
        details: dict,
        reason: str,
    ) -> Incident:
        incident = Incident(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            check_number=self.check_count,
            severity=severity,
            type=incident_type,
            action=action,
            progress=round(self._progress(), 4),
            details=details,
            reason=reason,
        )
        self.incidents.append(incident)

        # Append to monitor log
        try:
            self.monitor_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.monitor_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(incident.to_dict()) + "\n")
        except Exception as e:
            self._print(f"[!] Failed to write incident to {self.monitor_log}: {e}")

        # Console output (ASCII only — Windows default stdout is cp1252)
        sev_tag = {
            SEV_CRITICAL: "[!!!]",
            SEV_HIGH: "[!! ]",
            SEV_MEDIUM: "[!  ]",
            SEV_LOW: "[.  ]",
        }.get(severity, "[ . ]")
        self._print(f"{sev_tag} [{severity.upper()}] {incident_type}: {reason}")
        return incident

    def _log_heartbeat(self, payload: dict) -> None:
        """Append a check-time heartbeat line to the monitor log (no incident)."""
        try:
            self.monitor_log.parent.mkdir(parents=True, exist_ok=True)
            with open(self.monitor_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "type": "heartbeat",
                    "check_number": self.check_count,
                    **payload,
                }) + "\n")
        except Exception:
            pass

    def _print(self, msg: str) -> None:
        """Unbuffered console output for monitor messages."""
        print(f"[monitor] {msg}", flush=True)

    # ────────────────────────────────────────────────────────────────
    # Pre-flight check (blocks before benchmark starts)
    # ────────────────────────────────────────────────────────────────

    def preflight(self) -> bool:
        """
        Run all startup checks. Returns True if clean, False (and prints errors)
        if any fatal check failed. The caller should exit non-zero on False.

        Checks:
          1. Helix server is reachable
          2. Ollama is reachable
          3. Target benchmark model is loaded
          4. No unauthorized models are loaded
          5. Genome snapshot exists and is readable
        """
        self._print("=== PRE-FLIGHT CHECK ===")
        self._print(f"Benchmark model:   {self.benchmark_model}")
        self._print(f"Allowed models:    {self.allowed_models}")
        self._print(f"Genome path:       {self.genome_path}")
        self._print(f"Helix URL:         {self.helix_url}")
        self._print(f"Ollama URL:        {self.ollama_url}")
        self._print(f"Total needles:     {self.total_needles}")
        self._print(f"Restart threshold: {self._restart_threshold_pct()*100:.1f}% "
                    f"(~{int(self.total_needles * self._restart_threshold_pct())} needles)")

        fatal = []

        # 1. Helix server reachable?
        try:
            r = self._client.get(f"{self.helix_url}/health", timeout=5)
            if r.status_code != 200:
                fatal.append(f"Helix server returned HTTP {r.status_code}")
            else:
                health = r.json()
                self._print(f"[OK] Helix alive: {health.get('genes')} genes, "
                            f"ribosome={health.get('ribosome')}")
        except Exception as e:
            fatal.append(f"Helix server unreachable at {self.helix_url}: {e}")

        # 2. Ollama reachable?
        try:
            r = self._client.get(f"{self.ollama_url}/api/tags", timeout=5)
            if r.status_code != 200:
                fatal.append(f"Ollama returned HTTP {r.status_code}")
            else:
                self._print(f"[OK] Ollama alive")
        except Exception as e:
            fatal.append(f"Ollama unreachable at {self.ollama_url}: {e}")

        # 3 + 4. Loaded models
        loaded = self._loaded_models()
        loaded_names = [m.get("name") for m in loaded]
        self._print(f"Loaded models: {loaded_names or '(none)'}")

        if not self._benchmark_model_loaded(loaded):
            fatal.append(
                f"Benchmark model '{self.benchmark_model}' is not loaded. "
                f"Pre-load with: curl http://localhost:11434/api/generate "
                f"-d '{{\"model\":\"{self.benchmark_model}\",\"prompt\":\"hi\","
                f"\"stream\":false,\"keep_alive\":\"12h\",\"options\":{{\"num_predict\":1}}}}'"
            )

        unauthorized = self._unauthorized_models(loaded)
        if unauthorized:
            fatal.append(
                f"Unauthorized models loaded: {unauthorized}. "
                f"Unload with: curl -X POST {self.ollama_url}/api/generate "
                f"-d '{{\"model\":\"<name>\",\"keep_alive\":\"0s\"}}'"
            )

        # 5. Genome snapshot
        if not self.genome_path.exists():
            fatal.append(f"Genome snapshot missing: {self.genome_path}")
        else:
            size_mb = self.genome_path.stat().st_size / (1024 * 1024)
            self._print(f"[OK] Genome snapshot: {size_mb:.1f} MB at {self.genome_path}")
            self.snapshot_fingerprint = self._snapshot_fingerprint()

        # VRAM baseline (informational)
        vram = self._vram_state()
        if vram:
            self._print(f"[OK] VRAM baseline: {vram['used_mb']:.0f}/{vram['total_mb']:.0f} MB "
                        f"({vram['used_pct']*100:.1f}%), {vram['temp_c']:.0f}C")

        if fatal:
            self._print("")
            self._print("=== PRE-FLIGHT FAILED ===")
            for i, err in enumerate(fatal, 1):
                self._print(f"  {i}. {err}")
            self._print("")
            return False

        self._print("[OK] PRE-FLIGHT CLEAN - benchmark may proceed")
        self._print("")
        return True

    # ────────────────────────────────────────────────────────────────
    # Per-check logic
    # ────────────────────────────────────────────────────────────────

    def _run_check(self) -> None:
        """Execute all periodic checks. Sets self.abort_flag on fatal issues."""
        self.check_count += 1
        progress = self._progress()
        is_early = self._is_early_run()

        # Collect state
        vram = self._vram_state()
        loaded = self._loaded_models()
        unauthorized = self._unauthorized_models(loaded)
        snapshot_changed, snapshot_fp = self._snapshot_changed()
        completed = self._completed_needles()

        # Heartbeat
        self._log_heartbeat({
            "progress": round(progress, 4),
            "needles_done": completed,
            "needles_total": self.total_needles,
            "vram_used_pct": round(vram["used_pct"], 4) if vram else None,
            "vram_counter": self.vram_counter,
            "unauthorized_models": unauthorized,
            "is_early": is_early,
        })

        # ── CRITICAL: snapshot tampering (always abort) ─────────────
        if snapshot_changed:
            self._log_incident(
                severity=SEV_CRITICAL,
                incident_type="snapshot_tampered",
                action="abort",
                details={
                    "original": self.snapshot_fingerprint,
                    "current": snapshot_fp,
                },
                reason=(
                    "Genome snapshot file was modified mid-run. Benchmark integrity "
                    "is compromised — aborting regardless of progress."
                ),
            )
            self.abort_flag.set()
            return

        # ── FATAL: bench process dead ───────────────────────────────
        if not self._pid_alive():
            self._log_incident(
                severity=SEV_CRITICAL,
                incident_type="bench_pid_dead",
                action="abort",
                details={"pid": self.bench_pid},
                reason="Benchmark process is no longer running. Monitor exiting.",
            )
            self.abort_flag.set()
            return

        # ── HIGH: stall detection ───────────────────────────────────
        if completed > self.last_jsonl_lines:
            self.last_jsonl_lines = completed
            self.last_jsonl_change_ts = time.time()
        else:
            stall_s = time.time() - self.last_jsonl_change_ts
            if stall_s > self.cfg.stall_threshold_s:
                self._gated_incident(
                    incident_type="bench_stalled",
                    details={
                        "stall_seconds": round(stall_s, 1),
                        "last_needle": completed,
                        "threshold_s": self.cfg.stall_threshold_s,
                    },
                    reason_early=(
                        f"Benchmark stalled — no new results for {stall_s:.0f}s "
                        f"at needle {completed}/{self.total_needles}. Restart recommended."
                    ),
                    reason_late=(
                        f"Benchmark stalled at needle {completed}/{self.total_needles} "
                        f"({progress*100:.0f}% complete). Letting it continue — "
                        f"may be a slow broad-tier query."
                    ),
                )

        # ── HIGH: Helix server heartbeat ────────────────────────────
        helix_alive = True
        try:
            r = self._client.get(f"{self.helix_url}/health", timeout=5)
            helix_alive = r.status_code == 200
        except Exception:
            helix_alive = False

        if not helix_alive:
            self._gated_incident(
                incident_type="helix_server_down",
                details={"url": self.helix_url},
                reason_early="Helix server not responding. Restarting to recover.",
                reason_late=(
                    f"Helix server not responding at {progress*100:.0f}% progress. "
                    f"Warning — benchmark may stall imminently."
                ),
            )

        # ── HIGH: unauthorized model appeared mid-run ───────────────
        if unauthorized:
            self._gated_incident(
                incident_type="unauthorized_model_loaded",
                details={
                    "offending": unauthorized,
                    "benchmark_model": self.benchmark_model,
                },
                reason_early=(
                    f"New model loaded into VRAM mid-run: {unauthorized}. "
                    f"Restarting to reclaim clean VRAM."
                ),
                reason_late=(
                    f"New model loaded into VRAM mid-run at {progress*100:.0f}% "
                    f"progress: {unauthorized}. Flagging as contamination — "
                    f"results should be re-run in isolation."
                ),
            )

        # ── VRAM pressure tracking ──────────────────────────────────
        if vram:
            used_pct = vram["used_pct"]

            if used_pct >= self.cfg.vram_abort_pct:
                # 95%+ = saturated, progress-gated
                self._gated_incident(
                    incident_type="vram_saturated",
                    details={
                        "used_pct": round(used_pct, 4),
                        "used_mb": vram["used_mb"],
                        "total_mb": vram["total_mb"],
                        "temp_c": vram["temp_c"],
                    },
                    reason_early=(
                        f"VRAM saturated at {used_pct*100:.1f}% — thrashing likely. "
                        f"Restarting for clean run."
                    ),
                    reason_late=(
                        f"VRAM saturated at {used_pct*100:.1f}% at "
                        f"{progress*100:.0f}% progress. Warning — results may be biased."
                    ),
                )
                # 95%+ is distinct from the 90% counter — don't touch counter

            elif used_pct >= self.cfg.vram_warn_pct:
                # 90-95% = pressure, increment counter
                self.vram_counter += 1
                if self.vram_counter >= self.cfg.vram_warn_passes:
                    self._log_incident(
                        severity=SEV_MEDIUM,
                        incident_type="vram_pressure_sustained",
                        action="warn",
                        details={
                            "used_pct": round(used_pct, 4),
                            "passes": self.vram_counter,
                            "threshold_pct": self.cfg.vram_warn_pct,
                        },
                        reason=(
                            f"VRAM at or above {self.cfg.vram_warn_pct*100:.0f}% "
                            f"for {self.vram_counter} consecutive checks "
                            f"(~{self.vram_counter * self.cfg.check_interval_s:.0f}s). "
                            f"Sustained pressure — may indicate background process "
                            f"competing for VRAM."
                        ),
                    )
                    self.vram_counter = 0  # reset after firing
            else:
                # <90% = healthy, reset counter
                self.vram_counter = 0

    def _gated_incident(
        self,
        incident_type: str,
        details: dict,
        reason_early: str,
        reason_late: str,
    ) -> None:
        """Log a HIGH severity incident with progress-gated action."""
        if self._is_early_run():
            self._log_incident(
                severity=SEV_HIGH,
                incident_type=incident_type,
                action="restart_requested",
                details=details,
                reason=reason_early,
            )
            self.abort_flag.set()
        else:
            self._log_incident(
                severity=SEV_HIGH,
                incident_type=incident_type,
                action="warn",
                details=details,
                reason=reason_late,
            )

    def _pid_alive(self) -> bool:
        """Cross-platform PID liveness check."""
        try:
            if sys.platform == "win32":
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, self.bench_pid
                )
                if handle == 0:
                    return False
                exit_code = ctypes.c_ulong()
                alive = (
                    ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                    and exit_code.value == 259  # STILL_ACTIVE
                )
                ctypes.windll.kernel32.CloseHandle(handle)
                return alive
            else:
                os.kill(self.bench_pid, 0)
                return True
        except Exception:
            return False

    # ────────────────────────────────────────────────────────────────
    # Thread lifecycle
    # ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background monitor thread."""
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="bench-monitor")
        self._thread.start()
        self._print(f"Monitor thread started (interval={self.cfg.check_interval_s}s)")

    def _run_loop(self) -> None:
        """Main monitor loop. Exits when stop_flag or abort_flag is set."""
        while not self.stop_flag.is_set() and not self.abort_flag.is_set():
            try:
                self._run_check()
            except Exception as e:
                self._print(f"[!] Monitor check raised: {e}")
            # Wait with early-exit on flags
            self.stop_flag.wait(timeout=self.cfg.check_interval_s)

    def stop(self) -> None:
        """Clean shutdown."""
        self.stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def should_abort(self) -> bool:
        """True if the monitor has signalled the benchmark should stop."""
        return self.abort_flag.is_set()

    # ────────────────────────────────────────────────────────────────
    # Reporting
    # ────────────────────────────────────────────────────────────────

    def final_report(self) -> dict:
        """Summary for the benchmark results JSON."""
        by_severity: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for inc in self.incidents:
            by_severity[inc.severity] = by_severity.get(inc.severity, 0) + 1
            by_type[inc.type] = by_type.get(inc.type, 0) + 1

        has_critical = any(i.severity == SEV_CRITICAL for i in self.incidents)
        has_high = any(i.severity == SEV_HIGH for i in self.incidents)

        run_clean = not (has_critical or has_high)
        status = "clean"
        if has_critical:
            status = "critical"
        elif has_high:
            status = "warnings"
        elif self.incidents:
            status = "minor_warnings"

        return {
            "run_clean": run_clean,
            "status": status,
            "incident_count": len(self.incidents),
            "by_severity": by_severity,
            "by_type": by_type,
            "checks_performed": self.check_count,
            "elapsed_s": round(time.time() - self.start_ts, 1),
            "monitor_log_path": str(self.monitor_log),
            "incidents": [i.to_dict() for i in self.incidents],
        }


# ──────────────────────────────────────────────────────────────────────
# Standalone pre-flight mode
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Benchmark state monitor — standalone pre-flight check."
    )
    parser.add_argument("--model", required=True, help="Target benchmark model (e.g. qwen3:8b)")
    parser.add_argument("--incremental", default="/tmp/bench_incremental.jsonl",
                        help="Path to incremental JSONL (dummy value ok for pre-flight)")
    parser.add_argument("--n", type=int, default=50, help="Total needle count")
    parser.add_argument("--helix-url", default="http://127.0.0.1:11437")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--helix-config", default=None,
                        help="Path to helix.toml (default: HELIX_CONFIG env or cwd)")
    parser.add_argument("--check-only", action="store_true",
                        help="Run pre-flight check and exit")
    args = parser.parse_args()

    monitor = BenchmarkMonitor(
        benchmark_model=args.model,
        incremental_output_path=args.incremental,
        total_needles=args.n,
        helix_config_path=args.helix_config,
        helix_url=args.helix_url,
        ollama_url=args.ollama_url,
    )

    if not monitor.preflight():
        sys.exit(1)

    if args.check_only:
        print("[monitor] Pre-flight passed. Exiting (check-only mode).")
        sys.exit(0)

    print("[monitor] Pre-flight passed. In standalone mode, starting monitor and waiting.")
    print("[monitor] Press Ctrl+C to stop.")
    monitor.start()
    try:
        while not monitor.should_abort():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[monitor] Shutting down...")
    finally:
        monitor.stop()
        print(json.dumps(monitor.final_report(), indent=2))


if __name__ == "__main__":
    main()
