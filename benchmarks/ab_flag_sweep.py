"""A/B sweep: stand up a bench server on port 11438 with variant
helix.toml configs and run bench_skill_activation.py against each.

Does NOT touch the live server on :11437. Uses HELIX_CONFIG env var
to point at per-config tomls and HELIX_URL to redirect benches.

Output: ab_sweep_<config>.json per config + ab_sweep_summary.json
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
BASE_TOML = REPO / "helix.toml"
BENCH_PORT = 11438
BENCH_URL = f"http://127.0.0.1:{BENCH_PORT}"

CONFIGS = [
    ("baseline",       {"distance_metric": "cosine", "sr_enabled": "false",
                        "ray_trace_theta": "false", "seeded_edges_enabled": "false"}),
    ("w1",             {"distance_metric": "w1",     "sr_enabled": "false",
                        "ray_trace_theta": "false", "seeded_edges_enabled": "false"}),
    ("sr",             {"distance_metric": "cosine", "sr_enabled": "true",
                        "ray_trace_theta": "false", "seeded_edges_enabled": "false"}),
    ("seeded_plus_sr", {"distance_metric": "cosine", "sr_enabled": "true",
                        "ray_trace_theta": "false", "seeded_edges_enabled": "true"}),
    ("all_on",         {"distance_metric": "w1",     "sr_enabled": "true",
                        "ray_trace_theta": "true",  "seeded_edges_enabled": "true"}),
]

# Which benches to run per config. dim_lock retrieval-only (ASK_PROXY=0,
# N small) is the multi-hop signal SR needs to show lift on; skill_activation
# is the tier-heatmap signal.
RUN_SKILL_ACTIVATION = True
RUN_DIM_LOCK = True
DIM_LOCK_N = "10"
DIM_LOCK_ASK_PROXY = "0"  # retrieval-only, no LLM answer grading


def patch_toml(out_path: Path, flags: dict) -> None:
    """Copy base toml, patch flag lines + the FIRST `port =` only.

    helix.toml has two `port =` lines: the server port (under [server])
    and a synonym entry `port = ["proxy", ...]` under [synonyms]. Only
    rewrite the first one; overwriting the synonym list with an int
    breaks load_config().
    """
    src = BASE_TOML.read_text(encoding="utf-8")
    out = []
    port_patched = False
    current_section = None
    for line in src.splitlines():
        stripped = line.lstrip()
        # Track section to only patch port in [server]
        if stripped.startswith("[") and stripped.endswith("]"):
            current_section = stripped.strip("[]").strip()
        patched = False
        for flag, value in flags.items():
            if stripped.startswith(f"{flag} ="):
                indent = line[: len(line) - len(stripped)]
                if flag == "distance_metric":
                    out.append(f'{indent}{flag} = "{value}"')
                else:
                    out.append(f"{indent}{flag} = {value}")
                patched = True
                break
        if not patched:
            if (
                stripped.startswith("port =")
                and current_section == "server"
                and not port_patched
            ):
                indent = line[: len(line) - len(stripped)]
                out.append(f"{indent}port = {BENCH_PORT}")
                port_patched = True
                continue
            out.append(line)
    out_path.write_text("\n".join(out), encoding="utf-8")


def wait_healthy(url: str, timeout: float = 90) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def run_config(name: str, flags: dict) -> dict:
    """Start bench server, run skill_activation, tear down, return result."""
    toml_path = REPO / f"helix.bench.{name}.toml"
    patch_toml(toml_path, flags)

    env = os.environ.copy()
    env["HELIX_CONFIG"] = str(toml_path)
    env["PYTHONIOENCODING"] = "utf-8"

    log_path = REPO / f"benchmarks/ab_server_{name}.log"
    log_f = log_path.open("w", encoding="utf-8", errors="replace")
    print(f"[{name}] starting server on :{BENCH_PORT} ...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "helix_context.server:app",
         "--host", "127.0.0.1", "--port", str(BENCH_PORT)],
        env=env, stdout=log_f, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        if not wait_healthy(BENCH_URL):
            print(f"[{name}] server did not become healthy; skipping")
            return {"config": name, "error": "server_unhealthy"}

        bench_env = os.environ.copy()
        bench_env["HELIX_URL"] = BENCH_URL
        bench_env["PYTHONIOENCODING"] = "utf-8"

        out = {
            "config": name,
            "flags": {k: v for k, v in flags.items() if not k.startswith("__")},
        }

        if RUN_SKILL_ACTIVATION:
            print(f"[{name}] running bench_skill_activation ...", flush=True)
            subprocess.run(
                [sys.executable, "benchmarks/bench_skill_activation.py"],
                env=bench_env, cwd=str(REPO),
                capture_output=True, text=True, timeout=300,
            )
            sa_src = REPO / "benchmarks" / "skill_activation_results.json"
            if sa_src.exists():
                shutil.copy(sa_src, REPO / f"benchmarks/ab_sweep_{name}.json")

        if RUN_DIM_LOCK:
            print(f"[{name}] running bench_dimensional_lock N={DIM_LOCK_N} "
                  f"ASK_PROXY={DIM_LOCK_ASK_PROXY} ...", flush=True)
            dl_env = bench_env.copy()
            dl_env["N"] = DIM_LOCK_N
            dl_env["ASK_PROXY"] = DIM_LOCK_ASK_PROXY
            dl_env["SEED"] = "42"
            subprocess.run(
                [sys.executable, "benchmarks/bench_dimensional_lock.py"],
                env=dl_env, cwd=str(REPO),
                capture_output=True, text=True, timeout=1200,
            )
            dl_src = REPO / "benchmarks" / "dimensional_lock_results.json"
            if dl_src.exists():
                shutil.copy(dl_src, REPO / f"benchmarks/ab_dim_lock_{name}.json")

        return out
    finally:
        print(f"[{name}] stopping server ...", flush=True)
        try:
            proc.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        log_f.close()
        try:
            toml_path.unlink()
        except Exception:
            pass


def main() -> int:
    results = []
    for name, flags in CONFIGS:
        r = run_config(name, dict(flags))
        results.append(r)
        print(f"[{name}] done\n", flush=True)

    summary_path = REPO / "benchmarks/ab_sweep_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
