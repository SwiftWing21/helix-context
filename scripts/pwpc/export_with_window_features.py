"""Enrich a cwola_log export with per-row sliding-window correlation features.

For each row in the export, compute cwola.sliding_window_features() over
the 50 rows preceding it (same session, pre-ts). Adds two keys:

  window_features: dict[str, float]   # 36 unique off-diagonal pairs
  window_n_rows: int                  # rows that contributed (≤50)
  window_degenerate: bool             # True if extractor declined

This is the format batman's PWPC manifold port expects for the agreement
head training input. See:
  docs/collab/comms/LOCKSTEP_MATRIX_FINDINGS_2026-04-14.md
  helix_context/cwola.py  (sliding_window_features)

Usage:
    python scripts/pwpc/export_with_window_features.py \
        --in cwola_export/cwola_export_20260415.json \
        --out cwola_export/cwola_export_20260415_windowed.json \
        --db genome.db \
        [--window-size 50]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from helix_context import cwola


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=Path, required=True,
                    help="input JSON (cwola_export_*.json)")
    ap.add_argument("--out", dest="dst", type=Path, required=True,
                    help="output JSON (enriched)")
    ap.add_argument("--db", type=Path, default=Path("genome.db"),
                    help="genome.db path (for sliding-window queries)")
    ap.add_argument("--window-size", type=int, default=50,
                    help="rows per window (default 50)")
    args = ap.parse_args()

    with args.src.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise SystemExit(f"expected JSON array in {args.src}, got {type(rows).__name__}")
    print(f"read {len(rows)} rows from {args.src}")

    # Open read-only connection to genome.db for the window queries.
    conn = sqlite3.connect(f"file:{args.db.as_posix()}?mode=ro", uri=True)

    n_ok = 0
    n_deg = 0
    t0 = time.perf_counter()
    for row in rows:
        sid = row.get("session_id")
        ts = row.get("ts")
        if not sid or ts is None:
            row["window_features"] = {}
            row["window_n_rows"] = 0
            row["window_degenerate"] = True
            n_deg += 1
            continue
        out = cwola.sliding_window_features(
            conn, session_id=sid, before_ts=float(ts), window_size=args.window_size,
        )
        row["window_features"] = out["features"]
        row["window_n_rows"] = out["n_rows"]
        row["window_degenerate"] = out["degenerate"]
        if out["degenerate"]:
            n_deg += 1
        else:
            n_ok += 1

    dt = time.perf_counter() - t0
    print(f"enriched: ok={n_ok} degenerate={n_deg} in {dt:.1f}s")

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    with args.dst.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    size_kb = args.dst.stat().st_size / 1024
    print(f"wrote {args.dst} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
