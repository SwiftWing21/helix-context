"""
Density-gate compaction sweep — dry-run by default.

Applies the Struggle 1 density gate retroactively to an existing genome:
  1. Reads all currently-OPEN genes
  2. Runs the shared apply_density_gate() decision function
  3. Demotes failing genes to EUCHROMATIN or HETEROCHROMATIN
  4. Reports category-level stats + reason breakdown

Defaults to --dry-run (no DB writes). Pass --apply to actually write.

Usage:
    # Preview what would happen (safe):
    python scripts/compact_genome_sweep.py
    python scripts/compact_genome_sweep.py --db F:/Projects/helix-context/genome.db
    python scripts/compact_genome_sweep.py --output stats.json

    # Apply the sweep (writes to the DB — make a backup first):
    python scripts/compact_genome_sweep.py --apply --backup
    python scripts/compact_genome_sweep.py --apply --db /path/to/genome.db

Coordination notes:
  - If another process has the genome open (a running helix server, an
    active benchmark), SQLite's WAL mode allows concurrent reads from
    this script BUT writes will queue behind the server's writer lock.
    For large sweeps (~4000 demotions on the 2026-04-10 genome), the
    safe pattern is:
      1. Run with --dry-run first, review the preview
      2. Announce a restart via bridge.announce_restart() if a server
         is running, coordinating via ~/.helix/shared/signals/server_state.json
      3. Stop the server / wait for benchmarks to finish
      4. Back up the DB (cp genome.db genome.db.pre-compact.bak)
      5. Run with --apply
      6. Restart the server
      7. Re-run the benchmark to measure retrieval improvement
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from collections import Counter


def _open_genome(db_path: str):
    """Open a Genome pointed at the given path. Imports lazily so the
    script can print --help without pulling in the full helix stack."""
    # Force the helix-context root onto sys.path so "from helix_context ..." works
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from helix_context.genome import Genome
    return Genome(path=db_path)


def _source_bucket(src: str | None) -> str:
    """Categorize a source path for report aggregation."""
    s = (src or "").lower()
    if any(m in s for m in ("steamlibrary", "steamapps", "beamng.drive", "hades/", "hades\\")):
        return "steam"
    if "\\.next\\" in s or "/.next/" in s or "node_modules" in s or "__pycache__" in s:
        return "build_artifacts"
    if "helix-context" in s or "helix_context" in s:
        return "helix"
    if "cosmictasha" in s or "novabridge" in s:
        return "cosmic"
    if "bookkeeper" in s:
        return "bookkeeper"
    if "two-brain-audit" in s or "scorerift" in s:
        return "scorerift"
    if "biged" in s or "fleet/" in s or "education" in s:
        return "education"
    if "autoresearch" in s:
        return "autoresearch"
    if not src:
        return "no_source"
    return "other"


def _report(stats: dict, dry_run: bool) -> str:
    """Render a human-readable report from the compact_genome stats dict."""
    lines = []
    verb = "WOULD" if dry_run else "DID"
    lines.append("=" * 60)
    lines.append(f"Density gate compaction sweep ({'DRY RUN' if dry_run else 'APPLIED'})")
    lines.append("=" * 60)
    lines.append(f"  scanned               : {stats['scanned']}")
    lines.append(f"  {verb} stay OPEN          : {stats['kept_open']}")
    lines.append(f"  {verb} demote to EUCHRO   : {stats['to_euchromatin']}")
    lines.append(f"  {verb} demote to HETERO   : {stats['to_heterochromatin']}")
    lines.append(f"  skipped (no embedding): {stats['skipped_no_embedding']}")
    total_demoted = stats["to_euchromatin"] + stats["to_heterochromatin"]
    pct = total_demoted / max(stats["scanned"], 1) * 100
    lines.append(f"  {'total demoted':<22}: {total_demoted}  ({pct:.1f}%)")
    lines.append("")
    lines.append("Reasons:")
    for reason, count in sorted(stats.get("by_reason", {}).items(), key=lambda x: -x[1]):
        lines.append(f"  {reason:<22} {count:>6}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Density gate compaction sweep for helix-context genome",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        default="F:/Projects/helix-context/genome.db",
        help="Path to the genome database (default: F:/Projects/helix-context/genome.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply the sweep (writes to DB). Default is dry-run.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Make a .bak copy before applying. Implies --apply.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the full stats dict to this JSON file",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        sys.exit(2)

    apply_writes = args.apply or args.backup

    if args.backup:
        backup_path = db_path.with_suffix(f".db.pre-compact.{int(time.time())}.bak")
        print(f"Backing up {db_path} -> {backup_path}")
        shutil.copy2(db_path, backup_path)
        # Also back up the WAL and SHM files if they exist
        for suffix in ("-wal", "-shm"):
            src = Path(str(db_path) + suffix)
            if src.exists():
                shutil.copy2(src, str(backup_path) + suffix)
        print(f"  Backup complete: {backup_path}")

    print(f"Opening genome at {db_path}...")
    print(f"Mode: {'APPLY (writes)' if apply_writes else 'DRY RUN (no writes)'}")
    print()

    genome = _open_genome(str(db_path))
    try:
        # Pre-sweep snapshot for before/after comparison
        pre_stats = genome.stats()
        print(f"Pre-sweep genome state:")
        print(f"  total_genes : {pre_stats['total_genes']}")
        print(f"  open        : {pre_stats.get('open', pre_stats.get('total_genes', 0))}")
        print(f"  euchromatin : {pre_stats.get('euchromatin', 0)}")
        print(f"  heterochrom : {pre_stats.get('heterochromatin', 0)}")
        print(f"  compression : {pre_stats.get('compression_ratio', 0):.2f}x")
        print()

        t0 = time.perf_counter()
        stats = genome.compact_genome(dry_run=not apply_writes)
        elapsed = time.perf_counter() - t0

        print(_report(stats, dry_run=not apply_writes))
        print()
        print(f"Elapsed: {elapsed:.1f}s")

        # Post-sweep snapshot (only meaningful if we actually wrote)
        if apply_writes:
            post_stats = genome.stats()
            print()
            print("Post-sweep genome state:")
            print(f"  total_genes : {post_stats['total_genes']}")
            print(f"  open        : {post_stats.get('open', 0)}")
            print(f"  euchromatin : {post_stats.get('euchromatin', 0)}")
            print(f"  heterochrom : {post_stats.get('heterochromatin', 0)}")
            print(f"  compression : {post_stats.get('compression_ratio', 0):.2f}x")

            open_delta = post_stats.get("open", 0) - pre_stats.get("open", pre_stats.get("total_genes", 0))
            print()
            print(f"Delta in OPEN genes: {open_delta}")

        if args.output:
            out_path = Path(args.output)
            payload = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "db_path": str(db_path),
                "dry_run": not apply_writes,
                "pre_stats": pre_stats,
                "compact_stats": stats,
                "elapsed_s": round(elapsed, 2),
            }
            out_path.write_text(json.dumps(payload, indent=2, default=str))
            print(f"\nFull stats written to {out_path}")

        if not apply_writes:
            print()
            print("This was a DRY RUN. No changes were written.")
            print("To apply these changes:")
            print(f"  python scripts/compact_genome_sweep.py --db {args.db} --backup --apply")
    finally:
        genome.close()


if __name__ == "__main__":
    main()
