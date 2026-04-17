"""Fresh genome rebuild — phase 1 of the sharding migration.

Builds a clean genome.db at a new path by re-ingesting whitelisted
directories through HelixContextManager. Skips the dirty legacy genome
entirely. Layered fingerprints (parent genes + CHUNK_OF edges) are
created natively at ingest — no post-hoc backfill needed.

Runs safely while helix is serving from the old genome: the new
genome path is different, no lock contention, no shared state.

Cutover is a single helix.toml path change + supervisor restart.

Usage:
    python scripts/fresh_genome_build.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from pathlib import Path
from typing import Iterator

# Repo root on sys.path so helix_context imports resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from helix_context.context_manager import HelixContextManager  # noqa: E402
from helix_context.config import HelixConfig, load_config  # noqa: E402


# ── Defaults (override via CLI or edit in place) ─────────────────────

WHITELIST_DIRS = [
    "F:/Projects/helix-context",
    "F:/Projects/Education",
    "F:/Projects/BookKeeper",
    "F:/Projects/BigEd-ModuleHub",
    "F:/Projects/CosmicTasha",
    "F:/Projects/_plans",
]

IGNORE_DIR_NAMES = {
    ".next", "node_modules", "dist", "build", "target",
    "__pycache__", ".venv", ".git", ".claude",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "helix-cache", "Helix-backup",
    "genomes",  # don't ingest our own new genome storage
}

IGNORE_FILE_PATTERNS = [
    "*.min.js", "*.lock", "*.db", "*.db-wal", "*.db-shm",
    "*.pyc", "*.pyo",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.svg",
    "*.mp4", "*.mov", "*.avi", "*.webm",
    "*.zip", "*.tar", "*.gz", "*.7z", "*.rar",
    "*.exe", "*.dll", "*.so", "*.dylib", "*.bin",
    "*.woff", "*.woff2", "*.ttf", "*.otf",
    "*.pdf", "*.docx", "*.xlsx", "*.pptx",
]

MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MB cap per file

INGESTIBLE_EXTS = {
    ".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".java", ".c", ".cpp", ".h", ".hpp", ".rb", ".php",
    ".md", ".txt", ".rst", ".adoc",
    ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".ps1", ".bat",
    ".sql", ".css", ".scss", ".sass", ".html", ".htm",
    ".lua", ".vim", ".el",
}

TARGET_PATH = "F:/Projects/helix-context/genomes/main/genome.db"


# ── File walker ──────────────────────────────────────────────────────


def should_skip_dir(name: str) -> bool:
    return name in IGNORE_DIR_NAMES or name.startswith(".")


def should_skip_file(name: str, size: int) -> bool:
    if size > MAX_FILE_BYTES:
        return True
    if any(fnmatch.fnmatch(name, pat) for pat in IGNORE_FILE_PATTERNS):
        return True
    ext = os.path.splitext(name)[1].lower()
    if ext and ext not in INGESTIBLE_EXTS:
        return True
    return False


def walk(root: str) -> Iterator[Path]:
    """Yield ingestible file paths under root, respecting ignore rules."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in place so os.walk skips ignored subdirs entirely.
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
        for fn in filenames:
            full = Path(dirpath) / fn
            try:
                size = full.stat().st_size
            except OSError:
                continue
            if should_skip_file(fn, size):
                continue
            yield full


# ── Ingestion loop ───────────────────────────────────────────────────


def content_type_for(path: Path) -> str:
    """Map extension to helix content_type hint."""
    ext = path.suffix.lower()
    if ext in {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".go",
               ".java", ".c", ".cpp", ".h", ".rb", ".php", ".lua", ".sh",
               ".sql", ".html", ".css", ".scss"}:
        return "code"
    if ext in {".md", ".txt", ".rst", ".adoc"}:
        return "text"
    if ext in {".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"}:
        return "text"
    return "text"


def build_config(target_path: str) -> HelixConfig:
    """Clone the live helix.toml config but override the genome path.

    Ingest uses the same ribosome + tagger stack as production — the
    new genome is shaped identically to what a live install produces.
    """
    cfg = load_config()  # reads helix.toml by default
    # Swap the genome path — everything else (ribosome, ingestion
    # backend, tagger, budget) stays identical.
    cfg.genome.path = target_path
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=TARGET_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only ingest the first N files (smoke test)")
    args = ap.parse_args()

    target = Path(args.target)
    print(f"[fresh-build] target: {target}")
    print(f"[fresh-build] whitelist: {len(WHITELIST_DIRS)} root dirs")

    # Phase 1: enumerate files.
    t0 = time.perf_counter()
    files: list[Path] = []
    per_root_counts: dict[str, int] = {}
    for root in WHITELIST_DIRS:
        if not os.path.isdir(root):
            print(f"  skip missing root: {root}")
            continue
        before = len(files)
        files.extend(walk(root))
        per_root_counts[root] = len(files) - before
    total_bytes = sum(f.stat().st_size for f in files)
    print(f"[fresh-build] enumerated {len(files):,} files ({total_bytes/1024/1024:.1f} MB) in {time.perf_counter()-t0:.1f}s")
    for root, n in per_root_counts.items():
        print(f"  {n:>6,}  {root}")

    if args.limit:
        files = files[:args.limit]
        print(f"[fresh-build] --limit applied: will ingest {len(files)} files")

    if args.dry_run:
        print("\n[fresh-build] dry-run: no writes. Sample of next 10 files:")
        for f in files[:10]:
            print(f"  {f.stat().st_size:>8} b  {f}")
        return 0

    # Phase 2: ensure target path exists + build manager.
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"[fresh-build] WARNING: target already exists ({target.stat().st_size/1024/1024:.1f} MB)")
        print(f"             rebuild is IDEMPOTENT (INSERT OR REPLACE) but you likely want a clean start.")
        resp = input("  delete existing and start fresh? [y/N] ").strip().lower()
        if resp == "y":
            target.unlink()
            for suffix in ("-wal", "-shm"):
                p = Path(str(target) + suffix)
                if p.exists():
                    p.unlink()
            print("  cleared.")
        else:
            print("  keeping existing; rebuild will upsert on top.")

    cfg = build_config(str(target))
    print(f"[fresh-build] building ContextManager against {target}")
    mgr = HelixContextManager(cfg)

    # Phase 3: ingest loop.
    t_ingest = time.perf_counter()
    errors = 0
    total_genes = 0
    for i, path in enumerate(files):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            errors += 1
            continue
        if not content.strip():
            continue
        try:
            gene_ids = mgr.ingest(
                content,
                content_type=content_type_for(path),
                metadata={"path": str(path).replace("\\", "/")},
            )
            total_genes += len(gene_ids)
        except Exception as e:
            errors += 1
            if errors < 20:
                print(f"  ingest failed {path}: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_ingest
            rate = (i + 1) / max(elapsed, 0.001)
            remaining = (len(files) - (i + 1)) / max(rate, 0.001)
            print(f"  [{i+1:,}/{len(files):,}] {total_genes:,} genes, "
                  f"{errors} errors, {rate:.1f} f/s, ETA {remaining/60:.1f}m")

    # Phase 4: report.
    elapsed = time.perf_counter() - t_ingest
    print(f"\n[fresh-build] DONE in {elapsed/60:.1f}m")
    print(f"  files processed: {len(files):,}")
    print(f"  genes created:   {total_genes:,}")
    print(f"  errors:          {errors}")
    print(f"  target size:     {target.stat().st_size/1024/1024:.1f} MB")

    # Quick stats on what got ingested.
    import sqlite3
    conn = sqlite3.connect(str(target))
    parents = conn.execute("SELECT COUNT(*) FROM genes WHERE key_values LIKE '%is_parent=true%'").fetchone()[0]
    edges = conn.execute("SELECT COUNT(*) FROM gene_relations WHERE relation = 100").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM genes").fetchone()[0]
    conn.close()
    print(f"  total genes in DB: {total:,}")
    print(f"  parent genes:      {parents:,}")
    print(f"  CHUNK_OF edges:    {edges:,}")
    print(f"\nNext: update helix.toml [genome] path → {target}")
    print(f"      then stop + restart helix supervisor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
