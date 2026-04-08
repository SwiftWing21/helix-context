"""
Deep ingest of F:\Projects into the Helix genome.
Runs as a long-running background process.

Usage:
    python scripts/deep_ingest.py
    python scripts/deep_ingest.py --dry-run    # count files only
"""

import argparse
import os
import sys
import time

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
PROJECTS_ROOT = "F:/Projects"

# Directories to skip entirely
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "target", ".venv", "dist",
    "venv", ".claude", ".continue", ".pytest_cache", "site-packages",
    ".egg-info", ".mypy_cache", ".ruff_cache", "build", ".tox",
    "htmlcov", ".coverage", "migrations",
}

# File extensions to ingest
CODE_EXTS = {".py", ".rs", ".ts", ".js", ".toml"}
TEXT_EXTS = {".md", ".txt", ".rst"}

# Skip test files
SKIP_PREFIXES = ("test_", "conftest")
SKIP_SUFFIXES = ("_test.py",)

# Size limits (bytes)
MIN_SIZE = 500
MAX_SIZE = 50_000


def find_files(root):
    """Find all eligible files for ingestion."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip noise directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            _, ext = os.path.splitext(fname)
            if ext not in CODE_EXTS and ext not in TEXT_EXTS:
                continue

            # Skip test files
            if any(fname.startswith(p) for p in SKIP_PREFIXES):
                continue
            if any(fname.endswith(s) for s in SKIP_SUFFIXES):
                continue

            path = os.path.join(dirpath, fname)
            size = os.path.getsize(path)
            if size < MIN_SIZE or size > MAX_SIZE:
                continue

            content_type = "code" if ext in CODE_EXTS else "text"
            files.append((path, content_type, size))

    return files


def ingest(files, client):
    """Ingest files into the Helix genome."""
    total_genes = 0
    total_time = 0
    errors = 0
    skipped = 0

    for i, (path, content_type, size) in enumerate(files):
        rel = os.path.relpath(path, PROJECTS_ROOT)
        try:
            content = open(path, encoding="utf-8", errors="replace").read()
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] SKIP {rel}: {e}")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(files)}] {rel} ({size:,} bytes)...", end=" ", flush=True)

        try:
            t0 = time.time()
            resp = client.post(f"{HELIX_URL}/ingest", json={
                "content": content,
                "content_type": content_type,
                "metadata": {"path": os.path.abspath(path)},
            })
            dt = time.time() - t0
            total_time += dt

            if resp.status_code == 200:
                count = resp.json().get("count", 0)
                total_genes += count
                print(f"{count} genes ({dt:.1f}s)")
            else:
                print(f"HTTP {resp.status_code} ({dt:.1f}s)")
                errors += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errors += 1

        # Progress checkpoint every 50 files
        if (i + 1) % 50 == 0:
            try:
                stats = client.get(f"{HELIX_URL}/stats").json()
                print(f"\n  --- Checkpoint: {stats['total_genes']} genes, "
                      f"{stats['compression_ratio']:.1f}x, "
                      f"{total_time:.0f}s elapsed ---\n")
            except Exception:
                pass

    return total_genes, total_time, errors, skipped


def main():
    parser = argparse.ArgumentParser(description="Deep ingest F:\\Projects into Helix")
    parser.add_argument("--dry-run", action="store_true", help="Count files only")
    args = parser.parse_args()

    files = find_files(PROJECTS_ROOT)
    files.sort(key=lambda x: x[0])

    print(f"Found {len(files)} files to ingest")
    print(f"Total size: {sum(s for _, _, s in files) / 1024 / 1024:.1f} MB")
    print()

    if args.dry_run:
        # Show per-project breakdown
        projects = {}
        for path, _, size in files:
            proj = os.path.relpath(path, PROJECTS_ROOT).split(os.sep)[0]
            projects.setdefault(proj, [0, 0])
            projects[proj][0] += 1
            projects[proj][1] += size

        for proj, (count, size) in sorted(projects.items(), key=lambda x: -x[1][0]):
            print(f"  {proj}: {count} files ({size / 1024:.0f} KB)")
        print(f"\nEstimated time: {len(files) * 15 / 60:.0f} minutes")
        return

    client = httpx.Client(timeout=300)

    # Check server
    try:
        health = client.get(f"{HELIX_URL}/health").json()
        print(f"Server: {health['status']}, ribosome: {health['ribosome']}, "
              f"genes: {health['genes']}")
    except Exception:
        print(f"Cannot reach Helix at {HELIX_URL}")
        sys.exit(1)

    print()
    total_genes, total_time, errors, skipped = ingest(files, client)

    # Final stats
    try:
        stats = client.get(f"{HELIX_URL}/stats").json()
        print(f"\n=== COMPLETE ===")
        print(f"New genes: {total_genes}")
        print(f"Errors: {errors}, Skipped: {skipped}")
        print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"Genome: {stats['total_genes']} genes, {stats['compression_ratio']:.1f}x")
        print(f"Raw: {stats['total_chars_raw']:,} -> Compressed: {stats['total_chars_compressed']:,}")
    except Exception:
        print(f"\nDone. Genes: {total_genes}, Errors: {errors}, Time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
