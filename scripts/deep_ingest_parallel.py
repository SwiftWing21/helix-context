r"""
Parallel deep ingest of F:\Projects into the Helix genome.

Uses concurrent workers to overlap I/O with GPU inference.
Normal proxy operation is unaffected — only this script uses parallelism.

The bottleneck is Ollama (serializes GPU inference), but we gain speed by:
  1. Pre-reading and chunking files while the ribosome is busy
  2. Pipelining HTTP requests (Ollama can queue a few)
  3. Overlapping SQLite writes with the next inference call

Usage:
    python scripts/deep_ingest_parallel.py                    # default 3 workers
    python scripts/deep_ingest_parallel.py --workers 4        # more aggressive
    python scripts/deep_ingest_parallel.py --dry-run          # count files only
    python scripts/deep_ingest_parallel.py --resume            # skip already-done (default)
"""

import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import httpx

HELIX_URL = os.environ.get("HELIX_URL", "http://127.0.0.1:11437")
PROJECTS_ROOT = r"F:\Projects"

# Directories to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", "target", ".venv", "dist",
    "venv", ".claude", ".continue", ".pytest_cache", "site-packages",
    ".egg-info", ".mypy_cache", ".ruff_cache", "build", ".tox",
    "htmlcov", ".coverage", "migrations", ".next", "standalone",
    ".turbo", "coverage", ".nyc_output",
}

CODE_EXTS = {".py", ".rs", ".ts", ".js", ".toml"}
TEXT_EXTS = {".md", ".txt", ".rst"}
SKIP_PREFIXES = ("test_", "conftest")
SKIP_SUFFIXES = ("_test.py",)

MIN_SIZE = 500
MAX_SIZE = 500_000        # Accept very large files (skeleton extraction handles them)
CHUNK_THRESHOLD = 15_000
SKELETON_THRESHOLD = 50_000  # Files above this get skeleton-extracted instead of full ingest

PROGRESS_FILE = os.path.join(os.path.dirname(__file__), ".ingest_progress")
_progress_lock = Lock()
_print_lock = Lock()
_stats = {"genes": 0, "errors": 0, "skipped": 0, "files_done": 0}
_stats_lock = Lock()


def _load_progress():
    if os.path.exists(PROGRESS_FILE):
        return set(open(PROGRESS_FILE, encoding="utf-8").read().splitlines())
    return set()


def _save_progress(path):
    with _progress_lock:
        with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
            f.write(path + "\n")


def _log(msg):
    with _print_lock:
        print(msg, flush=True)


def _extract_skeleton(content, content_type):
    """Extract structural DNA from large files.

    For code: keeps imports, class/function signatures, docstrings, constants.
    For text/markdown: keeps headings, first paragraph of each section, lists.
    Result is typically 10-20% of the original — enough for the ribosome to
    understand the architecture without processing every line.
    """
    if content_type == "code":
        lines = content.split("\n")
        skeleton = []
        in_docstring = False
        docstring_char = None

        for line in lines:
            stripped = line.strip()

            # Always keep imports
            if stripped.startswith(("import ", "from ", "use ", "mod ")):
                skeleton.append(line)
                continue

            # Always keep class/function/struct definitions
            if re.match(r"^(class |def |async def |fn |pub fn |struct |impl |interface |type |export |const |UPPER)", stripped):
                skeleton.append(line)
                continue

            # Keep decorators
            if stripped.startswith("@"):
                skeleton.append(line)
                continue

            # Track and keep docstrings
            if not in_docstring:
                if stripped.startswith(('"""', "'''")):
                    in_docstring = True
                    docstring_char = stripped[:3]
                    skeleton.append(line)
                    if stripped.count(docstring_char) >= 2:
                        in_docstring = False
                    continue
            else:
                skeleton.append(line)
                if docstring_char in stripped:
                    in_docstring = False
                continue

            # Keep constants (ALL_CAPS assignments)
            if re.match(r"^[A-Z_][A-Z_0-9]+ *=", stripped):
                skeleton.append(line)
                continue

            # Keep type annotations / dataclass fields
            if re.match(r"^\w+\s*:\s*\w+", stripped):
                skeleton.append(line)
                continue

        return "\n".join(skeleton)

    else:
        # Markdown/text: keep headings, first paragraph per section, lists
        lines = content.split("\n")
        skeleton = []
        blank_count = 0
        after_heading = False
        kept_first_para = False

        for line in lines:
            stripped = line.strip()

            # Always keep headings
            if stripped.startswith("#"):
                skeleton.append(line)
                after_heading = True
                kept_first_para = False
                blank_count = 0
                continue

            # Keep list items
            if re.match(r"^[-*+|]\s", stripped) or re.match(r"^\d+\.", stripped):
                skeleton.append(line)
                continue

            # Keep tables
            if "|" in stripped and stripped.startswith("|"):
                skeleton.append(line)
                continue

            # Keep code fence markers
            if stripped.startswith("```"):
                skeleton.append(line)
                continue

            # Keep first paragraph after each heading
            if after_heading and not kept_first_para:
                if stripped == "":
                    blank_count += 1
                    if blank_count >= 2:
                        kept_first_para = True
                else:
                    skeleton.append(line)
                    blank_count = 0
                continue

        return "\n".join(skeleton)


def _chunk_content(content, content_type):
    if content_type == "code":
        blocks = re.split(
            r"(^(?:def |class |async def |struct |impl |fn |pub fn |export ))",
            content, flags=re.MULTILINE,
        )
        stitched = []
        if blocks and not re.match(
            r"^(?:def |class |async def |struct |impl |fn |pub fn |export )", blocks[0]
        ):
            stitched.append(blocks[0])
            blocks = blocks[1:]
        for j in range(0, len(blocks), 2):
            if j + 1 < len(blocks):
                stitched.append(blocks[j] + blocks[j + 1])
            elif blocks[j].strip():
                stitched.append(blocks[j])

        chunks = []
        current = ""
        for block in stitched:
            if len(current) + len(block) < CHUNK_THRESHOLD:
                current += block
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = block
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [content]
    else:
        paragraphs = re.split(r"\n\s*\n", content)
        chunks = []
        current = ""
        for p in paragraphs:
            if len(current) + len(p) < CHUNK_THRESHOLD:
                current += p + "\n\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = p + "\n\n"
        if current.strip():
            chunks.append(current.strip())
        return chunks if chunks else [content]


def find_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            _, ext = os.path.splitext(fname)
            if ext not in CODE_EXTS and ext not in TEXT_EXTS:
                continue
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


def ingest_one_file(index, total, path, content_type, size):
    """Ingest a single file (called from thread pool)."""
    rel = os.path.relpath(path, PROJECTS_ROOT)
    abs_path = os.path.abspath(path)

    try:
        content = open(path, encoding="utf-8", errors="replace").read()
    except Exception as e:
        _log(f"  [{index}/{total}] SKIP {rel}: {e}")
        with _stats_lock:
            _stats["skipped"] += 1
        return

    # Large files: extract skeleton (signatures, imports, docstrings, headings)
    if size > SKELETON_THRESHOLD:
        original_size = len(content)
        content = _extract_skeleton(content, content_type)
        reduction = (1 - len(content) / original_size) * 100
        _log(f"  [{index}/{total}] SKELETON {rel} ({original_size:,}B -> {len(content):,}B, -{reduction:.0f}%)")
        if len(content.strip()) < MIN_SIZE:
            _log(f"  [{index}/{total}] SKIP {rel} (skeleton too small)")
            _save_progress(abs_path)
            with _stats_lock:
                _stats["skipped"] += 1
            return

    # Chunk if still large after skeleton extraction
    if len(content) > CHUNK_THRESHOLD:
        chunks = _chunk_content(content, content_type)
        # Hard-cut fallback: if any chunk is still over threshold, slice it
        final_chunks = []
        for c in chunks:
            if len(c) > CHUNK_THRESHOLD:
                for i in range(0, len(c), CHUNK_THRESHOLD):
                    piece = c[i:i + CHUNK_THRESHOLD].strip()
                    if piece:
                        final_chunks.append(piece)
            else:
                final_chunks.append(c)
        chunks = final_chunks
        label = f"{rel} ({size:,}B, {len(chunks)} chunks)"
    else:
        chunks = [content]
        label = f"{rel} ({size:,}B)"

    # Each file gets its own HTTP client (thread-safe)
    client = httpx.Client(timeout=300)
    file_genes = 0
    t0 = time.time()

    for chunk in chunks:
        try:
            resp = client.post(f"{HELIX_URL}/ingest", json={
                "content": chunk,
                "content_type": content_type,
                "metadata": {"path": abs_path},
            })
            if resp.status_code == 200:
                file_genes += resp.json().get("count", 0)
        except httpx.ReadTimeout:
            pass
        except httpx.ConnectError:
            with _stats_lock:
                _stats["errors"] += 1
            _log(f"  [{index}/{total}] SERVER DOWN {label}")
            client.close()
            return
        except Exception:
            pass

    client.close()
    dt = time.time() - t0

    with _stats_lock:
        _stats["genes"] += file_genes
        _stats["files_done"] += 1
        if file_genes == 0:
            _stats["errors"] += 1

    _save_progress(abs_path)
    _log(f"  [{index}/{total}] {label} -> {file_genes} genes ({dt:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Parallel deep ingest into Helix")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent workers (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Count files only")
    args = parser.parse_args()

    files = find_files(PROJECTS_ROOT)
    files.sort(key=lambda x: x[0])

    # Load progress for resume
    done = _load_progress()
    remaining = [(p, ct, s) for p, ct, s in files if os.path.abspath(p) not in done]

    print(f"Total files: {len(files)}")
    print(f"Already done: {len(done)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Workers: {args.workers}")
    print()

    if args.dry_run:
        projects = {}
        for path, _, size in remaining:
            proj = os.path.relpath(path, PROJECTS_ROOT).split(os.sep)[0]
            projects.setdefault(proj, [0, 0])
            projects[proj][0] += 1
            projects[proj][1] += size
        for proj, (count, size) in sorted(projects.items(), key=lambda x: -x[1][0]):
            print(f"  {proj}: {count} files ({size / 1024:.0f} KB)")
        print(f"\nEstimated time ({args.workers} workers): {len(remaining) * 15 / args.workers / 60:.0f} minutes")
        return

    # Check server
    try:
        health = httpx.Client(timeout=10).get(f"{HELIX_URL}/health").json()
        print(f"Server: {health['status']}, ribosome: {health['ribosome']}, genes: {health['genes']}")
    except Exception:
        print(f"Cannot reach Helix at {HELIX_URL}")
        sys.exit(1)

    print()
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, (path, content_type, size) in enumerate(remaining):
            future = executor.submit(
                ingest_one_file, i + 1, len(remaining), path, content_type, size
            )
            futures[future] = path

        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    _log(f"  Worker error: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted. Progress saved.")
            executor.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - t_start

    # Final stats
    try:
        stats = httpx.Client(timeout=10).get(f"{HELIX_URL}/stats").json()
        print(f"\n=== COMPLETE ===")
        print(f"New genes: {_stats['genes']}")
        print(f"Files done: {_stats['files_done']}, Errors: {_stats['errors']}, Skipped: {_stats['skipped']}")
        print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"Genome: {stats['total_genes']} genes, {stats['compression_ratio']:.1f}x")
        print(f"Raw: {stats['total_chars_raw']/1024/1024:.1f}MB -> Compressed: {stats['total_chars_compressed']/1024/1024:.1f}MB")
    except Exception:
        print(f"\nDone. Genes: {_stats['genes']}, Errors: {_stats['errors']}, Time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
