"""
Ingest all text-based files from F:\\ into the Helix Context genome.
Safe: skips secrets, binaries, build artifacts, already-ingested files.
"""

import os
import re
import sys
import time
import sqlite3

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import json
import urllib.request
import urllib.error

# === Configuration ===
HELIX_URL = "http://127.0.0.1:11437/ingest"
GENOME_DB = "F:/Projects/helix-context/genome.db"
ROOT = "F:/"
CHUNK_SIZE = 4000
MIN_SIZE = 100
MAX_SIZE = 100_000
LOG_SKIP_MAX = 50  # max skip lines per category before suppressing

# Directories to skip entirely
SKIP_DIRS = {
    "windows", "$recycle.bin", "$sysreset", "system volume information",
    "wpsystem", "tmp", "node_modules", "__pycache__", ".git", "target",
    "dist", "build", ".next", ".venv", "venv", "site-packages", ".egg-info",
    "saves", ".mypy_cache", ".pytest_cache", ".tox", ".nox", ".cache",
    ".cargo", ".rustup", "deps", ".gradle", "obj", "bin",
}

# Top-level dirs to skip
SKIP_TOP = {
    "windows", "$recycle.bin", "$sysreset", "system volume information",
    "wpsystem", "tmp",
}

# File extensions to ingest
CODE_EXTS = {".lua", ".py", ".rs", ".ts", ".js", ".json", ".toml", ".yaml", ".yml"}
TEXT_EXTS = {".md", ".txt", ".rst", ".cfg", ".ini", ".conf", ".html"}
ALL_EXTS = CODE_EXTS | TEXT_EXTS

# Secret file patterns
SECRET_NAMES = {
    ".env", "credentials", "secrets", "token", "auth.json",
}
SECRET_EXTS = {".pem", ".key", ".cert"}

# Secret content patterns
SECRET_PATTERNS = re.compile(
    r'(API_KEY\s*=|SECRET\s*=|TOKEN\s*=|PASSWORD\s*=|aws_secret|PRIVATE.KEY)',
    re.IGNORECASE
)
LONG_HEX_B64 = re.compile(r'[a-zA-Z0-9+/=]{40,}')

# Binary/skip extensions
BINARY_EXTS = {
    ".exe", ".dll", ".so", ".bin", ".dat", ".db", ".sqlite", ".whl", ".egg",
    ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico",
    ".ogg", ".wav", ".mp3", ".mp4", ".avi", ".mkv", ".webm", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".pyc", ".pyo", ".class", ".o", ".obj", ".lib", ".a",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".pak", ".upk", ".uasset", ".umap",  # game assets
    ".dds", ".tga", ".psd",  # textures
    ".fbx", ".blend", ".3ds",  # 3D models
    ".bk2", ".bik",  # video
    ".bnk", ".wem",  # audio banks
    ".cache", ".tmp",
}


def is_secret_filename(name):
    lower = name.lower()
    if lower.startswith(".env"):
        return True
    if lower.startswith("service-account") and lower.endswith(".json"):
        return True
    base = os.path.splitext(lower)[0]
    ext = os.path.splitext(lower)[1]
    if ext in SECRET_EXTS:
        return True
    for pat in SECRET_NAMES:
        if lower == pat or lower.startswith(pat + "."):
            return True
    return False


def has_secret_content(content):
    """Check first 50 lines and full content for secret patterns."""
    lines = content.split('\n')
    for line in lines[:200]:
        if SECRET_PATTERNS.search(line):
            return True
        # Long hex/b64 strings that look like keys (but not in normal code context)
        stripped = line.strip()
        if stripped and not stripped.startswith(('#', '//', '--', '/*')):
            matches = LONG_HEX_B64.findall(stripped)
            for m in matches:
                # Filter out common false positives (hashes in lock files, etc.)
                if len(m) > 60 and not any(c in stripped.lower() for c in ['hash', 'sha', 'integrity', 'checksum', 'commit']):
                    return True
    return False


def content_type_for(ext):
    return "code" if ext in CODE_EXTS else "text"


def normalize_path(p):
    """Normalize path for comparison."""
    return p.replace("\\", "/").lower()


def load_existing_sources():
    """Load already-ingested source_ids from genome.db."""
    conn = sqlite3.connect(GENOME_DB)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT source_id FROM genes WHERE source_id IS NOT NULL")
    sources = set()
    for (sid,) in cur.fetchall():
        sources.add(normalize_path(sid))
    conn.close()
    return sources


def collect_files():
    """Walk F:\\ and collect eligible files."""
    files = []
    skipped = {"dir": 0, "ext": 0, "size": 0, "secret_name": 0, "binary": 0}
    dir_counts = {}

    for entry in os.scandir(ROOT):
        if not entry.is_dir():
            continue
        dirname_lower = entry.name.lower()
        if dirname_lower in SKIP_TOP:
            print(f"  SKIP top-level dir: {entry.name}")
            continue

        top_dir = entry.name
        count = 0

        for dirpath, dirnames, filenames in os.walk(entry.path, topdown=True):
            # Prune skip dirs
            dirnames[:] = [
                d for d in dirnames
                if d.lower() not in SKIP_DIRS
            ]

            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                ext = os.path.splitext(fname)[1].lower()

                # Extension check
                if ext not in ALL_EXTS:
                    skipped["ext"] += 1
                    continue

                # Secret filename check
                if is_secret_filename(fname):
                    skipped["secret_name"] += 1
                    continue

                # Size check
                try:
                    sz = os.path.getsize(fpath)
                except OSError:
                    continue

                if sz < MIN_SIZE:
                    skipped["size"] += 1
                    continue
                if sz > MAX_SIZE:
                    # Allow .js only up to MAX_SIZE (skip minified)
                    skipped["size"] += 1
                    continue

                # Log files > 50KB
                if ext == ".log" and sz > 50_000:
                    skipped["size"] += 1
                    continue

                files.append((fpath, ext, sz, top_dir))
                count += 1

        dir_counts[top_dir] = count

    return files, skipped, dir_counts


def ingest_file(fpath, ext, existing):
    """Ingest a single file. Returns (status, chunks_sent)."""
    norm = normalize_path(fpath)
    if norm in existing:
        return "skip_existing", 0

    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return "read_error", 0

    if len(content.strip()) < 50:
        return "skip_tiny", 0

    # Secret content check
    if has_secret_content(content):
        return "skip_secret", 0

    ct = content_type_for(ext)
    source_path = fpath.replace("/", "\\")

    # Chunk if needed
    chunks = []
    if len(content) > CHUNK_SIZE:
        for i in range(0, len(content), CHUNK_SIZE):
            chunks.append(content[i:i + CHUNK_SIZE])
    else:
        chunks = [content]

    for chunk in chunks:
        payload = json.dumps({
            "content": chunk,
            "content_type": ct,
            "metadata": {"path": source_path}
        }).encode("utf-8")

        req = urllib.request.Request(
            HELIX_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            if resp.status != 200:
                return f"http_{resp.status}", len(chunks)
        except urllib.error.HTTPError as e:
            return f"http_{e.code}", len(chunks)
        except Exception as e:
            return f"error:{type(e).__name__}", len(chunks)

    return "ok", len(chunks)


def main():
    print("=" * 60)
    print("Helix Context F:\\ Drive Ingest")
    print("=" * 60)

    # Step 1: Load existing sources
    print("\n[1/4] Loading existing source_ids from genome.db...")
    existing = load_existing_sources()
    print(f"  Found {len(existing)} already-ingested sources")

    # Step 2: Collect eligible files
    print("\n[2/4] Scanning F:\\ for eligible files...")
    files, skipped, dir_counts = collect_files()
    print(f"\n  === File Survey ===")
    print(f"  Total eligible files: {len(files)}")
    for d, c in sorted(dir_counts.items(), key=lambda x: -x[1]):
        if c > 0:
            print(f"    {d}: {c} files")
    print(f"\n  Skipped: ext={skipped['ext']}, size={skipped['size']}, "
          f"secret_name={skipped['secret_name']}, binary={skipped['binary']}")

    # Count how many are new vs existing
    new_files = []
    skip_existing = 0
    for fpath, ext, sz, top_dir in files:
        if normalize_path(fpath) in existing:
            skip_existing += 1
        else:
            new_files.append((fpath, ext, sz, top_dir))

    print(f"\n  Already ingested (will skip): {skip_existing}")
    print(f"  New files to ingest: {len(new_files)}")

    if not new_files:
        print("\nNothing new to ingest. Done!")
        return

    # Show breakdown of new files by directory
    new_by_dir = {}
    for fpath, ext, sz, top_dir in new_files:
        new_by_dir[top_dir] = new_by_dir.get(top_dir, 0) + 1
    print(f"\n  New files by directory:")
    for d, c in sorted(new_by_dir.items(), key=lambda x: -x[1]):
        print(f"    {d}: {c}")

    # Step 3: Ingest
    print(f"\n[3/4] Ingesting {len(new_files)} files...")
    start = time.time()
    stats = {"ok": 0, "skip_secret": 0, "skip_tiny": 0, "read_error": 0, "errors": 0}
    total_chunks = 0
    error_details = []

    for i, (fpath, ext, sz, top_dir) in enumerate(new_files, 1):
        status, chunks = ingest_file(fpath, ext, existing)

        if status == "ok":
            stats["ok"] += 1
            total_chunks += chunks
            # Add to existing so we don't re-process
            existing.add(normalize_path(fpath))
        elif status == "skip_secret":
            stats["skip_secret"] += 1
            if stats["skip_secret"] <= LOG_SKIP_MAX:
                print(f"  SKIP (potential secret): {fpath}")
        elif status == "skip_tiny":
            stats["skip_tiny"] += 1
        elif status == "read_error":
            stats["read_error"] += 1
        elif status == "skip_existing":
            pass  # shouldn't happen since we pre-filtered
        else:
            stats["errors"] += 1
            error_details.append((fpath, status))
            if len(error_details) <= 10:
                print(f"  ERROR: {fpath} -> {status}")

        # Progress
        if i % 10 == 0 or i == len(new_files):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            print(f"  [{i}/{len(new_files)}] ok={stats['ok']} err={stats['errors']} "
                  f"secret={stats['skip_secret']} tiny={stats['skip_tiny']} "
                  f"({rate:.1f} files/s, {elapsed:.0f}s elapsed)")

    # Step 4: Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"[4/4] INGEST COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Files ingested:      {stats['ok']}")
    print(f"  Total chunks sent:   {total_chunks}")
    print(f"  Skipped (secret):    {stats['skip_secret']}")
    print(f"  Skipped (tiny):      {stats['skip_tiny']}")
    print(f"  Read errors:         {stats['read_error']}")
    print(f"  HTTP/other errors:   {stats['errors']}")
    print(f"  Total time:          {elapsed:.1f}s")
    print(f"  Rate:                {stats['ok'] / elapsed:.1f} files/s" if elapsed > 0 else "")

    if error_details:
        print(f"\n  Error details (first 10):")
        for path, err in error_details[:10]:
            print(f"    {path}: {err}")


if __name__ == "__main__":
    main()
