"""
Seed a genome from files — bulk ingest a directory of documents.

Usage:
    python examples/seed_genome.py path/to/docs/
    python examples/seed_genome.py path/to/src/ --type code
    python examples/seed_genome.py README.md

Prerequisites:
    pip install helix-context
    ollama pull gemma4:e2b
    helix server running: python -m helix_context.server
"""

import argparse
import sys
from pathlib import Path

import httpx


def seed(target: str, content_type: str, helix_url: str) -> None:
    path = Path(target)
    client = httpx.Client(base_url=helix_url, timeout=300)

    # Check server is up
    try:
        health = client.get("/health").json()
        print(f"Helix server: {health['status']}, ribosome: {health['ribosome']}, "
              f"genes: {health['genes']}")
    except Exception as exc:
        # Surface the underlying reason (connection refused, timeout,
        # DNS failure, bad JSON, ...) so users don't have to guess why
        # the probe failed.
        print(f"Cannot reach Helix at {helix_url}: {exc}")
        print(f"  Start the server first:")
        print(f"  python -m uvicorn helix_context.server:app --host 127.0.0.1 --port 11437")
        sys.exit(1)

    # Collect files
    if path.is_file():
        files = [path]
    elif path.is_dir():
        exts = {"code": [".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp"],
                "text": [".txt", ".md", ".rst", ".html"]}
        allowed = exts.get(content_type, exts["text"] + exts["code"])
        files = sorted(f for f in path.rglob("*") if f.suffix in allowed and f.stat().st_size < 100_000)
    else:
        print(f"Not found: {target}")
        sys.exit(1)

    print(f"\nIngesting {len(files)} files as '{content_type}'...\n")

    total_genes = 0
    for f in files:
        content = f.read_text(encoding="utf-8", errors="replace")
        if not content.strip():
            continue

        ctype = "code" if f.suffix in [".py", ".js", ".ts", ".rs", ".go", ".java", ".c", ".cpp"] else "text"
        if content_type != "auto":
            ctype = content_type

        try:
            resp = client.post("/ingest", json={
                "content": content,
                "content_type": ctype,
                "metadata": {"path": str(f)},
            })
            if resp.status_code == 200:
                count = resp.json()["count"]
                total_genes += count
                print(f"  {f.name}: {count} genes ({len(content)} chars)")
            else:
                print(f"  {f.name}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  {f.name}: ERROR {e}")

    stats = client.get("/stats").json()
    print(f"\nDone. Genome: {stats['total_genes']} genes, "
          f"{stats['compression_ratio']:.1f}x compression")
    print(f"Raw: {stats['total_chars_raw']:,} chars -> "
          f"Compressed: {stats['total_chars_compressed']:,} chars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed a Helix genome from files")
    parser.add_argument("target", help="File or directory to ingest")
    parser.add_argument("--type", default="auto", choices=["text", "code", "auto"],
                        help="Content type (default: auto-detect by extension)")
    parser.add_argument("--url", default="http://127.0.0.1:11437",
                        help="Helix server URL (default: http://127.0.0.1:11437)")
    args = parser.parse_args()
    seed(args.target, args.type, args.url)
