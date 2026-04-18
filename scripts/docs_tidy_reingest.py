"""Re-ingest docs/*.md after the tidy-up move.

Walks docs/ recursively, POSTs each .md to /ingest with the new path in
metadata. Old source_ids remain as orphan genes (intentional — the probe
measures how the old vs new co-exist).

Usage:
    python scripts/docs_tidy_reingest.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HELIX = "http://127.0.0.1:11437"
DOCS_ROOT = Path(__file__).resolve().parent.parent / "docs"
SKIP_DIRS = {"FUTURE", "specs", "plans", "collab"}


def ingest_one(path: Path) -> dict:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"error": f"read failed: {e}"}
    if not content.strip():
        return {"skipped": "empty"}
    body = json.dumps(
        {
            "content": content,
            "content_type": "markdown",
            "metadata": {
                # Helix reads metadata["path"] for source_id (see
                # context_manager.py:397). Must be "path", not "source_id".
                "path": str(path).replace("\\", "/"),
                "tidy_reingest": True,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{HELIX}/ingest",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    targets = []
    for md in DOCS_ROOT.rglob("*.md"):
        rel_parts = md.relative_to(DOCS_ROOT).parts
        if rel_parts and rel_parts[0] in SKIP_DIRS:
            continue
        targets.append(md)

    print(f"[reingest] {len(targets)} docs under docs/ (excluding {SKIP_DIRS})")
    ok = err = 0
    t0 = time.time()
    for i, path in enumerate(targets, 1):
        res = ingest_one(path)
        if "error" in res:
            err += 1
            print(f"  [{i}/{len(targets)}] ERR {path.name}: {res['error']}")
        else:
            ok += 1
            print(f"  [{i}/{len(targets)}] OK  {path.relative_to(DOCS_ROOT)}")
    dt = time.time() - t0
    print(f"[reingest] done in {dt:.1f}s: {ok} ok, {err} errors")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
