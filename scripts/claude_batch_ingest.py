"""
Claude API batch ingest — Messages Batches API, 50% cost discount.

Two-phase design:
  PHASE 1 (submit): Walk roots, chunk files, build batch requests, submit
                    to /v1/messages/batches. Writes a .batch-manifest.json
                    with batch_id + custom_id → (fpath, strand_idx) map.
  PHASE 2 (ingest): Poll batch status, download results when ready, parse
                    each response into a Gene, upsert to genome.

Why two phases:
  The batch API is non-real-time (typically minutes, up to 24h). You submit,
  walk away, come back later and run phase 2. The manifest file survives
  process restarts.

NOTE: batches bypass Headroom and go direct to api.anthropic.com. Headroom
is a streaming proxy for /v1/messages; it does not proxy /v1/messages/batches.
The batch API discount is 50% flat — more savings than Headroom compression
would provide on inputs alone.

Usage:
    # Submit a batch (returns batch_id, writes manifest)
    python scripts/claude_batch_ingest.py submit \
        --roots F:/Projects/helix-context/helix_context \
        --manifest .batch-helix-core.json

    # Check batch status
    python scripts/claude_batch_ingest.py status --manifest .batch-helix-core.json

    # Poll until done, then ingest results into genome
    python scripts/claude_batch_ingest.py ingest --manifest .batch-helix-core.json

    # One-shot: submit + block until done + ingest
    python scripts/claude_batch_ingest.py run \
        --roots F:/Projects/helix-context/helix_context \
        --manifest .batch-helix-core.json

Requires:
    - ANTHROPIC_API_KEY in env
    - anthropic SDK >= 0.40 (for messages.batches support)
    - helix.toml for defaults
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_context.codons import CodonChunker, CodonEncoder
from helix_context.config import load_config
from helix_context.genome import Genome
from helix_context.ribosome import _PACK_SYSTEM  # reuse the stable system prompt
from helix_context.schemas import EpigeneticMarkers, Gene, PromoterTags

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.claude_batch")

TEXT_EXTS = {".txt", ".md", ".cfg", ".ini", ".conf", ".toml"}
CODE_EXTS = {".py", ".rs", ".ts", ".tsx", ".js", ".jsx", ".json", ".yaml", ".yml"}
INGEST_EXTS = TEXT_EXTS | CODE_EXTS
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", "dist", "build",
    ".pytest_cache", "target", ".claude", "knowledge",
}
MAX_FILE_SIZE = 100_000
MIN_FILE_SIZE = 100


def walk(roots: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in INGEST_EXTS:
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    continue
                if MIN_FILE_SIZE <= size <= MAX_FILE_SIZE:
                    out.append((fpath, ext))
    return out


def _make_client():
    import anthropic
    return anthropic.Anthropic()  # respects ANTHROPIC_API_KEY; bypasses BASE_URL for batches


def _build_prompt(content: str, encoder: CodonEncoder, ct: str) -> str:
    if ct == "code":
        groups = encoder.chunk_code(content)
    else:
        groups = encoder.chunk_text(content)
    numbered = "\n".join(f"[Group {i}]: {' '.join(g)}" for i, g in enumerate(groups))
    return f"Encode the following content into codons:\n\n{numbered}"


def cmd_submit(args) -> int:
    cfg = load_config(args.config)
    model = args.model or cfg.ribosome.claude_model
    max_tokens = cfg.budget.ribosome_tokens

    files = walk(args.roots)
    log.info("Found %d files across %d root(s)", len(files), len(args.roots))
    if args.limit:
        files = files[: args.limit]

    chunker = CodonChunker(max_chars_per_strand=4000)
    encoder = CodonEncoder()

    requests: list[dict[str, Any]] = []
    id_map: dict[str, dict[str, Any]] = {}

    for fpath, ext in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            continue
        ct = "code" if ext in CODE_EXTS else "text"
        strands = chunker.chunk(content, content_type=ct)
        for i, strand in enumerate(strands):
            custom_id = f"gene-{len(requests):06d}"
            prompt = _build_prompt(strand.content, encoder, ct)
            requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": _PACK_SYSTEM,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })
            id_map[custom_id] = {
                "fpath": fpath,
                "ext": ext,
                "content_type": ct,
                "strand_idx": i,
                "raw_content": strand.content,
                "is_fragment": strand.is_fragment,
            }

    if not requests:
        log.error("No requests built — nothing to submit")
        return 2

    log.info("Submitting batch of %d requests (model=%s)", len(requests), model)
    client = _make_client()
    batch = client.messages.batches.create(requests=requests)
    log.info("Batch submitted: id=%s processing_status=%s", batch.id, batch.processing_status)

    manifest = {
        "batch_id": batch.id,
        "model": model,
        "submitted_at": time.time(),
        "request_count": len(requests),
        "id_map": id_map,
        "db": args.db,
    }
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    log.info("Manifest written: %s", args.manifest)
    return 0


def cmd_status(args) -> int:
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    client = _make_client()
    batch = client.messages.batches.retrieve(manifest["batch_id"])
    log.info("batch_id=%s", batch.id)
    log.info("  processing_status=%s", batch.processing_status)
    log.info("  request_counts=%s", batch.request_counts)
    log.info("  created_at=%s", batch.created_at)
    log.info("  ended_at=%s", batch.ended_at)
    return 0 if batch.processing_status == "ended" else 1


def cmd_ingest(args) -> int:
    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    from helix_context.genome import Genome as _Genome
    client = _make_client()

    # Poll until done if asked, otherwise fail fast if not done
    while True:
        batch = client.messages.batches.retrieve(manifest["batch_id"])
        if batch.processing_status == "ended":
            break
        if not args.wait:
            log.error("Batch not ended yet (status=%s). Use --wait to block.", batch.processing_status)
            return 1
        log.info("Batch still %s — counts=%s. Sleeping 30s.", batch.processing_status, batch.request_counts)
        time.sleep(30)

    log.info("Batch ended. Streaming results…")
    genome = _Genome(path=manifest["db"], synonym_map={}, splade_enabled=True, entity_graph=True)
    id_map = manifest["id_map"]
    stats = {"ok": 0, "errors": 0}

    for result in client.messages.batches.results(manifest["batch_id"]):
        custom_id = result.custom_id
        meta = id_map.get(custom_id)
        if not meta:
            continue
        if result.result.type != "succeeded":
            log.warning("%s → %s", custom_id, result.result.type)
            stats["errors"] += 1
            continue
        try:
            text = result.result.message.content[0].text
            parsed = json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
            codon_meanings = [c.get("meaning", f"c{i}") for i, c in enumerate(parsed.get("codons", []))]
            promoter = parsed.get("promoter", {}) or {}
            content = meta["raw_content"]
            gene = Gene(
                gene_id=_Genome.make_gene_id(content),
                content=content,
                complement=parsed.get("complement", content[:500]),
                codons=codon_meanings,
                promoter=PromoterTags(
                    domains=promoter.get("domains", []),
                    entities=promoter.get("entities", []),
                    intent=promoter.get("intent", ""),
                    summary=promoter.get("summary", ""),
                ),
                epigenetics=EpigeneticMarkers(),
            )
            gene.source_id = meta["fpath"]
            gene.is_fragment = meta["is_fragment"]
            genome.upsert_gene(gene)
            stats["ok"] += 1
        except Exception as exc:
            log.warning("Parse/upsert failed for %s: %s", custom_id, exc)
            stats["errors"] += 1

    log.info("Ingest complete: %d ok, %d errors", stats["ok"], stats["errors"])
    return 0


def cmd_run(args) -> int:
    """Submit → wait → ingest."""
    rc = cmd_submit(args)
    if rc != 0:
        return rc
    args.wait = True
    return cmd_ingest(args)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Shared args
    def _common(p):
        p.add_argument("--db", default="genome.db")
        p.add_argument("--config", default="helix.toml")
        p.add_argument("--manifest", required=True)

    p_submit = sub.add_parser("submit")
    _common(p_submit)
    p_submit.add_argument("--roots", nargs="+", required=True)
    p_submit.add_argument("--model", default=None)
    p_submit.add_argument("--limit", type=int, default=0)
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser("status")
    _common(p_status)
    p_status.set_defaults(func=cmd_status)

    p_ingest = sub.add_parser("ingest")
    _common(p_ingest)
    p_ingest.add_argument("--wait", action="store_true", help="Poll until batch ends")
    p_ingest.set_defaults(func=cmd_ingest)

    p_run = sub.add_parser("run")
    _common(p_run)
    p_run.add_argument("--roots", nargs="+", required=True)
    p_run.add_argument("--model", default=None)
    p_run.add_argument("--limit", type=int, default=0)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
