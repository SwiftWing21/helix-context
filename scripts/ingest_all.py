r"""
Ingest sources into the genome in one pass.

Default sources (override with --sources):
  - F:\Projects (code, docs, configs)
  - F:\SteamLibrary (game Lua/JSON/configs + manifests)
  - F:\OpenModels (GGUF model headers)
  - E:\SteamLibrary, E:\Program Files, E:\NetMose

Skips binaries, limits file size to 200KB to avoid stalls on
massive JSON/XML blobs. Commits every 100 genes. Each gene receives
forward-only provenance via apply_metadata_hints + apply_provenance.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path

from helix_context.tagger import CpuTagger
from helix_context.genome import Genome
from helix_context.codons import CodonChunker
from helix_context.provenance import apply_metadata_hints, apply_provenance
from helix_context.sharding import (
    corpus_shard_db,
    agent_shard_db,
    main_db_path as _main_db_path,
)
from helix_context.shard_schema import (
    open_main_db,
    init_main_db,
    register_shard,
    upsert_source_index,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.all")

TEXT_EXTS = {".txt", ".md", ".cfg", ".ini", ".conf", ".properties", ".vdf", ".acf"}
CODE_EXTS = {
    ".lua", ".py", ".cs", ".js", ".json", ".yaml", ".yml", ".toml",
    ".bat", ".sh", ".html", ".rs", ".go", ".java", ".c", ".cpp", ".h",
    ".rb", ".ts", ".tsx", ".jsx", ".sql", ".r", ".ps1",
}
INGEST_EXTS = TEXT_EXTS | CODE_EXTS

SKIP_DIRS = {
    "shadercache", "temp", "downloading", "depotcache", "__pycache__",
    ".git", "node_modules", "Mono", "MonoBleedingEdge", ".venv", "venv",
    "dist", "build", ".pytest_cache", "target", ".claude",
    "$RECYCLE.BIN", "System Volume Information", "WpSystem",
    "WUDownloadCache", "WindowsApps",
    # Keep benchmark prompts, docs, and result artifacts out of the
    # live working genome so they cannot be retrieved as evidence.
    "benchmarks",
    # Archived genome backups — never re-ingest a previous genome.
    "Helix-backup blobs",
    # Unity / game engine runtime directories with no ingestable text.
    "D3D12",
}

MAX_FILE_SIZE = 200_000   # 200KB — avoids stalls on giant JSON/XML
MIN_FILE_SIZE = 50


def ingest_tree(root, genome, tagger, chunker, stats):
    """Walk a directory tree and ingest text/code files."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in INGEST_EXTS:
                stats["skipped"] += 1
                continue

            fpath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue

            if size < MIN_FILE_SIZE or size > MAX_FILE_SIZE:
                stats["skipped"] += 1
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                stats["errors"] += 1
                continue

            ct = "code" if ext in CODE_EXTS else "text"
            strands = chunker.chunk(content, content_type=ct)
            total_strands = len(strands)
            try:
                file_mtime = os.path.getmtime(fpath)
            except OSError:
                file_mtime = None
            metadata = {"mtime": file_mtime, "repo_root": root}
            for i, strand in enumerate(strands):
                gene = tagger.pack(
                    strand.content,
                    content_type=ct,
                    source_id=fpath,
                    sequence_index=i,
                )
                gene.is_fragment = strand.is_fragment
                apply_metadata_hints(
                    gene,
                    metadata,
                    content_type=ct,
                    total_strands=total_strands,
                )
                apply_provenance(
                    gene,
                    source_path=fpath,
                    observed_at=gene.observed_at,
                    content_type=ct,
                )
                genome.upsert_gene(gene)
                stats["genes"] += 1

            stats["files"] += 1

            if stats["genes"] % 200 == 0 and stats["genes"] > 0:
                elapsed = time.perf_counter() - stats["t0"]
                log.info(
                    "[%d files, %d genes] %.1f genes/s | %s",
                    stats["files"], stats["genes"],
                    stats["genes"] / elapsed,
                    os.path.basename(dirpath),
                )


def ingest_models(root, genome, tagger, stats):
    """Ingest Ollama model headers (GGUF metadata)."""
    # Reuse the model ingester logic inline
    try:
        from scripts.ingest_models import read_ollama_manifests, model_to_gene_content
    except ImportError:
        # Direct import if run from scripts dir
        sys.path.insert(0, os.path.dirname(__file__))
        from ingest_models import read_ollama_manifests, model_to_gene_content

    models = read_ollama_manifests(root)
    for model in models:
        content = model_to_gene_content(model)
        manifest_path = model["manifest_path"]
        try:
            manifest_mtime = os.path.getmtime(manifest_path)
        except OSError:
            manifest_mtime = None
        gene = tagger.pack(content, content_type="text", source_id=manifest_path)
        metadata = {
            "mtime": manifest_mtime,
            "repo_root": root,
            "source_kind": "config",  # GGUF manifests are metadata, not code
        }
        apply_metadata_hints(gene, metadata, content_type="text", total_strands=1)
        apply_provenance(
            gene,
            source_path=manifest_path,
            observed_at=gene.observed_at,
            content_type="text",
        )
        genome.upsert_gene(gene)
        stats["genes"] += 1
        stats["files"] += 1
        log.info("  Model: %s (%.1f GB, %d tensors)",
                 model["name"], model["size_gb"],
                 model["gguf"]["tensor_count"] if model.get("gguf") else 0)


def _parse_source_arg(spec: str) -> tuple[str, str, str]:
    """Parse a ``path=label[:category]`` source spec.

    ``category`` defaults to ``"reference"`` (external corpus) in sharded
    mode; use ``"participant"`` for operator-owned sources (e.g.,
    ``F:/Projects=projects:participant``) or ``"org"`` for shared-team
    sources. ``"agent"`` is reserved for ``--agent-source`` entries.

    Windows drive colons collide with the category separator; the category
    is only recognised if a ``:`` appears *after* the ``=label`` portion.
    """
    eq_pos = spec.find("=")
    last_colon = spec.rfind(":")
    if eq_pos >= 0 and last_colon > eq_pos:
        path_label = spec[:last_colon]
        category = spec[last_colon + 1:] or "reference"
    else:
        path_label = spec
        category = "reference"
    if "=" in path_label:
        path, label = path_label.split("=", 1)
    else:
        path, label = path_label, os.path.basename(path_label.rstrip("/\\")) or path_label
    return path, label, category


def _parse_agent_source(spec: str) -> tuple[str, str]:
    """Parse a ``handle=path`` agent-source spec."""
    if "=" not in spec:
        raise SystemExit(f"--agent-source expects HANDLE=PATH, got: {spec!r}")
    handle, path = spec.split("=", 1)
    if not handle or not path:
        raise SystemExit(f"--agent-source expects HANDLE=PATH, got: {spec!r}")
    return handle.strip(), path.strip()


_DEFAULT_SOURCES: list[tuple[str, str, str]] = [
    ("F:/Projects", "projects", "participant"),
    ("F:/SteamLibrary", "steam-f", "reference"),
    ("F:/OpenModels", "models", "reference"),
    ("E:/SteamLibrary", "steam-e", "reference"),
    ("E:/Program Files", "programs-e", "reference"),
    ("E:/NetMose", "netmose", "reference"),
]


def _copy_indexes_from_shard(main_conn, shard: Genome, shard_name: str) -> int:
    """Copy per-gene provenance + fingerprint rows into main.db.

    Populates ``source_index`` (for packet freshness + cross-shard provenance
    joins) and ``fingerprint_index`` (which ``ShardRouter.route`` queries to
    decide which shards to hit; if empty the router returns no shards and
    every cross-shard query returns empty).
    """
    rows = shard.conn.execute(
        "SELECT gene_id, source_id, repo_root, source_kind, observed_at, "
        "mtime, content_hash, volatility_class, authority_class, support_span, "
        "last_verified_at, promoter, key_values, is_fragment "
        "FROM genes"
    ).fetchall()
    if not rows:
        return 0
    now = time.time()
    si_payload = []
    fp_payload = []
    for (gid, src_id, repo_root, source_kind, observed_at, mtime, chash,
         vol, auth, span, last_verif, promoter_blob, kv_blob, is_frag) in rows:
        si_payload.append((
            gid, shard_name, src_id, repo_root, source_kind,
            observed_at, mtime, chash, vol or "medium", auth or "primary",
            span, last_verif, None, now,
        ))
        domains_json = None
        entities_json = None
        if promoter_blob:
            try:
                p = json.loads(promoter_blob)
                domains_json = json.dumps(p.get("domains") or [])
                entities_json = json.dumps(p.get("entities") or [])
            except Exception:
                log.debug("fingerprint promoter parse failed for %s", gid, exc_info=True)
        fp_payload.append((
            gid, shard_name, src_id, domains_json, entities_json, kv_blob,
            0 if is_frag else 1,
            None,
            now,
        ))

    main_conn.executemany(
        "INSERT OR REPLACE INTO source_index "
        "(gene_id, shard_name, source_id, repo_root, source_kind, observed_at, "
        "mtime, content_hash, volatility_class, authority_class, support_span, "
        "last_verified_at, invalidated_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        si_payload,
    )
    main_conn.executemany(
        "INSERT OR REPLACE INTO fingerprint_index "
        "(gene_id, shard_name, source_id, domains, entities, key_values, "
        "is_parent, sequence_idx, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        fp_payload,
    )
    main_conn.commit()
    return len(rows)


def _run_monolithic(args, tagger, chunker) -> None:
    genome = Genome(path=args.db, synonym_map={}, splade_enabled=True, entity_graph=True)
    stats = {"files": 0, "genes": 0, "skipped": 0, "errors": 0, "t0": time.perf_counter()}
    sources = (
        [_parse_source_arg(s) for s in args.sources]
        if args.sources else _DEFAULT_SOURCES
    )
    for root, label, _category in sources:
        if not os.path.isdir(root):
            log.info("Skipping %s (not found)", root)
            continue
        if label == "models" and not args.skip_models:
            log.info("=== Ingesting models from %s ===", root)
            ingest_models(root, genome, tagger, stats)
        else:
            log.info("=== Ingesting %s (%s) ===", root, label)
            ingest_tree(root, genome, tagger, chunker, stats)
    elapsed = time.perf_counter() - stats["t0"]
    log.info("=" * 60)
    log.info("Monolithic ingest complete")
    log.info("  Files: %d ingested, %d skipped, %d errors",
             stats["files"], stats["skipped"], stats["errors"])
    log.info("  Genes: %d in %.0fs (%.1f genes/s)",
             stats["genes"], elapsed, stats["genes"] / max(elapsed, 1))
    log.info("  Genome: %d total genes", genome.stats()["total_genes"])


def _run_sharded(args, tagger, chunker) -> None:
    genomes_root = Path(args.genomes_root)
    genomes_root.mkdir(parents=True, exist_ok=True)
    main_path = _main_db_path(genomes_root)
    main_conn = open_main_db(str(main_path))
    init_main_db(main_conn)
    log.info("Sharded mode: main registry at %s", main_path)

    totals = {"files": 0, "genes": 0, "skipped": 0, "errors": 0, "t0": time.perf_counter()}
    sources = (
        [_parse_source_arg(s) for s in args.sources]
        if args.sources else _DEFAULT_SOURCES
    )

    def ingest_into_shard(shard_db: Path, root: str, label: str,
                          category: str, is_models: bool) -> None:
        shard_db.parent.mkdir(parents=True, exist_ok=True)
        log.info("=== Ingesting %s (%s) -> %s [%s] ===",
                 root, label, shard_db, category)
        shard = Genome(path=str(shard_db), synonym_map={},
                       splade_enabled=True, entity_graph=True)
        s_stats = {"files": 0, "genes": 0, "skipped": 0, "errors": 0,
                   "t0": time.perf_counter()}
        try:
            if is_models:
                ingest_models(root, shard, tagger, s_stats)
            else:
                ingest_tree(root, shard, tagger, chunker, s_stats)
            byte_size = os.path.getsize(shard_db) if shard_db.is_file() else 0
            gene_count = shard.stats()["total_genes"]
            register_shard(
                main_conn,
                shard_name=label,
                category=category,
                path=str(shard_db),
                gene_count=gene_count,
                byte_size=byte_size,
            )
            copied = _copy_indexes_from_shard(main_conn, shard, label)
            log.info("  %s: %d genes, copied %d index rows in %.0fs",
                     label, gene_count, copied,
                     time.perf_counter() - s_stats["t0"])
        finally:
            shard.close()
        for k in ("files", "genes", "skipped", "errors"):
            totals[k] += s_stats[k]

    for root, label, category in sources:
        if not os.path.isdir(root):
            log.info("Skipping %s (not found)", root)
            continue
        is_models = (label == "models") and not args.skip_models
        shard_db = corpus_shard_db(root, label, genomes_root)
        ingest_into_shard(shard_db, root, label, category, is_models)

    for spec in (args.agent_source or []):
        handle, mem_root = _parse_agent_source(spec)
        if not os.path.isdir(mem_root):
            log.warning("Skipping agent %s (path not found: %s)", handle, mem_root)
            continue
        shard_db = agent_shard_db(handle, genomes_root)
        ingest_into_shard(shard_db, mem_root, handle, "agent", False)

    elapsed = time.perf_counter() - totals["t0"]
    shard_rows = main_conn.execute(
        "SELECT shard_name, category, path, gene_count, byte_size "
        "FROM shards WHERE health='ok' ORDER BY category, shard_name"
    ).fetchall()
    main_conn.close()
    log.info("=" * 60)
    log.info("Sharded ingest complete")
    log.info("  Files: %d ingested, %d skipped, %d errors",
             totals["files"], totals["skipped"], totals["errors"])
    log.info("  Genes: %d in %.0fs (%.1f genes/s)",
             totals["genes"], elapsed, totals["genes"] / max(elapsed, 1))
    log.info("  Registered %d shards in %s:", len(shard_rows), main_path)
    for r in shard_rows:
        log.info("    [%-11s] %-18s -> %s  (%d genes, %.1f MB)",
                 r["category"], r["shard_name"], r["path"],
                 r["gene_count"], r["byte_size"] / 1_048_576)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="genome.db",
                        help="Monolithic mode: target DB path")
    parser.add_argument("--sharded", action="store_true",
                        help="Enable sharded mode (one .genome.db per source)")
    parser.add_argument("--genomes-root", default="genomes",
                        help="Sharded mode: base directory for the layout")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument(
        "--sources",
        nargs="+",
        metavar="PATH=LABEL[:CATEGORY]",
        help=(
            "Override default sources. Each arg is a `path=label[:category]` "
            "triple; label 'models' triggers the GGUF-manifest reader; "
            "category defaults to 'reference' (sharded mode only)."
        ),
    )
    parser.add_argument(
        "--agent-source",
        nargs="+",
        metavar="HANDLE=PATH",
        help="Per-handle agent memory sources (sharded mode only).",
    )
    args = parser.parse_args()

    tagger = CpuTagger()
    chunker = CodonChunker()

    if args.sharded:
        _run_sharded(args, tagger, chunker)
    else:
        if args.agent_source:
            log.warning("--agent-source is ignored in monolithic mode")
        _run_monolithic(args, tagger, chunker)


if __name__ == "__main__":
    main()
