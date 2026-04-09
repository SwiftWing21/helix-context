"""
Ingest local model blobs — structural metadata without reading weights.

For each model in the Ollama blob store:
  1. Reads the GGUF header (magic, version, tensor count, metadata KV pairs)
  2. Reads the Ollama manifest (layers, sizes, params)
  3. Creates a "shadow gene" with full architecture metadata

The genome learns what models are available, their sizes, architectures,
quantization, and default parameters — enabling model-aware routing.

Usage:
    python scripts/ingest_models.py                           # default F:\OpenModels
    python scripts/ingest_models.py --root "D:\OpenModels"
"""

from __future__ import annotations

import json
import logging
import os
import struct
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helix_context.config import load_config
from helix_context.genome import Genome
from helix_context.tagger import CpuTagger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest.models")


# ── GGUF header parser ────────────────────────────────────────────

def read_gguf_header(fpath: str, max_kv: int = 60) -> dict | None:
    """Read GGUF metadata from file header without touching weights."""
    try:
        with open(fpath, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None

            version = struct.unpack("<I", f.read(4))[0]
            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            def read_string(f):
                length = struct.unpack("<Q", f.read(8))[0]
                return f.read(length).decode("utf-8", errors="replace")

            def read_value(f, vtype, depth=0):
                if depth > 2:
                    return None
                if vtype == 0: return struct.unpack("<B", f.read(1))[0]
                elif vtype == 1: return struct.unpack("<b", f.read(1))[0]
                elif vtype == 2: return struct.unpack("<H", f.read(2))[0]
                elif vtype == 3: return struct.unpack("<h", f.read(2))[0]
                elif vtype == 4: return struct.unpack("<I", f.read(4))[0]
                elif vtype == 5: return struct.unpack("<i", f.read(4))[0]
                elif vtype == 6: return round(struct.unpack("<f", f.read(4))[0], 6)
                elif vtype == 7: return struct.unpack("?", f.read(1))[0]
                elif vtype == 8: return read_string(f)
                elif vtype == 9:  # array
                    arr_type = struct.unpack("<I", f.read(4))[0]
                    arr_len = struct.unpack("<Q", f.read(8))[0]
                    # Cap array reads to prevent runaway
                    cap = min(arr_len, 20)
                    vals = [read_value(f, arr_type, depth + 1) for _ in range(cap)]
                    if arr_len > cap:
                        # Skip remaining
                        for _ in range(arr_len - cap):
                            read_value(f, arr_type, depth + 1)
                    return vals
                elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
                elif vtype == 11: return struct.unpack("<q", f.read(8))[0]
                elif vtype == 12: return round(struct.unpack("<d", f.read(8))[0], 6)
                return None

            meta = {}
            for _ in range(min(kv_count, max_kv)):
                try:
                    key = read_string(f)
                    vtype = struct.unpack("<I", f.read(4))[0]
                    value = read_value(f, vtype)
                    meta[key] = value
                except Exception:
                    break

            return {
                "format": "GGUF",
                "version": version,
                "tensor_count": tensor_count,
                "metadata_kv_count": kv_count,
                "metadata": meta,
            }
    except Exception as exc:
        log.warning("GGUF parse failed for %s: %s", fpath, exc)
        return None


def read_safetensors_header(fpath: str) -> dict | None:
    """Read safetensors JSON header without loading weights."""
    try:
        with open(fpath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            if header_size > 50_000_000:  # 50MB header = something wrong
                return None
            header_json = json.loads(f.read(header_size))

        tensors = [k for k in header_json if k != "__metadata__"]
        metadata = header_json.get("__metadata__", {})

        # Summarize tensor shapes
        shapes = {}
        for name in tensors[:50]:
            info = header_json[name]
            shape = info.get("shape", [])
            dtype = info.get("dtype", "unknown")
            shapes[name] = {"shape": shape, "dtype": dtype}

        return {
            "format": "safetensors",
            "tensor_count": len(tensors),
            "metadata": metadata,
            "sample_tensors": shapes,
        }
    except Exception as exc:
        log.warning("Safetensors parse failed for %s: %s", fpath, exc)
        return None


# ── Ollama manifest reader ────────────────────────────────────────

def read_ollama_manifests(root: str) -> list[dict]:
    """Read all Ollama model manifests and resolve blob references."""
    manifest_root = os.path.join(root, "ollama", "manifests")
    blob_dir = os.path.join(root, "ollama", "blobs")
    models = []

    if not os.path.isdir(manifest_root):
        return models

    for dirpath, _, filenames in os.walk(manifest_root):
        for fname in filenames:
            mpath = os.path.join(dirpath, fname)
            try:
                with open(mpath) as f:
                    manifest = json.load(f)
            except Exception:
                continue

            # Parse model identity from path
            # .../registry.ollama.ai/library/gemma4/e4b
            parts = dirpath.replace("\\", "/").split("/")
            try:
                lib_idx = parts.index("library")
                family = parts[lib_idx + 1] if lib_idx + 1 < len(parts) else "unknown"
            except ValueError:
                family = "unknown"
            variant = fname
            model_name = f"{family}:{variant}"

            # Extract layer info
            model_blob_digest = None
            model_blob_size = 0
            license_text = ""
            params = {}

            for layer in manifest.get("layers", []):
                media = layer.get("mediaType", "")
                digest = layer.get("digest", "")
                size = layer.get("size", 0)

                if "model" in media:
                    model_blob_digest = digest
                    model_blob_size = size
                elif "params" in media:
                    blob_path = os.path.join(blob_dir, digest.replace(":", "-"))
                    if os.path.exists(blob_path):
                        try:
                            with open(blob_path) as bf:
                                params = json.load(bf)
                        except Exception:
                            pass
                elif "license" in media:
                    blob_path = os.path.join(blob_dir, digest.replace(":", "-"))
                    if os.path.exists(blob_path):
                        try:
                            with open(blob_path, "r", errors="replace") as bf:
                                license_text = bf.read(500)  # First 500 chars
                        except Exception:
                            pass

            # Read GGUF header from model blob
            gguf_meta = None
            if model_blob_digest:
                blob_path = os.path.join(blob_dir, model_blob_digest.replace(":", "-"))
                if os.path.exists(blob_path):
                    gguf_meta = read_gguf_header(blob_path)

            models.append({
                "name": model_name,
                "family": family,
                "variant": variant,
                "size_gb": round(model_blob_size / 1e9, 2),
                "size_bytes": model_blob_size,
                "blob_digest": model_blob_digest,
                "params": params,
                "license_preview": license_text[:200],
                "gguf": gguf_meta,
                "manifest_path": mpath,
            })

    return models


def model_to_gene_content(model: dict) -> str:
    """Convert model metadata into a readable gene content string."""
    lines = [
        f"Model: {model['name']}",
        f"Family: {model['family']}",
        f"Variant: {model['variant']}",
        f"Size: {model['size_gb']} GB ({model['size_bytes']:,} bytes)",
        f"Blob: {model['blob_digest']}",
        "",
    ]

    if model["params"]:
        lines.append("Default Parameters:")
        for k, v in model["params"].items():
            lines.append(f"  {k} = {v}")
        lines.append("")

    gguf = model.get("gguf")
    if gguf:
        lines.append(f"Format: {gguf['format']} v{gguf['version']}")
        lines.append(f"Tensors: {gguf['tensor_count']}")
        lines.append(f"Metadata KV pairs: {gguf['metadata_kv_count']}")
        lines.append("")

        meta = gguf.get("metadata", {})
        # Architecture summary
        arch_keys = sorted(k for k in meta if "attention" in k or "block" in k
                           or "embedding" in k or "vocab" in k or "context" in k
                           or "layer" in k or "head" in k or "general" in k
                           or "quantize" in k or "file_type" in k)
        if arch_keys:
            lines.append("Architecture:")
            for k in arch_keys:
                v = meta[k]
                lines.append(f"  {k} = {v}")
            lines.append("")

        # Remaining metadata
        other_keys = sorted(k for k in meta if k not in arch_keys)
        if other_keys:
            lines.append("Other Metadata:")
            for k in other_keys[:20]:
                v = meta[k]
                if isinstance(v, str) and len(v) > 200:
                    v = v[:200] + "..."
                elif isinstance(v, list) and len(v) > 10:
                    v = v[:10]
                lines.append(f"  {k} = {v}")

    if model["license_preview"]:
        lines.append("")
        lines.append(f"License: {model['license_preview'][:200]}")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="F:/OpenModels")
    args = parser.parse_args()

    config = load_config()
    genome = Genome(
        path=config.genome.path,
        synonym_map=config.synonym_map,
        splade_enabled=config.ingestion.splade_enabled,
        entity_graph=config.ingestion.entity_graph,
    )
    tagger = CpuTagger(synonym_map=config.synonym_map)

    log.info("Scanning %s for model blobs...", args.root)
    t_start = time.perf_counter()

    models = read_ollama_manifests(args.root)
    log.info("Found %d Ollama models", len(models))

    genes_created = 0
    for model in models:
        content = model_to_gene_content(model)
        gene = tagger.pack(
            content,
            content_type="text",
            source_id=model["manifest_path"],
        )
        genome.upsert_gene(gene)
        genes_created += 1
        log.info(
            "  %s — %.1f GB, %d tensors, %d KV",
            model["name"],
            model["size_gb"],
            model["gguf"]["tensor_count"] if model.get("gguf") else 0,
            model["gguf"]["metadata_kv_count"] if model.get("gguf") else 0,
        )

    elapsed = time.perf_counter() - t_start
    stats = genome.stats()
    log.info("=" * 60)
    log.info("Model ingest complete")
    log.info("  Models: %d", len(models))
    log.info("  Genes created: %d in %.1fs", genes_created, elapsed)
    log.info("  Genome total: %d genes", stats["total_genes"])


if __name__ == "__main__":
    main()
