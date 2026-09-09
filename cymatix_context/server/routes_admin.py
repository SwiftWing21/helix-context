"""Admin, health, stats, debug, bridge, and replication routes.

Extracted from the monolithic server.py -- NO logic changes.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import helpers as _helpers
from .helpers import (
    _merge_tier_contributions,
    _paused_ribosomes,
    build_admin_auth,
)
from ..backends.sema_codec import decode_embedding

log = logging.getLogger("cymatix.server")


def setup_admin_routes(app: FastAPI, cymatix, config, registry, bridge, **_kw) -> None:
    """Register admin, health, stats, debug, bridge, and replication routes."""
    from ..accel import json_dumps, json_loads

    # Opt-in bearer guard (2026-08-08 audit): attached to every /admin/*
    # route below. Inert while [server] admin_token = "" (the default) —
    # the dependency returns before reading any header.
    _admin_auth = [Depends(build_admin_auth(config))]

    # -- Stats endpoint ------------------------------------------------

    @app.get("/stats")
    async def stats_endpoint():
        # Refresh OTel gauges
        try:
            from ..telemetry import emit_gauges_snapshot
            emit_gauges_snapshot(cymatix.genome)
        except Exception:
            log.debug("telemetry gauges snapshot failed", exc_info=True)
        return cymatix.stats()

    # -- Resonance introspection endpoint -------------------------------

    @app.get("/debug/resonance")
    async def resonance_endpoint(query: str, k: int = 10, downsample: int = 64):
        import json as _json

        try:
            genome = cymatix.genome
            codec = getattr(cymatix, "_sema_codec", None)

            from ..scoring.cymatics import query_spectrum, cached_doc_spectrum, resonance_score
            q_spec = query_spectrum(query)
            q_sema = codec.encode(query) if codec is not None else None

            neighbors: list = []
            if q_sema is not None:
                rows = genome.read_conn.execute(
                    "SELECT gene_id, embedding FROM genes "
                    "WHERE embedding IS NOT NULL AND chromatin < 2 "
                    "LIMIT 20000"
                ).fetchall()
                scored = []
                for r in rows:
                    try:
                        vec = decode_embedding(r["embedding"])
                    except Exception:
                        continue
                    sim = codec.similarity(q_sema, vec)
                    scored.append((sim, r["gene_id"]))
                scored.sort(key=lambda x: x[0], reverse=True)
                top = scored[:k]

                for sim, gid in top:
                    g = genome.get_doc(gid)
                    if g is None:
                        continue
                    try:
                        g_spec = cached_doc_spectrum(g)
                        cym_sim = resonance_score(q_spec, g_spec)
                    except Exception:
                        cym_sim = 0.0
                    path = None
                    if g.promoter and g.promoter.metadata:
                        path = g.promoter.metadata.get("path")
                    chrom = 0
                    if g.epigenetics:
                        chrom = getattr(g.epigenetics, "chromatin", 0)
                    neighbors.append({
                        "gene_id": gid,
                        "sema_cos_sim": round(float(sim), 4),
                        "cymatic_cos_sim": round(float(cym_sim), 4),
                        "path": path,
                        "preview": (g.content or "")[:120],
                        "chromatin": chrom,
                    })

            edges: list = []
            if neighbors:
                ids = [n["gene_id"] for n in neighbors]
                placeholders = ",".join("?" * len(ids))
                edge_rows = genome.read_conn.execute(
                    f"SELECT gene_id_a, gene_id_b, weight, source FROM harmonic_links "
                    f"WHERE gene_id_a IN ({placeholders}) AND gene_id_b IN ({placeholders})",
                    (*ids, *ids),
                ).fetchall()
                for r in edge_rows:
                    edges.append({
                        "from": r[0], "to": r[1],
                        "weight": round(float(r[2]), 4),
                        "source": r[3],
                    })

            def _downsample(spec, n):
                if len(spec) <= n:
                    return [round(float(x), 4) for x in spec]
                step = len(spec) / n
                out = []
                for i in range(n):
                    lo = int(i * step)
                    hi = int((i + 1) * step)
                    chunk = spec[lo:hi] or [spec[lo]]
                    out.append(round(sum(chunk) / len(chunk), 4))
                return out

            return {
                "query": query,
                "query_sema": [round(float(x), 4) for x in q_sema] if q_sema is not None else None,
                "query_spectrum": _downsample(q_spec, downsample),
                "spectrum_bins": downsample,
                "spectrum_bins_raw": len(q_spec),
                "neighbors": neighbors,
                "edges": edges,
                "edge_count": len(edges),
                "k": k,
                "sema_available": codec is not None,
            }
        except Exception:
            log.error(
                "/debug/resonance failed for query=%r", query, exc_info=True
            )
            raise HTTPException(status_code=500, detail="Internal error")

    # -- Debug: single-document fetch ---

    @app.get("/genes/{gene_id}")
    async def gene_get_endpoint(gene_id: str):
        """Fetch a single document by ID."""
        try:
            gene = cymatix.genome.get_doc(gene_id)
        except Exception as exc:
            log.warning("/genes/%s failed: %s", gene_id, exc, exc_info=True)
            return JSONResponse(
                {"error": f"Gene lookup failed: {exc}"}, status_code=500,
            )
        if gene is None:
            return JSONResponse(
                {"error": f"Unknown gene_id: {gene_id}"}, status_code=404,
            )
        return gene.model_dump()

    # Rosetta Tier 2 (docs/ROSETTA.md): software-lexicon alias of the
    # route above. Registered via add_api_route (rather than stacking a
    # second @app.get on gene_get_endpoint) so it gets its own operation
    # id and doesn't collide with /genes/{gene_id}'s in the OpenAPI
    # schema. Same handler, same behavior -- additive only.
    app.add_api_route(
        "/documents/{gene_id}",
        gene_get_endpoint,
        methods=["GET"],
        name="document_get_endpoint",
    )

    # -- Debug: lightweight SEMA neighbors ---

    @app.get("/debug/neighbors")
    async def neighbors_endpoint(query: str, k: int = 10):
        """Top-k SEMA neighbors for ``query``."""
        import json as _json

        try:
            rows = cymatix.genome.read_conn.execute(
                "SELECT gene_id, embedding FROM genes "
                "WHERE embedding IS NOT NULL AND chromatin < 2 "
                "LIMIT 20000"
            ).fetchall()
            if not rows:
                return {
                    "query": query,
                    "k": k,
                    "count": 0,
                    "neighbors": [],
                }
            codec = getattr(cymatix, "_sema_codec", None)
            if codec is None:
                return JSONResponse(
                    {
                        "error": "SEMA codec not available",
                        "hint": "Ingest must have populated embeddings first.",
                    },
                    status_code=503,
                )
            q_sema = codec.encode(query)
            scored: list = []
            for r in rows:
                try:
                    vec = decode_embedding(r["embedding"])
                except Exception:
                    continue
                sim = codec.similarity(q_sema, vec)
                scored.append((sim, r["gene_id"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:k]

            neighbors: list = []
            for sim, gid in top:
                g = cymatix.genome.get_doc(gid)
                if g is None:
                    continue
                path = None
                if g.promoter and g.promoter.metadata:
                    path = g.promoter.metadata.get("path")
                neighbors.append({
                    "gene_id": gid,
                    "sema_cos_sim": round(float(sim), 4),
                    "preview": (g.content or "")[:160],
                    "path": path,
                })
            return {
                "query": query,
                "k": k,
                "neighbors": neighbors,
                "count": len(neighbors),
            }
        except Exception:
            log.error("/debug/neighbors failed", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

    # -- Debug: context-pipeline dry run (no splice) ---

    @app.get("/debug/preview")
    async def preview_endpoint(
        query: str,
        max_genes: int = 12,
        profile: str = "balanced",
        score_floor: float = 0.0,
    ):
        """Dry-run the retrieval pipeline up to candidate selection."""
        profile = str(profile or "balanced").strip().lower()
        if profile not in {"fast", "balanced", "quality"}:
            return JSONResponse(
                {"error": "Invalid profile", "allowed": ["fast", "balanced", "quality"]},
                status_code=400,
            )

        if score_floor < 0:
            return JSONResponse(
                {"error": "score_floor must be >= 0"},
                status_code=400,
            )

        if score_floor > 0:
            eval_budget = min(max(max_genes * 3, 50), 200)
        else:
            eval_budget = max_genes

        expand_query = profile in {"balanced", "quality"}
        use_harmonic = profile == "quality"
        use_sr = profile == "quality"
        use_cymatics = profile == "quality"
        use_harmonic_bin = profile == "quality"
        use_tcm = True

        try:
            expanded_query, domains, entities = cymatix._prepare_query_signals(
                query,
                session_context=None,
                expand_query=expand_query,
            )
            candidates = cymatix._retrieve(
                domains=domains,
                entities=entities,
                max_genes=eval_budget,
                query_text=query,
                use_harmonic=use_harmonic,
                use_sr=use_sr,
            )
            # Issue #350 (W1.6b): snapshot scores + tier contributions
            # atomically under the writer lock IMMEDIATELY after THIS
            # request's retrieval (request-scoped capture, mirroring
            # /fingerprint) — the old post-refiner unlocked read could
            # hand this response a concurrent request's maps.
            _lock = getattr(cymatix.genome, "_last_query_scores_lock", None)
            if _lock is not None:
                with _lock:
                    scores = dict(cymatix.genome.last_query_scores or {})
                    _tier_contribs = dict(getattr(cymatix.genome, "last_tier_contributions", {}) or {})
            else:
                scores = dict(cymatix.genome.last_query_scores or {})
                _tier_contribs = dict(getattr(cymatix.genome, "last_tier_contributions", {}) or {})
            # query_scores= threads the snapshot through the refiners so
            # the cymatics/harmonic mutations land in THIS request's map
            # (byte-identical single-request; see /fingerprint).
            candidates, refiner_contrib = cymatix._apply_candidate_refiners(
                query,
                candidates,
                eval_budget,
                use_cymatics=use_cymatics,
                use_harmonic_bin=use_harmonic_bin,
                use_tcm=use_tcm,
                query_scores=scores,
            )
        except Exception:
            log.error("/debug/preview failed", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error")

        merged_tiers = _merge_tier_contributions(
            _tier_contribs,
            refiner_contrib,
        )

        def _final_score(gene_id: str) -> float:
            tcm_bonus = refiner_contrib.get(gene_id, {}).get("tcm", 0.0)
            return float(scores.get(gene_id, 0.0) + tcm_bonus)

        evaluated_total = len(candidates)
        above_floor = [g for g in candidates if _final_score(g.gene_id) >= score_floor]
        above_floor_total = len(above_floor)
        truncated = above_floor[:max_genes]
        returned = len(truncated)
        filtered_by_floor = evaluated_total - above_floor_total
        truncated_by_cap = above_floor_total - returned

        result = []
        for rank, g in enumerate(truncated):
            path = None
            if g.promoter and g.promoter.metadata:
                path = g.promoter.metadata.get("path")
            result.append({
                "rank": rank,
                "gene_id": g.gene_id,
                "score": round(_final_score(g.gene_id), 4),
                "preview": (g.content or "")[:160],
                "path": path,
                "domains": list(g.promoter.domains) if g.promoter else [],
                "entities": list(g.promoter.entities) if g.promoter else [],
                "chromatin": int(getattr(g, "chromatin", 0) or 0),
                "tier_contributions": {
                    k: round(float(v), 4)
                    for k, v in sorted(merged_tiers.get(g.gene_id, {}).items())
                },
            })

        if returned == 0 and evaluated_total > 0 and score_floor > 0:
            response_hint = (
                f"All {evaluated_total} evaluated candidates fell below "
                f"score_floor={score_floor}; consider lowering it or "
                f"refining the query."
            )
        elif truncated_by_cap > 0:
            response_hint = (
                f"{truncated_by_cap} additional candidates cleared the floor "
                f"but were truncated by max_genes={max_genes}; raise "
                f"max_genes to see more."
            )
        elif filtered_by_floor > 0:
            response_hint = (
                f"{filtered_by_floor} evaluated candidates fell below "
                f"score_floor={score_floor}."
            )
        else:
            response_hint = "No filtering or truncation applied."

        return {
            "query": query,
            "profile": profile,
            "extracted": {
                "expanded_query": expanded_query,
                "domains": list(domains),
                "entities": list(entities),
            },
            "candidates": result,
            "fingerprints": result,
            "count": len(result),
            "max_genes": max_genes,
            "score_floor": score_floor,
            "evaluated_total": evaluated_total,
            "above_floor_total": above_floor_total,
            "returned": returned,
            "filtered_by_floor": filtered_by_floor,
            "truncated_by_cap": truncated_by_cap,
            "response_hint": response_hint,
            "note": "Splice step skipped; these are pre-splice candidates.",
        }

    # -- Health history endpoint ---

    @app.get("/health/history")
    async def health_history_endpoint(limit: int = 50):
        return cymatix.genome.health_history(limit=limit)

    # -- Pipeline viewer feed (launcher dashboard) ---

    @app.get("/debug/pipeline/recent")
    async def pipeline_recent_endpoint(limit: int = 32):
        """Most recent build_context stage events for the launcher's
        pipeline-viewer panel. Each event is
            { request_id, stage, ms, ts }
        plus, on some stages (perf slice 1, W0.2/W0.3):
            express     + { signals: {name: ms}, model_load_ms: {name: ms} }
            tail_writes + { commits: int }   # per-request writer-commit delta
        Ordered oldest to newest. Backed by a bounded in-process ring
        (see cymatix_context.context_manager.get_recent_pipeline_events).
        Bench-side consumer: benchmarks/dogfood/_stage_receipts.py.
        """
        try:
            from ..context_manager import (
                get_recent_pipeline_events,
                get_pipeline_ring_max,
            )
            events = get_recent_pipeline_events(limit=max(1, min(int(limit), 200)))
            return {"events": events, "ring_max": get_pipeline_ring_max()}
        except Exception as exc:
            log.warning("/debug/pipeline/recent failed: %s", exc, exc_info=True)
            return JSONResponse({"error": str(exc)}, status_code=500)

    # -- Active genome path (launcher dashboard) ---

    @app.get("/admin/genome", dependencies=_admin_auth)
    async def admin_genome_endpoint():
        """Describe the genome file the running process actually opened.

        Used by the launcher's Database panel to confirm that the
        supervised cymatix is serving from the file the user expects (and
        to detect divergence when cymatix was started outside the launcher
        with a different CYMATIX_GENOME_PATH override).
        """
        from pathlib import Path as _Path

        path_str = str(getattr(config.genome, "path", "")) or "genome.db"
        resolved = _Path(path_str)
        try:
            resolved = resolved.resolve()
        except OSError:
            pass

        out = {
            "path": str(resolved),
            "configured_path": path_str,
            "env_override": os.environ.get("CYMATIX_GENOME_PATH"),
        }
        try:
            if resolved.exists() and resolved.is_file():
                st = resolved.stat()
                out["size_bytes"] = st.st_size
                out["mtime"] = st.st_mtime
            else:
                out["error"] = "genome file not found on disk"
        except OSError as exc:
            out["error"] = f"stat failed: {exc}"

        try:
            out["total_genes"] = int(cymatix.genome.stats().get("total_genes", 0))
        except Exception:
            log.debug("/admin/genome stats failed", exc_info=True)
        return out

    # -- Token metrics endpoint ---

    @app.get("/metrics/tokens")
    async def metrics_tokens_endpoint():
        """Session + lifetime token counters."""
        try:
            return cymatix.token_counter.snapshot()
        except Exception as exc:
            log.warning("Token metrics snapshot failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Token snapshot failed: {exc}"},
                status_code=500,
            )

    # -- Health endpoint ---

    @app.get("/health")
    async def health_endpoint():
        ribosome_disabled = getattr(cymatix.ribosome.backend, "is_disabled_backend", False)
        ribosome_model = "disabled" if ribosome_disabled else "unknown"
        if not ribosome_disabled and hasattr(cymatix.ribosome, "backend") and hasattr(cymatix.ribosome.backend, "model"):
            ribosome_model = cymatix.ribosome.backend.model
        elif not ribosome_disabled and hasattr(cymatix.ribosome, "ollama_ribosome"):
            ribosome_model = f"deberta+{cymatix.ribosome.ollama_ribosome.backend.model}"

        genome_ready = True
        total_genes = 0
        try:
            total_genes = cymatix.genome.stats()["total_genes"]
        except Exception:
            genome_ready = False
            log.warning("/health genome stats failed", exc_info=True)

        # Three-state upstream semantics (2026-07-30): the upstream model
        # server is a dependency only while a ribosome backend is live.
        #   active + reachable   -> upstream_state "ok"
        #   active + unreachable -> "unreachable" — the ONLY case that
        #                           degrades on this axis
        #   ribosome disabled    -> "inactive" — not a dependency; the
        #                           probe is skipped so a down Ollama does
        #                           not add probe latency to every health
        #                           poll on an LLM-free profile
        # Keyed on the RUNTIME backend (same signal as the `ribosome`
        # field above), not config, so programmatically wired backends
        # report honestly.
        if ribosome_disabled:
            upstream_reachable = None
            upstream_state = "inactive"
        else:
            # Late-bind via the package module so monkeypatch in tests
            # (``monkeypatch.setattr(server_mod, "_probe_upstream", ...)``)
            # takes effect.
            import cymatix_context.server as _srv
            _probe_fn = getattr(_srv, "_probe_upstream", _helpers._probe_upstream)
            upstream_probe = await asyncio.to_thread(
                _probe_fn, config.server.upstream
            )
            upstream_reachable = bool(upstream_probe.get("reachable"))
            upstream_state = "ok" if upstream_reachable else "unreachable"
        status = (
            "ok" if genome_ready and upstream_state != "unreachable" else "degraded"
        )

        if not genome_ready and upstream_state == "unreachable":
            message = "Genome stats failed and the upstream model server is unreachable."
        elif not genome_ready:
            message = "Genome stats failed; inspect the local knowledge store."
        elif upstream_state == "unreachable":
            message = "Upstream model server is unreachable; final chat proxy calls will fail."
        elif upstream_state == "inactive":
            message = (
                "Cymatix answered readiness checks; ribosome layer inactive — "
                "upstream model server not probed (only chat-proxy calls would need it)."
            )
        else:
            message = "Cymatix and its upstream model server answered readiness checks."

        from ..hardware import get_hardware
        hw_info = get_hardware()
        low_vram = bool(
            hw_info.vram_total_gb is not None
            and hw_info.vram_total_gb < config.hardware.low_vram_threshold_gb
        )
        hardware_block = {
            "device": hw_info.device,
            "device_name": hw_info.device_name,
            "requested_device": hw_info.requested_device,
            "fallback_active": hw_info.fallback_reason is not None,
            "fallback_reason": hw_info.fallback_reason,
            "vram_total_gb": hw_info.vram_total_gb,
            "system_ram_gb": hw_info.system_ram_gb,
            "low_vram_warning": low_vram,
        }

        calibration_block = {
            "ann_threshold_mode": config.retrieval.ann_threshold_mode,
            "abstain_mode": config.abstain.mode,
            "abstain_classes": sorted(config.abstain.per_class.keys()),
        }
        try:
            ann_meta = cymatix.genome.get_calibration_provenance()
        except Exception:
            ann_meta = None
            log.debug("/health calibration provenance read failed", exc_info=True)
        if ann_meta is not None:
            calibration_block["ann_threshold"] = ann_meta

        # Issue #239: surface KnowBlock confidence-logistic staleness so
        # agents can tell when the know/miss contract was last trued-up
        # against a clean bed (scripts/calibrate_know_confidence.py writes
        # calibrated_at/calibrated_on_n; see cymatix_context.scoring.know_calibration).
        from ..scoring.know_calibration import calibration_age_days, is_calibration_stale
        know_cfg = config.know
        know_calibrated_at = know_cfg.calibrated_at
        know_age_days = calibration_age_days(know_calibrated_at)
        # Uncalibrated (never run) counts as stale outright — not just
        # "age unknown" — so a fresh install doesn't silently read as
        # trustworthy in this block.
        know_stale = know_calibrated_at is None or is_calibration_stale(
            know_age_days, know_cfg.stale_after_days
        )
        know_calibration_block = {
            "calibrated_at": know_calibrated_at,
            "calibrated_on_n": know_cfg.calibrated_on_n,
            "stale": know_stale,
        }

        configured_backend = (
            config.ribosome.normalized_backend if config.ribosome.enabled else None
        )
        from .. import __version__ as _pkg_version
        return {
            "status": status,
            "message": message,
            "version": _pkg_version,
            # OS pid of the process answering this request. Lets callers
            # (e.g. the bench orchestrator's _wait_healthy) confirm they
            # are talking to the process they just spawned, not a stale
            # server that lost a port-bind race. See issue #127.
            "pid": os.getpid(),
            # Spawn-token identity (perf slice 2): on Windows a venv's
            # python.exe launcher re-execs the interpreter as a child, so
            # the pid above never matches the spawner's Popen.pid. The
            # orchestrator passes a unique token via env; echoing it is
            # the reliable "you are talking to the process you spawned"
            # signal. None when not bench-spawned.
            "bench_spawn_token": os.environ.get("CYMATIX_BENCH_SPAWN_TOKEN"),
            "ribosome": ribosome_model,
            "ribosome_backend": config.ribosome.effective_backend,
            "ribosome_configured_backend": configured_backend,
            "ribosome_cost_class": config.ribosome.cost_class,
            "genes": total_genes,
            "upstream": config.server.upstream,
            "upstream_reachable": upstream_reachable,
            "upstream_state": upstream_state,
            "hardware": hardware_block,
            "calibration": calibration_block,
            "know_calibration": know_calibration_block,
            "checks": {
                "genome_ready": genome_ready,
                "upstream_ready": upstream_reachable,
            },
        }

    @app.get("/replicas")
    async def replicas_endpoint():
        if cymatix._replication_mgr is None:
            return {"enabled": False, "replicas": []}
        return {"enabled": True, **cymatix._replication_mgr.status()}

    @app.post("/replicas/sync")
    async def replicas_sync_endpoint():
        if cymatix._replication_mgr is None:
            return {"synced": 0, "error": "replication not configured"}
        synced = cymatix._replication_mgr.sync_now()
        return {"synced": synced}

    # ---- Admin: knowledge store management ----

    @app.post("/admin/refresh", dependencies=_admin_auth)
    async def admin_refresh():
        """Reopen knowledge store connection to see external changes."""
        cymatix.genome.refresh()
        cymatix.genome._invalidate_dense_matrix(force=True)
        getattr(cymatix, "_mtime_cache", {}).clear()
        new_count = cymatix.genome.total_genes()
        return {"refreshed": True, "genes": new_count}

    @app.post("/admin/genes/tombstone", dependencies=_admin_auth)
    async def admin_genes_tombstone(request: Request):
        """Demote every gene matching ``source_id`` to HETEROCHROMATIN.

        Non-destructive lifecycle tombstone for callers that track content
        by provenance — mem_sync calls this when a watched memory file is
        deleted so the documents drop out of hot-tier retrieval. Content is
        preserved (``compress_to_heterochromatin`` only flips the tier
        flags), so the documents stay reachable via cold-tier retrieval.
        """
        data = await request.json()
        source_id = data.get("source_id")
        if not source_id or not isinstance(source_id, str):
            return JSONResponse(
                {"error": "source_id required"},
                status_code=400,
            )
        cur = cymatix.genome.conn.cursor()
        rows = cur.execute(
            "SELECT gene_id FROM genes WHERE source_id = ? AND chromatin < 2",
            (source_id,),
        ).fetchall()
        tombstoned = []
        for row in rows:
            try:
                if cymatix.genome.compress_to_heterochromatin(row["gene_id"]):
                    tombstoned.append(row["gene_id"])
            except Exception:
                log.warning("Tombstone failed for gene %s",
                            row["gene_id"], exc_info=True)
        return {
            "source_id": source_id,
            "tombstoned": len(tombstoned),
            "gene_ids": tombstoned,
        }

    @app.post("/admin/vacuum", dependencies=_admin_auth)
    async def admin_vacuum():
        """Reclaim free pages from the knowledge store database."""
        try:
            result = cymatix.genome.vacuum()
            return {"ok": True, **result}
        except Exception as exc:
            log.warning("VACUUM failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"ok": False, "error": str(exc)},
                status_code=500,
            )

    @app.post("/admin/kv-backfill", dependencies=_admin_auth)
    async def admin_kv_backfill():
        """Run CPU regex KV extraction on documents missing key_values."""
        import re as _re
        cur = cymatix.genome.conn.cursor()
        rows = cur.execute(
            "SELECT gene_id, content FROM genes "
            "WHERE key_values IS NULL OR key_values = '[]' OR key_values = 'null'"
        ).fetchall()
        if not rows:
            return {"backfilled": 0, "total": cymatix.genome.stats()["total_genes"]}

        patterns = [
            _re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*["\']([^"\'\n]{1,100})["\']', _re.MULTILINE),
            _re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(\d+(?:\.\d+)?)\s*$', _re.MULTILINE),
            _re.compile(r'"([a-z_]\w*)":\s*["\']?([^,}"\'\n]{1,80})["\']?'),
            _re.compile(r'(?:\*\*|[-*])\s*([A-Za-z ]{2,30})(?:\*\*)?:\s*(.{1,80})'),
        ]
        updated = 0
        # W2.3 Phase A interim (cwola pattern): the UPDATE loop + commit
        # is one writer transaction on the shared connection — serialize
        # it via the store's write lock. The per-row regex work is pure
        # CPU and bounded (3000 chars/row).
        import contextlib as _contextlib
        _wl = getattr(cymatix.genome, "_write_lock", None)
        with (_wl if _wl is not None else _contextlib.nullcontext()):
            for row in rows:
                content = row["content"][:3000]
                kvs = set()
                for pat in patterns:
                    for match in pat.finditer(content):
                        g = match.groups()
                        if len(g) == 2 and g[0] and g[1]:
                            kvs.add(f"{g[0].strip()[:40]}={g[1].strip()[:80]}")
                cur.execute(
                    "UPDATE genes SET key_values = ? WHERE gene_id = ?",
                    (json_dumps(sorted(kvs)[:15]), row["gene_id"]),
                )
                updated += 1
            cymatix.genome.conn.commit()
        return {"backfilled": updated, "total": cymatix.genome.stats()["total_genes"]}

    @app.post("/admin/compact", dependencies=_admin_auth)
    async def admin_compact(dry_run: bool = False, density_threshold: float = 0.3, access_threshold: int = 5):
        """Run compaction sweep."""
        result = cymatix.genome.compact_genome(
            density_threshold=density_threshold,
            access_threshold=access_threshold,
            dry_run=dry_run,
        )
        return result

    @app.post("/admin/compact-pki", dependencies=_admin_auth)
    async def admin_compact_pki(
        dry_run: bool = False, noise_cutoff: int = -1, drop: bool = False,
    ):
        """Issue #165 Option-B: compact ``path_key_index``.

        Drops the dead ``idx_pki_lookup``, prunes rows in pairs above the
        Tier-0 noise cutoff (score-invariant — those pairs are hard-skipped
        by the scorer), and rebuilds the table WITHOUT ROWID. Follow with
        ``/admin/vacuum`` to reclaim the freed pages. ``dry_run=true``
        reports what would change without touching the DB.

        Issue #370: ``drop=true`` drops the table entirely — the reclaim
        path for pki-off stores, where the scorer never reads it. Refused
        with 409 while ``[retrieval] pki_enabled = true``. The drop only
        frees pages to the freelist; call ``/admin/vacuum`` after or
        nothing shrinks on disk. Runbook 10 in docs/operator-runbooks.md.
        """
        from ..storage.indexes import (
            PKI_NOISE_CUTOFF, compact_path_key_index,
        )
        cutoff = noise_cutoff if noise_cutoff >= 0 else PKI_NOISE_CUTOFF
        # #370 guard input: the live store's effective Tier-0 read gate
        # (build_genome_kwargs threads [retrieval] pki_enabled into it).
        # Unknown attribute defaults to True — i.e. refuse the drop.
        pki_on = bool(getattr(cymatix.genome, "_pki_enabled", True))
        try:
            # W2.3 Phase A interim (cwola pattern): the compactor rewrites
            # path_key_index + commits on the shared writer — serialize at
            # the call site (storage.indexes stays store-agnostic).
            import contextlib as _contextlib
            _wl = getattr(cymatix.genome, "_write_lock", None)
            with (_wl if _wl is not None else _contextlib.nullcontext()):
                result = compact_path_key_index(
                    cymatix.genome.conn, noise_cutoff=cutoff, dry_run=dry_run,
                    drop=drop, pki_enabled=pki_on,
                )
            return {"ok": True, **result}
        except ValueError as exc:
            # #370 refuse-to-drop guard: the config forbids it — client
            # error (flip the knob first), not a server fault.
            log.warning("compact-pki drop refused: %s", exc)
            return JSONResponse(
                {"ok": False, "error": str(exc)}, status_code=409,
            )
        except Exception as exc:
            log.warning("compact-pki failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"ok": False, "error": str(exc)}, status_code=500,
            )

    @app.post("/admin/checkpoint", dependencies=_admin_auth)
    async def admin_checkpoint(mode: str = "PASSIVE"):
        """Force a WAL checkpoint."""
        cymatix.genome.checkpoint(mode)
        return {"checkpointed": True, "mode": mode}

    @app.post("/admin/ribosome/pause", dependencies=_admin_auth)
    async def admin_ribosome_pause():
        """Disable the compressor's LLM calls without unloading or restarting."""
        backend = cymatix.ribosome.backend
        backend_id = id(backend)
        if backend_id in _paused_ribosomes:
            return {
                "paused": True,
                "already": True,
                "model": getattr(backend, "model", "unknown"),
            }

        _paused_ribosomes[backend_id] = backend.complete

        def _raise_paused(*args, **kwargs):
            raise RuntimeError(
                "Ribosome paused by /admin/ribosome/pause -- "
                "learn() fallback path engaged"
            )

        backend.complete = _raise_paused
        log.info(
            "Ribosome backend paused (model=%s). LLM calls will raise.",
            getattr(backend, "model", "unknown"),
        )
        return {
            "paused": True,
            "model": getattr(backend, "model", "unknown"),
            "hint": (
                "LLM calls will raise. learn() builds minimal genes from "
                "raw exchange. Resume with POST /admin/ribosome/resume."
            ),
        }

    @app.post("/admin/ribosome/resume", dependencies=_admin_auth)
    async def admin_ribosome_resume():
        """Restore the compressor backend after /admin/ribosome/pause."""
        backend = cymatix.ribosome.backend
        backend_id = id(backend)
        if backend_id not in _paused_ribosomes:
            return {"resumed": False, "reason": "not paused"}

        backend.complete = _paused_ribosomes.pop(backend_id)
        log.info(
            "Ribosome backend resumed (model=%s)",
            getattr(backend, "model", "unknown"),
        )
        return {
            "resumed": True,
            "model": getattr(backend, "model", "unknown"),
        }

    @app.get("/admin/ribosome/status", dependencies=_admin_auth)
    async def admin_ribosome_status():
        """Check whether the compressor is currently paused."""
        backend = cymatix.ribosome.backend
        return {
            "paused": id(backend) in _paused_ribosomes,
            "model": getattr(backend, "model", "unknown"),
            "backend_type": type(backend).__name__,
        }

    @app.post("/admin/shutdown", dependencies=_admin_auth)
    async def admin_shutdown(request: Request):
        """Graceful shutdown."""
        import os as _os
        import signal as _signal

        try:
            data = await request.json()
        except Exception:
            data = {}
        actor = data.get("actor") or "unknown"
        reason = data.get("reason") or "manual shutdown"

        try:
            bridge.write_signal("server_state", {
                "state": "stopped",
                "actor": actor,
                "reason": reason,
                "pid": _os.getpid(),
                "expected_downtime_s": 0,
                "phase": "shutting_down",
            })
        except Exception:
            log.warning("Shutdown: failed to stamp signal", exc_info=True)

        log.info("Shutdown requested by %s: %s", actor, reason)

        try:
            _os.kill(_os.getpid(), _signal.SIGINT)
        except Exception:
            log.warning("SIGINT on self failed", exc_info=True)

        return {
            "shutting_down": True,
            "actor": actor,
            "reason": reason,
            "hint": "Poll GET /stats -- connection refused means shutdown complete.",
        }

    @app.post("/admin/announce_restart", dependencies=_admin_auth)
    async def admin_announce_restart(request: Request):
        """Announce an intentional server restart to other sessions."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "Invalid JSON body"},
                status_code=400,
            )

        reason = body.get("reason")
        actor = body.get("actor")
        if not reason or not actor:
            return JSONResponse(
                {"error": "Both 'reason' and 'actor' are required"},
                status_code=400,
            )

        expected_downtime_s = int(body.get("expected_downtime_s", 30))

        try:
            import os as _os
            bridge.announce_restart(
                reason=reason,
                actor=actor,
                expected_downtime_s=expected_downtime_s,
                pid=_os.getpid(),
            )
        except Exception as exc:
            log.warning("announce_restart failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"Announce failed: {exc}"},
                status_code=500,
            )

        log.info(
            "Restart announced by %s: %s (expected_downtime=%ds)",
            actor, reason, expected_downtime_s,
        )
        return {
            "announced": True,
            "actor": actor,
            "reason": reason,
            "expected_downtime_s": expected_downtime_s,
            "hint": "Sleep ~750ms before killing the server to let observers see the signal.",
        }

    @app.get("/admin/components", dependencies=_admin_auth)
    async def admin_components():
        """Return active subsystems with running/idle status + loaded-ness.

        #219 slice 2: encoders may be armed lazily ([hardware]
        lazy_encoders, default true), so a configured component is not
        necessarily resident. This endpoint must NEVER force a lazy
        component to materialize — loaded-ness is read through
        non-forcing probes only (``peek()``/``loaded`` on lazy proxies,
        the spaCy/SPLADE module globals, ``_model`` on the BGE-M3
        codec). Components that are configured but not yet loaded report
        ``status = "idle (not loaded)"`` and ``loaded = false``; loaded
        components keep the running/idle activity semantics.
        """
        import time as _time
        idle_threshold_s = 60.0
        age = _time.time() - getattr(cymatix, "_last_activity_ts", 0.0)
        active_status = "running" if age < idle_threshold_s else "idle"

        def _status(loaded: bool) -> str:
            return active_status if loaded else "idle (not loaded)"

        # ── ribosome (a LazyRibosome proxy when backend=deberta+lazy) ──
        # type()-level probe: instance attribute access on the proxy
        # would materialize the DeBERTa models.
        rib = cymatix.ribosome
        rib_is_lazy = bool(getattr(type(rib), "is_lazy_component", False))
        rib_real = rib.peek() if rib_is_lazy else rib
        rib_loaded = rib_real is not None
        backend_obj = getattr(rib_real, "backend", None) if rib_loaded else None
        ribosome_paused = backend_obj is not None and id(backend_obj) in _paused_ribosomes
        ribosome_disabled = bool(getattr(backend_obj, "is_disabled_backend", False))

        components = []

        if not ribosome_paused and not ribosome_disabled:
            if rib_is_lazy:
                ribosome_backend = rib.label or "unknown"
            elif backend_obj is not None and hasattr(backend_obj, "model"):
                ribosome_backend = backend_obj.model
            else:
                ribosome_backend = "unknown"
            components.append({
                "name": "ribosome",
                "kind": "decoder",
                "status": _status(rib_loaded),
                "backend": ribosome_backend,
                "loaded": rib_loaded,
            })

        sema = getattr(cymatix, "_sema_codec", None)
        if sema is not None:
            # LazySemaCodec exposes ``loaded`` without touching the model;
            # a plain (eager) SemaCodec has no such attr -> resident.
            sema_loaded = bool(getattr(sema, "loaded", True))
            components.append({
                "name": "sema",
                "kind": "encoder",
                "status": _status(sema_loaded),
                "loaded": sema_loaded,
                # #376: "daemon" / "in-process" — the transport the last
                # encode actually used; None for a purely local codec or
                # before the first encode.
                "transport": getattr(sema, "last_transport", None),
            })

        if getattr(cymatix, "_cpu_tagger", None) is not None:
            # spaCy loads lazily inside tagger._get_nlp(); the module
            # global reflects whether the pipeline is resident.
            try:
                from .. import tagger as _tagger_mod
                tagger_loaded = getattr(_tagger_mod, "_nlp", None) is not None
            except Exception:
                tagger_loaded = False
            components.append({
                "name": "cpu_tagger",
                "kind": "encoder",
                "status": _status(tagger_loaded),
                "loaded": tagger_loaded,
            })

        if getattr(cymatix.genome, "_splade_enabled", False):
            # splade_backend keeps its model in a module global, set on
            # first encode (already first-use-lazy pre-#219).
            try:
                from ..backends import splade_backend as _splade_mod
                splade_loaded = getattr(_splade_mod, "_model", None) is not None
            except Exception:
                splade_loaded = False
            components.append({
                "name": "splade",
                "kind": "encoder",
                "status": _status(splade_loaded),
                "loaded": splade_loaded,
            })

        # BGE-M3 dense codec — surfaced so operators can see whether the
        # ~2 GB model is resident. Loaded when either holder (manager
        # ingest-side, store query-side) has a materialized model; the
        # codec wrapper itself is cheap and loads weights on first encode.
        dense_enabled = bool(
            getattr(cymatix.genome, "_dense_embedding_enabled", False)
            or getattr(config.ingestion, "dense_embed_on_ingest", False)
        )
        if dense_enabled:
            dense_holders = (
                getattr(cymatix, "_dense_codec", None),
                getattr(cymatix.genome, "_dense_codec", None),
            )
            dense_loaded = any(
                holder is not None and getattr(holder, "_model", None) is not None
                for holder in dense_holders
            )
            # #376: the transport the last dense encode actually used, not
            # the wrapper class — "daemon" / "in-process"; None for a purely
            # local codec or before the first encode.
            dense_transport = next(
                (
                    t
                    for t in (getattr(h, "last_transport", None) for h in dense_holders)
                    if t is not None
                ),
                None,
            )
            components.append({
                "name": "dense_bgem3",
                "kind": "encoder",
                "status": _status(dense_loaded),
                "loaded": dense_loaded,
                "transport": dense_transport,
            })

        if getattr(cymatix.genome, "_entity_graph_enabled", False):
            components.append({
                "name": "entity_graph",
                "kind": "encoder",
                "status": active_status,
                "loaded": True,
            })

        try:
            from ..encoding.headroom_bridge import is_headroom_available
            if is_headroom_available():
                components.append({
                    "name": "headroom",
                    "kind": "decoder",
                    "status": active_status,
                    "loaded": True,
                })
        except Exception:
            pass

        return {
            "components": components,
            "count": len(components),
            "last_activity_s_ago": round(age, 1),
            "idle_threshold_s": idle_threshold_s,
        }

    @app.post("/admin/sema/rebuild", dependencies=_admin_auth)
    async def admin_sema_rebuild():
        """Force-rebuild the SEMA vector cache."""
        cymatix.genome.invalidate_sema_cache()
        # #339 rider: also drop the cold-tier cache (and its vectorless
        # gate) — it scans the disjoint heterochromatin partition and is
        # lazily rebuilt by query_cold_tier, so without this a forced
        # rebuild left stale cold vectors serving until process restart.
        cymatix.genome.invalidate_cold_sema_cache()
        cymatix.genome._build_sema_cache()
        cache = cymatix.genome._sema_cache
        return {
            "rebuilt": True,
            "vectors": len(cache["gene_ids"]) if cache else 0,
            "memory_kb": (cache["matrix"].nbytes // 1024) if cache else 0,
        }

    @app.post("/admin/reload", dependencies=_admin_auth)
    async def admin_reload():
        """Hot-reload server runtime state without killing the process."""
        changes = {}

        # 1. Reload config from cymatix.toml
        try:
            from ..config import load_config
            new_config = load_config()
            old_budget = cymatix.config.budget.max_genes_per_turn
            new_budget = new_config.budget.max_genes_per_turn
            cymatix.config = new_config
            if old_budget != new_budget:
                changes["max_genes_per_turn"] = {"old": old_budget, "new": new_budget}
            else:
                changes["config"] = "reloaded (no visible changes)"
        except Exception as exc:
            changes["config_error"] = str(exc)[:200]

        # 2. Refresh knowledge store snapshot
        try:
            cymatix.genome.refresh()
            total = cymatix.genome.stats().get("total_genes", 0)
            changes["genome_genes"] = total
        except Exception as exc:
            changes["genome_error"] = str(exc)[:200]

        # 3. Rebuild SEMA vector cache
        try:
            cymatix.genome.invalidate_sema_cache()
            # #339 rider: the refresh() above reopened the WAL snapshot to
            # see external writers, so the cold-tier cache may be stale
            # too. Invalidate it (query_cold_tier lazy-rebuilds).
            cymatix.genome.invalidate_cold_sema_cache()
            cymatix.genome._build_sema_cache()
            cache = cymatix.genome._sema_cache
            if cache:
                changes["sema_vectors"] = len(cache["gene_ids"])
        except Exception as exc:
            changes["sema_error"] = str(exc)[:200]

        # 4. Clear last_query_scores
        cymatix.genome.last_query_scores = {}

        log.info("Admin reload complete: %s", changes)
        return {"reloaded": True, "changes": changes}

    # ---- Admin: hot-swap knowledge store .db file ----

    @app.post("/admin/swap-db", dependencies=_admin_auth)
    async def admin_swap_db(request: Request):
        """Hot-swap the knowledge store .db file without restarting.

        Body: { "path": "genomes/bench/oauth/small.db", "read_only": false }
        """
        import os as _os
        import time as _time

        data = await request.json()
        path = data.get("path", "")
        read_only = bool(data.get("read_only", False))

        if not path or not _os.path.exists(path):
            return JSONResponse(
                {"error": f"path not found: {path}"}, status_code=400,
            )

        # Opt-in allowlist (2026-08-08 audit): a non-empty [server]
        # swap_db_roots restricts which paths this endpoint may open.
        # Default [] keeps today's unrestricted behavior.
        roots = [r for r in getattr(config.server, "swap_db_roots", []) if r]
        if roots:
            resolved = _os.path.realpath(path)
            allowed = False
            for root in roots:
                root_resolved = _os.path.realpath(root)
                try:
                    if _os.path.commonpath([root_resolved, resolved]) == root_resolved:
                        allowed = True
                        break
                except ValueError:
                    # Different drives (Windows) or abs/rel mix — no match.
                    continue
            if not allowed:
                return JSONResponse(
                    {"error": f"path outside [server] swap_db_roots: {path}"},
                    status_code=403,
                )

        t0 = _time.time()
        old_path = str(cymatix.genome.path)

        try:
            from ..context_manager import _apply_symbol_expansion_cap
            from ..sharding import build_genome_kwargs, open_read_source

            # 2026-08-12 audit: construct the swapped store from the SAME
            # shared kwargs-builder the boot path uses. The previous
            # hand-copied kwargs list here drifted behind boot four separate
            # times (dense/ANN 2026-05-18, #214, #222/#223, #260) and was
            # missing 32 knobs at audit time (filename_anchor/bm25 among
            # them — the 2026-08-09 receipt invalidations). See
            # tests/test_swap_db_knob_sync.py for the per-knob contract.
            new_store = open_read_source(
                genome_path=path,
                **build_genome_kwargs(
                    config, sema_codec=getattr(cymatix, "_sema_codec", None)
                ),
            )
            new_store.read_only = read_only
            # Boot applies this post-construction (WS3); mirror it here.
            _apply_symbol_expansion_cap(
                new_store, config.retrieval.symbol_expansion_cap
            )

            # Rebuild caches on the new store
            new_store.invalidate_sema_cache()
            new_store._build_sema_cache()
            # #339 rider: deliberately NO invalidate_cold_sema_cache() here.
            # new_store is freshly constructed by open_read_source(), so its
            # cold-tier cache is unbuilt (_cold_sema_cache=None,
            # _cold_sema_vectorless=False at __init__) — there is no stale
            # cold state to drop, and the old store's caches die with
            # old_store.close() below. query_cold_tier lazy-builds from the
            # swapped-in db on first use.

            # Atomic swap
            old_store = cymatix.genome
            cymatix.genome = new_store
            cymatix.genome.last_query_scores = {}

            # Tier-0 fix (2026-05-17): repoint the session Registry at the
            # new store. The Registry captures a genome reference at app
            # construction (app.py: Registry(cymatix.genome)) and uses
            # genome.conn directly for every read/write — including the
            # background sweep task. Without this repoint, after a swap the
            # Registry still holds the OLD store, which old_store.close()
            # below then closes; the next _background_registry_sweep tick
            # raises "sqlite3.ProgrammingError: Cannot operate on a closed
            # database". Must run BEFORE old_store.close() so a sweep that
            # fires mid-swap never observes a closed connection.
            try:
                registry.genome = new_store
            except Exception:
                log.warning("swap-db: failed to repoint registry genome", exc_info=True)

            # Tier-0 follow-up #4 (2026-05-17): repoint the VaultManager at
            # the new store too. Like the Registry above, VaultManager
            # captures cymatix.genome at construction (app.py:
            # VaultManager(genome=cymatix.genome)); its pruner thread runs
            # refresh_stale_view(genome=self.genome) on a timer, so without
            # this repoint a post-swap prune cycle would hit the closed old
            # store — the same closed-database failure. Vault is opt-in, so
            # this only bites when vault.enabled=true. Runs BEFORE close().
            try:
                request.app.state.vault.genome = new_store
            except Exception:
                log.warning("swap-db: failed to repoint vault genome", exc_info=True)

            # Close old store (best-effort)
            try:
                old_store.close()
            except Exception:
                log.warning("swap-db: failed to close old store", exc_info=True)

            elapsed_ms = round((_time.time() - t0) * 1000, 1)
            # 2026-08-03 ERB fix: stats() here cost 72.7s cold on a 49GB bed and
            # broke the bench 60s swap timeout; only the count is reported.
            genes = new_store.total_genes()

            log.info(
                "swap-db: %s -> %s (%d genes, read_only=%s, %.1fms)",
                old_path, path, genes, read_only, elapsed_ms,
            )
            return {
                "swapped": True,
                "old_path": old_path,
                "new_path": str(path),
                "read_only": read_only,
                "genes": genes,
                "elapsed_ms": elapsed_ms,
            }
        except Exception as exc:
            log.warning("swap-db failed: %s", exc, exc_info=True)
            return JSONResponse(
                {"error": f"swap failed: {exc}"}, status_code=500,
            )

    # ---- Bridge: shared memory between AI assistants ----

    @app.get("/bridge/status")
    async def bridge_status():
        def _collect_status() -> Dict[str, object]:
            signals = bridge.list_signals()
            inbox_count = (
                len(list(bridge.inbox.iterdir())) if bridge.inbox.exists() else 0
            )
            return {
                "shared_dir": str(bridge.shared_dir),
                "inbox_pending": inbox_count,
                "signals": signals,
            }

        return await asyncio.to_thread(_collect_status)

    @app.post("/bridge/collect")
    async def bridge_collect():
        """Collect inbox files and ingest into knowledge store."""
        items = await asyncio.to_thread(bridge.collect_inbox)
        gene_ids: list = []
        for item in items:
            try:
                ids = await cymatix.ingest_async(
                    item["content"],
                    content_type="text",
                    metadata={"path": f"__bridge_{item['source']}__"},
                )
                gene_ids.extend(ids)
            except Exception:
                log.warning("Bridge ingest failed for %s", item["path"], exc_info=True)

        # Update shared context
        try:
            stats_snapshot = cymatix.stats()
            await asyncio.to_thread(bridge.update_shared_context, stats_snapshot)
        except Exception:
            log.warning("Bridge shared-context update failed", exc_info=True)
        return {"collected": len(items), "genes_created": len(gene_ids)}

    @app.post("/bridge/signal")
    async def bridge_signal(request: Request):
        body = await request.json()
        name = body.get("name", "unnamed")
        data = body.get("data", {})
        if not isinstance(data, dict):
            return JSONResponse(
                {"error": "data must be a JSON object"}, status_code=400,
            )
        try:
            bridge.write_signal(name, data)
        except ValueError as exc:
            # Containment violation — traversal / absolute signal name.
            return JSONResponse({"error": str(exc)}, status_code=400)
        return {"ok": True, "signal": name}
