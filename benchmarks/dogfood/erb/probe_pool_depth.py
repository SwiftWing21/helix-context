"""probe_pool_depth.py -- ARM 1: pool-absence forensics on the ERB 947k bed.

THE QUESTION
    Does "pool_absent" mean gold is genuinely unretrievable by the lexical
    substrate, or does it mean gold sits deep in the BM25 ranking and the
    shipped pipeline truncates admission before it can be scored?

PRE-REGISTERED KILL CRITERION (written before any result was looked at)
    KILL the entire admission thesis -- and with it the candidate-depth A/B and
    the density-at-admission arm -- if FEWER THAN 30 of the 109 pool_absent
    misses have gold anywhere in a 1000-deep candidate pool.

------------------------------------------------------------------------
WHERE ADMISSION IS ACTUALLY BOUNDED (traced, this worktree, git 3ad934c)
------------------------------------------------------------------------
knowledge_store.query_genes:
  :2806  limit = max_genes * 2                     -> 24 at max_genes_per_turn=12
  :2812  _fts_fetch_depth = self._fts5_candidate_depth or (limit * 2)   -> 48
  :3195-3211  Tier-3 FTS5 content lane:
             SELECT gene_id, rank FROM genes_fts WHERE genes_fts MATCH ?
             ORDER BY rank LIMIT ?   <- the ONLY consumer of _fts_fetch_depth.
  Tier 1 tag_exact (:3155) and Tier 2 tag_prefix (:3180) carry NO LIMIT --
  fts5_candidate_depth does NOT bound them.
  :3838-3868  BM25 SHORTLIST POST-FILTER (retrieval.bm25_shortlist_enabled,
             default TRUE, retrieval.bm25_shortlist_size, default 50):
             gene_scores is filtered to genes in the BM25 top-N.  This DELETES
             every candidate -- from ANY tier, including the unbounded tag
             lanes -- that is not in the BM25 top-50 of the same OR-joined
             term query.
  :3899-3906 under fusion_mode="rrf", eligible_ids = set(gene_scores) --
             i.e. the shortlist filter also governs the fused output.
  :3987-3988 last_query_scores = final_scores (the combinator output over
             eligible_ids). THIS is the "observed score map" every ERB
             receipt reads.

=> The published ~45-deep observed map floor is NOT fts5_candidate_depth=48.
   It is bm25_shortlist_size=50 (minus rows dropped by the chromatin /
   party filters). Two knobs must move together to widen admission.

ARMS
  shipped   : fts5_candidate_depth=0 (auto 48), bm25_shortlist_size=50
  deep1000  : fts5_candidate_depth=1000,        bm25_shortlist_size=1000

BASIS SEMANTICS -- read this before quoting a number
  The route trace above describes the historical implementation, not a
  measurement of today's request. Final score-map size alone cannot locate
  a candidate drop.
  * deep1000 rank = rank in the final score map after build_context with
    both depth knobs raised. It does not establish upstream admission.
  * --stage-provenance records actual stage membership. The admission verdict
    requires post_shortlist measurements for the full 109-query cohort.
    Failures, unsupported paths, and incomplete cohorts are INCONCLUSIVE.
  * bm25_rank      = a PROXY: raw FTS5 bm25 ordering of the pipeline's own
    OR-joined query_terms to depth 20000. Not the fused ranking. Used only to
    say whether gold is reachable by the lexical substrate at all.

Usage:
    python benchmarks/dogfood/erb/probe_pool_depth.py --stage-provenance --cache NEW_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("CYMATIX_DISABLE_LEARN", "1")

BED = "F:/tmp/erb_blob_v09x.db"
CONFIG = HERE / "configs" / "w24_min_delivered_12.toml"
NEEDLES = HERE / "needles_resolved_v09x_full.json"
GOLD = HERE / "gold_by_needle_v09x.json"
MINE = HERE / "receipts" / "interloper_mine_erb947k_2026-08-31.json"
LEDGER = HERE / "receipts" / "bucket_ledger_2026-09-01.json"

K = 12
DEEP = 1000
BM25_PROXY_DEPTH = 20000
CONTROL_N = 60
CONTROL_SEED = 20260901

KILL_CRITERION = (
    "KILL the entire admission thesis -- and with it the candidate-depth A/B "
    "and the density-at-admission arm -- if FEWER THAN 30 of the 109 "
    "pool_absent misses have gold anywhere in a 1000-deep candidate pool "
    "(deep1000 arm, retrieval.fts5_candidate_depth=1000 + "
    "retrieval.bm25_shortlist_size=1000, shipped pipeline otherwise "
    "untouched). Evaluate against measured post_shortlist gold membership "
    "for all 109 misses. Missing stages, query failures, and partial cohorts "
    "make the verdict INCONCLUSIVE. KILL rejects this depth intervention; "
    "it does not establish that gold is absent from the corpus or that "
    "other retrieval interventions cannot help."
)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_sha() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=20,
        ).stdout.strip() or None
    except Exception:
        return None


def rank_gold(scores: Dict[str, float], gold) -> Tuple[Optional[int], List[int]]:
    """1-indexed rank of gold in a score map; ties broken by gene_id ascending
    (identical contract to ablation_ladder.rank_gold)."""
    gold_set = set(gold)
    ordered = [g for g, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]
    ranks = [i for i, g in enumerate(ordered, 1) if g in gold_set]
    return (ranks[0] if ranks else None, ranks)


def load_inputs():
    needles = {n["name"]: n for n in json.loads(
        NEEDLES.read_text(encoding="utf-8"))["needles"]}
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    bucket = {m["needle"]: m["bucket"] for m in ledger["per_miss"]}
    miss_names = [m["needle"] for m in json.loads(
        MINE.read_text(encoding="utf-8"))["misses"]]
    return needles, gold, bucket, miss_names


def pick_controls(needles, miss_names, n):
    """Fixed-seed, question_type-stratified sample of HIT needles (largest-
    remainder allocation so the strata sum to exactly n)."""
    hits = sorted(set(needles) - set(miss_names))
    by_type: Dict[str, List[str]] = {}
    for name in hits:
        by_type.setdefault(needles[name]["question_type"], []).append(name)
    total = len(hits)
    quotas = {t: len(v) * n / total for t, v in by_type.items()}
    alloc = {t: int(q) for t, q in quotas.items()}
    rem = n - sum(alloc.values())
    for t in sorted(quotas, key=lambda t: (-(quotas[t] - alloc[t]), t))[:rem]:
        alloc[t] += 1
    rng = random.Random(CONTROL_SEED)
    out: List[str] = []
    for t in sorted(by_type):
        pool = sorted(by_type[t])
        out.extend(rng.sample(pool, min(alloc.get(t, 0), len(pool))))
    return sorted(out)


class TermCapture:
    """Records the query_terms query_genes was actually called with, so the
    raw-BM25 proxy can be built from the pipeline's OWN expanded terms rather
    than a re-implementation of keyword extraction."""

    def __init__(self) -> None:
        self.last: Optional[List[str]] = None
        self._orig = None
        self._cls = None

    def install(self, store_cls) -> None:
        # NOTE: KnowledgeStore.query_genes is an ALIAS bound to the same
        # function object as query_docs, and context_manager.py:3100 calls
        # ``self.genome.query_docs(...)``. Patching the alias name therefore
        # captures nothing -- patch query_docs (and re-point the alias so the
        # two names stay the same object).
        orig = store_cls.query_docs
        cap = self

        def wrapper(self_store, domains, entities, *a, **kw):
            try:
                cap.last = (list(self_store._expand_terms(list(domains)))
                            + list(self_store._expand_terms(list(entities))))
            except Exception:
                cap.last = None
            return orig(self_store, domains, entities, *a, **kw)

        store_cls.query_docs = wrapper
        store_cls.query_genes = wrapper
        self._orig = orig
        self._cls = store_cls

    def uninstall(self) -> None:
        if self._cls is not None and self._orig is not None:
            self._cls.query_docs = self._orig
            self._cls.query_genes = self._orig


def open_capture(path: Path):
    """Reusable capture artifact: the FULL score map of every arm x needle,
    so downstream arms never have to re-run the expensive deep pass."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(
        "PRAGMA journal_mode=WAL;"
        "CREATE TABLE IF NOT EXISTS pool ("
        "  arm TEXT NOT NULL, needle TEXT NOT NULL, rank INTEGER NOT NULL,"
        "  gene_id TEXT NOT NULL, score REAL NOT NULL, is_gold INTEGER NOT NULL,"
        "  PRIMARY KEY (arm, needle, rank));"
        "CREATE INDEX IF NOT EXISTS idx_pool_gene ON pool(arm, needle, gene_id);"
        "CREATE TABLE IF NOT EXISTS terms ("
        "  needle TEXT PRIMARY KEY, query_terms TEXT NOT NULL);"
    )
    return con


def write_pool(con, arm_name, needle, scores, gold_set):
    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    con.executemany(
        "INSERT OR REPLACE INTO pool(arm,needle,rank,gene_id,score,is_gold) "
        "VALUES (?,?,?,?,?,?)",
        [(arm_name, needle, i, gid, float(sc), 1 if gid in gold_set else 0)
         for i, (gid, sc) in enumerate(ordered, 1)])


def run_arm(arm_name, knobs, targets, needles, gold, capture_terms,
            capture_con=None, ckpt_dir=None, stage_provenance=False):
    """Run one arm over *targets*.

    NEEDLE-LEVEL CHECKPOINT: when ``ckpt_dir`` is given, every completed
    needle is appended to ``<ckpt_dir>/<arm>.jsonl`` and a restart skips the
    needles already recorded there. This run was killed twice mid-arm by the
    supervising harness; arm-level caching alone loses ~an hour each time.
    Consequence for the latency section, on the record: a RESUMED arm's
    wall_ms vector mixes segments run against different SQLite page-cache
    states, so its p50 is a weaker latency estimate than a single
    uninterrupted pass. ``resumed_segments`` records how many restarts the
    arm took."""
    from cymatix_context.config import load_config
    from cymatix_context.context_manager import CymatixContextManager
    from cymatix_context.knowledge_store import KnowledgeStore
    from cymatix_context.retrieval.measurement import capture_stages

    cfg = load_config(str(CONFIG))
    cfg.genome.path = BED
    for path, val in knobs.items():
        sec, key = path.split(".", 1)
        obj = getattr(cfg, sec)
        if not hasattr(obj, key):
            raise SystemExit("unknown config path " + path)
        setattr(obj, key, val)

    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    ckpt_path = None
    done: Dict[str, Dict[str, Any]] = {}
    n_segments = 1
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / (arm_name + ".jsonl")
        if ckpt_path.exists():
            for line in ckpt_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if rec.get("__segment__"):
                    n_segments += 1
                    continue
                done[rec["needle"]] = rec
            if stage_provenance:
                require_stage_provenance(done.values(), ckpt_path, targets)
            print("  [" + arm_name + "] resuming: " + str(len(done))
                  + " needles already checkpointed", flush=True)
        with ckpt_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"__segment__": True,
                                 "started": time.time()}) + "\n")

    todo = [n for n in targets if n not in done]
    if not todo:
        records = [done[n] for n in targets if n in done]
        lat = [r["wall_ms"] for r in records]
        return {
            "arm": arm_name, "knobs": knobs, "n": len(records), "errors": [],
            "resumed_segments": n_segments,
            "wall_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "wall_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
            "wall_ms_p95": (round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)
                            if lat else None),
            "wall_ms_total": round(sum(lat), 1),
            "map_size_median": statistics.median([r["map_size"] for r in records]),
            "map_size_max": max([r["map_size"] for r in records]),
            "per_query": records,
        }

    cap = TermCapture()
    if capture_terms:
        cap.install(KnowledgeStore)

    manager = CymatixContextManager(cfg)
    try:
        # Untimed warmup (erb receipt convention) -- lazy encoders / SQLite
        # page cache must not be charged to the first needle.
        try:
            manager.build_context(needles[todo[0]]["query"], read_only=True,
                                  ignore_delivered=True, max_genes=K)
        except Exception as exc:
            errors.append("warmup: " + type(exc).__name__ + ": " + str(exc))

        for i, name in enumerate(todo):
            q = needles[name]["query"]
            g = gold.get(name, [])
            try:
                manager.genome.last_query_scores = {}
            except Exception:
                pass
            t0 = time.perf_counter()
            terms = None
            cap.last = None
            query_error = None
            try:
                with capture_stages(g, enabled=stage_provenance) as stage_capture:
                    window = manager.build_context(q, read_only=True,
                                                   ignore_delivered=True,
                                                   max_genes=K)
                wall = (time.perf_counter() - t0) * 1000.0
                scores = dict(manager.genome.last_query_scores or {})
                expressed = list(getattr(window, "expressed_gene_ids", None) or ())
            except Exception as exc:
                wall = (time.perf_counter() - t0) * 1000.0
                query_error = type(exc).__name__ + ": " + str(exc)
                errors.append(name + ": " + query_error)
                scores, expressed = {}, []
            if capture_terms:
                terms = cap.last
            first, all_r = rank_gold(scores, g)
            gset = set(g)
            if capture_con is not None:
                try:
                    write_pool(capture_con, arm_name, name, scores, gset)
                    if terms:
                        capture_con.execute(
                            "INSERT OR REPLACE INTO terms(needle,query_terms) "
                            "VALUES (?,?)", (name, json.dumps(terms)))
                    capture_con.commit()
                except Exception as exc:
                    errors.append(name + ": capture write failed: "
                                  + type(exc).__name__ + ": " + str(exc))
            drank = None
            for j, gid in enumerate(expressed, 1):
                if gid in gset:
                    drank = j
                    break
            rec = {
                "needle": name,
                "status": "failed" if query_error else "complete",
                "map_size": len(scores),
                "rank_of_first_gold": first,
                "gold_ranks": all_r,
                "delivered_gold": 1 if drank else 0,
                "delivered_gold_rank": drank,
                "delivered_count": len(expressed),
                "wall_ms": round(wall, 1),
                "query_terms": terms,
            }
            if query_error:
                rec["error"] = query_error
            if stage_provenance:
                rec["stage_provenance"] = stage_capture.report()
            records.append(rec)
            if ckpt_path is not None:
                with ckpt_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if (i + 1) % 25 == 0:
                print("  [" + arm_name + "] " + str(i + 1) + "/"
                      + str(len(todo)) + " (this segment)", flush=True)
    finally:
        if capture_terms:
            cap.uninstall()
        try:
            manager.genome.close()
        except Exception:
            pass

    # Merge checkpointed + freshly-run records back into target order.
    by_name = dict(done)
    for r in records:
        by_name[r["needle"]] = r
    records = [by_name[n] for n in targets if n in by_name]

    lat = [r["wall_ms"] for r in records]
    return {
        "arm": arm_name,
        "knobs": knobs,
        "n": len(records),
        "errors": errors,
        "resumed_segments": n_segments,
        "wall_ms_p50": round(statistics.median(lat), 1) if lat else None,
        "wall_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
        "wall_ms_p95": (round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)
                        if lat else None),
        "wall_ms_total": round(sum(lat), 1),
        "map_size_median": (statistics.median([r["map_size"] for r in records])
                            if records else None),
        "map_size_max": max([r["map_size"] for r in records]) if records else None,
        "per_query": records,
    }


def bm25_proxy(term_map, gold, depth):
    """Raw FTS5 bm25 ordering of the pipeline's own OR-joined query_terms.
    PROXY ONLY -- this is not the fused pipeline ranking."""
    con = sqlite3.connect("file:" + BED + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out: Dict[str, Any] = {}
    lat: List[float] = []
    try:
        cur = con.cursor()
        for name, terms in term_map.items():
            if not terms:
                out[name] = {"bm25_rank": None, "pool": None,
                             "status": "not_captured",
                             "note": "no query_terms captured"}
                continue
            match = " OR ".join('"' + t + '"' for t in terms if len(t) > 2)
            if not match:
                out[name] = {"bm25_rank": None, "pool": None,
                             "status": "not_captured",
                             "note": "no terms len>2"}
                continue
            t0 = time.perf_counter()
            try:
                rows = cur.execute(
                    "SELECT gene_id FROM genes_fts WHERE genes_fts MATCH ? "
                    "ORDER BY rank LIMIT ?", (match, depth)).fetchall()
            except Exception as exc:
                out[name] = {"bm25_rank": None, "pool": None,
                             "status": "failed",
                             "note": type(exc).__name__ + ": " + str(exc)}
                continue
            lat.append((time.perf_counter() - t0) * 1000.0)
            gs = set(gold.get(name, []))
            rank = None
            for i, r in enumerate(rows, 1):
                if r["gene_id"] in gs:
                    rank = i
                    break
            out[name] = {"bm25_rank": rank, "pool": len(rows), "status": "complete"}
    finally:
        con.close()
    return {"depth": depth, "per_needle": out,
            "wall_ms_p50": round(statistics.median(lat), 1) if lat else None,
            "wall_ms_total": round(sum(lat), 1)}


def source_map(wanted):
    """gene_id -> source_id for the ids in *wanted*. ONE full scan of genes
    (there is no index on genes.source_id on this bed)."""
    con = sqlite3.connect("file:" + BED + "?mode=ro", uri=True)
    out = {}
    t0 = time.perf_counter()
    try:
        for gid, sid in con.execute("SELECT gene_id, source_id FROM genes"):
            if gid in wanted:
                out[gid] = sid
    finally:
        con.close()
    return out, round((time.perf_counter() - t0), 1)


def doc_rollup(pool_rows, gold_set, src, k):
    """Rank parent documents by MAX (realizable, order-preserving) and by SUM
    of member chunk scores; return the rank of the first document holding any
    gold chunk under each rule.

    ERB CAVEAT on the record: gold here is DOCUMENT-COMPLETE (386 gold
    gene_ids == 386 chunks of the gold documents), so this measures ONLY
    whether NON-gold multi-chunk documents are fragmenting the ranking and
    consuming seats -- it is NOT a 'gold chunk vs its non-gold sibling'
    metric, which is an empty set on this bed."""
    best_max, best_sum = {}, {}
    for gid, sc in pool_rows:
        sid = src.get(gid, "__unmapped__:" + gid)
        best_max[sid] = max(best_max.get(sid, float("-inf")), sc)
        best_sum[sid] = best_sum.get(sid, 0.0) + sc
    gold_sids = {src.get(g, "__unmapped__:" + g) for g in gold_set}

    def rank_of(dmap):
        order = [s for s, _ in sorted(dmap.items(), key=lambda kv: (-kv[1], kv[0]))]
        for i, s in enumerate(order, 1):
            if s in gold_sids:
                return i
        return None

    rmax, rsum = rank_of(best_max), rank_of(best_sum)
    return {
        "n_docs": len(best_max),
        "doc_rank_max_rule": rmax,
        "doc_rank_sum_rule": rsum,
        "rescued_max_rule": 1 if (rmax is not None and rmax <= k) else 0,
        "rescued_sum_rule": 1 if (rsum is not None and rsum <= k) else 0,
    }


def stage_presence(record, stage_name):
    """Gold membership at a captured boundary, never inferred from final scores.

    Admission decisions require one completed lexical retrieval. Independent
    calls remain separate evidence, not a synthetic union of candidate pools.
    """
    if record.get("error") or record.get("status") == "failed":
        return "failed"
    report = record.get("stage_provenance")
    if not isinstance(report, dict) or report.get("version") != 1:
        return "not_captured"
    if report.get("status") == "failed":
        return "failed"
    if report.get("status") != "complete":
        return "not_captured"
    retrievals = report.get("retrievals")
    if not isinstance(retrievals, list):
        return "not_captured"
    if not retrievals:
        return "not_executed"
    if any(call.get("status") == "failed" for call in retrievals):
        return "failed"
    if len(retrievals) != 1 or retrievals[0].get("status") != "complete":
        return "not_captured"
    stage = retrievals[0].get("stages", {}).get(stage_name, {})
    status = stage.get("status")
    if status in ("failed", "not_executed"):
        return status
    if (status != "captured" or not isinstance(stage.get("gold_ids"), list)
            or not isinstance(stage.get("count"), int)):
        return "not_captured"
    return "present" if stage["gold_ids"] else "absent"


def admission_summary(records, *, expected_n=109, threshold=30):
    """Apply the registered threshold only to a complete measured cohort."""
    counts = dict.fromkeys(("present", "absent", "failed", "not_captured",
                           "not_executed"), 0)
    for rec in records:
        counts[stage_presence(rec, "post_shortlist")] += 1
    measured = counts["present"] + counts["absent"]
    complete = len(records) == expected_n and measured == expected_n
    return {
        "basis": "post_shortlist", "expected_n": expected_n,
        "n": len(records), "measured_n": measured, **counts,
        "threshold": threshold,
        "verdict": (("PASS" if counts["present"] >= threshold else "KILL")
                    if complete else "INCONCLUSIVE"),
    }


def require_stage_provenance(records, path, targets):
    """Refuse old caches rather than treating uncaptured boundaries as empty."""
    wanted = set(targets)
    for rec in records:
        if rec.get("needle") not in wanted:
            continue
        report = rec.get("stage_provenance")
        if (not isinstance(report, dict) or report.get("version") != 1
                or report.get("status") not in ("complete", "failed", "not_captured")
                or not isinstance(report.get("retrievals"), list)
                or not isinstance(report.get("stages"), dict)):
            raise ValueError(
                f"{path}: cached needle {rec.get('needle')!r} lacks stage provenance; "
                "use a fresh --cache directory to measure these boundaries"
            )


def histogram(ranks, statuses=None):
    """Final score-map ranks; absence here is not admission evidence."""
    h = {"le_48": 0, "r49_200": 0, "r201_1000": 0, "absent_at_1000": 0}
    if statuses is None:
        statuses = ["complete"] * len(ranks)
    else:
        h.update(failed=0, not_captured=0)
    for r, status in zip(ranks, statuses):
        if status != "complete":
            h["failed" if status == "failed" else "not_captured"] += 1
            continue
        if r is None:
            h["absent_at_1000"] += 1
        elif r <= 48:
            h["le_48"] += 1
        elif r <= 200:
            h["r49_200"] += 1
        else:
            h["r201_1000"] += 1
    return h


def bm25_window_attribution(rows):
    """Proxy ranks by window; these cannot identify an actual pipeline drop."""
    out = {"gold_in_bm25_top50": 0, "gold_bm25_51_1000": 0,
           "gold_bm25_1001_20000": 0, "gold_bm25_absent_20000": 0,
           "failed": 0, "not_captured": 0}
    for row in rows:
        status = row.get("bm25_proxy_status", "not_captured")
        rank = row["bm25_proxy_rank"]
        if status != "complete":
            out["failed" if status == "failed" else "not_captured"] += 1
        elif rank is None:
            out["gold_bm25_absent_20000"] += 1
        elif rank <= 50:
            out["gold_in_bm25_top50"] += 1
        elif rank <= 1000:
            out["gold_bm25_51_1000"] += 1
        else:
            out["gold_bm25_1001_20000"] += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", default="2026-09-01")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke mode: only run the first N targets")
    ap.add_argument("--cache", default=None,
                    help="directory for per-arm checkpoints (crash resume)")
    ap.add_argument("--capture", default=None,
                    help="path of the reusable pool-capture sqlite artifact")
    ap.add_argument("--stage-provenance", action="store_true",
                    help="capture retrieval/blend stage boundaries; required for an admission verdict")
    args = ap.parse_args(argv)

    needles, gold, bucket, miss_names = load_inputs()
    controls = pick_controls(needles, miss_names, CONTROL_N)
    targets = list(miss_names) + controls
    if args.limit:
        targets = targets[:args.limit]
    print("targets: " + str(len(miss_names)) + " misses + "
          + str(len(controls)) + " hit controls = " + str(len(targets)),
          flush=True)

    t_start = time.time()
    tkey = hashlib.sha256(("|".join(targets)).encode()).hexdigest()[:12]
    cache_dir = Path(args.cache) if args.cache else None

    capture_path = (Path(args.capture) if args.capture
                    else HERE / "receipts"
                    / ("pool_depth_capture_" + args.stamp + ".sqlite"))
    capture_con = open_capture(capture_path)

    def cached(arm_name, knobs, capture):
        cf = (cache_dir / ("arm_" + arm_name + "_" + tkey + ".json")
              if cache_dir else None)
        if cf is not None and cf.exists():
            print("resuming " + arm_name + " from " + str(cf), flush=True)
            result = json.loads(cf.read_text(encoding="utf-8"))
            if args.stage_provenance:
                records = result.get("per_query", [])
                if set(targets) != {r.get("needle") for r in records}:
                    raise ValueError(f"{cf}: stage provenance cache has a different target cohort")
                require_stage_provenance(records, cf, targets)
            return result
        res = run_arm(arm_name, knobs, targets, needles, gold, capture,
                      capture_con=capture_con, ckpt_dir=cache_dir,
                      stage_provenance=args.stage_provenance)
        if cf is not None:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps(res), encoding="utf-8")
        return res

    # ARM ORDER: deep1000 FIRST, on purpose. It is the irreplaceable arm --
    # the shipped arm is ~90% recoverable from the published basis ladder
    # receipt (per-needle rank_of_first_gold + delivered_gold, identical
    # config). Running deep first also means the deep arm pays the COLD
    # SQLite page cache and the shipped arm runs warm, which biases the
    # deep/shipped latency ratio UPWARD -- i.e. it OVERSTATES the cost of
    # depth rather than hiding it. Stated here so the latency section is not
    # read as a warm-vs-warm comparison.
    deep = cached("deep1000",
                  {"retrieval.fts5_candidate_depth": DEEP,
                   "retrieval.bm25_shortlist_size": DEEP}, True)
    print("deep1000 done p50=" + str(deep["wall_ms_p50"]) + "ms map_med="
          + str(deep["map_size_median"]), flush=True)
    shipped = cached("shipped", {}, False)
    print("shipped done p50=" + str(shipped["wall_ms_p50"]) + "ms map_med="
          + str(shipped["map_size_median"]), flush=True)

    term_map = {r["needle"]: (r["query_terms"] or [])
                if r.get("status") == "complete" else []
                for r in deep["per_query"]}
    proxy = bm25_proxy(term_map, gold, BM25_PROXY_DEPTH)
    print("bm25 proxy done p50=" + str(proxy["wall_ms_p50"]) + "ms", flush=True)

    ship_by = {r["needle"]: r for r in shipped["per_query"]}
    deep_by = {r["needle"]: r for r in deep["per_query"]}
    miss_set = set(miss_names)

    rows: List[Dict[str, Any]] = []
    for name in targets:
        s, d = ship_by[name], deep_by[name]
        px = proxy["per_needle"].get(name, {})
        rows.append({
            "needle": name,
            "cohort": "miss" if name in miss_set else "hit_control",
            "bucket": bucket.get(name),
            "question_type": needles[name]["question_type"],
            "shipped_status": ("failed" if s.get("stage_provenance", {}).get("status") == "failed"
                               else s.get("status", "not_captured")),
            "deep_status": ("failed" if d.get("stage_provenance", {}).get("status") == "failed"
                            else d.get("status", "not_captured")),
            "shipped_map_size": s["map_size"],
            "shipped_rank": s["rank_of_first_gold"],
            "shipped_delivered_gold": s["delivered_gold"],
            "deep_map_size": d["map_size"],
            "deep_rank": d["rank_of_first_gold"],
            "deep_gold_ranks": d["gold_ranks"],
            "deep_delivered_gold": d["delivered_gold"],
            "deep_delivered_gold_rank": d["delivered_gold_rank"],
            "bm25_proxy_rank": px.get("bm25_rank"),
            "bm25_proxy_pool": px.get("pool"),
            "bm25_proxy_status": px.get("status", "not_captured"),
            "shipped_wall_ms": s["wall_ms"],
            "deep_wall_ms": d["wall_ms"],
            "query_terms_n": len(d["query_terms"] or []),
        })
        if args.stage_provenance:
            rows[-1]["stage_provenance"] = {
                "shipped": s["stage_provenance"], "deep1000": d["stage_provenance"],
            }
            rows[-1]["shipped_admission"] = stage_presence(s, "post_shortlist")
            rows[-1]["deep_admission"] = stage_presence(d, "post_shortlist")

    def score_hist(grp):
        return histogram([r["deep_rank"] for r in grp],
                         statuses=[r["deep_status"] for r in grp])

    def subset(pred):
        return [r for r in rows if pred(r)]

    pool_absent = subset(lambda r: r["bucket"] == "pool_absent")
    near_band = subset(lambda r: r["bucket"] == "near_band")
    seat_capped = subset(lambda r: r["bucket"] == "seat_capped")
    misses = subset(lambda r: r["cohort"] == "miss")
    hitctl = subset(lambda r: r["cohort"] == "hit_control")

    pa_found = sum(1 for r in pool_absent
                   if r["deep_status"] == "complete" and r["deep_rank"] is not None)
    admission = admission_summary([deep_by[r["needle"]] for r in pool_absent])
    verdict = admission["verdict"]

    by_bucket = {}
    for label, grp in (("pool_absent", pool_absent), ("near_band", near_band),
                       ("seat_capped", seat_capped), ("all_misses", misses),
                       ("hit_control", hitctl)):
        measured = [r for r in grp if r["deep_status"] == "complete"]
        dr = [r["deep_rank"] for r in measured]
        found = [x for x in dr if x is not None]
        pxr = [r["bm25_proxy_rank"] for r in grp
               if r["bm25_proxy_status"] == "complete" and r["bm25_proxy_rank"] is not None]
        by_bucket[label] = {
            "n": len(grp),
            "n_measured": len(measured),
            "deep_hist": score_hist(grp),
            "n_found_at_1000": len(found),
            "found_rate": round(len(found) / len(measured), 4) if measured else None,
            "deep_rank_median": statistics.median(found) if found else None,
            "deep_rank_max": max(found) if found else None,
            "bm25_proxy_found": len(pxr),
            "bm25_proxy_median": statistics.median(pxr) if pxr else None,
            "deep_delivered_gold": sum(r["deep_delivered_gold"] for r in grp),
            "shipped_delivered_gold": sum(r["shipped_delivered_gold"] for r in grp),
        }

    by_qtype = {}
    for qt in sorted({r["question_type"] for r in misses}):
        grp = [r for r in misses if r["question_type"] == qt]
        by_qtype[qt] = {
            "n": len(grp),
            "deep_hist": score_hist(grp),
            "n_found_at_1000": sum(1 for r in grp if r["deep_status"] == "complete"
                                   and r["deep_rank"] is not None),
            "n_pool_absent": sum(1 for r in grp if r["bucket"] == "pool_absent"),
        }

    for label, grp in (("pool_absent", pool_absent), ("near_band", near_band),
                       ("seat_capped", seat_capped), ("all_misses", misses),
                       ("hit_control", hitctl)):
        by_bucket[label]["bm25_window_attribution"] = bm25_window_attribution(grp)

    sem_pa = [r for r in pool_absent if r["question_type"] == "semantic"]
    nonsem_pa = [r for r in pool_absent if r["question_type"] != "semantic"]

    nb_deep = [r for r in near_band
               if r["deep_status"] == "complete" and r["deep_rank"] is not None]

    # ── P2 residual: does a doc rollup lift the near_band misses? ───────
    # 32 of the 41 near_band misses sit at live_rank 19-45, outside the
    # 15-row mine window, so nobody had measured this. Run it over BOTH the
    # shipped map (the realizable question) and the deep map.
    cur_cap = capture_con.cursor()
    wanted = set()
    pools: Dict[str, Dict[str, List]] = {"shipped": {}, "deep1000": {}}
    for arm, needle, rank_, gid, sc in cur_cap.execute(
            "SELECT arm,needle,rank,gene_id,score FROM pool ORDER BY arm,needle,rank"):
        pools.setdefault(arm, {}).setdefault(needle, []).append((gid, sc))
        wanted.add(gid)
    for name in targets:
        wanted.update(gold.get(name, []))
    src, scan_s = source_map(wanted)

    rollup_rows = []
    for r in rows:
        name = r["needle"]
        gset = set(gold.get(name, []))
        entry = {"needle": name, "cohort": r["cohort"], "bucket": r["bucket"],
                 "shipped_rank": r["shipped_rank"], "deep_rank": r["deep_rank"]}
        for arm in ("shipped", "deep1000"):
            pr = pools.get(arm, {}).get(name, [])
            entry[arm] = doc_rollup(pr, gset, src, K) if pr else None
        rollup_rows.append(entry)

    def rollup_summary(pred):
        grp = [e for e in rollup_rows if pred(e)]
        out = {"n": len(grp)}
        for arm in ("shipped", "deep1000"):
            got = [e[arm] for e in grp if e[arm]]
            out[arm] = {
                "n_scored": len(got),
                "rescued_max_rule": sum(g["rescued_max_rule"] for g in got),
                "rescued_sum_rule": sum(g["rescued_sum_rule"] for g in got),
                "doc_rank_max_median": (statistics.median(
                    [g["doc_rank_max_rule"] for g in got
                     if g["doc_rank_max_rule"] is not None])
                    if any(g["doc_rank_max_rule"] is not None for g in got) else None),
                "n_docs_median": (statistics.median([g["n_docs"] for g in got])
                                  if got else None),
            }
        return out

    receipt = {
        "stamp": args.stamp,
        "probe": "ARM 1 -- pool-absence forensics (candidate-admission depth)",
        "tool": "benchmarks/dogfood/erb/probe_pool_depth.py",
        "tool_sha256": sha256_file(Path(__file__)),
        "git_sha": git_sha(),
        "bed": BED,
        "bed_bytes": Path(BED).stat().st_size,
        "bed_access": ("read-write open by KnowledgeStore construction "
                       "(idempotent DDL); all queries read_only=True, "
                       "CYMATIX_DISABLE_LEARN=1; raw BM25 proxy opened mode=ro"),
        "config": str(CONFIG),
        "basis_receipts": [
            "benchmarks/dogfood/erb/receipts/ladder_v09x_w24_min_delivered_2026-08-28.json",
            "benchmarks/dogfood/erb/receipts/interloper_mine_erb947k_2026-08-31.json",
            "benchmarks/dogfood/erb/receipts/bucket_ledger_2026-09-01.json",
        ],
        "kill_criterion": KILL_CRITERION,
        "verdict": verdict,
        "admission_measurement": admission,
        "k": K,
        "deep_depth": DEEP,
        "bm25_proxy_depth": BM25_PROXY_DEPTH,
        "n_misses": len(misses),
        "n_hit_controls": len(hitctl),
        "control_seed": CONTROL_SEED,
        "admission_trace": {
            "fts_lane": ("fts_raw records bounded SQL rows after any tier-0 BM25 "
                         "prefilter, before lifecycle and party filters"),
            "bm25_shortlist": ("pre_shortlist and post_shortlist record same-call "
                               "membership around the optional shortlist filter"),
            "map_publication": ("post_blend_scores records the request-local score "
                                "map after blend refiners; it is a separate stage"),
            "conclusion": ("Compare captured boundaries within each retrieval call. "
                           "Final map size and proxy ranks alone cannot identify "
                           "where a candidate was removed."),
        },
        "basis_semantics": {
            "deep_rank": ("final score-map rank after build_context with both "
                          "depth knobs raised to 1000; this map is not a "
                          "measurement of upstream candidate admission"),
            "deep_hist": ("legacy rank bins in the final score map; "
                          "absent_at_1000 means absent from that map in the "
                          "deep1000 arm, not absent at any upstream stage. "
                          "Failed and unmeasured queries have separate bins"),
            "admission_measurement": ("post_shortlist gold membership from one "
                                      "completed lexical retrieval per query; complete cohort "
                                      "measurement is required for a verdict"),
            "bm25_proxy_rank": ("PROXY -- raw FTS5 bm25 order of the pipeline's "
                                "own OR-joined query_terms to depth 20000; NOT "
                                "the fused ranking"),
        },
        "arms": {
            "shipped": {kk: vv for kk, vv in shipped.items() if kk != "per_query"},
            "deep1000": {kk: vv for kk, vv in deep.items() if kk != "per_query"},
        },
        "latency": {
            "arm_order": ("deep1000 ran FIRST (cold SQLite page cache), "
                          "shipped SECOND (warm) -- the ratio therefore "
                          "OVERSTATES the cost of depth"),
            "shipped_p50_ms": shipped["wall_ms_p50"],
            "deep1000_p50_ms": deep["wall_ms_p50"],
            "ratio_p50": (round(deep["wall_ms_p50"] / shipped["wall_ms_p50"], 3)
                          if shipped["wall_ms_p50"] else None),
            "shipped_p95_ms": shipped["wall_ms_p95"],
            "deep1000_p95_ms": deep["wall_ms_p95"],
            "bm25_proxy_p50_ms": proxy["wall_ms_p50"],
        },
        "headline": {
            "pool_absent_n": len(pool_absent),
            "pool_absent_gold_found_at_1000": pa_found,
            "pool_absent_hist": score_hist(pool_absent),
            "semantic_pool_absent_n": len(sem_pa),
            "semantic_pool_absent_found": sum(
                1 for r in sem_pa if r["deep_status"] == "complete" and r["deep_rank"] is not None),
            "nonsemantic_pool_absent_n": len(nonsem_pa),
            "nonsemantic_pool_absent_found": sum(
                1 for r in nonsem_pa if r["deep_status"] == "complete" and r["deep_rank"] is not None),
        },
        "by_bucket": by_bucket,
        "by_question_type_misses": by_qtype,
        "p2_residual_near_band": {
            "n_near_band": len(near_band),
            "n_shipped_rank_gt_15": sum(
                1 for r in near_band if (r["shipped_rank"] or 0) > 15),
            "n_found_deep": len(nb_deep),
            "deep_rank_hist": score_hist(near_band),
            "deep_delivered_gold": sum(r["deep_delivered_gold"] for r in near_band),
            "doc_rollup": {
                "definition": ("parent source_id rollup; doc score = MAX "
                               "(realizable) or SUM of member chunk scores; "
                               "rescue = gold DOCUMENT reaches doc-rank <= 12"),
                "erb_caveat": ("ERB gold is DOCUMENT-COMPLETE, so this "
                               "measures only whether NON-gold multi-chunk "
                               "docs fragment the ranking; it is NOT a 'gold "
                               "chunk vs non-gold sibling' metric (empty set "
                               "on this bed)"),
                "source_scan_s": scan_s,
                "near_band": rollup_summary(lambda e: e["bucket"] == "near_band"),
                "near_band_rank_gt_15": rollup_summary(
                    lambda e: e["bucket"] == "near_band"
                    and (e["shipped_rank"] or 0) > 15),
                "pool_absent": rollup_summary(lambda e: e["bucket"] == "pool_absent"),
                "seat_capped": rollup_summary(lambda e: e["bucket"] == "seat_capped"),
                "all_misses": rollup_summary(lambda e: e["cohort"] == "miss"),
                "hit_control": rollup_summary(lambda e: e["cohort"] == "hit_control"),
            },
        },
        "capture_artifact": {
            "path": str(capture_path),
            "bytes": capture_path.stat().st_size if capture_path.exists() else None,
            "rows": cur_cap.execute("SELECT COUNT(*) FROM pool").fetchone()[0],
            "schema": ("pool(arm, needle, rank, gene_id, score, is_gold) + "
                       "terms(needle, query_terms JSON)"),
            "note": ("Full score map of BOTH arms for all 216 targets. Reuse "
                     "this instead of re-running the deep pass."),
        },
        "rollup_rows": rollup_rows,
        "deep_arm_delivery": {
            "misses_rescued_by_deep_arm": sum(r["deep_delivered_gold"] for r in misses),
            "hit_controls_lost_by_deep_arm": sum(
                1 for r in hitctl if r["shipped_delivered_gold"] == 1
                and r["deep_delivered_gold"] == 0),
            "hit_controls_kept": sum(r["deep_delivered_gold"] for r in hitctl),
            "note": ("Delivery numbers here are a SIDE OBSERVATION on a "
                     "non-random subset (all 156 misses + 60 sampled hits); "
                     "they are NOT a delivered_gold_rate estimate for the bed."),
        },
        "wall_s_total": round(time.time() - t_start, 1),
        "rows": rows,
    }

    out = (Path(args.out) if args.out
           else HERE / "receipts" / ("pool_depth_forensics_" + args.stamp + ".json"))
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    capture_con.close()
    print("\nVERDICT " + verdict + ": measured post_shortlist gold presence = "
          + str(admission["present"]) + "/" + str(admission["measured_n"])
          + " (requires 109 measured queries; threshold 30)")
    print(json.dumps(receipt["headline"], indent=2))
    print(json.dumps(receipt["latency"], indent=2))
    print("receipt -> " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
