"""Stacked PLR fuser — query-quality confidence head.

Loads the GradientBoostingClassifier trained by `scripts/pwpc/sprint3.py` and
exposes a single scoring API: given the per-query tier totals + window
correlations + cos(query, top_candidate) that /context already computes, return
the log-odds that the user will re-query within 60s (bucket B) under the
current labelling discipline.

## Scope — read before wiring

STATISTICAL_FUSION.md §C3 describes a **per-(query, gene) stacked fuser** that
replaces the additive gene ranker. The CWoLa logger shipped since Sprint 1 has
been query-level, not per-gene: `cwola_log.tier_features` holds the *sum* of
tier contributions across all genes in the retrieval (see
`server.py::log_query` call site near line 900). The trained artifact is
therefore a **query-quality head**, not a ranker — every candidate in a given
query has the same feature vector, so the classifier can't order them.

Option B (per-(q,g) refactor): change `cwola.log_query()` to emit one row per
top-K candidate with per-gene tier scores, reset the ~2K-row label corpus,
re-accumulate. Not rejected — may be necessary if the additive `lex_anchor +291`
problem needs the spec's original per-gene fuser. For now we ship A and treat
B as a considered follow-up.

## How to use

Config: `[plr] enabled = true`, `model_path = "training/models/stacked_plr.joblib"`.

The /context packet path calls `StackedPLRFuser.query_confidence()` with the
same aggregates already computed for CWoLa logging, attaches the log-odds to
the response as `plr_confidence`, and leaves gene ranking untouched.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

log = logging.getLogger("helix.fusion_plr")

# Must match `scripts/pwpc/sprint3.py::MODEL_SCHEMA_VERSION`. Bumping the
# trainer schema rejects older artifacts at load time instead of silently
# scoring against the wrong feature layout.
EXPECTED_SCHEMA_VERSION = 1

# Keep in sync with the trainer. Tier keys first (9), then any window
# correlation keys the artifact provides, then cos_q_c as the last column.
TIER_KEYS = [
    "fts5", "splade", "sema_boost", "lex_anchor",
    "tag_exact", "tag_prefix", "pki", "harmonic", "sr",
]


class PLRLoadError(RuntimeError):
    """Raised when a model artifact cannot be loaded or trusted."""


class StackedPLRFuser:
    """Wraps the trained classifier + feature-layout contract.

    The fuser is immutable after construction: the feature-name list, schema
    version, and classifier are frozen at load time. The scoring path only
    reads them.
    """

    __slots__ = ("_clf", "_feat_names", "_feat_index", "_meta")

    def __init__(
        self,
        classifier: Any,
        feat_names: list[str],
        meta: Optional[dict] = None,
    ) -> None:
        if not feat_names:
            raise PLRLoadError("feat_names empty — refusing to construct fuser")
        self._clf = classifier
        self._feat_names = list(feat_names)
        self._feat_index = {name: i for i, name in enumerate(feat_names)}
        self._meta = dict(meta or {})

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path, expected_sha256: Optional[str] = None) -> "StackedPLRFuser":
        """Load from a joblib artifact produced by `scripts/pwpc/sprint3.py`.

        If `expected_sha256` is given (or a sidecar .sha256 file exists next to
        the artifact), verify it matches before loading.

        Raises `PLRLoadError` when the artifact is missing, malformed,
        schema-mismatched, or hash-mismatched. The caller is expected to
        catch this and leave `plr_confidence` off the packet.
        """
        p = Path(path)
        if not p.is_file():
            raise PLRLoadError(f"model artifact not found: {p}")

        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        sidecar = p.with_suffix(p.suffix + ".sha256")
        if expected_sha256 is None and sidecar.is_file():
            # Accept the first token of the sidecar (standard sha256sum layout).
            expected_sha256 = sidecar.read_text(encoding="utf-8").strip().split()[0]
        if expected_sha256 and expected_sha256 != digest:
            raise PLRLoadError(
                f"sha256 mismatch for {p.name}: expected {expected_sha256}, "
                f"got {digest}"
            )

        try:
            payload = joblib.load(p)
        except Exception as exc:
            raise PLRLoadError(f"joblib.load failed for {p}: {exc}") from exc

        if not isinstance(payload, dict):
            raise PLRLoadError(f"artifact {p} is not a dict payload")
        schema = payload.get("schema_version")
        if schema != EXPECTED_SCHEMA_VERSION:
            raise PLRLoadError(
                f"schema_version mismatch: artifact={schema}, expected={EXPECTED_SCHEMA_VERSION}"
            )
        clf = payload.get("classifier")
        feat_names = payload.get("feat_names")
        if clf is None or not feat_names:
            raise PLRLoadError(f"artifact {p} missing classifier or feat_names")
        meta = {
            k: payload.get(k) for k in (
                "label_set", "cos_threshold", "auc_mean", "auc_std",
                "n_A", "n_B", "source_export", "trained_at",
            )
        }
        log.info(
            "StackedPLRFuser loaded from %s — label_set=%s AUC=%.3f±%.3f "
            "(n_A=%s n_B=%s, trained_at=%s)",
            p, meta["label_set"], meta.get("auc_mean") or 0.0,
            meta.get("auc_std") or 0.0, meta["n_A"], meta["n_B"], meta["trained_at"],
        )
        return cls(clf, feat_names, meta=meta)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def meta(self) -> dict:
        """Read-only copy of the artifact's metadata block."""
        return dict(self._meta)

    @property
    def feat_names(self) -> list[str]:
        return list(self._feat_names)

    # ── Scoring ───────────────────────────────────────────────────────

    def _assemble_vector(
        self,
        tier_totals: dict[str, float] | None,
        window_features: dict[str, float] | None,
        cos_qc: Optional[float],
    ) -> np.ndarray:
        """Build a single feature vector matching the artifact's layout.

        Missing features fill to 0 — matches training where tiers that didn't
        fire were left at 0, and the aggregate over genes sums to 0 when no
        gene contributed a score for that tier.
        """
        x = np.zeros(len(self._feat_names), dtype=float)
        tt = tier_totals or {}
        wf = window_features or {}
        for k, v in tt.items():
            idx = self._feat_index.get(k)
            if idx is not None and v is not None:
                try:
                    x[idx] = float(v)
                except (TypeError, ValueError):
                    pass
        for k, v in wf.items():
            idx = self._feat_index.get(k)
            if idx is not None and v is not None:
                try:
                    x[idx] = float(v)
                except (TypeError, ValueError):
                    pass
        cos_idx = self._feat_index.get("cos_q_c")
        if cos_idx is not None and cos_qc is not None:
            try:
                x[cos_idx] = float(cos_qc)
            except (TypeError, ValueError):
                pass
        return x.reshape(1, -1)

    def query_confidence(
        self,
        tier_totals: dict[str, float] | None,
        window_features: dict[str, float] | None = None,
        cos_qc: Optional[float] = None,
    ) -> dict[str, float]:
        """Score one query. Returns {'logit', 'prob_B', 'score_A'}.

        - `prob_B` — P(bucket=B | features) — predicted re-query probability
          under the trained artifact's labelling discipline (cos threshold +
          60s window). Clipped away from {0, 1} so the logit stays finite.
        - `logit` — log(prob_B / (1 - prob_B)) — the scale-free fused
          statistic from STATISTICAL_FUSION.md §C3. Higher = more likely to
          trigger a re-query = lower confidence in the current retrieval.
        - `score_A` — 1 - prob_B. Packet-friendly confidence (higher = better).
        """
        x = self._assemble_vector(tier_totals, window_features, cos_qc)
        try:
            p = float(self._clf.predict_proba(x)[0, 1])
        except Exception:
            log.warning("PLR predict_proba failed", exc_info=True)
            # Neutral: 0.5 → logit 0. Caller can detect via score_A == 0.5.
            return {"prob_B": 0.5, "logit": 0.0, "score_A": 0.5}
        p_clipped = max(min(p, 1.0 - 1e-3), 1e-3)
        return {
            "prob_B": p,
            "logit": math.log(p_clipped / (1.0 - p_clipped)),
            "score_A": 1.0 - p,
        }


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton (lazy-loaded from config)
# ──────────────────────────────────────────────────────────────────────

_fuser: Optional[StackedPLRFuser] = None
_load_attempted: bool = False
_load_error: Optional[str] = None


def get_fuser(model_path: str | Path, *, force_reload: bool = False) -> Optional[StackedPLRFuser]:
    """Return the process-wide PLR fuser, loading on first call.

    Returns None when loading fails — callers treat that as "PLR unavailable"
    and skip attaching `plr_confidence`. The failure is cached in
    `_load_error` so we don't thrash the disk on every request.
    """
    global _fuser, _load_attempted, _load_error
    if force_reload:
        _fuser = None
        _load_attempted = False
        _load_error = None
    if _load_attempted:
        return _fuser
    _load_attempted = True
    try:
        _fuser = StackedPLRFuser.load(model_path)
        _load_error = None
    except PLRLoadError as exc:
        _load_error = str(exc)
        log.warning("PLR fuser unavailable: %s", exc)
        _fuser = None
    return _fuser


def last_load_error() -> Optional[str]:
    """Diagnostic for /health and tests."""
    return _load_error
