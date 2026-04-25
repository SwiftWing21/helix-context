"""Tests for the Stacked PLR query-confidence head.

These tests exercise StackedPLRFuser's load/score contracts without touching
the real trained artifact — they synthesize a tiny GradientBoostingClassifier
on the fly so CI doesn't depend on the 2026-04-15 export's training run.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier

from helix_context import fusion_plr
from helix_context.fusion_plr import (
    EXPECTED_SCHEMA_VERSION,
    PLRLoadError,
    StackedPLRFuser,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

FEAT_NAMES = [
    "fts5", "splade", "sema_boost", "lex_anchor",
    "tag_exact", "tag_prefix", "pki", "harmonic", "sr",
    "fts5__splade", "splade__sema_boost",  # pretend window correlations
    "cos_q_c",
]


def _toy_classifier(seed: int = 0) -> GradientBoostingClassifier:
    """Fit a shallow classifier on a separable 2D toy — we only need
    `predict_proba` to return sensible values."""
    rng = np.random.default_rng(seed)
    n = 50
    X_zero = rng.normal(loc=0.0, scale=0.5, size=(n, len(FEAT_NAMES)))
    X_one = rng.normal(loc=1.5, scale=0.5, size=(n, len(FEAT_NAMES)))
    X = np.vstack([X_zero, X_one])
    y = np.array([0] * n + [1] * n)
    clf = GradientBoostingClassifier(
        max_depth=2, n_estimators=20, random_state=seed,
    )
    clf.fit(X, y)
    return clf


@pytest.fixture
def trained_artifact(tmp_path: Path) -> Path:
    clf = _toy_classifier()
    payload = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "feat_names": list(FEAT_NAMES),
        "classifier": clf,
        "label_set": "test",
        "cos_threshold": 0.7,
        "auc_mean": 0.99,
        "auc_std": 0.01,
        "n_A": 50,
        "n_B": 50,
        "source_export": "synthetic.json",
        "trained_at": "2026-04-21T00:00:00+00:00",
    }
    out = tmp_path / "plr.joblib"
    joblib.dump(payload, out)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (out.with_suffix(out.suffix + ".sha256")).write_text(
        f"{digest}  {out.name}\n", encoding="utf-8",
    )
    return out


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Isolate the module-level fuser between tests."""
    fusion_plr._fuser = None
    fusion_plr._load_attempted = False
    fusion_plr._load_error = None
    yield
    fusion_plr._fuser = None
    fusion_plr._load_attempted = False
    fusion_plr._load_error = None


# ──────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────

def test_load_succeeds_on_valid_artifact(trained_artifact: Path):
    fuser = StackedPLRFuser.load(trained_artifact)
    assert fuser.feat_names == FEAT_NAMES
    assert fuser.meta["label_set"] == "test"
    assert fuser.meta["n_A"] == 50


def test_load_raises_when_file_missing(tmp_path: Path):
    with pytest.raises(PLRLoadError, match="not found"):
        StackedPLRFuser.load(tmp_path / "nope.joblib")


def test_load_raises_on_schema_mismatch(tmp_path: Path):
    """Bump the payload's schema_version so load should reject."""
    clf = _toy_classifier()
    path = tmp_path / "plr.joblib"
    joblib.dump({
        "schema_version": EXPECTED_SCHEMA_VERSION + 999,
        "feat_names": list(FEAT_NAMES),
        "classifier": clf,
    }, path)
    with pytest.raises(PLRLoadError, match="schema_version mismatch"):
        StackedPLRFuser.load(path)


def test_load_raises_on_sha256_mismatch(trained_artifact: Path):
    # Corrupt the sidecar so load refuses
    (trained_artifact.with_suffix(trained_artifact.suffix + ".sha256")).write_text(
        "0000000000000000000000000000000000000000000000000000000000000000  x\n",
        encoding="utf-8",
    )
    with pytest.raises(PLRLoadError, match="sha256 mismatch"):
        StackedPLRFuser.load(trained_artifact)


def test_load_accepts_explicit_sha256(trained_artifact: Path):
    digest = hashlib.sha256(trained_artifact.read_bytes()).hexdigest()
    fuser = StackedPLRFuser.load(trained_artifact, expected_sha256=digest)
    assert fuser is not None


def test_load_raises_on_bad_payload_shape(tmp_path: Path):
    path = tmp_path / "plr.joblib"
    joblib.dump("not a dict", path)
    with pytest.raises(PLRLoadError, match="not a dict"):
        StackedPLRFuser.load(path)


# ──────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────

def test_query_confidence_returns_expected_keys(trained_artifact: Path):
    fuser = StackedPLRFuser.load(trained_artifact)
    out = fuser.query_confidence(
        tier_totals={"fts5": 1.0, "splade": 0.5},
        window_features={"fts5__splade": 0.3},
        cos_qc=0.6,
    )
    assert set(out.keys()) == {"prob_B", "logit", "score_A"}
    assert 0.0 < out["prob_B"] < 1.0
    assert out["score_A"] == pytest.approx(1.0 - out["prob_B"])


def test_query_confidence_logit_matches_prob(trained_artifact: Path):
    """Clipped logit should be log(p/(1-p)) for the clipped prob."""
    import math

    fuser = StackedPLRFuser.load(trained_artifact)
    out = fuser.query_confidence(
        tier_totals={"fts5": 2.0, "splade": 2.0, "lex_anchor": 2.0},
        cos_qc=0.9,
    )
    p = out["prob_B"]
    p_clip = max(min(p, 1 - 1e-3), 1e-3)
    assert out["logit"] == pytest.approx(math.log(p_clip / (1 - p_clip)))


def test_query_confidence_tolerates_missing_features(trained_artifact: Path):
    """Unknown feature keys must not blow up the assembler."""
    fuser = StackedPLRFuser.load(trained_artifact)
    out = fuser.query_confidence(
        tier_totals={"fts5": 1.0, "made_up_tier": 42.0},
        window_features={"unknown__pair": 0.5},
        cos_qc=None,
    )
    assert 0.0 < out["prob_B"] < 1.0


def test_query_confidence_none_inputs_yield_zero_vector(trained_artifact: Path):
    """All-None inputs should still score (zero-vector, not crash)."""
    fuser = StackedPLRFuser.load(trained_artifact)
    out = fuser.query_confidence(None, None, None)
    assert 0.0 <= out["prob_B"] <= 1.0


def test_query_confidence_distinguishes_cold_from_hot(trained_artifact: Path):
    """Toy classifier was trained with B clustered at loc=1.5 — a vector
    with big tier_totals should score higher prob_B than all zeros."""
    fuser = StackedPLRFuser.load(trained_artifact)
    cold = fuser.query_confidence(
        tier_totals={k: 0.0 for k in FEAT_NAMES[:9]},
        window_features={k: 0.0 for k in FEAT_NAMES[9:11]},
        cos_qc=0.0,
    )
    hot = fuser.query_confidence(
        tier_totals={k: 2.0 for k in FEAT_NAMES[:9]},
        window_features={k: 2.0 for k in FEAT_NAMES[9:11]},
        cos_qc=2.0,
    )
    assert hot["prob_B"] > cold["prob_B"]


# ──────────────────────────────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────────────────────────────

def test_get_fuser_caches_result(trained_artifact: Path):
    first = fusion_plr.get_fuser(trained_artifact)
    second = fusion_plr.get_fuser(trained_artifact)
    assert first is second


def test_get_fuser_returns_none_on_missing_file(tmp_path: Path):
    assert fusion_plr.get_fuser(tmp_path / "missing.joblib") is None
    # Repeat call must also return None, and must not re-attempt load
    assert fusion_plr.get_fuser(tmp_path / "missing.joblib") is None
    assert fusion_plr.last_load_error() is not None
    assert "not found" in fusion_plr.last_load_error()


def test_get_fuser_force_reload_clears_error(tmp_path: Path, trained_artifact: Path):
    # First call misses; second (force_reload) succeeds on a real path.
    assert fusion_plr.get_fuser(tmp_path / "missing.joblib") is None
    assert fusion_plr.last_load_error() is not None
    fuser = fusion_plr.get_fuser(trained_artifact, force_reload=True)
    assert fuser is not None
    assert fusion_plr.last_load_error() is None
