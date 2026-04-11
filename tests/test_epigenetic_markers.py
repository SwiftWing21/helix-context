"""
Tests for EpigeneticMarkers — particularly the working-set inference
primitive added in Slice 1 of the 8D dimensional roadmap (Phase 1).

Slice 1 ships only the schema field + helper. The actual wiring (touch
path populating recent_accesses, density gate consuming access_rate) is
Slice 2 and lives in test_density_gate.py and test_genome.py respectively.

Reference: ~/.helix/shared/handoffs/2026-04-11_8d_dimensional_roadmap.md
"""

from __future__ import annotations

import time

import pytest

from helix_context.schemas import EpigeneticMarkers


# ── Field defaults / serialization ─────────────────────────────────────


def test_recent_accesses_defaults_to_empty_list():
    """A fresh EpigeneticMarkers has recent_accesses=[] (not None)."""
    epi = EpigeneticMarkers()
    assert epi.recent_accesses == []
    assert isinstance(epi.recent_accesses, list)


def test_recent_accesses_roundtrip_through_json():
    """JSON serialization preserves recent_accesses entries.

    EpigeneticMarkers is stored as a JSON blob in the genes table; if
    Pydantic loses the field on round-trip, every touch + reload would
    silently drop the working-set data. Guard explicitly.
    """
    now = time.time()
    epi = EpigeneticMarkers(recent_accesses=[now - 30, now - 15, now])
    blob = epi.model_dump_json()
    restored = EpigeneticMarkers.model_validate_json(blob)
    assert restored.recent_accesses == [now - 30, now - 15, now]


def test_existing_marker_blob_without_field_loads_with_empty_default():
    """Legacy genes that pre-date Slice 1 must still parse cleanly.

    The genome.db on disk has thousands of marker blobs that were
    written before recent_accesses existed. Pydantic must default the
    missing field to [] rather than raise on validation.
    """
    legacy_blob = (
        '{"created_at": 1000.0, "last_accessed": 1500.0, '
        '"access_count": 7, "co_activated_with": [], '
        '"typed_co_activated": [], "decay_score": 0.8}'
    )
    epi = EpigeneticMarkers.model_validate_json(legacy_blob)
    assert epi.access_count == 7
    assert epi.recent_accesses == []
    assert epi.access_rate() == 0.0  # empty buffer → zero rate, not error


# ── access_rate() helper ───────────────────────────────────────────────


def test_access_rate_zero_when_buffer_empty():
    """A gene with no recent_accesses entries reports rate=0.0."""
    epi = EpigeneticMarkers()
    assert epi.access_rate() == 0.0
    assert epi.access_rate(window_seconds=60) == 0.0
    assert epi.access_rate(window_seconds=86400) == 0.0


def test_access_rate_counts_only_entries_within_window():
    """Entries older than the window do not contribute to the rate.

    Distinguishes "hot recently" from "hot once a year ago" — the
    central reason for adding the windowed field over the existing
    monotonic access_count.
    """
    now = time.time()
    epi = EpigeneticMarkers(
        recent_accesses=[
            now - 7200,  # 2 hours ago — outside 1h window
            now - 5400,  # 1.5 hours ago — outside 1h window
            now - 1800,  # 30 minutes ago — inside 1h window
            now - 900,   # 15 minutes ago — inside 1h window
            now - 60,    # 1 minute ago — inside 1h window
        ]
    )
    rate_1h = epi.access_rate(window_seconds=3600)
    expected = 3 / 3600.0  # 3 entries in 1h window
    assert rate_1h == pytest.approx(expected, rel=1e-9)


def test_access_rate_window_zero_returns_zero_safely():
    """Zero/negative windows do not raise and return 0.0.

    Defensive: a caller computing a window from `now - last_seen`
    might pass a non-positive value; the helper should not throw
    a ZeroDivisionError on the production path.
    """
    now = time.time()
    epi = EpigeneticMarkers(recent_accesses=[now - 10, now - 5, now])
    assert epi.access_rate(window_seconds=0) == 0.0
    assert epi.access_rate(window_seconds=-100) == 0.0


def test_access_rate_full_buffer_with_wide_window():
    """All 100 buffer entries inside a sufficiently wide window."""
    now = time.time()
    # 100 timestamps spread evenly across the last 50 seconds
    timestamps = [now - 50 + (i * 0.5) for i in range(100)]
    epi = EpigeneticMarkers(recent_accesses=timestamps)
    rate = epi.access_rate(window_seconds=60)
    assert rate == pytest.approx(100 / 60.0, rel=1e-9)


def test_access_rate_distinguishes_recent_burst_from_old_history():
    """The point of the helper: gene A and gene B have identical
    access_count but very different rates — the rate signal sees what
    the monotonic counter cannot."""
    now = time.time()

    # Gene A: 10 accesses, all in the last minute
    gene_a = EpigeneticMarkers(
        access_count=10,
        recent_accesses=[now - 50 + i * 5 for i in range(10)],
    )

    # Gene B: 10 accesses, all from a year ago
    year_ago = now - 365 * 86400
    gene_b = EpigeneticMarkers(
        access_count=10,
        recent_accesses=[year_ago + i * 60 for i in range(10)],
    )

    # Same count, very different rates
    assert gene_a.access_count == gene_b.access_count
    assert gene_a.access_rate(window_seconds=3600) > 0
    assert gene_b.access_rate(window_seconds=3600) == 0


# ── Buffer-shape tolerance ─────────────────────────────────────────────


def test_access_rate_handles_oversized_buffer_gracefully():
    """The touch path (Slice 2) trims to 100 entries on append, but
    the helper itself does not enforce that bound — a marker blob
    deserialized from a corrupt or hand-crafted source with >100
    entries should still produce a correct rate, not crash."""
    now = time.time()
    big = [now - i for i in range(500)]  # 500 entries, one per second
    epi = EpigeneticMarkers(recent_accesses=big)
    # All 500 entries are within a 1000-second window
    rate = epi.access_rate(window_seconds=1000)
    assert rate == pytest.approx(500 / 1000.0, rel=1e-9)


def test_access_rate_handles_unsorted_buffer():
    """recent_accesses is appended in chronological order by the touch
    path, but `access_rate` does not assume sorting — it just counts
    entries above the cutoff. Verify that a reordered buffer still
    produces the same count."""
    now = time.time()
    sorted_buf = [now - 100, now - 50, now - 10]
    shuffled = [now - 10, now - 100, now - 50]
    sorted_epi = EpigeneticMarkers(recent_accesses=sorted_buf)
    shuffled_epi = EpigeneticMarkers(recent_accesses=shuffled)
    assert sorted_epi.access_rate(window_seconds=200) == shuffled_epi.access_rate(
        window_seconds=200
    )
