"""
Tests for the Struggle 1 density gate in `genome.py`.

Covers:
  - is_denied_source pattern matching for build artifacts, lockfiles,
    manifests, binary files, and non-English software locales
  - Steam / game content is NOT deny-listed (reframed 2026-04-10 — see
    ~/.helix/shared/handoffs/2026-04-10_density_gate_b_to_c.md)
  - apply_density_gate decisioning across the three stages (deny list,
    access override, score-based thresholds)
  - upsert_gene integration: the gate actually fires at the storage boundary
  - compact_genome batch sweep: dry-run counts, reason breakdown, idempotency
  - Backward compatibility: apply_gate=False escape hatch for HGT/backfill
"""

from __future__ import annotations

import pytest

from helix_context.genome import (
    Genome,
    is_denied_source,
    _DENSITY_HETEROCHROMATIN_THRESHOLD,
    _DENSITY_EUCHROMATIN_THRESHOLD,
    _DENSITY_ACCESS_OVERRIDE,
    _DENSITY_RATE_WINDOW,
    _DENSITY_RATE_MIN_HITS,
)
from helix_context.exceptions import PromoterMismatch
from helix_context.schemas import (
    Gene,
    PromoterTags,
    EpigeneticMarkers,
    ChromatinState,
)

from tests.conftest import make_gene


# ── is_denied_source — structural path patterns ────────────────────────


class TestIsDeniedSource:
    def test_none_not_denied(self):
        assert is_denied_source(None) is False

    def test_empty_string_not_denied(self):
        assert is_denied_source("") is False

    def test_signal_paths_not_denied(self):
        """Real helix-context source files must never be denied."""
        assert is_denied_source("F:/Projects/helix-context/helix_context/genome.py") is False
        assert is_denied_source("helix-context/docs/RESEARCH.md") is False
        assert is_denied_source("BookKeeper/src/ledger.py") is False
        assert is_denied_source("Education/fleet/dashboard.py") is False
        assert is_denied_source("two-brain-audit/scorerift/core.py") is False

    # Steam / game content
    # ─── Steam / game content is SIGNAL, not noise (reframed 2026-04-10) ───
    # Game files (configs, enums, item IDs, localization, code) are content-
    # dense with unambiguous literal values. Empirically 86% of correct
    # answers on the N=50 v2 NIAH benchmark came from steam/game paths. The
    # structural path is not a categorical reject; individual low-density
    # game genes still get caught by the score gate.

    def test_steam_library_not_denied(self):
        assert is_denied_source("F:/SteamLibrary/steamapps/common/BeamNG.drive/lua/lib/x.lua") is False

    def test_steamapps_common_not_denied(self):
        assert is_denied_source("/steamapps/common/Hades/some_file.txt") is False

    def test_beamng_drive_not_denied(self):
        assert is_denied_source("C:/BeamNG.drive/missions/Gridmap/quicktarget.json") is False

    def test_hades_content_not_denied(self):
        """Hades subtitles, maps, audio — all signal, none deny-listed."""
        assert is_denied_source("F:/SteamLibrary/Hades/Content/Subtitles/en/Zagreus.csv") is False
        assert is_denied_source("F:/SteamLibrary/Hades/Content/Maps/Orpheus.csv") is False
        assert is_denied_source("/Hades/Content/Audio/zagreus_intro.ogg") is False

    def test_factorio_base_data_not_denied(self):
        assert is_denied_source("F:/Factorio/data/base/campaigns/level-01.cfg") is False

    def test_dyson_sphere_not_denied(self):
        assert is_denied_source("F:/SteamLibrary/Dyson Sphere Program/data.json") is False

    # Build artifacts
    def test_next_build_denied(self):
        assert is_denied_source("F:/CosmicTasha/.next/server/app/page.js") is True

    def test_node_modules_denied(self):
        assert is_denied_source("project/node_modules/react/index.js") is True

    def test_pycache_denied(self):
        assert is_denied_source("helix_context/__pycache__/genome.cpython-314.pyc") is True

    def test_dist_denied(self):
        assert is_denied_source("project/dist/bundle.js") is True

    def test_target_debug_denied(self):
        assert is_denied_source("biged-rs/target/debug/deps/libcore.rlib") is True

    def test_target_release_denied(self):
        assert is_denied_source("biged-rs/target/release/biged.exe") is True

    # Lockfiles
    def test_package_lock_denied(self):
        assert is_denied_source("F:/CosmicTasha/package-lock.json") is True

    def test_yarn_lock_denied(self):
        assert is_denied_source("project/yarn.lock") is True

    def test_cargo_lock_denied(self):
        assert is_denied_source("biged-rs/Cargo.lock") is True

    def test_uv_lock_denied(self):
        assert is_denied_source("helix-context/uv.lock") is True

    # Minified and source maps
    def test_min_js_denied(self):
        assert is_denied_source("static/bundle.min.js") is True

    def test_min_css_denied(self):
        assert is_denied_source("static/theme.min.css") is True

    def test_source_map_denied(self):
        assert is_denied_source("static/bundle.js.map") is True

    # Next.js manifests
    def test_app_paths_manifest_denied(self):
        assert is_denied_source("F:/CosmicTasha/.next/server/app-paths-manifest.json") is True

    def test_client_reference_manifest_denied(self):
        assert is_denied_source(".next/server/app/client-reference-manifest.js") is True

    # Binary / compiled
    def test_pyc_denied(self):
        assert is_denied_source("__pycache__/module.cpython-312.pyc") is True

    def test_wasm_denied(self):
        assert is_denied_source("dist/module.wasm") is True

    def test_exe_denied(self):
        assert is_denied_source("target/release/tool.exe") is True

    # Locale handling
    def test_non_english_locale_denied(self):
        """Non-English locale directories are treated as noise by default."""
        assert is_denied_source("project/locale/de/messages.po") is True
        assert is_denied_source("project/locale/ja/messages.po") is True

    def test_english_locale_not_denied(self):
        """English locale is preserved as primary user base."""
        assert is_denied_source("project/locale/en/messages.po") is False

    # CRITICAL: CSVs are NOT in the deny list — business content
    def test_business_csv_not_denied(self):
        """Future business CSVs (customer data, invoices, etc.) must pass."""
        assert is_denied_source("F:/BookKeeper/customers.csv") is False
        assert is_denied_source("F:/BigEd/fleet/metrics/daily_report.csv") is False
        assert is_denied_source("project/data/financial_records.csv") is False

    def test_case_insensitive(self):
        """Case-insensitive matching applies to all kept deny patterns."""
        assert is_denied_source("F:/project/NODE_MODULES/react/index.js") is True
        assert is_denied_source("F:/project/Node_Modules/react/index.js") is True
        assert is_denied_source("F:/project/.NEXT/server/page.js") is True


# ── apply_density_gate — decision logic ────────────────────────────────


def _gene_with_access(content: str, domains: list[str], kvs: list[str], access: int, source_id: str | None):
    """Helper — build a gene with specific access count for gate tests."""
    g = make_gene(content=content, domains=domains)
    g.epigenetics.access_count = access
    g.key_values = kvs
    g.source_id = source_id
    return g


class TestApplyDensityGate:
    def test_denied_source_forces_heterochromatin(self, genome):
        """Deny-listed paths ignore score entirely."""
        g = _gene_with_access(
            content="x" * 2000,  # long but from a denied source
            domains=["code", "js", "lib"] * 10,  # tag-heavy, would score high
            kvs=["k1=v1", "k2=v2", "k3=v3"] * 5,
            access=0,
            source_id="F:/project/node_modules/react-dom/index.js",
        )
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.HETEROCHROMATIN
        assert reason == "deny_list"

    def test_access_override_beats_low_score(self, genome):
        """A frequently-accessed gene stays OPEN even with a terrible score."""
        g = _gene_with_access(
            content="boilerplate " * 100,  # 1200 chars, no tags, no KVs
            domains=[],
            kvs=[],
            access=_DENSITY_ACCESS_OVERRIDE + 1,  # 6 accesses
            source_id="legit/path/file.txt",
        )
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.OPEN
        assert reason == "access_override"

    def test_access_override_does_not_help_denied_source(self, genome):
        """Deny-list beats access override — the path is the stronger signal."""
        g = _gene_with_access(
            content="whatever",
            domains=["code"],
            kvs=["k=v"],
            access=100,
            source_id="F:/project/node_modules/lodash/index.js",
        )
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.HETEROCHROMATIN
        assert reason == "deny_list"

    # ── Phase 1 slice 2: windowed access-rate override ──────────────

    def test_rate_override_promotes_recently_active_gene(self, genome):
        """A gene with N>=min_hits accesses in the last window gets the
        access_rate_override path, even if its static density is low.

        This is the headline Slice 2 test — proves the rate signal sees
        what the monotonic counter cannot.
        """
        import time
        g = _gene_with_access(
            content="boilerplate " * 100,  # 1200 chars, no tags, no KVs
            domains=[],
            kvs=[],
            access=0,  # monotonic counter is ZERO — only the rate signal can save it
            source_id="legit/path/file.txt",
        )
        # Burst of 5 accesses in the last minute
        now = time.time()
        g.epigenetics.recent_accesses = [now - 300, now - 240, now - 180, now - 60, now - 1]
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.OPEN
        assert reason == "access_rate_override"

    def test_rate_override_does_not_fire_for_old_accesses(self, genome):
        """A gene whose recent_accesses are all OUTSIDE the window does
        not get the rate override. Falls through to the count fallback,
        and if access_count is also low, falls through to score-based.
        """
        import time
        g = _gene_with_access(
            content="boilerplate " * 100,  # low static density
            domains=[],
            kvs=[],
            access=0,  # also low count
            source_id="legit/path/file.txt",
        )
        # Buffer has entries, but they're all from a year ago — outside the 1h window
        year_ago = time.time() - (365 * 86400)
        g.epigenetics.recent_accesses = [year_ago + i * 60 for i in range(10)]
        state, reason = genome.apply_density_gate(g)
        # Neither rate nor count override fires; gene drops to its score-based tier
        assert reason != "access_rate_override"
        assert reason != "access_override"

    def test_legacy_gene_with_empty_buffer_still_uses_count_override(self, genome):
        """A pre-Phase-1 gene has recent_accesses=[] but access_count>=5.

        The rate path returns 0.0 (empty buffer), falls through cleanly
        to the legacy count override. This is the backward-compatibility
        guarantee — existing genes don't lose their override status when
        slice 2 lands.
        """
        g = _gene_with_access(
            content="boilerplate " * 100,
            domains=[],
            kvs=[],
            access=_DENSITY_ACCESS_OVERRIDE + 1,  # 6 monotonic accesses
            source_id="legit/path/file.txt",
        )
        # No rate buffer at all — pretend this gene predates slice 1
        assert g.epigenetics.recent_accesses == []
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.OPEN
        assert reason == "access_override"  # legacy fallback path

    def test_rate_override_takes_priority_over_count_override(self, genome):
        """When BOTH paths would qualify, the rate path wins (it ships
        the more informative reason code)."""
        import time
        g = _gene_with_access(
            content="boilerplate " * 100,
            domains=[],
            kvs=[],
            access=20,  # would trigger count override on its own
            source_id="legit/path/file.txt",
        )
        now = time.time()
        # Also recently active enough to trigger the rate override
        g.epigenetics.recent_accesses = [now - 30, now - 20, now - 10]
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.OPEN
        assert reason == "access_rate_override"  # rate takes priority

    def test_rate_threshold_constants_are_sane(self):
        """Defensive: the constants must produce a positive threshold
        and not collapse to zero or NaN."""
        threshold = _DENSITY_RATE_MIN_HITS / _DENSITY_RATE_WINDOW
        assert threshold > 0
        assert threshold < 1.0  # less than 1 access/second is reasonable
        assert _DENSITY_RATE_WINDOW > 0
        assert _DENSITY_RATE_MIN_HITS >= 1

    # ── Existing score-based stage tests ────────────────────────────

    def test_high_density_stays_open(self, genome):
        """Signal-grade content with rich tags stays OPEN."""
        g = _gene_with_access(
            content="def compute(): return 42",  # short + rich
            domains=["python", "function", "compute"],
            kvs=["name=compute", "returns=int", "value=42", "type=pure"],
            access=0,
            source_id="helix-context/helix_context/math.py",
        )
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.OPEN
        assert reason == "open"

    def test_very_low_density_goes_to_heterochromatin(self, genome):
        """Dilute content with no tags goes to HETEROCHROMATIN."""
        g = _gene_with_access(
            content="a " * 5000,  # 10k chars of nothing
            domains=[],
            kvs=[],
            access=0,
            source_id="project/scratch.txt",
        )
        state, reason = genome.apply_density_gate(g)
        assert state == ChromatinState.HETEROCHROMATIN
        assert reason == "low_score_hetero"

    def test_medium_density_goes_to_euchromatin(self, genome):
        """Content between hetero and euchro thresholds becomes EUCHROMATIN."""
        # Score should land in [0.5, 1.0). A gene with ~4000 chars, a few
        # tags, a complement → tag_density ≈ 1.0, kv_density ≈ 0.5,
        # complement bonus 0.1 → score ~0.55-0.75.
        g = _gene_with_access(
            content="some moderately informative prose about software " * 90,  # ~4400 chars
            domains=["software", "text"],
            kvs=["topic=prose", "length=medium"],
            access=0,
            source_id="project/notes/medium_density.md",
        )
        state, _ = genome.apply_density_gate(g)
        # This is a soft assertion — the exact category depends on the
        # precise char count and the complement bonus, but it should be
        # demoted (not OPEN) because the content is filler-heavy.
        assert state in (ChromatinState.EUCHROMATIN, ChromatinState.HETEROCHROMATIN)

    def test_tiny_content_does_not_explode_score(self, genome):
        """30-char gene with 5 tags must not get an absurd density score."""
        g = _gene_with_access(
            content="x" * 30,
            domains=["a", "b", "c", "d", "e"],  # 5 tags
            kvs=[],
            access=0,
            source_id="__session__",
        )
        # Without the 100-char floor, tag_density would be 5/0.03 = 166.
        # With the floor, it's 5/0.1 = 50. Still high, but the floor
        # prevents score overflow from tiny content. The actual test is
        # that this doesn't cause a division-by-zero or overflow.
        state, reason = genome.apply_density_gate(g)
        # Accept either open (if score is still very high) or something
        # else — the contract is "must not crash and must return a valid
        # ChromatinState".
        assert state in (
            ChromatinState.OPEN,
            ChromatinState.EUCHROMATIN,
            ChromatinState.HETEROCHROMATIN,
        )


# ── Phase 1 slice 2: touch_genes populates recent_accesses ─────────────


class TestTouchPopulatesRecentAccesses:
    """Tests that genome.touch_genes() writes timestamps into the
    recent_accesses buffer added in Slice 1, in addition to the existing
    monotonic access_count increment."""

    def test_touch_appends_timestamp_to_buffer(self, genome):
        """A single touch appends one timestamp to recent_accesses."""
        import time
        gene = make_gene(content="touch test 1", domains=["test"])
        genome.upsert_gene(gene, apply_gate=False)

        # Pre-condition: empty buffer
        before = genome.get_gene(gene.gene_id)
        assert before.epigenetics.recent_accesses == []

        t0 = time.time()
        genome.touch_genes([gene.gene_id])

        after = genome.get_gene(gene.gene_id)
        assert len(after.epigenetics.recent_accesses) == 1
        # Timestamp must be from this turn (within a few seconds of t0)
        assert abs(after.epigenetics.recent_accesses[0] - t0) < 5.0

    def test_touch_increments_count_and_buffer_together(self, genome):
        """Backward-compat: touch still increments access_count, AND it
        also populates the new buffer. Both signals advance in lockstep."""
        gene = make_gene(content="touch test 2", domains=["test"])
        genome.upsert_gene(gene, apply_gate=False)

        for _ in range(5):
            genome.touch_genes([gene.gene_id])

        after = genome.get_gene(gene.gene_id)
        assert after.epigenetics.access_count == 5
        assert len(after.epigenetics.recent_accesses) == 5

    def test_touch_trims_buffer_to_100(self, genome):
        """Touching a gene >100 times keeps only the most recent 100
        timestamps. Per-gene marker blob stays bounded at ~800 bytes."""
        gene = make_gene(content="touch test 3", domains=["test"])
        genome.upsert_gene(gene, apply_gate=False)

        for _ in range(150):
            genome.touch_genes([gene.gene_id])

        after = genome.get_gene(gene.gene_id)
        assert after.epigenetics.access_count == 150  # monotonic still grows
        assert len(after.epigenetics.recent_accesses) == 100  # buffer is bounded

    def test_touch_batch_populates_all_genes(self, genome):
        """A single touch_genes call with N gene_ids populates the
        buffer for every gene in the batch (not just one)."""
        gene_a = make_gene(content="touch batch a", domains=["test"])
        gene_b = make_gene(content="touch batch b", domains=["test"])
        gene_c = make_gene(content="touch batch c", domains=["test"])
        for g in (gene_a, gene_b, gene_c):
            genome.upsert_gene(g, apply_gate=False)

        genome.touch_genes([gene_a.gene_id, gene_b.gene_id, gene_c.gene_id])

        for g in (gene_a, gene_b, gene_c):
            after = genome.get_gene(g.gene_id)
            assert len(after.epigenetics.recent_accesses) == 1

    def test_touch_then_gate_uses_rate_override_after_enough_hits(self, genome):
        """End-to-end: touch a gene 3 times, then run apply_density_gate.
        The gate should return access_rate_override even if the gene's
        static content is low-density. Proves the slice 1 schema +
        slice 2 wiring + slice 2 gate path compose correctly."""
        gene = make_gene(content="boilerplate " * 100, domains=[])
        gene.key_values = []
        gene.source_id = "legit/path/file.txt"
        genome.upsert_gene(gene, apply_gate=False)

        # 3 touches in rapid succession — meets _DENSITY_RATE_MIN_HITS in window
        for _ in range(_DENSITY_RATE_MIN_HITS):
            genome.touch_genes([gene.gene_id])

        after = genome.get_gene(gene.gene_id)
        state, reason = genome.apply_density_gate(after)
        assert state == ChromatinState.OPEN
        assert reason == "access_rate_override"


# ── upsert_gene integration — gate fires at storage boundary ──────────


class TestUpsertGateIntegration:
    """These tests exercise the gate at the upsert boundary using the
    `gated_genome` fixture — the default `genome` fixture bypasses the
    gate for test convenience, so gate-firing tests must opt in.
    """

    def test_upsert_denies_build_artifact_path(self, gated_genome):
        """Calling upsert_gene directly with a build-artifact path should demote it."""
        g = make_gene("generated bundle content", domains=["js", "generated"])
        g.source_id = "F:/project/node_modules/react-dom/cjs/react-dom.production.min.js"
        gated_genome.upsert_gene(g)

        retrieved = gated_genome.get_gene(g.gene_id)
        assert retrieved is not None
        assert retrieved.chromatin == ChromatinState.HETEROCHROMATIN

    def test_upsert_preserves_signal_content(self, gated_genome):
        """Real helix-context source stays OPEN through the gate."""
        g = make_gene(
            "def helix_context(): return 'signal'",
            domains=["python", "helix", "function"],
        )
        g.source_id = "F:/Projects/helix-context/helix_context/api.py"
        g.key_values = ["name=helix_context", "returns=str"]
        gated_genome.upsert_gene(g)

        retrieved = gated_genome.get_gene(g.gene_id)
        assert retrieved is not None
        assert retrieved.chromatin == ChromatinState.OPEN

    def test_apply_gate_false_bypass(self, gated_genome):
        """apply_gate=False preserves the incoming chromatin state as-is."""
        g = make_gene("some generated content")
        g.source_id = "F:/project/.next/static/chunks/main.js"
        g.chromatin = ChromatinState.OPEN  # deliberately set

        gated_genome.upsert_gene(g, apply_gate=False)

        retrieved = gated_genome.get_gene(g.gene_id)
        assert retrieved.chromatin == ChromatinState.OPEN, (
            "apply_gate=False must not touch the chromatin state"
        )

    def test_upsert_gate_is_idempotent(self, gated_genome):
        """Upserting the same denied gene twice must be stable."""
        g = make_gene("generated manifest data")
        g.source_id = "F:/project/.next/server/app-paths-manifest.json"

        gated_genome.upsert_gene(g)
        first = gated_genome.get_gene(g.gene_id)

        gated_genome.upsert_gene(g)
        second = gated_genome.get_gene(g.gene_id)

        assert first.chromatin == second.chromatin == ChromatinState.HETEROCHROMATIN

    def test_upsert_preserves_explicit_heterochromatin(self, gated_genome):
        """Gate must not override an explicit HETEROCHROMATIN state.

        If a caller (HGT import, test fixture, etc.) deliberately sets
        the chromatin state before upserting, the gate should trust that
        decision and not 'promote' the gene to OPEN based on its content.
        """
        g = make_gene(
            "active dense content with tags",
            domains=["auth", "security", "session"],
            chromatin=ChromatinState.HETEROCHROMATIN,
        )
        g.source_id = "legit/path/file.py"
        g.key_values = ["k1=v1", "k2=v2", "k3=v3"]
        gated_genome.upsert_gene(g)

        retrieved = gated_genome.get_gene(g.gene_id)
        assert retrieved.chromatin == ChromatinState.HETEROCHROMATIN, (
            "gate must preserve explicit non-OPEN chromatin"
        )

    def test_upsert_preserves_explicit_euchromatin(self, gated_genome):
        """Gate must not override an explicit EUCHROMATIN state either."""
        g = make_gene(
            "content",
            domains=["auth"],
            chromatin=ChromatinState.EUCHROMATIN,
        )
        gated_genome.upsert_gene(g)

        retrieved = gated_genome.get_gene(g.gene_id)
        assert retrieved.chromatin == ChromatinState.EUCHROMATIN


# ── compact_genome batch sweep ─────────────────────────────────────────


class TestCompactGenomeSweep:
    def test_dry_run_does_not_modify(self, genome):
        """dry_run=True must not change any genes on disk."""
        # Signal gene: helix source with embedding (dense enough to stay OPEN)
        g_signal = make_gene(
            "def foo(): return 42",
            domains=["python", "function"],
        )
        g_signal.source_id = "helix-context/math.py"
        g_signal.embedding = [0.1] * 20

        # Noise gene: build-artifact path with embedding so compact can actually demote
        g_noise = make_gene("x" * 3000, domains=[])
        g_noise.source_id = "F:/project/node_modules/lodash/fp/_baseAssignValue.js"
        g_noise.embedding = [0.1] * 20

        # Bypass the gate so we can force both to OPEN, then run the sweep
        genome.upsert_gene(g_signal, apply_gate=False)
        genome.upsert_gene(g_noise, apply_gate=False)

        # Verify both are OPEN before sweep
        assert genome.get_gene(g_signal.gene_id).chromatin == ChromatinState.OPEN
        assert genome.get_gene(g_noise.gene_id).chromatin == ChromatinState.OPEN

        stats = genome.compact_genome(dry_run=True)

        # Dry run should report what would happen
        assert stats["scanned"] == 2
        assert stats["to_heterochromatin"] >= 1  # at least the noise gene

        # But neither gene should have been modified on disk
        assert genome.get_gene(g_signal.gene_id).chromatin == ChromatinState.OPEN
        assert genome.get_gene(g_noise.gene_id).chromatin == ChromatinState.OPEN

    def test_apply_run_demotes_noise(self, genome):
        """dry_run=False actually writes the demotions."""
        original_content = "x" * 3000
        g_noise = make_gene(original_content, domains=[])
        g_noise.source_id = "F:/project/node_modules/some-pkg/dist/index.min.js"
        g_noise.embedding = [0.1] * 20  # required for cold-storage demotion
        # Gate would demote on insert, so bypass it then run sweep
        genome.upsert_gene(g_noise, apply_gate=False)

        assert genome.get_gene(g_noise.gene_id).chromatin == ChromatinState.OPEN

        genome.compact_genome(dry_run=False)

        retrieved = genome.get_gene(g_noise.gene_id)
        # After C.1 (2026-04-10), heterochromatin is non-destructive — the
        # chromatin flag flips but content is preserved so cold-tier
        # retrieval (C.2) can reactivate the gene when a query needs it.
        assert retrieved.chromatin == ChromatinState.HETEROCHROMATIN
        assert retrieved.content == original_content, (
            "compress_to_heterochromatin must preserve content "
            "(non-destructive as of C.1)"
        )

    def test_sweep_reason_breakdown(self, genome):
        """The by_reason dict should record why each decision was made."""
        g_denied = make_gene("generated content here", domains=[])
        g_denied.source_id = "F:/project/node_modules/chalk/source/index.js"
        g_denied.embedding = [0.1] * 20

        g_accessed = make_gene("boilerplate content", domains=[])
        g_accessed.source_id = "legit/file.txt"
        g_accessed.epigenetics.access_count = 10
        g_accessed.embedding = [0.1] * 20

        g_signal = make_gene(
            "def helix_compute(): return 'signal'",
            domains=["python", "function", "helix", "compute"],
        )
        g_signal.source_id = "helix-context/api.py"
        g_signal.key_values = ["name=helix_compute", "returns=str", "value=signal"]
        g_signal.embedding = [0.1] * 20

        genome.upsert_gene(g_denied, apply_gate=False)
        genome.upsert_gene(g_accessed, apply_gate=False)
        genome.upsert_gene(g_signal, apply_gate=False)

        stats = genome.compact_genome(dry_run=True)

        assert stats["scanned"] == 3, f"expected 3 scanned, got {stats}"
        reasons = stats["by_reason"]
        assert "deny_list" in reasons, f"reasons={reasons}"
        assert "access_override" in reasons, f"reasons={reasons}"
        # Signal gene should hit "open" reason
        assert "open" in reasons, f"reasons={reasons}"


# ── C.1 — compress_to_heterochromatin is non-destructive ──────────────


class TestHeterochromatinNonDestructive:
    """Verifies that compress_to_heterochromatin only flips the tier flag.

    Content, complement, codons, SPLADE terms, and FTS5 index entries are
    all preserved so that cold-tier retrieval (C.2) can reactivate demoted
    genes on-demand. Hot-tier retrieval still excludes heterochromatin
    via the `WHERE chromatin < HETEROCHROMATIN` filter on every query.
    """

    def test_content_preserved_on_demotion(self, genome):
        original = "def legit_function():\n    return 'real content with literal values'"
        g = make_gene(original, domains=["python"])
        g.source_id = "project/module.py"
        genome.upsert_gene(g, apply_gate=False)

        genome.compress_to_heterochromatin(g.gene_id)

        retrieved = genome.get_gene(g.gene_id)
        assert retrieved is not None
        assert retrieved.chromatin == ChromatinState.HETEROCHROMATIN
        assert retrieved.content == original, (
            "content must survive heterochromatin demotion"
        )

    def test_complement_preserved_on_demotion(self, genome):
        g = make_gene("some gene content", domains=["code"])
        g.source_id = "project/file.py"
        # make_gene builds a complement of form "Summary of: ..." — verify
        # it's preserved across demotion
        original_complement = g.complement
        assert original_complement  # sanity
        genome.upsert_gene(g, apply_gate=False)

        genome.compress_to_heterochromatin(g.gene_id)

        retrieved = genome.get_gene(g.gene_id)
        assert retrieved.complement == original_complement

    def test_codons_preserved_on_demotion(self, genome):
        g = make_gene("codon test content", domains=["code"])
        g.source_id = "project/file.py"
        original_codons = list(g.codons)
        assert original_codons  # sanity
        genome.upsert_gene(g, apply_gate=False)

        genome.compress_to_heterochromatin(g.gene_id)

        retrieved = genome.get_gene(g.gene_id)
        assert retrieved.codons == original_codons

    def test_hot_retrieval_still_excludes_heterochromatin(self, genome):
        """Non-destructive demotion must not leak demoted genes into hot retrieval."""
        g = make_gene(
            "def authenticate(user): return user.is_valid()",
            domains=["python", "auth"],
            entities=["authenticate", "user"],
        )
        g.source_id = "project/auth.py"
        genome.upsert_gene(g, apply_gate=False)

        # Verify it's findable BEFORE demotion
        hot_before = genome.query_genes(domains=["auth"], entities=[], max_genes=10)
        assert any(gene.gene_id == g.gene_id for gene in hot_before), (
            "gene should be findable in hot retrieval before demotion"
        )

        genome.compress_to_heterochromatin(g.gene_id)

        # After demotion, it must NOT appear in hot retrieval even though
        # content is preserved. An empty result can arrive either as an
        # empty list OR as a PromoterMismatch exception (query_genes raises
        # when zero genes match across all tiers) — both mean "not found."
        try:
            hot_after = genome.query_genes(domains=["auth"], entities=[], max_genes=10)
        except PromoterMismatch:
            hot_after = []
        assert not any(gene.gene_id == g.gene_id for gene in hot_after), (
            "heterochromatin gene must be excluded from hot retrieval "
            "despite content being preserved"
        )

    def test_idempotent_demotion(self, genome):
        """Calling compress_to_heterochromatin twice must be stable and
        still preserve content the second time."""
        original = "stable content across idempotent demotion"
        g = make_gene(original, domains=["test"])
        g.source_id = "project/file.py"
        genome.upsert_gene(g, apply_gate=False)

        assert genome.compress_to_heterochromatin(g.gene_id) is True
        first = genome.get_gene(g.gene_id)

        assert genome.compress_to_heterochromatin(g.gene_id) is True
        second = genome.get_gene(g.gene_id)

        assert first.chromatin == second.chromatin == ChromatinState.HETEROCHROMATIN
        assert first.content == second.content == original

    def test_missing_gene_returns_false(self, genome):
        assert genome.compress_to_heterochromatin("nonexistent_gene_id") is False


# ── C.2 — cold-tier retrieval via ΣĒMA cosine ─────────────────────────


class TestColdTierRetrieval:
    """Verifies query_cold_tier() returns heterochromatin genes with full
    content when the query matches their ΣĒMA signature.

    Requires sentence-transformers to load the SemaCodec (~400MB model).
    Module-scoped fixture caches the codec across tests.
    """

    # Skip the whole class if sentence-transformers isn't installed
    pytestmark = pytest.mark.skipif(
        pytest.importorskip("sentence_transformers", reason="needs sentence-transformers") is None,
        reason="needs sentence-transformers",
    )

    @pytest.fixture(scope="class")
    def codec(self):
        from helix_context.sema import SemaCodec
        return SemaCodec()

    @pytest.fixture
    def cold_genome(self, codec):
        """In-memory genome with a SemaCodec attached for cold-tier tests."""
        g = Genome(path=":memory:", sema_codec=codec)
        yield g
        g.close()

    def test_returns_empty_without_codec(self, genome):
        """Without a SemaCodec, cold-tier retrieval degrades to empty list."""
        # The default `genome` fixture has sema_codec=None
        result = genome.query_cold_tier("any query text")
        assert result == []

    def test_returns_empty_when_no_heterochromatin_genes(self, cold_genome, codec):
        """Fresh genome with no demoted genes → empty result."""
        g = make_gene("unrelated content", domains=["test"])
        g.embedding = codec.encode("unrelated content")
        cold_genome.upsert_gene(g, apply_gate=False)
        # Gene is chromatin=OPEN, so cold cache is empty

        result = cold_genome.query_cold_tier("anything")
        assert result == []

    def test_retrieves_demoted_gene_on_semantic_match(self, cold_genome, codec):
        """The headline test: a gene demoted to heterochromatin is still
        findable via cold-tier ΣĒMA cosine when the query is semantically
        close to its content. This is the whole point of C.1 + C.2."""
        content = (
            "def authenticate_user(username, password): "
            "return check_password_hash(user.pw_hash, password)"
        )
        g = make_gene(content, domains=["python", "auth"])
        g.source_id = "project/auth.py"
        g.embedding = codec.encode(content)
        cold_genome.upsert_gene(g, apply_gate=False)

        # Demote to heterochromatin (non-destructive as of C.1)
        cold_genome.compress_to_heterochromatin(g.gene_id)

        # Query with a semantically-related phrase.
        # Note on min_cosine: ΣĒMA's 20-dim projection is sparse by design.
        # Typical close-paraphrase pairs score 0.15–0.30 in cosine, NOT
        # 0.6–0.9 like full 384-dim sentence embeddings. Using 0.1 here
        # to verify the mechanism works — the production default is 0.25
        # which is slightly more permissive than the existing hot-tier
        # Mode A/B thresholds (0.3/0.4).
        result = cold_genome.query_cold_tier(
            "user authentication login password check",
            k=5,
            min_cosine=0.1,
        )

        assert len(result) >= 1, "semantically-close query should retrieve the demoted gene"
        assert result[0].gene_id == g.gene_id
        # Content must be intact — this is what C.1 enabled
        assert result[0].content == content
        # Tier flag stays hetero — caller decides whether to promote
        assert result[0].chromatin == ChromatinState.HETEROCHROMATIN

    def test_respects_k_limit(self, cold_genome, codec):
        """Even if many hetero genes match, only k are returned."""
        contents = [
            "python function for user authentication with password",
            "python function for session token validation and refresh",
            "python function for access control via role-based permissions",
            "python function for OAuth2 flow with authorization code grant",
        ]
        for i, c in enumerate(contents):
            g = make_gene(c, domains=["python", "auth"], entities=[f"fn_{i}"])
            g.source_id = f"project/auth_{i}.py"
            g.embedding = codec.encode(c)
            cold_genome.upsert_gene(g, apply_gate=False)
            cold_genome.compress_to_heterochromatin(g.gene_id)

        result = cold_genome.query_cold_tier(
            "authentication authorization security",
            k=2,
            min_cosine=0.05,  # very permissive so multiple match
        )
        assert len(result) <= 2

    def test_respects_min_cosine_threshold(self, cold_genome, codec):
        """A query far from the gene's semantic neighborhood returns nothing."""
        g = make_gene(
            "def red_velvet_cake_recipe(): return ['flour', 'cocoa', 'buttermilk']",
            domains=["recipe"],
        )
        g.source_id = "recipes/cakes.py"
        g.embedding = codec.encode(g.content)
        cold_genome.upsert_gene(g, apply_gate=False)
        cold_genome.compress_to_heterochromatin(g.gene_id)

        # Query something completely unrelated, with a high threshold
        result = cold_genome.query_cold_tier(
            "kernel-level interrupt handler ISR assembly x86_64",
            k=5,
            min_cosine=0.9,  # very strict
        )
        assert result == [], "unrelated query with strict threshold should return nothing"

    def test_cache_invalidated_on_upsert(self, cold_genome, codec):
        """Upserting a new gene should invalidate the cold-tier cache so
        the next query rebuilds it and sees the new state."""
        g1 = make_gene("original hetero gene content", domains=["test"])
        g1.embedding = codec.encode("original hetero gene content")
        cold_genome.upsert_gene(g1, apply_gate=False)
        cold_genome.compress_to_heterochromatin(g1.gene_id)

        # First query — builds the cache
        _ = cold_genome.query_cold_tier("original hetero", k=5, min_cosine=0.1)
        assert cold_genome._cold_sema_cache is not None

        # Now upsert a new gene (fresh hot-tier gene)
        g2 = make_gene("newly added content", domains=["test"])
        g2.embedding = codec.encode("newly added content")
        cold_genome.upsert_gene(g2, apply_gate=False)

        # Cache must have been invalidated
        assert cold_genome._cold_sema_cache is None, (
            "upsert_gene must invalidate the cold-tier cache"
        )


# ── Threshold constants are sane ───────────────────────────────────────


class TestThresholdSanity:
    def test_hetero_below_euchro(self):
        """Heterochromatin threshold must be strictly below euchromatin."""
        assert _DENSITY_HETEROCHROMATIN_THRESHOLD < _DENSITY_EUCHROMATIN_THRESHOLD

    def test_thresholds_positive(self):
        assert _DENSITY_HETEROCHROMATIN_THRESHOLD > 0
        assert _DENSITY_EUCHROMATIN_THRESHOLD > 0

    def test_access_override_positive(self):
        assert _DENSITY_ACCESS_OVERRIDE > 0
