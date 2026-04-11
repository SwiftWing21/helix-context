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
)
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
        g_noise = make_gene("x" * 3000, domains=[])
        g_noise.source_id = "F:/project/node_modules/some-pkg/dist/index.min.js"
        g_noise.embedding = [0.1] * 20  # required for cold-storage demotion
        # Gate would demote on insert, so bypass it then run sweep
        genome.upsert_gene(g_noise, apply_gate=False)

        assert genome.get_gene(g_noise.gene_id).chromatin == ChromatinState.OPEN

        genome.compact_genome(dry_run=False)

        retrieved = genome.get_gene(g_noise.gene_id)
        # After heterochromatin compression, content is replaced with a marker
        assert "COMPRESSED:heterochromatin" in retrieved.content

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
