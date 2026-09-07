"""
Tests for the launcher graph summary — the read-only layer ledger on the
Knowledge store tab.

Four things are pinned here, in this order:

  1. Aggregation semantics. Reciprocal COVER rows count as one undirected
     link; COVER-connected genes are the union of both endpoints; entities
     are distinct by identifier while memberships stay a row count. A
     SYMBOL_REF row shares `gene_relations` with CHUNK_OF, so every count
     filters on the relation code.
  2. The bounded TTL cache. Same path inside the TTL reads once, a second
     path is a miss, an expired entry re-reads, and the cache evicts.
  3. The unavailable state. A missing table, a missing file, or a corrupt
     database yields `available: False` with the reason in the log and in
     `error` — never invented zeroes, never a raise, and never the reason
     on the page.
  4. The two CSS registrations. `.panel--graph-summary` has to join both
     the `span 12` group and the Knowledge-store-tab allowlist; drop
     either and every HTML assertion still passes while the panel renders
     half-width or invisible.

The panel path is also the reason this file asserts that
`state["database"]["active_path"]` round-trips an uppercase genome path
unchanged: the summary opens that string with sqlite, so case-folding it
makes the panel unreadable on any case-sensitive filesystem.

The `.modal-backdrop[hidden]` guard is deliberately not re-pinned here.
It already ships in `launcher.css` and
`tests/test_launcher_dashboard_wiring.py::test_modal_backdrop_has_hidden_display_guard`
already covers it.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cymatix_context.launcher import genome_registry as gr
from cymatix_context.launcher import graph_summary
from cymatix_context.launcher.collector import StateCollector
from cymatix_context.schemas import NLRelation, StructuralRelation

LAUNCHER_DIR = Path(__file__).resolve().parent.parent / "cymatix_context" / "launcher"
CSS_PATH = LAUNCHER_DIR / "static" / "launcher.css"

COVER = int(NLRelation.COVER)
CHUNK_OF = int(StructuralRelation.CHUNK_OF)
SYMBOL_REF = int(StructuralRelation.SYMBOL_REF)


# ── fixture genome ────────────────────────────────────────────────────

# Column shapes follow cymatix_context/storage/ddl.py; only the columns the
# summary reads are carried, so the fixture stays legible.
_SCHEMA = """
CREATE TABLE genes (
    gene_id TEXT PRIMARY KEY,
    content TEXT
);
CREATE TABLE gene_relations (
    gene_id_a  TEXT,
    gene_id_b  TEXT,
    relation   INTEGER,
    confidence REAL,
    updated_at REAL,
    PRIMARY KEY (gene_id_a, gene_id_b)
);
CREATE TABLE harmonic_links (
    gene_id_a  TEXT NOT NULL,
    gene_id_b  TEXT NOT NULL,
    weight     REAL NOT NULL,
    updated_at REAL NOT NULL,
    source     TEXT NOT NULL DEFAULT 'co_retrieved',
    PRIMARY KEY (gene_id_a, gene_id_b)
);
CREATE TABLE entity_graph (
    entity  TEXT NOT NULL,
    gene_id TEXT NOT NULL,
    PRIMARY KEY (entity, gene_id)
);
"""

# Five genes; three of them touched by a COVER relation.
#   COVER: g1<->g2 stored both ways (one undirected link), g2->g3 (a second).
#          So 3 stored rows, 2 distinct links, 3 connected genes — and g3 is
#          only ever a `gene_id_b`, which is what catches an a-only union.
#   CHUNK_OF: g4->g1, g5->g1.
#   SYMBOL_REF: g3->g4, which must not be counted as anything.
_RELATIONS = [
    ("g1", "g2", COVER),
    ("g2", "g1", COVER),
    ("g2", "g3", COVER),
    ("g4", "g1", CHUNK_OF),
    ("g5", "g1", CHUNK_OF),
    ("g3", "g4", SYMBOL_REF),
]

_HARMONIC = [
    ("g1", "g2", 0.9, 1.0, "co_retrieved"),
    ("g2", "g3", 0.7, 2.0, "co_retrieved"),
    ("g3", "g1", 0.5, 3.0, "seeded"),
]

# Entity "ent:alpha" repeats across three genes: 3 distinct entities,
# 5 membership rows.
_ENTITIES = [
    ("ent:alpha", "g1"),
    ("ent:alpha", "g2"),
    ("ent:alpha", "g3"),
    ("ent:beta", "g4"),
    ("ent:gamma", "g5"),
]

EXPECTED_COUNTS = {
    "available": True,
    "total_genes": 5,
    "cover_connected_genes": 3,
    "cover_links": 2,
    "cover_rows": 3,
    "chunk_of_rows": 2,
    "harmonic_links": 3,
    "entities": 3,
    "entity_memberships": 5,
}


def _graph_db(path: Path) -> Path:
    """Write a small but schema-faithful genome at `path`."""
    conn = sqlite3.connect(str(path))
    try:
        # WAL, the journal mode `KnowledgeStore` puts the real genome in
        # (`knowledge_store.py`), so what these tests measure is what the
        # launcher meets on disk.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO genes (gene_id, content) VALUES (?, ?)",
            [(f"g{i}", f"body {i}") for i in range(1, 6)],
        )
        conn.executemany(
            "INSERT INTO gene_relations "
            "(gene_id_a, gene_id_b, relation, confidence, updated_at) "
            "VALUES (?, ?, ?, 1.0, 0.0)",
            _RELATIONS,
        )
        conn.executemany(
            "INSERT INTO harmonic_links "
            "(gene_id_a, gene_id_b, weight, updated_at, source) "
            "VALUES (?, ?, ?, ?, ?)",
            _HARMONIC,
        )
        conn.executemany(
            "INSERT INTO entity_graph (entity, gene_id) VALUES (?, ?)",
            _ENTITIES,
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _genes_only_db(path: Path) -> Path:
    """A genome predating the graph tables: `genes` and nothing else."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript("CREATE TABLE genes (gene_id TEXT PRIMARY KEY);")
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture(autouse=True)
def _isolated_launcher_state(tmp_path, monkeypatch):
    """Keep the registry off the developer's real ~/.cymatix/launcher.

    Same reason as tests/test_launcher_dashboard_wiring.py: without this a
    test run can hijack the real tray's active genome."""
    monkeypatch.setenv(
        "CYMATIX_LAUNCHER_STATE_DIR", str(tmp_path / "launcher-state"),
    )
    monkeypatch.delenv("CYMATIX_GENOME_PATH", raising=False)
    gr.clear_cache()
    yield
    gr.clear_cache()


# ── aggregation ───────────────────────────────────────────────────────


class TestAggregation:
    def test_summary_reports_every_layer(self, tmp_path):
        db = _graph_db(tmp_path / "genome.db")

        summary = graph_summary._read_graph_summary(db)

        assert summary["path"] == str(db)
        assert {k: v for k, v in summary.items() if k != "path"} == EXPECTED_COUNTS

    def test_reciprocal_cover_rows_are_one_undirected_link(self, tmp_path):
        """g1->g2 and g2->g1 are the same link. The stored row count still
        reports both, because that is the number a diagnosis needs."""
        db = _graph_db(tmp_path / "genome.db")

        summary = graph_summary._read_graph_summary(db)

        assert summary["cover_links"] == 2
        assert summary["cover_rows"] == 3

    def test_cover_connected_genes_unions_both_endpoints(self, tmp_path):
        """g3 appears only as a `gene_id_b`. Counting `gene_id_a` alone
        returns 2 and the panel under-reports for good."""
        db = _graph_db(tmp_path / "genome.db")

        assert graph_summary._read_graph_summary(db)["cover_connected_genes"] == 3

    def test_entities_are_distinct_and_memberships_are_rows(self, tmp_path):
        db = _graph_db(tmp_path / "genome.db")

        summary = graph_summary._read_graph_summary(db)

        assert summary["entities"] == 3
        assert summary["entity_memberships"] == 5

    def test_relation_codes_come_from_the_enums(self, tmp_path):
        """`gene_relations` is a discriminated union, so a bare 5 or 100 in
        the module survives a renumbering of the enum and silently counts
        the wrong layer. SYMBOL_REF (101) sits in the same table and must
        not land in either count."""
        assert NLRelation.COVER == 5
        assert StructuralRelation.CHUNK_OF == 100

        source = Path(graph_summary.__file__).read_text(encoding="utf-8")
        assert "NLRelation.COVER" in source
        assert "StructuralRelation.CHUNK_OF" in source
        assert not re.search(r"relation\s*=\s*(?:5|100)\b", source)

        db = _graph_db(tmp_path / "genome.db")
        summary = graph_summary._read_graph_summary(db)
        assert summary["chunk_of_rows"] == 2
        assert summary["cover_rows"] == 3

    def test_cover_links_is_skipped_above_the_row_ceiling(self, tmp_path, monkeypatch):
        """Distinct undirected pairs need a temp B-tree over every COVER
        row, so the query is bounded. Above the ceiling the field is None
        and the panel reports stored rows instead."""
        assert graph_summary.COVER_LINKS_MAX_ROWS == 500_000

        db = _graph_db(tmp_path / "genome.db")
        monkeypatch.setattr(graph_summary, "COVER_LINKS_MAX_ROWS", 2)

        summary = graph_summary._read_graph_summary(db)

        assert summary["cover_links"] is None
        assert summary["cover_rows"] == 3
        assert summary["total_genes"] == 5

    def test_summary_does_not_modify_the_genome(self, tmp_path):
        # A `mode=ro` open of a WAL database does create the `-wal` and
        # `-shm` sidecars, so the claim pinned here is the one the spec
        # makes: the genome file itself is not modified.
        db = _graph_db(tmp_path / "genome.db")
        before = db.stat().st_mtime_ns

        graph_summary._read_graph_summary(db)

        assert db.stat().st_mtime_ns == before

    def test_a_hash_in_the_genome_path_reads_and_creates_nothing(self, tmp_path):
        # Built by interpolation, `file:{path}?mode=ro` re-parses here: `#`
        # opens the URI fragment, `?mode=ro` goes with it, and sqlite opens
        # `.../gen` read-write and creates it. `?` cannot appear in a Windows
        # filename; `#` can.
        home = tmp_path / "genomes"
        home.mkdir()
        db = _graph_db(home / "gen#1.db")

        summary = graph_summary._read_graph_summary(db)

        assert summary["available"] is True
        assert summary["total_genes"] == 5
        assert not (home / "gen").exists()
        assert {entry.name for entry in home.iterdir()} <= {
            "gen#1.db",
            "gen#1.db-wal",
            "gen#1.db-shm",
        }


# ── cache ─────────────────────────────────────────────────────────────


class TestCache:
    @pytest.mark.parametrize("unavailable", [False, True])
    @pytest.mark.parametrize("expired", [False, True])
    def test_concurrent_misses_share_one_read(self, tmp_path, unavailable, expired):
        db = tmp_path / "genome.db"
        clock = [100.0]
        cache = graph_summary.GraphSummaryCache(clock=lambda: clock[0])
        payload = {"available": True, "path": str(db), "total_genes": 5}
        if expired:
            with patch.object(graph_summary, "_read_graph_summary", return_value=payload):
                cache.get(db)
            clock[0] = 131.0

        started = threading.Event()
        follower_checked = threading.Event()
        caller = threading.local()
        lock = threading.Lock()

        class ObservedLock:
            def __enter__(self):
                lock.acquire()

            def __exit__(self, *args):
                lock.release()
                if getattr(caller, "follower", False):
                    follower_checked.set()

        cache._lock = ObservedLock()

        def read(path):
            started.set()
            assert follower_checked.wait(5), "second caller never checked the cache"
            if unavailable:
                raise sqlite3.OperationalError("unreadable")
            return payload

        def follow():
            caller.follower = True
            return cache.get(db)

        with patch.object(graph_summary, "_read_graph_summary", side_effect=read) as reader:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(cache.get, db)
                assert started.wait(5)
                second = pool.submit(follow)
                a, b = first.result(timeout=5), second.result(timeout=5)
            assert reader.call_count == 1
            assert a == b
            assert a is not b
            assert a["available"] is not unavailable
            assert cache.get(db) == a

    def test_slow_path_does_not_block_another_path(self, tmp_path):
        slow = tmp_path / "slow.db"
        other = tmp_path / "other.db"
        started = threading.Event()
        release = threading.Event()
        cache = graph_summary.GraphSummaryCache()

        def read(path):
            if path == slow:
                started.set()
                assert release.wait(5)
            return {"available": True, "path": str(path)}

        with patch.object(graph_summary, "_read_graph_summary", side_effect=read):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(cache.get, slow)
                try:
                    assert started.wait(5)
                    second = pool.submit(cache.get, other)
                    assert second.result(timeout=5)["path"] == str(other)
                finally:
                    release.set()
                assert first.result(timeout=5)["path"] == str(slow)

    def test_ttl_starts_when_the_read_finishes(self, tmp_path):
        clock = [100.0]
        cache = graph_summary.GraphSummaryCache(clock=lambda: clock[0])

        def read(path):
            clock[0] = 131.0
            return {"available": True, "path": str(path)}

        with patch.object(graph_summary, "_read_graph_summary", side_effect=read) as reader:
            cache.get(tmp_path / "slow.db")
            clock[0] = 132.0
            cache.get(tmp_path / "slow.db")
            assert reader.call_count == 1

    def test_ttl_default_is_about_thirty_seconds(self):
        assert graph_summary.GRAPH_SUMMARY_TTL_S == 30.0

    def test_hits_reuse_a_path_and_expiry_refreshes(self, tmp_path):
        first = tmp_path / "first.db"
        second = tmp_path / "second.db"
        first.touch()
        second.touch()
        clock = [100.0]
        cache = graph_summary.GraphSummaryCache(ttl_s=30.0, clock=lambda: clock[0])
        payload = {"available": True, "path": str(first), "total_genes": 1}

        with patch.object(
            graph_summary, "_read_graph_summary", return_value=payload,
        ) as read:
            assert cache.get(first)["total_genes"] == 1
            assert cache.get(first)["total_genes"] == 1
            assert read.call_count == 1

            cache.get(second)
            assert read.call_count == 2

            clock[0] = 131.0
            cache.get(first)
            assert read.call_count == 3

    def test_cache_is_bounded(self, tmp_path):
        """A launcher that walks a bench directory can see many genomes.
        The cache holds `max_paths` and evicts the oldest."""
        paths = []
        for i in range(3):
            p = tmp_path / f"g{i}.db"
            p.touch()
            paths.append(p)
        clock = [100.0]
        cache = graph_summary.GraphSummaryCache(
            ttl_s=30.0, max_paths=2, clock=lambda: clock[0],
        )

        def _read(path):
            return {"available": True, "path": str(path), "total_genes": 1}

        with patch.object(
            graph_summary, "_read_graph_summary", side_effect=_read,
        ) as read:
            for p in paths:
                cache.get(p)
                clock[0] += 1.0
            assert read.call_count == 3

            cache.get(paths[0])  # oldest, evicted — reads again
            assert read.call_count == 4

            cache.get(paths[2])  # still resident — no read
            assert read.call_count == 4

    def test_cached_payload_is_not_shared_mutable_state(self, tmp_path):
        db = _graph_db(tmp_path / "genome.db")
        cache = graph_summary.GraphSummaryCache()

        first = cache.get(db)
        first["total_genes"] = -1

        assert cache.get(db)["total_genes"] == 5


# ── unavailable state ─────────────────────────────────────────────────


class TestUnavailable:
    def test_missing_graph_tables_are_unavailable_and_logged(self, tmp_path, caplog):
        db = _genes_only_db(tmp_path / "old.genome.db")
        cache = graph_summary.GraphSummaryCache()

        with caplog.at_level(logging.WARNING):
            result = cache.get(db)

        assert result["available"] is False
        assert result["path"] == str(db.resolve())
        assert "gene_relations" in result["error"]
        assert "Graph summary unavailable" in caplog.text

    def test_missing_file_is_unavailable(self, tmp_path):
        missing = tmp_path / "nowhere" / "genome.db"

        result = graph_summary.GraphSummaryCache().get(missing)

        assert result["available"] is False
        assert result["error"]

    def test_corrupt_database_is_unavailable(self, tmp_path):
        db = tmp_path / "corrupt.genome.db"
        db.write_bytes(b"SQLite format 3\x00" + b"\x00" * 512)

        result = graph_summary.GraphSummaryCache().get(db)

        assert result["available"] is False
        assert result["error"]

    def test_unavailable_invents_no_counts(self, tmp_path):
        """An empty genome and an unreadable one must not look alike. The
        unavailable payload carries no count keys at all."""
        db = _genes_only_db(tmp_path / "old.genome.db")

        result = graph_summary.GraphSummaryCache().get(db)

        for field in (
            "total_genes",
            "cover_connected_genes",
            "cover_links",
            "cover_rows",
            "chunk_of_rows",
            "harmonic_links",
            "entities",
            "entity_memberships",
        ):
            assert field not in result


# ── collector wiring ──────────────────────────────────────────────────


class TestCollectorWiring:
    def _stopped_supervisor(self):
        supervisor = MagicMock()
        supervisor.is_running.return_value = False
        supervisor.find_orphan_cymatix.return_value = None
        supervisor.get_last_error.return_value = None
        supervisor.owns_process.return_value = False
        return supervisor

    def test_summary_is_attached_while_cymatix_is_stopped(self, tmp_path):
        """The summary reads the genome file, not the server, so it must be
        attached above the stopped-server early return in `collect`."""
        db = _graph_db(tmp_path / "genome.db")
        cache = MagicMock()
        cache.get.return_value = {"available": True, "total_genes": 5_636}
        subject = StateCollector(
            supervisor=self._stopped_supervisor(), graph_summary_cache=cache,
        )

        with patch.object(
            subject,
            "_database_panel",
            return_value={"active_path": str(db), "entries": []},
        ):
            state = subject.collect()

        assert state["cymatix"]["running"] is False
        cache.get.assert_called_once_with(str(db))
        assert state["graph_summary"]["total_genes"] == 5_636

    def test_active_path_reaches_the_cache_untouched(self, tmp_path):
        """Whatever the database panel publishes is what gets opened. If
        anything folds the case between the two, this is where it shows."""
        genome_dir = tmp_path / "Genomes"
        genome_dir.mkdir()
        db = _graph_db(genome_dir / "Dogfood.genome.db")
        cache = MagicMock()
        cache.get.return_value = {"available": True, "total_genes": 5}
        subject = StateCollector(
            supervisor=self._stopped_supervisor(), graph_summary_cache=cache,
        )

        with patch.object(
            subject,
            "_database_panel",
            return_value={"active_path": str(db), "entries": []},
        ):
            subject.collect()

        cache.get.assert_called_once_with(str(db))
        assert str(db) != str(db).lower()

    def test_no_active_path_means_no_summary(self):
        """`_database_panel` returns `{"error": ..., "entries": []}` when the
        registry cannot be read. No path, no panel."""
        cache = MagicMock()
        subject = StateCollector(
            supervisor=self._stopped_supervisor(), graph_summary_cache=cache,
        )

        with patch.object(
            subject, "_database_panel", return_value={"error": "boom", "entries": []},
        ):
            state = subject.collect()

        assert "graph_summary" not in state
        cache.get.assert_not_called()

    def test_absent_genome_file_is_not_summarised(self, tmp_path):
        """`active_genome_path()` falls back to a repo-root `genome.db`
        whether or not it exists, so a fresh install with no genome would
        otherwise log a warning every thirty seconds forever."""
        cache = MagicMock()
        subject = StateCollector(
            supervisor=self._stopped_supervisor(), graph_summary_cache=cache,
        )

        with patch.object(
            subject,
            "_database_panel",
            return_value={"active_path": str(tmp_path / "absent.db"), "entries": []},
        ):
            state = subject.collect()

        assert "graph_summary" not in state
        cache.get.assert_not_called()


# ── the published path ────────────────────────────────────────────────


def test_database_panel_publishes_an_openable_path(tmp_path, monkeypatch):
    """The defect this panel exposes: the database panel case-folded the
    active genome path before publishing it. On a case-sensitive
    filesystem nothing downstream can open the result — including the
    summary, which hands the string straight to sqlite.

    `os.path.exists` on the published value is the assertion a Windows box
    also passes and a case-folded string cannot."""
    genome_dir = tmp_path / "Genomes"
    genome_dir.mkdir()
    db = _graph_db(genome_dir / "Dogfood.genome.db")
    monkeypatch.setenv("CYMATIX_GENOME_PATH", str(db))
    gr.clear_cache()

    panel = StateCollector(supervisor=MagicMock())._database_panel()

    assert str(db) != str(db).lower(), "fixture must carry uppercase to be a test"
    assert panel["active_path"] == str(db.resolve())
    assert os.path.exists(panel["active_path"])


def test_is_active_still_matches_case_insensitively(tmp_path, monkeypatch):
    """The registry does its own comparison and keeps folding case, so a
    Windows genome selected as `F:/Genomes/...` and discovered as
    `f:/genomes/...` still shows as active. Dropping the fold in the
    collector must not disturb that."""
    genome_dir = tmp_path / "Genomes"
    genome_dir.mkdir()
    db = _graph_db(genome_dir / "Dogfood.genome.db")
    monkeypatch.setenv("CYMATIX_GENOME_PATH", str(db))
    gr.clear_cache()

    panel = StateCollector(supervisor=MagicMock())._database_panel()

    active = [e for e in panel["entries"] if e["is_active"]]
    assert [Path(e["path"]) for e in active] == [db.resolve()]


# ── rendering ─────────────────────────────────────────────────────────


def _state_with_summary(summary):
    return {
        "cymatix": {"running": False, "availability": "unavailable"},
        "graph_summary": summary,
    }


DOGFOOD_SUMMARY = {
    "available": True,
    "path": "F:/Genomes/Dogfood/Genome.db",
    "total_genes": 5_636,
    "cover_connected_genes": 4_049,
    "cover_links": 29_061,
    "cover_rows": 58_122,
    "chunk_of_rows": 4_240,
    "harmonic_links": 28,
    "entities": 14_186,
    "entity_memberships": 42_236,
}


@pytest.fixture
def panels_client():
    """A launcher app whose collector returns exactly the state under test."""
    pytest.importorskip("jinja2", reason="launcher extra not installed")
    from fastapi.testclient import TestClient

    from cymatix_context.launcher.app import create_app

    store = MagicMock()
    store.state.cymatix_pid = None
    supervisor = MagicMock()
    supervisor.store = store
    supervisor.is_running.return_value = False
    collector = MagicMock()
    app = create_app(store=store, supervisor=supervisor, collector=collector)
    with TestClient(app) as client:
        yield client, collector


class TestPanelRender:
    def test_panel_renders_with_formatted_counts(self, panels_client):
        client, collector = panels_client
        collector.collect.return_value = _state_with_summary(DOGFOOD_SUMMARY)

        html = client.get("/api/state/panels").text

        assert "panel--graph-summary" in html
        assert "Graph summary" in html
        assert "Renderer deferred" in html
        assert "5,636" in html
        assert "4,049 COVER-connected" in html
        assert "29,061" in html
        assert "distinct COVER links" in html
        assert "4,240" in html
        assert "42,236 memberships" in html
        assert "F:/Genomes/Dogfood/Genome.db" in html

    def test_bounded_semantic_row_reports_stored_rows(self, panels_client):
        client, collector = panels_client
        summary = dict(DOGFOOD_SUMMARY, cover_links=None, cover_rows=7_224_963)
        collector.collect.return_value = _state_with_summary(summary)

        html = client.get("/api/state/panels").text

        assert "7,224,963" in html
        assert "distinct links not counted at this scale" in html

    def test_unavailable_state_does_not_leak_the_reason(self, panels_client):
        client, collector = panels_client
        collector.collect.return_value = _state_with_summary({
            "available": False,
            "path": "F:/Genomes/Dogfood/Genome.db",
            "error": "missing graph tables: entity_graph, harmonic_links",
        })

        html = client.get("/api/state/panels").text

        assert "panel--graph-summary" in html
        assert "Graph summary unavailable" in html
        assert "missing graph tables" not in html
        assert "F:/Genomes/Dogfood/Genome.db" in html

    def test_no_summary_means_no_panel(self, panels_client):
        client, collector = panels_client
        collector.collect.return_value = {
            "cymatix": {"running": False, "availability": "unavailable"},
        }

        html = client.get("/api/state/panels").text

        assert "panel--graph-summary" not in html

    def test_panel_ships_no_script(self, panels_client):
        """A ledger, not a renderer: the partial adds no client-side work."""
        client, collector = panels_client
        collector.collect.return_value = _state_with_summary(DOGFOOD_SUMMARY)

        html = client.get("/api/state/panels").text

        assert "<script" not in html


# ── stylesheet registration ───────────────────────────────────────────


def _selector_tokens(css: str, declaration: str) -> set:
    """Every individual selector of every rule whose body carries
    `declaration`, whitespace-normalised."""
    body = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    tokens = set()
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", body):
        if declaration in match.group(2):
            for part in match.group(1).split(","):
                tokens.add(" ".join(part.split()))
    return tokens


def test_panel_is_registered_full_width_and_on_the_knowledge_store_tab():
    """The silent regression. `launcher.css` hides every `.panel` on the
    genome, agents and monitoring tabs and brings back an explicit
    allowlist, and `.panel` defaults to `span 6`. Miss either registration
    and the HTML assertions above all still pass while the panel is
    half-width or invisible on the one tab it was built for."""
    css = CSS_PATH.read_text(encoding="utf-8")

    assert ".panel--graph-summary" in _selector_tokens(css, "grid-column: span 12")

    hidden = _selector_tokens(css, "display: none")
    assert '.panels[data-active-tab="genome"] .panel' in hidden

    shown = _selector_tokens(css, "display: grid")
    assert '.panels[data-active-tab="genome"] .panel--graph-summary' in shown


def test_panel_styles_use_tokens_only():
    """No raw hex, no new radius, no animation in the added block — every
    value comes from the stylesheet's own custom properties."""
    css = CSS_PATH.read_text(encoding="utf-8")
    match = re.search(
        r"\.graph-layers\s*\{.*?(?=\n/\* ──|\Z)", css, flags=re.DOTALL,
    )
    assert match, ".graph-layers block missing from launcher.css"
    block = match.group(0)

    assert not re.search(r"#[0-9a-fA-F]{3,8}\b", block)
    assert "@keyframes" not in block
    assert "transition" not in block
    assert "outline: none" not in block


# ── cost ──────────────────────────────────────────────────────────────


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("CYMATIX_TEST_PERF") != "1",
    reason="Set CYMATIX_TEST_PERF=1 to run the graph summary cost measurement",
)
def test_summary_stays_inside_its_budget_on_a_large_bed(tmp_path):
    """A dev-box measurement, not a CI gate. The dashboard polls every two
    seconds and reads the cache; this bounds what one cache miss costs.
    Budget is deliberately loose — it is a smoke alarm for an accidental
    full scan, not a benchmark."""
    db = tmp_path / "big.genome.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO genes (gene_id, content) VALUES (?, ?)",
            [(f"g{i}", "x") for i in range(50_000)],
        )
        conn.executemany(
            "INSERT INTO gene_relations "
            "(gene_id_a, gene_id_b, relation, confidence, updated_at) "
            "VALUES (?, ?, ?, 1.0, 0.0)",
            [(f"g{i}", f"g{(i + 1) % 50_000}", COVER) for i in range(50_000)],
        )
        conn.commit()
    finally:
        conn.close()

    started = time.perf_counter()
    summary = graph_summary._read_graph_summary(db)
    elapsed = time.perf_counter() - started

    assert summary["total_genes"] == 50_000
    assert elapsed < 2.0, f"graph summary took {elapsed:.2f}s on a 50k-gene bed"
