"""Tests for scripts/tombstone_orphan_parents.py.

Builds a throwaway genome with known-orphan and known-live parent
genes, then verifies the tombstone logic:
    - orphan parents are deleted
    - their CHUNK_OF edges are deleted
    - live parents are preserved
    - child genes are preserved by default
    - --include-children sweeps orphan children
    - --dry-run mutates nothing
    - non-path source_ids are left alone (no false-positive deletes)
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "tombstone_orphan_parents.py"
CHUNK_OF = 100


def _mk_genome(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE genes (
            gene_id      TEXT PRIMARY KEY,
            content      TEXT,
            complement   TEXT,
            codons       TEXT,
            promoter     TEXT,
            epigenetics  TEXT,
            chromatin    INTEGER,
            is_fragment  INTEGER,
            embedding    TEXT,
            source_id    TEXT,
            version      INTEGER,
            supersedes   TEXT,
            key_values   TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE gene_relations (
            gene_id_a    TEXT,
            gene_id_b    TEXT,
            relation     INTEGER,
            confidence   REAL,
            updated_at   REAL,
            PRIMARY KEY (gene_id_a, gene_id_b, relation)
        )
    """)
    conn.commit()
    return conn


def _insert_gene(
    conn: sqlite3.Connection,
    gene_id: str,
    source_id: str,
    is_parent: bool = False,
) -> None:
    kv = json.dumps(["is_parent=true"]) if is_parent else json.dumps([])
    conn.execute(
        "INSERT INTO genes (gene_id, content, complement, codons, promoter, "
        "epigenetics, chromatin, is_fragment, embedding, source_id, version, "
        "supersedes, key_values) VALUES (?, 'c', 'cmp', '[]', '{}', '{}', 0, 0, "
        "NULL, ?, 1, NULL, ?)",
        (gene_id, source_id, kv),
    )


def _insert_chunk_of(conn: sqlite3.Connection, child: str, parent: str) -> None:
    conn.execute(
        "INSERT INTO gene_relations (gene_id_a, gene_id_b, relation, "
        "confidence, updated_at) VALUES (?, ?, ?, 1.0, 0)",
        (child, parent, CHUNK_OF),
    )


@pytest.fixture
def genome_with_orphans(tmp_path: Path):
    """Set up a genome with 1 live parent, 1 orphan parent, 1 non-path parent,
    plus children for each."""
    # Create a real file so we have a live source_id
    live_file = tmp_path / "live.md"
    live_file.write_text("live content")

    genome_path = tmp_path / "genome.db"
    conn = _mk_genome(str(genome_path))

    # Live parent — file exists
    _insert_gene(conn, "parent_live", str(live_file).replace("\\", "/"), is_parent=True)
    _insert_gene(conn, "child_live_1", str(live_file).replace("\\", "/"))
    _insert_gene(conn, "child_live_2", str(live_file).replace("\\", "/"))
    _insert_chunk_of(conn, "child_live_1", "parent_live")
    _insert_chunk_of(conn, "child_live_2", "parent_live")

    # Orphan parent — file does NOT exist
    orphan_path = str((tmp_path / "gone.md").as_posix())
    _insert_gene(conn, "parent_orphan", orphan_path, is_parent=True)
    _insert_gene(conn, "child_orphan_1", orphan_path)
    _insert_gene(conn, "child_orphan_2", orphan_path)
    _insert_chunk_of(conn, "child_orphan_1", "parent_orphan")
    _insert_chunk_of(conn, "child_orphan_2", "parent_orphan")

    # Non-path source — free-form string, MUST be preserved
    _insert_gene(conn, "parent_nonpath", "manual_note:ops", is_parent=True)

    conn.commit()
    conn.close()
    return genome_path


def _run(genome: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--genome", str(genome), *args],
        capture_output=True, text=True, check=False,
    )


def _gene_ids(genome: Path) -> set[str]:
    conn = sqlite3.connect(str(genome))
    try:
        return {row[0] for row in conn.execute("SELECT gene_id FROM genes")}
    finally:
        conn.close()


def _edge_count(genome: Path) -> int:
    conn = sqlite3.connect(str(genome))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM gene_relations WHERE relation = ?",
            (CHUNK_OF,),
        ).fetchone()[0]
    finally:
        conn.close()


def test_dry_run_mutates_nothing(genome_with_orphans: Path):
    before = _gene_ids(genome_with_orphans)
    before_edges = _edge_count(genome_with_orphans)
    res = _run(genome_with_orphans, "--dry-run")
    assert res.returncode == 0, res.stderr
    assert "orphan parents:" in res.stdout
    assert "dry-run" in res.stdout
    assert _gene_ids(genome_with_orphans) == before
    assert _edge_count(genome_with_orphans) == before_edges


def test_live_run_deletes_orphan_parents_and_edges(genome_with_orphans: Path):
    res = _run(genome_with_orphans)
    assert res.returncode == 0, res.stderr

    remaining = _gene_ids(genome_with_orphans)

    # Orphan parent is gone
    assert "parent_orphan" not in remaining
    # Live parent is preserved
    assert "parent_live" in remaining
    # Non-path "parent" is preserved (no deletion for unresolvable but non-path source)
    assert "parent_nonpath" in remaining
    # Children preserved by default (even the orphan's children — they might
    # still be valid as content-addressable, just with a stale source_id)
    assert "child_live_1" in remaining
    assert "child_live_2" in remaining
    assert "child_orphan_1" in remaining
    assert "child_orphan_2" in remaining


def test_chunk_of_edges_to_orphan_parent_are_removed(genome_with_orphans: Path):
    _run(genome_with_orphans)
    conn = sqlite3.connect(str(genome_with_orphans))
    try:
        orphan_edges = conn.execute(
            "SELECT COUNT(*) FROM gene_relations WHERE gene_id_b = 'parent_orphan'"
        ).fetchone()[0]
        live_edges = conn.execute(
            "SELECT COUNT(*) FROM gene_relations WHERE gene_id_b = 'parent_live'"
        ).fetchone()[0]
        assert orphan_edges == 0, "CHUNK_OF edges to orphan parent should be removed"
        assert live_edges == 2, "CHUNK_OF edges to live parent must be preserved"
    finally:
        conn.close()


def test_include_children_sweeps_orphan_children(genome_with_orphans: Path):
    res = _run(genome_with_orphans, "--include-children")
    assert res.returncode == 0, res.stderr
    remaining = _gene_ids(genome_with_orphans)

    # Orphan children are gone now
    assert "child_orphan_1" not in remaining
    assert "child_orphan_2" not in remaining
    # Live children preserved
    assert "child_live_1" in remaining
    assert "child_live_2" in remaining
    # Non-path parent still preserved
    assert "parent_nonpath" in remaining


def test_nothing_to_do_is_harmless(tmp_path: Path):
    """Running on a genome with no orphans should print 'nothing to do'
    and return 0."""
    live = tmp_path / "live.md"
    live.write_text("x")
    genome = tmp_path / "clean.db"
    conn = _mk_genome(str(genome))
    _insert_gene(conn, "p", str(live).replace("\\", "/"), is_parent=True)
    conn.commit()
    conn.close()

    res = _run(genome)
    assert res.returncode == 0, res.stderr
    assert "nothing to do" in res.stdout
    assert _gene_ids(genome) == {"p"}


def test_free_form_source_id_never_orphaned(tmp_path: Path):
    """Source IDs without slashes are treated as free-form labels, not paths,
    and must never be considered orphans even when the file doesn't exist."""
    genome = tmp_path / "g.db"
    conn = _mk_genome(str(genome))
    _insert_gene(conn, "p_note", "agent:laude", is_parent=True)
    _insert_gene(conn, "p_bare", "somelabel", is_parent=True)
    conn.commit()
    conn.close()

    res = _run(genome)
    assert res.returncode == 0, res.stderr
    remaining = _gene_ids(genome)
    assert "p_note" in remaining
    assert "p_bare" in remaining
