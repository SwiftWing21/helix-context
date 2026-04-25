"""
Tests for scripts/backfill_filename_domains.py.

Verifies that the backfill:
  - Inserts correct domain tokens into promoter_index for genes with source_id
  - Is idempotent (double-run produces no duplicates)
  - Skips genes without source_id
  - Dry-run writes nothing
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from helix_context.genome import Genome
from tests.conftest import make_gene


def _run_backfill(db_path: str, dry_run: bool = False) -> int:
    """Import and run the backfill main() against a specific db path."""
    import importlib, types

    # Force reload so args parse fresh each time
    spec = importlib.util.spec_from_file_location(
        "backfill_filename_domains",
        Path(__file__).resolve().parent.parent / "scripts" / "backfill_filename_domains.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    import sys as _sys
    old_argv = _sys.argv
    argv = ["backfill_filename_domains.py", "--db", db_path]
    if dry_run:
        argv.append("--dry-run")
    _sys.argv = argv
    try:
        return mod.main()
    finally:
        _sys.argv = old_argv


def _promoter_domains(conn: sqlite3.Connection, gene_id: str) -> set[str]:
    rows = conn.execute(
        "SELECT tag_value FROM promoter_index WHERE gene_id=? AND tag_type='domain'",
        (gene_id,),
    ).fetchall()
    return {r[0] for r in rows}


class TestBackfillFilenameDomainsScript:
    def test_injects_filename_token_for_gene_with_source_id(self, tmp_path):
        """After backfill, promoter_index gains 'claims' for claims.py gene."""
        db = str(tmp_path / "genome.db")
        genome = Genome(db)
        try:
            gene = make_gene("pass", domains=[])
            gene.source_id = "/repo/helix_context/claims.py"
            gene_id = genome.upsert_gene(gene, apply_gate=False)

            # Pre-condition: 'claims' not yet in promoter_index
            conn = sqlite3.connect(db)
            before = _promoter_domains(conn, gene_id)
            conn.close()
            assert "claims" not in before

            _run_backfill(db)

            conn = sqlite3.connect(db)
            after = _promoter_domains(conn, gene_id)
            conn.close()
        finally:
            genome.close()

        assert "claims" in after, f"'claims' not in promoter_index after backfill: {after}"

    def test_injects_tokenized_parts_for_compound_stem(self, tmp_path):
        """claim_types_handler.py → 'claim', 'types', 'handler' all injected."""
        db = str(tmp_path / "genome.db")
        genome = Genome(db)
        try:
            gene = make_gene("pass", domains=[])
            gene.source_id = "/repo/helix_context/claim_types_handler.py"
            gene_id = genome.upsert_gene(gene, apply_gate=False)
            _run_backfill(db)
            conn = sqlite3.connect(db)
            after = _promoter_domains(conn, gene_id)
            conn.close()
        finally:
            genome.close()

        assert "claim" in after
        assert "types" in after
        assert "handler" in after
        assert "claim_types_handler" in after

    def test_skips_gene_without_source_id(self, tmp_path):
        """Genes with NULL source_id get no new promoter_index rows."""
        db = str(tmp_path / "genome.db")
        genome = Genome(db)
        try:
            gene = make_gene("pass", domains=[])
            gene.source_id = None
            gene_id = genome.upsert_gene(gene, apply_gate=False)

            conn = sqlite3.connect(db)
            before = _promoter_domains(conn, gene_id)
            conn.close()

            _run_backfill(db)

            conn = sqlite3.connect(db)
            after = _promoter_domains(conn, gene_id)
            conn.close()
        finally:
            genome.close()

        assert after == before

    def test_idempotent_double_run(self, tmp_path):
        """Running backfill twice produces no duplicate rows."""
        db = str(tmp_path / "genome.db")
        genome = Genome(db)
        try:
            gene = make_gene("pass", domains=[])
            gene.source_id = "/repo/helix_context/claims.py"
            gene_id = genome.upsert_gene(gene, apply_gate=False)
            _run_backfill(db)
            _run_backfill(db)
            conn = sqlite3.connect(db)
            rows = conn.execute(
                "SELECT tag_value, COUNT(*) as cnt FROM promoter_index "
                "WHERE gene_id=? AND tag_type='domain' GROUP BY tag_value HAVING cnt > 1",
                (gene_id,),
            ).fetchall()
            conn.close()
        finally:
            genome.close()

        assert rows == [], f"Duplicate rows after double run: {rows}"

    def test_dry_run_writes_nothing(self, tmp_path):
        """--dry-run leaves promoter_index unchanged."""
        db = str(tmp_path / "genome.db")
        genome = Genome(db)
        try:
            gene = make_gene("pass", domains=[])
            gene.source_id = "/repo/helix_context/claims.py"
            gene_id = genome.upsert_gene(gene, apply_gate=False)

            conn = sqlite3.connect(db)
            before = _promoter_domains(conn, gene_id)
            conn.close()

            _run_backfill(db, dry_run=True)

            conn = sqlite3.connect(db)
            after = _promoter_domains(conn, gene_id)
            conn.close()
        finally:
            genome.close()

        assert after == before, f"Dry-run wrote rows: {after - before}"
