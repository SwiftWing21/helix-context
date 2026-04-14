#!/bin/bash
# Sequential n=1000 sike sweep — 2026-04-14
# All 3 benches against fresh snapshot post Sprint 1/2/4/5A + 192K-edge backfill.
# Helix is assumed running on :11437 with OTel ON (so the dashboard captures
# the load profile alongside the bench results).

set -u  # don't 'set -e' — we want all 3 to attempt even if one errors
SNAPSHOT="F:/Projects/helix-context/genome-bench-2026-04-14.db"
TS=$(date +%Y-%m-%d_%H%M)
LOG_DIR="benchmarks"
cd /f/Projects/helix-context

export GENOME_DB="$SNAPSHOT"
export HELIX_MODEL="qwen3:4b"

echo "==============================================="
echo "  n=1000 sike sweep starting at $(date)"
echo "  snapshot: $SNAPSHOT"
echo "  model:    $HELIX_MODEL"
echo "  helix:    http://127.0.0.1:11437"
echo "==============================================="

echo ""
echo "=========================================="
echo "  TEST A: bench_needle_1000.py (canonical)"
echo "=========================================="
python benchmarks/bench_needle_1000.py 2>&1 | tee "$LOG_DIR/n1000_A_needle_${TS}.log"
echo "TEST A finished at $(date)"

echo ""
echo "=========================================="
echo "  TEST B: bench_dimensional_lock.py N=200"
echo "=========================================="
N=200 SEED=42 python benchmarks/bench_dimensional_lock.py 2>&1 | tee "$LOG_DIR/n1000_B_dim_lock_${TS}.log"
echo "TEST B finished at $(date)"

echo ""
echo "=========================================="
echo "  TEST C: bench_rag_vs_sike_tokens.py N=200"
echo "=========================================="
N=200 python benchmarks/bench_rag_vs_sike_tokens.py 2>&1 | tee "$LOG_DIR/n1000_C_rag_vs_sike_${TS}.log"
echo "TEST C finished at $(date)"

echo ""
echo "==============================================="
echo "  SWEEP COMPLETE at $(date)"
echo "  results in $LOG_DIR/n1000_{A,B,C}_*_${TS}.log"
echo "==============================================="
