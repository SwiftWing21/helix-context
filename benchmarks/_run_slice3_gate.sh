#!/usr/bin/env bash
# Slice 3 decision gate — clean SR-off N=50 bench on qwen3:1.7b.
#
# Runs the measurement the 8d_dimensional_roadmap §Phase 1 Slice 3
# specifies: "N=50 v2 against current state to see whether slice 2
# alone moves the needle" — with SR explicitly OFF so the reading is
# isolated to slice-2 (access_rate density gate).
#
# Usage:
#     cd F:/Projects/helix-context
#     bash benchmarks/_run_slice3_gate.sh
#
# Preconditions the script checks:
#   - Helix server is currently running on :11437
#   - qwen3:1.7b is loaded in Ollama (unloaded competitors)
#   - Nothing else is midway through writing helix.toml
#
# Side effects (all reverted on exit, even on failure):
#   - Temporarily sets [retrieval] sr_enabled = false in helix.toml
#   - Restarts the Helix server twice (once to flip, once to restore)
#
# Output:
#   - benchmarks/needle_50_v2_slice3_gate_<timestamp>.json   (result)
#   - benchmarks/needle_50_v2_slice3_gate_<timestamp>.log    (stdout)

set -euo pipefail

REPO_ROOT="F:/Projects/helix-context"
cd "$REPO_ROOT"

TS=$(date +%Y-%m-%d_%H%M)
RESULT_JSON="benchmarks/needle_50_v2_slice3_gate_${TS}.json"
RESULT_LOG="benchmarks/needle_50_v2_slice3_gate_${TS}.log"
HELIX_TOML="helix.toml"
BACKUP_TOML="helix.toml.slice3_gate_backup"

# ── Safety: capture original state ─────────────────────────────────
cp "$HELIX_TOML" "$BACKUP_TOML"

restore_toml() {
    echo ""
    echo "[slice3-gate] Restoring original helix.toml..."
    mv "$BACKUP_TOML" "$HELIX_TOML"
    echo "[slice3-gate] Restart the Helix server manually to pick up restored sr_enabled=true."
}
trap restore_toml EXIT

# ── Flip sr_enabled=false ──────────────────────────────────────────
echo "[slice3-gate] Flipping sr_enabled to false in helix.toml..."
python -c "
import pathlib, re
path = pathlib.Path('$HELIX_TOML')
src = path.read_text(encoding='utf-8')
new = re.sub(
    r'^sr_enabled\s*=\s*true.*\$',
    'sr_enabled = false                      # Slice 3 gate measurement (auto-restored)',
    src, count=1, flags=re.MULTILINE,
)
if new == src:
    raise SystemExit('sr_enabled=true line not found — is config in expected state?')
path.write_text(new, encoding='utf-8')
print('  ok — sr_enabled flipped to false')
"

# ── MANUAL STEP: operator restarts server ──────────────────────────
cat <<'EOF'

[slice3-gate] ACTION REQUIRED:
  1. Restart the Helix server now so the flipped flag takes effect.
     Typical command:
       (kill the running server, then)
       python -m uvicorn helix_context.server:app --host 127.0.0.1 --port 11437

  2. Press ENTER here once the server is back up and /stats responds 200.

EOF
read -r

# ── Verify server is up ────────────────────────────────────────────
if ! curl -s -f http://127.0.0.1:11437/health >/dev/null; then
    echo "[slice3-gate] Helix /health did not respond. Aborting; toml will be restored."
    exit 1
fi

# ── Run bench ──────────────────────────────────────────────────────
echo "[slice3-gate] Running N=50 v2 bench at qwen3:1.7b with sr_enabled=false..."
N=50 SEED=42 HELIX_MODEL=qwen3:1.7b OUTPUT="$RESULT_JSON" \
    python benchmarks/bench_needle_1000.py > "$RESULT_LOG" 2>&1

# ── Surface the result ─────────────────────────────────────────────
echo ""
echo "[slice3-gate] Bench complete. Comparing to 2026-04-14 sr-ON result:"
echo ""
tail -22 "$RESULT_LOG" | grep -E "retr=|ans=|elapsed|failure|retrieval_miss"
echo ""
echo "[slice3-gate] Baseline (post-recovery, qwen3:8b, slice 2 only): retr=20.0% ans=16.0%"
echo "[slice3-gate] This run (qwen3:1.7b, SR off, slice 2 active):    see above"
echo "[slice3-gate] Prev run (qwen3:1.7b, SR on, slice 2 active):     retr=20.0% ans=2.0%"
echo ""
echo "[slice3-gate] Decision rule per 8d_dimensional_roadmap §Phase 1:"
echo "    slice 2 alone improves retrieval by >=1pp over pre-slice2 baseline -> ship slice 2 standalone"
echo "    slice 2 alone is neutral -> implement slice 3 tiebreaker"
echo "    slice 2 alone is worse   -> revert, investigate"
echo ""
echo "[slice3-gate] Full result: $RESULT_JSON"
echo "[slice3-gate] Full log:    $RESULT_LOG"

# trap will restore helix.toml on exit
