# Memory-Watch SessionStart Hook + Per-Persona Hook Config

**Status:** Design sketch, 2026-04-17. Not on the current roadmap. Captured
because the mechanism is cheap and the pattern generalizes to any hook
behavior. First use case is memory folder archival; the real insight is
that launcher `.bat` files double as per-persona config surface.

Related memory note: `project_helix_personas_as_config_surface.md`.

---

## The first use case — memory-watch

Shared-memory folder (`C:\Users\max\.claude\shared-memory\` after the
2026-04-17 junction move) grows over time. At ~60 files the hot shelf
gets noisy and MEMORY.md index approaches its 200-line cap. We want a
**low-friction heads-up** when the count crosses a threshold, without
gating every session.

## The hook

Claude Code `SessionStart` hooks run once per new conversation and can
inject text into the conversation context via stdout (captured as a
`<system-reminder>`). That's the right shape for this — visible to both
human and agent, blocks nothing.

```bash
# ~/.claude/hooks/memory-watch.sh
set -eu

MEMORY_DIR="${HELIX_MEMORY_DIR:-$HOME/.claude/shared-memory}"
THRESHOLD="${HELIX_MEMORY_THRESHOLD:-60}"
MODE="${HELIX_MEMORY_MODE:-nod}"     # nod | confirm | block

# Count hot-shelf markdown files (exclude MEMORY.md and archive/)
HOT_COUNT=$(find "$MEMORY_DIR" -maxdepth 1 -name "*.md" ! -name "MEMORY.md" \
  | wc -l | tr -d ' ')

[ "$HOT_COUNT" -le "$THRESHOLD" ] && exit 0

# Over threshold — produce a reminder
echo "[memory-watch] HOT shelf at $HOT_COUNT files (target: 40-60)."

# Archive candidates: project_* and session_* not linked in MEMORY.md index
INDEX="$MEMORY_DIR/MEMORY.md"
echo "Candidates for archive (not linked in MEMORY.md):"
for f in "$MEMORY_DIR"/project_*.md "$MEMORY_DIR"/session_*.md; do
  [ -f "$f" ] || continue
  name=$(basename "$f")
  if ! grep -q "($name)" "$INDEX" 2>/dev/null; then
    echo "  - $name"
  fi
done | head -10

case "$MODE" in
  nod)     echo "Say 'memory archive' to sweep." ;;
  confirm) echo "Archive suggested. Agent will pause before running sweep." ;;
  block)   echo "Destructive sweeps blocked until you say 'memory archive unlock'." ;;
esac
```

## Three modes — knob for alignment intensity

| Mode    | Behavior                                                           |
|---------|--------------------------------------------------------------------|
| `nod`   | Reminder injected. Agent sees it, may mention it. No block.        |
| `confirm` | Reminder + agent instructed to ask before running archive op.    |
| `block` | Archive operation blocked via PreToolUse hook until unlock said.   |

Default: `nod`. Lowest friction, still keeps the decision visible.

## Per-persona config via launcher .bat

The mode (and, generalizing, any hook toggle) is an env var. Each
persona's launcher .bat sets its own:

```bat
REM start-helix-tray-raude.bat
set HELIX_AGENT=raude
set HELIX_MEMORY_MODE=nod

REM start-helix-tray-laude.bat
set HELIX_AGENT=laude
set HELIX_MEMORY_MODE=confirm

REM start-helix-tray-taude.bat
set HELIX_AGENT=taude
set HELIX_MEMORY_MODE=block
```

Same settings.json, same hook script, different behavior per agent
because the env flows through from launch.

## Federation-layer precedence

The 4-layer identity model (org/party/participant/agent) maps cleanly:

| Layer          | Where the config lives             | Example                             |
|----------------|------------------------------------|-------------------------------------|
| Org (team)     | Shared repo `.claude/settings.json`| "This org: block destructive ops"   |
| Party (device) | User-global `~/.claude/settings.json` | "On this laptop default to nod"  |
| Participant    | Memory file read by hook           | "Max always wants confirm over nod"|
| Agent          | Per-launcher `.bat` env var        | "Raude nod, Laude confirm"          |

Precedence order is the hook script's to decide. Simplest rule: org
strictness wins, agent preference loses — but agents can be stricter
than org if they want.

## What else this pattern unlocks (beyond memory-watch)

Any hook-driven behavior can be toggled per layer using the same
env-var-in-.bat mechanism:

- **Destructive-op gating** — `HELIX_DESTRUCTIVE_GATE=warn|block`
- **Auto-commit message style** — `HELIX_COMMIT_STYLE=terse|descriptive`
- **Audio notification style** — `HELIX_AUDIO_STYLE=subtle|focus|none`
- **Context-hygiene pressure** — `HELIX_CTX_THRESHOLD=25|40|60`
- **Telemetry sampling** — `HELIX_OTEL_SAMPLER_RATIO=0.1|1.0`

The launcher .bat files aren't just "which pane I am" — they're the
**personality-config surface** for each agent. Laude and Raude can
differ by more than name.

## Archive sweep behavior (when mode triggers run)

When the user says "memory archive" (or similar), the sweep script:

1. For each `project_*.md` or `session_*.md` NOT linked in MEMORY.md:
   a. Grep the note for a `superseded_by:` frontmatter field
   b. Or check filename date → compare to current date
   c. Move to `archive/YYYY-Qn/` based on last-modified time
2. Update MEMORY.md with a single "See archive/" line per moved file
3. Print a summary: "Archived N files, HOT shelf now at M."

Never silently archive — always print before the move, error-out on
anything that looks load-bearing.

## Implementation cost

- Hook script: ~30 LOC bash
- settings.json entry: 5 lines
- One-time archive sweep script: ~50 LOC
- Per-persona .bat templates: copy-paste

Total ~90 LOC + ~15 min wiring. Cheap when you're ready.

## Why not now

- Helix sharding, layered fingerprints, and retrieval rank bottleneck
  all sit ahead of this on priority
- Memory folder is at 34 files today — no bloat yet
- Pattern only pays off at ~60+ files, estimated 2-3 months out

Revisit trigger: file count > 55, or first time someone asks "how do I
make Laude more cautious than Raude."

## Out of scope for this doc

- The actual archive sweep tool (companion doc if/when we build it)
- How memory notes get a `superseded_by` field (minor frontmatter
  addition)
- Cross-machine memory federation (shared-memory/ + Dropbox-style sync
  is a separate direction)
