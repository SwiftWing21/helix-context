# Helix Cross-Session Restart Protocol

When multiple Claude sessions share a single Helix server, one session can
announce an intentional restart so that observing sessions don't misread the
outage as a crash.

**Since:** helix-context v0.3.0b4

## Signal location

One canonical file, single-slot, always overwritten:

```
~/.helix/shared/signals/server_state.json
```

## Signal schema

```json
{
  "state": "running" | "restarting" | "stopped",
  "actor": "laude",
  "reason": "swapping ribosome model for benchmark",
  "pid": 48213,
  "expected_downtime_s": 30,
  "phase": "shutting_down" | "up",
  "timestamp": 1775980800.123,
  "timestamp_human": "2026-04-09 14:20:00"
}
```

## If you're restarting the server

Before killing the Helix process, POST to the announce endpoint, sleep ~750ms,
then trigger your restart:

```bash
curl -X POST http://127.0.0.1:11437/admin/announce_restart \
     -H "Content-Type: application/json" \
     -d '{
       "reason": "swapping ribosome model",
       "actor": "your_name",
       "expected_downtime_s": 30
     }'
sleep 0.75
# ... now kill and restart Helix ...
```

Or from Python:

```python
from helix_context.bridge import AgentBridge
import time

bridge = AgentBridge()
bridge.announce_restart(
    reason="swapping ribosome model",
    actor="laude",
    expected_downtime_s=30,
)
time.sleep(0.75)  # let filesystem flush + observers see the signal
# ... trigger restart ...
```

The new server process will automatically stamp `state="running"` on startup,
clearing the restart announcement.

## If you're observing the server

When you get `ConnectionRefused` / `background command failed` from Helix,
read the signal file directly — **no HTTP needed**, the server is down:

```bash
cat ~/.helix/shared/signals/server_state.json
```

Interpret the `state` field:

| State | Meaning | What to do |
|-------|---------|------------|
| `running` | Server is up | Retry your request |
| `restarting` | Another session announced a restart | Report to user, wait `expected_downtime_s + ~5`, retry |
| `stopped` | Clean shutdown happened | Restart needed; escalate to user |
| *(missing file)* | Legacy server OR genuine crash | Treat as crash |

### Staleness check

A `restarting` signal older than `timestamp + expected_downtime_s + 15` seconds
means the announced restart never completed. Fall back to normal crash handling.

From Python:

```python
from helix_context.bridge import AgentBridge
import time

bridge = AgentBridge()
result = bridge.read_server_state()

if result is None:
    # No signal file — legacy server or genuine outage
    handle_crash()
else:
    signal, is_stale, age_s = result
    if signal["state"] == "running":
        retry()
    elif signal["state"] == "restarting" and not is_stale:
        print(f"Waiting for {signal['actor']}: {signal['reason']}")
        time.sleep(signal["expected_downtime_s"] - age_s + 2)
        retry()
    else:
        # stale restarting, or stopped, or unknown state
        handle_crash()
```

## Design notes

- **The filesystem is the broker.** The signal file survives any server
  restart because it lives outside the server process. This mirrors MQTT's
  Last Will and Testament pattern, using the filesystem as the persistent
  third party.

- **Atomic writes.** `write_signal` uses write-to-temp + `os.replace` so
  readers never see a partially-written file. Works on POSIX and Windows.

- **Single-slot, always overwritten.** No stale accumulation — the latest
  state is always the only state. No TTL sweep job needed.

- **Belt and suspenders.** The lifespan shutdown hook ALSO stamps a signal
  on clean shutdown, but it does NOT run under `kill -9`. Agents should
  always call `announce_restart` BEFORE killing the process for intentional
  restarts — the shutdown hook is only a fallback for Ctrl+C / OS shutdown.

- **`kill -9` is handled gracefully.** The agent's pre-kill `restarting`
  signal is already on disk; the new process's startup hook overwrites it
  with `running` on respawn.

## Failure modes

| Failure | Behavior |
|---------|----------|
| Agent crashes mid-announcement | Atomic rename → readers see old signal or new, never partial |
| Agent announces but doesn't restart | TTL → observer ignores stale `restarting` after `expected_downtime_s + 15` |
| Server restarts but startup hook fails | Observer's successful Helix call is an implicit "running" regardless of signal |
| Two agents restart simultaneously | Last writer wins. Both wanted a restart; both see it happen |
| Fresh observer session | Reads signal file on first Helix failure — no prior context needed |
| `kill -9` | Agent's pre-kill signal is already on disk; new process stamps `running` on boot |

## Related

- `POST /admin/announce_restart` — convenience HTTP wrapper
- `POST /bridge/signal` — low-level signal write (use if you need custom signal names)
- `GET /bridge/status` — read all active signals (server must be up)
- `bridge.announce_restart()` — Python client method
- `bridge.read_server_state()` — Python observer helper with TTL-aware staleness check
