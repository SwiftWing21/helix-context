# Deploy templates for `helix-launcher`

Service / daemon templates for running the helix launcher in the
background as a system-managed process. The launcher itself works
fine as a foreground app — these templates are for "set it and forget
it" deployments.

| Platform | File | Notes |
|---|---|---|
| Linux | [`systemd/helix-launcher.service`](systemd/helix-launcher.service) | User-level systemd unit. Edit `ExecStart` and copy to `~/.config/systemd/user/`. |
| macOS | [`launchd/com.swiftwing21.helix-launcher.plist`](launchd/com.swiftwing21.helix-launcher.plist) | Per-user LaunchAgent. Edit `ProgramArguments` and copy to `~/Library/LaunchAgents/`. |
| Windows | [`windows/README.md`](windows/README.md) | NSSM-based service install recipe. Manual setup via NSSM GUI. |

These templates are deliberately **not auto-installed**. Each platform
has its own conventions for paths, permissions, and user vs system
scope; a one-size-fits-all installer would inevitably guess wrong.
A future `helix-launcher install-service` CLI may automate the most
common cases, but template-first is the right starting point.

## What the launcher does once it's running

- Spawns and supervises one `helix-context` server child process
- Serves the dashboard at `http://127.0.0.1:11438/`
- Restarts the helix child via the announce-then-kill protocol when
  asked
- Adopts an already-running helix from the state file at
  `~/.helix/launcher/state.json` if the launcher itself is restarted

See [`docs/LAUNCHER.md`](../docs/LAUNCHER.md) for the full architecture.
