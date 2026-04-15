"""Emit a helix participant heartbeat with optional presence state.

Wraps POST /sessions/{participant_id}/heartbeat so each agent persona
(laude, raude, taude, ...) can publish what it's currently doing to a
retrievable presence gene — other personas see the state through their
normal /context retrievals.

Usage:
    # Simple ping (unchanged behavior — just refreshes TTL)
    python scripts/helix_heartbeat.py laude

    # Publish presence state
    python scripts/helix_heartbeat.py laude \\
        --party swift_wing21 \\
        --focus "PWPC Phase 1 followup" \\
        --blocked-on "batman access" \\
        --in-flight "heartbeat endpoint" \\
        --in-flight "lockstep test" \\
        --notes "See docs/collab/comms/LOCKSTEP_TEST.md"

    # Read git state automatically for last_commit_hash
    python scripts/helix_heartbeat.py laude --git

The participant must already be registered (POST /sessions/register).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


DEFAULT_HELIX_URL = "http://localhost:11437"


def read_git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            timeout=5,
        )
        return out.decode("utf-8").strip() or None
    except Exception:
        return None


def post_heartbeat(
    base_url: str, participant_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/sessions/{participant_id}/heartbeat"
    data = json.dumps(body).encode("utf-8") if body else b""
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("participant_id",
                    help="Participant id (e.g. laude / raude / taude / UUID)")
    ap.add_argument("--url", default=DEFAULT_HELIX_URL,
                    help=f"Helix base URL (default: {DEFAULT_HELIX_URL})")
    ap.add_argument("--handle", default=None,
                    help="Display handle — defaults to participant_id")
    ap.add_argument("--party", default=None, help="party_id")
    ap.add_argument("--focus", default=None, help="One-line current focus")
    ap.add_argument("--blocked-on", action="append", default=[],
                    help="Repeatable. Thing blocking progress")
    ap.add_argument("--in-flight", action="append", default=[],
                    help="Repeatable. Task currently in progress")
    ap.add_argument("--commit", default=None,
                    help="Last commit hash (use --git to auto-read)")
    ap.add_argument("--git", action="store_true",
                    help="Auto-populate --commit from `git rev-parse --short HEAD`")
    ap.add_argument("--notes", default=None, help="Free-form markdown")
    args = ap.parse_args()

    body: dict[str, Any] = {}
    if args.handle:
        body["handle"] = args.handle
    if args.party:
        body["party_id"] = args.party
    if args.focus:
        body["current_focus"] = args.focus
    if args.blocked_on:
        body["blocked_on"] = args.blocked_on
    if args.in_flight:
        body["in_flight"] = args.in_flight
    if args.commit:
        body["last_commit_hash"] = args.commit
    elif args.git:
        h = read_git_head()
        if h:
            body["last_commit_hash"] = h
    if args.notes:
        body["notes"] = args.notes

    try:
        result = post_heartbeat(args.url, args.participant_id, body)
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8"))
        except Exception:
            detail = {"error": str(exc)}
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Heartbeat request failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
