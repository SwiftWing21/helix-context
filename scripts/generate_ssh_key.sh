#!/usr/bin/env bash
# Generate an ed25519 SSH key for the Celestia × Helix collab, with a
# passphrase (deviating from the onboarding doc's "no passphrase"
# default — the private key lives on your laptop, not on the compute
# instance, so encrypting at rest matters).
#
# After generation:
#   - prints the PUBLIC key for sharing with Todd via Discord DM
#   - reminds you where the private key lives (never share this)
#   - optionally loads the key into ssh-agent for the session
#
# Windows (Git Bash) and macOS/Linux safe.

set -euo pipefail

KEY_NAME="${KEY_NAME:-helix_collab_ed25519}"
KEY_PATH="${HOME}/.ssh/${KEY_NAME}"
COMMENT="${COMMENT:-max-helix-collab}"

# ── Pre-flight ───────────────────────────────────────────────────────

if ! command -v ssh-keygen >/dev/null 2>&1; then
    echo "error: ssh-keygen not found. Install OpenSSH client." >&2
    exit 1
fi

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh" 2>/dev/null || true  # no-op on Windows

if [[ -f "${KEY_PATH}" ]]; then
    echo "error: key already exists at ${KEY_PATH}" >&2
    echo "       refusing to overwrite. Options:" >&2
    echo "         - use the existing key (cat ${KEY_PATH}.pub to share)" >&2
    echo "         - remove the existing key first if truly stale" >&2
    echo "         - set KEY_NAME=<other_name> to generate a second key" >&2
    exit 2
fi

# ── Generate ─────────────────────────────────────────────────────────

echo "Generating ed25519 key at ${KEY_PATH}"
echo "You'll be prompted for a passphrase. Use something strong — this"
echo "protects the private key if your laptop is ever compromised."
echo ""

ssh-keygen -t ed25519 -C "${COMMENT}" -f "${KEY_PATH}"

# ── Post-check ───────────────────────────────────────────────────────

if [[ ! -f "${KEY_PATH}" || ! -f "${KEY_PATH}.pub" ]]; then
    echo "error: key files not created as expected" >&2
    exit 3
fi

chmod 600 "${KEY_PATH}" 2>/dev/null || true
chmod 644 "${KEY_PATH}.pub" 2>/dev/null || true

# ── Optional: add to ssh-agent ──────────────────────────────────────

if command -v ssh-add >/dev/null 2>&1; then
    echo ""
    read -p "Load into ssh-agent now? (Y/n) " -n 1 -r REPLY
    echo ""
    if [[ ! ${REPLY} =~ ^[Nn]$ ]]; then
        # On Windows Git Bash, ssh-agent may need to be started first.
        if ! ssh-add -l >/dev/null 2>&1; then
            echo "starting ssh-agent..."
            eval "$(ssh-agent -s)" >/dev/null
        fi
        ssh-add "${KEY_PATH}"
    fi
fi

# ── Report ──────────────────────────────────────────────────────────

cat <<EOF

══════════════════════════════════════════════════════════════════════
  DONE. Here's what to do next:
══════════════════════════════════════════════════════════════════════

  1. Send the PUBLIC key (below) to Todd via Discord DM.
     It's safe to share in any channel — that's what public keys are for.

     ───────────── PUBLIC KEY (share this) ─────────────
$(cat "${KEY_PATH}.pub")
     ───────────────────────────────────────────────────

  2. The PRIVATE key lives at:
       ${KEY_PATH}

     NEVER share this. NEVER paste it anywhere. NEVER commit it.
     If you ever suspect it's been exposed, regenerate immediately.

  3. When Todd confirms the key is added to vast.ai instances,
     connect with:

       ssh -i ${KEY_PATH} -p <PORT> root@<HOST>

══════════════════════════════════════════════════════════════════════
EOF
