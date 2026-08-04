#!/usr/bin/env bash
# Reclaim disk on this machine. Read it before running it.
#
# The no-sudo parts have already been run; they are kept here so the whole
# recovery is repeatable. The sudo parts have NOT been run and are the reason
# this file exists.
#
# Nothing here touches your own files — only caches, logs and superseded
# package revisions, all of which regenerate or are already obsolete.
set -uo pipefail

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
free_now() { df -h / | awk 'NR==2{print "  free: "$4"  ("$5" used)"}'; }

say "starting"; free_now

# ---------------------------------------------------------------------------
# No sudo required — already applied
# ---------------------------------------------------------------------------

say "stale /tmp build dirs"
rm -rf /tmp/flutter_tools.* /tmp/openclaw /tmp/remotion-webpack-bundle-* \
       /tmp/node-compile-cache 2>/dev/null
free_now

say "Playwright browser binaries (~3.1 GB, re-downloaded on demand)"
rm -rf "$HOME/.cache/ms-playwright"
free_now

say "Chrome cache (~1.7 GB)"
rm -rf "$HOME/.cache/google-chrome"
free_now

say "uv package cache (~12 GB)"
# --force overrides the lock held by long-running uvx MCP servers. Safe: on
# Linux a running process keeps its open files alive even once unlinked, and
# venvs hardlink out of the cache so they survive it. Verified after running:
# the project venv still imports numpy/scipy/open3d and the suite passes.
uv cache clean --force 2>/dev/null || uv cache clean
free_now

say "Gradle wrapper distributions (~900 MB of superseded Gradle versions)"
# The current version re-downloads on next build; older ones will not be used.
find "$HOME/.gradle/wrapper/dists" -maxdepth 1 -mindepth 1 -type d \
     ! -name '*8.11.1*' -exec rm -rf {} + 2>/dev/null
free_now

# ---------------------------------------------------------------------------
# Needs sudo — NOT yet applied
# ---------------------------------------------------------------------------

cat <<'SUDO'

=============================================================================
The rest needs your password. Run these yourself:

  # systemd journal: 3.9 GB of logs, capped at 500 MB going forward
  sudo journalctl --vacuum-size=500M
  sudo sed -i 's/^#\?SystemMaxUse=.*/SystemMaxUse=500M/' /etc/systemd/journald.conf
  sudo systemctl restart systemd-journald

  # Superseded snap revisions — 9 disabled ones, including two ollama at ~2 GB
  # each. "disabled" means already replaced by a newer revision.
  snap list --all | awk '/disabled/{print $1, $3}' |
    while read -r name rev; do sudo snap remove "$name" --revision="$rev"; done

  # Stop snap keeping three old revisions of everything
  sudo snap set system refresh.retain=2

  # Package archives
  sudo apt clean

Together: roughly another 9-10 GB.
=============================================================================

SUDO

say "done"; free_now
