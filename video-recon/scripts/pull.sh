#!/usr/bin/env bash
# Pull the newest recording (or the one named) off the phone into data/recordings/.
#   video-recon/scripts/pull.sh            # newest
#   video-recon/scripts/pull.sh rec_20260902-101500
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ADB="${ADB:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}/platform-tools/adb}"
REMOTE=/sdcard/Android/data/com.recon.videoscan/files/recordings
DEST="$ROOT/data/recordings"
mkdir -p "$DEST"
name="${1:-}"
if [ -z "$name" ]; then
    name=$("$ADB" shell "ls -1 $REMOTE" | tr -d '\r' | sort | tail -1)
    [ -n "$name" ] || { echo "no recordings on the phone" >&2; exit 1; }
fi
echo "pulling $name"
"$ADB" pull "$REMOTE/$name" "$DEST/"
ls -la "$DEST/$name"
