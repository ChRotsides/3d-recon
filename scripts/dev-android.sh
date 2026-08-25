#!/usr/bin/env bash
# Build, install and relaunch the Android pose logger on a wirelessly attached
# phone, and keep doing it whenever a source file changes.
#
# The build box has no USB link to the phone, so adb reaches it over the LAN via
# Android's wireless debugging (Android 11+). Pairing is a one-off; the connect
# port changes on every reboot of the phone.
#
#   scripts/dev-android.sh pair 192.168.0.42:37155 483619
#   scripts/dev-android.sh connect 192.168.0.42:5555
#   scripts/dev-android.sh watch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANDROID="$ROOT/clients/android"
SDK="${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}"
ADB="$SDK/platform-tools/adb"
PKG=com.recon.poselogger
ACTIVITY="$PKG/.MainActivity"

# Java 21 on this box is JRE-only; Gradle needs the 17 JDK (clients/android/README.md).
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"
export ANDROID_HOME="$SDK"
export ANDROID_NDK_HOME="${ANDROID_NDK_HOME:-$(ls -d "$SDK"/ndk/* 2>/dev/null | tail -1)}"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# One device is the normal case; pin it so gradle and adb agree when it is not.
# A wirelessly attached phone shows up twice — once as ip:port and once under its
# mDNS service name — so prefer the address, which is the same transport either way.
pick_device() {
    [ -n "${ANDROID_SERIAL:-}" ] && return
    local devices
    devices=$("$ADB" devices | awk '$2=="device" {print $1}')
    [ -z "$devices" ] && die "no device. Run: $0 connect <ip:port>   (see '$0 help')"
    export ANDROID_SERIAL="$(echo "$devices" | grep -m1 ':[0-9]*$' || echo "$devices" | head -1)"
    [ "$(echo "$devices" | grep -cv '_tcp$')" -gt 1 ] && say "several devices; using $ANDROID_SERIAL"
    return 0
}

cmd_pair() {   # one-off: phone shows a *different* port and a 6-digit code
    [ $# -eq 2 ] || die "usage: $0 pair <ip:pairing-port> <code>"
    "$ADB" pair "$1" "$2"
    say "now connect on the port from the main Wireless debugging screen:"
    say "  $0 connect ${1%%:*}:<port>"
}

# The connect port is randomised on every phone reboot, but a paired phone
# advertises it over mDNS, so with no argument we just go and find it.
cmd_connect() {
    local target="${1:-}"
    if [ -z "$target" ]; then
        say "looking for a paired phone on the network"
        target=$("$ADB" mdns services | awk '/_adb-tls-connect/ {print $3; exit}')
        [ -z "$target" ] && die "none advertised. Is Wireless debugging on? Else: $0 connect <ip:port>"
    fi
    "$ADB" connect "$target"
    "$ADB" devices -l
}

cmd_core() {   # only needed when core/ changes; the .so is committed
    say "cargo ndk -> jniLibs"
    ( cd "$ROOT/core" && cargo ndk -t arm64-v8a -t armeabi-v7a \
        -o "$ANDROID/app/src/main/jniLibs" build --release )
}

cmd_run() {
    pick_device
    say "installDebug -> $ANDROID_SERIAL"
    ( cd "$ANDROID" && ./gradlew --console=plain -q installDebug )
    "$ADB" shell am force-stop "$PKG" || true
    "$ADB" shell am start -n "$ACTIVITY" >/dev/null
    say "running"
}

# No true hot reload exists for a native Android app outside Android Studio's
# Apply Changes, so this is reinstall-and-relaunch. Kotlin-only edits land in a
# few seconds; a core/ edit pays for a cargo-ndk build first.
cmd_watch() {
    local stamp; stamp="$(mktemp)"
    local watched=("$ANDROID/app/src" "$ROOT/core/src")
    cmd_run
    say "watching ${watched[*]} — ^C to stop"
    while true; do
        sleep 1
        local changed
        changed=$(find "${watched[@]}" -type f -newer "$stamp" -print -quit)
        [ -z "$changed" ] && continue
        touch "$stamp"
        say "changed: ${changed#$ROOT/}"
        case "$changed" in "$ROOT/core/"*) cmd_core ;; esac
        cmd_run || say "build failed — still watching"
    done
}

cmd_logs() { pick_device; "$ADB" logcat --pid="$("$ADB" shell pidof -s "$PKG")"; }

cmd_help() { awk 'NR>1 && /^#/ {sub(/^# ?/,""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; cat <<EOF

commands:
  pair <ip:port> <code>   one-off, from the phone's pairing dialog
  connect [ip:port]       after each phone reboot; finds it over mDNS if omitted
  run                     build, install, relaunch, once
  watch                   the same, on every source change
  core                    rebuild librecon_core.so for both ABIs
  logs                    logcat for this app only
EOF
}

case "${1:-help}" in
    pair|connect|core|run|watch|logs|help) c="$1"; shift; "cmd_$c" "$@" ;;
    *) die "unknown command: $1" ;;
esac
