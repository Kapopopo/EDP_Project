#!/usr/bin/env bash
# Installe un watchdog launchd qui surveille et enchaîne les 3 phases.
# Survit à la fermeture du terminal et aux sessions Cursor.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LABEL="com.dattyai.fastest"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string>
    <string>-i</string>
    <string>${ROOT}/scripts/fastest_to_best.sh</string>
    <string>--foreground</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT}</string>
  <key>StandardOutPath</key>
  <string>${ROOT}/runs/fastest.log</string>
  <key>StandardErrorPath</key>
  <string>${ROOT}/runs/fastest.log</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl enable "gui/$(id -u)/${LABEL}"
launchctl kickstart -k "gui/$(id -u)/${LABEL}"

echo "Watchdog installé : ${LABEL}"
echo "  Log : tail -f ${ROOT}/runs/fastest.log"
echo "  Stop : launchctl bootout gui/$(id -u)/${LABEL}"
