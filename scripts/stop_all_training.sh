#!/usr/bin/env bash
# Arrête tous les entraînements actifs (un seul autorisé normalement).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LABEL="com.dattyai.fastest"

launchctl bootout "gui/$(id -u)/${LABEL}" 2>/dev/null || true
rm -f runs/fastest.pid runs/.global_training.lock

stopped=0
for lock in runs/*/training.lock; do
  [[ -f "$lock" ]] || continue
  run="$(dirname "$lock")"
  pid="$(tr -d '[:space:]' < "$lock" 2>/dev/null || true)"
  touch "$run/STOP" 2>/dev/null || true
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    echo "Arrêt $(basename "$run") PID $pid..."
    kill "$pid" 2>/dev/null || true
    stopped=1
  fi
  rm -f "$lock" "$run/STOP"
done

# Tuer les python train orphelins
while read -r pid; do
  echo "Arrêt processus orphelin PID $pid"
  kill "$pid" 2>/dev/null || true
  stopped=1
done < <(pgrep -f "llm_fr\.train" 2>/dev/null || true)

sleep 2
if [[ "$stopped" -eq 0 ]]; then
  echo "Aucun entraînement actif."
else
  echo "Tous les entraînements arrêtés."
fi
