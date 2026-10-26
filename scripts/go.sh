#!/usr/bin/env bash
# Point d'entrée unique : lance l'entraînement + pipeline auto 3 phases.
# Une commande, puis dormir — tout s'enchaîne jusqu'au checkpoint refine2.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="runs/legal_large_25k_ram15"
LOCK="$OUT/training.lock"
PIPE_PID_FILE="runs/fastest.pid"

echo "══════════════════════════════════════════════════════════"
echo "  DATTY AI — Lancement automatique vers perte minimale"
echo "══════════════════════════════════════════════════════════"

# 1. Entraînement phase 1 (si pas déjà actif)
train_running=0
if [[ -f "$LOCK" ]]; then
  pid="$(tr -d '[:space:]' < "$LOCK" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    train_running=1
    echo "✓ Phase 1 déjà en cours (PID $pid)"
  else
    rm -f "$LOCK"
  fi
fi

if [[ "$train_running" -eq 0 ]]; then
  echo "▶ Démarrage phase 1 (ram15)..."
  "$ROOT/scripts/start_ram15.sh"
else
  echo "  (phase 1 laissée telle quelle)"
fi

# 2. Pipeline 3 phases (attend phase 1 puis enchaîne refine + refine2)
pipeline_running=0
if [[ -f "$PIPE_PID_FILE" ]]; then
  ppid="$(tr -d '[:space:]' < "$PIPE_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$ppid" ]] && kill -0 "$ppid" 2>/dev/null; then
    pipeline_running=1
    echo "✓ Pipeline auto déjà actif (PID $ppid)"
  else
    rm -f "$PIPE_PID_FILE"
  fi
fi

if [[ "$pipeline_running" -eq 0 ]] && ! launchctl print "gui/$(id -u)/com.dattyai.fastest" 2>/dev/null | grep -q "state = running"; then
  echo "▶ Installation watchdog (survit fermeture terminal)..."
  "$ROOT/scripts/install_watchdog.sh"
else
  echo "✓ Pipeline / watchdog déjà actif"
fi

echo
echo "Tout est lancé. Suivi :"
echo "  ./scripts/progress.sh --watch"
echo "  tail -f runs/fastest.log"
echo "══════════════════════════════════════════════════════════"
