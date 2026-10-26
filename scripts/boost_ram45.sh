#!/usr/bin/env bash
# Reprend ram15 (rapide) — NE PAS utiliser ram55 (trop lent sur CPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "Reprise ram15 (rapide, ~11 s/step)..."
"$ROOT/scripts/stop_all_training.sh"
sleep 2
"$ROOT/scripts/install_watchdog.sh"
echo "Lancé. Suivi : ./scripts/progress.sh --watch"
