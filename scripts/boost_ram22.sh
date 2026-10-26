#!/usr/bin/env bash
# Bascule ram22 : batch 1024, ~22 Go (entre ram15 rapide et ram55 lent).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "Boost ram22 — batch 1024, ~22 Go RAM..."
"$ROOT/scripts/stop_all_training.sh"
sleep 2
"$ROOT/scripts/install_watchdog.sh"
echo "Lancé depuis $(ls runs/legal_large_25k_ram15/best.pt 2>/dev/null && echo 'ram15/best.pt' || echo 'checkpoint')."
echo "Suivi : ./scripts/progress.sh --watch"
