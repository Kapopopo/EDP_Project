#!/usr/bin/env bash
# Arrête le CPU lent, lance refine MPS (GPU) depuis le meilleur checkpoint.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "══════════════════════════════════════════════════════════"
echo "  ACCÉLÉRATION MPS — GPU M1 Max"
echo "  Saute le reste de la phase 1 (déjà à 2,999)"
echo "  Lance refine + refine2 sur GPU"
echo "══════════════════════════════════════════════════════════"

"$ROOT/scripts/stop_all_training.sh"
sleep 2

if ! .venv/bin/python -c "import torch; assert torch.backends.mps.is_available()"; then
  echo "ERREUR : MPS indisponible"
  exit 1
fi

"$ROOT/scripts/install_watchdog.sh"
echo "Pipeline MPS lancé. Suivi : ./scripts/progress.sh --watch"
