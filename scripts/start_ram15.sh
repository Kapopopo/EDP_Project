#!/usr/bin/env bash
# OBSOLÈTE — utiliser ./scripts/go.sh ou ./scripts/boost_ram45.sh
# Refuse de lancer si un autre entraînement tourne déjà.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

active="$(.venv/bin/python -c "from llm_fr.runtime import active_training; print(active_training() or '')")"
if [[ -n "$active" ]]; then
  echo "Refusé : entraînement déjà actif ($active)."
  echo "Un seul à la fois. Voir : ./scripts/progress.sh --watch"
  exit 1
fi

echo "⚠ start_ram15.sh est obsolète. Utilise : ./scripts/go.sh"
exec "$ROOT/scripts/go.sh"
