#!/usr/bin/env bash
# Pose une question en mode synthèse (extraits officiels, sans génération).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
QUESTION="${*:-}"
if [ -z "$QUESTION" ]; then
  echo "Usage: ./scripts/ask_synthese.sh votre question ici" >&2
  exit 1
fi
exec .venv/bin/python -m llm_fr.ask_synthese_cli "$QUESTION"
