#!/usr/bin/env bash
# Chat interactif (bloque le terminal jusqu'à /quitter).
# Une seule question sans chat : ./scripts/ask_synthese.sh "votre question"
# Démo non interactive : ./scripts/demo_rag_non_interactif.sh
# Tests auto (sans chat) : ./scripts/run_all_auto_tests.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec .venv/bin/python -m llm_fr.rag_chat "$@"
