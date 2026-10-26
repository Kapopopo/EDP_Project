#!/usr/bin/env bash
# Affiche le profil matériel et les limites adaptatives.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
.venv/bin/python -m llm_fr.hardware --save
