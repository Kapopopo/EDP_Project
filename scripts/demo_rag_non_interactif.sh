#!/usr/bin/env bash
# Démo rapide sans ouvrir le chat interactif (une commande = tout s'exécute).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

echo "=== Statut ==="
./scripts/rag_status.sh

echo ""
echo "=== Synthèse (secret professionnel) ==="
./scripts/ask_synthese.sh "secret professionnel avocat"

echo ""
echo "=== Synthèse JSON (honoraires) ==="
"$PY" -m llm_fr.ask_synthese_cli "honoraires avocat" --json | head -n 40

echo ""
echo "=== Audit (1 ligne test) ==="
export RAG_AUDIT_LOG="${RAG_AUDIT_LOG:-logs/rag_audit.jsonl}"
"$PY" -m llm_fr.ask_synthese_cli "test audit démo" --no-audit
"$PY" - <<'PY'
import os
from llm_fr.synthesis_audit import append_audit_record
append_audit_record("test audit démo", [{"domain": "legifrance", "title": "demo", "score": 1.0}], index_path="data/rag/index_legal")
print(f"Écrit dans {os.environ['RAG_AUDIT_LOG']}")
PY

echo ""
echo "Pour le chat interactif (bloquant) : ./scripts/rag.sh --sources"
echo "Quitter le chat : /quitter"
