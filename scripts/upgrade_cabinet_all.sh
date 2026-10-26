#!/usr/bin/env bash
# Met à jour corpus cabinet + évaluations (sans rebuild index 5M).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

echo "=== 1/4 Index cabinet (RIN, mentions légales) ==="
./scripts/build_cabinet_index.sh

echo ""
echo "=== 2/4 Index sémantique curé (si index principal présent) ==="
if [ -f data/rag/index_legal/manifest.json ]; then
  ./scripts/build_semantic_curated.sh data/rag/index_legal || true
else
  echo "Index principal absent — étape sémantique ignorée."
fi

echo ""
echo "=== 3/5 Tests unitaires RAG ==="
"$PY" -m unittest \
  tests.test_legal_curated \
  tests.test_synthesis_enriched \
  tests.test_synthesis_filters \
  tests.test_hybrid_retrieval \
  tests.test_cabinet_merge \
  tests.test_query_expansion \
  tests.test_llm_redact \
  tests.test_retrieval_diagnostics \
  -q

echo ""
echo "=== 4/5 Smoke citations (prompts stricts) ==="
"$PY" scripts/eval_redact_smoke.py

echo ""
echo "=== 5/5 Évaluations golden + cabinet ==="
"$PY" scripts/run_eval.py --suite all

echo ""
echo "Terminé. Lance : ./scripts/rag.sh"
