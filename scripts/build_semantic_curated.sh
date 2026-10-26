#!/usr/bin/env bash
# Construit l'index d'embeddings sur le sous-ensemble métier curé (hybride TF-IDF + sémantique).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INDEX="${1:-data/rag/index_legal}"
MODEL="${SEMANTIC_MODEL:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
BATCH="${SEMANTIC_BATCH:-32}"
MIN_PRIORITY="${SEMANTIC_MIN_PRIORITY:-2}"
if [ ! -f "$INDEX/meta.jsonl" ]; then
  echo "Index introuvable : $INDEX (lancez d'abord build_rag_legal.sh ou rebuild_index_fast.sh)"
  exit 1
fi
if ! .venv/bin/python -c "import sentence_transformers" 2>/dev/null; then
  echo "Installation de sentence-transformers..."
  .venv/bin/pip install -r requirements-rag.txt
fi
echo "Index sémantique curé : modèle=$MODEL index=$INDEX"
.venv/bin/python -m llm_fr.semantic_index --index "$INDEX" --model "$MODEL" --batch-size "$BATCH" --min-priority "$MIN_PRIORITY"
echo "Terminé. Redémarrez ./scripts/rag.sh pour activer le mode hybride."
