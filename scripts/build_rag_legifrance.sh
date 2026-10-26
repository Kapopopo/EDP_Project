#!/usr/bin/env bash
# Pipeline RAG 100 % juridique : Légifrance LEGI → chunks → index.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CORPUS="${1:-data/legal/raw/legifrance.jsonl}"
CHUNKS="${2:-data/rag/chunks_legifrance.jsonl}"
INDEX="${3:-data/rag/index_legifrance}"
if [ ! -f "$CORPUS" ]; then
  echo "Corpus Légifrance absent : $CORPUS"
  echo "Téléchargement et conversion LEGI (environ 1,1 Go + 30-60 min)..."
  .venv/bin/python -m llm_fr.download_legifrance --output "$CORPUS"
fi
echo "Découpage : $CORPUS"
.venv/bin/python -m llm_fr.chunks --input "$CORPUS" --output "$CHUNKS" --size 1200 --overlap 150
echo "Indexation : $CHUNKS"
.venv/bin/python -m llm_fr.rag_index --chunks "$CHUNKS" --output "$INDEX"
echo "Index juridique prêt : $INDEX"
