#!/usr/bin/env bash
# Pipeline RAG combiné : Légifrance + Judilibre → index unique.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LEGI_CORPUS="${1:-data/legal/raw/legifrance.jsonl}"
JUDI_CORPUS="${2:-data/legal/raw/judilibre.jsonl}"
CHUNKS="${3:-data/rag/chunks_legal.jsonl}"
INDEX="${4:-data/rag/index_legal}"
LIMIT="${JUDILIBRE_LIMIT:-0}"
if [ ! -f "$LEGI_CORPUS" ]; then
  echo "Construction Légifrance..."
  ./scripts/build_rag_legifrance.sh "$LEGI_CORPUS" data/rag/chunks_legifrance.jsonl data/rag/index_legifrance
fi
if [ ! -f "$JUDI_CORPUS" ]; then
  echo "Construction Judilibre (limit=${LIMIT})..."
  JUDILIBRE_LIMIT="$LIMIT" ./scripts/build_rag_judilibre.sh "$JUDI_CORPUS" data/rag/chunks_judilibre.jsonl data/rag/index_judilibre
fi
echo "Fusion des fragments Légifrance + Judilibre..."
cat data/rag/chunks_legifrance.jsonl data/rag/chunks_judilibre.jsonl > "$CHUNKS"
echo "Indexation combinée : $CHUNKS"
.venv/bin/python -m llm_fr.rag_index --chunks "$CHUNKS" --output "$INDEX"
echo "Index juridique combiné prêt : $INDEX"
if [ "${BUILD_SEMANTIC:-0}" = "1" ]; then
  ./scripts/build_semantic_curated.sh "$INDEX"
fi
