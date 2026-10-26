#!/usr/bin/env bash
# Pipeline RAG jurisprudence : Judilibre → chunks → index.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CORPUS="${1:-data/legal/raw/judilibre.jsonl}"
CHUNKS="${2:-data/rag/chunks_judilibre.jsonl}"
INDEX="${3:-data/rag/index_judilibre}"
LIMIT="${JUDILIBRE_LIMIT:-0}"
if [ "${JUDILIBRE_REFRESH:-0}" = "1" ] && [ -f "$CORPUS" ]; then
  echo "JUDILIBRE_REFRESH=1 : suppression de $CORPUS"
  rm -f "$CORPUS"
fi
if [ ! -f "$CORPUS" ]; then
  echo "Corpus Judilibre absent : $CORPUS"
  echo "Téléchargement Cour de cassation (export open data, ~950 Mo compressé si LIMIT=0)..."
  if [ "$LIMIT" != "0" ]; then
    .venv/bin/python -m llm_fr.download_judilibre --output "$CORPUS" --limit "$LIMIT"
  else
    .venv/bin/python -m llm_fr.download_judilibre --output "$CORPUS"
  fi
fi
echo "Découpage : $CORPUS"
.venv/bin/python -m llm_fr.chunks --input "$CORPUS" --output "$CHUNKS" --size 1200 --overlap 150
echo "Indexation : $CHUNKS"
.venv/bin/python -m llm_fr.rag_index --chunks "$CHUNKS" --output "$INDEX"
echo "Index jurisprudence prêt : $INDEX"
