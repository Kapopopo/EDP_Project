#!/usr/bin/env bash
# Construit l'index RAG à partir d'un corpus JSONL préparé.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CORPUS="${1:-data/legal/prepared_214885/val.jsonl}"
CHUNKS="${2:-data/rag/chunks.jsonl}"
INDEX="${3:-data/rag/index}"
echo "Corpus : $CORPUS"
.venv/bin/python -m llm_fr.chunks --input "$CORPUS" --output "$CHUNKS"
.venv/bin/python -m llm_fr.rag_index --chunks "$CHUNKS" --output "$INDEX"
echo "Index prêt : $INDEX"
