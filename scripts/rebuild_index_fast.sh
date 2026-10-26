#!/usr/bin/env bash
# Reconstruit l'index combiné depuis les chunks existants (format npz + profils).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CHUNKS="${1:-data/rag/chunks_legal.jsonl}"
INDEX="${2:-data/rag/index_legal}"
if [ ! -f "$CHUNKS" ]; then
  echo "Fichier de fragments absent : $CHUNKS"
  echo "Lancez d'abord : ./scripts/build_rag_legal.sh"
  exit 1
fi
echo "Reconstruction index : $CHUNKS → $INDEX"
.venv/bin/python -m llm_fr.rag_index --chunks "$CHUNKS" --output "$INDEX"
echo "Index prêt : $INDEX"
if [ "${BUILD_SEMANTIC:-0}" = "1" ]; then
  ./scripts/build_semantic_curated.sh "$INDEX"
fi
