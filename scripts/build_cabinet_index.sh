#!/usr/bin/env bash
# Index léger cabinet (RIN, mentions légales) — quelques secondes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SRC="${1:-data/legal/curated/cabinet_metier.jsonl}"
CHUNKS="${2:-data/rag/chunks_cabinet.jsonl}"
INDEX="${3:-data/rag/index_cabinet}"
.venv/bin/python -m llm_fr.chunks --input "$SRC" --output "$CHUNKS" --size 1200 --overlap 100
.venv/bin/python -m llm_fr.rag_index --chunks "$CHUNKS" --output "$INDEX"
echo "Index cabinet : $INDEX"
