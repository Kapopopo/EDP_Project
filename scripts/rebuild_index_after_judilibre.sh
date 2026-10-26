#!/usr/bin/env bash
# Enchaîne découpage Judilibre → fusion → index TF-IDF → sémantique (sans re-télécharger).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
LOG="${JUDILIBRE_REBUILD_LOG:-logs/judilibre_rebuild.log}"
CORPUS="data/legal/raw/judilibre.jsonl"
exec > >(tee -a "$LOG") 2>&1
echo "=== Rebuild index après Judilibre — $(date -Iseconds) ==="
if [ ! -f "$CORPUS" ]; then
  echo "Corpus absent : $CORPUS"
  exit 1
fi
echo "Corpus : $(wc -l < "$CORPUS" | tr -d ' ') décisions"
echo "Découpage Judilibre..."
.venv/bin/python -m llm_fr.chunks --input "$CORPUS" --output data/rag/chunks_judilibre.jsonl --size 1200 --overlap 150
echo "Fusion Légifrance + Judilibre..."
cat data/rag/chunks_legifrance.jsonl data/rag/chunks_judilibre.jsonl > data/rag/chunks_legal.jsonl
echo "Index TF-IDF..."
.venv/bin/python -m llm_fr.rag_index --chunks data/rag/chunks_legal.jsonl --output data/rag/index_legal
echo "Index sémantique curé..."
./scripts/build_semantic_curated.sh data/rag/index_legal
echo "=== Terminé — $(date -Iseconds) ==="
