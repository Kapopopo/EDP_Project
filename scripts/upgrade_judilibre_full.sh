#!/usr/bin/env bash
# Judilibre Cassation complet → chunks → index combiné → index sémantique curé.
# Durée typique : 30–90 min (téléchargement + indexation selon la machine).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
LOG="${JUDILIBRE_UPGRADE_LOG:-/tmp/judilibre_full_upgrade.log}"
CORPUS="data/legal/raw/judilibre.jsonl"
CHUNKS_JUDI="data/rag/chunks_judilibre.jsonl"
CHUNKS_LEGI="data/rag/chunks_legifrance.jsonl"
CHUNKS_LEGAL="data/rag/chunks_legal.jsonl"
INDEX="data/rag/index_legal"

exec > >(tee -a "$LOG") 2>&1
echo "=== Judilibre complet — $(date -Iseconds) ==="
echo "Journal : $LOG"

FREE_GIB="$(df -g . | awk 'NR==2 {print $4}')"
if [ "${FREE_GIB:-0}" -lt 15 ]; then
  echo "Attention : moins de 15 Gio libres (recommandé pour corpus + index)."
fi

LINES=0
if [ -f "$CORPUS" ]; then
  LINES="$(wc -l < "$CORPUS" | tr -d ' ')"
fi
if [ "$LINES" -gt 100000 ] && [ "${JUDILIBRE_FORCE_DOWNLOAD:-0}" != "1" ]; then
  echo "Corpus déjà complet ($LINES lignes) — saut téléchargement."
else
  if [ -f "$CORPUS" ]; then
    BACKUP="data/legal/raw/judilibre_$(date +%Y%m%d_%H%M%S).bak.jsonl"
    echo "Sauvegarde de l'ancien corpus : $BACKUP"
    mv "$CORPUS" "$BACKUP"
  fi
  echo "Téléchargement et conversion Cassation (LIMIT=0)..."
  .venv/bin/python -m llm_fr.download_judilibre --output "$CORPUS"
fi
rm -f "$CHUNKS_JUDI"

echo "Découpage Judilibre..."
.venv/bin/python -m llm_fr.chunks --input "$CORPUS" --output "$CHUNKS_JUDI" --size 1200 --overlap 150

if [ ! -f "$CHUNKS_LEGI" ]; then
  echo "Fragments Légifrance absents — lancez build_rag_legifrance.sh d'abord."
  exit 1
fi

echo "Fusion Légifrance + Judilibre..."
cat "$CHUNKS_LEGI" "$CHUNKS_JUDI" > "$CHUNKS_LEGAL"

echo "Reconstruction index TF-IDF : $INDEX"
.venv/bin/python -m llm_fr.rag_index --chunks "$CHUNKS_LEGAL" --output "$INDEX"

echo "Reconstruction index sémantique curé..."
./scripts/build_semantic_curated.sh "$INDEX"

echo "=== Terminé — $(date -Iseconds) ==="
echo "Redémarrez ./scripts/rag.sh pour utiliser le nouvel index."
