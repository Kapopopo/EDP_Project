#!/usr/bin/env bash
# Suivi du rebuild Judilibre (sans pièges zsh sur ~ dans les commentaires).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CHUNKS="data/rag/chunks_judilibre.jsonl"
TARGET=3982129
echo "=== Rebuild Judilibre — $(date '+%H:%M:%S') ==="
if pgrep -f "llm_fr.chunks.*judilibre" >/dev/null 2>&1; then
  echo "Découpage : EN COURS"
elif pgrep -f "llm_fr.rag_index" >/dev/null 2>&1; then
  echo "Index TF-IDF : EN COURS"
elif pgrep -f "build_semantic_curated" >/dev/null 2>&1; then
  echo "Index sémantique : EN COURS"
else
  echo "Aucun job rebuild détecté (terminé ou arrêté)."
fi
if [ -f "$CHUNKS" ]; then
  N=$(wc -l < "$CHUNKS" | tr -d ' ')
  PCT=$((N * 100 / TARGET))
  [ "$PCT" -gt 100 ] && PCT=100
  echo "Fragments Judilibre : $N / ~${TARGET} (${PCT}%)"
  ls -lh "$CHUNKS" | awk '{print "Taille fichier :", $5}'
else
  echo "Fragments Judilibre : fichier absent"
fi
META="data/rag/index_legal/meta.jsonl"
POST="data/rag/index_legal/postings.npz"
MANI="data/rag/index_legal/manifest.json"
if [ -f "$META" ]; then
  ML=$(wc -l < "$META" | tr -d ' ')
  echo "meta.jsonl : $ML lignes (modifié $(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$META" 2>/dev/null || echo '?'))"
fi
if [ -f "$POST" ]; then
  echo "postings.npz : $(stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$POST" 2>/dev/null) ($(ls -lh "$POST" | awk '{print $5}'))"
fi
if [ -f "$MANI" ]; then
  python3 -c "import json; m=json.load(open('$MANI')); print('manifest (actif RAG tant que postings ancien) :', m.get('n_docs'), 'fragments')"
fi
if pgrep -f 'llm_fr.rag_index' >/dev/null 2>&1; then
  PID=$(pgrep -f 'llm_fr.rag_index' | head -1)
  ps -p "$PID" -o etime=,pcpu= 2>/dev/null | awk '{print "Processus index : PID '"$PID"' — durée",$1,"CPU",$2"%"}'
  if [ -f "$META" ] && [ -f "$POST" ]; then
    MT=$(stat -f '%m' "$META" 2>/dev/null)
    PT=$(stat -f '%m' "$POST" 2>/dev/null)
    if [ "$ML" -ge 5000000 ] 2>/dev/null && [ "$PT" -lt "$MT" ] 2>/dev/null; then
      echo "Phase probable : compression postings.npz (meta déjà à jour, manifest pas encore écrit)."
    fi
  fi
fi
echo "Log : tail -f logs/judilibre_rebuild.log"
