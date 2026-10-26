#!/usr/bin/env bash
# Pose une question (mode synthèse par défaut ; --generer pour le modèle).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GENERER=0
while [ $# -gt 0 ]; do
  case "$1" in
    --generer) GENERER=1; shift ;;
    --synthese) shift ;;  # Alias conservé : synthèse = défaut.
    *) break ;;
  esac
done
QUESTION="${*:-}"
if [ -z "$QUESTION" ]; then
  echo "Usage: ./scripts/ask_rag.sh [--generer] votre question ici" >&2
  exit 1
fi
if [ "$GENERER" -eq 0 ]; then
  exec "$ROOT/scripts/ask_synthese.sh" "$QUESTION"
fi
.venv/bin/python -c "
import sys
from pathlib import Path
from llm_fr.rag_chat import default_index, format_sources, load_model, rag_reply
from llm_fr.rag_index import RagIndex

question = sys.argv[1]
index = RagIndex.load(default_index())
candidates = ('runs/legal_xxl_adaptive_refine2/best.pt', 'runs/legal_xxl_adaptive_refine/best.pt', 'runs/legal_xxl_adaptive/best.pt')
checkpoint = next(p for p in candidates if Path(p).is_file())
model, tokenizer, step, context = load_model(checkpoint)
result = rag_reply(index, model, tokenizer, question, top_k=3, count=120, temperature=0.7, max_context_chars=min(900, context * 3))
if result['sources']:
    print(format_sources(result['sources']))
    print()
print(f'Modèle (étape {step}) > {result[\"text\"]}')
" "$QUESTION"
