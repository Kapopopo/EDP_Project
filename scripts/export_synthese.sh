#!/usr/bin/env bash
# Une question → synthèse exportée en Markdown.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
QUESTION="${*:-}"
OUT="${EXPORT:-exports/derniere_synthese.md}"
if [ -z "$QUESTION" ]; then
  echo "Usage: ./scripts/export_synthese.sh votre question" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"
export SYNTH_QUESTION="$QUESTION" SYNTH_OUT="$OUT"
.venv/bin/python - <<'PY'
import os
from llm_fr.rag_chat import default_index, synthese_reply
from llm_fr.rag_index import RagIndex
from llm_fr.synthesis_export import export_synthese_markdown

q = os.environ["SYNTH_QUESTION"]
out = os.environ["SYNTH_OUT"]
index_dir = default_index()
index = RagIndex.load(index_dir)
result = synthese_reply(index, q)
path = export_synthese_markdown(
    out,
    q,
    result["text"],
    hits=result.get("hits"),
    index_path=index_dir,
)
print(result["text"])
print(f"\n---\nExport : {path}")
PY
