#!/usr/bin/env bash
# Prépare meta_offsets.npy (une passe sur meta.jsonl) sans reconstruire l'index TF-IDF.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INDEX="${1:-data/rag/index_legal}"
case "$INDEX" in
  ""|"#"*) INDEX="data/rag/index_legal" ;;
esac
if [ ! -f "$INDEX/meta.jsonl" ]; then
  echo "Index invalide ou meta.jsonl absent : $INDEX"
  echo "Usage : $0 [data/rag/index_legal]"
  exit 1
fi
.venv/bin/python - <<PY
from pathlib import Path
from llm_fr.meta_store import LazyMetaStore

index = Path("$INDEX")
store = LazyMetaStore.load(index)
print(f"meta_offsets prêt : {index} ({len(store)} fragments)")
PY
