#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-runs/legal_xxl_adaptive_burn_refine}"
cd "$ROOT"
echo "=== Refine burn — $(date '+%H:%M:%S') ==="
if pgrep -f "llm_fr.train" >/dev/null 2>&1; then
  pid=$(pgrep -f "llm_fr.train" | head -1)
  ps -p "$pid" -o etime=,pcpu=,%mem= 2>/dev/null | awk -v p="$pid" '{print "Train PID", p, "—", $0}'
else
  echo "Aucun llm_fr.train actif."
fi
if [[ -f "$OUT/progress.json" ]]; then
  .venv/bin/python -c "import json; p=json.load(open('$OUT/progress.json')); print('Step', p.get('step'), '/ 5000 | train_loss', p.get('train_loss'))"
fi
if [[ -f "$OUT/metrics.jsonl" ]]; then
  tail -1 "$OUT/metrics.jsonl" | .venv/bin/python -c "import sys,json; m=json.load(sys.stdin); print('val_loss',m.get('val_loss'),'ppl',round(m.get('perplexity',0),2))"
fi
echo "Log : tail -f runs/burn_refine_train.log"
