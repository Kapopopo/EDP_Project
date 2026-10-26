#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-runs/legal_xxl_adaptive_burn}"
echo "=== Burn train — $(date '+%H:%M:%S') ==="
if pgrep -f "llm_fr.train" >/dev/null 2>&1; then
  pid=$(pgrep -f "llm_fr.train" | head -1)
  ps -p "$pid" -o etime=,pcpu=,%mem= 2>/dev/null | awk '{print "Processus : PID '"$pid"' — durée",$1,"CPU",$2"% RAM",$3}'
else
  echo "Aucun processus llm_fr.train burn détecté."
fi
if [[ -f "$OUT/progress.json" ]]; then
  .venv/bin/python -c "import json; p=json.load(open('$OUT/progress.json')); print('Step', p.get('step'), '| train_loss', p.get('train_loss'), '| tok/s', round(p.get('tokens_per_second',0),1))"
fi
if [[ -f "$OUT/metrics.jsonl" ]]; then
  tail -1 "$OUT/metrics.jsonl" | .venv/bin/python -c "import sys,json; m=json.load(sys.stdin); print('Dernière val : step',m.get('step'),'val_loss',m.get('val_loss'),'ppl',round(m.get('perplexity',0),2))"
fi
echo "Log : tail -f runs/burn_train.log"
