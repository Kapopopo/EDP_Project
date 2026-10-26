#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$ROOT/runs/pipeline.log"; }

wait_run() {
  local output="$1"
  while [[ -f "$output/training.lock" ]]; do
    pid="$(cat "$output/training.lock" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "En cours : $output (PID $pid)"
      sleep 60
    else
      rm -f "$output/training.lock"
      break
    fi
  done
}

log "Pipeline large 25k démarré"
wait_run runs/legal_large_25k_ram48

"$PY" -m llm_fr.train --config configs/legal_large_refine2_cpu.json \
  --data data/legal/prepared_25k --output runs/legal_large_25k_refine \
  --init-from runs/legal_large_25k_ram48/best.pt

log "Terminé. Perte : $( "$PY" -c "import json,math; b=math.inf
for l in open('runs/legal_large_25k_refine/metrics.jsonl'):
 m=json.loads(l); v=m.get('val_loss')
 if v and v<b: b=v
print(f'{b:.4f}')" )"
