#!/usr/bin/env bash
# Pipeline max RAM/CPU après XXL max.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$ROOT/runs/pipeline_max.log"; }

wait_run() {
  local output="$1"
  while [[ -f "$output/training.lock" ]]; do
    pid="$(cat "$output/training.lock" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "En cours : $output (PID $pid)"
      sleep 120
    else
      rm -f "$output/training.lock"
      break
    fi
  done
}

train() {
  local config="$1" data="$2" output="$3"
  shift 3
  log "START $output"
  "$PY" -m llm_fr.train --config "$config" --data "$data" --output "$output" "$@"
  log "DONE  $output"
}

best_loss() {
  [[ -f "$1" ]] || { echo "999"; return; }
  "$PY" -c "import json,math; b=math.inf
for l in open('$1'):
 m=json.loads(l); v=m.get('val_loss')
 if v and v<b: b=v
print(f'{b:.4f}')"
}

log "Pipeline MAX démarré"
wait_run runs/legal_xxl_fast
log "XXL fast best : $(best_loss runs/legal_xxl_fast/metrics.jsonl)"

train configs/legal_xxl_refine_cpu.json data/legal/prepared_15k runs/legal_xxl_fast_refine \
  --init-from runs/legal_xxl_fast/best.pt

train configs/legal_xxl_refine2_cpu.json data/legal/prepared_15k runs/legal_xxl_fast_refine2 \
  --init-from runs/legal_xxl_fast_refine/best.pt

if [[ -d data/legal/prepared_25k ]]; then
  train configs/legal_xxl_25k_cpu.json data/legal/prepared_25k runs/legal_xxl_25k_fast \
    --init-from runs/legal_xxl_fast_refine2/best.pt
fi

log "Terminé. Meilleures pertes :"
for d in runs/legal_xxl_25k_fast runs/legal_xxl_fast_refine2 runs/legal_xxl_fast_refine runs/legal_xxl_fast; do
  [[ -f "$d/metrics.jsonl" ]] && log "  $d → $(best_loss "$d/metrics.jsonl")"
done
