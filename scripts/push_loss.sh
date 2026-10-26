#!/usr/bin/env bash
# Enchaîne automatiquement les phases d'entraînement vers une perte minimale.
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
      log "XXL base en cours (PID $pid)..."
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
  log "START $output ← $config"
  "$PY" -m llm_fr.train --config "$config" --data "$data" --output "$output" "$@"
  log "DONE  $output"
}

best_loss() {
  local metrics="$1"
  [[ -f "$metrics" ]] || { echo "999"; return; }
  "$PY" -c "
import json, math
best = math.inf
for line in open('$metrics'):
    m = json.loads(line)
    v = m.get('val_loss')
    if v is not None and v < best:
        best = v
print(f'{best:.4f}')
"
}

log "Pipeline démarré — attente fin XXL base..."
# Attendre la fin du XXL base s'il tourne déjà
wait_run "runs/legal_xxl"

log "Meilleure perte XXL base : $(best_loss runs/legal_xxl/metrics.jsonl)"

# Phase 2 : refine
train configs/legal_xxl_refine_cpu.json data/legal/prepared_15k runs/legal_xxl_refine \
  --init-from runs/legal_xxl/best.pt

log "Meilleure perte refine : $(best_loss runs/legal_xxl_refine/metrics.jsonl)"

# Phase 3 : refine2 ultra-basse LR
train configs/legal_xxl_refine2_cpu.json data/legal/prepared_15k runs/legal_xxl_refine2 \
  --init-from runs/legal_xxl_refine/best.pt

log "Meilleure perte refine2 : $(best_loss runs/legal_xxl_refine2/metrics.jsonl)"

# Phase 4 : corpus 25k si disponible
if [[ -f data/legal/raw/corpus_25k.jsonl ]]; then
  if [[ ! -f data/legal/prepared_25k/manifest.json ]]; then
    log "Préparation corpus 25k..."
    "$PY" -m llm_fr.prepare data/legal/raw/corpus_25k.jsonl data/legal/prepared_25k \
      --tokenizer data/legal/prepared/tokenizer.json
  fi
  train configs/legal_xxl_25k_cpu.json data/legal/prepared_25k runs/legal_xxl_25k \
    --init-from runs/legal_xxl_refine2/best.pt
  log "Meilleure perte 25k : $(best_loss runs/legal_xxl_25k/metrics.jsonl)"
fi

# Phase 5 : warm-start large sur 5k (rapide, complémentaire)
if [[ -d data/legal/prepared_5k ]]; then
  if [[ ! -f runs/legal_large_5k/best.pt ]] || [[ $(best_loss runs/legal_large_5k/metrics.jsonl) > 3.0 ]]; then
    train configs/legal_large_5k_cpu.json data/legal/prepared_5k runs/legal_large_5k \
      --init-from runs/legal_large_turbo/best.pt
    log "Meilleure perte large_5k : $(best_loss runs/legal_large_5k/metrics.jsonl)"
  fi
fi

log "Pipeline terminé. Meilleurs checkpoints :"
for d in runs/legal_xxl_refine2 runs/legal_xxl_25k runs/legal_xxl_refine runs/legal_xxl runs/legal_large_5k; do
  [[ -f "$d/metrics.jsonl" ]] && log "  $d → $(best_loss "$d/metrics.jsonl")"
done
