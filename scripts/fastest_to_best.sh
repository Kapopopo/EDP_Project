#!/usr/bin/env bash
# Pipeline MPS (GPU M1) — refine + refine2 depuis le meilleur checkpoint CPU.
# Phase 1 CPU sautée si perte déjà < 3,0.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
LOG="$ROOT/runs/fastest.log"
PID_FILE="$ROOT/runs/fastest.pid"

if [[ "${1:-}" != "--foreground" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    old="$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then exit 0; fi
    rm -f "$PID_FILE"
  fi
  mkdir -p runs
  nohup caffeinate -i "$0" --foreground >> "$LOG" 2>&1 &
  echo $! > "$PID_FILE"
  disown -h 2>/dev/null || true
  exit 0
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

best_loss() {
  [[ -f "$1" ]] || { echo "999"; return; }
  "$PY" -c "import json,math; b=math.inf
for l in open('$1'):
 m=json.loads(l); v=m.get('val_loss')
 if v and v<b: b=v
print(f'{b:.4f}')"
}

wait_run() {
  local output="$1"
  while [[ -f "$output/training.lock" ]]; do
    pid="$(tr -d '[:space:]' < "$output/training.lock" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "⏳ $(basename "$output")"
      sleep 60
    else
      rm -f "$output/training.lock"
      break
    fi
  done
}

train() {
  local config="$1" data="$2" output="$3"
  shift 3
  local extra=("$@") attempt=0
  while [[ $attempt -lt 5 ]]; do
    attempt=$((attempt + 1))
    rm -f "$output/training.lock" runs/.global_training.lock
    log "▶ START $output"
    local cmd=( "$PY" -m llm_fr.train --config "$config" --data "$data" --output "$output" )
    ((${#extra[@]})) && cmd+=("${extra[@]}")
    if "${cmd[@]}" 2>&1 | tee -a "$LOG"; then
      log "✓ DONE $output → $(best_loss "$output/metrics.jsonl")"
      return 0
    fi
    sleep 30
    [[ -f "$output/last.pt" ]] && extra=(--resume)
  done
  return 1
}

pick_parent() {
  for c in runs/legal_large_25k_ram22/best.pt runs/legal_large_25k_ram15/best.pt runs/legal_large_25k_fast/best.pt; do
    [[ -f "$c" ]] && { echo "$c"; return; }
  done
  return 1
}

log "══════════════════════════════════════════════════════════"
log "PIPELINE MPS — GPU M1 Max, refine direct (perte déjà < 3,0)"
log "══════════════════════════════════════════════════════════"

P2="runs/legal_large_25k_mps_refine"
P3="runs/legal_large_25k_mps_refine2"

for old in runs/legal_large_25k_ram22 runs/legal_large_25k_ram15; do
  wait_run "$old"
done
wait_run "$P2"

parent="$(pick_parent)" || { log "ERREUR : pas de checkpoint"; exit 1; }
log "Parent : $parent (perte $(best_loss "$(dirname "$parent")/metrics.jsonl"))"

if [[ -f "$P2/last.pt" ]]; then
  train configs/legal_large_25k_mps_refine_cpu.json data/legal/prepared_25k "$P2" --resume
else
  train configs/legal_large_25k_mps_refine_cpu.json data/legal/prepared_25k "$P2" --init-from "$parent"
fi

[[ -f "$P2/best.pt" ]] || exit 1

if [[ -f "$P3/last.pt" ]]; then
  train configs/legal_large_25k_mps_refine2_cpu.json data/legal/prepared_25k "$P3" --resume
else
  train configs/legal_large_25k_mps_refine2_cpu.json data/legal/prepared_25k "$P3" --init-from "$P2/best.pt"
fi

log "TERMINÉ — $P3/best.pt"
rm -f "$PID_FILE"
