#!/usr/bin/env bash
# Sprint entraînement : ~10 minutes à fond (MPS, gros batch), puis arrêt propre (STOP).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"

DURATION="${DURATION_SEC:-600}"
DATA="${DATA:-data/legal/prepared_214885}"
CONFIG="${CONFIG:-configs/generated/legal_xxl_sprint_10min_cpu.json}"
PARENT="${PARENT:-runs/legal_xxl_adaptive_burn_refine/best.pt}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
OUTPUT="${OUTPUT:-runs/sprint_10min_${STAMP}}"
LOG="${LOG:-runs/sprint_10min.log}"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10

if [[ "${1:-}" != "--foreground" ]]; then
  mkdir -p runs
  nohup caffeinate -dims "$0" --foreground >> "$LOG" 2>&1 &
  echo "Sprint 10 min lancé (PID $!) — log : $LOG"
  echo "Suivi : tail -f $LOG"
  exit 0
fi

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

rm -f runs/.global_training.lock
for lock in runs/*/training.lock; do
  [[ -f "$lock" ]] || continue
  pid="$(tr -d '[:space:]' < "$lock" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "ERREUR : entraînement déjà actif ($pid)"
    exit 1
  fi
  rm -f "$lock"
done

[[ -f "$DATA/manifest.json" ]] || { log "Corpus absent"; exit 1; }
[[ -f "$PARENT" ]] || { log "Parent absent $PARENT"; exit 1; }

log "════════════════════════════════════════"
log "SPRINT ${DURATION}s — sortie $OUTPUT"
log "Config max : batch 32×3, MPS, eval rare"
log "════════════════════════════════════════"

"$PY" -m llm_fr.train --config "$CONFIG" --data "$DATA" --output "$OUTPUT" --init-from "$PARENT" &
TRAIN_PID=$!

START=$(date +%s)
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  NOW=$(date +%s)
  ELAPSED=$((NOW - START))
  if (( ELAPSED >= DURATION )); then
    log "Chronomètre ${DURATION}s — demande STOP"
    touch "$OUTPUT/STOP"
    break
  fi
  sleep 3
done

wait "$TRAIN_PID" 2>/dev/null || true
rm -f "$OUTPUT/STOP"

if [[ -f "$OUTPUT/progress.json" ]]; then
  "$PY" -c "import json; p=json.load(open('$OUTPUT/progress.json')); print('Bilan : step', p.get('step'), 'train_loss', p.get('train_loss'), 'tok/s', round(p.get('tokens_per_second',0),1))"
fi
log "Sprint terminé → $OUTPUT"
