#!/usr/bin/env bash
# Entraînement XXL intensif (MPS + gros batch) — chauffe CPU/GPU au maximum raisonnable sur 64 Go.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
LOG="$ROOT/runs/burn_train.log"
PID_FILE="$ROOT/runs/burn_train.pid"
DATA="${DATA:-data/legal/prepared_214885}"
CONFIG="${CONFIG:-configs/generated/legal_xxl_adaptive_burn_cpu.json}"
OUTPUT="${OUTPUT:-runs/legal_xxl_adaptive_burn}"
PARENT="${PARENT:-runs/legal_xxl_adaptive_refine2/best.pt}"

if [[ "${1:-}" != "--foreground" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    old="$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
      echo "Entraînement burn déjà lancé (PID $old). tail -f $LOG"
      exit 0
    fi
    rm -f "$PID_FILE"
  fi
  mkdir -p runs
  nohup caffeinate -dims "$0" --foreground >> "$LOG" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Burn lancé en arrière-plan — PID $(cat "$PID_FILE")"
  echo "Suivi : tail -f $LOG"
  exit 0
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

rm -f runs/.global_training.lock
for lock in runs/*/training.lock; do
  [[ -f "$lock" ]] || continue
  pid="$(tr -d '[:space:]' < "$lock" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "ERREUR : entraînement déjà actif ($pid). ./scripts/stop_all_training.sh"
    exit 1
  fi
  rm -f "$lock"
done

if [[ ! -f "$DATA/manifest.json" ]]; then
  log "ERREUR : corpus absent $DATA"
  exit 1
fi
if [[ ! -f "$PARENT" ]]; then
  log "ERREUR : checkpoint parent absent $PARENT"
  exit 1
fi

"$PY" -m llm_fr.hardware --save | tee -a "$LOG"
log "══════════════════════════════════════════════════════════"
log "BURN XXL — $OUTPUT"
log "Data : $DATA | Config : $CONFIG"
log "Parent : $PARENT"
log "══════════════════════════════════════════════════════════"

extra=()
if [[ -f "$OUTPUT/last.pt" ]]; then
  extra+=(--resume)
  log "Reprise depuis $OUTPUT/last.pt"
else
  extra+=(--init-from "$PARENT")
fi

attempt=0
while [[ $attempt -lt 5 ]]; do
  attempt=$((attempt + 1))
  rm -f "$OUTPUT/training.lock" runs/.global_training.lock
  if "$PY" -m llm_fr.train --config "$CONFIG" --data "$DATA" --output "$OUTPUT" "${extra[@]}" 2>&1 | tee -a "$LOG"; then
    log "TERMINÉ — meilleure val : $(tail -1 "$OUTPUT/metrics.jsonl" 2>/dev/null || echo '?')"
    rm -f "$PID_FILE"
    exit 0
  fi
  log "Échec tentative $attempt — reprise si last.pt existe"
  sleep 20
  [[ -f "$OUTPUT/last.pt" ]] && extra=(--resume)
done
log "ÉCHEC après 5 tentatives"
rm -f "$PID_FILE"
exit 1
