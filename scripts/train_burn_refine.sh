#!/usr/bin/env bash
# Affinage après burn XXL (nouvelle phase depuis best.pt burn).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
LOG="$ROOT/runs/burn_refine_train.log"
PID_FILE="$ROOT/runs/burn_refine_train.pid"
DATA="${DATA:-data/legal/prepared_214885}"
CONFIG="${CONFIG:-configs/generated/legal_xxl_adaptive_burn_refine_cpu.json}"
OUTPUT="${OUTPUT:-runs/legal_xxl_adaptive_burn_refine}"
PARENT="${PARENT:-runs/legal_xxl_adaptive_burn/best.pt}"

if [[ "${1:-}" != "--foreground" ]]; then
  if pgrep -f "llm_fr.train.*$OUTPUT" >/dev/null 2>&1; then
    echo "Refine déjà en cours. ./scripts/monitor_burn_refine.sh"
    exit 0
  fi
  if [[ -f "$PID_FILE" ]]; then
    old="$(tr -d '[:space:]' < "$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
      echo "Refine lancé (PID $old). tail -f $LOG"
      exit 0
    fi
    rm -f "$PID_FILE"
  fi
  mkdir -p runs
  nohup caffeinate -dims "$0" --foreground >> "$LOG" 2>&1 &
  echo $! > "$PID_FILE"
  echo "Refine burn lancé — PID $(cat "$PID_FILE")"
  echo "Suivi : ./scripts/monitor_burn_refine.sh"
  exit 0
fi

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

rm -f runs/.global_training.lock
for lock in runs/*/training.lock; do
  [[ -f "$lock" ]] || continue
  pid="$(tr -d '[:space:]' < "$lock" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    log "ERREUR : entraînement actif ($pid). ./scripts/stop_all_training.sh"
    exit 1
  fi
  rm -f "$lock"
done

[[ -f "$DATA/manifest.json" ]] || { log "Corpus absent $DATA"; exit 1; }
[[ -f "$PARENT" ]] || { log "Parent absent $PARENT"; exit 1; }

"$PY" -m llm_fr.hardware --save | tee -a "$LOG"
log "REFINE — $OUTPUT depuis $PARENT"

extra=()
if [[ -f "$OUTPUT/last.pt" ]]; then
  extra+=(--resume)
  log "Reprise $OUTPUT/last.pt"
else
  if [[ -d "$OUTPUT" ]] && [[ ! -f "$OUTPUT/last.pt" ]]; then
    log "Dossier partiel sans checkpoint — réinitialisation $OUTPUT"
    rm -rf "$OUTPUT"
  fi
  extra+=(--init-from "$PARENT")
fi

"$PY" -m llm_fr.train --config "$CONFIG" --data "$DATA" --output "$OUTPUT" "${extra[@]}" 2>&1 | tee -a "$LOG"
log "Terminé — $(tail -1 "$OUTPUT/metrics.jsonl" 2>/dev/null || echo '?')"
rm -f "$PID_FILE"
