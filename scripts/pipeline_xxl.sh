#!/usr/bin/env bash
# Pipeline adaptatif : corpus + XXL calibrés sur la machine (hardware.py).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
cd "$ROOT"
LOG="$ROOT/runs/xxl_pipeline.log"
PID_FILE="$ROOT/runs/xxl_pipeline.pid"
TOK="data/legal/prepared_25k/tokenizer.json"

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
      step="$(cat "$output/progress.json" 2>/dev/null | "$PY" -c "import sys,json; print(json.load(sys.stdin).get('step',0))" 2>/dev/null || echo "?")"
      log "⏳ $(basename "$output") step $step"
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

# Profil matériel
PROFILE="$("$PY" -m llm_fr.hardware --save)"
CORPUS_DOCS="$("$PY" -c "import json; print(json.load(open('runs/hardware_profile.json'))['corpus_docs'])")"
MAX_MB="$("$PY" -c "import json; print(json.load(open('runs/hardware_profile.json'))['max_mb'])")"
TIER="$("$PY" -c "import json; print(json.load(open('runs/hardware_profile.json'))['tier'])")"
DEVICE="$("$PY" -c "import json; print(json.load(open('runs/hardware_profile.json'))['device'])")"
RAW="data/legal/raw/corpus_${CORPUS_DOCS}.jsonl"
DATA="data/legal/prepared_${CORPUS_DOCS}"
if [[ ! -f "$DATA/manifest.json" ]]; then
  best=0
  for m in data/legal/prepared_*/manifest.json; do
    [[ -f "$m" ]] || continue
    n="${m#data/legal/prepared_}"; n="${n%/manifest.json}"
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > best )) && best=$n
  done
  if (( best > 0 )); then
    log "↪ Corpus adaptatif $CORPUS_DOCS → prepared existant $best"
    CORPUS_DOCS=$best
    RAW="data/legal/raw/corpus_${CORPUS_DOCS}.jsonl"
    DATA="data/legal/prepared_${CORPUS_DOCS}"
  fi
fi
CONFIG_DIR="configs/generated"
mkdir -p "$CONFIG_DIR" runs

log "══════════════════════════════════════════════════════════"
log "PIPELINE ADAPTATIF — tier $TIER | ${CORPUS_DOCS} docs | $DEVICE"
log "$PROFILE"
log "══════════════════════════════════════════════════════════"

# Génère configs XXL calibrées
"$PY" - <<'PYEOF' | tee -a "$LOG"
import json
from pathlib import Path
from llm_fr.hardware import profile
p = profile(".")
base = dict(seed=42, context=256, width=640, heads=10, layers=10, dropout=0.1,
            device=p["device"], threads=p["threads"], batch_size=p["batch_size"],
            accumulation=p["accumulation"], eval_batches=4,
            max_rss_mb=p["max_rss_mb"], min_available_mb=p["min_available_mb"], min_disk_mb=400)
out = Path("configs/generated")
phases = [
    ("legal_xxl_adaptive_cpu.json", dict(steps=12000, warmup=200, learning_rate=0.00008, eval_every=250, patience=25, weight_decay=0.08)),
    ("legal_xxl_adaptive_refine_cpu.json", dict(steps=8000, warmup=100, learning_rate=0.00002, eval_every=250, patience=20, weight_decay=0.05)),
    ("legal_xxl_adaptive_refine2_cpu.json", dict(steps=5000, warmup=60, learning_rate=0.000008, eval_every=250, patience=15, weight_decay=0.05)),
]
for name, extra in phases:
    cfg = {**base, **extra}
    (out / name).write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Config : {out / name} (batch {cfg['batch_size']}, device {cfg['device']})")
Path("runs/hardware_profile.json").write_text(json.dumps(p, indent=2))
PYEOF

# Reprend un téléchargement interrompu sous un autre nom
for old in data/legal/raw/corpus_*.jsonl.part; do
  [[ -f "$old" ]] || continue
  [[ "$old" == "${RAW}.part" ]] && continue
  if [[ ! -f "${RAW}.part" ]]; then
    log "↪ Reprise $(basename "$old") → $(basename "$RAW").part"
    mv "$old" "${RAW}.part"
  fi
done

# Téléchargement
if [[ ! -f "$RAW" ]]; then
  log "▶ Téléchargement $CORPUS_DOCS documents (max ${MAX_MB} Mo)..."
  "$PY" -m llm_fr.download_legal --output "$RAW" --limit "$CORPUS_DOCS" --max-mb "$MAX_MB" 2>&1 | tee -a "$LOG"
else
  lines=$(wc -l < "$RAW" | tr -d ' ')
  if [[ "$lines" -lt "$CORPUS_DOCS" ]]; then
    log "▶ Complément corpus ($lines → $CORPUS_DOCS)..."
    "$PY" -m llm_fr.download_legal --output "$RAW" --limit "$CORPUS_DOCS" --max-mb "$MAX_MB" 2>&1 | tee -a "$LOG" || true
  else
    log "✓ Corpus $RAW ($lines docs)"
  fi
fi

# Préparation
if [[ ! -f "$DATA/manifest.json" ]]; then
  log "▶ Préparation $DATA ..."
  "$PY" -m llm_fr.prepare "$RAW" "$DATA" --tokenizer "$TOK" 2>&1 | tee -a "$LOG"
else
  log "✓ $DATA prêt"
fi

# Attendre refine large MPS
for run in runs/legal_large_25k_mps_refine runs/legal_large_25k_mps_refine2; do
  wait_run "$run"
done

P1="runs/legal_xxl_adaptive"
P2="runs/legal_xxl_adaptive_refine"
P3="runs/legal_xxl_adaptive_refine2"
C1="$CONFIG_DIR/legal_xxl_adaptive_cpu.json"
C2="$CONFIG_DIR/legal_xxl_adaptive_refine_cpu.json"
C3="$CONFIG_DIR/legal_xxl_adaptive_refine2_cpu.json"

wait_run "$P1"

if [[ -f "$P1/last.pt" ]]; then
  train "$C1" "$DATA" "$P1" --resume
else
  log "▶ XXL 55M from scratch sur $CORPUS_DOCS docs"
  train "$C1" "$DATA" "$P1"
fi
[[ -f "$P1/best.pt" ]] || exit 1

if [[ -f "$P2/last.pt" ]]; then
  train "$C2" "$DATA" "$P2" --resume
else
  train "$C2" "$DATA" "$P2" --init-from "$P1/best.pt"
fi
[[ -f "$P2/best.pt" ]] || exit 1

if [[ -f "$P3/last.pt" ]]; then
  train "$C3" "$DATA" "$P3" --resume
else
  train "$C3" "$DATA" "$P3" --init-from "$P2/best.pt"
fi

log "TERMINÉ — tier $TIER — $P3/best.pt"
rm -f "$PID_FILE"
