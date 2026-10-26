#!/usr/bin/env bash
# Affiche l'avancement de l'entraînement dans le terminal.
# Usage :
#   ./scripts/progress.sh          # un coup d'œil
#   ./scripts/progress.sh --watch  # rafraîchit toutes les 10 s
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
WATCH=false
INTERVAL=10
[[ "${1:-}" == "--watch" || "${1:-}" == "-w" ]] && WATCH=true
[[ "${2:-}" =~ ^[0-9]+$ ]] && INTERVAL="$2"

bar() {
  local current="$1" total="$2" width=30
  [[ "$total" -le 0 ]] && { printf '[%s]' "$(printf '%*s' "$width" '' | tr ' ' '?')"; return; }
  local filled=$((current * width / total))
  local empty=$((width - filled))
  printf '[%s%s] %d/%d (%.1f%%)' \
    "$(printf '%*s' "$filled" '' | tr ' ' '█')" \
    "$(printf '%*s' "$empty" '' | tr ' ' '░')" \
    "$current" "$total" "$(awk "BEGIN {print 100*$current/$total}")"
}

show() {
  clear 2>/dev/null || true
  echo "═══════════════════════════════════════════════════════════"
  echo "  DATTY AI — Avancement entraînement"
  echo "  $(date '+%d/%m/%Y %H:%M:%S')"
  echo "═══════════════════════════════════════════════════════════"
  if [[ -f "$ROOT/runs/hardware_profile.json" ]]; then
    "$PY" -c "import json; p=json.load(open('$ROOT/runs/hardware_profile.json')); print(f'  Machine : tier {p[\"tier\"]} | {p[\"ram_gb\"]} Go RAM | corpus {p[\"corpus_docs\"]} docs | {p[\"device\"]}')" 2>/dev/null || true
  fi
  echo

  # Entraînement actif (nettoie les verrous orphelins)
  local active=0 active_count=0
  while IFS= read -r lock; do
    local run_dir
    run_dir="$(dirname "$lock")"
    local name
    name="$(basename "$run_dir")"
    local pid
    pid="$(cat "$lock" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$lock"  # PID mort → verrou orphelin
      continue
    fi
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      active=1
      active_count=$((active_count + 1))
      local rss cpu elapsed step best total stale
      read -r cpu rss elapsed <<< "$(ps -p "$pid" -o pcpu=,rss=,etime= 2>/dev/null | awk '{print $1, $2, $3}')"
      rss_gb=$(awk "BEGIN {printf \"%.1f\", $rss/1024/1024}")
      read -r step best total stale train_loss tps eval_every sec_per_step next_val eta_val_min <<< "$("$PY" - <<PYEOF 2>/dev/null || echo "0 999 0 0 0 0 500 0 500 0"
import json, torch, math
from pathlib import Path
run = Path("$run_dir")
step, best, total, stale = 0, 999.0, 0, 0
train_loss, tps, eval_every = 0.0, 0.0, 500
progress = run / "progress.json"
last = run / "last.pt"
cfg = run / "config.json"
if progress.exists():
    p = json.loads(progress.read_text())
    step = p.get("step", 0)
    train_loss = p.get("train_loss", 0.0)
    tps = p.get("tokens_per_second", 0.0)
if last.exists():
    c = torch.load(last, map_location="cpu", weights_only=True)
    if not progress.exists():
        step = c["step"]
    best, stale = c["best"], c.get("stale", 0)
if cfg.exists():
    c = json.loads(cfg.read_text())
    total = c["steps"]
    eval_every = c.get("eval_every", 500)
next_val = math.ceil((step + 1) / eval_every) * eval_every
sec_per_step = 0
if tps > 0 and cfg.exists():
    tps_step = c["batch_size"] * c["context"] * c.get("accumulation", 1)
    sec_per_step = tps_step / tps
steps_to_val = max(0, next_val - step)
eta_val_min = (steps_to_val * sec_per_step) / 60 if sec_per_step else 0
print(step, f"{best:.4f}", total, stale, f"{train_loss:.4f}", f"{tps:.0f}", eval_every, f"{sec_per_step:.1f}", f"{next_val}", f"{eta_val_min:.0f}")
PYEOF
)"
      echo "▶ EN COURS : $name"
      echo "  PID $pid | CPU ${cpu:-?}% | RAM ${rss_gb} Go | temps ${elapsed:-?}"
      if [[ "$total" -gt 0 ]]; then
        echo "  Progression : $(bar "$step" "$total")"
      fi
      echo "  Perte train (live) : $train_loss  |  Perte val (best) : $best"
      echo "  Vitesse : ~${tps} tok/s (~${sec_per_step:-?} s/step) — prochaine val step ${next_val:-?} (~${eta_val_min:-?} min)"
      echo "  Objectifs : 3,0 ○  |  2,0 ○  |  1,5 ○"
      if [[ -f "$run_dir/metrics.jsonl" ]]; then
        echo
        echo "  Dernières mesures :"
        tail -5 "$run_dir/metrics.jsonl" | "$PY" -c "
import sys, json, math
for line in sys.stdin:
    m = json.loads(line)
    s = m.get('step', '?')
    v = m.get('val_loss')
    t = m.get('train_loss')
    p = m.get('perplexity')
    if v is not None:
        print(f'    step {s:>6}  train {t:.3f}  val {v:.4f}  ppl {p:.1f}')
    elif s == 0:
        print(f'    step {s:>6}  baseline val {v:.4f}')
" 2>/dev/null || tail -3 "$run_dir/metrics.jsonl"
      fi
      echo
    fi
  done < <(find "$ROOT/runs" -name training.lock 2>/dev/null | sort -r)

  if [[ "$active_count" -gt 1 ]]; then
    echo "⚠ ATTENTION : $active_count entraînements en parallèle !"
    echo "  Lancer : ./scripts/stop_all_training.sh && ./scripts/install_watchdog.sh"
    echo
  fi

  if [[ "$active" -eq 0 ]]; then
    echo "○ Aucun entraînement actif"
    echo
  fi

  # Pipeline auto 3 phases (launchd ou nohup)
  local pipeline_ok=0
  if launchctl print "gui/$(id -u)/com.dattyai.fastest" 2>/dev/null | grep -q "state = running"; then
    pipeline_ok=1
    echo "⟳ Watchdog actif (launchd) — phases 1→2→3 enchaînées auto"
  elif [[ -f "$ROOT/runs/fastest.pid" ]]; then
    fpid="$(tr -d '[:space:]' < "$ROOT/runs/fastest.pid" 2>/dev/null || true)"
    if [[ -n "$fpid" ]] && kill -0 "$fpid" 2>/dev/null; then
      pipeline_ok=1
      echo "⟳ Pipeline auto actif (PID $fpid) — phases 1→2→3"
    fi
  fi
  if [[ "$pipeline_ok" -eq 1 ]]; then
    tail -1 "$ROOT/runs/fastest.log" 2>/dev/null | sed 's/^/  /' || true
    echo
  fi
  if [[ -f "$ROOT/runs/xxl_pipeline.pid" ]]; then
    xpid="$(tr -d '[:space:]' < "$ROOT/runs/xxl_pipeline.pid" 2>/dev/null || true)"
    if [[ -n "$xpid" ]] && kill -0 "$xpid" 2>/dev/null; then
      echo "⟳ Pipeline XXL en préparation (PID $xpid) — 100k docs → modèle 55M"
      tail -1 "$ROOT/runs/xxl_pipeline.log" 2>/dev/null | sed 's/^/  /' || true
      echo
    fi
  fi

  # Meilleurs checkpoints
  echo "───────────────────────────────────────────────────────────"
  echo "  Meilleurs modèles (perte validation)"
  echo "───────────────────────────────────────────────────────────"
  "$PY" - <<'PYEOF'
import json, math
from pathlib import Path
root = Path("runs")
rows = []
for metrics in sorted(root.glob("*/metrics.jsonl")):
    best, step = math.inf, 0
    for line in metrics.read_text().splitlines():
        m = json.loads(line)
        v = m.get("val_loss")
        if v is not None and v < best:
            best, step = v, m.get("step", 0)
    if best < math.inf:
        rows.append((best, metrics.parent.name, step))
rows.sort()
for best, name, step in rows[:8]:
    flag = " ★" if best < 3.5 else ""
    print(f"  {best:.4f}  {name:<28} step {step}{flag}")
PYEOF

  echo
  echo "───────────────────────────────────────────────────────────"
  local log="$ROOT/runs/fastest.log"
  [[ -f "$log" ]] || log="$ROOT/runs/legal_large_25k_ram15/train.log"
  if [[ -f "$log" ]]; then
    echo "  Journal (dernières lignes)"
    tail -3 "$log" 2>/dev/null | sed 's/^/  /'
  fi
  echo
  echo "  Commandes :"
  echo "    ./scripts/progress.sh --watch     # suivi live"
  echo "    tail -f runs/legal_large_25k_ram15/metrics.jsonl"
  echo "    tail -f runs/fastest.log          # pipeline auto 3 phases"
  echo "═══════════════════════════════════════════════════════════"
}

cd "$ROOT"
if $WATCH; then
  while true; do show; sleep "$INTERVAL"; done
else
  show
fi
