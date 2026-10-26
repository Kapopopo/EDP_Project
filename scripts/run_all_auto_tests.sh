#!/usr/bin/env bash
# Tous les tests automatiques — rien à taper dans rag.sh.
# Usage :
#   ./scripts/run_all_auto_tests.sh          # rapide (golden + unitaires + smoke citations)
#   ./scripts/run_all_auto_tests.sh --full   # + eval 2000 questions (long, ~chargement index)
#   ./scripts/run_all_auto_tests.sh --ci     # comme rapide, sans benchmark perf lourd
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
MODE="${1:-quick}"

echo "=== Tests unitaires ==="
"$PY" -m unittest discover -s tests -q

echo ""
echo "=== Smoke citations ==="
"$PY" scripts/eval_redact_smoke.py

echo ""
echo "=== Eval golden (19 cas stricts) ==="
"$PY" scripts/run_eval.py --suite all

if [ "$MODE" = "full" ]; then
  if [ ! -f eval/question_bank_2000.jsonl ]; then
    echo "=== Génération banque 2000 questions ==="
    "$PY" scripts/generate_question_bank_2000.py
  fi
  echo ""
  echo "=== Batch eval 2000 (smoke retrieval, seuil 95 %) ==="
  "$PY" scripts/run_batch_eval.py --quiet --min-rate 0.95 --report logs/batch_eval_failures.jsonl
fi

if [ "$MODE" != "ci" ]; then
  echo ""
  echo "=== Perf smoke (optionnel) ==="
  "$PY" scripts/benchmark_perf.py || true
fi

echo ""
echo "=== Tous les tests auto OK ($MODE) ==="
