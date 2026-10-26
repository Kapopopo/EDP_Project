#!/usr/bin/env bash
# Batterie d'évaluation technique rapide (une commande, sans commentaires # en fin de ligne).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"

echo "=== Tests unitaires ==="
"$PY" -m unittest discover -s tests -v

echo ""
echo "=== Index / rebuild ==="
./scripts/monitor_judilibre_rebuild.sh

echo ""
echo "=== meta_offsets ==="
./scripts/warm_meta_offsets.sh

echo ""
echo "=== Benchmark ranking ==="
"$PY" scripts/benchmark_ranking.py

echo ""
echo "=== Eval golden retrieval ==="
"$PY" scripts/run_eval.py

echo ""
echo "=== Smoke citations (extractif + prompts stricts) ==="
"$PY" scripts/eval_redact_smoke.py

echo ""
echo "=== Perf smoke ==="
"$PY" scripts/benchmark_perf.py

echo ""
echo "=== Fin eval smoke ==="
