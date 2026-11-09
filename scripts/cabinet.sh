#!/usr/bin/env bash
# Point d'entrée cabinet — recherche juridique locale (Légifrance + Judilibre + métier).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CMD="${1:-help}"
shift || true

usage() {
  cat <<'EOF'
Usage : ./scripts/cabinet.sh <commande> [options]

  status              État des index RAG et du checkpoint local
  ask "question"      Synthèse (extraits officiels, sans génération)
  chat                Session interactive (./scripts/rag.sh)
  demo                Démo non interactive (statut + 2 questions)
  test                Tests auto rapides (unitaires + eval golden)
  test-full           Tests + eval 2000 questions (long)
  export "question"   Synthèse → fichier Markdown (logs/synthese.md)

Exemples :
  ./scripts/cabinet.sh ask "secret professionnel avocat"
  ./scripts/cabinet.sh chat --sources
  ./scripts/cabinet.sh export "mentions légales site avocat"

Variables utiles (.env) :
  RAG_AUDIT_LOG=logs/rag_audit.jsonl   journal des questions
  RAG_OLLAMA_MODEL=flint:40b-a2b-qat-int4  modèle Ollama pour ./scripts/rag.sh --rediger (défaut : local)
  OPENAI_API_KEY / RAG_LLM_API_KEY     seulement avec --rediger --backend api (extraits envoyés à l'API)
EOF
}

mkdir -p logs

case "$CMD" in
  help|-h|--help)
    usage
    ;;
  status)
    exec ./scripts/rag_status.sh
    ;;
  ask)
    if [ $# -lt 1 ]; then
      echo "Usage: ./scripts/cabinet.sh ask \"votre question\"" >&2
      exit 1
    fi
    exec ./scripts/ask_synthese.sh "$@"
    ;;
  chat)
    exec ./scripts/rag.sh "$@"
    ;;
  demo)
    exec ./scripts/demo_rag_non_interactif.sh
    ;;
  test)
    exec ./scripts/run_all_auto_tests.sh --ci
    ;;
  test-full)
    exec ./scripts/run_all_auto_tests.sh --full
    ;;
  export)
    if [ $# -lt 1 ]; then
      echo "Usage: ./scripts/cabinet.sh export \"votre question\" [fichier.md]" >&2
      exit 1
    fi
    OUT="${2:-logs/synthese.md}"
    exec .venv/bin/python -m llm_fr.ask_synthese_cli "$1" --export "$OUT" --sources
    ;;
  *)
    echo "Commande inconnue : $CMD" >&2
    usage >&2
    exit 1
    ;;
esac
