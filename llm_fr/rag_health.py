"""État des index RAG et du modèle local (sans charger le corpus entier en mémoire longue)."""

import json
from pathlib import Path

from .rag_chat import default_index, resolve_checkpoint_path

INDEX_CANDIDATES = (
    "data/rag/index_legal",
    "data/rag/index_cabinet",
    "data/rag/index_legifrance",
    "data/rag/index_judilibre",
)


def _read_manifest(path):
    manifest = path / "manifest.json"
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "n_docs": data.get("n_docs"),
        "n_terms": data.get("n_terms"),
        "meta_lazy": data.get("meta_lazy"),
    }


def collect_health():
    active = default_index()
    indexes = []
    for rel in INDEX_CANDIDATES:
        info = _read_manifest(Path(rel))
        if info:
            info["active"] = rel == active
            indexes.append(info)
    ckpt = resolve_checkpoint_path()
    return {
        "default_index": active,
        "indexes": indexes,
        "checkpoint": ckpt,
        "audit_log": __import__("os").environ.get("RAG_AUDIT_LOG") or None,
    }


def print_health():
    h = collect_health()
    print(f"Index actif : {h['default_index']}")
    for idx in h["indexes"]:
        flag = " (actif)" if idx.get("active") else ""
        print(f"  • {idx['path']}{flag} — {idx['n_docs']} fragments, {idx['n_terms']} termes")
    if h["checkpoint"]:
        print(f"Checkpoint rédaction/génération : {h['checkpoint']}")
    else:
        print("Checkpoint local : aucun (API ou --generer indisponible sans poids)")
    if h["audit_log"]:
        print(f"Journal audit : {h['audit_log']}")
    else:
        print("Journal audit : désactivé (export RAG_AUDIT_LOG=logs/rag_audit.jsonl pour activer)")


if __name__ == "__main__":
    print_health()
