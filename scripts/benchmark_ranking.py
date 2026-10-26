#!/usr/bin/env python3
"""Compare le ranking avant/après reranking sur des requêtes de référence."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Racine du projet.
if str(ROOT) not in sys.path:  # Garantit l'import du package local.
    sys.path.insert(0, str(ROOT))  # Ajoute la racine au PYTHONPATH.

from llm_fr.query_understanding import analyze_query
from llm_fr.rag_chat import default_index
from llm_fr.rag_index import RagIndex
from llm_fr.rerank import rerank_hits

QUERIES = [
    "secret professionnel avocat",
    "jurisprudence responsabilité avocat",
    "obligations déontologiques avocat conflit d'intérêts",
    "mentions légales site internet avocat",
    "secret médical médecin",
    "obligations liées au secret de la défense nationale",
]


def summarize(hits):  # Résume les trois premiers résultats.
    lines = []  # Lignes de sortie.
    for index, hit in enumerate(hits[:3], 1):  # Parcourt le top 3.
        badge = hit.get("domain", "?")  # Source du fragment.
        title = (hit.get("title") or "")[:70]  # Titre tronqué.
        score = hit.get("score")  # Score affiché.
        base = hit.get("base_score", score)  # Score TF-IDF initial.
        lines.append(f"  {index}. [{badge}] base={base:.3f} final={score:.3f} | {title}")  # Formate la ligne.
    return "\n".join(lines)  # Renvoie le résumé.


def main():  # Lance le benchmark local.
    index = RagIndex.load(default_index())  # Charge l'index actif.
    print(f"Index : {default_index()} ({len(index.meta)} fragments)\n")  # Affiche l'index utilisé.
    for query in QUERIES:  # Parcourt les requêtes de référence.
        analysis = analyze_query(query)  # Analyse la requête.
        raw = index.retrieve(query, top_k=40)  # Récupère les candidats bruts.
        ranked = rerank_hits(raw, analysis=analysis, top_k=5)  # Applique le reranking.
        print(f"=== {query}")  # Affiche la requête.
        print(f"profession={analysis.profession} ({analysis.profession_confidence:.2f}) intents={list(analysis.intents)} domains={sorted(analysis.active_domains)}")  # Contexte.
        print("AVANT (TF-IDF):")  # Section baseline.
        print(summarize(raw))  # Top 3 brut.
        print("APRÈS (rerank):")  # Section améliorée.
        print(summarize(ranked))  # Top 3 reranké.
        print()  # Sépare les requêtes.


if __name__ == "__main__":  # Point d'entrée script.
    main()  # Exécute le benchmark.
