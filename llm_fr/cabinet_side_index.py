"""Mini-index cabinet (RIN, mentions légales) fusionné à la recherche principale."""

from pathlib import Path

from .hybrid_retrieval import reciprocal_rank_fusion
from .query_understanding import QueryAnalysis, analyze_query

DEFAULT_CABINET_INDEX = Path("data/rag/index_cabinet")
DEFAULT_CABINET_CHUNKS = Path("data/rag/chunks_cabinet.jsonl")
DEFAULT_CABINET_SOURCE = Path("data/legal/curated/cabinet_metier.jsonl")

# Poids RRF du mini-index cabinet (canal secondaire).
CABINET_RRF_WEIGHT_BOOST = 1.35
CABINET_RRF_WEIGHT_DEFAULT = 1.05
CABINET_RRF_WEIGHT_LOW = 0.55


def _cabinet_boost_query(analysis: QueryAnalysis) -> bool:
    """Requêtes où les fiches cabinet (RIN, mentions, honoraires) sont prioritaires."""
    norm = analysis.normalized
    triggers = (
        "mention",
        "honoraires",
        " rin",
        "rin ",
        "reglement interieur",
        "règlement intérieur",
        "reglement interieur national",
        "règlement intérieur national",
        "site internet",
        "publicite",
        "publicité",
        "convention honoraires",
        "barreau",
    )
    return any(token in norm for token in triggers) or norm.startswith("rin")


def cabinet_merge_weight(analysis: QueryAnalysis) -> float:
    """Retourne 0 pour ignorer la fusion cabinet, sinon un poids RRF."""
    intents = set(analysis.intents)
    if "jurisprudence_strong" in intents:
        return 0.0
    norm = analysis.normalized
    if "secret" in norm and "professionnel" in norm and analysis.profession == "avocat":
        return 0.0
    if _cabinet_boost_query(analysis) and "jurisprudence_strong" not in intents:
        return CABINET_RRF_WEIGHT_BOOST
    if "professional_rules" in intents:
        return CABINET_RRF_WEIGHT_DEFAULT
    return CABINET_RRF_WEIGHT_LOW


def should_skip_semantic_for_analysis(analysis: QueryAnalysis) -> bool:
    """Désactive le canal sémantique curé quand Judilibre / TF-IDF doit primer."""
    if "jurisprudence_strong" in analysis.intents:
        return True
    norm = analysis.normalized
    if (
        "secret" in norm
        and "professionnel" in norm
        and analysis.profession == "avocat"
        and "jurisprudence" not in norm
    ):
        return True
    return False


def load_cabinet_index(path=None):
    from .rag_index import RagIndex

    root = Path(path or DEFAULT_CABINET_INDEX)
    if not (root / "manifest.json").is_file():
        return None
    return RagIndex.load(root)


def merge_cabinet_candidates(main_hits, cabinet_index, query, analysis=None, pool=18):
    if cabinet_index is None:
        return main_hits
    analysis = analysis or analyze_query(query)
    weight = cabinet_merge_weight(analysis)
    if weight <= 0:
        return main_hits
    side = cabinet_index.retrieve(query, top_k=pool)
    if not side:
        return main_hits
    for row in side:
        row["retrieval"] = "cabinet_curated"
    return reciprocal_rank_fusion([main_hits, side], weights=[1.0, weight])
