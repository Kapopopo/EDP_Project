"""Fusion des candidats TF-IDF et sémantiques (RRF)."""

from copy import deepcopy

RRF_K = 60  # Constante de lissage reciprocal rank fusion.


def _hit_identity(hit):  # Clé stable pour dédoublonner les fragments.
    for key in ("chunk_id", "sha256", "article_id"):
        value = hit.get(key)
        if value:
            return str(value)
    return f"{hit.get('source', '')}|{hit.get('title', '')}"


def reciprocal_rank_fusion(ranked_lists, weights=None, rrf_k=RRF_K):  # Fusionne plusieurs classements.
    if not ranked_lists:
        return []
    weights = weights or [1.0] * len(ranked_lists)
    scores = {}
    rows = {}
    for weight, hits in zip(weights, ranked_lists):
        for rank, hit in enumerate(hits, start=1):
            key = _hit_identity(hit)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            if key not in rows:
                rows[key] = deepcopy(hit)
    merged = []
    for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        row = rows[key]
        row["fusion_score"] = round(score, 6)
        base = float(row.get("score") or 0.0)
        row["score"] = round(max(base, min(1.0, score * 3.0)), 4)
        if row.get("retrieval") != "semantic":
            row.setdefault("retrieval", "hybrid")
        merged.append(row)
    return merged


def merge_retrieval_candidates(tfidf_hits, semantic_hits, semantic_weight=0.55):  # Combine deux pools de candidats.
    if not semantic_hits:
        return list(tfidf_hits)
    if not tfidf_hits:
        return list(semantic_hits)
    tfidf_weight = max(0.2, 1.0 - semantic_weight)
    for hit in tfidf_hits:
        hit.setdefault("retrieval", "tfidf")
    for hit in semantic_hits:
        hit["retrieval"] = "semantic"
    return reciprocal_rank_fusion([tfidf_hits, semantic_hits], weights=[tfidf_weight, semantic_weight])
