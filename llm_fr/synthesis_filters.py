"""Filtres de domaine sur les résultats RAG."""

VALID_DOMAINS = frozenset({"legifrance", "judilibre", "cabinet_curated", "all"})


def normalize_domain_filter(value):
    if not value or value == "all":
        return None
    key = value.strip().lower()
    if key not in VALID_DOMAINS:
        raise ValueError(f"Domaine invalide : {value} (legifrance, judilibre, cabinet_curated, all)")
    return key


def filter_hits_by_domain(hits, domain_filter):
    if not domain_filter:
        return list(hits)
    return [h for h in hits if (h.get("domain") or "") == domain_filter]
