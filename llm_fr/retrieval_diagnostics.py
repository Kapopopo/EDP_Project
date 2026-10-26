"""Indicateurs de confiance retrieval pour la synthèse cabinet."""

from .synthesis_export import count_sources_by_domain


def _top_score(hits):
    scores = [h.get("score") for h in hits if h.get("score") is not None]
    return max(scores) if scores else None


def confidence_label(hits):
    """Heuristique simple : fort / moyen / faible selon le meilleur score."""
    top = _top_score(hits)
    if top is None:
        return "inconnue"
    if top >= 0.65:
        return "élevée"
    if top >= 0.35:
        return "moyenne"
    return "faible"


def format_retrieval_diagnostics(question, hits, analysis=None):
    """Ligne compacte pour le terminal (après la synthèse)."""
    if not hits:
        return "Retrieval : aucun fragment — reformulez ou vérifiez l'index."
    by_domain = count_sources_by_domain(hits)
    domain_bits = ", ".join(f"{k}={v}" for k, v in sorted(by_domain.items()))
    top = _top_score(hits)
    score_bit = f"score max {top:.3f}" if top is not None else "scores non disponibles"
    conf = confidence_label(hits)
    intent_bit = ""
    if analysis is not None and getattr(analysis, "intents", None):
        intent_bit = f" | intents: {', '.join(sorted(analysis.intents))}"
    prof = getattr(analysis, "profession", None) if analysis else None
    prof_bit = f" | métier détecté: {prof}" if prof else ""
    return f"Retrieval — confiance {conf} ({score_bit}) | {len(hits)} source(s) [{domain_bits}]{prof_bit}{intent_bit}"
