"""Synthèse documentaire structurée (sans génération LLM)."""

from .chunks import tokenize
from .synthesis_config import (
    DEFAULT_FALLBACK_CHARS,
    DEFAULT_MAX_SENTENCES,
    HIGHLIGHT_BULLETS,
    HIGHLIGHT_MAX_CHARS,
)

LIMITES_INTRO = (
    "Synthèse extractive uniquement : aucune interprétation ni conseil juridique du modèle. "
    "Les extraits peuvent être incomplets ; vérifier les textes officiels et la jurisprudence à jour."
)


def hit_dedup_key(hit):
    """Clé stable pour éviter les doublons d'affichage."""
    article_id = hit.get("article_id")
    if article_id:
        return f"article:{article_id}"
    source = (hit.get("source") or "").strip()
    if "courdecassation.fr" in source.lower():
        return f"decision:{source}"
    chunk_id = hit.get("chunk_id")
    if chunk_id:
        return f"chunk:{chunk_id}"
    title = (hit.get("title") or "").strip()
    return f"raw:{title}:{source[:96]}"


def dedupe_hits(hits, max_items=None):
    """Conserve le meilleur score par document logique."""
    seen = {}
    order = []
    for hit in hits:
        key = hit_dedup_key(hit)
        score = float(hit.get("score") or 0.0)
        if key not in seen or score > float(seen[key].get("score") or 0.0):
            if key not in seen:
                order.append(key)
            seen[key] = hit
    out = [seen[k] for k in order]
    if max_items is not None:
        out = out[:max_items]
    return out


def _sentence_overlap_score(sentence, query_tokens):
    st = set(tokenize(sentence))
    return len(query_tokens & st)


def build_highlights(hits, question, max_bullets=HIGHLIGHT_BULLETS, max_chars=HIGHLIGHT_MAX_CHARS):
    """Bullet points issus des phrases les plus liées à la question (multi-sources)."""
    from .rag_chat import extractive_summary, split_sentences

    query_tokens = set(tokenize(question))
    if not query_tokens:
        return []
    candidates = []
    for hit in hits[:10]:
        text = hit.get("text") or ""
        for index, sentence in enumerate(split_sentences(text)):
            overlap = _sentence_overlap_score(sentence, query_tokens)
            if overlap < 2:
                continue
            candidates.append((overlap, index, sentence.strip(), hit))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    bullets = []
    seen_norm = set()
    for overlap, _, sentence, hit in candidates:
        norm = sentence.casefold()[:80]
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        badge = hit.get("domain") or "corpus"
        line = sentence
        if len(line) > max_chars:
            line = line[: max_chars - 1].rsplit(" ", 1)[0] + "…"
        bullets.append(f"• {line} [{badge}]")
        if len(bullets) >= max_bullets:
            break
    if not bullets and hits:
        excerpt = extractive_summary(hits[0]["text"], question, max_sentences=2, fallback_chars=400)
        if excerpt:
            bullets.append(f"• {excerpt}")
    return bullets


def _group_hits(hits):
    legifrance = []
    judilibre = []
    cabinet = []
    other = []
    for hit in hits:
        domain = hit.get("domain")
        if domain == "legifrance":
            legifrance.append(hit)
        elif domain == "judilibre":
            judilibre.append(hit)
        elif domain == "cabinet_curated":
            cabinet.append(hit)
        else:
            other.append(hit)
    return legifrance, judilibre, cabinet, other


def source_count_line(hits):
    """Ligne de confiance : volume par type de source (badges affichés)."""
    from .rag_chat import source_badge

    n = len(hits)
    n_legi = sum(1 for h in hits if source_badge(h) == "Légifrance")
    n_judi = sum(1 for h in hits if source_badge(h) == "Judilibre")
    n_metier = sum(1 for h in hits if source_badge(h) == "Métier curé")
    return (
        f"{n} source(s) retenue(s) — {n_legi} Légifrance, "
        f"{n_judi} Judilibre, {n_metier} métier curé."
    )


def format_limites_section(hits):
    """Section Limites : rappels méthodologiques et effectifs."""
    lines = ["── Limites ──", LIMITES_INTRO]
    if hits:
        lines.append(source_count_line(hits))
    else:
        lines.append("Aucune source indexée n'a été associée à cette question.")
    return lines


def format_hit_block(index, hit, question, max_sentences, fallback_chars, format_source_line):
    """Formate un bloc source numéroté."""
    from .rag_chat import extractive_summary, source_badge

    title = hit.get("title") or hit.get("code") or hit["source"]
    score = hit.get("score")
    suffix = f" (score {score:.3f})" if score is not None else ""
    badge = source_badge(hit)
    lines = [
        "",
        f"[{index}] [{badge}] {title}{suffix}",
        f"    {format_source_line(hit)}",
        extractive_summary(hit["text"], question, max_sentences, fallback_chars),
    ]
    return lines


def _append_source_section(lines, section_title, section_hits, counter, question, max_sentences, fallback_chars):
    if not section_hits:
        return counter
    lines.append("")
    lines.append(f"── {section_title} ──")
    for hit in section_hits:
        lines.extend(
            format_hit_block(
                counter,
                hit,
                question,
                max_sentences,
                fallback_chars,
                lambda h: h["source"],
            )
        )
        counter += 1
    return counter


def format_synthese_enriched(
    question,
    hits,
    max_sentences=DEFAULT_MAX_SENTENCES,
    fallback_chars=DEFAULT_FALLBACK_CHARS,
):
    """Réponse structurée : faits, textes, jurisprudence, points d'attention, limites."""
    q = question.strip()
    lines = [f"Question : {q}", ""]
    if not hits:
        lines.extend(format_limites_section(hits))
        lines.append(
            "Précisez la question (texte visé, profession, délai, juridiction) ou reformulez avec des termes juridiques."
        )
        return "\n".join(lines)

    highlights = build_highlights(hits, q)
    lines.append("── Faits juridiques ──")
    if highlights:
        lines.append("(extraits les plus liés à la question, sans interprétation du modèle)")
        lines.extend(highlights)
    else:
        lines.append("(aucun extrait saillant automatique ; voir les sections sources ci-dessous)")

    legifrance, judilibre, cabinet, other = _group_hits(hits)
    attention = cabinet + other

    counter = 1
    counter = _append_source_section(
        lines, "Textes", legifrance, counter, q, max_sentences, fallback_chars
    )
    counter = _append_source_section(
        lines, "Jurisprudence", judilibre, counter, q, max_sentences, fallback_chars
    )
    counter = _append_source_section(
        lines, "Points d'attention", attention, counter, q, max_sentences, fallback_chars
    )

    lines.append("")
    lines.extend(format_limites_section(hits))
    return "\n".join(lines)
