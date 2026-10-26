"""Rédaction structurée à partir des extraits RAG (API compatible OpenAI ou modèle local)."""

import json
import os
import urllib.error
import urllib.request
from typing import Callable, Optional

from .rag_chat import build_rag_prompt, source_badge, truncate_excerpt

REDACT_SYSTEM = (
    "Tu es l'assistant de recherche d'un cabinet d'avocats français. "
    "Tu rédiges UNIQUEMENT à partir des extraits fournis, sans inventer de textes, articles ou décisions. "
    "Structure : (1) rappel de la question, (2) cadre juridique / textes, (3) jurisprudence si présente, "
    "(4) points pratiques, (5) limites (ce que les extraits ne permettent pas de conclure). "
    "Cite les sources entre crochets [1], [2], etc. Français juridique clair. "
    "Si les extraits sont insuffisants, dis-le explicitement."
)

REDACT_SYSTEM_STRICT = (
    "Tu es l'assistant de recherche d'un cabinet d'avocats français. "
    "Tu rédiges UNIQUEMENT à partir des extraits fournis, sans inventer de textes, articles ou décisions. "
    "Mode citations strictes : chaque puce contient UNE seule affirmation factuelle, "
    "immédiatement suivie des identifiants de source entre crochets, ex. [S1] ou [S1][S3]. "
    "N'utilise que les identifiants [S1], [S2], … listés dans le message utilisateur. "
    "Structure : (1) rappel de la question, (2) cadre juridique / textes, (3) jurisprudence si présente, "
    "(4) points pratiques, (5) limites (ce que les extraits ne permettent pas de conclure). "
    "Français juridique clair. Si les extraits sont insuffisants, dis-le explicitement."
)

DEFAULT_API_BASE = "https://api.openai.com/v1"
DEFAULT_API_MODEL = "gpt-4o-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 1400


def llm_api_key():
    return (os.environ.get("RAG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()


def llm_api_base():
    return (os.environ.get("RAG_LLM_BASE_URL") or DEFAULT_API_BASE).rstrip("/")


def llm_api_model():
    return (os.environ.get("RAG_LLM_MODEL") or DEFAULT_API_MODEL).strip()


def build_redaction_system(strict_citations=False):
    """Prompt système pour la rédaction (standard ou citations strictes [S1]…)."""
    return REDACT_SYSTEM_STRICT if strict_citations else REDACT_SYSTEM


def _source_label(index, strict_citations):
    return f"S{index}" if strict_citations else str(index)


def build_redaction_user_prompt(
    question,
    hits,
    max_excerpt_chars=420,
    max_context_chars=12000,
    strict_citations=False,
):
    """Construit le message utilisateur pour la rédaction."""
    lines = [f"Question du client : {question.strip()}", "", "Extraits indexés (seule source autorisée) :"]
    used = 0
    selected = []
    for index, hit in enumerate(hits, 1):
        title = hit.get("title") or hit.get("code") or hit["source"]
        badge = source_badge(hit)
        excerpt = truncate_excerpt(hit.get("text") or "", max_excerpt_chars)
        sid = _source_label(index, strict_citations)
        block = f"[{sid}] ({badge}) {title}\n{excerpt}"
        if used + len(block) > max_context_chars and selected:
            break
        lines.append(block)
        used += len(block)
        selected.append(hit)
    lines.append("")
    if strict_citations:
        lines.append(
            "Rédige le mémo de synthèse demandé. "
            "Chaque puce = une affirmation + citations [S1], [S2], … (une ou plusieurs sources par puce)."
        )
    else:
        lines.append("Rédige le mémo de synthèse demandé.")
    return "\n\n".join(lines), selected


def redact_via_api(
    question,
    hits,
    temperature=0.25,
    max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    strict_citations=False,
):
    """Appelle une API chat compatible OpenAI. Retourne None si indisponible."""
    key = llm_api_key()
    if not key or not hits:
        return None
    user_prompt, _ = build_redaction_user_prompt(question, hits, strict_citations=strict_citations)
    payload = {
        "model": llm_api_model(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": build_redaction_system(strict_citations)},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        f"{llm_api_base()}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content")
        return (content or "").strip() or None
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None


def redact_via_local(
    question,
    hits,
    model,
    tokenizer,
    count=400,
    temperature=0.45,
    top_p_k=40,
    max_context_chars=2200,
    strict_citations=False,
):
    """Génère un mémo avec le checkpoint local (même stack que --generer)."""
    if not hits:
        return None
    prompt, _ = build_rag_prompt(question, hits, max_context_chars=max_context_chars, max_excerpt_chars=360)
    system = build_redaction_system(strict_citations)
    cite_hint = "[S1], [S2]… (une affirmation par puce)" if strict_citations else "[1], [2]…"
    extended = (
        f"{system}\n\n{prompt}\n\n"
        f"Rédige un mémo structuré en français (sections numérotées), en citant {cite_hint}\nMémo :"
    )
    pieces = []
    from .chat import stream_reply

    for piece in stream_reply(model, tokenizer, extended, count, temperature, top_p_k):
        pieces.append(piece)
    text = "".join(pieces).strip()
    return text or None


def redact_answer(
    question,
    hits,
    backend="auto",
    local_runner: Optional[Callable[[], Optional[str]]] = None,
    strict_citations=False,
):
    """
    backend: auto | api | local
    local_runner: callable sans argument qui retourne le texte (modèle déjà chargé).
    """
    backend = (backend or "auto").lower()
    if backend == "off":
        return None, "off"

    if backend in ("auto", "api") and llm_api_key():
        text = redact_via_api(question, hits, strict_citations=strict_citations)
        if text:
            return text, "api"

    if backend in ("auto", "local") and local_runner is not None:
        text = local_runner()
        if text:
            return text, "local"

    return None, "unavailable"


def append_redaction_to_synthesis(extractive_text, redacted_text, backend_label):
    if not redacted_text:
        return extractive_text
    header = {
        "api": "Mémo rédigé (API — relecture avocat obligatoire)",
        "local": "Mémo rédigé (modèle local — relecture avocat obligatoire)",
    }.get(backend_label, "Mémo rédigé (IA)")
    return f"{extractive_text}\n\n{'═' * 60}\n{header}\n{'═' * 60}\n\n{redacted_text}"
