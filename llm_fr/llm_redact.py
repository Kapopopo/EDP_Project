"""Rédaction structurée à partir des extraits RAG.

Moteurs (``backend``) :
- ``auto`` (défaut) : modèle local via Ollama, puis checkpoint local ; n'appelle JAMAIS une API distante ;
- ``ollama`` : Ollama seul (API compatible OpenAI exposée sur la machine) ;
- ``local`` : checkpoint local seul (runs/…/best.pt) ;
- ``api`` : API compatible OpenAI distante (api.openai.com par défaut), uniquement sur demande explicite ;
- ``off`` : pas de rédaction.
"""

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
DEFAULT_OLLAMA_BASE = "http://localhost:11434/v1"
# Nom du modèle tel qu'affiché par `ollama list` ; surcharge possible avec RAG_OLLAMA_MODEL.
DEFAULT_OLLAMA_MODEL = "flint:40b-a2b-qat-int4"
DEFAULT_MAX_OUTPUT_TOKENS = 1400
API_TIMEOUT_S = 120
OLLAMA_TIMEOUT_S = 600  # Un modèle local peut mettre plusieurs minutes à charger puis répondre.
# Les modèles locaux « à raisonnement » (Flint…) dépensent d'abord des tokens à réfléchir (champ `reasoning`)
# avant d'écrire la réponse : avec le budget de l'API, le mémo restait vide. Surcharge : RAG_OLLAMA_MAX_TOKENS.
DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS = 8000

REDACTION_BACKENDS = ("auto", "ollama", "local", "api", "off")


def llm_api_key():
    return (os.environ.get("RAG_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()


def llm_api_base():
    return (os.environ.get("RAG_LLM_BASE_URL") or DEFAULT_API_BASE).rstrip("/")


def llm_api_model():
    return (os.environ.get("RAG_LLM_MODEL") or DEFAULT_API_MODEL).strip()


def ollama_base_url():
    return (os.environ.get("RAG_OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE).rstrip("/")


def ollama_model():
    return (os.environ.get("RAG_OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()


def ollama_max_tokens():
    try:
        return int(os.environ.get("RAG_OLLAMA_MAX_TOKENS") or DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS)
    except ValueError:
        return DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS


def ollama_available(timeout=2):
    """Vrai si un serveur Ollama répond à l'adresse configurée (GET /models)."""
    try:
        with urllib.request.urlopen(f"{ollama_base_url()}/models", timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def normalize_backend(backend):
    """Valide le moteur demandé ; ``None`` ou vide vaut ``auto``."""
    name = (backend or "auto").strip().lower()
    if name not in REDACTION_BACKENDS:
        raise ValueError(f"Moteur de rédaction inconnu : {backend!r} (attendu : {', '.join(REDACTION_BACKENDS)})")
    return name


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


def _chat_completion(
    base_url,
    model,
    question,
    hits,
    headers=None,
    temperature=0.25,
    max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    strict_citations=False,
    timeout=API_TIMEOUT_S,
):
    """POST /chat/completions (format OpenAI). Retourne le texte ou None si indisponible."""
    user_prompt, _ = build_redaction_user_prompt(question, hits, strict_citations=strict_citations)
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": build_redaction_system(strict_citations)},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            return None
        content = choices[0].get("message", {}).get("content")
        return (content or "").strip() or None
    except (urllib.error.URLError, json.JSONDecodeError, OSError, ValueError):
        return None


def redact_via_api(
    question,
    hits,
    temperature=0.25,
    max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    strict_citations=False,
):
    """Appelle une API chat compatible OpenAI distante. Retourne None si indisponible."""
    key = llm_api_key()
    if not key or not hits:
        return None
    return _chat_completion(
        llm_api_base(),
        llm_api_model(),
        question,
        hits,
        headers={"Authorization": f"Bearer {key}"},
        temperature=temperature,
        max_tokens=max_tokens,
        strict_citations=strict_citations,
        timeout=API_TIMEOUT_S,
    )


def redact_via_ollama(
    question,
    hits,
    temperature=0.25,
    max_tokens=None,
    strict_citations=False,
):
    """Rédige avec le modèle local servi par Ollama. Retourne None si Ollama ne répond pas."""
    if not hits:
        return None
    return _chat_completion(
        ollama_base_url(),
        ollama_model(),
        question,
        hits,
        temperature=temperature,
        max_tokens=max_tokens or ollama_max_tokens(),
        strict_citations=strict_citations,
        timeout=OLLAMA_TIMEOUT_S,
    )


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
    backend: auto | ollama | local | api | off
      - auto   : Ollama, puis checkpoint local ; jamais d'API distante, même si une clé est définie ;
      - api    : API distante (OpenAI-compatible), seulement si demandée explicitement.
    local_runner: callable sans argument qui retourne le texte (modèle déjà chargé).
    Retourne (texte ou None, libellé du moteur utilisé ou "unavailable"/"off").
    """
    backend = normalize_backend(backend)
    if backend == "off":
        return None, "off"

    if backend == "api":
        text = redact_via_api(question, hits, strict_citations=strict_citations)
        return (text, "api") if text else (None, "unavailable")

    if backend in ("auto", "ollama"):
        text = redact_via_ollama(question, hits, strict_citations=strict_citations)
        if text:
            return text, "ollama"

    if backend in ("auto", "local") and local_runner is not None:
        text = local_runner()
        if text:
            return text, "local"

    return None, "unavailable"


def redaction_unavailable_hint(backend, has_local_model=False):
    """Explique quoi vérifier quand aucun moteur n'a pu rédiger."""
    backend = normalize_backend(backend)
    if backend == "api":
        return "OPENAI_API_KEY ou RAG_LLM_API_KEY" if not llm_api_key() else "API injoignable"
    if backend == "local":
        return "checkpoint local introuvable (runs/…/best.pt)"
    hint = (
        f"Ollama injoignable ou modèle « {ollama_model()} » absent sur {ollama_base_url()} "
        "(lancer `ollama serve`, vérifier `ollama list`, ou définir RAG_OLLAMA_MODEL)"
    )
    if backend == "auto" and not has_local_model:
        hint += " ; ou checkpoint local (runs/…/best.pt)"
    return hint


def append_redaction_to_synthesis(extractive_text, redacted_text, backend_label):
    if not redacted_text:
        return extractive_text
    header = {
        "api": "Mémo rédigé (API distante — relecture avocat obligatoire)",
        "ollama": "Mémo rédigé (Ollama local — relecture avocat obligatoire)",
        "local": "Mémo rédigé (modèle local — relecture avocat obligatoire)",
    }.get(backend_label, "Mémo rédigé (IA)")
    return f"{extractive_text}\n\n{'═' * 60}\n{header}\n{'═' * 60}\n\n{redacted_text}"
