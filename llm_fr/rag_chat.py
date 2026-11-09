"""Chat local avec RAG : recherche documentaire + génération ou synthèse extractive."""

import argparse  # Lit les options du terminal.
import math  # Valide la température de génération.
import re  # Découpe les phrases pour la synthèse extractive.
from pathlib import Path  # Localise le checkpoint et l'index.

import torch  # Charge le modèle de génération.
from tokenizers import Tokenizer  # Recharge le tokenizer du checkpoint.

from .chat import stream_reply  # Réutilise le décodage progressif.
from .chunks import tokenize  # Tokenise les requêtes pour la synthèse.
from .model import LanguageModel  # Reconstruit l'architecture sauvegardée.
from .query_understanding import VAGUE_HINT, prepare_query
from .rag_index import RagIndex  # Charge l'index TF-IDF local.


DISCLAIMER = (  # Avertissement affiché à chaque session RAG générative.
    "RAG local — les extraits proviennent du corpus indexé. "
    "La réponse générée peut être incomplète ou erronée : relecture avocat obligatoire."
)

SYNTHESIS_DISCLAIMER = (  # Avertissement du mode synthèse sans génération.
    "Mode synthèse enrichi — faits juridiques, textes, jurisprudence et fiches métier "
    "(Légifrance / Judilibre / métier curé), extraits uniquement, sans génération du modèle. "
    "Relecture avocat obligatoire."
)

SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+")  # Coupe les phrases françaises courantes.

_SESSION_EXIT_COMMANDS = frozenset(
    {
        "/quitter",
        "quitter",
        "/quit",
        "quit",
        "/exit",
        "exit",
        "/q",
        "q",
    }
)


def is_session_exit_command(message):
    """Reconnaît les commandes de sortie du chat (avec ou sans slash)."""
    return message.strip().lower() in _SESSION_EXIT_COMMANDS

INDEX_CANDIDATES = ("data/rag/index_legal", "data/rag/index_legifrance", "data/rag/index_judilibre", "data/rag/index_full", "data/rag/index")  # Priorise l'index combiné.


def default_index():  # Choisit le meilleur index RAG disponible sur disque.
    return next((path for path in INDEX_CANDIDATES if (Path(path) / "manifest.json").is_file()), INDEX_CANDIDATES[-1])  # Retombe sur l'index léger.


def resolve_checkpoint_path(explicit=None):  # Sélectionne un checkpoint local pour la génération.
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return str(path)
    from .hardware import profile

    deploy = profile().get("deploy_model", "large")
    xxl = (
        "runs/legal_xxl_adaptive_burn_refine/best.pt",
        "runs/legal_xxl_adaptive_burn/best.pt",
        "runs/legal_xxl_adaptive_refine2/best.pt",
        "runs/legal_xxl_adaptive_refine/best.pt",
        "runs/legal_xxl_adaptive/best.pt",
    )
    large = (
        "runs/legal_large_25k_mps_refine2/best.pt",
        "runs/legal_large_25k_fast/best.pt",
        "runs/pilot/last.pt",
    )
    candidates = xxl + large if deploy == "xxl" else large + xxl
    return next((p for p in candidates if Path(p).is_file()), None)


def truncate_excerpt(text, max_chars):  # Borne un extrait pour respecter le contexte du modèle.
    text = text.strip()  # Retire les espaces superflus.
    if len(text) <= max_chars:  # Conserve les extraits déjà courts.
        return text  # Rien à tronquer.
    cut = text[:max_chars].rsplit(" ", 1)[0]  # Coupe proprement au dernier mot entier.
    return (cut or text[:max_chars]).rstrip() + "…"  # Ajoute une ellipse si tronqué.


def build_rag_prompt(question, hits, max_context_chars=900, max_excerpt_chars=280):  # Construit l'amorce avec les extraits trouvés.
    if not question.strip():  # Refuse une question vide.
        raise ValueError("Question non vide requise.")  # Explique l'erreur.
    lines = ["Extraits du corpus :"]  # En-tête du contexte documentaire.
    used = len(lines[0])  # Compte les caractères déjà utilisés.
    selected = []  # Conserve les extraits réellement injectés.
    for index, hit in enumerate(hits, 1):  # Parcourt les fragments pertinents.
        title = hit.get("title") or hit["source"]  # Titre lisible pour la citation.
        excerpt = truncate_excerpt(hit["text"], max_excerpt_chars)  # Borne la longueur de l'extrait.
        block = f"[{index}] {title} ({hit['source']})\n{excerpt}"  # Formate le bloc cité.
        if used + len(block) + 2 > max_context_chars and selected:  # Arrête si le contexte est saturé.
            break  # Ne garde que les extraits qui tiennent dans la fenêtre.
        lines.append(block)  # Ajoute l'extrait au prompt.
        used += len(block) + 1  # Met à jour la taille du contexte.
        selected.append(hit)  # Mémorise l'extrait utilisé.
    if not selected:  # Injecte au moins le premier extrait si la fenêtre est très petite.
        hit = hits[0]  # Prend le meilleur résultat.
        title = hit.get("title") or hit["source"]  # Titre de secours.
        excerpt = truncate_excerpt(hit["text"], max_excerpt_chars)  # Tronque l'extrait.
        lines.append(f"[1] {title} ({hit['source']})\n{excerpt}")  # Ajoute un seul bloc minimal.
        selected = [hit]  # Conserve la référence.
    prompt = "\n\n".join(lines) + f"\n\nQuestion : {question.strip()}\nRéponse :"  # Assemble l'amorce finale.
    return prompt, selected  # Renvoie le prompt et les sources citées.


def split_sentences(text):  # Découpe un texte en phrases pour la synthèse extractive.
    text = text.strip()  # Retire les espaces superflus.
    if not text:  # Ignore les textes vides.
        return []  # Aucune phrase à extraire.
    return [sentence.strip() for sentence in SENTENCE_SPLIT.split(text) if sentence.strip()]  # Conserve les phrases non vides.


def extractive_summary(text, query, max_sentences=4, fallback_chars=600):  # Sélectionne les phrases les plus liées à la question.
    sentences = split_sentences(text)  # Découpe le fragment en phrases.
    if not sentences:  # Retombe sur un extrait brut si la découpe échoue.
        return truncate_excerpt(text, fallback_chars)  # Tronque proprement le texte source.
    query_tokens = set(tokenize(query))  # Mots-clés de la question.
    if not query_tokens:  # Sans requête exploitable, affiche le début du fragment.
        return truncate_excerpt(text, fallback_chars)  # Tronque proprement le texte source.
    scored = []  # Classe les phrases par recouvrement lexical.
    for index, sentence in enumerate(sentences):  # Parcourt chaque phrase du fragment.
        overlap = len(query_tokens & set(tokenize(sentence)))  # Compte les mots communs.
        if overlap:  # Ignore les phrases sans lien lexical.
            scored.append((overlap, index, sentence))  # Mémorise le score et la position.
    if not scored:  # Aucune phrase ne matche la requête.
        return truncate_excerpt(text, fallback_chars)  # Retombe sur un extrait brut.
    top = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]  # Garde les meilleures phrases.
    ordered = sorted(index for _, index, _ in top)  # Réordonne dans l'ordre du texte source.
    return " ".join(sentences[index] for index in ordered)  # Assemble la synthèse extractive.


def source_badge(hit):  # Indique la provenance officielle d'un extrait.
    domain = hit.get("domain")  # Domaine du document source.
    if domain == "legifrance":  # Texte normatif officiel.
        return "Légifrance"  # Badge Légifrance.
    if domain == "judilibre":  # Jurisprudence Cour de cassation.
        return "Judilibre"  # Badge Judilibre.
    if domain == "cabinet_curated":
        return "Métier curé"
    return "Corpus"  # Badge générique.


def format_synthese(question, hits, max_sentences=4, fallback_chars=600):  # Formate une réponse documentaire sans génération.
    from .synthesis_enriched import format_synthese_enriched

    return format_synthese_enriched(question, hits, max_sentences=max_sentences, fallback_chars=fallback_chars)


def synthese_reply(
    index,
    question,
    top_k=None,
    max_sentences=None,
    fallback_chars=None,
    redact=False,
    redact_backend="auto",
    strict_citations=False,
    local_model=None,
    local_tokenizer=None,
    domain_filter=None,
):  # Produit une synthèse sans charger le modèle (rédaction IA optionnelle).
    from .llm_redact import (
        append_redaction_to_synthesis,
        redact_answer,
        redact_via_local,
        redaction_unavailable_hint,
    )
    from .synthesis_config import (
        DEFAULT_FALLBACK_CHARS,
        DEFAULT_MAX_SENTENCES,
        DEFAULT_TOP_K,
        RETRIEVAL_TOP_K,
    )
    from .query_understanding import analyze_query
    from .synthesis_enriched import dedupe_hits, format_synthese_enriched

    top_k = DEFAULT_TOP_K if top_k is None else top_k
    max_sentences = DEFAULT_MAX_SENTENCES if max_sentences is None else max_sentences
    fallback_chars = DEFAULT_FALLBACK_CHARS if fallback_chars is None else fallback_chars
    if not question.strip():  # Refuse une question vide.
        raise ValueError("Question non vide requise.")  # Explique l'erreur.
    prepared = prepare_query(question)  # Nettoie la requête et détecte les formulations vagues.
    if prepared.is_vague:  # Requête trop imprécise pour interroger le corpus.
        return dict(
            sources=[],
            hits=[],
            text=f"Question : {question.strip()}\n\n{prepared.vague_hint or VAGUE_HINT}",
            analysis=None,
        )
    analysis = analyze_query(question)
    pool = max(top_k, RETRIEVAL_TOP_K)
    hits = dedupe_hits(index.search(question, top_k=pool, domain_filter=domain_filter), max_items=top_k)
    text = format_synthese_enriched(question, hits, max_sentences=max_sentences, fallback_chars=fallback_chars)
    redaction_backend = None
    if redact and hits:
        runner = None
        if local_model is not None and local_tokenizer is not None:
            runner = lambda: redact_via_local(
                question,
                hits,
                local_model,
                local_tokenizer,
                strict_citations=strict_citations,
            )
        redacted, redaction_backend = redact_answer(
            question,
            hits,
            backend=redact_backend,
            local_runner=runner,
            strict_citations=strict_citations,
        )
        if redacted:
            text = append_redaction_to_synthesis(text, redacted, redaction_backend)
        elif redact_backend != "off":
            hint = redaction_unavailable_hint(redact_backend, has_local_model=local_model is not None)
            text += f"\n\n(Rédaction IA non disponible : {hint}.)"
    return dict(
        sources=hits,
        hits=hits,
        text=text,
        redaction_backend=redaction_backend,
        analysis=analysis,
    )


def format_sources(hits):  # Formate la liste des sources pour l'affichage terminal.
    lines = ["Sources :"]  # En-tête des citations.
    for index, hit in enumerate(hits, 1):  # Numérote chaque source.
        title = hit.get("title") or hit.get("code") or hit["source"]  # Titre affiché.
        score = hit.get("score")  # Score de pertinence éventuel.
        suffix = f" (score {score})" if score is not None else ""  # Ajoute le score si disponible.
        badge = source_badge(hit)  # Indique la provenance officielle.
        lines.append(f"  [{index}] [{badge}] {title} — {hit['source']}{suffix}")  # Ligne de citation.
    return "\n".join(lines)  # Renvoie le bloc formaté.


def load_model(checkpoint_path):  # Charge le modèle et le tokenizer depuis un checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # Lit la sauvegarde locale.
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])  # Récupère le vocabulaire exact.
    model = LanguageModel(**checkpoint["architecture"])  # Recrée les dimensions du réseau.
    model.load_state_dict(checkpoint["model"])  # Charge les poids appris.
    model.eval()  # Désactive le dropout pour l'inférence.
    step = checkpoint.get("step", "?")  # Retient l'étape d'entraînement affichée.
    context = checkpoint["architecture"]["context"]  # Taille de fenêtre du modèle.
    del checkpoint  # Libère les états inutiles à l'inférence.
    return model, tokenizer, step, context  # Renvoie les objets prêts à l'emploi.


def rag_reply(index, model, tokenizer, question, top_k=3, count=100, temperature=0.7, top_p_k=40, max_context_chars=900):  # Produit une réponse RAG complète.
    if count < 1 or top_p_k < 1 or not math.isfinite(temperature) or temperature <= 0:  # Valide les paramètres de génération.
        raise ValueError("count et top-k doivent être positifs ; température finie et strictement positive.")  # Message d'erreur explicite.
    hits = index.search(question, top_k=top_k)  # Recherche les fragments pertinents.
    if not hits:  # Gère l'absence de résultats.
        return dict(prompt="", sources=[], hits=[], text="Aucun extrait pertinent trouvé dans le corpus indexé.")  # Réponse vide explicite.
    prompt, sources = build_rag_prompt(question, hits, max_context_chars=max_context_chars)  # Construit l'amorce documentée.
    pieces = list(stream_reply(model, tokenizer, prompt, count, temperature, top_p_k))  # Génère la réponse token par token.
    return dict(prompt=prompt, sources=sources, hits=hits, text="".join(pieces))  # Renvoie la réponse et les métadonnées.


def synthese_chat(
    index_dir,
    top_k=None,
    max_sentences=None,
    fallback_chars=None,
    redact=False,
    redact_backend="auto",
    strict_citations=False,
    checkpoint_path=None,
    domain_filter=None,
    export_path=None,
    show_sources=False,
    show_diagnostics=True,
    audit=True,
):  # Lance une session interactive sans génération.
    from .llm_redact import llm_api_base, llm_api_key, normalize_backend, ollama_available, ollama_base_url, ollama_model
    from .retrieval_diagnostics import format_retrieval_diagnostics
    from .synthesis_audit import append_audit_record
    from .synthesis_config import DEFAULT_FALLBACK_CHARS, DEFAULT_MAX_SENTENCES, DEFAULT_TOP_K
    from .synthesis_filters import normalize_domain_filter

    top_k = DEFAULT_TOP_K if top_k is None else top_k
    domain_filter = normalize_domain_filter(domain_filter)
    max_sentences = DEFAULT_MAX_SENTENCES if max_sentences is None else max_sentences
    fallback_chars = DEFAULT_FALLBACK_CHARS if fallback_chars is None else fallback_chars
    index = RagIndex.load(index_dir)  # Charge l'index documentaire.
    local_model = local_tokenizer = None
    if redact:
        redact_backend = normalize_backend(redact_backend)
        ollama_up = redact_backend in ("auto", "ollama") and ollama_available()
        # Le checkpoint local n'est chargé que s'il est demandé, ou en secours quand Ollama ne répond pas.
        use_local = redact_backend == "local" or (redact_backend == "auto" and not ollama_up)
        ckpt = resolve_checkpoint_path(checkpoint_path) if use_local else None
        if ckpt:
            print(f"Rédaction : modèle local {ckpt}", flush=True)
            local_model, local_tokenizer, _, _ = load_model(ckpt)
        elif redact_backend in ("auto", "ollama"):
            print(f"Rédaction : Ollama local ({ollama_model()} sur {ollama_base_url()})", flush=True)
            if not ollama_up:
                print("Attention : Ollama ne répond pas (lancer `ollama serve`, vérifier `ollama list`).", flush=True)
        elif redact_backend == "api":
            print(
                f"Rédaction : API distante {llm_api_base()} (--backend api) — les extraits quittent la machine.",
                flush=True,
            )
            if not llm_api_key():
                print("Attention : OPENAI_API_KEY ou RAG_LLM_API_KEY attendue (voir .env.example).", flush=True)
        else:
            print("Rédaction : checkpoint local attendu (runs/…/best.pt).", flush=True)
    print(f"Index : {index_dir} ({index.n_docs} fragments)", flush=True)  # Indique la taille de l'index.
    if index.semantic is not None:  # Recherche hybride activée.
        print(f"Hybride : TF-IDF + sémantique métier ({len(index.semantic.meta)} fragments curés)", flush=True)  # Indique le sous-index.
    if index.cabinet_index is not None:
        print(f"Index cabinet : {index.cabinet_index.n_docs} fiches métier (RIN, mentions légales)", flush=True)
    if domain_filter:
        print(f"Filtre actif : {domain_filter}", flush=True)
    if export_path:
        print(f"Export Markdown : {export_path}", flush=True)
    print(SYNTHESIS_DISCLAIMER, flush=True)  # Affiche l'avertissement du mode synthèse.
    print(
        f"Réponse : jusqu'à {top_k} sources, {max_sentences} phrases/source"
        f"{' + mémo IA' if redact else ''}. Sortie : /quitter (ou quitter, exit).",
        flush=True,
    )
    try:  # Permet une sortie propre avec Ctrl+C ou Ctrl+D.
        while True:  # Attend plusieurs messages sans relancer Python.
            message = input("\nToi > ").strip()  # Lit une question dans le terminal.
            if is_session_exit_command(message):
                break  # Termine la boucle.
            if not message:  # Ignore une entrée vide.
                continue  # Attend une véritable question.
            result = synthese_reply(
                index,
                message,
                top_k,
                max_sentences,
                fallback_chars,
                redact=redact,
                redact_backend=redact_backend,
                strict_citations=strict_citations,
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                domain_filter=domain_filter,
            )
            print(result["text"], flush=True)  # Affiche les extraits sélectionnés.
            if show_sources and result.get("hits"):
                print(format_sources(result["hits"]), flush=True)
            if show_diagnostics:
                print(
                    format_retrieval_diagnostics(
                        message, result.get("hits") or [], result.get("analysis")
                    ),
                    flush=True,
                )
            if audit:
                logged = append_audit_record(
                    message,
                    result.get("hits"),
                    index_path=index_dir,
                    extra={"redact": redact, "domain_filter": domain_filter},
                )
                if logged:
                    print(f"(audit → {logged})", flush=True)
            if export_path:
                from .synthesis_export import export_synthese_markdown

                saved = export_synthese_markdown(
                    export_path,
                    message,
                    result["text"],
                    hits=result.get("hits"),
                    index_path=index_dir,
                )
                print(f"(exporté → {saved})", flush=True)
    except (KeyboardInterrupt, EOFError):  # Gère les interruptions clavier habituelles.
        print("\nSession fermée.", flush=True)  # Termine sans trace d'erreur.


def rag_chat(checkpoint_path, index_dir, top_k=3, count=100, temperature=0.7, top_p_k=40):  # Lance une session interactive RAG.
    torch.set_num_threads(4)  # Laisse de la marge CPU si un entraînement tourne en parallèle.
    index = RagIndex.load(index_dir)  # Charge l'index documentaire.
    path = Path(checkpoint_path)  # Conserve le chemin du checkpoint.
    print(f"Index : {index_dir} ({len(index.meta)} fragments)", flush=True)  # Indique la taille de l'index.
    print(f"Sauvegarde : {path}", flush=True)  # Indique le modèle utilisé.
    print(DISCLAIMER, flush=True)  # Affiche l'avertissement juridique.
    print("Sortie : /quitter (ou quitter, exit). Chaque message est indépendant.", flush=True)
    try:  # Permet une sortie propre avec Ctrl+C ou Ctrl+D.
        while True:  # Attend plusieurs messages sans relancer Python.
            message = input("\nToi > ").strip()  # Lit une question dans le terminal.
            if is_session_exit_command(message):
                break  # Termine la boucle.
            if not message:  # Ignore une entrée vide.
                continue  # Attend une véritable question.
            model, tokenizer, step, context = load_model(path)  # Recharge le modèle à chaque message.
            max_context = min(2400, context * 4)  # Contexte RAG élargi pour des réponses plus développées.
            result = rag_reply(index, model, tokenizer, message, top_k, count, temperature, top_p_k, max_context)  # Produit la réponse.
            del model  # Libère le réseau avant la prochaine saisie.
            if result["sources"]:  # Affiche les sources si des extraits ont été trouvés.
                print(format_sources(result["sources"]), flush=True)  # Montre les citations avant la réponse.
            print(f"Modèle (étape {step}) > ", end="", flush=True)  # Affiche le début de la réponse générée.
            print(result["text"], flush=True)  # Affiche le texte produit.
    except (KeyboardInterrupt, EOFError):  # Gère les interruptions clavier habituelles.
        print("\nSession fermée.", flush=True)  # Termine sans trace d'erreur.


if __name__ == "__main__":  # Expose python -m llm_fr.rag_chat.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide intégrée.
    parser.add_argument("--checkpoint")  # Permet de choisir explicitement un checkpoint.
    parser.add_argument("--index", default=None)  # Dossier de l'index TF-IDF (auto si omis).
    parser.add_argument("--generer", action="store_true", help="Active la génération du modèle (désactivée par défaut).")  # Mode RAG génératif expérimental.
    parser.add_argument(
        "--rediger",
        action="store_true",
        help="Ajoute un mémo rédigé après les extraits (modèle local via Ollama par défaut).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "ollama", "local", "api"),
        default="auto",
        help=(
            "Moteur de rédaction pour --rediger. auto (défaut) : Ollama local puis checkpoint local, "
            "jamais d'API distante ; ollama ; local (checkpoint) ; "
            "api : API OpenAI-compatible distante, les extraits quittent la machine."
        ),
    )
    parser.add_argument(
        "--strict-citations",
        action="store_true",
        help="Avec --rediger : une affirmation par puce, citations [S1], [S2], …",
    )
    parser.add_argument("--synthese", action="store_true", help="Explicite le mode synthèse (comportement par défaut).")  # Alias de compatibilité.
    parser.add_argument("--top-k", type=int, default=None)  # Nombre d'extraits à récupérer.
    parser.add_argument("--count", type=int, default=320)  # Longueur maximale de la réponse générée.
    parser.add_argument("--temperature", type=float, default=0.7)  # Diversité de la génération.
    parser.add_argument("--top-p-k", type=int, default=40, dest="top_p_k")  # Limite des candidats par token.
    parser.add_argument("--max-sentences", type=int, default=None, dest="max_sentences")  # Phrases par extrait (défaut : config enrichie).
    parser.add_argument(
        "--domain",
        choices=("all", "legifrance", "judilibre", "cabinet_curated"),
        default="all",
        help="Filtre les résultats par provenance.",
    )
    parser.add_argument("--export", dest="export_path", default=None, help="Fichier Markdown de sortie (écrasé à chaque réponse).")
    parser.add_argument("--sources", action="store_true", help="Affiche la liste détaillée des sources après chaque réponse.")
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Masque la ligne de confiance retrieval.",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="N'écrit pas dans RAG_AUDIT_LOG même si défini.",
    )
    args = parser.parse_args()  # Lit les options demandées.
    index_dir = args.index or default_index()  # Utilise l'index complet dès qu'il est prêt.
    from .synthesis_config import DEFAULT_TOP_K

    top_k = args.top_k if args.top_k is not None else (6 if args.generer else DEFAULT_TOP_K)
    if args.generer:  # Mode RAG génératif expérimental.
        checkpoint_path = args.checkpoint or resolve_checkpoint_path()
        if not checkpoint_path:
            raise SystemExit("Aucun checkpoint local trouvé pour --generer.")
        rag_chat(checkpoint_path, index_dir, top_k, args.count, args.temperature, args.top_p_k)  # Ouvre le terminal interactif.
    else:  # Mode synthèse par défaut : extraits officiels sans modèle.
        synthese_chat(
            index_dir,
            top_k,
            args.max_sentences,
            None,
            redact=args.rediger,
            redact_backend=args.backend,
            strict_citations=args.strict_citations,
            checkpoint_path=args.checkpoint,
            domain_filter=args.domain,
            export_path=args.export_path,
            show_sources=args.sources,
            show_diagnostics=not args.no_diagnostics,
            audit=not args.no_audit,
        )
