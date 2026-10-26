"""Analyse déterministe des requêtes pour le reranking RAG."""

import re
import unicodedata
from dataclasses import dataclass, field

from .chunks import tokenize


def normalize_text(text):  # Uniformise une requête pour la détection de motifs.
    text = unicodedata.normalize("NFC", text).casefold()  # Normalise casse et accents.
    text = text.replace("'", " ").replace("’", " ").replace("‑", "-")  # Unifie apostrophes et tirets.
    text = re.sub(r"\s+", " ", text).strip()  # Réduit les espaces multiples.
    return text  # Renvoie le texte normalisé.


@dataclass(frozen=True)
class ProfessionSpec:  # Décrit une profession réglementée détectable.
    profession_id: str  # Identifiant interne.
    phrases: tuple[str, ...]  # Expressions multi-mots en premier.
    token_aliases: tuple[str, ...] = ()  # Tokens unitaires complémentaires.


@dataclass
class QueryAnalysis:  # Représentation explicable d'une requête analysée.
    raw_query: str  # Requête brute.
    normalized: str  # Requête normalisée.
    tokens: tuple[str, ...]  # Tokens de la requête.
    profession: str | None = None  # Métier détecté.
    profession_confidence: float = 0.0  # Confiance 0-1.
    intents: tuple[str, ...] = ()  # Intentions détectées.
    active_domains: frozenset[str] = frozenset()  # Domaines explicitement pertinents.
    search_query: str = ""  # Requête nettoyée pour le retrieval.
    is_vague: bool = False  # Requête trop imprécise pour chercher.
    debug: dict = field(default_factory=dict)  # Traces de détection.


@dataclass(frozen=True)
class PreparedQuery:  # Résultat du nettoyage d'une requête utilisateur.
    raw_query: str  # Requête brute.
    search_query: str  # Requête exploitable pour le retrieval.
    is_vague: bool  # Indique si la requête manque de substance juridique.
    vague_hint: str  # Message d'aide si la requête est trop vague.


# Formules de politesse et bruit conversationnel à retirer avant recherche.
FILLER_PHRASES = (
    "s il te plait",
    "s il vous plait",
    "s il te plaît",
    "s il vous plaît",
    "stp",
    "svp",
    "merci beaucoup",
    "merci",
    "bonjour",
    "salut",
    "coucou",
    "bonsoir",
    "hello",
    "hey",
)

FILLER_TOKENS = frozenset({  # Tokens isolés sans valeur juridique.
    "stp", "svp", "merci", "bonjour", "salut", "coucou", "bonsoir", "hello", "hey",
    "svp", "please", "svp", "stp",
})

# Mots génériques insuffisants seuls pour lancer une recherche.
GENERIC_QUERY_TOKENS = frozenset({
    "question", "loi", "juridique", "droit", "legal", "juridique", "info", "infos",
    "information", "besoin", "aide", "expliquer", "explication", "quoi", "comment",
    "pourquoi", "peux", "peut", "pourrais", "pourriez", "donne", "donner", "dis",
    "dire", "moi", "stp", "svp", "merci",
})

# Mots vides français ignorés pour évaluer la substance d'une requête.
STOPWORDS = frozenset({
    "de", "du", "des", "le", "la", "les", "un", "une", "au", "aux", "en", "et", "ou",
    "a", "d", "l", "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "on", "y", "ne", "pas", "que", "qui", "dont", "par", "pour", "sur", "sous",
    "dans", "avec", "sans", "entre", "vers", "est", "sont", "été", "etre", "être",
})

VAGUE_HINT = (  # Message affiché quand la requête est trop imprécise.
    "Votre question est trop vague pour interroger le corpus. "
    "Précisez le sujet juridique, par exemple : "
    "« secret professionnel avocat », « prescription civile », "
    "« jurisprudence responsabilité médicale »."
)


# Registre extensible des professions réglementées.
PROFESSION_REGISTRY: tuple[ProfessionSpec, ...] = (
    ProfessionSpec(
        "avocat",
        ("mandataire de justice", "conseil national des barreaux", "barreau", "avocat", "avocats"),
        ("rin",),
    ),
    ProfessionSpec(
        "medecin",
        ("secret medical", "code de la sante publique", "praticien", "medecin", "médecin", "docteur"),
    ),
    ProfessionSpec(
        "notaire",
        ("chambre des notaires", "profession notariale", "notaire", "notaires"),
        ("notariat",),
    ),
)

# Intentions jurisprudentielles — expressions complètes d'abord.
JURISPRUDENCE_STRONG_PHRASES = (
    "cour de cassation",
    "decision de justice",
    "décision de justice",
    "chambre civile",
    "chambre commerciale",
    "chambre sociale",
    "chambre criminelle",
    "jurisprudence",
    "arret de cassation",
    "arrêt de cassation",
    "pourvoi",
    "que disent les tribunaux",
    "solution jurisprudentielle",
)

JURISPRUDENCE_MODERATE_PHRASES = (
    "tribunaux",
    "tribunal",
    "cassation",
    "arret",
    "arrêt",
    "decision",
    "décision",
    "cour d appel",
    "cour d'appel",
)

# Intentions règles professionnelles.
PROFESSIONAL_RULES_PHRASES = (
    "conflit d interets",
    "conflit d'intérêts",
    "obligations deontologiques",
    "obligations déontologiques",
    "regles deontologiques",
    "règles déontologiques",
    "deontologie",
    "déontologie",
    "reglement interieur national",
    "règlement intérieur national",
    "mentions legales",
    "mentions légales",
    "obligations professionnelles",
    "secret professionnel",
    "site internet",
    "rin",
)

# Domaines sensibles activés explicitement par la requête.
DOMAIN_PHRASES = {
    "health": (
        "secret medical",
        "secret médical",
        "medecin",
        "médecin",
        "sante publique",
        "santé publique",
        "malade",
        "hopital",
        "hôpital",
        "pharmacien",
    ),
    "defense": (
        "secret de la defense",
        "secret de la défense",
        "defense nationale",
        "défense nationale",
        "ministere de la defense",
        "ministère de la défense",
        "secret defense",
        "secret défense",
        "classifie",
        "classifié",
    ),
}


def _contains_phrase(normalized, phrase):  # Vérifie la présence d'une expression normalisée.
    return phrase in normalized  # Recherche exacte sur texte normalisé.


def _contains_any(normalized, phrases):  # Vérifie si une expression de la liste est présente.
    return any(_contains_phrase(normalized, phrase) for phrase in phrases)  # Teste chaque expression.


def detect_profession(normalized, tokens):  # Détecte la profession et sa confiance.
    best_id, best_score, best_match = None, 0.0, ""  # Meilleur candidat trouvé.
    token_set = set(tokens)  # Ensemble des tokens de la requête.
    for spec in PROFESSION_REGISTRY:  # Parcourt le registre des professions.
        for phrase in spec.phrases:  # Teste les expressions multi-mots.
            if _contains_phrase(normalized, normalize_text(phrase)):  # Correspondance phrase.
                score = 0.92 if " " in phrase else 0.78  # Phrase complète = confiance plus haute.
                if score > best_score:  # Conserve le meilleur score.
                    best_id, best_score, best_match = spec.profession_id, score, phrase  # Mémorise.
        for alias in spec.token_aliases:  # Teste les alias tokenisés.
            if alias in token_set:  # Correspondance token.
                score = 0.70  # Confiance modérée pour un alias court.
                if score > best_score:  # Conserve le meilleur score.
                    best_id, best_score, best_match = spec.profession_id, score, alias  # Mémorise.
    return best_id, best_score, best_match  # Renvoie la profession détectée.


def detect_intents(normalized, profession, profession_confidence):  # Infère les intentions de recherche.
    intents = []  # Accumule les intentions détectées.
    if _contains_any(normalized, JURISPRUDENCE_STRONG_PHRASES):  # Jurisprudence explicite.
        intents.append("jurisprudence_strong")  # Intention forte.
    elif _contains_any(normalized, JURISPRUDENCE_MODERATE_PHRASES):  # Indices jurisprudentiels.
        intents.append("jurisprudence_moderate")  # Intention modérée.
    if _contains_any(normalized, PROFESSIONAL_RULES_PHRASES):  # Règles professionnelles.
        intents.append("professional_rules")  # Intention déontologique.
    if profession and profession_confidence >= 0.45 and "jurisprudence_strong" not in intents:  # Métier sans jurisprudence.
        if "professional_rules" in intents or any(word in normalized for word in ("mention", "site", "obligation", "deontolog", "déontolog")):  # Contexte réglementaire.
            intents.append("professional_rules")  # Renforce l'intention métier.
    if not intents:  # Aucune intention spécialisée.
        intents.append("general_legal")  # Recherche juridique générale.
    return tuple(dict.fromkeys(intents))  # Déduplique en conservant l'ordre.


def detect_active_domains(normalized, profession):  # Active les domaines sensibles pertinents pour la requête.
    active = set()  # Domaines explicitement demandés.
    for domain, phrases in DOMAIN_PHRASES.items():  # Parcourt les domaines configurés.
        if _contains_any(normalized, phrases):  # Le domaine est mentionné.
            active.add(domain)  # Active le domaine.
    if profession == "medecin":  # Profession médicale détectée.
        active.add("health")  # Active le domaine santé.
    return frozenset(active)  # Renvoie les domaines actifs.


def _strip_fillers(normalized):  # Retire les formules de politesse d'une requête normalisée.
    text = f" {normalized} "  # Ajoute des bornes pour éviter les faux positifs partiels.
    for phrase in FILLER_PHRASES:  # Parcourt les expressions à retirer.
        text = text.replace(f" {phrase} ", " ")  # Supprime chaque formule entourée d'espaces.
    return re.sub(r"\s+", " ", text).strip()  # Réduit les espaces résiduels.


def prepare_query(query):  # Nettoie une requête et détecte si elle est trop vague.
    raw = (query or "").strip()  # Nettoie la requête brute.
    if not raw:  # Requête vide.
        return PreparedQuery(raw, "", True, "Posez une question juridique précise.")  # Refuse la recherche.
    normalized = _strip_fillers(normalize_text(raw))  # Retire politesse et bruit.
    tokens = tuple(token for token in tokenize(normalized) if token not in FILLER_TOKENS)  # Filtre les fillers résiduels.
    search_query = " ".join(tokens).strip()  # Reconstruit la requête de recherche.
    if not search_query:  # Plus rien d'exploitable après nettoyage.
        return PreparedQuery(raw, "", True, VAGUE_HINT)  # Requête trop vague.
    substantive = [token for token in tokens if token not in GENERIC_QUERY_TOKENS and token not in STOPWORDS]  # Tokens juridiquement utiles.
    is_vague = len(substantive) == 0  # Trop générique si aucun mot de fond.
    return PreparedQuery(raw, search_query, is_vague, VAGUE_HINT if is_vague else "")  # Renvoie la requête préparée.


def analyze_query(query):  # Analyse complète d'une requête utilisateur.
    prepared = prepare_query(query)  # Nettoie la requête avant analyse.
    raw = prepared.raw_query  # Conserve la requête brute.
    normalized = normalize_text(prepared.search_query or raw)  # Normalise la requête utile.
    tokens = tuple(tokenize(prepared.search_query or raw))  # Tokenise la requête de recherche.
    profession, confidence, match = detect_profession(normalized, tokens)  # Détecte le métier.
    intents = detect_intents(normalized, profession, confidence)  # Infère les intentions.
    active_domains = detect_active_domains(normalized, profession)  # Active les domaines sensibles.
    return QueryAnalysis(  # Construit l'analyse explicable.
        raw_query=raw,
        normalized=normalized,
        tokens=tokens,
        profession=profession,
        profession_confidence=confidence,
        intents=intents,
        active_domains=active_domains,
        search_query=prepared.search_query,
        is_vague=prepared.is_vague,
        debug=dict(profession_match=match, tokens=list(tokens), prepared_search_query=prepared.search_query),
    )
