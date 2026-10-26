"""Expansion lexicale des requêtes pour améliorer le retrieval TF-IDF."""

import re

from .query_understanding import QueryAnalysis, normalize_text

# Citations d'articles (ex. art 1147) qui noient le pool Judilibre en TF-IDF.
_ARTICLE_CITATION = re.compile(
    r"\bart(?:icle)?\.?\s*(?:l\.?\s*)?\d[\w-]*\b",
    re.IGNORECASE,
)


def strip_legal_citations(text):  # Retire les renvois article/numéro du texte de retrieval.
    cleaned = _ARTICLE_CITATION.sub(" ", text or "")
    return re.sub(r"\s+", " ", cleaned).strip()


def judilibre_retrieval_query(analysis: QueryAnalysis):  # Requête TF-IDF sans citations bruyantes.
    base = analysis.search_query or analysis.raw_query
    return strip_legal_citations(base) or base

# Règles (déclencheurs normalisés → termes additionnels pour le retrieval uniquement).
EXPANSION_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("rin", "reglement interieur national", "règlement intérieur national"), ("deontologie avocat", "profession avocat", "conseil national des barreaux")),
    (("cassation", "cour de cassation"), ("jurisprudence", "arret cour de cassation")),
    (("jurisprudence",), ("arret", "decision justice", "pourvoi")),
    (("deontologie", "déontologie"), ("obligations professionnelles", "regles professionnelles")),
    (("secret professionnel",), ("confidentialite", "devoir secret")),
    (("notaire", "notaires"), ("chambre des notaires", "profession notaire", "ordonnance notariat", "statut notariat")),
    (("notaire", "notaires"), ("deontologie notaire", "déontologie notaire", "2023-1297", "code deontologie des notaires")),
    (("deontologie", "déontologie"), ("code deontologie", "obligations professionnelles")),
    (("responsabilite", "responsabilité"), ("avocat", "faute professionnelle", "indemnisation client")),
    (("jurisprudence",), ("cour de cassation", "arret pourvoi", "judilibre")),
    (("medecin", "médecin"), ("secret medical", "code sante publique")),
    (("defense nationale", "défense nationale"), ("secret defense", "ministere defense")),
    (("mentions legales", "mentions légales"), ("reglement interieur national", "conseil national des barreaux", "site web")),
    (("mentions legales", "mentions légales"), ("rin", "deontologie avocat")),
    (("responsabilite", "responsabilité"), ("faute", "dommage", "indemnisation")),
)


def _triggered(normalized, tokens, triggers):  # Vérifie si un déclencheur est actif.
    token_set = set(tokens)  # Tokens de la requête.
    for trigger in triggers:  # Parcourt les déclencheurs.
        trigger_norm = normalize_text(trigger)  # Normalise le déclencheur.
        if " " in trigger_norm and trigger_norm in normalized:  # Expression multi-mots.
            return True  # Déclencheur trouvé.
        if trigger_norm in token_set:  # Token isolé.
            return True  # Déclencheur trouvé.
    return False  # Aucun déclencheur.


def expand_query(analysis: QueryAnalysis):  # Enrichit la requête pour le retrieval sans altérer l'analyse.
    additions = []  # Termes additionnels à injecter.
    seen = set(analysis.tokens)  # Évite les doublons évidents.
    for triggers, extras in EXPANSION_RULES:  # Parcourt les règles d'expansion.
        if not _triggered(analysis.normalized, analysis.tokens, triggers):  # Règle inactive.
            continue  # Passe à la règle suivante.
        for extra in extras:  # Ajoute les synonymes associés.
            for token in normalize_text(extra).split():  # Découpe en tokens indexables.
                if token and token not in seen:  # Ignore les doublons.
                    additions.append(token)  # Mémorise le terme.
                    seen.add(token)  # Marque comme utilisé.
    base = analysis.search_query or analysis.raw_query  # Utilise la requête nettoyée.
    if not additions:  # Aucune expansion utile.
        return base  # Conserve la requête de recherche.
    return f"{base} {' '.join(additions)}"  # Concatène pour le TF-IDF.
