"""Reranking contextuel des résultats TF-IDF."""

from copy import deepcopy

from .chunks import tokenize
from .document_profile import profile_document
from .query_understanding import GENERIC_QUERY_TOKENS, QueryAnalysis, STOPWORDS, analyze_query
from .legal_curated import classify_curated_pack, curated_priority
from .ranking_config import (
    CANDIDATE_POOL_MAX,
    CANDIDATE_POOL_MIN,
    CANDIDATE_POOL_MULTIPLIER,
    CURATED_PACK_HIGH_BOOST,
    CURATED_PACK_MATCH_BOOST,
    EXAM_PROGRAM_PENALTY,
    IRRELEVANT_DEFENSE_PENALTY,
    IRRELEVANT_HEALTH_PENALTY,
    IRRELEVANT_PROFESSION_PENALTY,
    JUDILIBRE_MODERATE_JURISPRUDENCE_BOOST,
    JUDILIBRE_STRONG_JURISPRUDENCE_BOOST,
    JUDILIBRE_WEAK_PROFESSION_ONLY_BOOST,
    MAX_SCORE_CAP,
    MIN_DISPLAY_SCORE,
    MIN_FINAL_SCORE,
    PROFESSION_BAR_DECREE_BOOST,
    PROFESSION_CONFIDENCE_HIGH,
    PROFESSION_CONFIDENCE_MEDIUM,
    NOTAIRE_RULES_HIGH_BOOST,
    NOTAIRE_RULES_MEDIUM_BOOST,
    PROFESSION_RULES_HIGH_BOOST,
    PROFESSION_RULES_MEDIUM_BOOST,
    SECRET_ENUMERATION_PENALTY,
)


def candidate_pool_size(top_k):  # Calcule le nombre de candidats à récupérer avant reranking.
    return min(CANDIDATE_POOL_MAX, max(CANDIDATE_POOL_MIN, top_k * CANDIDATE_POOL_MULTIPLIER))  # Borne la taille du pool.


PROFESSION_TOKENS = frozenset({"avocat", "avocats", "notaire", "notaires", "medecin", "médecin", "praticien"})  # Mots métier seuls.


def _hit_blob(hit):  # Concatène les champs textuels d'un fragment.
    return " ".join([hit.get("title") or "", hit.get("code") or "", hit.get("text") or ""]).casefold()  # Texte combiné normalisé.


def _substantive_tokens(analysis):  # Extrait les termes juridiques utiles hors métier et mots vides.
    tokens = set(analysis.tokens)  # Tokens de la requête.
    tokens -= PROFESSION_TOKENS  # Retire les professions.
    if analysis.profession:  # Retire la profession détectée.
        tokens.discard(analysis.profession)  # Évite le biais métier seul.
    return [token for token in tokens if token not in STOPWORDS and token not in GENERIC_QUERY_TOKENS]  # Conserve la substance.


def _lexical_overlap(analysis, hit, exclude_profession=False):  # Compte les termes de la requête présents dans le document.
    hit_tokens = set(tokenize(_hit_blob(hit)))  # Tokens du document.
    query_tokens = set(analysis.tokens)  # Tokens de la requête.
    if exclude_profession:  # Ignore le métier pour mesurer la substance juridique.
        query_tokens -= PROFESSION_TOKENS  # Retire les professions génériques.
        if analysis.profession:  # Retire aussi la profession détectée.
            query_tokens.discard(analysis.profession)  # Évite de compter « avocat » seul.
    return len(query_tokens & hit_tokens)  # Nombre de termes communs.


def _intent_flags(analysis):  # Résume les intentions sous forme booléenne.
    intents = set(analysis.intents)  # Ensemble des intentions.
    return {  # Drapeaux pratiques pour le reranking.
        "jurisprudence_strong": "jurisprudence_strong" in intents,
        "jurisprudence_moderate": "jurisprudence_moderate" in intents,
        "professional_rules": "professional_rules" in intents,
        "general_legal": "general_legal" in intents and len(intents) == 1,
    }


def compute_adjustments(hit, analysis, profile):  # Calcule les ajustements explicables d'un résultat.
    adjustments = []  # Liste des deltas appliqués.
    flags = _intent_flags(analysis)  # Intentions booléennes.
    base = float(hit.get("score") or 0.0)  # Score TF-IDF de départ.

    substantive = _substantive_tokens(analysis)  # Termes juridiques hors métier.
    substantive_overlap = len(set(substantive) & set(tokenize(_hit_blob(hit))))  # Recouvrement de fond.

    if profile.source_type == "judilibre":  # Source jurisprudentielle.
        if flags["jurisprudence_strong"]:  # Intention jurisprudentielle forte.
            if substantive_overlap >= 1:  # Le document traite le sujet posé.
                adjustments.append(("jurisprudence_strong_judilibre", JUDILIBRE_STRONG_JURISPRUDENCE_BOOST))  # Boost fort.
            else:  # Jurisprudence hors sujet malgré le mot « avocat ».
                adjustments.append(("jurisprudence_irrelevant_substance", -0.10))  # Recule les arrêts hors sujet.
        elif flags["jurisprudence_moderate"]:  # Intention jurisprudentielle modérée.
            if substantive_overlap >= 1:  # Le document traite le sujet posé.
                adjustments.append(("jurisprudence_moderate_judilibre", JUDILIBRE_MODERATE_JURISPRUDENCE_BOOST))  # Boost modéré.
        elif analysis.profession and analysis.profession_confidence >= PROFESSION_CONFIDENCE_MEDIUM:  # Métier sans jurisprudence.
            adjustments.append(("profession_only_judilibre", JUDILIBRE_WEAK_PROFESSION_ONLY_BOOST))  # Boost faible.
        if flags["professional_rules"] and not flags["jurisprudence_strong"] and not flags["jurisprudence_moderate"]:  # Déontologie sans jurisprudence.
            adjustments.append(("judilibre_without_jurisprudence_intent", -0.04))  # Léger recul.
        if flags["professional_rules"] and "secret" in analysis.normalized:  # Secret professionnel : la jurisprudence peut éclairer la norme.
            if "secret" in (hit.get("text") or "").casefold():  # Extrait aligné sur le secret.
                adjustments.append(("professional_rules_secret_jurisprudence", 0.12))  # Boost modéré ciblé.

    if flags["jurisprudence_strong"] and profile.source_type == "judilibre":  # Jurisprudence explicitement demandée.
        if analysis.profession == "avocat" and analysis.profession_confidence >= PROFESSION_CONFIDENCE_MEDIUM:  # Filtre métier avocat.
            adjustments.append(("jurisprudence_profession_match", 0.05))  # Léger bonus de cohérence métier.

    if flags["professional_rules"] and "mention" in analysis.normalized and analysis.profession == "avocat":  # Mentions légales avocat.
        blob = _hit_blob(hit)  # Texte du document.
        if profile.is_professional_rules_avocat or "reglement interieur national" in blob or "rin" in blob.split():  # Texte RIN / déontologie.
            adjustments.append(("avocat_mentions_legales_rin", 0.14))  # Favorise le RIN.
        elif "site" in blob and "internet" in blob and "avocat" not in blob:  # Site internet sans lien avocat.
            adjustments.append(("generic_site_internet_noise", -0.12))  # Pénalise le bruit homonyme.

    if substantive_overlap >= 2:  # Forte adéquation lexicale requête/document.
        adjustments.append(("substantive_overlap", min(0.10, 0.03 * substantive_overlap)))  # Boost progressif.

    if flags["professional_rules"] and any(token.startswith("deontolog") for token in substantive):  # Requête déontologique.
        blob = _hit_blob(hit)  # Texte du document.
        if "deontolog" in blob:  # Document déontologique.
            adjustments.append(("deontologie_match", 0.12))  # Boost la déontologie.
        if analysis.profession == "notaire" and "notaire" in blob:  # Texte lié à la profession notariale.
            adjustments.append(("notaire_deontologie_context", 0.08))  # Favorise les textes notariaux.

    if analysis.profession == "notaire" and flags["professional_rules"]:  # Requête réglementaire notaire.
        blob_plain = _hit_blob(hit).replace("é", "e").replace("è", "e")  # Normalise les accents.
        if "creation" in blob_plain and "chambre" in blob_plain and "deontolog" not in blob_plain:  # Décret administratif.
            adjustments.append(("notaire_administrative_noise", -0.22))  # Pénalise les décrets de création de chambre.

    if "secret" in analysis.normalized and "professionnel" in analysis.normalized:  # Requête sur le secret professionnel.
        blob = _hit_blob(hit)  # Texte du document.
        if "secret" in blob and "professionnel" in blob:  # Document aligné sur le sujet.
            adjustments.append(("secret_professionnel_match", 0.15))  # Boost la substance recherchée.

    if analysis.profession == "avocat":  # Requête relative aux avocats.
        if profile.is_professional_rules_avocat:  # Texte RIN / déontologie avocat.
            overlap = _lexical_overlap(analysis, hit, exclude_profession=True)  # Termes juridiques partagés.
            boost = 0.0  # Boost métier calculé selon le contexte.
            if overlap > 0 and analysis.profession_confidence >= PROFESSION_CONFIDENCE_HIGH:  # Confiance élevée et sujet partagé.
                boost = PROFESSION_RULES_HIGH_BOOST  # Boost fort par défaut.
            elif overlap > 0 and analysis.profession_confidence >= PROFESSION_CONFIDENCE_MEDIUM:  # Confiance modérée et sujet partagé.
                boost = PROFESSION_RULES_MEDIUM_BOOST  # Boost modéré par défaut.
            if boost and flags["jurisprudence_strong"]:  # L'utilisateur cherche de la jurisprudence.
                adjustments.append(("normative_during_jurisprudence_search", -0.14))  # Recule les textes normatifs.
                boost *= 0.35  # Réduit fortement le boost normatif.
            if boost:  # Applique le boost métier résiduel.
                label = "avocat_professional_rules_high" if boost >= PROFESSION_RULES_MEDIUM_BOOST else "avocat_professional_rules_medium"  # Nom explicite.
                adjustments.append((label, boost))  # Boost déontologie / RIN.
        if not flags["jurisprudence_strong"]:  # Pas de recherche jurisprudentielle explicite.
            if "91-1197" in (hit.get("code") or "").casefold() or "91-1197" in (hit.get("title") or "").casefold():  # Décret barreau.
                if _lexical_overlap(analysis, hit, exclude_profession=True) > 0:  # Le document traite aussi le sujet posé.
                    adjustments.append(("avocat_bar_decret", PROFESSION_BAR_DECREE_BOOST))  # Boost décret profession.

    if analysis.profession == "notaire":  # Requête relative aux notaires.
        if profile.is_professional_rules_notaire:  # Texte profession notaire.
            overlap = _lexical_overlap(analysis, hit, exclude_profession=True)  # Termes juridiques partagés.
            boost = 0.0  # Boost métier calculé selon le contexte.
            if overlap > 0 and analysis.profession_confidence >= PROFESSION_CONFIDENCE_HIGH:  # Confiance élevée et sujet partagé.
                boost = NOTAIRE_RULES_HIGH_BOOST  # Boost fort par défaut.
            elif overlap > 0 and analysis.profession_confidence >= PROFESSION_CONFIDENCE_MEDIUM:  # Confiance modérée et sujet partagé.
                boost = NOTAIRE_RULES_MEDIUM_BOOST  # Boost modéré par défaut.
            if boost and flags["jurisprudence_strong"]:  # L'utilisateur cherche de la jurisprudence.
                adjustments.append(("notaire_normative_during_jurisprudence_search", -0.12))  # Recule les textes normatifs.
                boost *= 0.35  # Réduit fortement le boost normatif.
            if boost:  # Applique le boost métier résiduel.
                label = "notaire_professional_rules_high" if boost >= NOTAIRE_RULES_MEDIUM_BOOST else "notaire_professional_rules_medium"  # Nom explicite.
                adjustments.append((label, boost))  # Boost profession notaire.

    if "health" in profile.topic_domains and "health" not in analysis.active_domains:  # Santé hors contexte.
        penalty = IRRELEVANT_HEALTH_PENALTY  # Pénalité santé par défaut.
        if analysis.profession == "avocat":  # Requête avocat avec bruit médical.
            penalty = max(penalty, 0.20)  # Pénalité légèrement renforcée.
        adjustments.append(("irrelevant_health_domain", -penalty))  # Applique la pénalité.

    if "defense" in profile.topic_domains and "defense" not in analysis.active_domains:  # Défense hors contexte.
        adjustments.append(("irrelevant_defense_domain", -IRRELEVANT_DEFENSE_PENALTY))  # Pénalise la défense.

    if profile.is_secret_enumeration and not any(domain in analysis.active_domains for domain in ("health", "defense")):  # Liste de secrets.
        if not flags["jurisprudence_strong"]:  # Pas une recherche jurisprudentielle ciblée.
            adjustments.append(("secret_enumeration", -SECRET_ENUMERATION_PENALTY))  # Pénalise les énumérations.

    if profile.is_exam_program and (flags["jurisprudence_strong"] or flags["professional_rules"]):  # Programme d'examen hors sujet.
        adjustments.append(("exam_program_noise", -EXAM_PROGRAM_PENALTY))  # Pénalise les programmes.

    if analysis.profession and analysis.profession_confidence >= PROFESSION_CONFIDENCE_MEDIUM:  # Métier détecté avec confiance.
        if analysis.profession not in profile.profession_tags and profile.profession_tags:  # Document d'un autre métier.
            other = next(iter(profile.profession_tags))  # Profession dominante du document.
            if other != analysis.profession:  # Conflit métier requête/document.
                adjustments.append((f"irrelevant_profession_{other}", -IRRELEVANT_PROFESSION_PENALTY))  # Pénalise le métier étranger.

    pack = hit.get("curated_pack") or classify_curated_pack(hit)  # Pack métier curé indexé.
    if pack:
        priority = curated_priority(pack)
        if priority >= 2:
            adjustments.append(("curated_pack_high", CURATED_PACK_HIGH_BOOST))
        elif priority >= 1:
            adjustments.append(("curated_pack", CURATED_PACK_MATCH_BOOST))
        if flags["professional_rules"] and analysis.profession == "notaire" and "notaire" in pack:
            adjustments.append(("curated_notaire_professional", 0.10))
        if flags["professional_rules"] and analysis.profession == "avocat" and "avocat" in pack:
            adjustments.append(("curated_avocat_professional", 0.10))
    if hit.get("semantic_score") is not None and substantive_overlap >= 1:
        adjustments.append(("semantic_substance", min(0.08, 0.03 * float(hit["semantic_score"]))))

    final = base + sum(delta for _, delta in adjustments)  # Score ajusté.
    final = max(MIN_FINAL_SCORE, min(MAX_SCORE_CAP, final))  # Borne le score final.
    return final, adjustments  # Renvoie le score et les ajustements.


def rerank_hits(hits, analysis=None, top_k=5, debug=False):  # Rerank une liste de candidats TF-IDF.
    if not hits:  # Aucun candidat à reranker.
        return []  # Renvoie une liste vide.
    analysis = analysis or analyze_query(hits[0].get("_query", ""))  # Analyse la requête si absente.
    ranked = []  # Résultats rerankés.
    for hit in hits:  # Parcourt chaque candidat.
        row = deepcopy(hit)  # Copie pour ne pas modifier l'original.
        profile = profile_document(row)  # Profile le fragment.
        final_score, adjustments = compute_adjustments(row, analysis, profile)  # Calcule le score ajusté.
        row["base_score"] = row.get("score")  # Conserve le score TF-IDF initial.
        row["score"] = round(final_score, 4)  # Met à jour le score affiché.
        if debug:  # Mode explicabilité développeur.
            row["ranking_debug"] = {  # Détail des ajustements.
                "base_score": row["base_score"],
                "adjustments": [{name: round(delta, 4)} for name, delta in adjustments],
                "profile": {
                    "source_type": profile.source_type,
                    "topic_domains": sorted(profile.topic_domains),
                    "profession_tags": sorted(profile.profession_tags),
                    "is_secret_enumeration": profile.is_secret_enumeration,
                    "is_exam_program": profile.is_exam_program,
                    "is_professional_rules_avocat": profile.is_professional_rules_avocat,
                    "is_professional_rules_notaire": profile.is_professional_rules_notaire,
                },
                "query_analysis": {
                    "profession": analysis.profession,
                    "profession_confidence": round(analysis.profession_confidence, 3),
                    "intents": list(analysis.intents),
                    "active_domains": sorted(analysis.active_domains),
                },
            }
        ranked.append(row)  # Conserve le résultat reranké.
    ranked.sort(key=lambda item: item["score"], reverse=True)  # Trie par score final.
    filtered = [row for row in ranked if float(row.get("score") or 0.0) >= MIN_DISPLAY_SCORE]
    if filtered:
        return filtered[:top_k]
    return ranked[:top_k]
