"""Poids et seuils centralisés pour le reranking RAG."""

# Multiplicateur de candidats récupérés avant reranking (base TF-IDF).
CANDIDATE_POOL_MULTIPLIER = 8
CANDIDATE_POOL_MIN = 40
CANDIDATE_POOL_MAX = 120

# Seuils de confiance métier.
PROFESSION_CONFIDENCE_HIGH = 0.75
PROFESSION_CONFIDENCE_MEDIUM = 0.45

# Boosts Judilibre (additifs sur le score cosinus 0-1).
JUDILIBRE_STRONG_JURISPRUDENCE_BOOST = 0.26
JUDILIBRE_MODERATE_JURISPRUDENCE_BOOST = 0.10
JUDILIBRE_WEAK_PROFESSION_ONLY_BOOST = 0.03

# Boosts corpus professionnel (Légifrance métier).
PROFESSION_RULES_HIGH_BOOST = 0.16
PROFESSION_RULES_MEDIUM_BOOST = 0.08
PROFESSION_BAR_DECREE_BOOST = 0.10
NOTAIRE_RULES_HIGH_BOOST = 0.14
NOTAIRE_RULES_MEDIUM_BOOST = 0.07

# Pénalités contextuelles (soustractives, plafonnées).
IRRELEVANT_HEALTH_PENALTY = 0.22
IRRELEVANT_DEFENSE_PENALTY = 0.18
SECRET_ENUMERATION_PENALTY = 0.14
EXAM_PROGRAM_PENALTY = 0.12
IRRELEVANT_PROFESSION_PENALTY = 0.10

# Plancher du score final après reranking.
MIN_FINAL_SCORE = 0.01

# Score maximum après ajustements (évite les boosts cumulés excessifs).
MAX_SCORE_CAP = 1.0

# Recherche hybride (corpus métier curé + TF-IDF global).
SEMANTIC_RETRIEVAL_WEIGHT = 0.55
SEMANTIC_POOL_SIZE = 35

# Boosts pack métier curé (additifs).
CURATED_PACK_HIGH_BOOST = 0.12
CURATED_PACK_MATCH_BOOST = 0.08

# Seuil minimal pour afficher un résultat (après reranking).
MIN_DISPLAY_SCORE = 0.22
