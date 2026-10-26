"""Paramètres du mode synthèse enrichi (extraits officiels, sans LLM)."""

# Nombre de fragments affichés après déduplication.
DEFAULT_TOP_K = 16
# Candidats récupérés avant déduplication (pool élargi).
RETRIEVAL_TOP_K = 30
# Phrases extraites par source.
DEFAULT_MAX_SENTENCES = 10
# Longueur max si découpage phrase impossible.
DEFAULT_FALLBACK_CHARS = 1400
# Points saillants en tête de réponse.
HIGHLIGHT_BULLETS = 6
HIGHLIGHT_MAX_CHARS = 220
