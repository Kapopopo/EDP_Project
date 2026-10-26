"""Classification des fragments indexés pour le reranking contextuel."""

import unicodedata
from dataclasses import dataclass


def _normalize(text):  # Uniformise un texte pour la détection de motifs.
    return unicodedata.normalize("NFC", (text or "")).casefold()  # Normalise casse et accents.


@dataclass(frozen=True)
class DocumentProfile:  # Profil thématique d'un fragment indexé.
    source_type: str  # legifrance, judilibre, corpus.
    topic_domains: frozenset[str]  # Domaines thématiques détectés.
    profession_tags: frozenset[str]  # Professions réglementées liées au document.
    is_secret_enumeration: bool  # Liste plusieurs types de secrets.
    is_exam_program: bool  # Programme d'examen ou concours.
    is_professional_rules_avocat: bool  # Texte déontologique / RIN avocat.
    is_professional_rules_notaire: bool  # Texte profession notaire.


SECRET_MARKERS = (  # Marqueurs de secrets listés dans un même extrait.
    "secret professionnel",
    "secret medical",
    "secret médical",
    "secret de la defense",
    "secret de la défense",
    "secret des deliberations",
    "secret des délibérations",
    "secret de l enquete",
    "secret de l'enquête",
)

HEALTH_MARKERS = (  # Indices domaine santé.
    "secret medical",
    "secret médical",
    "code de la sante publique",
    "code de la santé publique",
    "medecin",
    "médecin",
    "malade",
    "praticien",
    "pharmacien",
)

DEFENSE_MARKERS = (  # Indices domaine défense.
    "ministere de la defense",
    "ministère de la défense",
    "secret de la defense",
    "secret de la défense",
    "defense nationale",
    "défense nationale",
    "classifie",
    "classifié",
    "nato secret",
)

AVOCAT_PROFESSION_MARKERS = (  # Indices textes professionnels avocat.
    "decret n° 91-1197",
    "décret n° 91-1197",
    "organisant la profession d'avocat",
    "conseil national des barreaux",
    "college de deontologie",
    "collège de déontologie",
    "reglement interieur national",
    "règlement intérieur national",
    "profession d'avocat",
    "barreau",
)

NOTAIRE_PROFESSION_MARKERS = (  # Indices textes profession notaire.
    "chambre des notaires",
    "profession de notaire",
    "profession notariale",
    "code de deontologie des notaires",
    "code de déontologie des notaires",
    "decret n° 2023-1297",
    "décret n° 2023-1297",
    "notaire",
)

EXAM_PROGRAM_MARKERS = (  # Indices programmes d'examen / concours.
    "programme de l'enseignement",
    "programme et les modalites",
    "programme et les modalités",
    "epreuve",
    "épreuve",
    "concours d'entree",
    "concours d'entrée",
    "coefficient",
    "durée :",
    "duree :",
)


def _count_markers(text, markers):  # Compte les marqueurs présents dans un texte.
    return sum(1 for marker in markers if marker in text)  # Nombre de correspondances.


def profile_to_dict(profile):  # Sérialise un profil pour l'index.
    return {  # Format JSON compact.
        "source_type": profile.source_type,
        "topic_domains": sorted(profile.topic_domains),
        "profession_tags": sorted(profile.profession_tags),
        "is_secret_enumeration": profile.is_secret_enumeration,
        "is_exam_program": profile.is_exam_program,
        "is_professional_rules_avocat": profile.is_professional_rules_avocat,
        "is_professional_rules_notaire": profile.is_professional_rules_notaire,
    }


def profile_from_dict(data):  # Recharge un profil sérialisé.
    return DocumentProfile(  # Reconstruit le profil.
        source_type=data.get("source_type") or "corpus",
        topic_domains=frozenset(data.get("topic_domains") or ()),
        profession_tags=frozenset(data.get("profession_tags") or ()),
        is_secret_enumeration=bool(data.get("is_secret_enumeration")),
        is_exam_program=bool(data.get("is_exam_program")),
        is_professional_rules_avocat=bool(data.get("is_professional_rules_avocat")),
        is_professional_rules_notaire=bool(data.get("is_professional_rules_notaire", False)),
    )


def compute_profile(hit):  # Calcule le profil thématique d'un fragment.
    source_type = hit.get("domain") or "corpus"  # Type de source indexée.
    title = _normalize(hit.get("title"))  # Titre normalisé.
    code = _normalize(hit.get("code"))  # Code ou texte parent normalisé.
    text = _normalize(hit.get("text"))  # Contenu normalisé.
    blob = f"{title}\n{code}\n{text}"  # Texte combiné pour l'analyse.
    topic_domains = set()  # Domaines thématiques détectés.
    profession_tags = set()  # Professions liées au document.
    if _count_markers(blob, HEALTH_MARKERS):  # Document lié à la santé.
        topic_domains.add("health")  # Marque le domaine santé.
    if _count_markers(blob, DEFENSE_MARKERS):  # Document lié à la défense.
        topic_domains.add("defense")  # Marque le domaine défense.
    if source_type == "judilibre":  # Jurisprudence indexée.
        topic_domains.add("jurisprudence")  # Marque la jurisprudence.
    is_professional_rules_avocat = source_type == "legifrance" and _count_markers(blob, AVOCAT_PROFESSION_MARKERS) >= 1  # Normes métier avocat.
    is_professional_rules_notaire = source_type == "legifrance" and _count_markers(blob, NOTAIRE_PROFESSION_MARKERS) >= 2  # Normes métier notaire.
    if is_professional_rules_avocat:  # Corpus déontologique avocat.
        topic_domains.add("professional_rules_avocat")  # Marque les règles professionnelles.
        profession_tags.add("avocat")  # Associe la profession avocat.
    if is_professional_rules_notaire:  # Corpus profession notaire.
        topic_domains.add("professional_rules_notaire")  # Marque les règles notariales.
        profession_tags.add("notaire")  # Associe la profession notaire.
    if "profession d'avocat" in blob or "avocat" in title:  # Mention explicite avocat.
        profession_tags.add("avocat")  # Tag profession avocat.
    if _count_markers(blob, ("medecin", "médecin", "praticien")):  # Mention médicale.
        profession_tags.add("medecin")  # Tag profession médecin.
    secret_hits = _count_markers(blob, SECRET_MARKERS)  # Compte les types de secrets cités.
    is_secret_enumeration = secret_hits >= 2  # Plusieurs secrets dans le même extrait.
    is_exam_program = _count_markers(blob, EXAM_PROGRAM_MARKERS) >= 2  # Programme d'examen probable.
    return DocumentProfile(  # Renvoie le profil complet.
        source_type=source_type,
        topic_domains=frozenset(topic_domains),
        profession_tags=frozenset(profession_tags),
        is_secret_enumeration=is_secret_enumeration,
        is_exam_program=is_exam_program,
        is_professional_rules_avocat=is_professional_rules_avocat,
        is_professional_rules_notaire=is_professional_rules_notaire,
    )


def profile_document(hit):  # Charge ou calcule le profil d'un fragment.
    cached = hit.get("profile")  # Profil pré-calculé à l'indexation.
    if isinstance(cached, dict):  # Format sérialisé disponible.
        return profile_from_dict(cached)  # Recharge sans recalcul.
    return compute_profile(hit)  # Calcule à la volée pour les anciens index.
