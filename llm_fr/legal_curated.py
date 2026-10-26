"""Corpus métier prioritaire (avocat, notaire, déontologie) pour retrieval hybride."""

import re
import unicodedata

# (identifiant pack, motifs sur code/titre normalisés, priorité 1=standard 2=haute)
CURATED_PACK_RULES: tuple[tuple[str, tuple[str, ...], int], ...] = (
    ("avocat_organique", ("91-1197", "organisant la profession d'avocat"), 2),
    ("avocat_rin", ("règlement intérieur national", "reglement interieur national", " rin ", "mentions légales", "publicité des avocats"), 2),
    ("avocat_deontologie_2023", ("code de déontologie des avocats", "code de deontologie des avocats"), 2),
    ("avocat_loi_fondamentale", ("loi n° 71-1130", "loi no 71-1130", "réforme certaines professions judiciaires"), 2),
    ("notaire_statut", ("ordonnance n° 45-2590", "ordonnance no 45-2590", "institution du notariat"), 2),
    ("notaire_deontologie_2023", ("2023-1297", "code de déontologie des notaires", "code de deontologie des notaires"), 2),
    ("notaire_ordonnance_2016", ("ordonnance n° 2016-131", "réforme du notariat"), 1),
    ("code_civil_fondamental", ("code civil",), 1),
    ("code_procedure_civile", ("code de procédure civile", "code de procedure civile"), 1),
)

_PACK_BY_ID = {pack_id: priority for pack_id, _, priority in CURATED_PACK_RULES}  # Lookup priorité.


def normalize_field(text):  # Uniformise code/titre pour le matching.
    text = unicodedata.normalize("NFC", text or "").casefold()
    return re.sub(r"\s+", " ", text).strip()


def classify_curated_pack(hit):  # Retourne l'id du pack métier ou None.
    code = normalize_field(hit.get("code"))
    title = normalize_field(hit.get("title"))
    blob = f"{code} {title}".strip()
    if not blob:
        return None
    best_pack = None
    best_priority = 0
    for pack_id, markers, priority in CURATED_PACK_RULES:
        if any(marker in blob for marker in markers):
            if priority > best_priority:
                best_pack = pack_id
                best_priority = priority
    return best_pack


def curated_priority(pack_id):  # Priorité numérique d'un pack (0 si inconnu).
    return _PACK_BY_ID.get(pack_id, 0)


def is_curated_hit(hit):  # Indique si le fragment appartient au corpus métier.
    return bool(hit.get("curated_pack") or classify_curated_pack(hit))
