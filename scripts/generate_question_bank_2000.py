#!/usr/bin/env python3
"""Génère eval/question_bank_2000.jsonl — banque de questions pour tests auto (smoke retrieval).

Les lignes avec champs expect_* sont des cas « stricts » (optionnels, à compléter).
Par défaut chaque ligne a smoke=true : le batch eval exige au moins 1 hit avec score > seuil.
"""

import json
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "question_bank_2000.jsonl"

# Thèmes cabinet : (préfixe_id, formulations de base, profession optionnelle)
THEMES = [
    (
        "secret_pro_avocat",
        "avocat",
        [
            "secret professionnel avocat",
            "confidentialité avocat client correspondances",
            "levée du secret professionnel avocat",
            "violation secret professionnel avocat cassation",
            "secret professionnel correspondances avocat client",
            "avocat tenu au secret professionnel décret 91-1197",
            "divulgation information avocat secret professionnel",
            "recel violation secret professionnel avocat",
            "secret professionnel avocat instruction pénale",
            "bâtonnier secret professionnel avocat",
        ],
    ),
    (
        "resp_avocat",
        "avocat",
        [
            "responsabilité civile avocat faute conseil",
            "responsabilité professionnelle avocat cassation",
            "faute avocat perte de chance",
            "mise en cause avocat responsabilité civile",
            "jurisprudence responsabilité avocat",
            "obligation de conseil avocat responsabilité",
            "avocat responsabilité faute art 1240",
            "indemnisation client avocat faute",
        ],
    ),
    (
        "honoraires",
        "avocat",
        [
            "convention honoraires avocat",
            "honoraires avocat information client",
            "provision sur honoraires avocat",
            "facturation honoraires avocat décret 91-1197",
            "contestation honoraires avocat bâtonnier",
            "honoraires au résultat avocat interdiction",
            "devis honoraires avocat",
            "honoraires avocat aide juridictionnelle",
        ],
    ),
    (
        "publicite",
        "avocat",
        [
            "publicité avocat décret 91-1197",
            "site internet avocat publicité loyauté dignité",
            "réseaux sociaux avocat publicité",
            "interdiction publicité comparative avocat",
            "communication avocat barreau",
            "publicité avocat honoraires",
        ],
    ),
    (
        "mentions_rin",
        "avocat",
        [
            "mentions légales site internet avocat",
            "règlement intérieur national avocat RIN",
            "obligations site web cabinet avocat",
            "coordonnées avocat mentions obligatoires",
            "barreau mentions légales avocat",
        ],
    ),
    (
        "deonto_avocat",
        "avocat",
        [
            "conflit d'intérêts avocat",
            "obligations déontologiques avocat",
            "indépendance avocat déontologie",
            "devoir de loyauté avocat client",
            "refus de mission avocat déontologie",
            "collaboration avocat conflit intérêts",
            "code déontologie avocat 2023",
        ],
    ),
    (
        "cassation",
        "avocat",
        [
            "arrêt Cour de cassation avocat",
            "jurisprudence Cour de cassation secret professionnel avocat",
            "pourvoi cassation responsabilité avocat",
            "chambre criminelle secret professionnel avocat",
        ],
    ),
    (
        "notaire",
        "notaire",
        [
            "obligations déontologiques notaire décret 2023",
            "secret professionnel notaire",
            "conflit d'intérêts notaire",
            "devoir de conseil notaire",
            "responsabilité civile notaire",
        ],
    ),
    (
        "medical",
        "medecin",
        [
            "secret médical médecin",
            "obligations déontologiques médecin",
            "violation secret médical",
        ],
    ),
    (
        "defense",
        None,
        [
            "secret défense nationale obligations",
            "classification secret défense",
        ],
    ),
    (
        "procedure",
        "avocat",
        [
            "assignation avocat délais",
            "forclusion préclusion avocat",
            "frais de justice avocat",
            "expertise judiciaire avocat",
            "communication pièces avocat adversaire",
            "dépôt conclusions avocat",
            "appel civil avocat délai",
            "référé avocat urgence",
        ],
    ),
    (
        "barreau",
        "avocat",
        [
            "inscription barreau avocat",
            "stage avocat réglementation",
            "associe avocat SELARL",
            "déménagement barreau avocat",
            "assurance RCP avocat",
            "compte séquestre avocat",
        ],
    ),
    (
        "client",
        "avocat",
        [
            "rupture relation avocat client",
            "restitution dossier avocat client",
            "mandat avocat fin de mission",
            "information client avocat droits",
            "plainte contre avocat bâtonnier",
        ],
    ),
]

# Variantes « question naturelle » (préfixe oral)
OPENERS = ["", "Quelles sont les ", "Quel est le ", "Peut-on ", "Est-ce que ", "Comment "]

QUALIFIERS = [
    "",
    "jurisprudence",
    "texte applicable",
    "obligations",
    "sanctions",
    "exceptions",
    "cassation récente",
    "décret",
    "article de loi",
    "cadre général",
    "principes",
    "conditions",
    "portée",
    "limites",
    "contentieux",
    "preuve",
    "procédure",
    "réforme",
    "2024",
    "2023",
    "2022",
]


def build_rows(target=2000):
    rows = []
    n = 0
    seen = set()
    for theme_id, profession, bases in THEMES:
        for base, qual, opener in itertools.product(bases, QUALIFIERS, OPENERS):
            core = f"{base} {qual}".strip() if qual else base
            q = f"{opener}{core} ?".replace("  ", " ").strip() if opener else core
            if q in seen:
                continue
            seen.add(q)
            n += 1
            row = {
                "id": f"bank_{n:04d}",
                "theme": theme_id,
                "query": q,
                "smoke": True,
            }
            if profession:
                row["hint_profession"] = profession  # informatif ; pas de critère strict
            rows.append(row)
            if len(rows) >= target:
                return rows
    return rows


def main():
    rows = build_rows(2000)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Écrit {len(rows)} questions → {OUT}")
    print("Complétez expect_intents / prefer_source sur les lignes importantes (voir docs/TESTS_AUTO.md).")


if __name__ == "__main__":
    main()
