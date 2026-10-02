"""Nettoyage déterministe, traçabilité et séparation des documents sans PyTorch."""

import hashlib  # Calcule les empreintes des documents et fichiers.
import json  # Lit le format JSONL : un document JSON par ligne.
import re  # Regroupe les espaces superflus.
import unicodedata  # Uniformise les caractères accentués sans les supprimer.
from pathlib import Path  # Manipule les chemins de manière portable.


def digest(path):  # Identifie exactement un fichier sans le charger en RAM.
    result = hashlib.sha256()  # Initialise une empreinte SHA-256.
    with Path(path).open("rb") as stream:  # Ouvre les octets du fichier.
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):  # Lit par blocs de 1 Mio.
            result.update(chunk)  # Intègre chaque bloc à l'empreinte.
    return result.hexdigest()  # Renvoie une chaîne enregistrable en JSON.


def clean(text):  # Normalise un texte en conservant ses paragraphes.
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n")  # Unifie accents et fins de ligne.
    return "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()).strip()  # Réduit les espaces.


def records(path):  # Parcourt un corpus sans le charger entièrement.
    with Path(path).open(encoding="utf-8") as stream:  # Utilise UTF-8 pour le français.
        for number, line in enumerate(stream, 1):  # Garde le numéro pour les erreurs.
            if not line.strip():  # Accepte les lignes vides.
                continue  # Passe à la ligne suivante.
            row = json.loads(line)  # Décode un document.
            required = ("text", "source", "license", "language")  # Exige une provenance et une langue déclarée.
            if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):  # Vérifie le schéma.
                raise ValueError(f"{path}:{number} : champs texte/source/license/language invalides (clé text attendue).")  # Signale l'entrée fautive.
            yield row  # Fournit un seul document à la fois.


def split_corpus(source, target, val_fraction=0.05, seed=42, min_chars=200):  # Sépare avant d'entraîner le tokenizer.
    if not 0 < val_fraction < 1:  # Refuse une fraction inutilisable.
        raise ValueError("La fraction de validation doit être entre 0 et 1.")  # Explique le paramètre.
    target = Path(target)  # Convertit le dossier de destination.
    target.mkdir(parents=True, exist_ok=False)  # Empêche d'écraser un corpus déjà préparé.
    seen = set()  # Conserve uniquement les empreintes, pas les textes complets.
    stats = dict(train=0, val=0, duplicates=0, rejected=0)  # Compte les décisions de filtrage.
    with (target / "train.jsonl").open("w", encoding="utf-8") as train, (target / "val.jsonl").open("w", encoding="utf-8") as val:  # Ouvre les deux sorties.
        for row in records(source):  # Parcourt les documents d'origine.
            text = clean(row["text"])  # Nettoie avant déduplication.
            if row["language"] != "fr" or len(text) < min_chars or "\x00" in text:  # Écarte langue déclarée incorrecte, texte trop court ou binaire.
                stats["rejected"] += 1  # Compte le rejet.
                continue  # Ignore ce document.
            key = hashlib.sha256(text.casefold().encode()).hexdigest()  # Détecte les doublons exacts sans distinction de casse.
            if key in seen:  # Vérifie les documents déjà acceptés.
                stats["duplicates"] += 1  # Compte le doublon.
                continue  # Évite sa présence dans deux ensembles.
            seen.add(key)  # Mémorise le document accepté.
            group = row.get("split_group", key)  # Garde tous les morceaux d'un livre dans le même ensemble ; les anciens documents conservent leur règle.
            if not isinstance(group, str) or not group:  # Vérifie la clé de regroupement facultative.
                raise ValueError("split_group doit être une chaîne non vide.")  # Refuse une affectation ambiguë.
            score = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:16], 16) / 2**64  # Affectation stable indépendante de l'ordre.
            split = "val" if score < val_fraction else "train"  # Réserve une fraction pour l'évaluation.
            row = dict(row, text=text, sha256=key)  # Conserve les métadonnées de provenance.
            (val if split == "val" else train).write(json.dumps(row, ensure_ascii=False) + "\n")  # Écrit le document nettoyé.
            stats[split] += 1  # Compte le document dans son ensemble.
    if not stats["train"] or not stats["val"]:  # Refuse une validation vide même pour un petit corpus.
        raise ValueError("Corpus trop petit : train ou validation vide. Ajouter des documents et choisir un nouveau dossier.")  # Donne une correction.
    return stats  # Rend les statistiques au pipeline.
