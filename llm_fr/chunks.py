"""Découpe un corpus JSONL en fragments indexables pour le RAG."""

import argparse  # Lit les chemins source et destination.
import hashlib  # Identifie chaque fragment de façon stable.
import json  # Lit et écrit le format JSONL.
import re  # Repère les mots pour les statistiques futures.
import unicodedata  # Normalise les accents avant découpage.
from pathlib import Path  # Manipule les fichiers de corpus.

from .corpus import clean, records  # Réutilise le nettoyage et la lecture JSONL.


TOKEN = re.compile(r"[a-zàâäéèêëïîôùûüçœæ0-9]+", re.IGNORECASE)  # Mots français simples pour le RAG.


def tokenize(text):  # Transforme un texte en liste de mots normalisés.
    text = unicodedata.normalize("NFC", text).casefold()  # Uniformise la casse et les accents composés.
    return TOKEN.findall(text)  # Extrait les tokens alphanumériques.


def chunk_text(text, size=800, overlap=100):  # Découpe un texte long en fragments chevauchants.
    if size < 1 or overlap < 0 or overlap >= size:  # Refuse des paramètres incohérents.
        raise ValueError("size doit être positif et overlap strictement inférieur à size.")  # Explique la contrainte.
    text = clean(text)  # Nettoie avant découpage.
    if not text:  # Ignore les textes vides après nettoyage.
        return []  # Aucun fragment à produire.
    if len(text) <= size:  # Conserve les documents déjà courts.
        return [text]  # Un seul fragment suffit.
    step = size - overlap  # Avance la fenêtre en conservant un chevauchement.
    chunks = []  # Accumule les fragments extraits.
    start = 0  # Position de départ du fragment courant.
    while start < len(text):  # Parcourt tout le document.
        end = min(start + size, len(text))  # Borne la fin du fragment.
        piece = text[start:end].strip()  # Retire les espaces en bordure.
        if piece:  # Ignore les fragments vides.
            chunks.append(piece)  # Conserve le morceau utile.
        if end >= len(text):  # Termine après le dernier caractère.
            break  # Sort de la boucle.
        start += step  # Décale la fenêtre suivante.
    return chunks  # Renvoie tous les fragments.


def chunk_id(sha256, index):  # Fabrique un identifiant stable pour un fragment.
    return hashlib.sha256(f"{sha256}:{index}".encode()).hexdigest()[:16]  # Empreinte courte et déterministe.


def build_chunks(sources, output, size=800, overlap=100):  # Découpe un ou plusieurs JSONL en fragments RAG.
    if size < 1 or overlap < 0 or overlap >= size:  # Valide les paramètres de découpage.
        raise ValueError("size doit être positif et overlap strictement inférieur à size.")  # Message d'erreur explicite.
    paths = [Path(p) for p in sources]  # Convertit les entrées en chemins.
    for path in paths:  # Vérifie chaque source avant traitement.
        if not path.is_file():  # Refuse un fichier manquant.
            raise FileNotFoundError(f"Corpus introuvable : {path}")  # Signale le chemin fautif.
    target = Path(output)  # Localise le fichier de sortie.
    target.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier RAG si besoin.
    stats = dict(documents=0, chunks=0, skipped=0)  # Compte les documents et fragments produits.
    with target.open("w", encoding="utf-8") as stream:  # Écrit les fragments au fil de l'eau.
        for path in paths:  # Traite chaque corpus source.
            for row in records(path):  # Lit les documents un par un.
                text = clean(row["text"])  # Nettoie le texte source.
                if len(text) < 50:  # Ignore les entrées trop courtes pour être utiles.
                    stats["skipped"] += 1  # Compte le rejet.
                    continue  # Passe au document suivant.
                stats["documents"] += 1  # Compte le document traité.
                doc_sha = row.get("sha256") or hashlib.sha256(text.casefold().encode()).hexdigest()  # Réutilise l'empreinte existante.
                title = row.get("title") or row["source"]  # Fournit un titre lisible pour les citations.
                for index, piece in enumerate(chunk_text(text, size, overlap)):  # Découpe le document.
                    chunk = dict(  # Conserve la provenance pour chaque fragment.
                        chunk_id=chunk_id(doc_sha, index),  # Identifiant unique du fragment.
                        text=piece,  # Texte du fragment.
                        source=row["source"],  # URL ou référence d'origine.
                        title=title,  # Titre affiché dans les citations.
                        license=row["license"],  # Licence du document source.
                        language=row["language"],  # Langue déclarée.
                        sha256=doc_sha,  # Empreinte du document parent.
                        chunk_index=index,  # Position du fragment dans le document.
                    )
                    for key in (  # Propage les métadonnées juridiques.
                        "domain", "dataset", "article_id", "article_num", "code", "etat",
                        "decision_id", "jurisdiction", "chamber", "solution", "decision_date", "decision_type", "ecli",
                    ):
                        if row.get(key):  # Ignore les champs absents.
                            chunk[key] = row[key]  # Conserve la référence Légifrance.
                    stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")  # Écrit une ligne JSONL.
                    stats["chunks"] += 1  # Compte le fragment produit.
    if not stats["chunks"]:  # Refuse une sortie vide.
        raise ValueError("Aucun fragment produit : corpus vide ou trop court.")  # Demande un corpus plus riche.
    return stats  # Renvoie les statistiques de découpage.


if __name__ == "__main__":  # Expose python -m llm_fr.chunks.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide intégrée.
    parser.add_argument("--input", nargs="+", required=True)  # Accepte un ou plusieurs JSONL.
    parser.add_argument("--output", default="data/rag/chunks.jsonl")  # Chemin de sortie par défaut.
    parser.add_argument("--size", type=int, default=800)  # Taille maximale d'un fragment.
    parser.add_argument("--overlap", type=int, default=100)  # Chevauchement entre fragments.
    args = parser.parse_args()  # Lit les options du terminal.
    result = build_chunks(args.input, args.output, args.size, args.overlap)  # Lance le découpage.
    print(f"Fragments : {result['chunks']} ({result['documents']} documents, {result['skipped']} ignorés) → {args.output}")  # Affiche le bilan.
