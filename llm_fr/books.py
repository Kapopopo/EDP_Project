"""Collecte bornée de classiques français, avec originaux et provenance conservés."""

import argparse  # Lit les chemins du catalogue et du corpus.
import json  # Lit le catalogue vérifié et écrit les documents JSONL.
import re  # Repère les limites Gutenberg et les paragraphes.
import shutil  # Vérifie l'espace disponible sur le disque.
import urllib.request  # Télécharge sans nouvelle dépendance Python.
from pathlib import Path  # Manipule les chemins locaux.
from .corpus import digest  # Calcule l'empreinte des éditions téléchargées.


def body(text):  # Retire les notices Gutenberg du texte utilisé pour apprendre.
    start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", text, re.I)  # Repère le début du texte de l'œuvre.
    end = re.search(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*", text, re.I)  # Repère la fin avant les conditions de distribution.
    if start is None or end is None or end.start() <= start.end():  # Refuse un format inconnu au lieu d'apprendre les notices anglaises.
        raise ValueError("Marqueurs Gutenberg absents ou invalides.")  # Permet de revoir manuellement cette édition.
    if not re.search(r"Language:\s*French", text[:start.start()], re.I):  # Vérifie la langue déclarée dans le fichier original.
        raise ValueError("L'édition ne déclare pas la langue française.")  # Évite une traduction anglaise de même titre.
    return text[start.end():end.start()].strip()  # Conserve uniquement le corps de l'œuvre.


def chunks(text, size=12000):  # Limite la taille des documents présentés au tokenizer.
    if size < 200:  # Évite des morceaux incompatibles avec le filtre de longueur.
        raise ValueError("La taille des morceaux doit être au moins 200 caractères.")  # Signale un réglage inutile.
    text = text.replace("\r\n", "\n")  # Uniformise les fins de ligne des éditions.
    pending = ""  # Accumule au plus un morceau de texte.
    for paragraph in re.split(r"\n\s*\n", text):  # Privilégie les frontières naturelles des paragraphes.
        paragraph = re.sub(r"\s+", " ", paragraph).strip()  # Rejoint les lignes de l'édition imprimée.
        for offset in range(0, len(paragraph), size):  # Découpe aussi les paragraphes exceptionnellement longs.
            piece = paragraph[offset:offset + size]  # Conserve une tranche bornée.
            if pending and len(pending) + len(piece) + 2 > size:  # Ferme le morceau avant de dépasser la limite.
                yield pending  # Fournit un document complet.
                pending = ""  # Réinitialise le tampon.
            pending += ("\n\n" if pending else "") + piece  # Assemble les paragraphes qui tiennent ensemble.
    if pending:  # Conserve le dernier morceau même s'il est plus court.
        yield pending  # Le filtre commun décidera s'il est assez long.


def download_books(catalog, output):  # Traite un petit catalogue explicitement sélectionné.
    path = Path(output)  # Définit la sortie JSONL.
    path.parent.mkdir(parents=True, exist_ok=True)  # Prépare son dossier.
    if path.exists():  # Protège un corpus déjà constitué.
        raise FileExistsError(path)  # Exige une nouvelle sortie pour une nouvelle collecte.
    originals = path.parent / "books_originals"  # Conserve les éditions complètes, dont les notices.
    originals.mkdir(exist_ok=True)  # Crée un cache réutilisable pour les huit ouvrages.
    temporary = path.with_suffix(".jsonl.part")  # Distingue une collecte incomplète d'un corpus fini.
    with temporary.open("x", encoding="utf-8") as stream:  # Refuse d'écraser un travail interrompu.
        for book in json.loads(Path(catalog).read_text(encoding="utf-8")):  # Lit uniquement le petit catalogue en mémoire.
            identifier = int(book["id"])  # Valide l'identifiant du livre.
            url = f"https://www.gutenberg.org/ebooks/{identifier}.txt.utf-8"  # Utilise l'édition texte UTF-8 officielle.
            original = originals / f"{identifier}.txt"  # Sépare chaque édition locale.
            if not original.exists():  # Réutilise une édition déjà reçue sans redemander le réseau.
                if shutil.disk_usage(path.parent).free < 1024**3:  # Réserve un Gio pour le fonctionnement de la machine.
                    raise OSError("Moins de 1 Gio libre avant le téléchargement du livre.")  # Arrête sans remplir le disque.
                request = urllib.request.Request(url, headers={"User-Agent": "llm-fr-local-corpus/1.0"})  # Identifie la collecte sans usurper un navigateur.
                partial = original.with_suffix(".part")  # Prépare un fichier de téléchargement intermédiaire.
                with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as target:  # Ferme toujours le réseau et le fichier.
                    total = 0  # Compte les octets reçus pour ce livre.
                    while block := response.read(65536):  # Lit seulement 64 Kio à la fois.
                        total += len(block)  # Actualise la taille transférée.
                        if total > 16 * 1024**2:  # Refuse une réponse anormalement volumineuse.
                            raise ValueError("Édition supérieure au plafond de 16 Mio.")  # Borne la mémoire de son décodage ultérieur.
                        target.write(block)  # Écrit progressivement les octets sur disque.
                partial.replace(original)  # Publie l'original uniquement après réception complète.
            if original.stat().st_size > 16 * 1024**2:  # Applique aussi le plafond aux éditions déjà présentes sur disque.
                raise ValueError("Édition locale supérieure au plafond de 16 Mio.")  # Refuse un chargement mémoire trop volumineux.
            text = body(original.read_text(encoding="utf-8-sig"))  # Vérifie la langue et isole le texte, un livre à la fois.
            fingerprint = digest(original)  # Identifie exactement l'édition reçue.
            count = 0  # Compte les fragments de cet ouvrage.
            for count, piece in enumerate(chunks(text), 1):  # Produit des documents d'au plus 12 000 caractères.
                row = dict(text=piece, source=f"https://www.gutenberg.org/ebooks/{identifier}", download_url=url, title=book["title"], author=book["author"], language="fr", license="Public domain in the USA (fiche Project Gutenberg) ; notice complète conservée dans l'original", split_group=f"gutenberg:{identifier}", edition_sha256=fingerprint, fragment=count)  # Conserve attribution, version et groupe de séparation.
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")  # Écrit un fragment traçable.
            print(f"{book['title']} : {count} fragments", flush=True)  # Affiche l'avancement livre par livre.
    temporary.replace(path)  # Rend le corpus disponible seulement lorsqu'il est terminé.


if __name__ == "__main__":  # Expose la collecte comme commande Python.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide du terminal.
    parser.add_argument("--catalog", default="configs/books.json")  # Sélectionne le catalogue de classiques vérifiés.
    parser.add_argument("--output", default="data/raw/books_fr.jsonl")  # Choisit la sortie par défaut.
    args = parser.parse_args()  # Lit les paramètres de la commande.
    download_books(args.catalog, args.output)  # Lance la collecte bornée.
