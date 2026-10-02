"""Collecte un échantillon juridique français à partir de Wikipédia filtrée."""

import argparse  # Configure la collecte juridique.
import json  # Écrit le corpus au format JSONL attendu par corpus.py.
import re  # Filtre les articles par mots-clés juridiques.
import shutil  # Surveille l'espace disque disponible.
from pathlib import Path  # Crée les chemins de sortie.
import pyarrow.parquet as pq  # Lit les partitions Parquet distantes.
from huggingface_hub import HfApi, HfFileSystem  # Résout la version figée du dataset.


KEYWORDS = (  # Mots-clés orientant la sélection vers le droit français.
    "droit", "avocat", "tribunal", "jurisprudence", "juridique", "cour ", "code civil",
    "code pénal", "procédure", "contrat", "responsabilité", "secret professionnel",
    "divorce", "succession", "bail", "pénal", "civil", "administratif", "constitution",
    "décret", "loi ", "article ", "juge", "audience", "conclusions", "assignation",
)


def legal_score(title, text):  # Estime la pertinence juridique d'un article.
    haystack = f"{title}\n{text[:4000]}".casefold()  # Limite la lecture pour accélérer le filtrage.
    return sum(1 for keyword in KEYWORDS if keyword in haystack)  # Compte les occurrences thématiques.


def download(output, limit, revision, min_score=1, max_mb=400):  # Collecte des documents juridiques bornés.
    if limit <= 0 or max_mb <= 0:  # Refuse des limites invalides.
        raise ValueError("--limit et --max-mb doivent être positifs.")  # Explique l'erreur de paramètre.
    path = Path(output)  # Définit le fichier de sortie.
    if path.exists():  # Protège un corpus déjà collecté.
        raise FileExistsError(path)  # Demande un autre chemin.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier juridique.
    temporary = path.with_suffix(path.suffix + ".part")  # Marque les collectes incomplètes.
    repository = "wikimedia/wikipedia"  # Réutilise la source déjà validée par le projet.
    info = HfApi().dataset_info(repository, revision=revision)  # Figera le commit réellement utilisé.
    files = sorted(item.rfilename for item in info.siblings if item.rfilename.startswith("20231101.fr/train-") and item.rfilename.endswith(".parquet"))  # Sélectionne les partitions françaises.
    if not files:  # Refuse une révision sans données françaises.
        raise ValueError("Aucun fichier Parquet français dans cette révision.")  # Évite un corpus vide.
    filesystem = HfFileSystem()  # Ouvre la lecture distante par blocs.
    kept = 0  # Compte les documents retenus.
    scanned = 0  # Compte les articles examinés.
    with temporary.open("x", encoding="utf-8") as stream:  # Refuse d'écraser une collecte partielle.
        for name in files:  # Parcourt les partitions disponibles.
            with filesystem.open(f"datasets/{repository}@{info.sha}/{name}", "rb") as remote:  # Ouvre la partition distante.
                with pq.ParquetFile(remote, pre_buffer=False) as parquet:  # Désactive la prélecture asynchrone.
                    for batch in parquet.iter_batches(batch_size=32, columns=["text", "url", "title"], use_threads=False):  # Décode de petits lots.
                        for row in batch.to_pylist():  # Convertit le lot courant.
                            scanned += 1  # Incrémente le nombre d'articles lus.
                            title = row.get("title") or ""  # Récupère le titre encyclopédique.
                            text = row.get("text") or ""  # Récupère le corps de l'article.
                            if legal_score(title, text) < min_score:  # Ignore les articles hors thème.
                                continue  # Passe au document suivant.
                            if shutil.disk_usage(path.parent).free < 1024**3:  # Préserve un Gio libre.
                                raise OSError("Collecte arrêtée : moins de 1 Gio libre sur le disque.")  # Arrête proprement.
                            document = dict(  # Construit un document conforme à corpus.py.
                                text=text,
                                source=row["url"],
                                title=title,
                                license="CC-BY-SA-3.0 / GFDL (fiche du dataset ; vérifier les conditions applicables)",
                                language="fr",
                                domain="juridique",
                                dataset=repository,
                                snapshot="20231101.fr",
                                revision=info.sha,
                                legal_score=legal_score(title, text),
                            )
                            line = json.dumps(document, ensure_ascii=False) + "\n"  # Prépare la ligne JSONL.
                            if stream.tell() + len(line.encode("utf-8")) > max_mb * 1024**2:  # Respecte le plafond disque.
                                raise OSError("Limite de taille atteinte ; réduire --limit ou augmenter --max-mb.")  # Signale la borne atteinte.
                            stream.write(line)  # Enregistre l'article juridique.
                            kept += 1  # Compte le document retenu.
                            if kept >= limit:  # Atteint l'objectif demandé.
                                break  # Sort du lot courant.
                        if kept >= limit:  # Atteint l'objectif dans la partition.
                            break  # Passe à la finalisation.
            if kept >= limit:  # Atteint l'objectif global.
                break  # Termine la collecte.
    temporary.replace(path)  # Publie le fichier seulement après succès.
    print(f"Corpus juridique : {path} ({kept} documents retenus sur {scanned} examinés, commit {info.sha}).")  # Rappelle de vérifier un échantillon.


if __name__ == "__main__":  # Exécute la commande depuis le terminal.
    parser = argparse.ArgumentParser(description=__doc__)  # Décrit la collecte juridique.
    parser.add_argument("--output", default="data/legal/raw/corpus.jsonl")  # Chemin de sortie par défaut.
    parser.add_argument("--limit", type=int, default=1000)  # Vise un échantillon intermédiaire du plan T1.
    parser.add_argument("--revision", default="b04c8d1ceb2f5cd4588862100d08de323dccfbaa")  # Réutilise le commit pilote.
    parser.add_argument("--min-score", type=int, default=1)  # Exige au moins un mot-clé juridique.
    parser.add_argument("--max-mb", type=int, default=400)  # Borne la taille locale.
    args = parser.parse_args()  # Lit les paramètres utilisateur.
    download(args.output, args.limit, args.revision, args.min_score, args.max_mb)  # Lance la collecte.
