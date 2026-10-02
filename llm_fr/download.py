"""Télécharge un échantillon borné de Wikipédia français avec sa provenance."""

import argparse  # Configure les limites du téléchargement.
import json  # Écrit un document JSON par ligne.
import shutil  # Surveille l'espace disponible pendant la collecte.
from pathlib import Path  # Crée les chemins de sortie.
import pyarrow.parquet as pq  # Décode les fichiers Parquet sans threads de lecture persistants.
from huggingface_hub import HfApi, HfFileSystem  # Résout une version précise et lit les fichiers distants par blocs.


def download(output, limit, revision, max_mb=400):  # Borne le nombre d'articles et la taille locale du téléchargement.
    if limit <= 0 or max_mb <= 0:  # Refuse des limites vides ou négatives.
        raise ValueError("--limit doit être positif.")  # Explique le paramètre incorrect.
    path = Path(output)  # Définit le fichier local.
    if path.exists():  # Protège les données existantes.
        raise FileExistsError(path)  # Demande implicitement un autre nom de fichier.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier de réception.
    temporary = path.with_suffix(path.suffix + ".part")  # Rend visible un téléchargement incomplet.
    repository = "wikimedia/wikipedia"  # Identifie le dataset public.
    info = HfApi().dataset_info(repository, revision=revision)  # Résout même main vers un commit immuable.
    files = sorted(item.rfilename for item in info.siblings if item.rfilename.startswith("20231101.fr/train-") and item.rfilename.endswith(".parquet"))  # Sélectionne uniquement les partitions françaises.
    if not files:  # Détecte une version dont le format ne correspond pas.
        raise ValueError("Aucun fichier Parquet français dans cette révision.")  # Évite un corpus silencieusement vide.
    filesystem = HfFileSystem()  # Initialise la lecture distante avec accès partiel.
    count = 0  # Compte les articles effectivement reçus.
    with temporary.open("x", encoding="utf-8") as stream:  # Refuse aussi d'écraser un téléchargement interrompu.
        for name in files:  # Parcourt les partitions sans les charger toutes.
            with filesystem.open(f"datasets/{repository}@{info.sha}/{name}", "rb") as remote:  # Ouvre la version figée du fichier.
                with pq.ParquetFile(remote, pre_buffer=False) as parquet:  # Désactive toute prélecture asynchrone.
                    for batch in parquet.iter_batches(batch_size=32, columns=["text", "url", "title"], use_threads=False):  # Décode de petits lots sur le thread courant.
                        for row in batch.to_pylist():  # Convertit uniquement le lot courant en objets Python.
                            if shutil.disk_usage(path.parent).free < 1024**3:  # Préserve un Gio libre avant chaque écriture.
                                raise OSError("Collecte arrêtée : moins de 1 Gio libre sur le disque.")  # Laisse le fichier .part identifiable comme incomplet.
                            document = dict(text=row["text"], source=row["url"], title=row["title"], license="CC-BY-SA-3.0 / GFDL (fiche du dataset ; vérifier les conditions applicables)", language="fr", dataset=repository, snapshot="20231101.fr", revision=info.sha)  # Conserve la provenance et le commit résolu.
                            line = json.dumps(document, ensure_ascii=False) + "\n"  # Prépare une seule ligne à la fois.
                            if stream.tell() + len(line.encode("utf-8")) > max_mb * 1024**2:  # Évite de dépasser le plafond demandé.
                                raise OSError("Limite de taille atteinte ; fichier .part conservé, réduire --limit.")  # N'annonce pas un téléchargement complet.
                            stream.write(line)  # Enregistre un article complet.
                            count += 1  # Actualise la limite de téléchargement.
                            if count >= limit:  # Vérifie la limite dans le lot.
                                break  # Arrête les documents de ce lot.
                        if count >= limit:  # Vérifie la limite dans la partition.
                            break  # Ferme le lecteur Parquet sans demander un autre lot.
            if count >= limit:  # Vérifie la limite entre partitions.
                break  # Évite l'ouverture d'un autre fichier distant.
    temporary.replace(path)  # Publie le fichier local seulement après succès.
    print(f"Corpus téléchargé : {path} ({count} articles, commit {info.sha}). Vérifier un échantillon avant entraînement.")  # Affiche le volume réel et rappelle la revue des données.


if __name__ == "__main__":  # Exécute uniquement la commande appelée.
    parser = argparse.ArgumentParser(description=__doc__)  # Décrit le téléchargement optionnel.
    parser.add_argument("--output", default="data/raw/wikipedia_fr.jsonl")  # Définit le fichier cible.
    parser.add_argument("--limit", type=int, default=10000)  # Commence par un volume borné.
    parser.add_argument("--revision", default="main")  # Accepte un commit Hugging Face pour figer la source.
    parser.add_argument("--max-mb", type=int, default=400)  # Borne la taille locale de la collecte en Mio.
    args = parser.parse_args()  # Lit les choix de l'utilisateur.
    download(args.output, args.limit, args.revision, args.max_mb)  # Lance le téléchargement.
