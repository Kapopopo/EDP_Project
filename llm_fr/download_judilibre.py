"""Télécharge et convertit les décisions Judilibre (Cour de cassation) en corpus JSONL local."""

import argparse  # Configure la collecte Judilibre.
import gzip  # Lit les exports JSONL compressés.
import hashlib  # Empreinte les documents convertis.
import json  # Écrit le corpus au format JSONL.
import shutil  # Surveille l'espace disque disponible.
from pathlib import Path  # Manipule les chemins de sortie.
from urllib.request import urlopen  # Télécharge les exports distants.

from .corpus import clean  # Réutilise la normalisation de texte du projet.


LICENSE = (  # Licence des données Judilibre.
    "Licence Ouverte 2.0 (Cour de cassation — API Judilibre, https://www.data.gouv.fr/datasets/api-judilibre)"
)
HF_BASE = "https://huggingface.co/datasets/antoinejeannot/jurisprudence/resolve/main"  # Miroir JSONL open source.
JURISDICTIONS = {  # Exports disponibles par juridiction.
    "cassation": ("cour_de_cassation.jsonl.gz", "Cour de cassation"),
    "appel": ("cour_d_appel.jsonl.gz", "Cour d'appel"),
    "tribunal": ("tribunal_judiciaire.jsonl.gz", "Tribunal judiciaire"),
}


def decision_url(decision_id, ecli=None):  # Construit le lien public vers une décision.
    if ecli:  # Privilégie l'identifiant européen quand il est disponible.
        return f"https://www.courdecassation.fr/decision/{ecli}"  # Lien ECLI officiel.
    return f"https://www.courdecassation.fr/decision/{decision_id}"  # Lien Judilibre par identifiant.


def build_title(row):  # Formate un titre lisible pour une décision.
    parts = []  # Accumule les éléments du titre.
    decision_type = (row.get("type") or "Décision").strip()  # Type de décision.
    jurisdiction = (row.get("jurisdiction") or "Cour de cassation").strip()  # Juridiction source.
    parts.append(f"{decision_type} — {jurisdiction}")  # Préfixe principal.
    chamber = row.get("chamber")  # Chambre de la juridiction.
    if chamber:  # Ajoute la chambre si connue.
        parts.append(str(chamber).strip())  # Conserve la chambre.
    decision_date = row.get("decision_date") or row.get("update_date")  # Date de la décision.
    if decision_date:  # Ajoute la date si disponible.
        parts.append(str(decision_date).strip())  # Conserve la date.
    number = row.get("number") or (row.get("numbers") or [None])[0]  # Numéro de pourvoi ou d'affaire.
    if number and str(number).strip() not in {".", "-."}:  # Ignore les numéros vides ou placeholders.
        parts.append(f"n° {str(number).strip()}")  # Ajoute le numéro.
    solution = row.get("solution")  # Solution juridictionnelle.
    if solution:  # Ajoute la solution si présente.
        parts.append(str(solution).strip())  # Conserve la solution.
    return " — ".join(parts)  # Assemble le titre final.


def build_text(row):  # Assemble le texte indexable d'une décision.
    summary = clean(row.get("summary") or "")  # Résumé éditorial éventuel.
    body = clean(row.get("text") or "")  # Texte intégral pseudonymisé.
    if summary and body and summary not in body:  # Évite de dupliquer le résumé dans le corps.
        return f"{summary}\n\n{body}"  # Préfixe le texte par le résumé.
    return body or summary  # Retombe sur le contenu disponible.


def parse_decision(row):  # Convertit une décision Judilibre en document JSONL du projet.
    decision_id = (row.get("id") or "").strip()  # Identifiant Judilibre.
    text = build_text(row)  # Assemble le texte utile.
    if not decision_id or len(text) < 80:  # Ignore les décisions trop courtes ou sans identifiant.
        return None  # Aucun document à produire.
    title = build_title(row)  # Formate le titre affiché.
    source = decision_url(decision_id, row.get("ecli"))  # Lien public vers la décision.
    document = dict(  # Construit un document compatible avec corpus.py.
        text=f"{title}\n\n{text}",
        source=source,
        title=title,
        license=LICENSE,
        language="fr",
        domain="judilibre",
        dataset="Judilibre",
        decision_id=decision_id,
        jurisdiction=row.get("jurisdiction"),
        chamber=row.get("chamber"),
        solution=row.get("solution"),
        decision_date=row.get("decision_date") or row.get("update_date"),
        decision_type=row.get("type"),
        ecli=row.get("ecli"),
    )
    document["sha256"] = hashlib.sha256(document["text"].casefold().encode()).hexdigest()  # Empreinte déterministe.
    return document  # Renvoie le document converti.


def download_archive(url, target):  # Télécharge un export JSONL.gz si absent.
    path = Path(target)  # Localise le fichier cible.
    if path.is_file() and path.stat().st_size > 0:  # Ignore un téléchargement déjà présent.
        print(f"Archive déjà présente : {path} ({path.stat().st_size // (1024 ** 2)} Mio)")  # Indique la réutilisation.
        return path  # Renvoie le chemin existant.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier de stockage.
    temporary = path.with_suffix(path.suffix + ".part")  # Marque les téléchargements incomplets.
    print(f"Téléchargement Judilibre : {url}")  # Affiche la source distante.
    with urlopen(url) as remote, temporary.open("wb") as local:  # Télécharge en flux.
        shutil.copyfileobj(remote, local)  # Écrit l'archive compressée.
    temporary.replace(path)  # Publie le fichier seulement après succès.
    print(f"Archive téléchargée : {path}")  # Confirme la fin du téléchargement.
    return path  # Renvoie le chemin final.


def convert_stream(url, output, limit=0):  # Télécharge et convertit un export distant sans archiver le fichier complet.
    path = Path(output)  # Localise la sortie.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier cible.
    temporary = path.with_suffix(path.suffix + ".part")  # Marque les conversions incomplètes.
    kept, scanned, skipped = 0, 0, 0  # Compte les décisions traitées.
    print(f"Conversion en flux : {url}")  # Indique la source distante.
    with urlopen(url) as remote, gzip.open(remote, "rt", encoding="utf-8") as stream, temporary.open("w", encoding="utf-8") as out:  # Lit et convertit en flux.
        for line in stream:  # Parcourt les décisions exportées.
            if not line.strip():  # Ignore les lignes vides.
                continue  # Passe à la ligne suivante.
            scanned += 1  # Compte une décision examinée.
            try:  # Isole les erreurs JSON sans interrompre la conversion.
                row = json.loads(line)  # Décode la décision source.
            except json.JSONDecodeError:  # Ignore les lignes corrompues.
                skipped += 1  # Compte le rejet.
                continue  # Passe à la ligne suivante.
            document = parse_decision(row)  # Convertit la décision.
            if document is None:  # Ignore les décisions filtrées.
                skipped += 1  # Compte le rejet.
                continue  # Passe à la décision suivante.
            if shutil.disk_usage(path.parent).free < 1024 ** 3:  # Préserve au moins 1 Gio libre.
                raise OSError("Conversion arrêtée : moins de 1 Gio libre sur le disque.")  # Arrête proprement.
            out.write(json.dumps(document, ensure_ascii=False) + "\n")  # Écrit le document juridique.
            kept += 1  # Compte le document conservé.
            if kept % 5000 == 0:  # Affiche une progression régulière.
                print(f"  {kept} décisions converties ({scanned} lignes examinées)...", flush=True)  # Informe l'utilisateur.
            if limit and kept >= limit:  # Respecte une limite de test éventuelle.
                break  # Termine la conversion anticipée.
    if not kept:  # Refuse une sortie vide.
        raise ValueError("Aucune décision Judilibre convertie ; vérifier l'URL ou la limite.")  # Signale l'échec.
    temporary.replace(path)  # Publie le corpus seulement après succès.
    return dict(kept=kept, scanned=scanned, skipped=skipped, output=str(path))  # Renvoie les statistiques.


def convert_archive(archive_path, output, limit=0):  # Convertit un export JSONL.gz en corpus JSONL du projet.
    path = Path(output)  # Localise la sortie.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier cible.
    temporary = path.with_suffix(path.suffix + ".part")  # Marque les conversions incomplètes.
    kept, scanned, skipped = 0, 0, 0  # Compte les décisions traitées.
    opener = gzip.open if str(archive_path).endswith(".gz") else open  # Ouvre selon le format source.
    with opener(archive_path, "rt", encoding="utf-8") as stream, temporary.open("w", encoding="utf-8") as out:  # Lit et écrit en flux.
        for line in stream:  # Parcourt les décisions exportées.
            if not line.strip():  # Ignore les lignes vides.
                continue  # Passe à la ligne suivante.
            scanned += 1  # Compte une décision examinée.
            try:  # Isole les erreurs JSON sans interrompre la conversion.
                row = json.loads(line)  # Décode la décision source.
            except json.JSONDecodeError:  # Ignore les lignes corrompues.
                skipped += 1  # Compte le rejet.
                continue  # Passe à la ligne suivante.
            document = parse_decision(row)  # Convertit la décision.
            if document is None:  # Ignore les décisions filtrées.
                skipped += 1  # Compte le rejet.
                continue  # Passe à la décision suivante.
            if shutil.disk_usage(path.parent).free < 1024 ** 3:  # Préserve au moins 1 Gio libre.
                raise OSError("Conversion arrêtée : moins de 1 Gio libre sur le disque.")  # Arrête proprement.
            out.write(json.dumps(document, ensure_ascii=False) + "\n")  # Écrit le document juridique.
            kept += 1  # Compte le document conservé.
            if kept % 5000 == 0:  # Affiche une progression régulière.
                print(f"  {kept} décisions converties ({scanned} lignes examinées)...", flush=True)  # Informe l'utilisateur.
            if limit and kept >= limit:  # Respecte une limite de test éventuelle.
                break  # Termine la conversion anticipée.
    if not kept:  # Refuse une sortie vide.
        raise ValueError("Aucune décision Judilibre convertie ; vérifier l'archive ou la limite.")  # Signale l'échec.
    temporary.replace(path)  # Publie le corpus seulement après succès.
    return dict(kept=kept, scanned=scanned, skipped=skipped, output=str(path))  # Renvoie les statistiques.


def download(output, jurisdiction="cassation", archive=None, limit=0):  # Télécharge puis convertit un export Judilibre.
    filename, label = JURISDICTIONS[jurisdiction]  # Résout le fichier distant.
    archive_path = Path(archive or Path(output).with_name(f"judilibre_{jurisdiction}.jsonl.gz"))  # Localise l'archive locale.
    url = f"{HF_BASE}/{filename}?download=true"  # URL de téléchargement direct.
    if limit and not archive_path.is_file():  # Évite un dump complet quand une limite est demandée.
        stats = convert_stream(url, output, limit=limit)  # Convertit directement depuis le flux distant.
    else:  # Import complet ou reprise depuis une archive locale.
        download_archive(url, archive_path)  # Télécharge l'archive si nécessaire.
        stats = convert_archive(archive_path, output, limit=limit)  # Convertit les décisions en JSONL.
    print(  # Affiche le bilan de conversion.
        f"Corpus Judilibre ({label}) : {stats['output']} "
        f"({stats['kept']} décisions, {stats['scanned']} lignes examinées, {stats['skipped']} ignorées)."
    )
    return stats  # Renvoie les statistiques.


if __name__ == "__main__":  # Expose python -m llm_fr.download_judilibre.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide intégrée.
    parser.add_argument("--output", default="data/legal/raw/judilibre.jsonl")  # Corpus JSONL de sortie.
    parser.add_argument("--jurisdiction", choices=JURISDICTIONS, default="cassation")  # Juridiction à importer.
    parser.add_argument("--archive")  # Archive JSONL.gz déjà téléchargée.
    parser.add_argument("--limit", type=int, default=0)  # Limite de test (0 = tout convertir).
    args = parser.parse_args()  # Lit les options du terminal.
    download(args.output, args.jurisdiction, args.archive, args.limit)  # Lance la collecte Judilibre.
