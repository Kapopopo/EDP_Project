"""Télécharge et convertit les données LEGI (Légifrance) en corpus JSONL local."""

import argparse  # Configure la collecte Légifrance.
import hashlib  # Empreinte les documents convertis.
import json  # Écrit le corpus au format JSONL.
import re  # Nettoie le HTML léger des articles LEGI.
import shutil  # Surveille l'espace disque disponible.
import tarfile  # Lit les archives DILA sans extraction complète.
import xml.etree.ElementTree as ET  # Parse les articles LEGI en XML.
from datetime import date  # Filtre les versions en vigueur.
from pathlib import Path  # Manipule les chemins de sortie.
from urllib.request import urlretrieve  # Télécharge le dump LEGI officiel.

from .corpus import clean  # Réutilise la normalisation de texte du projet.


DEFAULT_URL = "https://echanges.dila.gouv.fr/OPENDATA/LEGI/Freemium_legi_global_20250713-140000.tar.gz"  # Stock LEGI complet.
LICENSE = "Licence Ouverte 2.0 (DILA — https://www.legifrance.gouv.fr)"  # Licence des données LEGI.
TAG_RE = re.compile(r"<[^>]+>")  # Retire les balises HTML simples des articles.


def download_archive(url, target):  # Télécharge une archive LEGI si elle est absente.
    path = Path(target)  # Localise le fichier cible.
    if path.is_file() and path.stat().st_size > 0:  # Ignore un téléchargement déjà présent.
        print(f"Archive déjà présente : {path} ({path.stat().st_size // (1024 ** 2)} Mio)")  # Indique la réutilisation.
        return path  # Renvoie le chemin existant.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier de stockage.
    temporary = path.with_suffix(path.suffix + ".part")  # Marque les téléchargements incomplets.
    print(f"Téléchargement LEGI : {url}")  # Affiche la source officielle.
    urlretrieve(url, temporary)  # Télécharge l'archive distante.
    temporary.replace(path)  # Publie le fichier seulement après succès.
    print(f"Archive téléchargée : {path}")  # Confirme la fin du téléchargement.
    return path  # Renvoie le chemin final.


def strip_html(text):  # Retire les balises HTML et normalise les sauts de ligne.
    text = TAG_RE.sub(" ", text or "")  # Supprime les balises HTML légères.
    text = re.sub(r"\s+", " ", text).strip()  # Réduit les espaces superflus.
    return text  # Renvoie le texte brut.


def pick_title(context):  # Choisit le titre juridique le plus pertinent d'un article.
    if context is None:  # Gère les articles sans contexte.
        return "Texte LEGI"  # Titre de secours.
    titles = []  # Accumule les titres disponibles.
    for node in context.iter("TITRE_TXT"):  # Parcourt les titres du contexte.
        title = strip_html(node.text or "") or node.attrib.get("c_titre_court", "")  # Lit le titre long ou court.
        if title:  # Ignore les titres vides.
            titles.append((node.attrib.get("fin", ""), title))  # Conserve la date de fin associée.
    if not titles:  # Retombe sur le titre du texte parent.
        text_node = context.find("TEXTE")  # Repère le nœud texte principal.
        if text_node is not None:  # Utilise les attributs du texte si présents.
            return text_node.attrib.get("nature", "Texte") + " " + text_node.attrib.get("num", "")  # Compose un titre minimal.
        return "Texte LEGI"  # Titre de secours.
    for fin, title in titles:  # Privilégie une version encore en vigueur.
        if fin.startswith("2999"):  # La DILA marque ainsi les versions actuelles.
            return title  # Retourne le titre en vigueur.
    return titles[-1][1]  # Sinon conserve le dernier titre connu.


def in_force(etat, date_fin):  # Détermine si une version d'article est exploitable pour le RAG.
    if etat == "ABROGE":  # Exclut les articles abrogés.
        return False  # Article hors service.
    if not date_fin or date_fin.startswith("2999"):  # Conserve les versions sans fin ou encore actives.
        return True  # Article exploitable.
    try:  # Compare la date de fin au jour courant.
        year, month, day = (int(part) for part in date_fin.split("-"))  # Décode la date LEGI.
        return date(year, month, day) >= date.today()  # Conserve les versions encore valides.
    except ValueError:  # Tolère un format de date inattendu.
        return True  # Préfère conserver l'article plutôt que le perdre.


def parse_article(xml_bytes):  # Convertit un fichier LEGIARTI XML en document JSONL.
    root = ET.fromstring(xml_bytes)  # Charge l'article XML.
    article_id = (root.findtext(".//META_COMMUN/ID") or "").strip()  # Identifiant Légifrance.
    article_num = (root.findtext(".//META_ARTICLE/NUM") or "").strip()  # Numéro d'article.
    etat = (root.findtext(".//META_ARTICLE/ETAT") or "").strip()  # État juridique de la version.
    date_fin = (root.findtext(".//META_ARTICLE/DATE_FIN") or "").strip()  # Date de fin de validité.
    if not article_id or not in_force(etat, date_fin):  # Ignore les articles abrogés ou expirés.
        return None  # Aucun document à produire.
    body = root.find(".//BLOC_TEXTUEL/CONTENU")  # Localise le corps de l'article.
    content = clean(strip_html("".join(body.itertext()) if body is not None else ""))  # Extrait le texte utile.
    if len(content) < 40:  # Ignore les articles trop courts pour le RAG.
        return None  # Fragment inutile.
    context = root.find(".//CONTEXTE")  # Récupère le contexte juridique.
    code_title = pick_title(context)  # Déduit le code ou le texte parent.
    title = f"Article {article_num} — {code_title}".strip(" —") if article_num else code_title  # Formate le titre affiché.
    source = f"https://www.legifrance.gouv.fr/codes/article_lc/{article_id}"  # Lien officiel vers l'article.
    text = f"{title}\n\n{content}"  # Préfixe le contenu par sa référence juridique.
    key = hashlib.sha256(text.casefold().encode()).hexdigest()  # Empreinte déterministe du document.
    return dict(  # Construit un document compatible avec corpus.py.
        text=text,
        source=source,
        title=title,
        license=LICENSE,
        language="fr",
        domain="legifrance",
        dataset="LEGI",
        article_id=article_id,
        article_num=article_num,
        code=code_title,
        etat=etat or "VIGUEUR",
        sha256=key,
    )


def convert_archive(archive_path, output, limit=0):  # Parcourt une archive LEGI et écrit le corpus JSONL.
    path = Path(output)  # Localise la sortie.
    path.parent.mkdir(parents=True, exist_ok=True)  # Crée le dossier cible.
    temporary = path.with_suffix(path.suffix + ".part")  # Marque les conversions incomplètes.
    kept, scanned, skipped = 0, 0, 0  # Compte les articles traités.
    mode = "a" if temporary.exists() else "x"  # Reprend ou crée le flux de sortie.
    with tarfile.open(archive_path, "r:gz") as archive, temporary.open(mode, encoding="utf-8") as stream:  # Lit l'archive sans l'extraire entièrement.
        for member in archive:  # Parcourt les fichiers de l'archive.
            if not member.isfile() or not member.name.endswith(".xml") or "LEGIARTI" not in member.name:  # Filtre les articles LEGI.
                continue  # Ignore les autres fichiers XML.
            scanned += 1  # Compte un article examiné.
            payload = archive.extractfile(member)  # Ouvre le fichier courant.
            if payload is None:  # Ignore les entrées illisibles.
                skipped += 1  # Compte le rejet.
                continue  # Passe au fichier suivant.
            try:  # Isole les erreurs XML sans interrompre la conversion.
                document = parse_article(payload.read())  # Convertit l'article XML.
            except ET.ParseError:  # Ignore les fichiers XML corrompus.
                skipped += 1  # Compte le rejet.
                continue  # Passe au fichier suivant.
            if document is None:  # Ignore les articles filtrés.
                skipped += 1  # Compte le rejet.
                continue  # Passe au fichier suivant.
            if shutil.disk_usage(path.parent).free < 1024 ** 3:  # Préserve au moins 1 Gio libre.
                raise OSError("Conversion arrêtée : moins de 1 Gio libre sur le disque.")  # Arrête proprement.
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")  # Écrit le document juridique.
            kept += 1  # Compte le document conservé.
            if kept % 5000 == 0:  # Affiche une progression régulière.
                print(f"  {kept} articles convertis ({scanned} XML examinés)...", flush=True)  # Informe l'utilisateur.
            if limit and kept >= limit:  # Respecte une limite de test éventuelle.
                break  # Termine la conversion anticipée.
    if not kept:  # Refuse une sortie vide.
        raise ValueError("Aucun article LEGI converti ; vérifier l'archive ou le filtre de vigueur.")  # Signale l'échec.
    temporary.replace(path)  # Publie le corpus seulement après succès.
    return dict(kept=kept, scanned=scanned, skipped=skipped, output=str(path))  # Renvoie les statistiques.


def download(output, archive=None, url=DEFAULT_URL, limit=0):  # Télécharge puis convertit le fonds LEGI.
    archive_path = Path(archive or Path(output).with_suffix(".tar.gz"))  # Localise l'archive locale.
    download_archive(url, archive_path)  # Télécharge l'archive si nécessaire.
    stats = convert_archive(archive_path, output, limit=limit)  # Convertit les articles en JSONL.
    print(  # Affiche le bilan de conversion.
        f"Corpus Légifrance : {stats['output']} "
        f"({stats['kept']} articles, {stats['scanned']} XML examinés, {stats['skipped']} ignorés)."
    )
    return stats  # Renvoie les statistiques.


if __name__ == "__main__":  # Expose python -m llm_fr.download_legifrance.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide intégrée.
    parser.add_argument("--output", default="data/legal/raw/legifrance.jsonl")  # Corpus JSONL de sortie.
    parser.add_argument("--archive")  # Archive LEGI déjà téléchargée.
    parser.add_argument("--url", default=DEFAULT_URL)  # URL du dump LEGI à récupérer.
    parser.add_argument("--limit", type=int, default=0)  # Limite de test (0 = tout convertir).
    args = parser.parse_args()  # Lit les options du terminal.
    download(args.output, args.archive, args.url, args.limit)  # Lance la collecte officielle.
