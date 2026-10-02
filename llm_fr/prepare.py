"""Crée un tokenizer BPE français et deux flux de tokens sur disque."""

import argparse  # Expose une commande utilisable dans le terminal.
import json  # Enregistre les paramètres de préparation.
import shutil  # Copie exactement le tokenizer existant lors d'une extension de corpus.
from pathlib import Path  # Construit les chemins de sortie.
import numpy as np  # Écrit les identifiants en entiers compacts.
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # Fournit un BPE entraînable de zéro.
from .corpus import digest, records, split_corpus  # Réutilise les règles de qualité communes.


def prepare(source, output, vocab_size=4096, val_fraction=0.05, seed=42, tokenizer_path=None):  # Prépare les données avec un BPE neuf ou conservé.
    if not 257 <= vocab_size <= 65535:  # Réserve 256 octets et un marqueur de fin, dans un uint16.
        raise ValueError("Le vocabulaire doit contenir entre 257 et 65535 tokens.")  # Évite les débordements.
    root = Path(output)  # Définit le dossier préparé.
    parent = root.parent  # Cherche le volume qui accueillera les données préparées.
    parent.mkdir(parents=True, exist_ok=True)  # Rend la mesure d'espace libre possible.
    required = Path(source).stat().st_size * 3 + 400 * 1024**2  # Prévoit textes nettoyés, tokens et réserve disque avec une marge conservatrice.
    if shutil.disk_usage(parent).free < required:  # Vérifie l'espace avant de commencer les écritures volumineuses.
        raise OSError("Espace insuffisant pour préparer ce corpus en conservant 400 Mio de réserve.")  # Évite une préparation interrompue par saturation.
    stats = split_corpus(source, root, val_fraction, seed)  # Nettoie, déduplique et sépare les documents.
    if tokenizer_path is None:  # Apprend un vocabulaire seulement pour un nouveau modèle.
        tokenizer = Tokenizer(models.BPE())  # Initialise un vocabulaire sans poids ni vocabulaire préentraîné.
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)  # Rend tous les caractères UTF-8 représentables.
        tokenizer.decoder = decoders.ByteLevel()  # Reconstruit correctement les accents depuis les octets.
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())  # Définit les fusions et le token de fin.
        tokenizer.train_from_iterator((r["text"] for r in records(root / "train.jsonl")), trainer=trainer)  # N'apprend jamais sur la validation.
        tokenizer.save(str(root / "tokenizer.json"))  # Sauvegarde le vocabulaire exact.
    else:  # Préserve le sens des identifiants appris lors de la phase précédente.
        tokenizer = Tokenizer.from_file(str(tokenizer_path))  # Recharge le vocabulaire existant sans aucune fusion supplémentaire.
        if tokenizer.get_vocab_size() > 65535 or tokenizer.token_to_id("<|endoftext|>") is None:  # Vérifie le format du flux de tokens.
            raise ValueError("Tokenizer incompatible avec le format uint16 ou sans token de fin.")  # Refuse des données incompatibles.
        shutil.copyfile(tokenizer_path, root / "tokenizer.json")  # Préserve aussi son empreinte octet pour octet.
    eos = tokenizer.token_to_id("<|endoftext|>")  # Récupère la frontière entre documents.
    counts = {}  # Stocke le nombre de tokens de chaque ensemble.
    for split in ("train", "val"):  # Encode séparément entraînement et validation.
        counts[split] = 0  # Initialise le compteur.
        with (root / f"{split}.bin").open("wb") as stream:  # Écrit un flux binaire compact.
            for row in records(root / f"{split}.jsonl"):  # Lit un document à la fois.
                ids = tokenizer.encode(row["text"]).ids + [eos]  # Termine chaque document explicitement.
                np.asarray(ids, dtype="<u2").tofile(stream)  # Utilise deux octets par token, petit-boutiste.
                counts[split] += len(ids)  # Compte les tokens réellement produits.
    names = ("train.bin", "val.bin", "tokenizer.json")  # Liste les artefacts nécessaires à l'apprentissage.
    manifest = dict(documents=stats, tokens=counts, vocab_size=tokenizer.get_vocab_size(), seed=seed, val_fraction=val_fraction, source_sha256=digest(source), sha256={n: digest(root / n) for n in names})  # Identifie données et tokenizer.
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # Écrit le manifeste en dernier, une fois tout terminé.
    print(json.dumps(manifest, indent=2))  # Affiche le bilan de préparation.


if __name__ == "__main__":  # Exécute la commande uniquement lors d'un lancement direct.
    parser = argparse.ArgumentParser(description=__doc__)  # Initialise l'aide intégrée.
    parser.add_argument("source")  # Attend un JSONL documenté.
    parser.add_argument("output")  # Attend un nouveau dossier.
    parser.add_argument("--vocab-size", type=int, default=4096)  # Choisit la taille maximale du vocabulaire.
    parser.add_argument("--val-fraction", type=float, default=0.05)  # Réserve 5 % des documents par défaut.
    parser.add_argument("--seed", type=int, default=42)  # Stabilise la séparation.
    parser.add_argument("--tokenizer")  # Réutilise le tokenizer du modèle pour une nouvelle phase de préentraînement.
    args = parser.parse_args()  # Lit les arguments du terminal.
    prepare(args.source, args.output, args.vocab_size, args.val_fraction, args.seed, args.tokenizer)  # Lance la préparation.
