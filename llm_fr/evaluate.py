"""Compare des checkpoints sur exactement les mêmes fenêtres de validation."""

import argparse  # Lit la liste des checkpoints à comparer.
import json  # Enregistre un rapport exploitable.
import math  # Convertit la perte en perplexité.
from pathlib import Path  # Localise corpus et rapports.
import numpy as np  # Lit le binaire sans le charger entièrement en mémoire.
import torch  # Exécute les modèles sur CPU.
from .corpus import digest  # Contrôle l'identité de la validation et du vocabulaire.
from .model import LanguageModel  # Reconstruit les modèles sauvegardés.
from .train import measure_loss  # Réutilise le calcul de perte sans gradients.


def evaluate(checkpoints, data, batches=128, seed=12345, output=None):  # Compare les modèles sur un échantillon fixe distinct du suivi d'entraînement.
    if batches <= 0:  # Refuse une mesure sans lots.
        raise ValueError("Le nombre de lots doit être positif.")  # Signale l'argument incorrect.
    torch.set_num_threads(4)  # Borne la consommation de cœurs CPU.
    root = Path(data)  # Identifie le corpus de validation.
    manifest = json.loads((root / "manifest.json").read_text())  # Charge les empreintes des données.
    for name in ("val.bin", "tokenizer.json"):  # Contrôle les fichiers effectivement utilisés.
        if digest(root / name) != manifest["sha256"][name]:  # Vérifie leur intégrité.
            raise ValueError(f"Empreinte incorrecte : {name}")  # Arrête une comparaison incohérente.
    tokens = np.memmap(root / "val.bin", dtype="<u2", mode="r")  # Accède progressivement à la validation.
    results, reference_context = [], None  # Accumule les résultats et impose un contexte commun.
    for path in checkpoints:  # Évalue un modèle à la fois pour limiter la RAM.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # Recharge les poids demandés.
        if checkpoint["manifest"]["sha256"]["tokenizer.json"] != manifest["sha256"]["tokenizer.json"]:  # Exige une échelle de perte comparable.
            raise ValueError("Comparaison impossible avec des tokenizers différents.")  # Refuse une perplexité trompeuse.
        context = checkpoint["architecture"]["context"]  # Récupère le contexte réellement appris.
        if len(tokens) <= context or (reference_context is not None and reference_context != context):  # Vérifie la longueur des fenêtres.
            raise ValueError("Il faut assez de tokens et des modèles de même contexte.")  # Empêche des fenêtres incomparables.
        reference_context = context  # Fixe la longueur pour les modèles suivants.
        model = LanguageModel(**checkpoint["architecture"])  # Reconstruit le Transformer sur CPU.
        model.load_state_dict(checkpoint["model"])  # Charge les paramètres appris.
        config = dict(context=context, batch_size=4, eval_batches=batches)  # Fixe un lot identique pour tous les checkpoints.
        loss = measure_loss(model, tokens, config, torch.device("cpu"), seed)  # Évalue exactement les mêmes positions du flux.
        result = dict(checkpoint=str(path), checkpoint_sha256=digest(path), step=checkpoint["step"], val_loss=loss, perplexity=math.exp(min(loss, 80)), evaluated_tokens=batches * 4 * context)  # Rend la mesure traçable.
        results.append(result)  # Ajoute ce modèle au rapport.
        print(json.dumps(result), flush=True)  # Affiche le résultat dès qu'il est disponible.
        del model, checkpoint  # Libère les poids et l'optimiseur chargé avant le modèle suivant.
    report = dict(data=str(root), validation_sha256=manifest["sha256"]["val.bin"], seed=seed, batches=batches, results=results)  # Décrit le protocole de comparaison.
    if output is not None:  # Écrit un rapport uniquement lorsqu'il est demandé.
        with Path(output).open("x", encoding="utf-8") as stream:  # Évite d'écraser un rapport antérieur.
            json.dump(report, stream, indent=2)  # Conserve la comparaison complète.
    return report  # Permet aux tests d'examiner les mesures.


if __name__ == "__main__":  # Expose le comparateur dans le terminal.
    parser = argparse.ArgumentParser(description=__doc__)  # Crée l'aide de la commande.
    parser.add_argument("checkpoints", nargs="+")  # Accepte un ou plusieurs checkpoints.
    parser.add_argument("--data", required=True)  # Exige le corpus servant à la comparaison.
    parser.add_argument("--batches", type=int, default=128)  # Choisit la précision de l'estimation.
    parser.add_argument("--seed", type=int, default=12345)  # Fixe un échantillon reproductible.
    parser.add_argument("--output")  # Choisit un nouveau rapport JSON facultatif.
    args = parser.parse_args()  # Lit les paramètres du terminal.
    evaluate(args.checkpoints, args.data, args.batches, args.seed, args.output)  # Lance les évaluations successives.
