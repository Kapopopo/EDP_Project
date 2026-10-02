"""Inspecte une continuation de texte ; ce modèle n'est pas encore un chatbot."""

import argparse  # Lit le checkpoint et l'amorce.
import torch  # Exécute le réseau et l'échantillonnage.
from tokenizers import Tokenizer  # Recharge le vocabulaire associé aux poids.
from .model import LanguageModel  # Reconstruit l'architecture sauvegardée.


def generate(checkpoint_path, prompt, count=100, temperature=0.8, top_k=40):  # Produit une courte continuation sur CPU.
    if not prompt or count < 1 or temperature <= 0 or top_k < 1:  # Valide les paramètres de génération.
        raise ValueError("Amorce non vide, count/top-k positifs et temperature strictement positive requis.")  # Évite les distributions invalides.
    torch.set_num_threads(4)  # Garde une consommation CPU modérée.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # Charge le checkpoint local.
    tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])  # Utilise exactement le tokenizer de l'entraînement.
    model = LanguageModel(**checkpoint["architecture"])  # Recrée les mêmes dimensions.
    model.load_state_dict(checkpoint["model"])  # Remplace les poids aléatoires par les poids appris.
    model.eval()  # Désactive le dropout.
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)  # Transforme l'amorce en tokens.
    eos = tokenizer.token_to_id("<|endoftext|>")  # Identifie le token de fin.
    with torch.inference_mode():  # Économise la mémoire des gradients.
        for _ in range(count):  # Borne la longueur produite.
            logits, _ = model(ids[:, -model.context:])  # Conserve uniquement le contexte disponible.
            values, indices = torch.topk(logits[:, -1, :] / temperature, min(top_k, logits.size(-1)))  # Restreint aux tokens les mieux classés.
            choice = torch.multinomial(torch.softmax(values, dim=-1), 1)  # Tire une continuation selon les probabilités.
            token = indices.gather(-1, choice)  # Retrouve l'identifiant dans le vocabulaire complet.
            if token.item() == eos:  # Arrête à une frontière de document.
                break  # Termine la génération.
            ids = torch.cat((ids, token), dim=1)  # Ajoute le token à la séquence.
    return tokenizer.decode(ids[0].tolist())  # Reconstruit le texte UTF-8.


if __name__ == "__main__":  # Rend le fichier exécutable comme module.
    parser = argparse.ArgumentParser(description=__doc__)  # Initialise l'aide.
    parser.add_argument("checkpoint")  # Attend best.pt ou last.pt.
    parser.add_argument("prompt")  # Attend une amorce française.
    parser.add_argument("--count", type=int, default=100)  # Fixe le maximum de tokens ajoutés.
    parser.add_argument("--temperature", type=float, default=0.8)  # Contrôle la diversité.
    parser.add_argument("--top-k", type=int, default=40)  # Contrôle le nombre de candidats.
    args = parser.parse_args()  # Lit les arguments.
    print(generate(args.checkpoint, args.prompt, args.count, args.temperature, args.top_k))  # Affiche la continuation.
