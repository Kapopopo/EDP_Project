"""Terminal interactif avec affichage progressif des sorties du modèle local."""

import argparse  # Lit les options du terminal.
import math  # Vérifie que la température est un nombre fini.
import unicodedata  # Identifie les caractères de contrôle du terminal.
from pathlib import Path  # Localise la sauvegarde à recharger.
import torch  # Exécute le modèle et tire les tokens suivants.
from tokenizers import Tokenizer  # Recharge le tokenizer associé aux poids.
from .model import LanguageModel  # Reconstruit notre Transformer.


def visible(text):  # Évite que des sorties aléatoires commandent le terminal.
    return "".join(c for c in text if c in "\n\t" or not unicodedata.category(c).startswith("C"))  # Conserve le texte et les sauts de ligne utiles.


def stream_reply(model, tokenizer, prompt, count, temperature, top_k):  # Produit uniquement les nouveaux morceaux de texte.
    ids = torch.tensor([tokenizer.encode(prompt).ids], dtype=torch.long)  # Encode l'amorce sur CPU.
    generated, emitted = [], 0  # Mémorise les tokens produits et la longueur déjà affichée.
    eos = tokenizer.token_to_id("<|endoftext|>")  # Repère la fin d'un document.
    with torch.inference_mode():  # Désactive les gradients pendant la réponse.
        for _ in range(count):  # Borne la durée et la longueur de sortie.
            logits, _ = model(ids[:, -model.context:])  # Utilise le contexte récent disponible.
            values, indices = torch.topk(logits[:, -1] / temperature, min(top_k, logits.size(-1)))  # Retient les meilleurs candidats.
            token = indices.gather(-1, torch.multinomial(torch.softmax(values, dim=-1), 1))  # Échantillonne un token.
            if token.item() == eos:  # Respecte une fin de texte prédite.
                break  # Arrête la réponse.
            generated.append(token.item())  # Conserve les octets nécessaires au décodage UTF-8.
            ids = torch.cat((ids, token), dim=1)  # Ajoute le token au contexte suivant.
            text = tokenizer.decode(generated).rstrip("\ufffd")  # Attend les octets suivants si le dernier caractère est incomplet.
            if len(text) > emitted:  # Affiche seulement un nouveau fragment décodable.
                yield visible(text[emitted:])  # Fournit le fragment sans réafficher le texte précédent.
                emitted = len(text)  # Actualise la position d'affichage.
    tail = tokenizer.decode(generated)[emitted:]  # Récupère aussi une éventuelle fin UTF-8 invalide.
    if tail:  # N'émet pas de fragment vide.
        yield visible(tail)  # Termine l'affichage sans masquer un caractère de remplacement final.


def chat(checkpoint_path, count=100, temperature=0.8, top_k=40):  # Lance une session interactive persistante.
    if count < 1 or top_k < 1 or not math.isfinite(temperature) or temperature <= 0:  # Refuse les paramètres de génération invalides.
        raise ValueError("count et top-k doivent être positifs ; température finie et strictement positive.")  # Explique la correction.
    torch.set_num_threads(4)  # Laisse de la marge CPU si un entraînement tourne en parallèle.
    path = Path(checkpoint_path)  # Conserve le chemin du checkpoint à actualiser.
    print(f"Sauvegarde suivie : {path}", flush=True)  # Indique quelle phase fournit les réponses.
    print("Modèle local — /quitter pour sortir. Chaque message est indépendant.", flush=True)  # Présente les commandes et l'absence de mémoire de dialogue.
    print("Préentraînement seulement : les réponses peuvent être incohérentes. Les nouveaux poids sont chargés entre les messages.", flush=True)  # Décrit les capacités réelles.
    try:  # Permet une sortie propre avec Ctrl+C ou Ctrl+D.
        while True:  # Attend plusieurs messages sans relancer Python.
            message = input("\nToi > ").strip()  # Lit un message dans le terminal.
            if message == "/quitter":  # Reconnaît la commande de fermeture.
                break  # Termine la boucle.
            if not message:  # Ignore une entrée vide.
                continue  # Attend un véritable message.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)  # Lit une sauvegarde complète publiée par renommage atomique.
            tokenizer = Tokenizer.from_str(checkpoint["tokenizer"])  # Récupère le vocabulaire exact de cette sauvegarde.
            model = LanguageModel(**checkpoint["architecture"])  # Crée les dimensions correspondantes.
            model.load_state_dict(checkpoint["model"])  # Charge les poids disponibles au début du message.
            model.eval()  # Désactive le dropout pour l'inférence.
            step = checkpoint["step"]  # Retient l'étape affichée à l'utilisateur.
            del checkpoint  # Libère les états de l'optimiseur inutiles au dialogue.
            prompt = f"Question : {message}\nRéponse :"  # Propose un format de réponse sans prétendre qu'il a été appris.
            print(f"Modèle (étape {step}) > ", end="", flush=True)  # Affiche immédiatement le début de la sortie.
            for piece in stream_reply(model, tokenizer, prompt, count, temperature, top_k):  # Consomme les fragments au rythme de leur génération.
                print(piece, end="", flush=True)  # Affiche chaque fragment sans attendre la réponse entière.
            print(flush=True)  # Termine la ligne de réponse.
            del model  # Libère le réseau avant la prochaine saisie.
    except (KeyboardInterrupt, EOFError):  # Gère les interruptions clavier habituelles.
        print("\nSession fermée.", flush=True)  # Termine sans trace d'erreur.


if __name__ == "__main__":  # Expose python -m llm_fr.chat.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide intégrée.
    parser.add_argument("--checkpoint")  # Permet de choisir explicitement une phase d'entraînement.
    parser.add_argument("--count", type=int, default=100)  # Fixe la longueur maximale de chaque sortie.
    parser.add_argument("--temperature", type=float, default=0.8)  # Règle la diversité des tokens.
    parser.add_argument("--top-k", type=int, default=40)  # Limite le nombre de candidats par position.
    args = parser.parse_args()  # Lit les options demandées.
    candidates = ("runs/legal_large_turbo/best.pt", "runs/legal_large/best.pt", "runs/legal_medium_refine2/best.pt", "runs/legal_medium_refine/best.pt", "runs/legal_medium/best.pt", "runs/legal_refine3/best.pt", "runs/long/last.pt", "runs/expanded/last.pt", "runs/pilot/last.pt")  # Préfère le meilleur modèle juridique disponible.
    checkpoint_path = args.checkpoint or next((p for p in candidates if Path(p).is_file()), candidates[-1])  # Suit automatiquement la session longue dès qu'elle possède une sauvegarde.
    chat(checkpoint_path, args.count, args.temperature, args.top_k)  # Ouvre le terminal interactif.
