"""Affiche le suivi de la session longue ou demande son arrêt avec sauvegarde."""

import argparse  # Lit le dossier à suivre et la demande d'arrêt.
import fcntl  # Détermine si le verrou est actuellement détenu.
import json  # Lit les métriques et la configuration sauvegardées.
from collections import deque  # Retient seulement la dernière ligne du journal.
from pathlib import Path  # Localise l'expérience et le fichier STOP.


def status(output, stop=False):  # Inspecte une expérience sans charger les poids en mémoire.
    root = Path(output)  # Définit le dossier suivi.
    active = False  # Ne suppose pas qu'un PID écrit indique un processus vivant.
    lock = root / "training.lock"  # Localise le verrou partagé avec train.py.
    if lock.exists():  # Vérifie les expériences utilisant le verrou de cette version.
        with lock.open("r+") as stream:  # Ouvre sans modifier le contenu du verrou.
            try:  # Essaie brièvement d'acquérir le verrou.
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Réussit uniquement si aucun entraînement ne le détient.
                fcntl.flock(stream, fcntl.LOCK_UN)  # Relâche immédiatement le verrou acquis.
            except BlockingIOError:  # Un autre processus détient réellement le verrou.
                active = True  # Signale l'entraînement en cours.
    print("Entraînement actif." if active else "Aucun entraînement actif dans ce dossier.")  # Affiche l'état réellement observé.
    metrics = root / "metrics.jsonl"  # Localise les mesures périodiques.
    if metrics.exists():  # Autorise le suivi même avant la première validation.
        with metrics.open(encoding="utf-8") as stream:  # Parcourt le journal sans le charger entier.
            last = deque(stream, maxlen=1)  # Ne conserve que sa dernière ligne.
        if last:  # Vérifie que le journal n'est pas vide.
            try:  # Tolère une écriture de journal momentanément incomplète.
                print(json.dumps(json.loads(last[0]), ensure_ascii=False, indent=2))  # Présente étape, pertes, débit et ressources.
            except json.JSONDecodeError:  # Reconnaît une mesure en cours d'écriture.
                print("Une mesure est en cours d'écriture ; relancer le suivi.")  # Évite une erreur bruyante lors du suivi en direct.
    if stop and active:  # Ne laisse pas une demande d'arrêt pour une session qui n'existe pas.
        (root / "STOP").touch()  # Demande une sauvegarde dès la prochaine mise à jour complète.
        print("Arrêt demandé ; attendre la confirmation de sauvegarde dans le journal.")  # Distingue la demande de la fin effective du processus.


if __name__ == "__main__":  # Expose une commande de suivi indépendante du terminal de lancement.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide du module.
    parser.add_argument("--output", default="runs/long")  # Suit par défaut la session longue.
    parser.add_argument("--stop", action="store_true")  # Demande un arrêt propre facultatif.
    args = parser.parse_args()  # Lit les paramètres demandés.
    status(args.output, args.stop)  # Affiche le suivi et transmet éventuellement la demande.
