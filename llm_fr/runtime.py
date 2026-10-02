"""Garde-fous Linux : un seul entraînement par dossier et surveillance des ressources."""

import fcntl  # Utilise les verrous du système, libérés même si le processus meurt.
import os  # Obtient l'identifiant du processus d'entraînement.
import shutil  # Mesure l'espace libre du disque.
from contextlib import contextmanager  # Encadre automatiquement acquisition et libération du verrou.
from pathlib import Path  # Lit les informations Linux et les dossiers du projet.


@contextmanager  # Transforme la fonction en gestionnaire utilisable avec with.
def training_lock(output):  # Empêche deux processus de modifier les mêmes checkpoints.
    root = Path(output)  # Localise l'expérience.
    root.mkdir(parents=True, exist_ok=True)  # Crée son dossier s'il n'existe pas.
    with (root / "training.lock").open("a+") as stream:  # Conserve toujours le même fichier de verrouillage.
        try:  # Détecte un verrou déjà pris sans attendre silencieusement.
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Demande un verrou exclusif non bloquant.
        except BlockingIOError as error:  # Signale un entraînement déjà actif.
            raise RuntimeError(f"Un entraînement utilise déjà {root}.") from error  # Refuse une corruption concurrente des sauvegardes.
        stream.seek(0)  # Revient au début du fichier après acquisition.
        stream.truncate()  # Efface l'ancien PID dont le verrou a été libéré.
        stream.write(str(os.getpid()))  # Identifie le processus qui détient le verrou.
        stream.flush()  # Rend ce PID lisible pour le suivi.
        try:  # Maintient le verrou pendant tout l'apprentissage.
            yield  # Rend le contrôle à la boucle d'entraînement.
        finally:  # Libère le verrou même en cas d'exception.
            fcntl.flock(stream, fcntl.LOCK_UN)  # Autorise une future reprise.


def resources(output):  # Lit une photographie de RAM et disque sans bibliothèque supplémentaire.
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)  # Lit la mémoire du processus courant.
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)  # Lit la disponibilité mémoire globale estimée par Linux.
    return dict(rss_mb=int(status["VmRSS"].split()[0]) / 1024, available_mb=int(memory["MemAvailable"].split()[0]) / 1024, disk_free_mb=shutil.disk_usage(output).free / 1024**2)  # Convertit les mesures en Mio.


def stop_reason(config, output, usage):  # Demande un arrêt propre avant d'épuiser les ressources configurées.
    if (Path(output) / "STOP").exists():  # Vérifie la demande explicite d'arrêt entre mises à jour.
        return "Arrêt demandé par le fichier STOP."  # Laisse la boucle sauvegarder avant de sortir.
    if usage["rss_mb"] > config.get("max_rss_mb", float("inf")):  # Compare la RAM résidente au plafond de surveillance.
        return "Plafond de RAM du processus atteint."  # Évite de poursuivre un entraînement devenu trop lourd.
    if usage["available_mb"] < config.get("min_available_mb", 0):  # Surveille aussi la mémoire utilisée par les autres applications.
        return "Réserve de RAM disponible trop faible."  # Préserve la réactivité de l'ordinateur.
    if usage["disk_free_mb"] < config.get("min_disk_mb", 0):  # Réserve de la place pour les dernières sauvegardes.
        return "Réserve d'espace disque trop faible."  # Arrête avant saturation lorsque la réserve est suffisante.
    return None  # Autorise la prochaine mise à jour.
