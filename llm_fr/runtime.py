"""Garde-fous multi-plateforme : un seul entraînement par dossier et surveillance des ressources."""

import fcntl  # Utilise les verrous du système, libérés même si le processus meurt.
import os  # Obtient l'identifiant du processus d'entraînement.
import shutil  # Mesure l'espace libre du disque.
import subprocess  # Lit la mémoire sur macOS sans /proc.
import sys  # Distingue Linux et macOS pour l'interprétation des mesures.
from contextlib import contextmanager  # Encadre automatiquement acquisition et libération du verrou.
from pathlib import Path  # Lit les informations système et les dossiers du projet.


def active_training(runs_dir=None):  # Repère un entraînement vivant dans runs/.
    root = Path(runs_dir or "runs")  # Localise le dossier des expériences.
    for lock in root.glob("*/training.lock"):  # Parcourt chaque verrou local.
        pid = lock.read_text().strip()  # Lit le PID enregistré.
        if not pid:  # Ignore un verrou vide.
            continue  # Passe au dossier suivant.
        try:  # Vérifie que le processus existe encore.
            os.kill(int(pid), 0)  # Signale zéro sans tuer le processus.
        except (ProcessLookupError, ValueError, OSError):  # Le PID est mort ou invalide.
            continue  # Traite ce verrou comme orphelin.
        return lock.parent.name  # Retourne le nom du run actif.
    return None  # Aucun entraînement en cours.


@contextmanager  # Transforme la fonction en gestionnaire utilisable avec with.
def training_lock(output):  # Empêche deux processus de modifier les mêmes checkpoints.
    root = Path(output).resolve()  # Localise l'expérience.
    runs_dir = root.parent  # Regroupe toutes les sessions sous runs/.
    global_lock = runs_dir / ".global_training.lock"  # Verrou machine : un seul entraînement à la fois.
    root.mkdir(parents=True, exist_ok=True)  # Crée son dossier s'il n'existe pas.
    runs_dir.mkdir(parents=True, exist_ok=True)  # Crée runs/ si besoin.
    global_stream = global_lock.open("a+")  # Ouvre le verrou global.
    local_stream = (root / "training.lock").open("a+")  # Ouvre le verrou local.
    try:  # Acquiert les deux verrous ou refuse immédiatement.
        fcntl.flock(global_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Réserve la machine entière.
        fcntl.flock(local_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Réserve ce dossier de sortie.
    except BlockingIOError as error:  # Signale un entraînement déjà actif.
        other = active_training(runs_dir)  # Identifie le run concurrent.
        hint = f" ({other})" if other and other != root.name else ""  # Précise quel run bloque.
        raise RuntimeError(f"Un seul entraînement autorisé{hint}. Arrêter l'autre run avant de relancer.") from error
    for stream in (global_stream, local_stream):  # Enregistre le PID sur les deux verrous.
        stream.seek(0)  # Revient au début du fichier.
        stream.truncate()  # Efface l'ancien PID.
        stream.write(str(os.getpid()))  # Identifie le processus courant.
        stream.flush()  # Rend le PID lisible pour le suivi.
    try:  # Maintient les verrous pendant tout l'apprentissage.
        yield  # Rend le contrôle à la boucle d'entraînement.
    finally:  # Libère les verrous même en cas d'exception.
        fcntl.flock(local_stream, fcntl.LOCK_UN)  # Libère le dossier.
        local_stream.close()  # Ferme le descripteur local.
        fcntl.flock(global_stream, fcntl.LOCK_UN)  # Libère la machine.
        global_stream.close()  # Ferme le descripteur global.


def _linux_resources(output):  # Lit /proc quand le noyau Linux l'expose.
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)  # Lit la mémoire du processus courant.
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)  # Lit la disponibilité mémoire globale estimée par Linux.
    return dict(rss_mb=int(status["VmRSS"].split()[0]) / 1024, available_mb=int(memory["MemAvailable"].split()[0]) / 1024, disk_free_mb=shutil.disk_usage(output).free / 1024**2)  # Convertit les mesures en Mio.


def _mac_available_mb():  # Estime la RAM libre sur macOS via vm_stat.
    page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())  # Obtient la taille d'une page mémoire.
    pages = {}  # Accumule les compteurs de pages signalés par le noyau.
    for line in subprocess.check_output(["vm_stat"], text=True).splitlines()[1:]:  # Ignore l'en-tête du rapport.
        if ":" not in line:  # Passe les lignes vides ou mal formées.
            continue  # Continue la lecture sans interrompre l'estimation.
        key, value = line.split(":", 1)  # Sépare le nom du compteur de sa valeur.
        pages[key.strip()] = int(value.strip().rstrip("."))  # Convertit la valeur numérique en entier.
    free_pages = pages.get("Pages free", 0) + pages.get("Pages inactive", 0) + pages.get("Pages speculative", 0)  # Approxime la mémoire réutilisable.
    return free_pages * page_size / 1024**2  # Retourne la mémoire disponible en Mio.


def _portable_resources(output):  # Mesure RAM et disque sans dépendre de /proc.
    rss_kb = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True).strip())  # Lit la RSS courante du processus.
    available_mb = _mac_available_mb() if sys.platform == "darwin" else None  # Utilise vm_stat uniquement sur macOS.
    if available_mb is None:  # Retombe sur /proc si Linux ne l'a pas déjà fourni.
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)  # Lit la mémoire disponible Linux.
        available_mb = int(memory["MemAvailable"].split()[0]) / 1024  # Convertit la valeur en Mio.
    return dict(rss_mb=rss_kb / 1024, available_mb=available_mb, disk_free_mb=shutil.disk_usage(output).free / 1024**2)  # Assemble la photographie portable.


def resources(output):  # Lit une photographie de RAM et disque sans bibliothèque supplémentaire.
    if Path("/proc/self/status").exists() and Path("/proc/meminfo").exists():  # Préfère la voie Linux historique quand elle existe.
        return _linux_resources(output)  # Conserve le comportement déjà validé sous Linux.
    return _portable_resources(output)  # Utilise ps et vm_stat sur macOS ou autres systèmes sans /proc.


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
