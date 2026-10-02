"""Garde-fous Linux et macOS : un seul entraînement par dossier et surveillance des ressources."""

import fcntl
import os
import platform
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def training_lock(output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)

    with (root / "training.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Un entraînement utilise déjà {root}.") from error

        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()

        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _mac_memory():
    """Retourne la RAM du processus et la RAM disponible sur macOS."""

    # RAM utilisée par le processus Python.
    process = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        text=True,
        check=True,
    )

    rss_kb = int(process.stdout.strip())
    rss_mb = rss_kb / 1024

    # RAM disponible estimée à partir de vm_stat.
    vm = subprocess.run(
        ["vm_stat"],
        capture_output=True,
        text=True,
        check=True,
    )

    page_size = 4096

    for line in vm.stdout.splitlines():
        if "page size of" in line:
            page_size = int(line.split("page size of")[1].split("bytes")[0].strip())
            break

    values = {}

    for line in vm.stdout.splitlines():
        if ":" not in line:
            continue

        name, value = line.split(":", 1)
        value = value.strip().rstrip(".").replace(".", "")

        try:
            values[name.strip()] = int(value)
        except ValueError:
            continue

    available_pages = (
        values.get("Pages free", 0)
        + values.get("Pages inactive", 0)
        + values.get("Pages speculative", 0)
    )

    available_mb = available_pages * page_size / 1024**2

    return rss_mb, available_mb


def resources(output):
    """Lit une photographie de RAM et disque compatible Linux et macOS."""

    if platform.system() == "Darwin":
        rss_mb, available_mb = _mac_memory()

    else:
        status = dict(
            line.split(":", 1)
            for line in Path("/proc/self/status").read_text().splitlines()
            if ":" in line
        )

        memory = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
            if ":" in line
        )

        rss_mb = int(status["VmRSS"].split()[0]) / 1024
        available_mb = int(memory["MemAvailable"].split()[0]) / 1024

    return {
        "rss_mb": rss_mb,
        "available_mb": available_mb,
        "disk_free_mb": shutil.disk_usage(output).free / 1024**2,
    }


def stop_reason(config, output, usage):
    if (Path(output) / "STOP").exists():
        return "Arrêt demandé par le fichier STOP."

    if usage["rss_mb"] > config.get("max_rss_mb", float("inf")):
        return "Plafond de RAM du processus atteint."

    if usage["available_mb"] < config.get("min_available_mb", 0):
        return "Réserve de RAM disponible trop faible."

    if usage["disk_free_mb"] < config.get("min_disk_mb", 0):
        return "Réserve d'espace disque trop faible."

    return None