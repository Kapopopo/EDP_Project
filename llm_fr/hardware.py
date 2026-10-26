"""Détection matérielle : adapte corpus, entraînement et déploiement à la machine."""

import json  # Sérialise le profil détecté.
import os  # Compte les cœurs sous Linux.
import shutil  # Mesure l'espace disque libre.
import subprocess  # Lit la RAM et le nombre de cœurs.
import sys  # Distingue macOS et Linux.
from pathlib import Path  # Écrit le profil sur disque.


def _total_ram_mb():  # Retourne la RAM physique totale en Mio.
    if sys.platform == "darwin":  # Utilise sysctl sur macOS.
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / 1024**2
    for line in Path("/proc/meminfo").read_text().splitlines():  # Parcourt meminfo sous Linux.
        if line.startswith("MemTotal:"):  # Repère la ligne totale.
            return int(line.split()[1]) / 1024  # Convertit les kio en Mio.
    return 8192.0  # Valeur de repli prudente.


def _cpu_count():  # Compte les cœurs logiques disponibles.
    return int(subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()) if sys.platform == "darwin" else os.cpu_count() or 4


def _accelerator():  # Identifie le meilleur accélérateur utilisable.
    try:  # Importe PyTorch seulement si disponible.
        import torch  # Détecte CUDA ou MPS.
    except ImportError:  # Continue sans accélérateur détecté.
        return "cpu"  # Reste sur processeur.
    if torch.cuda.is_available():  # Préfère NVIDIA lorsque présent.
        return "cuda"  # Active le chemin GPU NVIDIA.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # Détecte Apple Silicon.
        return "mps"  # Active le GPU intégré Apple.
    return "cpu"  # Retombe sur CPU.


def profile(root="."):
    ram_mb = _total_ram_mb()  # Lit la mémoire installée.
    ram_gb = ram_mb / 1024  # Convertit en gibioctets lisibles.
    disk_mb = shutil.disk_usage(root).free / 1024**2  # Mesure l'espace libre.
    disk_gb = disk_mb / 1024  # Convertit le disque en gibioctets.
    cores = _cpu_count()  # Compte les cœurs logiques.
    device = _accelerator()  # Choisit cpu, mps ou cuda.

    if ram_gb >= 48:  # Station de travail haut de gamme (ex. 64 Go).
        tier, corpus_docs, max_mb, model, batch, accum = "max", 250_000, 20_000, "xxl", 24, 2
    elif ram_gb >= 24:  # Machine performante.
        tier, corpus_docs, max_mb, model, batch, accum = "pro", 100_000, 8_000, "xxl", 16, 2
    elif ram_gb >= 12:  # Poste standard récent.
        tier, corpus_docs, max_mb, model, batch, accum = "standard", 25_000, 2_000, "large", 512, 1
    else:  # PC modeste ou ancien.
        tier, corpus_docs, max_mb, model, batch, accum = "minimal", 5_000, 500, "large", 256, 1

    corpus_docs = min(corpus_docs, max(5_000, int(disk_gb * 200)))  # Borne par le disque (~5 Mo/doc brut).
    max_mb = min(max_mb, max(500, int(disk_mb * 0.05)))  # Ne consomme pas plus de 5 % du disque libre.

    deploy = "xxl" if ram_gb >= 16 else "large"  # Modèle conseillé pour l'inférence locale.
    threads = max(2, min(cores, 10))  # Laisse une marge système sur les gros CPU.

    return dict(
        tier=tier,
        ram_gb=round(ram_gb, 1),
        disk_gb=round(disk_gb, 1),
        cores=cores,
        device=device,
        corpus_docs=corpus_docs,
        corpus_label=f"{corpus_docs // 1000}k" if corpus_docs >= 1000 else str(corpus_docs),
        max_mb=max_mb,
        train_model=model,
        batch_size=batch,
        accumulation=accum,
        threads=threads,
        deploy_model=deploy,
        min_available_mb=max(2000, int(ram_mb * 0.08)),
        max_rss_mb=int(ram_mb * 0.85),
    )


def save(path="runs/hardware_profile.json", root="."):
    data = profile(root)  # Détecte le profil courant.
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # Crée runs/ si besoin.
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")  # Persiste pour les scripts shell.
    return data


if __name__ == "__main__":  # Affiche ou enregistre le profil détecté.
    import argparse  # Lit --save depuis le terminal.
    parser = argparse.ArgumentParser(description="Profil matériel pour corpus et entraînement.")  # Aide intégrée.
    parser.add_argument("--save", action="store_true")  # Écrit runs/hardware_profile.json.
    parser.add_argument("--root", default=".")  # Racine pour mesurer le disque.
    args = parser.parse_args()  # Parse les options.
    data = save() if args.save else profile(args.root)  # Détecte et éventuellement sauvegarde.
    print(json.dumps(data, indent=2))  # Affiche le profil lisible.
