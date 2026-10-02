"""Préentraînement avec mémoire bornée, validation fixe et reprise complète."""

import argparse  # Lit les options de lancement.
import json  # Lit les configurations et écrit les métriques.
import math  # Calcule la décroissance cosinus et la perplexité.
import time  # Mesure le débit observé.
from pathlib import Path  # Manipule les dossiers de données et de sauvegarde.
import numpy as np  # Accède aux tokens par projection mémoire.
import torch  # Effectue l'apprentissage différentiable.
from .corpus import digest  # Vérifie l'identité des données.
from .model import LanguageModel  # Importe notre architecture écrite de zéro.
from .runtime import resources, stop_reason, training_lock  # Protège la machine et les checkpoints pendant les sessions longues.


def batch(tokens, config, generator, device):  # Tire des fenêtres de tokens sans tout charger en RAM.
    length = config["context"]  # Fixe le nombre de prédictions par fenêtre.
    starts = torch.randint(len(tokens) - length, (config["batch_size"],), generator=generator).tolist()  # Choisit des départs reproductibles.
    windows = np.stack([tokens[s:s + length + 1] for s in starts]).astype(np.int64)  # Copie seulement un petit lot avec sa dernière cible.
    values = torch.from_numpy(windows).to(device)  # Transfère les identifiants vers le matériel choisi.
    return values[:, :-1], values[:, 1:]  # Décale les cibles d'un token.


def atomic_save(payload, path):  # Évite une sauvegarde finale partiellement écrite.
    temporary = path.with_suffix(".tmp")  # Utilise un fichier temporaire voisin.
    torch.save(payload, temporary)  # Sérialise les tenseurs et les états.
    temporary.replace(path)  # Remplace atomiquement la sauvegarde précédente.


def measure_loss(model, tokens, config, device, seed):  # Mesure une perte sur un échantillon fixe sans modifier le hasard d'entraînement.
    model.eval()  # Désactive le dropout pour comparer équitablement les étapes.
    generator = torch.Generator().manual_seed(seed)  # Isole et fixe les fenêtres évaluées.
    total = 0.0  # Initialise la moyenne des pertes.
    with torch.inference_mode():  # Évite de conserver des gradients pour l'évaluation.
        for _ in range(config["eval_batches"]):  # Parcourt le nombre de lots choisi.
            x, y = batch(tokens, config, generator, device)  # Tire les mêmes fenêtres à chaque mesure.
            _, loss = model(x, y)  # Calcule la perte FP32.
            total += loss.item() / config["eval_batches"]  # Moyenne les lots de même taille.
    if not math.isfinite(total):  # Repère une divergence numérique.
        raise FloatingPointError("Évaluation non finie.")  # Empêche une comparaison invalide.
    return total  # Renvoie la perte moyenne en nats par token.


def initialize_from(model, checkpoint_path, architecture, manifest):  # Transfère les poids d'une phase antérieure, sans ses moments AdamW.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)  # Charge une sauvegarde compatible avec notre format.
    if checkpoint["architecture"] != architecture:  # Exige les mêmes dimensions et opérations.
        raise ValueError("L'initialisation exige la même architecture que le checkpoint.")  # Refuse une adaptation implicite du réseau.
    if checkpoint["manifest"]["sha256"]["tokenizer.json"] != manifest["sha256"]["tokenizer.json"]:  # Compare les identifiants via l'empreinte du tokenizer entier.
        raise ValueError("Tokenizer différent : préparer le corpus avec --tokenizer pour conserver les identifiants.")  # Empêche de changer le sens des poids appris.
    if any(checkpoint["manifest"][key] != manifest[key] for key in ("seed", "val_fraction")):  # Préserve l'affectation des documents identiques entre phases.
        raise ValueError("Conserver seed et val_fraction pour éviter de mélanger les ensembles entre phases.")  # Protège l'ancienne validation de la réutilisation comme entraînement.
    model.load_state_dict(checkpoint["model"])  # Transfère effectivement tous les poids déjà appris.
    old_config = checkpoint["config"]  # Retrouve le lot effectif de la phase précédente.
    previous_tokens = checkpoint.get("origin", {}).get("tokens_before_phase", 0) + checkpoint["step"] * old_config["batch_size"] * old_config["context"] * old_config["accumulation"]  # Cumule l'exposition des phases successives.
    return dict(checkpoint=str(Path(checkpoint_path).resolve()), sha256=digest(checkpoint_path), parent_step=checkpoint["step"], tokens_before_phase=previous_tokens)  # Enregistre la filiation de la nouvelle expérience.


def train(config_path, data_path, output_path, resume=False, stop_after=None, init_from=None):  # Verrouille le dossier pendant une session entière.
    with training_lock(output_path):  # Refuse un second écrivain sur les mêmes fichiers.
        return _train(config_path, data_path, output_path, resume, stop_after, init_from)  # Lance la session protégée.


def _train(config_path, data_path, output_path, resume=False, stop_after=None, init_from=None):  # Pilote une reprise exacte ou une nouvelle phase depuis des poids existants.
    if resume and init_from is not None:  # Distingue reprise complète et transfert de poids.
        raise ValueError("Choisir --resume ou --init-from, jamais les deux.")  # Évite une ambiguïté sur l'état à restaurer.
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))  # Charge les réglages demandés.
    positive = ("threads", "context", "width", "heads", "layers", "batch_size", "accumulation", "steps", "eval_every", "eval_batches", "patience")  # Liste les compteurs obligatoirement positifs.
    if any(type(config[k]) is not int or config[k] <= 0 for k in positive):  # Refuse les valeurs ambiguës ou nulles.
        raise ValueError("Les dimensions et compteurs doivent être des entiers positifs.")  # Explique l'erreur.
    if not 0 <= config["warmup"] < config["steps"] or config["learning_rate"] <= 0 or config["weight_decay"] < 0:  # Vérifie la programmation du taux.
        raise ValueError("Taux positif, weight_decay positif ou nul et warmup inférieur à steps requis.")  # Guide la correction.
    if stop_after is not None and stop_after <= 0:  # Vérifie la durée d'une session courte.
        raise ValueError("--stop-after doit être positif.")  # Refuse une session vide.
    device = torch.device(config["device"])  # Choisit explicitement CPU ou CUDA.
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):  # N'annonce que les matériels pris en charge.
        raise ValueError("Choisir cpu, ou cuda avec un PyTorch CUDA et une carte NVIDIA disponibles.")  # Évite une sélection silencieuse.
    torch.set_num_threads(config["threads"])  # Évite de saturer inutilement tous les cœurs.
    torch.manual_seed(config["seed"])  # Stabilise initialisation et dropout.
    root, output = Path(data_path), Path(output_path)  # Résout les chemins de travail.
    manifest = json.loads((root / "manifest.json").read_text())  # Charge l'identité du corpus terminé.
    for name, expected in manifest["sha256"].items():  # Contrôle les artefacts consommés.
        if digest(root / name) != expected:  # Détecte une modification ou une corruption.
            raise ValueError(f"Empreinte incorrecte : {name}")  # Refuse un entraînement incohérent.
    streams = {s: np.memmap(root / f"{s}.bin", dtype="<u2", mode="r") for s in ("train", "val")}  # Laisse les gros flux sur disque.
    literature = None  # Garde le comportement historique lorsqu'aucun corpus littéraire n'est configuré.
    probability = config.get("literature_probability", 0.0)  # Définit la proportion de micro-lots littéraires.
    if not 0 <= probability < 1 or bool(probability) != bool(config.get("literature_data")):  # Exige un chemin et une proportion cohérents.
        raise ValueError("literature_data et literature_probability dans ]0,1[ doivent être fournis ensemble.")  # Refuse un mélange mal défini.
    if probability:  # Ouvre le second corpus sans le recopier ni le charger entièrement.
        literature_root = Path(config["literature_data"])  # Localise les textes littéraires préparés.
        literature_manifest = json.loads((literature_root / "manifest.json").read_text())  # Charge leur identité.
        for name, expected in literature_manifest["sha256"].items():  # Vérifie tous les fichiers de ce second corpus.
            if digest(literature_root / name) != expected:  # Détecte corruption ou modification.
                raise ValueError(f"Empreinte littéraire incorrecte : {name}")  # Évite une reprise sur des données altérées.
        if literature_manifest["sha256"]["tokenizer.json"] != manifest["sha256"]["tokenizer.json"]:  # Impose le même vocabulaire pour les deux domaines.
            raise ValueError("Les deux corpus doivent utiliser le même tokenizer.")  # Protège le sens des identifiants.
        manifest["literature"] = literature_manifest  # Inclut le second corpus dans la comparaison de reprise exacte.
        literature = {s: np.memmap(literature_root / f"{s}.bin", dtype="<u2", mode="r") for s in ("train", "val")}  # Ouvre uniquement des projections mémoire.
    all_streams = list(streams.values()) + ([] if literature is None else list(literature.values()))  # Réunit les flux pour vérifier leurs longueurs.
    if any(len(stream) <= config["context"] for stream in all_streams):  # Vérifie que chaque ensemble fournit une fenêtre.
        raise ValueError("Pas assez de tokens : ajouter des documents ou réduire context.")  # Évite un échantillonnage vide.
    if not resume and output.exists() and any(p.name != "training.lock" for p in output.iterdir()):  # Protège une expérience existante tout en autorisant son verrou.
        raise ValueError("Dossier de sortie non vide : utiliser --resume ou un nouveau dossier.")  # Explique les deux choix.
    output.mkdir(parents=True, exist_ok=True)  # Crée le dossier de l'expérience.
    architecture = {k: config[k] for k in ("context", "width", "heads", "layers", "dropout")}  # Isole les paramètres du réseau.
    architecture["vocab_size"] = manifest["vocab_size"]  # Utilise la taille réellement apprise du tokenizer.
    model = LanguageModel(**architecture).to(device)  # Crée le modèle aléatoire sur le matériel choisi.
    origin = {} if init_from is None else initialize_from(model, init_from, architecture, manifest)  # Charge les poids avant de créer le nouvel optimiseur.
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": config["weight_decay"]}, {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}]  # Exclut biais et normalisations de la pénalisation.
    optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"])  # Met à jour les poids avec des moments adaptatifs.
    amp = device.type == "cuda"  # Réserve la précision mixte au GPU.
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16  # Préfère BF16 lorsque disponible.
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)  # Protège les petits gradients FP16.
    generator = torch.Generator().manual_seed(config["seed"] + 1)  # Isole le hasard du tirage des données.
    start, best, stale = 0, float("inf"), 0  # Initialise progression, meilleure validation et patience.
    if resume:  # Recharge une expérience interrompue.
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)  # Lit uniquement le format de checkpoint prévu.
        if checkpoint["config"] != config or checkpoint["manifest"] != manifest:  # Interdit de changer silencieusement l'expérience.
            raise ValueError("La reprise exige la même configuration et le même corpus.")  # Protège la comparabilité.
        model.load_state_dict(checkpoint["model"])  # Restaure les paramètres appris.
        optimizer.load_state_dict(checkpoint["optimizer"])  # Restaure les moments de l'optimiseur.
        scaler.load_state_dict(checkpoint["scaler"])  # Restaure l'échelle de précision mixte.
        generator.set_state(checkpoint["generator"])  # Reprend le prochain tirage de fenêtres.
        torch.set_rng_state(checkpoint["rng"])  # Reprend le hasard CPU après création du modèle.
        if amp:  # Restaure également le hasard du dropout GPU.
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])  # Reprend les générateurs des cartes.
        start, best, stale = checkpoint["step"], checkpoint["best"], checkpoint["stale"]  # Restaure la progression et l'arrêt anticipé.
        origin = checkpoint.get("origin", {})  # Conserve la filiation pendant les reprises de cette phase.
    (output / "config.json").write_text(json.dumps(config, indent=2))  # Conserve les réglages avec les poids.
    parameters = sum(p.numel() for p in model.parameters())  # Compte chaque poids partagé une seule fois.
    tokens_per_step = config["batch_size"] * config["context"] * config["accumulation"]  # Calcule le lot effectif.
    print(f"Paramètres : {parameters:,} ; tokens/mise à jour : {tokens_per_step:,} ; appareil : {device}", flush=True)  # Affiche la taille réelle.
    if start >= config["steps"]:  # Repère une reprise dont le budget est déjà terminé.
        print("Budget déjà atteint : pour continuer, créer une nouvelle phase avec --init-from et un nouveau dossier.", flush=True)  # Évite de faire croire que --resume ajoute automatiquement des étapes.
        return  # Ne lance aucune mise à jour hors du calendrier prévu.
    if init_from is not None:  # Établit un point de comparaison sur le nouveau corpus avant tout apprentissage.
        best = measure_loss(model, streams["val"], config, device, config["seed"] + 2)  # Mesure le modèle parent sur les nouvelles fenêtres réservées.
        reference_loss = best  # Conserve la référence encyclopédique distincte du score mélangé.
        literature_loss = None if literature is None else measure_loss(model, literature["val"], config, device, config["seed"] + 3)  # Mesure le livre réservé sans gradients.
        best = reference_loss if literature_loss is None else (1 - probability) * reference_loss + probability * literature_loss  # Pondère la validation comme le mélange d'entraînement.
        baseline = dict(step=0, val_loss=best, reference_val_loss=reference_loss, literature_val_loss=literature_loss, perplexity=math.exp(min(best, 80)), origin=origin)  # Décrit séparément les domaines et la référence combinée.
        (output / "baseline.json").write_text(json.dumps(baseline, indent=2))  # Garde une mesure avant/après comparable.
        initial = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), config=config, architecture=architecture, manifest=manifest, tokenizer=(root / "tokenizer.json").read_text(), step=0, best=best, stale=0, generator=generator.get_state(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if amp else [], origin=origin)  # Rend la phase reprenable dès son initialisation.
        atomic_save(initial, output / "last.pt")  # Sauvegarde l'état de départ avant la première mise à jour.
        atomic_save(initial, output / "best.pt")  # Ne remplace les poids parents que si la validation s'améliore réellement.
        print(json.dumps(baseline), flush=True)  # Affiche le point de départ mesuré.
    end = config["steps"] if stop_after is None else min(config["steps"], start + stop_after)  # Permet une pause sans modifier le calendrier global.
    for step in range(start + 1, end + 1):  # Répète les mises à jour restantes.
        if stale >= config["patience"]:  # Respecte un arrêt anticipé déjà atteint lors d'une reprise.
            print("Arrêt anticipé : la validation ne s'améliore plus. Examiner best.pt et les données avant une nouvelle phase.", flush=True)  # Rend la raison de l'arrêt visible.
            break  # Ne reprend pas une expérience jugée stagnante.
        clock = time.perf_counter()  # Démarre le chronomètre de la mise à jour.
        model.train()  # Active le dropout.
        optimizer.zero_grad(set_to_none=True)  # Libère les anciens gradients.
        progress = max(0.0, (step - config["warmup"]) / max(1, config["steps"] - config["warmup"]))  # Positionne la décroissance après warmup.
        factor = step / max(1, config["warmup"]) if step <= config["warmup"] else 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2  # Monte puis décroît jusqu'à 10 % du taux initial.
        for group in optimizer.param_groups:  # Applique le même taux aux deux groupes.
            group["lr"] = config["learning_rate"] * factor  # Actualise le taux avant la mise à jour.
        train_loss = 0.0  # Accumule la perte non divisée pour le journal.
        for _ in range(config["accumulation"]):  # Simule un plus gros lot sans augmenter la mémoire du micro-lot.
            selected = literature if literature is not None and torch.rand((), generator=generator).item() < probability else streams  # Tire le domaine avec la proportion configurée et un hasard sauvegardé.
            x, y = batch(selected["train"], config, generator, device)  # Tire un lot uniquement dans l'entraînement de ce domaine.
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):  # Utilise FP32 sur CPU, précision mixte sur GPU.
                _, loss = model(x, y)  # Calcule l'erreur de prédiction.
            if not torch.isfinite(loss):  # Détecte immédiatement une divergence numérique.
                raise FloatingPointError("Perte non finie : réduire le taux d'apprentissage.")  # Arrête avant d'écrire des poids invalides.
            train_loss += loss.item() / config["accumulation"]  # Moyenne les pertes des micro-lots.
            scaler.scale(loss / config["accumulation"]).backward()  # Accumule les gradients correctement normalisés.
        scaler.unscale_(optimizer)  # Rétablit les gradients réels avant de les borner.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)  # Limite les explosions de gradients.
        scaler.step(optimizer)  # Effectue une mise à jour des poids.
        scaler.update()  # Ajuste l'échelle FP16 si nécessaire.
        elapsed = time.perf_counter() - clock  # Mesure le temps hors validation et sauvegarde.
        usage = resources(output)  # Mesure la RAM et le disque après chaque mise à jour.
        reason = stop_reason(config, output, usage)  # Détecte une demande d'arrêt ou une réserve insuffisante.
        if step % config["eval_every"] == 0 or step == end or reason:  # Sauvegarde aussi immédiatement lors d'un arrêt demandé.
            reference_loss = None if reason else measure_loss(model, streams["val"], config, device, config["seed"] + 2)  # Évite une validation coûteuse lorsque les ressources manquent.
            literature_loss = None if reason or literature is None else measure_loss(model, literature["val"], config, device, config["seed"] + 3)  # Suit séparément les textes littéraires réservés.
            val_loss = reference_loss if literature_loss is None else (1 - probability) * reference_loss + probability * literature_loss  # Calcule le score correspondant au mélange.
            improved = val_loss is not None and val_loss < best  # N'élit jamais meilleur un checkpoint non évalué.
            best, stale = (val_loss, 0) if improved else (best, stale if reason else stale + 1)  # Ne consomme pas la patience pour un arrêt de ressources.
            report = dict(step=step, train_loss=train_loss, val_loss=val_loss, reference_val_loss=reference_loss, literature_val_loss=literature_loss, perplexity=None if val_loss is None else math.exp(min(val_loss, 80)), learning_rate=optimizer.param_groups[0]["lr"], tokens_seen=step * tokens_per_step, tokens_per_second=tokens_per_step / elapsed, resources=usage, stop_reason=reason)  # Journalise aussi la consommation réelle et la cause d'arrêt.
            report["total_tokens_seen"] = origin.get("tokens_before_phase", 0) + report["tokens_seen"]  # Distingue le budget de cette phase de l'exposition cumulée.
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:  # Conserve l'historique des validations.
                stream.write(json.dumps(report) + "\n")  # Écrit une ligne par mesure.
            print(json.dumps(report), flush=True)  # Rend les résultats visibles pendant l'exécution.
            payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), config=config, architecture=architecture, manifest=manifest, tokenizer=(root / "tokenizer.json").read_text(), step=step, best=best, stale=stale, generator=generator.get_state(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if amp else [])  # Sauvegarde tout ce qui est nécessaire à la reprise.
            payload["origin"] = origin  # Conserve la provenance des poids et le total des phases précédentes.
            atomic_save(payload, output / "last.pt")  # Garde la dernière progression validée.
            if improved:  # Conserve séparément le meilleur modèle.
                atomic_save(payload, output / "best.pt")  # Évite de perdre les meilleurs poids.
            if reason:  # Termine après la sauvegarde de la dernière mise à jour complète.
                (output / "STOP").unlink(missing_ok=True)  # Consomme la demande d'arrêt pour permettre une future reprise.
                print(f"{reason} Dernier état sauvegardé.", flush=True)  # Explique clairement pourquoi le calcul s'arrête.
                break  # Rend les ressources à la machine.
    print("Session terminée ; voir metrics.jsonl, last.pt et best.pt.", flush=True)  # Indique les artefacts à examiner.


if __name__ == "__main__":  # Expose la commande de préentraînement.
    parser = argparse.ArgumentParser(description=__doc__)  # Crée l'aide en français.
    parser.add_argument("--config", default="configs/cpu.json")  # Choisit un petit profil par défaut.
    parser.add_argument("--data", default="data/prepared")  # Localise les tokens préparés.
    parser.add_argument("--output", default="runs/cpu")  # Localise les checkpoints.
    parser.add_argument("--resume", action="store_true")  # Active la reprise exacte de configuration.
    parser.add_argument("--init-from")  # Commence une nouvelle phase à partir de poids existants avec un optimiseur neuf.
    parser.add_argument("--stop-after", type=int)  # Limite cette session en nombre de mises à jour.
    args = parser.parse_args()  # Lit les paramètres demandés.
    train(args.config, args.data, args.output, args.resume, args.stop_after, args.init_from)  # Lance l'apprentissage.
