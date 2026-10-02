"""Vérifie séparation, causalité, apprentissage et reprise sur un corpus fictif."""

import json  # Fabrique un petit corpus de test.
import tempfile  # Isole les artefacts des tests.
import unittest  # Utilise le framework de test standard de Python.
from pathlib import Path  # Manipule les fichiers temporaires.
from llm_fr.corpus import clean, records, split_corpus  # Importe les fonctions sans dépendances lourdes.


class CorpusTests(unittest.TestCase):  # Vérifie les propriétés du nettoyage.
    def test_clean(self):  # Vérifie accents et paragraphes.
        self.assertEqual(clean("  e\u0301cole  française\r\n  Bonjour "), "école française\nBonjour")  # Compare la normalisation attendue.

    def test_split_and_duplicates(self):  # Vérifie l'absence de doublons entre ensembles.
        with tempfile.TemporaryDirectory() as folder:  # Supprime les fichiers à la fin du test.
            root = Path(folder)  # Crée un chemin pratique.
            rows = [dict(text=f"Document {i}. " + "La langue française conserve ses accents. " * 8, source=f"test:{i}", license="test", language="fr") for i in range(100)]  # Crée assez de documents distincts.
            source = root / "raw.jsonl"  # Localise le corpus artificiel.
            source.write_text("\n".join(json.dumps(r) for r in rows + rows), encoding="utf-8")  # Duplique intentionnellement chaque document.
            stats = split_corpus(source, root / "clean")  # Exécute la séparation réelle.
            train = {r["sha256"] for r in records(root / "clean/train.jsonl")}  # Lit les empreintes d'entraînement.
            val = {r["sha256"] for r in records(root / "clean/val.jsonl")}  # Lit les empreintes de validation.
            self.assertFalse(train & val)  # Aucun doublon exact ne doit traverser la séparation.
            self.assertEqual(stats["duplicates"], 100)  # Tous les doublons doivent être détectés.
            self.assertEqual(len(train | val), 100)  # Tous les documents uniques doivent être conservés.


class ModelTests(unittest.TestCase):  # Regroupe les vérifications nécessitant les dépendances ML.
    @classmethod  # Prépare les bibliothèques une fois pour la classe.
    def setUpClass(cls):  # Signale clairement les dépendances manquantes.
        import torch  # Échoue explicitement si PyTorch n'est pas installé.
        torch.set_num_threads(1)  # Accélère les très petits réseaux de test.

    def test_causality_and_learning(self):  # Vérifie que le réseau n'utilise pas le futur et peut apprendre.
        import torch  # Charge les opérations de tenseurs.
        from llm_fr.model import LanguageModel  # Charge l'architecture testée.
        torch.manual_seed(7)  # Rend le test reproductible.
        model = LanguageModel(32, 8, 16, 2, 1, 0.0)  # Crée un réseau miniature sans dropout.
        x = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Définit une séquence connue.
        changed = x.clone()  # Copie la séquence pour le test de causalité.
        changed[:, 4:] = 12  # Modifie seulement les positions futures.
        torch.testing.assert_close(model(x)[0][:, :4], model(changed)[0][:, :4])  # Le passé doit rester inchangé.
        target = (x + 1) % 32  # Définit un motif simple à mémoriser.
        initial = model(x, target)[1].item()  # Mesure l'erreur avant apprentissage.
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)  # Utilise un taux adapté au test court.
        for _ in range(30):  # Répète quelques mises à jour.
            optimizer.zero_grad()  # Efface les anciens gradients.
            model(x, target)[1].backward()  # Calcule les gradients de l'erreur.
            optimizer.step()  # Modifie réellement les poids.
        self.assertLess(model(x, target)[1].item(), initial / 2)  # Exige une baisse nette de la perte.

    def test_full_pipeline_and_resume(self):  # Compare une session continue à deux sessions reprises.
        import torch  # Lit et compare les checkpoints.
        from tokenizers import Tokenizer  # Vérifie les accents du tokenizer.
        from llm_fr.prepare import prepare  # Prépare le corpus réellement.
        from llm_fr.train import train  # Utilise la boucle d'entraînement réelle.
        from llm_fr.generate import generate  # Vérifie aussi la lecture des poids pour l'inférence.
        with tempfile.TemporaryDirectory() as folder:  # Isole la chaîne complète.
            root = Path(folder)  # Localise les sorties de test.
            source = root / "raw.jsonl"  # Définit le corpus de test.
            rows = [dict(text=f"Chapitre {i}. " + "Les élèves étudient la lumière et écrivent en français. " * 6, source=f"test:{i}", license="test", language="fr") for i in range(80)]  # Fournit un corpus artificiel, sans ambition linguistique.
            source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")  # Écrit les documents.
            prepare(source, root / "data", vocab_size=300, val_fraction=0.2)  # Entraîne le BPE et encode les deux ensembles.
            tokenizer = Tokenizer.from_file(str(root / "data/tokenizer.json"))  # Recharge le BPE sauvegardé.
            phrase = "Éléphant, cœur, où es-tu ?"  # Inclut des caractères français absents du corpus.
            self.assertEqual(tokenizer.decode(tokenizer.encode(phrase).ids), phrase)  # Exige la reconstruction exacte des accents.
            config = dict(seed=42, device="cpu", threads=1, context=8, width=16, heads=2, layers=1, dropout=0.1, batch_size=2, accumulation=2, steps=4, warmup=1, learning_rate=0.001, weight_decay=0.1, eval_every=2, eval_batches=2, patience=10)  # Configure un test rapide incluant le dropout.
            config_path = root / "config.json"  # Localise les réglages.
            config_path.write_text(json.dumps(config))  # Écrit une expérience identique pour les deux parcours.
            train(config_path, root / "data", root / "continuous")  # Effectue quatre mises à jour continues.
            train(config_path, root / "data", root / "resumed", stop_after=2)  # S'arrête après deux mises à jour sauvegardées.
            train(config_path, root / "data", root / "resumed", resume=True)  # Termine les deux mises à jour restantes.
            continuous = torch.load(root / "continuous/last.pt", weights_only=True)  # Charge le résultat continu.
            resumed = torch.load(root / "resumed/last.pt", weights_only=True)  # Charge le résultat repris.
            self.assertEqual(resumed["step"], 4)  # Vérifie le compteur final.
            for name in continuous["model"]:  # Compare tous les poids, pas seulement la perte.
                torch.testing.assert_close(continuous["model"][name], resumed["model"][name], rtol=0, atol=0)  # Exige une reprise identique sur CPU.
            self.assertIsInstance(generate(root / "resumed/best.pt", "Bonjour", count=3), str)  # Vérifie une inférence complète.
            from copy import deepcopy  # Prépare des variantes incompatibles sans modifier les références.
            from llm_fr.model import LanguageModel  # Vérifie le transfert exact des poids parents.
            from llm_fr.train import initialize_from  # Exerce les protections de changement de phase.
            from llm_fr.evaluate import evaluate  # Vérifie les comparaisons à fenêtres fixes.
            from llm_fr.corpus import digest  # Compare les fichiers du tokenizer octet pour octet.
            extra = [dict(text=f"Ajout {i}. " + "La géographie décrit les montagnes et les rivières. " * 6, source=f"extra:{i}", license="test", language="fr") for i in range(30)]  # Ajoute de nouveaux documents distincts.
            extended_source = root / "extended.jsonl"  # Localise le corpus élargi.
            extended_source.write_text("\n".join(json.dumps(r) for r in rows + extra), encoding="utf-8")  # Conserve aussi les anciens documents.
            prepare(extended_source, root / "extended", val_fraction=0.2, tokenizer_path=root / "data/tokenizer.json")  # Encode davantage de données sans changer le BPE.
            self.assertEqual(digest(root / "data/tokenizer.json"), digest(root / "extended/tokenizer.json"))  # Exige un tokenizer rigoureusement identique.
            old_val = {r["sha256"] for r in records(root / "data/val.jsonl")}  # Identifie les anciennes données réservées.
            new_train = {r["sha256"] for r in records(root / "extended/train.jsonl")}  # Identifie le nouvel entraînement.
            self.assertFalse(old_val & new_train)  # Vérifie qu'aucun ancien document de validation n'entre dans l'entraînement.
            manifest = json.loads((root / "extended/manifest.json").read_text())  # Charge les métadonnées élargies.
            model = LanguageModel(**continuous["architecture"])  # Crée un modèle neuf pour tester le transfert.
            origin = initialize_from(model, root / "continuous/last.pt", continuous["architecture"], manifest)  # Transfère les poids existants.
            for name, value in model.state_dict().items():  # Vérifie toutes les couches initialisées.
                torch.testing.assert_close(value, continuous["model"][name], rtol=0, atol=0)  # Aucun poids ne doit être réinitialisé.
            self.assertEqual(origin["tokens_before_phase"], 128)  # Vérifie la comptabilité des tokens déjà présentés.
            incompatible = deepcopy(manifest)  # Prépare un faux tokenizer de même taille.
            incompatible["sha256"]["tokenizer.json"] = "incompatible"  # Change son identité sans toucher aux dimensions.
            with self.assertRaisesRegex(ValueError, "Tokenizer différent"):  # Exige une erreur explicite.
                initialize_from(model, root / "continuous/last.pt", continuous["architecture"], incompatible)  # Refuse le transfert vers un autre vocabulaire.
            incompatible = dict(manifest, seed=999)  # Prépare une séparation susceptible de contaminer la validation.
            with self.assertRaisesRegex(ValueError, "seed et val_fraction"):  # Exige la conservation du protocole de séparation.
                initialize_from(model, root / "continuous/last.pt", continuous["architecture"], incompatible)  # Refuse de mélanger les ensembles.
            train(config_path, root / "extended", root / "phase2", init_from=root / "continuous/last.pt", stop_after=2)  # Démarre une phase avec nouvel optimiseur et référence initiale.
            self.assertTrue((root / "phase2/baseline.json").exists())  # Vérifie la mesure avant apprentissage.
            train(config_path, root / "extended", root / "phase2", resume=True)  # Vérifie la reprise de cette nouvelle phase.
            phase2 = torch.load(root / "phase2/last.pt", weights_only=True)  # Examine la progression reprise.
            self.assertEqual(phase2["origin"]["tokens_before_phase"], 128)  # La reprise doit conserver la provenance.
            self.assertEqual(phase2["step"], 4)  # Les étapes repartent de zéro dans chaque nouvelle phase.
            report = evaluate([root / "phase2/last.pt", root / "phase2/last.pt"], root / "extended", batches=2)  # Compare un checkpoint à lui-même.
            self.assertEqual(report["results"][0]["val_loss"], report["results"][1]["val_loss"])  # Les fenêtres et pertes doivent être identiques.
            mixed_config = dict(config, literature_data=str(root / "extended"), literature_probability=0.2)  # Exerce le mélange avec un second flux miniature.
            mixed_path = root / "mixed.json"  # Localise les réglages du mélange.
            mixed_path.write_text(json.dumps(mixed_config))  # Écrit une configuration indépendante.
            train(mixed_path, root / "data", root / "mixed", init_from=root / "continuous/last.pt")  # Entraîne le mélange sans interruption.
            train(mixed_path, root / "data", root / "mixed_resume", init_from=root / "continuous/last.pt", stop_after=2)  # Interrompt à une sauvegarde connue.
            train(mixed_path, root / "data", root / "mixed_resume", resume=True)  # Restaure aussi le hasard du choix du domaine.
            mixed = torch.load(root / "mixed/last.pt", weights_only=True)  # Charge le parcours continu.
            mixed_resume = torch.load(root / "mixed_resume/last.pt", weights_only=True)  # Charge le parcours repris.
            for name in mixed["model"]:  # Compare tous les poids après les deux parcours.
                torch.testing.assert_close(mixed["model"][name], mixed_resume["model"][name], rtol=0, atol=0)  # Le choix de domaine doit être reproductible après reprise.
            self.assertIn("literature", mixed["manifest"])  # Exige l'identité du second corpus dans la sauvegarde.
            stopped = root / "stopped"  # Prépare une expérience avec arrêt utilisateur.
            train(mixed_path, root / "data", stopped, init_from=root / "continuous/last.pt", stop_after=1)  # Crée une première sauvegarde valide.
            (stopped / "STOP").touch()  # Demande l'arrêt au prochain pas complet.
            train(mixed_path, root / "data", stopped, resume=True)  # Exerce la sauvegarde avant arrêt.
            checkpoint = torch.load(stopped / "last.pt", weights_only=True)  # Lit le dernier état conservé.
            self.assertEqual(checkpoint["step"], 2)  # Ne doit effectuer qu'un pas supplémentaire.
            self.assertFalse((stopped / "STOP").exists())  # La demande doit être consommée pour une reprise ultérieure.


if __name__ == "__main__":  # Permet de lancer ce fichier directement.
    unittest.main()  # Exécute toutes les vérifications.
