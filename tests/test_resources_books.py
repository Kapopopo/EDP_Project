"""Vérifie les frontières de livres, la mémoire bornée et les arrêts protégés."""

import json  # Écrit un corpus miniature de livres.
import tempfile  # Isole les fichiers des tests.
import unittest  # Organise les tests automatiques.
from pathlib import Path  # Manipule les chemins temporaires.
from llm_fr.books import body, chunks  # Teste le nettoyage des éditions et leur découpage.
from llm_fr.corpus import records, split_corpus  # Vérifie la séparation par ouvrage.
from llm_fr.runtime import training_lock, stop_reason  # Vérifie les protections des sessions longues.


class BooksResourcesTests(unittest.TestCase):  # Regroupe les nouvelles propriétés à protéger.
    def test_notices_and_language(self):  # Refuse d'apprendre les notices ou une traduction anglaise.
        text = "Language: French\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nBonjour.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicence"  # Simule une édition minimale.
        self.assertEqual(body(text), "Bonjour.")  # Exige l'exclusion des notices.
        with self.assertRaises(ValueError):  # Exige la vérification de la langue déclarée.
            body(text.replace("French", "English"))  # Simule une édition dans la mauvaise langue.
        with self.assertRaises(ValueError):  # Exige des limites de texte explicites.
            body("Un fichier dépourvu de marqueurs.")  # Refuse un format non reconnu.

    def test_chunk_bound(self):  # Vérifie que même un paragraphe géant ne crée pas un document géant.
        parts = list(chunks("é" * 30001, size=12000))  # Découpe un long paragraphe accentué.
        self.assertTrue(all(len(p) <= 12000 for p in parts))  # Borne la taille de tous les morceaux.
        self.assertEqual("".join(parts), "é" * 30001)  # Ne perd aucun caractère au découpage.

    def test_book_groups(self):  # Empêche un même livre de traverser entraînement et validation.
        with tempfile.TemporaryDirectory() as folder:  # Isole le corpus artificiel.
            root = Path(folder)  # Définit les chemins de test.
            rows = [dict(text=f"Livre {book}, extrait {part}. " + "Une phrase française suffisamment longue. " * 8, source=f"test:{book}", license="test", language="fr", split_group=f"livre:{book}") for book in range(80) for part in range(3)]  # Produit plusieurs fragments par livre.
            source = root / "books.jsonl"  # Définit l'entrée du séparateur.
            source.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")  # Écrit les documents artificiels.
            split_corpus(source, root / "prepared", val_fraction=0.2)  # Utilise la vraie séparation du pipeline.
            train = {r["split_group"] for r in records(root / "prepared/train.jsonl")}  # Identifie les ouvrages d'entraînement.
            val = {r["split_group"] for r in records(root / "prepared/val.jsonl")}  # Identifie les ouvrages réservés.
            self.assertFalse(train & val)  # Aucun livre ne doit être partagé.
            self.assertEqual(len(train | val), 80)  # Aucun ouvrage ne doit être perdu.

    def test_lock_and_resource_stop(self):  # Vérifie le verrou exclusif et les seuils de ressources.
        with tempfile.TemporaryDirectory() as folder:  # Isole le fichier de verrouillage.
            with training_lock(folder):  # Simule un premier entraînement actif.
                with self.assertRaises(RuntimeError):  # Un second entraînement doit être refusé.
                    with training_lock(folder):  # Essaie de prendre le même verrou.
                        self.fail("Le second verrou ne doit pas être acquis.")  # Signale une protection défectueuse.
            usage = dict(rss_mb=500, available_mb=2000, disk_free_mb=1000)  # Simule une machine avec assez de ressources.
            config = dict(max_rss_mb=1800, min_available_mb=600, min_disk_mb=400)  # Définit les garde-fous du profil long.
            self.assertIsNone(stop_reason(config, folder, usage))  # Autorise la situation normale.
            for key, value in (("rss_mb", 1900), ("available_mb", 500), ("disk_free_mb", 300)):  # Exerce chaque seuil séparément.
                self.assertIsNotNone(stop_reason(config, folder, dict(usage, **{key: value})))  # Chaque dépassement doit demander l'arrêt.
            (Path(folder) / "STOP").touch()  # Simule la commande d'arrêt utilisateur.
            self.assertIn("STOP", stop_reason(config, folder, usage))  # Vérifie la priorité de la demande utilisateur.


if __name__ == "__main__":  # Autorise l'exécution directe du fichier.
    unittest.main()  # Lance les tests de cette classe.
