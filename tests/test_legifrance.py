"""Vérifie la conversion LEGI et le filtrage des articles en vigueur."""

import json  # Lit les documents convertis.
import tempfile  # Isole les artefacts des tests.
import unittest  # Organise les vérifications automatiques.
from pathlib import Path  # Manipule les fichiers temporaires.

from llm_fr.download_legifrance import convert_archive, in_force, parse_article


class LegifranceTests(unittest.TestCase):  # Regroupe les tests Légifrance.
    SAMPLE = Path("data/legal/raw/legi_sample/sample.tar.gz")  # Archive LEGI miniature locale.

    def test_in_force(self):  # Vérifie le filtre de vigueur juridique.
        self.assertTrue(in_force("VIGUEUR", "2999-01-01"))  # Conserve une version actuelle.
        self.assertFalse(in_force("ABROGE", "2999-01-01"))  # Exclut un article abrogé.
        self.assertFalse(in_force("MODIFIE", "2018-03-01"))  # Exclut une version expirée.

    def test_convert_sample_archive(self):  # Vérifie la conversion d'une archive LEGI réelle.
        self.assertTrue(self.SAMPLE.is_file(), "Archive LEGI de test absente.")  # Exige l'échantillon local.
        with tempfile.TemporaryDirectory() as folder:  # Supprime les fichiers à la fin du test.
            output = Path(folder) / "legifrance.jsonl"  # Corpus temporaire.
            stats = convert_archive(self.SAMPLE, output)  # Convertit l'archive miniature.
            self.assertGreater(stats["kept"], 10)  # Exige plusieurs articles convertis.
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])  # Lit le premier document.
            self.assertEqual(row["domain"], "legifrance")  # Marque la provenance officielle.
            self.assertIn("legifrance.gouv.fr", row["source"])  # Conserve le lien officiel.
            self.assertIn("Article", row["title"])  # Formate un titre juridique lisible.


if __name__ == "__main__":  # Autorise le lancement autonome des tests.
    unittest.main()  # Exécute les tests déclarés.
