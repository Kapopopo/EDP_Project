"""Vérifie la conversion Judilibre et le format des décisions."""

import gzip  # Lit l'échantillon compressé.
import json  # Manipule les documents convertis.
import tempfile  # Isole les artefacts des tests.
import unittest  # Organise les vérifications automatiques.
from pathlib import Path  # Manipule les fichiers temporaires.

from llm_fr.download_judilibre import build_title, convert_archive, parse_decision


class JudilibreTests(unittest.TestCase):  # Regroupe les tests Judilibre.
    SAMPLE = Path("data/legal/raw/judilibre_sample/sample.jsonl.gz")  # Export miniature local.

    def test_build_title(self):  # Vérifie le titre d'une décision.
        row = dict(
            type="Arrêt",
            jurisdiction="Cour de cassation",
            chamber="chambre civile 1",
            decision_date="2020-01-15",
            number="19-12.345",
            solution="Rejet",
        )
        title = build_title(row)  # Formate le titre.
        self.assertIn("Cour de cassation", title)  # Conserve la juridiction.
        self.assertIn("19-12.345", title)  # Conserve le numéro.

    def test_parse_decision(self):  # Vérifie la conversion d'une décision brute.
        row = dict(
            id="abc123",
            type="Arrêt",
            jurisdiction="Cour de cassation",
            chamber="chambre commerciale",
            decision_date="2021-06-01",
            number="20-10.111",
            solution="Cassation",
            summary="Le contrat est nul pour vice du consentement.",
            text="Attendu que le consentement a été vicié par une erreur substantielle. Par ces motifs, la cour casse et annule le jugement attaqué.",
        )
        document = parse_decision(row)  # Convertit la décision.
        self.assertIsNotNone(document)  # Produit un document.
        self.assertEqual(document["domain"], "judilibre")  # Marque la provenance.
        self.assertIn("courdecassation.fr", document["source"])  # Conserve le lien public.
        self.assertIn("vice du consentement", document["text"].casefold())  # Conserve le résumé.

    def test_convert_sample_archive(self):  # Vérifie la conversion d'un export miniature.
        self.assertTrue(self.SAMPLE.is_file(), "Échantillon Judilibre de test absent.")  # Exige l'échantillon local.
        with tempfile.TemporaryDirectory() as folder:  # Supprime les fichiers à la fin du test.
            output = Path(folder) / "judilibre.jsonl"  # Corpus temporaire.
            stats = convert_archive(self.SAMPLE, output)  # Convertit l'échantillon.
            self.assertGreater(stats["kept"], 1)  # Exige plusieurs décisions converties.
            row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])  # Lit le premier document.
            self.assertEqual(row["domain"], "judilibre")  # Marque la provenance officielle.
            self.assertIn("courdecassation.fr", row["source"])  # Conserve le lien public.


if __name__ == "__main__":  # Autorise le lancement autonome des tests.
    unittest.main()  # Exécute les tests déclarés.
