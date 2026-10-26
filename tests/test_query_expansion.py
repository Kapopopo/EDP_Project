"""Vérifie l'expansion lexicale des requêtes."""

import unittest

from llm_fr.query_expansion import expand_query, strip_legal_citations
from llm_fr.query_understanding import analyze_query


class QueryExpansionTests(unittest.TestCase):  # Regroupe les tests d'expansion.
    def test_rin_expansion(self):  # Étend RIN vers déontologie avocat.
        analysis = analyze_query("rin avocat")  # Analyse la requête.
        expanded = expand_query(analysis)  # Enrichit pour le retrieval.
        self.assertIn("deontologie", expanded.casefold())  # Ajoute la déontologie.
        self.assertIn("rin", expanded.casefold())  # Conserve la requête initiale.

    def test_cassation_expansion(self):  # Étend cassation vers jurisprudence.
        analysis = analyze_query("cassation responsabilité avocat")  # Analyse la requête.
        expanded = expand_query(analysis)  # Enrichit pour le retrieval.
        self.assertIn("jurisprudence", expanded.casefold())  # Ajoute la jurisprudence.

    def test_strip_article_citation(self):  # Retire les renvois art. pour Judilibre.
        stripped = strip_legal_citations("jurisprudence responsabilité civile avocat faute art 1147")
        self.assertNotIn("1147", stripped)
        self.assertIn("faute", stripped)

    def test_no_expansion_for_generic_query(self):  # Ne modifie pas une requête générique.
        analysis = analyze_query("prescription civile")  # Analyse la requête.
        expanded = expand_query(analysis)  # Tente l'expansion.
        self.assertEqual(expanded, analysis.raw_query)  # Conserve la requête brute.


if __name__ == "__main__":  # Autorise le lancement autonome des tests.
    unittest.main()  # Exécute les tests déclarés.
