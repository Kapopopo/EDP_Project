"""Vérifie l'analyse déterministe des requêtes."""

import unittest

from llm_fr.query_understanding import analyze_query, prepare_query


class QueryUnderstandingTests(unittest.TestCase):  # Regroupe les tests d'analyse de requête.
    def test_avocat_jurisprudence(self):  # Détecte métier et intention jurisprudentielle.
        analysis = analyze_query("jurisprudence responsabilité avocat")  # Analyse la requête.
        self.assertEqual(analysis.profession, "avocat")  # Métier avocat.
        self.assertGreaterEqual(analysis.profession_confidence, 0.75)  # Confiance élevée.
        self.assertIn("jurisprudence_strong", analysis.intents)  # Intention jurisprudentielle.

    def test_cassation_strong_intent(self):  # Détecte une intention jurisprudentielle forte.
        analysis = analyze_query("arrêt Cour de cassation responsabilité professionnelle avocat")  # Analyse la requête.
        self.assertIn("jurisprudence_strong", analysis.intents)  # Intention forte.
        self.assertEqual(analysis.profession, "avocat")  # Métier avocat.

    def test_deontologie_without_jurisprudence(self):  # Détecte déontologie sans jurisprudence forte.
        analysis = analyze_query("obligations déontologiques avocat conflit d'intérêts")  # Analyse la requête.
        self.assertIn("professional_rules", analysis.intents)  # Intention déontologique.
        self.assertNotIn("jurisprudence_strong", analysis.intents)  # Pas de jurisprudence forte.

    def test_mentions_legales_no_strong_jurisprudence(self):  # Mentions légales sans jurisprudence.
        analysis = analyze_query("mentions légales site internet avocat")  # Analyse la requête.
        self.assertEqual(analysis.profession, "avocat")  # Métier avocat.
        self.assertNotIn("jurisprudence_strong", analysis.intents)  # Pas de jurisprudence forte.

    def test_medical_domain_active(self):  # Active le domaine santé quand pertinent.
        analysis = analyze_query("secret médical médecin obligations déontologiques")  # Analyse la requête.
        self.assertIn("health", analysis.active_domains)  # Domaine santé actif.
        self.assertEqual(analysis.profession, "medecin")  # Métier médecin.

    def test_defense_domain_active(self):  # Active le domaine défense quand pertinent.
        analysis = analyze_query("obligations liées au secret de la défense nationale")  # Analyse la requête.
        self.assertIn("defense", analysis.active_domains)  # Domaine défense actif.

    def test_general_low_confidence(self):  # Requête générale sans métier fort.
        analysis = analyze_query("obligations professionnelles")  # Analyse la requête.
        self.assertIsNone(analysis.profession)  # Pas de métier détecté.
        self.assertIn("professional_rules", analysis.intents)  # Intention déontologique générale.

    def test_notaire_profession(self):  # Détecte le métier notaire.
        analysis = analyze_query("obligations déontologiques notaire chambre des notaires")  # Analyse la requête.
        self.assertEqual(analysis.profession, "notaire")  # Métier notaire.
        self.assertGreaterEqual(analysis.profession_confidence, 0.75)  # Confiance élevée.

    def test_stp_filler_removed(self):  # Retire « stp » avant la recherche.
        prepared = prepare_query("question de loi stp")  # Prépare la requête.
        self.assertNotIn("stp", prepared.search_query)  # Ne conserve pas le filler.
        self.assertTrue(prepared.is_vague)  # Reste trop vague pour chercher.

    def test_specific_query_not_vague(self):  # Conserve une requête juridique précise.
        prepared = prepare_query("secret professionnel avocat stp")  # Prépare la requête.
        self.assertFalse(prepared.is_vague)  # Requête exploitable.
        self.assertIn("secret", prepared.search_query)  # Conserve le fond juridique.


if __name__ == "__main__":  # Autorise le lancement autonome des tests.
    unittest.main()  # Exécute les tests déclarés.
