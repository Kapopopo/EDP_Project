"""Tests du corpus métier curé."""

import unittest

from llm_fr.legal_curated import classify_curated_pack, curated_priority, is_curated_hit


class LegalCuratedTests(unittest.TestCase):
    def test_avocat_decret(self):
        hit = dict(code="Décret n° 91-1197 du 27 novembre 1991 organisant la profession d'avocat", title="Article 24")
        self.assertEqual(classify_curated_pack(hit), "avocat_organique")
        self.assertTrue(is_curated_hit(hit))

    def test_notaire_deontologie(self):
        hit = dict(code="Décret n° 2023-1297 du 28 décembre 2023 relatif au code de déontologie des notaires", title="Article 1")
        self.assertEqual(classify_curated_pack(hit), "notaire_deontologie_2023")
        self.assertGreaterEqual(curated_priority("notaire_deontologie_2023"), 2)

    def test_non_curated(self):
        hit = dict(code="Arrêté du 1 janvier 2010 relatif aux pesticides", title="Article 2")
        self.assertIsNone(classify_curated_pack(hit))
        self.assertFalse(is_curated_hit(hit))


if __name__ == "__main__":
    unittest.main()
