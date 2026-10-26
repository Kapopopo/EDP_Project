"""Filtres de domaine pour la synthèse."""

import unittest

from llm_fr.synthesis_filters import filter_hits_by_domain, normalize_domain_filter


class SynthesisFiltersTests(unittest.TestCase):
    def test_normalize(self):
        self.assertIsNone(normalize_domain_filter(None))
        self.assertIsNone(normalize_domain_filter("all"))
        self.assertEqual(normalize_domain_filter("judilibre"), "judilibre")

    def test_filter(self):
        hits = [
            {"domain": "legifrance", "text": "a"},
            {"domain": "cabinet_curated", "text": "b"},
        ]
        out = filter_hits_by_domain(hits, "cabinet_curated")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["domain"], "cabinet_curated")


if __name__ == "__main__":
    unittest.main()
