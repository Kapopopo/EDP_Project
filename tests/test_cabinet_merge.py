"""Fusion conditionnelle de l'index cabinet."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from llm_fr.cabinet_side_index import (
    cabinet_merge_weight,
    merge_cabinet_candidates,
    should_skip_semantic_for_analysis,
)
from llm_fr.query_understanding import analyze_query
from llm_fr.rag_index import RagIndex, build_index


class CabinetMergeTests(unittest.TestCase):
    def test_skip_merge_jurisprudence_strong(self):
        analysis = analyze_query("jurisprudence responsabilité civile avocat faute art 1147")
        self.assertIn("jurisprudence_strong", analysis.intents)
        self.assertEqual(cabinet_merge_weight(analysis), 0.0)

    def test_boost_merge_mentions_honoraires(self):
        for query in (
            "mentions légales site internet avocat",
            "convention honoraires avocat information client",
            "règlement intérieur national avocat barreau obligations",
        ):
            analysis = analyze_query(query)
            self.assertGreater(cabinet_merge_weight(analysis), 1.2, msg=query)

    def test_skip_merge_secret_professionnel_avocat(self):
        analysis = analyze_query("secret professionnel avocat")
        self.assertEqual(cabinet_merge_weight(analysis), 0.0)
        self.assertTrue(should_skip_semantic_for_analysis(analysis))

    def test_merge_respects_zero_weight(self):
        main = [dict(chunk_id="m1", score=0.9, text="judilibre hit", domain="judilibre")]
        cabinet = MagicMock()
        cabinet.retrieve.return_value = [
            dict(chunk_id="c1", score=0.95, text="fiche cabinet", domain="cabinet_curated"),
        ]
        q = "jurisprudence responsabilité civile avocat"
        out = merge_cabinet_candidates(main, cabinet, q)
        self.assertIs(out, main)
        cabinet.retrieve.assert_not_called()

    def test_merge_fuses_when_boosted(self):
        main = [dict(chunk_id="m1", score=0.5, text="main")]
        cabinet = MagicMock()
        cabinet.retrieve.return_value = [dict(chunk_id="c1", score=0.8, text="cabinet fiche")]
        q = "mentions légales site internet avocat"
        out = merge_cabinet_candidates(main, cabinet, q)
        self.assertEqual(len(out), 2)
        self.assertTrue(cabinet.retrieve.called)

    def test_cabinet_index_not_loaded_recursively(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [
                dict(
                    text="Mentions légales et site internet de l'avocat. " * 8,
                    source="test:cabinet",
                    title="Mentions légales",
                    domain="cabinet_curated",
                    license="test",
                    language="fr",
                ),
            ]
            chunks = root / "chunks.jsonl"
            chunks.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            build_index(chunks, root / "index_cabinet")
            idx = RagIndex.load(root / "index_cabinet")
            self.assertIsNone(idx.cabinet_index)


if __name__ == "__main__":
    unittest.main()
