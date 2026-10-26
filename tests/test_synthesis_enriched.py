"""Tests de la synthèse enrichie."""

import tempfile
import unittest
from pathlib import Path

from llm_fr.synthesis_enriched import (
    dedupe_hits,
    format_synthese_enriched,
    source_count_line,
)
from llm_fr.synthesis_export import build_export_metadata_footer, export_synthese_markdown


class SynthesisEnrichedTests(unittest.TestCase):
    def test_dedupe_article_id(self):
        hits = [
            dict(text="a", source="u1", article_id="X", score=0.5, domain="legifrance", title="A"),
            dict(text="b", source="u2", article_id="X", score=0.9, domain="legifrance", title="A bis"),
        ]
        out = dedupe_hits(hits)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["score"], 0.9)

    def test_format_has_sections(self):
        hits = [
            dict(
                text="Le secret professionnel de l'avocat protège les échanges.",
                source="https://test",
                title="Art. 1",
                domain="legifrance",
                score=0.9,
            ),
            dict(
                text="Attendu que la responsabilité de l'avocat est engagée.",
                source="https://www.courdecassation.fr/decision/ECLI:TEST",
                title="Arrêt test",
                domain="judilibre",
                score=0.8,
            ),
            dict(
                text="La convention d'honoraires doit informer le client des modes de facturation.",
                source="cabinet:curated:honoraires",
                title="Honoraires",
                domain="cabinet_curated",
                score=0.85,
            ),
        ]
        text = format_synthese_enriched("secret professionnel avocat honoraires", hits, max_sentences=2, fallback_chars=400)
        self.assertIn("Faits juridiques", text)
        self.assertIn("Textes", text)
        self.assertIn("Jurisprudence", text)
        self.assertIn("Points d'attention", text)
        self.assertIn("Limites", text)
        self.assertIn("[Légifrance]", text)
        self.assertIn("[Judilibre]", text)
        self.assertIn("[Métier curé]", text)
        self.assertIn("Légifrance", source_count_line(hits))

    def test_export_metadata_footer(self):
        hits = [
            dict(domain="legifrance", text="x", source="s1"),
            dict(domain="cabinet_curated", text="y", source="s2"),
        ]
        footer = build_export_metadata_footer(
            "Question test ?",
            hits=hits,
            index_path="data/rag/index_legal",
        )
        self.assertIn("Question test", footer)
        self.assertIn("legifrance", footer)
        self.assertIn("cabinet_curated", footer)
        self.assertIn("index_legal", footer)

    def test_export_writes_footer(self):
        hits = [dict(domain="legifrance", text="t", source="s", title="T")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.md"
            export_synthese_markdown(
                path,
                "Ma question",
                "Corps synthèse",
                hits=hits,
                index_path="data/rag/index_legal",
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("Corps synthèse", content)
            self.assertIn("Métadonnées d'export", content)
            self.assertIn("Ma question", content)


if __name__ == "__main__":
    unittest.main()
