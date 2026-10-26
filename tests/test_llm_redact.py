"""Tests de construction des prompts de rédaction."""

import unittest

from llm_fr.llm_redact import (
    REDACT_SYSTEM,
    REDACT_SYSTEM_STRICT,
    build_redaction_system,
    build_redaction_user_prompt,
)


class LlmRedactTests(unittest.TestCase):
    def test_build_prompt_cites_indices(self):
        hits = [
            dict(text="Secret professionnel de l'avocat.", title="Art. 1", source="https://a", domain="legifrance"),
            dict(text="Responsabilité de l'avocat.", title="Arrêt", source="https://b", domain="judilibre"),
        ]
        prompt, selected = build_redaction_user_prompt("secret avocat", hits)
        self.assertIn("[1]", prompt)
        self.assertIn("[2]", prompt)
        self.assertEqual(len(selected), 2)

    def test_strict_system_differs(self):
        self.assertIs(build_redaction_system(False), REDACT_SYSTEM)
        self.assertIs(build_redaction_system(True), REDACT_SYSTEM_STRICT)
        self.assertIn("[S1]", REDACT_SYSTEM_STRICT)

    def test_strict_user_prompt_s_labels(self):
        hits = [
            dict(text="Texte A.", title="A", source="https://a", domain="legifrance"),
            dict(text="Texte B.", title="B", source="https://b", domain="judilibre"),
        ]
        prompt, _ = build_redaction_user_prompt("question", hits, strict_citations=True)
        self.assertIn("[S1]", prompt)
        self.assertIn("[S2]", prompt)
        self.assertNotIn("[1]", prompt.split("Rédige")[0])
        self.assertIn("Chaque puce", prompt)


if __name__ == "__main__":
    unittest.main()
