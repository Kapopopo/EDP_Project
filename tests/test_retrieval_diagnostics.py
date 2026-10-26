import unittest

from llm_fr.retrieval_diagnostics import confidence_label, format_retrieval_diagnostics
from llm_fr.synthesis_audit import append_audit_record, audit_log_path


class RetrievalDiagnosticsTests(unittest.TestCase):
    def test_confidence(self):
        self.assertEqual(confidence_label([{"score": 0.8}]), "élevée")
        self.assertEqual(confidence_label([{"score": 0.4}]), "moyenne")
        self.assertEqual(confidence_label([]), "inconnue")

    def test_format_line(self):
        hits = [{"domain": "judilibre", "score": 0.7}]
        line = format_retrieval_diagnostics("test", hits, None)
        self.assertIn("judilibre=1", line)
        self.assertIn("élevée", line)

    def test_audit_disabled_without_env(self):
        self.assertIsNone(audit_log_path())
        self.assertIsNone(append_audit_record("q", [], index_path="data/rag/index_legal"))


if __name__ == "__main__":
    unittest.main()
