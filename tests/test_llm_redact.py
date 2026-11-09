"""Tests de construction des prompts de rédaction et du choix du moteur."""

import io
import json
import os
import unittest
import urllib.error
from unittest import mock

from llm_fr import llm_redact
from llm_fr.llm_redact import (
    REDACT_SYSTEM,
    REDACT_SYSTEM_STRICT,
    append_redaction_to_synthesis,
    build_redaction_system,
    build_redaction_user_prompt,
    redact_answer,
    redaction_unavailable_hint,
)

HITS = [dict(text="Secret professionnel de l'avocat.", title="Art. 1", source="https://a", domain="legifrance")]
ENV_WITH_KEY = {
    "OPENAI_API_KEY": "sk-test-factice",
    "RAG_LLM_API_KEY": "",
    "RAG_LLM_BASE_URL": "",
    "RAG_OLLAMA_BASE_URL": "",
    "RAG_OLLAMA_MODEL": "",
}


class _FakeResponse(io.BytesIO):
    status = 200


def _fake_urlopen(calls, reply="Mémo de test [1]", fail=False):
    """Remplace urlopen : note chaque URL appelée, répond comme /chat/completions ou échoue."""

    def fake(req, timeout=None):
        calls.append(getattr(req, "full_url", req))
        if fail:
            raise urllib.error.URLError("connexion refusée")
        return _FakeResponse(json.dumps({"choices": [{"message": {"content": reply}}]}).encode("utf-8"))

    return fake


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


class RedactBackendTests(unittest.TestCase):
    """--rediger : Ollama local par défaut, API distante seulement sur demande explicite."""

    def _run(self, backend, fail=False, local_runner=None, env=None):
        calls = []
        with mock.patch.dict(os.environ, env if env is not None else ENV_WITH_KEY), mock.patch.object(
            llm_redact.urllib.request, "urlopen", _fake_urlopen(calls, fail=fail)
        ):
            result = redact_answer("secret avocat", HITS, backend=backend, local_runner=local_runner)
        return result, calls

    def test_auto_uses_ollama_never_openai_even_with_key(self):
        (text, label), calls = self._run("auto")
        self.assertEqual(label, "ollama")
        self.assertEqual(text, "Mémo de test [1]")
        self.assertEqual(calls, ["http://localhost:11434/v1/chat/completions"])
        self.assertFalse(any("openai.com" in url for url in calls))

    def test_auto_falls_back_to_local_checkpoint_when_ollama_down(self):
        (text, label), calls = self._run("auto", fail=True, local_runner=lambda: "Mémo checkpoint")
        self.assertEqual((text, label), ("Mémo checkpoint", "local"))
        self.assertFalse(any("openai.com" in url for url in calls))

    def test_auto_unavailable_without_ollama_nor_checkpoint(self):
        result, calls = self._run("auto", fail=True)
        self.assertEqual(result, (None, "unavailable"))
        self.assertFalse(any("openai.com" in url for url in calls))

    def test_api_only_when_explicit(self):
        (text, label), calls = self._run("api")
        self.assertEqual(label, "api")
        self.assertEqual(calls, ["https://api.openai.com/v1/chat/completions"])

    def test_api_without_key_does_not_fall_back(self):
        result, calls = self._run("api", env={"OPENAI_API_KEY": "", "RAG_LLM_API_KEY": ""})
        self.assertEqual(result, (None, "unavailable"))
        self.assertEqual(calls, [])

    def test_ollama_env_overrides(self):
        env = dict(ENV_WITH_KEY, RAG_OLLAMA_BASE_URL="http://127.0.0.1:9999/v1/", RAG_OLLAMA_MODEL="flint:test")
        (_, label), calls = self._run("ollama", env=env)
        self.assertEqual(label, "ollama")
        self.assertEqual(calls, ["http://127.0.0.1:9999/v1/chat/completions"])
        with mock.patch.dict(os.environ, env):
            self.assertEqual(llm_redact.ollama_model(), "flint:test")

    def test_ollama_budget_leaves_room_for_reasoning(self):
        bodies = []

        def fake(req, timeout=None):
            bodies.append(json.loads(req.data.decode("utf-8")))
            return _FakeResponse(json.dumps({"choices": [{"message": {"content": "Mémo"}}]}).encode("utf-8"))

        with mock.patch.dict(os.environ, dict(ENV_WITH_KEY, RAG_OLLAMA_MAX_TOKENS="")), mock.patch.object(
            llm_redact.urllib.request, "urlopen", fake
        ):
            redact_answer("q", HITS, backend="ollama")
        self.assertEqual(bodies[0]["max_tokens"], llm_redact.DEFAULT_OLLAMA_MAX_OUTPUT_TOKENS)
        with mock.patch.dict(os.environ, {"RAG_OLLAMA_MAX_TOKENS": "1234"}):
            self.assertEqual(llm_redact.ollama_max_tokens(), 1234)

    def test_off_and_unknown_backend(self):
        self.assertEqual(redact_answer("q", HITS, backend="off"), (None, "off"))
        with self.assertRaises(ValueError):
            redact_answer("q", HITS, backend="openai")

    def test_ollama_header_and_hints(self):
        text = append_redaction_to_synthesis("Extraits", "Mémo", "ollama")
        self.assertIn("Ollama local", text)
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "", "RAG_LLM_API_KEY": ""}):
            self.assertIn("ollama serve", redaction_unavailable_hint("auto"))
            self.assertIn("checkpoint local", redaction_unavailable_hint("auto"))
            self.assertNotIn("OPENAI", redaction_unavailable_hint("auto"))
            self.assertIn("OPENAI_API_KEY", redaction_unavailable_hint("api"))


if __name__ == "__main__":
    unittest.main()
