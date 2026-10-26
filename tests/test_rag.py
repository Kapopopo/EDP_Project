"""Vérifie le découpage, l'index TF-IDF et l'assemblage des prompts RAG."""

import json  # Fabrique des corpus de test.
import tempfile  # Isole les artefacts des tests.
import unittest  # Organise les vérifications automatiques.
from pathlib import Path  # Manipule les fichiers temporaires.

from llm_fr.chunks import build_chunks, chunk_text, tokenize
from llm_fr.rag_chat import (
    build_rag_prompt,
    extractive_summary,
    format_sources,
    format_synthese,
    is_session_exit_command,
    synthese_reply,
)
from llm_fr.rag_index import RagIndex, build_index


class RagTests(unittest.TestCase):  # Regroupe les tests du pipeline RAG minimal.
    def test_tokenize_french(self):  # Vérifie la tokenisation des accents français.
        self.assertEqual(tokenize("Le secret professionnel de l'avocat"), ["le", "secret", "professionnel", "de", "l", "avocat"])  # Normalise la casse.

    def test_chunk_overlap(self):  # Vérifie le découpage avec chevauchement.
        text = "A" * 100 + " mot-clé juridique " + "B" * 100  # Texte plus long que la fenêtre.
        chunks = chunk_text(text, size=80, overlap=20)  # Découpe avec chevauchement.
        self.assertGreater(len(chunks), 1)  # Produit plusieurs fragments.
        self.assertTrue(all(len(chunk) <= 80 for chunk in chunks))  # Respecte la taille maximale.

    def test_build_chunks_and_index(self):  # Vérifie la chaîne découpage → index → recherche.
        with tempfile.TemporaryDirectory() as folder:  # Supprime les fichiers à la fin du test.
            root = Path(folder)  # Localise les sorties temporaires.
            rows = [  # Crée deux documents juridiques fictifs distincts.
                dict(
                    text="Le secret professionnel de l'avocat protège les échanges avec le client. " * 12,
                    source="test:secret",
                    title="Secret professionnel",
                    license="test",
                    language="fr",
                ),
                dict(
                    text="La prescription en matière civile fixe un délai de droit commun. " * 12,
                    source="test:prescription",
                    title="Prescription civile",
                    license="test",
                    language="fr",
                ),
            ]
            source = root / "corpus.jsonl"  # Fichier source du test.
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")  # Écrit le corpus.
            chunks_path = root / "chunks.jsonl"  # Fichier de fragments.
            stats = build_chunks([source], chunks_path, size=120, overlap=20)  # Découpe le corpus.
            self.assertGreater(stats["chunks"], 1)  # Produit plusieurs fragments.
            manifest = build_index(chunks_path, root / "index")  # Construit l'index TF-IDF.
            self.assertGreater(manifest["n_docs"], 1)  # Indexe tous les fragments.
            self.assertTrue(manifest.get("profile_cached"))  # Profils pré-calculés.
            self.assertEqual(manifest.get("postings_format"), "npz")  # Format mmap.
            index = RagIndex.load(root / "index")  # Recharge l'index depuis le disque.
            self.assertEqual(index.postings_mode, "mmap")  # Postings chargés en mmap.
            hits = index.search("secret professionnel avocat", top_k=2)  # Recherche un terme juridique.
            self.assertTrue(hits)  # Retourne au moins un résultat.
            self.assertIn("secret", hits[0]["text"].casefold())  # Le meilleur résultat est pertinent.
            prompt, selected = build_rag_prompt("Qu'est-ce que le secret professionnel ?", hits)  # Construit le prompt.
            self.assertIn("Question :", prompt)  # Contient la question utilisateur.
            self.assertTrue(selected)  # Conserve au moins une source.
            sources = format_sources(selected)  # Formate les citations.
            self.assertIn("[1]", sources)  # Numérote la première source.
            self.assertIn("test:secret", sources)  # Affiche la provenance.

    def test_extractive_summary(self):  # Vérifie la sélection de phrases pertinentes.
        text = "La prescription fixe un délai. Le secret professionnel protège les échanges. Un autre sujet sans lien."
        summary = extractive_summary(text, "secret professionnel avocat", max_sentences=2)  # Extrait les phrases utiles.
        self.assertIn("secret professionnel", summary.casefold())  # Conserve la phrase pertinente.
        self.assertNotIn("prescription", summary.casefold())  # Écarte les phrases hors sujet.

    def test_format_synthese(self):  # Vérifie le rendu documentaire sans génération.
        hits = [  # Simule un résultat de recherche.
            dict(
                text="Le secret professionnel de l'avocat protège les échanges avec le client.",
                source="test:secret",
                title="Secret professionnel",
                domain="legifrance",
                score=0.91,
            )
        ]
        output = format_synthese("Qu'est-ce que le secret professionnel ?", hits)  # Formate la synthèse.
        self.assertIn("Question :", output)  # Contient la question posée.
        self.assertIn("[Légifrance]", output)  # Affiche la provenance officielle.
        self.assertIn("secret professionnel", output.casefold())  # Reprend le contenu utile.

    def test_synthese_reply(self):  # Vérifie la chaîne recherche → synthèse sans modèle.
        with tempfile.TemporaryDirectory() as folder:  # Supprime les fichiers à la fin du test.
            root = Path(folder)  # Localise les sorties temporaires.
            rows = [dict(text="Le secret professionnel de l'avocat protège les échanges. " * 8, source="test:secret", title="Secret professionnel", license="test", language="fr")]
            source = root / "corpus.jsonl"  # Fichier source du test.
            source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")  # Écrit le corpus.
            chunks_path = root / "chunks.jsonl"  # Fichier de fragments.
            build_chunks([source], chunks_path, size=120, overlap=20)  # Découpe le corpus.
            build_index(chunks_path, root / "index")  # Construit l'index TF-IDF.
            index = RagIndex.load(root / "index")  # Recharge l'index depuis le disque.
            result = synthese_reply(index, "secret professionnel avocat", top_k=1)  # Produit une synthèse.
            self.assertIn("secret professionnel", result["text"].casefold())  # La réponse reste documentaire.
            self.assertEqual(len(result["sources"]), 1)  # Conserve la source trouvée.

    def test_session_exit_commands(self):
        self.assertTrue(is_session_exit_command("/quitter"))
        self.assertTrue(is_session_exit_command("quitter"))
        self.assertTrue(is_session_exit_command("EXIT"))
        self.assertFalse(is_session_exit_command("secret professionnel"))


if __name__ == "__main__":  # Autorise le lancement autonome des tests.
    unittest.main()  # Exécute les tests déclarés.
