"""Vérifie le reranking contextuel et les tests de classement."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from llm_fr.chunks import build_chunks
from llm_fr.document_profile import profile_document, profile_from_dict, profile_to_dict
from llm_fr.query_understanding import analyze_query
from llm_fr.rag_index import RagIndex, build_index
from llm_fr.rerank import compute_adjustments, rerank_hits


class RerankUnitTests(unittest.TestCase):  # Tests unitaires du reranking synthétique.
    def _hit(self, **kwargs):  # Fabrique un fragment de test.
        base = dict(
            text="Texte de test.",
            source="test:source",
            title="Titre test",
            license="test",
            language="fr",
            score=0.50,
        )
        base.update(kwargs)  # Applique les surcharges.
        return base  # Renvoie le fragment.

    def test_penalize_health_when_irrelevant(self):  # Pénalise la santé hors contexte avocat.
        analysis = analyze_query("obligations déontologiques avocat")  # Requête avocat.
        hit = self._hit(
            domain="legifrance",
            title="Secret médical",
            text="Le secret médical couvre tout ce qui est parvenu à la connaissance du médecin.",
            score=0.60,
        )
        profile = profile_document(hit)  # Profile le fragment.
        final, adjustments = compute_adjustments(hit, analysis, profile)  # Calcule les ajustements.
        self.assertLess(final, hit["score"])  # Score final inférieur au score de base.
        self.assertTrue(any(name == "irrelevant_health_domain" for name, _ in adjustments))  # Pénalité santé.

    def test_no_health_penalty_when_relevant(self):  # Ne pénalise pas la santé si pertinente.
        analysis = analyze_query("secret médical médecin")  # Requête médicale.
        hit = self._hit(
            domain="legifrance",
            title="Secret médical",
            text="Le secret médical couvre tout ce qui est parvenu à la connaissance du médecin.",
            score=0.60,
        )
        profile = profile_document(hit)  # Profile le fragment.
        final, adjustments = compute_adjustments(hit, analysis, profile)  # Calcule les ajustements.
        self.assertFalse(any(name == "irrelevant_health_domain" for name, _ in adjustments))  # Pas de pénalité.

    def test_boost_judilibre_on_jurisprudence(self):  # Booste Judilibre sur intention jurisprudentielle.
        analysis = analyze_query("jurisprudence responsabilité avocat")  # Requête jurisprudentielle.
        hit = self._hit(domain="judilibre", title="Arrêt Cour de cassation", text="Attendu que la responsabilité de l'avocat...", score=0.40)
        profile = profile_document(hit)  # Profile le fragment.
        final, adjustments = compute_adjustments(hit, analysis, profile)  # Calcule les ajustements.
        self.assertGreater(final, hit["score"])  # Score amélioré.
        self.assertTrue(any("judilibre" in name for name, _ in adjustments))  # Boost Judilibre.

    def test_no_strong_judilibre_boost_for_mentions_legales(self):  # Pas de gros boost Judilibre pour mentions légales.
        analysis = analyze_query("mentions légales site internet avocat")  # Requête non jurisprudentielle.
        hit = self._hit(domain="judilibre", title="Arrêt Cour de cassation", text="Attendu que l'avocat...", score=0.40)
        profile = profile_document(hit)  # Profile le fragment.
        final, adjustments = compute_adjustments(hit, analysis, profile)  # Calcule les ajustements.
        boost = sum(delta for name, delta in adjustments if "jurisprudence" in name)  # Somme des boosts jurisprudence.
        self.assertLessEqual(boost, 0.05)  # Boost jurisprudentiel faible ou nul.

    def test_boost_avocat_professional_rules(self):  # Favorise les textes déontologiques avocat.
        analysis = analyze_query("obligations déontologiques avocat conflit d'intérêts")  # Requête déontologique.
        hit = self._hit(
            domain="legifrance",
            code="Décret n° 91-1197 du 27 novembre 1991 organisant la profession d'avocat",
            title="Article 1 — Décret organisant la profession d'avocat",
            text="Le collège de déontologie et le règlement intérieur national encadrent la profession d'avocat.",
            score=0.35,
        )
        profile = profile_document(hit)  # Profile le fragment.
        final, adjustments = compute_adjustments(hit, analysis, profile)  # Calcule les ajustements.
        self.assertGreater(final, hit["score"])  # Score amélioré.
        self.assertTrue(profile.is_professional_rules_avocat)  # Document métier avocat.

    def test_boost_notaire_professional_rules(self):  # Favorise les textes profession notaire.
        analysis = analyze_query("obligations déontologiques notaire chambre des notaires")  # Requête notaire.
        hit = self._hit(
            domain="legifrance",
            title="Chambre des notaires — profession de notaire",
            text="La chambre des notaires encadre la profession de notaire et ses obligations déontologiques.",
            score=0.35,
        )
        profile = profile_document(hit)  # Profile le fragment.
        final, adjustments = compute_adjustments(hit, analysis, profile)  # Calcule les ajustements.
        self.assertGreater(final, hit["score"])  # Score amélioré.
        self.assertTrue(profile.is_professional_rules_notaire)  # Document métier notaire.
        self.assertTrue(any("notaire" in name for name, _ in adjustments))  # Boost notaire appliqué.

    def test_profile_cache_roundtrip(self):  # Vérifie la sérialisation des profils.
        hit = self._hit(
            domain="legifrance",
            title="Chambre des notaires",
            text="La chambre des notaires encadre la profession de notaire.",
        )
        cached = profile_to_dict(profile_document(hit))  # Sérialise le profil.
        restored = profile_from_dict(cached)  # Recharge le profil.
        self.assertTrue(restored.is_professional_rules_notaire)  # Conserve le marqueur notaire.

    def test_rerank_ordering(self):  # Vérifie le reclassement global.
        analysis = analyze_query("jurisprudence responsabilité avocat")  # Requête jurisprudentielle.
        hits = [
            self._hit(domain="legifrance", title="Arrêté défense", text="secret de la défense nationale secret médical secret professionnel", score=0.70),
            self._hit(domain="judilibre", title="Arrêt Cour de cassation", text="Responsabilité de l'avocat", score=0.45),
        ]
        ranked = rerank_hits(hits, analysis=analysis, top_k=2)  # Rerank les candidats.
        self.assertEqual(ranked[0]["domain"], "judilibre")  # Judilibre remonte en tête.


class RankingGoldenTests(unittest.TestCase):  # Tests de classement sur index miniature.
    @classmethod
    def setUpClass(cls):  # Construit un index de test une seule fois.
        cls.fixture = json.loads(Path("tests/fixtures/ranking_golden_queries.json").read_text(encoding="utf-8"))  # Charge le golden set.
        cls._tmpdir = tempfile.mkdtemp()  # Dossier temporaire persistant pour la classe.
        root = Path(cls._tmpdir)  # Dossier temporaire.
        rows = [  # Corpus miniature multi-sources.
            dict(text="Le secret professionnel de l'avocat est protégé. " * 8, source="test:secret", title="Secret professionnel avocat", license="test", language="fr", domain="legifrance", code="Code général"),
            dict(text="Secret médical du médecin et secret de la défense nationale listés ensemble. " * 6, source="test:list", title="Liste de secrets", license="test", language="fr", domain="legifrance"),
            dict(text="Attendu que la responsabilité de l'avocat devant la Cour de cassation est engagée. " * 6, source="https://www.courdecassation.fr/decision/test1", title="Arrêt Cour de cassation — avocat", license="test", language="fr", domain="judilibre"),
            dict(text="Le décret n° 91-1197 organisant la profession d'avocat et le collège de déontologie encadrent les conflits d'intérêts. " * 4, source="test:rin", title="Décret profession d'avocat", license="test", language="fr", domain="legifrance", code="Décret n° 91-1197 du 27 novembre 1991 organisant la profession d'avocat"),
        ]
        source = root / "corpus.jsonl"  # Corpus source.
        source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")  # Écrit le corpus.
        chunks = root / "chunks.jsonl"  # Fragments.
        build_chunks([source], chunks, size=200, overlap=20)  # Découpe le corpus.
        build_index(chunks, root / "index")  # Construit l'index.
        cls.index = RagIndex.load(root / "index")  # Charge l'index.

    @classmethod
    def tearDownClass(cls):  # Nettoie l'index temporaire.
        shutil.rmtree(cls._tmpdir, ignore_errors=True)  # Supprime le dossier de test.

    def test_golden_queries_analysis(self):  # Vérifie l'analyse attendue du golden set.
        for case in self.fixture:  # Parcourt chaque requête de référence.
            analysis = analyze_query(case["query"])  # Analyse la requête.
            if case.get("expect_profession") is not None:  # Métier attendu.
                self.assertEqual(analysis.profession, case["expect_profession"], case["id"])  # Vérifie le métier.
            else:  # Pas de métier attendu.
                self.assertIsNone(analysis.profession, case["id"])  # Confirme l'absence de métier.
            for intent in case.get("expect_intents", []):  # Intentions attendues.
                self.assertIn(intent, analysis.intents, case["id"])  # Vérifie l'intention.
            for domain in case.get("active_domains", []):  # Domaines actifs attendus.
                self.assertIn(domain, analysis.active_domains, case["id"])  # Vérifie le domaine.

    def test_secret_professionnel_avocat_penalizes_secret_list(self):  # Cas réel amélioré sur mini-index.
        hits = self.index.search("secret professionnel avocat", top_k=3)  # Recherche rerankée.
        titles = [hit.get("title", "") for hit in hits]  # Titres classés.
        self.assertNotEqual(titles[0], "Liste de secrets")  # La liste multi-secrets ne doit pas être en tête.
        self.assertIn(titles[0], {"Secret professionnel avocat", "Arrêt Cour de cassation — avocat", "Décret profession d'avocat"})  # Source pertinente en tête.


if __name__ == "__main__":  # Autorise le lancement autonome des tests.
    unittest.main()  # Exécute les tests déclarés.
