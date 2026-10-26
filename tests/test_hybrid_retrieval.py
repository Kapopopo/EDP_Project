"""Tests de fusion hybride TF-IDF / sémantique."""

import unittest

from llm_fr.hybrid_retrieval import merge_retrieval_candidates, reciprocal_rank_fusion


class HybridRetrievalTests(unittest.TestCase):
    def test_rrf_dedupes(self):
        a = [dict(chunk_id="x", score=0.9, text="a"), dict(chunk_id="y", score=0.5, text="b")]
        b = [dict(chunk_id="y", score=0.8, text="b"), dict(chunk_id="z", score=0.4, text="c")]
        merged = reciprocal_rank_fusion([a, b])
        ids = [hit["chunk_id"] for hit in merged]
        self.assertEqual(ids[0], "y")
        self.assertEqual(len(ids), 3)

    def test_merge_prefers_both_channels(self):
        tfidf = [dict(chunk_id="1", score=0.3)]
        semantic = [dict(chunk_id="2", score=0.7, semantic_score=0.9)]
        merged = merge_retrieval_candidates(tfidf, semantic, semantic_weight=0.6)
        self.assertEqual(len(merged), 2)
        self.assertTrue(any(hit["chunk_id"] == "2" for hit in merged))


if __name__ == "__main__":
    unittest.main()
