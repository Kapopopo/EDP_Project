"""Index d'embeddings sur le sous-ensemble métier curé (recherche sémantique locale)."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .corpus import digest
from .legal_curated import classify_curated_pack

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_SUBDIR = "semantic_curated"
MAX_CHARS = 900  # Limite le texte encodé par fragment.


def _embedding_text(hit):  # Construit le passage à encoder.
    parts = [hit.get("title") or "", hit.get("code") or "", hit.get("text") or ""]
    blob = "\n".join(part.strip() for part in parts if part and str(part).strip())
    return blob[:MAX_CHARS]


def _load_encoder(model_name):  # Charge sentence-transformers si disponible.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers requis pour l'index sémantique : "
            "pip install -r requirements-rag.txt"
        ) from exc
    return SentenceTransformer(model_name)


def encode_texts(texts, model_name=DEFAULT_MODEL, batch_size=32, show_progress=True):  # Encode une liste de textes.
    model = _load_encoder(model_name)
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def _normalize_rows(matrix):  # Normalise L2 chaque vecteur (sécurité si modèle sans normalize).
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def build_semantic_curated_index(index_dir, model_name=DEFAULT_MODEL, batch_size=32, min_priority=1):  # Construit l'index sémantique.
    root = Path(index_dir)
    meta_path = root / "meta.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.jsonl introuvable dans {index_dir}")
    curated = []
    with meta_path.open(encoding="utf-8") as stream:
        for doc_id, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            pack = row.get("curated_pack") or classify_curated_pack(row)
            if not pack:
                continue
            from .legal_curated import curated_priority

            if curated_priority(pack) < min_priority:
                continue
            row = dict(row)
            row["doc_id"] = doc_id
            row["curated_pack"] = pack
            curated.append(row)
    if not curated:
        raise ValueError("Aucun fragment métier curé trouvé dans meta.jsonl.")
    texts = [_embedding_text(hit) for hit in curated]
    vectors = encode_texts(texts, model_name=model_name, batch_size=batch_size)
    vectors = _normalize_rows(vectors)
    out = root / SEMANTIC_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors)
    mapping_path = out / "doc_ids.json"
    mapping_path.write_text(json.dumps([hit["doc_id"] for hit in curated]), encoding="utf-8")
    meta_out = out / "meta.jsonl"
    with meta_out.open("w", encoding="utf-8") as stream:
        for hit in curated:
            stream.write(json.dumps(hit, ensure_ascii=False) + "\n")
    manifest = {
        "parent_index": str(root.resolve()),
        "model": model_name,
        "n_vectors": len(curated),
        "dim": int(vectors.shape[1]),
        "meta_digest": digest(meta_out),
        "min_priority": min_priority,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class SemanticCuratedIndex:
    def __init__(self, embeddings, doc_ids, meta, model_name):  # Index sémantique chargé.
        self.embeddings = embeddings
        self.doc_ids = doc_ids
        self.meta = meta
        self.model_name = model_name
        self._model = None

    @classmethod
    def load(cls, index_dir):  # Charge depuis semantic_curated/ ou renvoie None.
        root = Path(index_dir) / SEMANTIC_SUBDIR
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
        doc_ids = json.loads((root / "doc_ids.json").read_text(encoding="utf-8"))
        meta = []
        with (root / "meta.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    meta.append(json.loads(line))
        return cls(embeddings, doc_ids, meta, manifest.get("model", DEFAULT_MODEL))

    def _encode_query(self, query):  # Encode une requête utilisateur.
        if self._model is None:
            self._model = _load_encoder(self.model_name)
        vector = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        return np.asarray(vector, dtype=np.float32)

    def retrieve(self, query, top_k=25, main_meta=None):  # Recherche sémantique sur le corpus curé.
        if top_k < 1:
            return []
        query = (query or "").strip()
        if not query:
            return []
        q = self._encode_query(query)
        scores = self.embeddings @ q
        k = min(top_k, len(scores))
        if k == 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        hits = []
        for local_id in top_idx:
            score = float(scores[local_id])
            if not math.isfinite(score):
                continue
            row = dict(self.meta[int(local_id)])
            doc_id = int(self.doc_ids[int(local_id)])
            if main_meta is not None and 0 <= doc_id < len(main_meta):
                row = dict(main_meta[doc_id])
                row["curated_pack"] = row.get("curated_pack") or self.meta[int(local_id)].get("curated_pack")
            row["doc_id"] = doc_id
            row["semantic_score"] = round(score, 4)
            row["score"] = round(max(0.0, min(1.0, (score + 1.0) / 2.0)), 4)
            row["retrieval"] = "semantic"
            hits.append(row)
        return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/rag/index_legal")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-priority", type=int, default=2, help="1=inclut code civil/CPC ; 2=métier avocat/notaire uniquement")
    args = parser.parse_args()
    manifest = build_semantic_curated_index(
        args.index,
        model_name=args.model,
        batch_size=args.batch_size,
        min_priority=args.min_priority,
    )
    print(f"Sémantique curé : {manifest['n_vectors']} vecteurs ({manifest['dim']}d) → {args.index}/{SEMANTIC_SUBDIR}")
