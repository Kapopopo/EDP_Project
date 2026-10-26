"""Index TF-IDF local pour la recherche documentaire du RAG."""

import argparse  # Lit les chemins des fragments et de l'index.
import heapq
import json  # Sérialise le vocabulaire et les métadonnées.
import math  # Calcule les normes et les scores cosinus.
from collections import Counter, defaultdict  # Agrège les fréquences et les scores.
from pathlib import Path  # Manipule les fichiers d'index.

import numpy as np  # Stocke les vecteurs IDF et les normes documentaires.

from .chunks import tokenize  # Réutilise la tokenisation des fragments.
from .corpus import digest  # Empreinte les artefacts pour le manifeste.
from .document_profile import compute_profile, profile_to_dict  # Pré-calcule les profils.
from .legal_curated import classify_curated_pack  # Tag corpus métier prioritaire.
from .meta_store import LazyMetaStore, write_meta_jsonl  # Métadonnées mmap / lazy.


def term_freq(tokens):  # Calcule la fréquence normalisée d'un document.
    if not tokens:  # Refuse un document vide.
        return {}  # Aucun terme à indexer.
    counts = Counter(tokens)  # Compte les occurrences de chaque mot.
    total = sum(counts.values())  # Normalise par la longueur du document.
    return {term: count / total for term, count in counts.items()}  # Renvoie les poids relatifs.


def load_chunks(path):  # Charge tous les fragments d'un fichier JSONL.
    chunks = []  # Accumule les fragments lus.
    with Path(path).open(encoding="utf-8") as stream:  # Ouvre le fichier de fragments.
        for number, line in enumerate(stream, 1):  # Parcourt chaque ligne.
            if not line.strip():  # Ignore les lignes vides.
                continue  # Passe à la suivante.
            row = json.loads(line)  # Décode un fragment.
            if not isinstance(row, dict) or not row.get("text", "").strip():  # Vérifie le contenu minimal.
                raise ValueError(f"{path}:{number} : fragment invalide (texte manquant).")  # Signale l'erreur.
            chunks.append(row)  # Conserve le fragment.
    return chunks  # Renvoie la liste complète.


def _serialize_postings(postings, n_terms):  # Convertit l'index inversé en tableaux compacts.
    offsets = np.zeros(n_terms + 1, dtype=np.int64)  # Pointeurs de début par terme.
    doc_ids = []  # Identifiants de documents.
    weights = []  # Poids TF-IDF associés.
    pos = 0  # Position courante dans les tableaux plats.
    for term_id in range(n_terms):  # Parcourt chaque terme du vocabulaire.
        offsets[term_id] = pos  # Marque le début des postings du terme.
        for doc_id, weight in postings.get(term_id, []):  # Parcourt les documents indexés.
            doc_ids.append(doc_id)  # Ajoute l'identifiant document.
            weights.append(weight)  # Ajoute le poids TF-IDF.
            pos += 1  # Avance le curseur.
    offsets[n_terms] = pos  # Marque la fin de la dernière liste.
    return offsets, np.array(doc_ids, dtype=np.uint32), np.array(weights, dtype=np.float32)  # Renvoie les tableaux.


def build_index(chunks_path, output_dir, min_df=1):  # Construit un index TF-IDF à partir des fragments.
    chunks = load_chunks(chunks_path)  # Charge les fragments à indexer.
    if not chunks:  # Refuse un corpus vide.
        raise ValueError("Aucun fragment à indexer.")  # Demande un fichier de fragments valide.
    doc_tokens = [tokenize(chunk["text"]) for chunk in chunks]  # Tokenise chaque fragment.
    df = Counter()  # Compte les documents contenant chaque terme.
    for tokens in doc_tokens:  # Parcourt les documents tokenisés.
        df.update(set(tokens))  # Incrémente la fréquence documentaire.
    vocab = sorted(term for term, count in df.items() if count >= min_df)  # Retient le vocabulaire utile.
    if not vocab:  # Refuse un vocabulaire vide.
        raise ValueError("Vocabulaire vide après filtrage.")  # Signale un corpus trop pauvre.
    term_to_id = {term: index for index, term in enumerate(vocab)}  # Associe chaque mot à un identifiant.
    n_docs = len(chunks)  # Nombre total de fragments indexés.
    idf = np.array([math.log((n_docs + 1) / (df[term] + 1)) + 1.0 for term in vocab], dtype=np.float32)  # Poids inverse de fréquence.
    postings = defaultdict(list)  # Index inversé : terme → (doc_id, poids_tf).
    norms = np.zeros(n_docs, dtype=np.float32)  # Normes L2 des vecteurs TF-IDF.
    for doc_id, tokens in enumerate(doc_tokens):  # Parcourt chaque document.
        tf = term_freq(tokens)  # Calcule les fréquences relatives.
        weights = {}  # Stocke les poids TF-IDF du document.
        for term, value in tf.items():  # Parcourt les termes du document.
            if term not in term_to_id:  # Ignore les termes hors vocabulaire.
                continue  # Passe au terme suivant.
            term_id = term_to_id[term]  # Récupère l'identifiant du terme.
            weight = value * idf[term_id]  # Combine TF et IDF.
            weights[term_id] = weight  # Mémorise le poids du terme.
            postings[term_id].append((doc_id, weight))  # Ajoute l'entrée à l'index inversé.
        norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0  # Calcule la norme L2.
        norms[doc_id] = norm  # Enregistre la norme du document.
    output = Path(output_dir)  # Localise le dossier de sortie.
    output.mkdir(parents=True, exist_ok=True)  # Crée le dossier d'index.
    meta_rows = []  # Fragments enrichis pour meta.jsonl.
    for chunk in chunks:  # Parcourt chaque fragment.
        row = dict(chunk)  # Copie le fragment source.
        row["profile"] = profile_to_dict(compute_profile(row))  # Pré-calcule le profil thématique.
        pack = classify_curated_pack(row)  # Identifie le pack métier curé.
        if pack:
            row["curated_pack"] = pack  # Facilite retrieval hybride et reranking.
        meta_rows.append(row)
    meta_path = write_meta_jsonl(meta_rows, output / "meta.jsonl")[0]  # Écrit meta + offsets.
    offsets, doc_ids, weights = _serialize_postings(postings, len(vocab))  # Compacte l'index inversé.
    np.savez_compressed(output / "index.npz", idf=idf, norms=norms)  # Sauvegarde les tableaux numériques.
    np.savez_compressed(output / "postings.npz", offsets=offsets, doc_ids=doc_ids, weights=weights)  # Sauvegarde l'index inversé mmap.
    (output / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")  # Écrit le vocabulaire.
    manifest = dict(  # Décrit l'index pour la traçabilité.
        chunks_path=str(Path(chunks_path).resolve()),  # Source des fragments.
        chunks_digest=digest(chunks_path),  # Empreinte du fichier source.
        meta_digest=digest(meta_path),  # Empreinte des métadonnées.
        n_docs=n_docs,  # Nombre de fragments indexés.
        n_terms=len(vocab),  # Taille du vocabulaire.
        min_df=min_df,  # Seuil de fréquence documentaire.
        postings_format="npz",  # Format binaire mmap pour chargement rapide.
        profile_cached=True,  # Profils documentaires pré-calculés.
        meta_lazy=True,  # meta_offsets.npy pour lecture paresseuse.
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")  # Écrit le manifeste.
    return manifest  # Renvoie les statistiques de construction.


class RagIndex:  # Représente un index TF-IDF chargé en mémoire.
    def __init__(self, vocab, idf, norms, meta, postings=None, offsets=None, doc_ids=None, weights=None, semantic=None, cabinet_index=None):  # Initialise les structures de recherche.
        self.vocab = vocab  # Liste des termes indexés.
        self.idf = idf  # Poids IDF par terme.
        self.norms = norms  # Normes L2 des documents.
        self.meta = meta  # Métadonnées des fragments.
        self.term_to_id = {term: index for index, term in enumerate(vocab)}  # Mappe les mots vers leurs identifiants.
        self.postings = postings  # Index inversé dict (legacy JSON).
        self._offsets = offsets  # Pointeurs mmap par terme.
        self._doc_ids = doc_ids  # Tableau mmap des doc_id.
        self._weights = weights  # Tableau mmap des poids TF-IDF.
        self.semantic = semantic  # Index sémantique optionnel (corpus curé).
        self.cabinet_index = cabinet_index  # Mini-index RIN / mentions légales.

    @property
    def postings_mode(self):  # Indique le mode de stockage des postings.
        return "mmap" if self._offsets is not None else "dict"  # mmap ou dict legacy.

    @property
    def n_docs(self):  # Nombre de fragments indexés.
        return int(len(self.norms))

    @classmethod
    def load(cls, index_dir):  # Charge un index depuis le disque.
        root = Path(index_dir)  # Localise le dossier d'index.
        required = ("vocab.json", "index.npz", "meta.jsonl", "manifest.json")  # Fichiers obligatoires.
        for name in required:  # Vérifie les fichiers requis.
            if not (root / name).is_file():  # Refuse un index incomplet.
                raise FileNotFoundError(f"Index incomplet : {root / name} manquant.")  # Signale le fichier absent.
        if not (root / "postings.npz").is_file() and not (root / "postings.json").is_file():  # Postings requis.
            raise FileNotFoundError(f"Index incomplet : postings.npz ou postings.json manquant dans {root}.")  # Signale l'absence.
        vocab = json.loads((root / "vocab.json").read_text(encoding="utf-8"))  # Charge le vocabulaire.
        arrays = np.load(root / "index.npz", mmap_mode="r")  # Charge les tableaux numériques en mmap.
        postings = None  # Index inversé dict (legacy).
        offsets = doc_ids = weights = None  # Tableaux mmap.
        if (root / "postings.npz").is_file():  # Format binaire compact.
            posting_arrays = np.load(root / "postings.npz", mmap_mode="r")  # Charge les postings en mmap.
            offsets = posting_arrays["offsets"]  # Pointeurs par terme.
            doc_ids = posting_arrays["doc_ids"]  # Identifiants document.
            weights = posting_arrays["weights"]  # Poids TF-IDF.
        else:  # Fallback legacy JSON.
            raw_postings = json.loads((root / "postings.json").read_text(encoding="utf-8"))  # Charge l'index inversé.
            postings = {int(term_id): [(doc_id, weight) for doc_id, weight in entries] for term_id, entries in raw_postings.items()}  # Reconvertit les clés.
        meta = LazyMetaStore.load(root)  # Métadonnées lues à la demande (pas 5M JSON en RAM).
        from .semantic_index import SemanticCuratedIndex  # Import tardif (dépendance optionnelle).

        semantic = SemanticCuratedIndex.load(root)  # Charge l'index sémantique s'il existe.
        from .cabinet_side_index import load_cabinet_index

        cabinet = None
        if root.name != "index_cabinet":
            cabinet = load_cabinet_index()  # Corpus métier cabinet (léger).
        return cls(
            vocab,
            arrays["idf"],
            arrays["norms"],
            meta,
            postings=postings,
            offsets=offsets,
            doc_ids=doc_ids,
            weights=weights,
            semantic=semantic,
            cabinet_index=cabinet,
        )

    def _iter_postings(self, term_id):  # Itère les (doc_id, poids) d'un terme.
        if self._offsets is not None:  # Mode mmap.
            start = int(self._offsets[term_id])  # Début des postings.
            end = int(self._offsets[term_id + 1])  # Fin des postings.
            for doc_id, weight in zip(self._doc_ids[start:end], self._weights[start:end]):  # Parcourt les entrées.
                yield int(doc_id), float(weight)  # Renvoie doc_id et poids.
            return  # Fin du mode mmap.
        for doc_id, weight in self.postings.get(term_id, []):  # Mode dict legacy.
            yield doc_id, weight  # Renvoie doc_id et poids.

    def retrieve(self, query, top_k=5):  # Retourne les candidats TF-IDF bruts sans reranking.
        if top_k < 1:  # Refuse un nombre de résultats invalide.
            raise ValueError("top_k doit être positif.")  # Explique le paramètre.
        tokens = tokenize(query)  # Tokenise la requête.
        if not tokens:  # Refuse une requête vide.
            return []  # Aucun résultat possible.
        q_tf = term_freq(tokens)  # Calcule les fréquences de la requête.
        q_weights = {}  # Stocke les poids TF-IDF de la requête.
        for term, value in q_tf.items():  # Parcourt les termes de la requête.
            term_id = self.term_to_id.get(term)  # Cherche le terme dans le vocabulaire.
            if term_id is None:  # Ignore les mots hors vocabulaire.
                continue  # Passe au terme suivant.
            q_weights[term_id] = value * self.idf[term_id]  # Applique le poids IDF.
        q_norm = math.sqrt(sum(value * value for value in q_weights.values())) or 1.0  # Norme L2 de la requête.
        ranked = self._rank_tfidf_scores(q_weights, q_norm, top_k)  # Top-k sans trier tout le corpus.
        results = []  # Construit les résultats enrichis.
        for doc_id, score in ranked:  # Parcourt les meilleurs documents.
            row = self.meta[doc_id]  # Métadonnées (copie si store paresseux).
            row["score"] = round(score, 4)  # Ajoute le score de pertinence.
            results.append(row)  # Conserve le résultat.
        return results  # Renvoie les fragments classés.

    def _rank_tfidf_scores(self, q_weights, q_norm, top_k):  # Sélectionne les meilleurs doc_id (cosinus).
        if not q_weights:
            return []
        if self._offsets is not None and len(self.norms) > 50_000:  # Gros index : accumulation vectorisée.
            scores = np.zeros(len(self.norms), dtype=np.float32)
            for term_id, q_weight in q_weights.items():
                start = int(self._offsets[term_id])
                end = int(self._offsets[term_id + 1])
                if start >= end:
                    continue
                slice_ids = self._doc_ids[start:end]
                slice_w = self._weights[start:end]
                np.add.at(scores, slice_ids, q_weight * slice_w)
            k = min(top_k, len(scores))
            if k <= 0 or not np.any(scores):
                return []
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            norms = self.norms
            return [(int(doc_id), float(scores[doc_id] / (norms[doc_id] * q_norm))) for doc_id in top_idx]
        scores = defaultdict(float)  # Petits index ou postings dict legacy.
        for term_id, q_weight in q_weights.items():
            for doc_id, d_weight in self._iter_postings(term_id):
                scores[doc_id] += q_weight * d_weight
        if not scores:
            return []
        norms = self.norms
        selected = heapq.nlargest(
            min(top_k, len(scores)),
            scores.items(),
            key=lambda item: item[1] / (norms[item[0]] * q_norm),
        )
        return [(doc_id, raw / (norms[doc_id] * q_norm)) for doc_id, raw in selected]

    def search(self, query, top_k=5, debug=False, domain_filter=None):  # Recherche avec reranking contextuel.
        from .hybrid_retrieval import merge_retrieval_candidates  # Fusion TF-IDF + sémantique.
        from .query_expansion import expand_query, judilibre_retrieval_query, strip_legal_citations  # Enrichit la requête pour le retrieval.
        from .query_understanding import analyze_query  # Analyse la requête.
        from .ranking_config import SEMANTIC_POOL_SIZE, SEMANTIC_RETRIEVAL_WEIGHT
        from .rerank import candidate_pool_size, rerank_hits  # Applique le reranking.

        analysis = analyze_query(query)  # Comprend la requête avant classement.
        if analysis.is_vague:  # Requête trop imprécise pour chercher.
            return []  # Évite les faux positifs lexicaux (ex. « stp » → acronyme STP).
        retrieval_query = expand_query(analysis)  # Étend la requête pour le TF-IDF.
        if "jurisprudence_strong" in analysis.intents:
            retrieval_query = strip_legal_citations(retrieval_query) or retrieval_query
        pool_size = candidate_pool_size(top_k)  # Élargit le pool de candidats.
        tfidf_hits = self.retrieve(retrieval_query or analysis.search_query, top_k=pool_size)  # Candidats lexicaux.
        candidates = tfidf_hits
        if "jurisprudence_strong" in analysis.intents:  # Second passage ciblé Cassation / Judilibre.
            from .hybrid_retrieval import reciprocal_rank_fusion

            focus_q = f"cour de cassation arrêt pourvoi {judilibre_retrieval_query(analysis)}"
            jud_hits = [
                h
                for h in self.retrieve(focus_q, top_k=min(80, pool_size + 20))
                if h.get("domain") == "judilibre"
            ]
            if jud_hits:
                for row in jud_hits:
                    row["retrieval"] = "judilibre_focus"
                candidates = reciprocal_rank_fusion([candidates, jud_hits], weights=[1.0, 0.9])
        from .cabinet_side_index import merge_cabinet_candidates, should_skip_semantic_for_analysis

        skip_semantic = should_skip_semantic_for_analysis(analysis)
        if self.semantic is not None and not skip_semantic:
            semantic_query = analysis.search_query or analysis.raw_query
            semantic_hits = self.semantic.retrieve(semantic_query, top_k=SEMANTIC_POOL_SIZE, main_meta=self.meta)
            candidates = merge_retrieval_candidates(tfidf_hits, semantic_hits, semantic_weight=SEMANTIC_RETRIEVAL_WEIGHT)
        if self.cabinet_index is not None:
            candidates = merge_cabinet_candidates(
                candidates,
                self.cabinet_index,
                retrieval_query or analysis.search_query or query,
                analysis=analysis,
            )
        if debug:  # Mode explicabilité développeur.
            for row in candidates:  # Parcourt les candidats.
                row.setdefault("ranking_debug", {})  # Initialise le debug.
                row["ranking_debug"]["retrieval_query"] = retrieval_query  # Conserve la requête enrichie.
                row["ranking_debug"]["hybrid"] = self.semantic is not None  # Indique le mode hybride.
        hits = rerank_hits(candidates, analysis=analysis, top_k=top_k, debug=debug)
        if domain_filter:
            from .synthesis_filters import filter_hits_by_domain

            hits = filter_hits_by_domain(hits, domain_filter)[:top_k]
        return hits


if __name__ == "__main__":  # Expose python -m llm_fr.rag_index.
    parser = argparse.ArgumentParser(description=__doc__)  # Prépare l'aide intégrée.
    parser.add_argument("--chunks", default="data/rag/chunks.jsonl")  # Fichier de fragments source.
    parser.add_argument("--output", default="data/rag/index")  # Dossier de sortie de l'index.
    parser.add_argument("--min-df", type=int, default=1)  # Fréquence documentaire minimale.
    args = parser.parse_args()  # Lit les options du terminal.
    manifest = build_index(args.chunks, args.output, args.min_df)  # Construit l'index.
    print(f"Index : {manifest['n_docs']} fragments, {manifest['n_terms']} termes → {args.output}")  # Affiche le bilan.
