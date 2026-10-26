#!/usr/bin/env python3
"""Smoke test de performance (chargement index + recherche). Exit 1 si seuils dépassés."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fr.rag_chat import default_index
from llm_fr.rag_index import RagIndex

DEFAULT_QUERY = "jurisprudence responsabilité avocat faute art 1147"
MAX_LOAD_S = 30.0
MAX_SEARCH_S = 2.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=None)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-load", type=float, default=MAX_LOAD_S)
    parser.add_argument("--max-search", type=float, default=MAX_SEARCH_S)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    index_dir = args.index or default_index()

    t0 = time.perf_counter()
    index = RagIndex.load(index_dir)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    hits = index.search(args.query, top_k=args.top_k)
    search_s = time.perf_counter() - t1

    print(f"Index : {index_dir} ({index.n_docs} fragments)")
    print(f"load={load_s:.2f}s (seuil {args.max_load}s)")
    print(f"search={search_s:.3f}s (seuil {args.max_search}s) hits={len(hits)}")

    ok = True
    if load_s > args.max_load:
        print("ÉCHEC : chargement trop lent")
        ok = False
    if search_s > args.max_search:
        print("ÉCHEC : recherche trop lente")
        ok = False
    if not ok:
        sys.exit(1)
    print("OK perf smoke")


if __name__ == "__main__":
    main()
