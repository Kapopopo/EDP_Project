#!/usr/bin/env python3
"""Évalue un fichier JSONL de questions (smoke ou critères stricts). Aucun test manuel."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fr.query_understanding import analyze_query
from llm_fr.rag_chat import default_index
from llm_fr.rag_index import RagIndex

# Réutilise le scoring golden
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from run_eval import load_cases, score_case

STRICT_CASE_KEYS = frozenset(
    {
        "expect_intents",
        "prefer_source",
        "prefer_any_of",
        "avoid_domains",
        "judilibre_should_not_dominate",
    }
)


def smoke_pass(hits, min_score=0.08):
    if not hits:
        return False, "aucun hit"
    top = hits[0].get("score")
    if top is None:
        return True, ""
    if top < min_score:
        return False, f"score top {top:.3f} < {min_score}"
    return True, ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=str(ROOT / "eval" / "question_bank_2000.jsonl"),
        help="Fichier JSONL (query + id + smoke ou expect_*)",
    )
    parser.add_argument("--min-rate", type=float, default=0.95, help="Taux min de réussite (défaut 95 %).")
    parser.add_argument("--smoke-min-score", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=0, help="Limiter le nombre de lignes (0 = tout).")
    parser.add_argument("--quiet", action="store_true", help="Résumé seulement.")
    parser.add_argument("--report", default="", help="Écrire les échecs en JSONL.")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.is_file():
        print(f"Fichier absent : {path}", file=sys.stderr)
        print("Générez-le : .venv/bin/python scripts/generate_question_bank_2000.py", file=sys.stderr)
        sys.exit(2)

    cases = load_cases(path)
    if args.limit:
        cases = cases[: args.limit]

    index_dir = default_index()
    if not args.quiet:
        print(f"Index : {index_dir} — {len(cases)} questions")
    t0 = time.perf_counter()
    index = RagIndex.load(index_dir)
    load_s = time.perf_counter() - t0

    ok = fail = 0
    failures = []
    for case in cases:
        q = case["query"]
        analysis = analyze_query(q)
        hits = index.search(q, top_k=5)
        strict = case.get("expect_profession") is not None or any(case.get(k) for k in STRICT_CASE_KEYS)
        if case.get("smoke") and not strict:
            passed, notes = smoke_pass(hits, args.smoke_min_score)
        else:
            passed, notes = score_case(case, hits, analysis)
        if passed:
            ok += 1
        else:
            fail += 1
            failures.append({"id": case.get("id"), "query": q, "notes": notes})
            if not args.quiet:
                print(f"[FAIL] {case.get('id')}: {q[:70]} → {', '.join(notes)}")

    rate = ok / len(cases) if cases else 0
    elapsed = time.perf_counter() - t0
    print(
        f"\nBatch {path.name}: {ok}/{len(cases)} ({rate:.1%}) "
        f"— chargement index {load_s:.1f}s, total {elapsed:.1f}s"
    )
    if args.report and failures:
        Path(args.report).write_text(
            "\n".join(json.dumps(f, ensure_ascii=False) for f in failures) + "\n",
            encoding="utf-8",
        )
        print(f"Échecs → {args.report}")

    if rate < args.min_rate:
        print(f"ÉCHEC : taux < {args.min_rate:.0%}")
        sys.exit(1)
    print("OK batch eval")


if __name__ == "__main__":
    main()
