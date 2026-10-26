#!/usr/bin/env python3
"""Évaluation retrieval sur le golden set (index production)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fr.query_understanding import analyze_query
from llm_fr.rag_chat import default_index
from llm_fr.rag_index import RagIndex

FIXTURE_RANKING = ROOT / "tests" / "fixtures" / "ranking_golden_queries.json"
FIXTURE_CABINET = ROOT / "eval" / "cabinet_queries.jsonl"


def load_cases(path):
    if path.suffix == ".jsonl":
        cases = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cases.append(json.loads(line))
        return cases
    return json.loads(path.read_text(encoding="utf-8"))


def _domain(hit):
    return hit.get("domain") or ""


def _pack(hit):
    return hit.get("curated_pack") or ""


def score_case(case, hits, analysis):
    """Retourne (passed: bool, notes: list[str])."""
    notes = []
    passed = True
    top5 = hits[:5]

    for intent in case.get("expect_intents", []):
        if intent not in analysis.intents:
            passed = False
            notes.append(f"intent manquant: {intent}")

    if case.get("expect_profession") is not None:
        if analysis.profession != case["expect_profession"]:
            passed = False
            notes.append(f"profession={analysis.profession}")

    prefer = case.get("prefer_source")
    if prefer:
        count = sum(1 for h in top5 if _domain(h) == prefer)
        if count < 2:
            passed = False
            notes.append(f"<2 résultats {prefer} dans top5 ({count})")

    if case.get("judilibre_should_not_dominate"):
        jud = sum(1 for h in top5[:3] if _domain(h) == "judilibre")
        if jud >= 3:
            passed = False
            notes.append("Judilibre domine top3")

    avoid = case.get("avoid_domains") or []
    for domain in avoid:
        if any(_domain(h) == domain for h in top5[:2]):
            passed = False
            notes.append(f"domaine {domain} en top2")

    prefer_any = case.get("prefer_any_of") or []
    if prefer_any:
        ok = False
        for h in top5[:3]:
            d = _domain(h)
            p = _pack(h)
            if d in prefer_any or p in prefer_any:
                ok = True
                break
        if not ok:
            passed = False
            notes.append(f"aucun de {prefer_any} en top3")

    if not top5:
        passed = False
        notes.append("aucun hit")

    return passed, notes


def run_suite(name, cases, index):
    print(f"\n### Suite {name} ({len(cases)} requêtes)")
    n_ok = 0
    for case in cases:
        q = case["query"]
        analysis = analyze_query(q)
        hits = index.search(q, top_k=5)
        ok, notes = score_case(case, hits, analysis)
        n_ok += int(ok)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {case['id']}")
        print(f"  Q: {q}")
        for i, h in enumerate(hits[:3], 1):
            title = (h.get("title") or "")[:65]
            print(f"  {i}. [{_domain(h)}] score={h.get('score')} | {title}")
        if notes:
            print(f"  → {', '.join(notes)}")
        print()

    rate = n_ok / len(cases) if cases else 0
    print(f"Score {name} : {n_ok}/{len(cases)} ({rate:.0%})")
    return n_ok, len(cases)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("ranking", "cabinet", "all"), default="all")
    args = parser.parse_args()

    suites = []
    if args.suite in ("ranking", "all"):
        if not FIXTURE_RANKING.is_file():
            print(f"Fixture absente : {FIXTURE_RANKING}", file=sys.stderr)
            sys.exit(2)
        suites.append(("ranking", load_cases(FIXTURE_RANKING)))
    if args.suite in ("cabinet", "all"):
        if not FIXTURE_CABINET.is_file():
            print(f"Fixture absente : {FIXTURE_CABINET}", file=sys.stderr)
            sys.exit(2)
        suites.append(("cabinet", load_cases(FIXTURE_CABINET)))

    index_dir = default_index()
    print(f"Index : {index_dir}")
    index = RagIndex.load(index_dir)
    if index.cabinet_index:
        print(f"Index cabinet : {index.cabinet_index.n_docs} fiches")

    total_ok = total_n = 0
    for name, cases in suites:
        ok, n = run_suite(name, cases, index)
        total_ok += ok
        total_n += n

    rate = total_ok / total_n if total_n else 0
    print(f"\n=== Total : {total_ok}/{total_n} ({rate:.0%}) ===")
    if rate < 0.6:
        print("ÉCHEC : taux global < 60%")
        sys.exit(1)
    print("OK eval")


if __name__ == "__main__":
    main()
