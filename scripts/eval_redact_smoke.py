#!/usr/bin/env python3
"""Smoke test : synthèse extractive + prompts de rédaction (citations strictes).

Sans clé API : vérifie les prompts et la synthèse enrichie sur hits figés (SKIP rédaction live).
Avec --live-redact et OPENAI_API_KEY / RAG_LLM_API_KEY : appelle l'API une fois par cas.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_fr.llm_redact import (
    build_redaction_system,
    build_redaction_user_prompt,
    llm_api_key,
    redact_via_api,
)
from llm_fr.synthesis_enriched import format_synthese_enriched

FIXTURE = ROOT / "eval" / "citation_smoke.jsonl"


def load_cases(path):
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def check_extractive(case):
    text = format_synthese_enriched(case["question"], case["hits"])
    notes = []
    ok = True
    for needle in case.get("extract_substrings") or []:
        if needle.lower() not in text.lower():
            ok = False
            notes.append(f"extrait manquant: {needle!r}")
    return ok, notes, text


def check_strict_prompt(case):
    system = build_redaction_system(strict_citations=True)
    user, selected = build_redaction_user_prompt(
        case["question"],
        case["hits"],
        strict_citations=True,
    )
    notes = []
    ok = True
    if "une affirmation" not in system.lower() and "UNE seule affirmation" not in system:
        ok = False
        notes.append("system strict incomplet")
    for label in case.get("strict_labels") or []:
        if label not in user:
            ok = False
            notes.append(f"label absent du user prompt: {label}")
    if len(selected) != len(case["hits"]):
        ok = False
        notes.append(f"hits sélectionnés {len(selected)} != {len(case['hits'])}")
    return ok, notes


def check_live_redact(case):
    if not llm_api_key():
        return None, ["SKIP live redact (pas de OPENAI_API_KEY / RAG_LLM_API_KEY)"]
    text = redact_via_api(case["question"], case["hits"], strict_citations=True)
    if not text:
        return False, ["API rédaction indisponible ou réponse vide"]
    notes = []
    ok = True
    if "[S" not in text:
        ok = False
        notes.append("aucune citation [Sx] dans la réponse API")
    return ok, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        type=Path,
        default=FIXTURE,
        help="Fichier JSONL (défaut : eval/citation_smoke.jsonl)",
    )
    parser.add_argument(
        "--live-redact",
        action="store_true",
        help="Appeler l'API de rédaction (nécessite une clé ; sinon SKIP documenté)",
    )
    args = parser.parse_args()
    if not args.cases.is_file():
        raise SystemExit(f"Fichier introuvable : {args.cases}")

    cases = load_cases(args.cases)
    passed = failed = skipped = 0
    print(f"Cas : {len(cases)} ({args.cases.name})", flush=True)

    for case in cases:
        cid = case.get("id", "?")
        ok_e, notes_e, _ = check_extractive(case)
        ok_p, notes_p = check_strict_prompt(case)
        ok = ok_e and ok_p
        notes = notes_e + notes_p

        if args.live_redact:
            live = check_live_redact(case)
            if live[0] is None:
                skipped += 1
                notes.extend(live[1])
            elif not live[0]:
                ok = False
                notes.extend(live[1])
            else:
                notes.append("live redact OK")

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        detail = "; ".join(notes) if notes else "ok"
        print(f"  [{status}] {cid} — {detail}", flush=True)

    print(f"\nRésumé : {passed} pass, {failed} fail, {skipped} skip (live)", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
