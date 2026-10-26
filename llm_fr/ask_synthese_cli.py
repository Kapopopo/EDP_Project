"""Question unique en synthèse (script / CI)."""

import argparse
import json
import sys

from .rag_chat import default_index, format_sources, synthese_reply
from .rag_index import RagIndex
from .retrieval_diagnostics import format_retrieval_diagnostics
from .synthesis_audit import append_audit_record
from .synthesis_export import export_synthese_markdown
from .synthesis_filters import normalize_domain_filter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="Question juridique")
    parser.add_argument("--domain", default="all", choices=("all", "legifrance", "judilibre", "cabinet_curated"))
    parser.add_argument("--export", dest="export_path", default=None)
    parser.add_argument("--sources", action="store_true")
    parser.add_argument("--json", action="store_true", help="Sortie JSON (texte + diagnostics)")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    index_dir = default_index()
    index = RagIndex.load(index_dir)
    domain_filter = normalize_domain_filter(args.domain)
    result = synthese_reply(index, question, domain_filter=domain_filter)
    if not args.no_audit:
        append_audit_record(question, result.get("hits"), index_path=index_dir)
    diag = format_retrieval_diagnostics(question, result.get("hits") or [], result.get("analysis"))
    if args.export_path:
        export_synthese_markdown(
            args.export_path,
            question,
            result["text"],
            hits=result.get("hits"),
            index_path=index_dir,
        )
    if args.json:
        payload = {
            "text": result["text"],
            "diagnostics": diag,
            "hits": [
                {
                    "domain": h.get("domain"),
                    "title": h.get("title"),
                    "score": h.get("score"),
                    "source": h.get("source"),
                }
                for h in (result.get("hits") or [])
            ],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(result["text"])
    if args.sources and result.get("hits"):
        print()
        print(format_sources(result["hits"]))
    print()
    print(diag)
    if args.export_path:
        print(f"\nExport : {args.export_path}")


if __name__ == "__main__":
    main()
