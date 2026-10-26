"""Export des synthèses RAG (Markdown)."""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def count_sources_by_domain(hits):
    """Compte les fragments par domaine RAG (clés brutes du corpus)."""
    if not hits:
        return {}
    return dict(Counter((h.get("domain") or "unknown") for h in hits))


def build_export_metadata_footer(question, hits=None, index_path=None, timestamp=None):
    """Pied de page Markdown : question, horodatage, effectifs, indice d'index."""
    ts = timestamp or datetime.now(timezone.utc).astimezone()
    ts_label = ts.isoformat(timespec="seconds")
    lines = [
        "## Métadonnées d'export",
        "",
        f"- **Question :** {question.strip()}",
        f"- **Horodatage :** {ts_label}",
    ]
    if hits is not None:
        total = len(hits)
        by_domain = count_sources_by_domain(hits)
        if by_domain:
            parts = ", ".join(f"{domain} : {count}" for domain, count in sorted(by_domain.items()))
            lines.append(f"- **Sources ({total}) :** {parts}")
        else:
            lines.append(f"- **Sources :** {total}")
    if index_path:
        lines.append(f"- **Index RAG (indication) :** `{index_path}`")
    lines.append("")
    lines.append(
        "*Document généré en mode synthèse extractive (EDP) — relecture professionnelle obligatoire.*"
    )
    return "\n".join(lines)


def export_synthese_markdown(path, question, text, hits=None, index_path=None, timestamp=None):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    header = f"<!-- Généré le {datetime.now().isoformat(timespec='seconds')} -->\n\n"
    footer = build_export_metadata_footer(
        question, hits=hits, index_path=index_path, timestamp=timestamp
    )
    body = f"{header}{text}\n\n---\n\n{footer}\n"
    target.write_text(body, encoding="utf-8")
    return str(target.resolve())
