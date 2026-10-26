"""Journal optionnel des questions / hits (conformité cabinet)."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def audit_log_path():
    raw = os.environ.get("RAG_AUDIT_LOG", "").strip()
    return Path(raw) if raw else None


def append_audit_record(question, hits, index_path=None, extra=None):
    """Ajoute une ligne JSONL si RAG_AUDIT_LOG est défini."""
    path = audit_log_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "question": question.strip(),
        "n_hits": len(hits or []),
        "domains": {},
        "top_titles": [],
        "index": str(index_path) if index_path else None,
    }
    if hits:
        from .synthesis_export import count_sources_by_domain

        record["domains"] = count_sources_by_domain(hits)
        record["top_titles"] = [
            {"domain": h.get("domain"), "title": (h.get("title") or "")[:120], "score": h.get("score")}
            for h in hits[:5]
        ]
    if extra:
        record.update(extra)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return str(path.resolve())
