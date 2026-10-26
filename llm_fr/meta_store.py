"""Accès paresseux aux métadonnées meta.jsonl (évite de charger 5M+ JSON en RAM)."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np


def build_meta_line_offsets(meta_path):  # Calcule la position de chaque ligne non vide.
    path = Path(meta_path)
    offsets = []
    with path.open("rb") as stream:
        while True:
            pos = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
    return np.asarray(offsets, dtype=np.uint64)


def write_meta_jsonl(chunks, meta_path):  # Écrit meta.jsonl et meta_offsets.npy.
    path = Path(meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = []
    with path.open("wb") as stream:
        for chunk in chunks:
            offsets.append(stream.tell())
            line = json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n"
            stream.write(line)
    off_path = path.parent / "meta_offsets.npy"
    np.save(off_path, np.asarray(offsets, dtype=np.uint64))
    return path, off_path


class LazyMetaStore:
    """Métadonnées indexées par doc_id, texte lu à la demande."""

    def __init__(self, meta_path, offsets, cache_size=512):
        self._path = Path(meta_path)
        self._offsets = offsets
        self._n_docs = int(len(offsets))
        self._cache_size = max(16, cache_size)
        self._get = lru_cache(maxsize=self._cache_size)(self._load_doc)

    @classmethod
    def load(cls, index_dir, cache_size=512):
        root = Path(index_dir)
        meta_path = root / "meta.jsonl"
        if not meta_path.is_file():
            raise FileNotFoundError(f"meta.jsonl manquant dans {index_dir}")
        off_path = root / "meta_offsets.npy"
        if off_path.is_file():
            offsets = np.load(off_path, mmap_mode="r")
        else:
            offsets = build_meta_line_offsets(meta_path)
            np.save(off_path, np.asarray(offsets, dtype=np.uint64))
        return cls(meta_path, offsets, cache_size=cache_size)

    def __len__(self):
        return self._n_docs

    def __getitem__(self, doc_id):
        doc_id = int(doc_id)
        if doc_id < 0 or doc_id >= self._n_docs:
            raise IndexError(doc_id)
        return dict(self._get(doc_id))

    def _load_doc(self, doc_id):
        with self._path.open("rb") as stream:
            stream.seek(int(self._offsets[doc_id]))
            line = stream.readline()
        row = json.loads(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise ValueError(f"meta.jsonl doc_id={doc_id} : entrée invalide.")
        return row
