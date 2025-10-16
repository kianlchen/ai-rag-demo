from __future__ import annotations

import uuid
from typing import Dict, List, Tuple

import numpy as np

from .embeddings import embed


class DocumentStore:
    def __init__(self):
        self.docs: Dict[str, str] = {}
        self.vecs: Dict[str, List[float]] = {}

    def clear(self):
        self.docs.clear()
        self.vecs.clear()

    def add(self, text: str) -> str:
        doc_id = str(uuid.uuid4())
        self.docs[doc_id] = text
        self.vecs[doc_id] = embed(text)
        return doc_id

    def query(self, keyword: str, limit: int = 3) -> List[Dict[str, str]]:
        """Return top-N documents ranked by a simple keyword score.
        Ranking priority:
          1) Higher frequency of the keyword (descending)
          2) Earlier first occurrence position (ascending)
          3) Shorter document length (ascending)
        """
        k = (keyword or "").lower().strip()
        if not k:
            return []

        scored = []
        for doc_id, text in self.docs.items():
            tl = text.lower()
            if k in tl:
                freq = tl.count(k)
                first_pos = tl.find(k)
                scored.append((freq, first_pos, len(text), doc_id, text))

        # Sort by: frequency (desc), first occurrence (asc), length (asc)
        scored.sort(key=lambda s: (-s[0], s[1], s[2]))

        return [{"id": doc_id, "text": text} for _, _, _, doc_id, text in scored[:limit]]

    def query_vector(self, query_text: str, limit: int = 3) -> List[Dict[str, str]]:
        """
        Semantic search:
          - Embed the query
          - Compute cosine similarity with stored vectors (dot product; vectors are normalized)
          - Return top-N docs with their scores
        """
        qv = np.array(embed(query_text), dtype=np.float32)
        if qv.size == 0 or not np.isfinite(qv).all():
            return []

        scored: List[Tuple[float, str, str]] = []  # (score, id, text)
        for doc_id, dv in self.vecs.items():
            dv_arr = np.array(dv, dtype=np.float32)
            if dv_arr.size != qv.size:
                # dimension mismatch shouldn't happen if all via same backend
                continue
            score = float(qv @ dv_arr)  # cosine similarity because vectors are normalized
            scored.append((score, doc_id, self.docs[doc_id]))

        scored.sort(key=lambda s: -s[0])  # highest score first
        return [
            {"id": doc_id, "text": text, "score": round(score, 6)}
            for score, doc_id, text in scored[: max(0, limit)]
        ]


STORE = DocumentStore()
