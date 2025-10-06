import uuid
from typing import Dict, List


class DocumentStore:
    def __init__(self):
        self.docs: Dict[str, str] = {}

    def add(self, text: str) -> str:
        doc_id = str(uuid.uuid4())
        self.docs[doc_id] = text
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


STORE = DocumentStore()
