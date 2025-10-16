from __future__ import annotations

import hashlib
from typing import List, Optional

import numpy as np

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from app.summarize.config import settings


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize to unit length; returns zeros if norm~=0 to avoid NaNs."""
    norm = np.linalg.norm(vec)
    if norm == 0.0 or not np.isfinite(norm):
        return np.zeros_like(vec, dtype=np.float32)
    return (vec / norm).astype(np.float32)


def embed_mock(text: str, dim: int = 128) -> List[float]:
    """
    Deterministic, offline embedding for tests/demo.
    - Uses SHA-256 of the text to produce a repeatable vector.
    - Same input -> same output across machines/CI.
    - No API calls; fast and reproducible.
    """
    # 32 bytes digest; repeat to reach dim
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = (digest * ((dim + len(digest) - 1) // len(digest)))[:dim]
    vec = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    return _l2_normalize(vec).tolist()


def embed_openai(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Real embedding via OpenAI, if OPENAI_API_KEY is set.
    - Normalizes to unit length so cosine similarity works reliably.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not available; install `openai` to use embed_openai")
    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot use embed_openai")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(input=text, model=model)
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    return _l2_normalize(vec).tolist()


def embed(text: str, *, backend: Optional[str] = None) -> List[float]:
    """
    Public entrypoint:
      - backend="openai" to force OpenAI
      - backend="mock" to force mock (for testing)
      - default: follows settings.llm_provider
    """
    chosen = (backend or settings.llm_provider or "").lower()
    if chosen == "openai":
        return embed_openai(text)
    if chosen == "mock":
        return embed_mock(text)
    return embed_mock(text)
