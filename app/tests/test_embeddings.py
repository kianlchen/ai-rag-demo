import numpy as np
import pytest


def test_embed_openai_import_failure(monkeypatch):
    """Simulate ImportError when openai package is not installed"""
    # Patch import system to simulate ImportError
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    import importlib
    import sys

    sys.modules.pop("app.rag.embeddings", None)
    emb = importlib.import_module("app.rag.embeddings")
    assert emb.OpenAI is None
    with pytest.raises(
        RuntimeError, match="openai package not available; install `openai` to use embed_openai"
    ):
        emb.embed_openai("test")
    sys.modules.pop("app.rag.embeddings", None)


def test_embed_openai_no_api_key(monkeypatch):
    """Simulate openai package installed but no API key set"""
    import app.rag.embeddings as emb

    # ensure OpenAI exists
    if emb.OpenAI is None:  # pragma: no cover
        monkeypatch.setattr(emb, "OpenAI", type("DummyOpenAI", (), {}))
    from app.summarize.config import settings

    settings.openai_api_key = None
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY not set; cannot use embed_openai"):
        emb.embed("test", backend="openai")


def test_embed_norm_zero_vector():
    """Test _l2_normalize returns zeros for a zero-norm vector"""
    import app.rag.embeddings as emb

    vec = np.zeros(5, dtype=np.float32)
    results = emb._l2_normalize(vec)
    assert np.all(results == 0.0)
    assert np.array_equal(results, vec)


def test_embed_choose_mock(monkeypatch):
    import app.rag.embeddings as emb

    monkeypatch.setattr(emb, "embed_mock", lambda text, dim=128: "mocked")
    results = emb.embed("test", backend="mock")
    assert results == "mocked"
