from fastapi.testclient import TestClient

from app.main import app
from app.summarize.config import settings

client = TestClient(app)


def test_ping():
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_summarize_empty():
    r = client.post("/summarize", json={"text": "   "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Text cannot be empty or whitespace"


def test_summarize_dummyllm():
    settings.llm_provider = "dummy"  # Ensure using DummyLLM for this test
    settings.openai_api_key = None

    payload = {
        "text": "Hello world, last word of this sentence should be truncated in DummyLLM.\nMore lines here.",
        "max_words": 11,
    }
    r = client.post("/summarize", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "Hello world, last word of this sentence should be truncated in"
    assert data["confidence"] == 0.85
