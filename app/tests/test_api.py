from fastapi.testclient import TestClient
from app.main import app
from app import llm as llm_module

client = TestClient(app)

class _TestLLM(llm_module.DummyLLM):
    def summarize(self, text: str, max_words: int = 80, strict: bool = False):
        # deterministic summary for testing
        return ("Test summary", 0.9)
    
def test_ping():
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

def test_summarize_ok(monkeypatch):
    # Force deterministic LLM for testing
    monkeypatch.setattr(llm_module, "get_llm", lambda: _TestLLM())
    payload = {"text": "Hello world.\nMore lines here." , "max_words": 10}
    r = client.post("/summarize", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["summary"] == "Test summary"
    assert 0.0 <= data["confidence"] <= 1.0

def test_summarize_empty():
    r = client.post("/summarize", json={"text": "   "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Text cannot be empty or whitespace"