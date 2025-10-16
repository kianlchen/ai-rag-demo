from fastapi.testclient import TestClient

from app.main import app
from app.rag.store import STORE

client = TestClient(app)


def test_rag_add_query():
    # add two docs
    r1 = client.post("/rag/add", json={"text": "Embedding power semantic search"})
    r2 = client.post("/rag/add", json={"text": "RAG retrieves context before generation"})
    assert r1.status_code == 200 and r2.status_code == 200

    # query should hit the second doc
    q = client.post("/rag/query", json={"query": "RAG", "limit": 3})
    assert q.status_code == 200
    results = q.json()["results"]
    assert len(results) == 1
    assert any("RAG retrieves context" in d["text"] for d in results)


def test_rag_add_empty_fails():
    r = client.post("/rag/add", json={"text": "   "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Text cannot be empty"


def test_rag_invalid_query_returns_empty_list():
    q = client.post("/rag/query", json={"query": "no-such-keyword"})
    assert q.status_code == 200
    assert q.json()["results"] == []


def test_rag_query_limit():
    # add three docs
    client.post("/rag/add", json={"text": "doc one"})
    client.post("/rag/add", json={"text": "doc two"})
    client.post("/rag/add", json={"text": "doc three"})

    # query with limit=2 should return only 2 results
    q = client.post("/rag/query", json={"query": "doc", "limit": 2})
    assert q.status_code == 200
    results = q.json()["results"]
    assert len(results) == 2


def test_rag_query_empty_keyword():
    q = client.post("/rag/query", json={"query": "   "})
    assert q.status_code == 200
    assert q.json()["results"] == []


def test_rag_query_vector():
    from app.summarize.config import settings

    settings.llm_provider = "dummy"  # Ensure using DummyLLM for this test
    settings.openai_api_key = None

    # add three docs
    client.post("/rag/add", json={"text": "doc one"})
    client.post("/rag/add", json={"text": "doc two"})
    client.post("/rag/add", json={"text": "doc three"})

    q = client.post("/rag/query_vector", json={"query": "doc", "limit": 2})
    # query with limit=2 should return only 2 results
    assert q.status_code == 200
    results = q.json()["results"]
    assert len(results) == 2
    assert all("id" in d and "text" in d and "score" in d for d in results)
    assert results[0]["score"] >= results[1]["score"]  # descending order


def test_rag_query_vector_dimension_mismatch(monkeypatch):
    STORE.clear()
    bad_id = "bad-12345"
    STORE.docs[bad_id] = "This doc has a bad embedding vector"
    STORE.vecs[bad_id] = [0.1, 0.2]
    q = client.post("/rag/query_vector", json={"query": "test", "limit": 3})
    assert all(r["id"] != bad_id for r in q.json().get("results", []))


def test_rag_query_vector_query_dimension_empty(monkeypatch):
    import app.rag.store as store

    STORE.clear()
    # Patch embed to return empty vector
    monkeypatch.setattr(store, "embed", lambda text: [])
    results = client.post("/rag/query_vector", json={"query": "test", "limit": 3})
    assert results.status_code == 200
    assert results.json()["results"] == []
