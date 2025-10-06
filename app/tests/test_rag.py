import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.rag.store import STORE

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_store():
    STORE.docs.clear()
    yield
    STORE.docs.clear()


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
