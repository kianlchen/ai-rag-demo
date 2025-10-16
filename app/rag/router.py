from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .store import STORE

router = APIRouter(prefix="/rag", tags=["rag"])


class AddRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    query: str
    limit: int = 3


@router.post("/add")
def add_doc(req: AddRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    doc_id = STORE.add(req.text)
    return {"id": doc_id}


@router.post("/query")
def query_docs(req: QueryRequest):
    return {"results": STORE.query(req.query, limit=req.limit)}


@router.post("/query_vector")
def query_vector(req: QueryRequest):
    return {"results": STORE.query_vector(req.query, limit=req.limit)}
