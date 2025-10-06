from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .store import STORE

router = APIRouter(prefix="/rag", tags=["rag"])
store = STORE


class AddRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    query: str
    limit: int = 3


@router.post("/add")
def add_doc(req: AddRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    doc_id = store.add(req.text)
    return {"id": doc_id}


@router.post("/query")
def query_docs(req: QueryRequest):
    results = store.query(req.query, limit=req.limit)
    return {"results": results}
