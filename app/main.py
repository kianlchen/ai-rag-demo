from fastapi import FastAPI, HTTPException
from .schemas import SummarizeRequest, SummarizeResponse
from .llm import summarize_with_retry
from .utils import truncate_words

app = FastAPI(title="AI RAG Demo", version="1.0.0")


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(
            status_code=400, detail="Text cannot be empty or whitespace"
        )
    # defensive cap
    text = req.text.strip()
    summary, conf, _ = summarize_with_retry(text, req.max_words)
    summary = truncate_words(summary or "", req.max_words)
    return SummarizeResponse(summary=summary, confidence=float(conf))
