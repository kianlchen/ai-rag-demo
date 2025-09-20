# AI RAG Demo (Week-1 scaffold)

Small FastAPI service with a `/summarize` endpoint that calls a pluggable LLM (dummy by default) and returns structured JSON with a basic retry/self-check.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit as needed
```