# AI RAG Demo

This project demonstrates how to build and run a simple AI-powered FastAPI service with multiple capabilities:
- **`/summarize`**: Uses a pluggable LLM (dummy by default) to shorten or summarize text with basic retry/self-check logic.
- **`/agent`**: A minimal, rule-based agent that decides which tool to run (calculator, ping, echo), execute it, and return structured results with history.
- **`/ping`**: Simple health check that returns `{"status": "ok"}`. Useful for monitoring or testing service availability.

## Setup

### Local development
```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit with your API key if needed
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -t ai-rag-demo:dev .
docker run --rm -p 8000:8000 --env-file .env ai-rag-demo:dev
```

## Run tests
```bash
PYTHONPATH=. pytest --cov=app --cov-report=term-missing
```

## Examples

### summarize
```bash
curl -s -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text":"This is a long sentence that needs to be shorter","max_words":10}'
```

### agent
```bash
curl -s -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text":"(-1)*(2+2)"}'
```

### ping
```bash
curl -s http://127.0.0.1:8000/ping
```
