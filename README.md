# AI RAG Demo
![coverage](https://img.shields.io/badge/coverage-94%25-brightgreen)

This project demonstrates how to build and run a simple AI-powered FastAPI service with multiple capabilities:
- **`/summarize`**: Uses a pluggable LLM (dummy by default) to shorten or summarize text with basic retry/self-check logic.
- **`/agent`**: A minimal, rule-based agent that decides which tool to run (calculator, ping, echo), execute it, and return structured results with history.
- **`/ping`**: Simple health check that returns `{"status": "ok"}`. Useful for monitoring or testing service availability.
- **`/rag`**: Simple retrieval API to add and query text documents, used by the agent's `rag:` tool for keyword-based search.

## System Overview
- FastAPI service exposing `/summarize`, `/agent`, `/rag`
- Core packages:
  - `app/agents/` - tool routing & agent logic
  - `app/rag/` - document store, embeddings, retrieval
  - `app/summarize/` - LLM abstraction layer

## Agent Scaffolding

The `/agent` endpoint is backed by a minimal, rule-based agent.
It decides which tool to run based on simple heuristics:

1. **Command prefixes**
   - `calc: <expr>` -> runs the calculator
   - `echo: <text>` -> runs the echo tool
   - `ping` -> runs the ping tool
   - `rag: <text>` -> runs a dummy rag tool based on top-N docs ranked by simple keyword scoring.
   - `rag_anwer: <text>` -> runs a rag tool based on coine similarity with stored vectors

2. **Math detection**
   - If the input looks like a math expression (digits/operators only), the calculator tool is used.
   - Example: `"12*(3+1)"` -> calculator -> `48`.

3. **Fallback**
   - If no prefix or math is detected, the input is passed to the echo tool.
   - Example: `"hello world"` -> echo -> `"hello world"`

4. **Error handling**
   - Invalid math (`foo+bar`) returns `calc_error`.
   - If a tool raises an exception (like division by zero in a custom tool), the agent catches it and returns `tool_error:<ExceptionType>`.
   - If no tools are available, the agent responds with `{"tool": "none", "output": "no_suitable_tool"}`.

The agent also maintains a **short history** of the last 10 interactions.
This is the foundation for more advanced features in the future.

## RAG Retrieval

The RAG (Retrieval-Augmented Generation) module provides a minimal in-memory document store and keyword search API.

- **`POST /rag/add`** - add a new document
  Example:
  ```bash
  curl -s -X POST http://127.0.0.1:8000/rag/add \
    -H "Content-Type: application/json" \
    -d '{"text":"LangChain is a framework for LLM applications"}'
  ```
- **`POST /rag/query`** - search for documents by keyword
  Example:
  ```bash
  curl -s -X POST http://127.0.0.1:8000/rag/query \
    -H "Content-Type: application/json" \
    -d '{"query":"LangChain"}'
  ```

You can also access retrieval through the agent:
```bash
curl -s -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text":"rag: LangChain"}'
```

Results are ranked by:
1. Keyword frequency
2. First occurrence position
3. Document length

## RAG Answering

The `rag_answer` tool extends the basic retrieval API by combining retrieved documents
with a summarization step to form a minimal **Retrieval-Augmented Generation (RAG)** pipeline.

- It searches the vector store using mock or real embeddings.
- Builds a short prompt that includes the retrieved context.
- Summarize the results with the configured LLM (by default, a deterministic `DummyLLM`)

### Example
```bash
curl -s -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text":"rag_answer: Explain RAG"}' | jq
```
Example response:
```json
{
  "tool": "rag_answer",
  "output": "{\"answer\": \"RAG = retrieve first, then generate.\", \"sources\": [\"b9d4...\", \"9a12...\"]}",
  "history": [...]
}
```
When no relevant documents are found, the tool returns:
```json
  "answer": "no relevant context found",
  "sources": []
```

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
pytest --cov=app --cov-report=term-missing
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


### RAG API
You can directly use the `/rag` routes to add and query text documents.

#### Add documents
```bash
curl -s -X POST http://127.0.0.1:8000/rag/add \
  -H "Content-Type: application/json" \
  -d '{"text":"RAG retrieves context before generation"}'
```

#### Query documents
```bash
curl -s -X POST http://127.0.0.1:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query":"RAG","limit":3}'
```

#### Example response
```json
{
  "results": [
    {
      "id": "2d3c6c22-5d3c-46e7-8d3f-b67c02b1dbd2",
      "text": "RAG retrieves context before generation"
    }
  ]
}
```

#### Query through the Agent
```bash
curl -s -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"text":"rag: RAG"}' | jq
```

#### Example response
```json
{
  "tool": "rag_search",
  "output": "[{\"id\":\"...\",\"text\":\"RAG retrieves context before generation\"}]",
  "history": [...]
}
```

## Testing and Development

All core modules (`agent`, `rag`, `embeddings`, and `tools`) include full pytest coverage,
including import-time behavior and fallback logic.

Key test coverage:
- **Embeddings:** Safe normalization, import fallback when `openai` not installed
- **Agent Tools:** `calculator`, `echo`, `ping`, `rag_search`, `rag_answer`.
- **RAG Answer:** Verifies both "no results" and "context found" branches.

Run tests:
```bash
pytest --cov=app --cov-report=term-missing
```

> Current test coverage: **~94%**
