# AI RAG Demo

This project demonstrates how to build and run a simple AI-powered FastAPI service with multiple capabilities:
- **`/summarize`**: Uses a pluggable LLM (dummy by default) to shorten or summarize text with basic retry/self-check logic.
- **`/agent`**: A minimal, rule-based agent that decides which tool to run (calculator, ping, echo), execute it, and return structured results with history.
- **`/ping`**: Simple health check that returns `{"status": "ok"}`. Useful for monitoring or testing service availability.

## Agent Scaffolding

The `/agent` endpoint is backed by a minimal, rule-based agent.
It decides which tool to run based on simple heuristics"

1. **Command prefixes**
   - `calc: <expr>` -> runs the calculator
   - `echo: <text>` -> runs the echo tool
   - `ping` -> runs the ping tool

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
