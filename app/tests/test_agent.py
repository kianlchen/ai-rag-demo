import ast
import json

import pytest
from fastapi.testclient import TestClient

from app.agents.agent import Agent
from app.agents.tools import REGISTRY, _eval_node
from app.main import app
from app.rag.store import STORE

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_store():
    STORE.docs.clear()
    yield
    STORE.docs.clear()


def test_decide_and_run_calculator():
    agent = Agent(REGISTRY)
    resp = agent.run("(-1)*(2 + 2)")
    assert resp.tool == "calculator"
    assert resp.output in ("-4", "-4.0")


def test_calculator_invalid():
    agent = Agent(REGISTRY)
    resp = agent.run("foo + bar")
    assert resp.tool == "calculator"
    assert resp.output == "calc_error"


def test_agent_calc_prefix():
    agent = Agent(REGISTRY)
    resp = agent.run("calc: 3 * (4 + 5)")
    assert resp.tool == "calculator"
    assert resp.output in ("27", "27.0")


def test_agent_tool_error():
    # a tool that raises an error
    def boom(_: str) -> str:
        return 1 / 0  # ZeroDivisionError

    agent = Agent({"boom": boom})
    # force the agent to choose erroring tool
    agent.decide_tool = lambda _: "boom"
    resp = agent.run("anything")
    assert resp.tool == "boom"
    assert resp.output.startswith("tool_error:") and "ZeroDivisionError" in resp.output


def test_eval_node_unsupported_raises():
    node = ast.parse("foo", mode="eval")
    with pytest.raises(ValueError, match="unsupported expression"):
        _eval_node(node)


def test_run_ping():
    agent = Agent(REGISTRY)
    resp = agent.run("ping")
    assert resp.tool == "ping"
    assert resp.output == "pong"


def test_run_echo():
    agent = Agent(REGISTRY)
    resp = agent.run("hello, this is a test")
    assert resp.tool == "echo"
    assert resp.output == "hello, this is a test"


def test_agent_history_grows():
    agent = Agent(REGISTRY)
    agent.run("ping")
    agent.run("hello world")
    agent.run("3 * 5")
    assert len(agent.history) == 3
    assert agent.history[0].tool == "ping"
    assert agent.history[1].tool == "echo"
    assert agent.history[2].tool == "calculator"
    assert agent.history[0].output == "pong"
    assert agent.history[1].output == "hello world"
    assert agent.history[2].output in ("15", "15.0")


def test_api_agent_calculator():
    r = client.post("/agent", json={"text": "12*(3+1)"})
    assert r.status_code == 200
    data = r.json()
    assert data["tool"] == "calculator"
    assert data["output"] in ("48", "48.0")


def test_agent_no_tool():
    agent = Agent({})
    resp = agent.run("No tools available")
    assert resp.tool == "none"
    assert resp.output == "no_suitable_tool"


def test_agent_rag_prefix():
    STORE.add("RAG retrieves context before generation")
    agent = Agent(REGISTRY)
    resp = agent.run("rag: RAG")
    assert resp.tool == "rag_search"
    results = json.loads(resp.output)
    assert any("RAG retrieves context before generation" in r.get("text") for r in results)


def test_agent_rag_no_results():
    agent = Agent(REGISTRY)
    resp = agent.run("rag: no-such-keyword")
    assert resp.tool == "rag_search"
    assert resp.output == "no_results"
