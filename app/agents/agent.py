from __future__ import annotations

import re
from typing import Dict

from .schemas import AgentResponse, AgentResult
from .tools import Tool

_DIGIT_CHARS = set("0123456789")
_OP_CHARS = set("+-*/()")

MATH_RE = re.compile(r"^[0-9+\-*/().\s]+$")

TOOL_ALIASES = {
    "calc": "calculator",
    "calculate": "calculator",
    "echo": "echo",
    "ping": "ping",
    "rag": "rag_search",
    "rag_answer": "rag_answer",
}


class Agent:
    """
    Minimal agent
      - decide_tool: naive rule-based planner
      - run: execute tool, append to history, return structured response
    """

    def __init__(self, tools: Dict[str, Tool]) -> None:
        self.tools = tools
        self.history: list[AgentResult] = []
        self._last_payload = ""

    def decide_tool(self, text: str) -> str:
        t = text.strip()
        if ":" in t:
            alias, payload = t.split(":", 1)
            alias = alias.strip().lower()
            tool_name = TOOL_ALIASES.get(alias)
            if tool_name and tool_name in self.tools:
                self._last_payload = payload.strip()
                return tool_name
        if t.lower() == "ping":
            self._last_payload = ""
            return "ping"
        if MATH_RE.match(t) or any((ch in _DIGIT_CHARS) or (ch in _OP_CHARS) for ch in t):
            self._last_payload = t
            return "calculator"
        self._last_payload = t
        return "echo"

    def run(self, text: str) -> AgentResponse:
        tool_name = self.decide_tool(text)
        tool = self.tools.get(tool_name)
        if not tool:
            result = AgentResult(tool="none", output="no_suitable_tool")
            self.history.append(result)
            return AgentResponse(tool=result.tool, output=result.output, history=self.history[-10:])
        try:
            output = tool(self._last_payload)
        except Exception as e:
            output = f"tool_error: {type(e).__name__}"
        result = AgentResult(tool=tool_name, output=output)
        self.history.append(result)
        return AgentResponse(tool=tool_name, output=output, history=self.history[-10:])
