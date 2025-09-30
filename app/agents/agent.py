from __future__ import annotations
from typing import Dict
from .schemas import AgentResponse, AgentResult
from .tools import Tool

_DIGIT_CHARS = set("0123456789")
_OP_CHARS = set("+-*/()")


class Agent:
    """
    Minimal agent
      - decide_tool: naive rule-based planner
      - run: execute tool, append to history, return structured response
    """

    def __init__(self, tools: Dict[str, Tool]) -> None:
        self.tools = tools
        self.history: list[AgentResult] = []

    def decide_tool(self, text: str) -> str:
        t = text.strip().lower()
        if t == "ping":
            return "ping"
        if any(ch in _DIGIT_CHARS or ch in _OP_CHARS for ch in text):
            return "calculator"
        return "echo"

    def run(self, text: str) -> AgentResponse:
        tool_name = self.decide_tool(text)
        tool = self.tools.get(tool_name)
        if not tool:
            result = AgentResult(tool="none", output="no_suitable_tool")
            self.history.append(result)
            return AgentResponse(
                tool=result.tool, output=result.output, history=self.history[-10:]
            )
        output = tool(text)
        result = AgentResult(tool=tool_name, output=output)
        self.history.append(result)
        return AgentResponse(tool=tool_name, output=output, history=self.history[-10:])
