from __future__ import annotations

import ast
import json
import operator as op
from typing import Callable, Dict

from app.rag.store import STORE
from app.summarize.llm import summarize_with_retry

Tool = Callable[[str], str]

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_OPS[type(node.op)](left, right)
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    raise ValueError("unsupported expression")


def _safe_calc(expr: str) -> str:
    try:
        parsed = ast.parse(expr, mode="eval").body
        val = _eval_node(parsed)
        return str(int(val)) if float(val).is_integer() else str(val)
    except Exception:
        return "calc_error"


def calculator(query: str) -> str:
    import re

    expr = "".join(re.findall(r"[0-9+\-*/(). ]", query))
    return _safe_calc(expr) if expr.strip() else "calc_error"


def echo(query: str) -> str:
    return query


def ping(_: str) -> str:
    return "pong"


def rag_search(query: str) -> str:
    results = STORE.query(query, limit=3)
    if not results:
        return "no_results"
    # return a compact json string; to be structured later
    return json.dumps(results)


def rag_answer(query: str) -> str:
    """
    Retrieve top-k semantically similar docs, then summarize into an answer.
    Returns a compact JSON string with {answer, sources:[ids]}
    """
    results = STORE.query_vector(query, limit=3)
    if not results:
        return json.dumps({"answer": "no relevant context found", "sources": []})
    context = "\n\n".join(f"- {r['text']}" for r in results)
    prompt = f"Using only the context below, answer the question.\n\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"
    summary, conf, _ = summarize_with_retry(prompt, max_words=80)
    return json.dumps({"answer": summary or "", "sources": [r["id"] for r in results]})


REGISTRY: Dict[str, Tool] = {
    "calculator": calculator,
    "echo": echo,
    "ping": ping,
    "rag_search": rag_search,
    "rag_answer": rag_answer,
}
