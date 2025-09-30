from __future__ import annotations
import ast
import operator as op
from typing import Callable, Dict

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
        parsed = ast.parse(expr, mode="eval")
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


REGISTRY: Dict[str, Tool] = {"calculator": calculator, "echo": echo, "ping": ping}
