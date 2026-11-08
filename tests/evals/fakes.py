"""Shared eval test helpers: a shop registry, scripted models and cases."""

from __future__ import annotations

from typing import Any

from agent_adapter.core.registry import Registry
from agent_adapter.evals.model import EvalCase, EvalRun


def shop_registry() -> Registry:
    """A registry with a lookup tool and a confirm tool."""
    registry = Registry()

    @registry.tool
    def order_status(order_id: str) -> str:
        """Where an order is."""
        return {"1042": "shipped"}.get(order_id, "unknown")

    @registry.tool(confirm=True, effects="additive")
    def refund(order_id: str) -> str:
        """Refund an order (needs approval)."""
        return f"refunded {order_id}"

    return registry


def case(case_id: str = "c1", *, expected: Any = "shipped", **fields: Any) -> EvalCase:
    """A case asking about order 1042."""
    return EvalCase(
        id=case_id, input=fields.pop("input", "where is 1042"), expected=expected, **fields
    )


def run(output: Any = "It is shipped.", **fields: Any) -> EvalRun:
    """What a grader sees for one run."""
    return EvalRun(case_id="c1", output=output, **fields)
