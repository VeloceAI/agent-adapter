"""The runner an exported tool uses when it is given no executor: the default, found per call."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from agent_adapter.core.context import ToolContext
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.tool import ToolSpec
from agent_adapter.exec.defaults import default_executor
from agent_adapter.exec.executor import Executor
from agent_adapter.integrations.base import ToolRunner


class DefaultRunner:
    """Runs each call on `default_executor()` as it is at call time.

    With `approval`, calls run on a copy of the default with that port; the copy is rebuilt
    only when the default changes, and the port itself is kept.
    """

    def __init__(self, approval: ApprovalPort | None = None) -> None:
        self.approval = approval
        self._lock = threading.Lock()
        self._base: Executor | None = None
        self._copy: Executor | None = None

    def executor(self) -> Executor:
        """The executor the next call runs on."""
        base = default_executor()
        if self.approval is None:
            return base
        with self._lock:
            if self._copy is None or self._base is not base:
                self._base, self._copy = base, base.replace(approval=self.approval)
            return self._copy

    async def run_spec(
        self,
        spec: ToolSpec,
        args: Mapping[str, Any] | None = None,
        ctx: ToolContext | None = None,
        *,
        call_id: str | None = None,
    ) -> Any:
        """`self.executor().run_spec(...)`; failures propagate."""
        return await self.executor().run_spec(spec, args, ctx, call_id=call_id)

    def __repr__(self) -> str:
        return f"DefaultRunner(approval={self.approval!r})"


DEFAULT_RUNNER = DefaultRunner()


def lazy_runner(executor: ToolRunner | None) -> ToolRunner:
    """`executor`, else `DEFAULT_RUNNER`, so a later `set_default_executor` still applies."""
    return executor if executor is not None else DEFAULT_RUNNER


__all__ = ["DEFAULT_RUNNER", "DefaultRunner", "lazy_runner"]
