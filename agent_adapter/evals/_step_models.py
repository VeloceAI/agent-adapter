"""`StepModels`: one `LLMPort` that sends each flow node or stage to its own model."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any

from agent_adapter.core.context import NODE_KEY
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, LLMResponse, StreamEvent
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import LedgerEntry
from agent_adapter.runtime.current import current_context

ALL_STEPS = "*"  # a mapping key that covers every step without its own entry
NODE_PREFIX = "node:"
STAGE_PREFIX = "stage:"


def step_of(stage: str, node: str = "") -> str:
    """The step of a model call: `node:<node>` inside a flow node, else `stage:<stage>`."""
    return NODE_PREFIX + node if node else STAGE_PREFIX + (stage or "llm")


def entry_step(entry: LedgerEntry) -> str:
    """The step of a ledger entry, from its `metadata["node"]` and `stage`."""
    return step_of(entry.stage, entry.metadata.get("node", ""))


def running_node() -> str:
    """The flow node the current context runs in, or ""."""
    ctx = current_context()
    node = ctx.extras.get(NODE_KEY) if ctx is not None else None
    return node if isinstance(node, str) else ""


def current_step(request: LLMRequest) -> str:
    """The step of a model call running now."""
    return step_of(request.stage, running_node())


class StepModels:
    """An `LLMPort` that answers each step with `steps[step]`, else with `baseline`.

    Steps are `node:<name>` (any call made inside a flow node) or `stage:<stage>` (a tool
    loop, agent or other stage); the key `"*"` covers every step not listed. Pass it as
    `FlowRunner(model=...)`, `ToolLoopTarget(model, ...)` or `run_tool_loop(model, ...)`.
    A routed port gets the request with `model=None`, so it uses its own model.

    Raises:
        ConfigError: `baseline` or a step model is not an `LLMPort`, or a step key is empty.
    """

    provider = "steps"

    def __init__(self, baseline: LLMPort, steps: Mapping[str, LLMPort] | None = None) -> None:
        given = dict(steps or {})
        for key, port in {"baseline": baseline, **given}.items():
            if not isinstance(key, str) or not key:
                raise ConfigError("StepModels step keys must be non-empty strings")
            if not callable(getattr(port, "generate", None)):
                raise ConfigError(f"StepModels {key!r} is not an LLMPort: {type(port).__name__}")
        self.baseline = baseline
        self.steps: Mapping[str, LLMPort] = given
        self.model = baseline.model

    def port_for(self, step: str) -> LLMPort:
        """The port that answers `step`."""
        return self.steps.get(step) or self.steps.get(ALL_STEPS) or self.baseline

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """The reply of the port mapped to the current step.

        Raises:
            ModelError: as the chosen port.
        """
        port, routed = self._route(request)
        return await port.generate(routed)

    async def stream(self, request: LLMRequest) -> AsyncIterator[StreamEvent]:
        """The stream of the port mapped to the current step.

        Raises:
            ModelError: as the chosen port.
        """
        port, routed = self._route(request)
        async for event in port.stream(routed):
            yield event

    def cache_key_fields(self) -> dict[str, Any]:
        """Every routed port's key material, plus the flow node running now (if any)."""
        return {
            "baseline": _key_of(self.baseline),
            "steps": {step: _key_of(port) for step, port in sorted(self.steps.items())},
            "node": running_node(),
        }

    def _route(self, request: LLMRequest) -> tuple[LLMPort, LLMRequest]:
        port = self.port_for(current_step(request))
        if port is self.baseline or request.model is None:
            return port, request
        return port, request.model_copy(update={"model": None})


def apply_mapping(
    baseline: LLMPort, mapping: Mapping[str, str], candidates: Mapping[str, LLMPort]
) -> StepModels:
    """A `StepModels` from an advisor mapping `{step: candidate name}`.

    Raises:
        ConfigError: a mapped name is not in `candidates`.
    """
    unknown = sorted(name for name in mapping.values() if name not in candidates)
    if unknown:
        raise ConfigError(f"mapping names unknown candidates: {unknown}")
    return StepModels(baseline, {step: candidates[name] for step, name in mapping.items()})


def _key_of(port: LLMPort) -> dict[str, Any]:
    fields = getattr(port, "cache_key_fields", None)
    extra = fields() if callable(fields) else {}
    return {"provider": getattr(port, "provider", ""), "model": port.model, "fields": extra}
