"""Node actions (`Call`, `Map`, `Join`, `SubFlow`), routes, the node cache, node context reads."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Generic, Literal, TypeVar, cast

from pydantic import BaseModel

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import FlowDefinitionError
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.tool import ToolSpec
from agent_adapter.core.types import CacheScope

S = TypeVar("S", bound=BaseModel)

# A node function: fn(state) or fn(state, ctx), sync or async, returning a mapping update or None.
NodeFn = Callable[[S], Any] | Callable[[S, ToolContext], Any]

PROMPT_KEY = "aa.node_prompt"
MODEL_KEY = "aa.model"


@dataclass(frozen=True)
class Call:
    """Run one tool through the executor; `args` may be a function of the state."""

    tool: str | ToolSpec
    args: Mapping[str, Any] | Callable[[Any], Mapping[str, Any]] = field(default_factory=dict)
    into: str | None = None  # state field for the tool's value; None = no write


@dataclass(frozen=True)
class Map:
    """Fan out over the list field `over`: one tool call (or `fn` call) per item."""

    over: str
    tool: str | ToolSpec | None = None  # same tool per item; args = item, or args_fn(item)
    tool_from: str | None = None  # item[tool_from] names the tool
    args_from: str = "args"  # with tool_from: item[args_from], else the item minus tool_from
    args_fn: Callable[[Any], Mapping[str, Any]] | None = None
    fn: Callable[..., Any] | None = None  # fn(item, state, ctx) -> value
    into: str | None = None  # default: the node's name
    on_error: Literal["fail", "skip"] = "fail"

    def __post_init__(self) -> None:
        chosen = [self.tool is not None, self.tool_from is not None, self.fn is not None]
        if sum(chosen) != 1:
            raise FlowDefinitionError(
                [f"Map({self.over!r}) needs exactly one of tool=, tool_from= or fn="]
            )


@dataclass(frozen=True)
class Join:
    """Fan-in: runs once nothing else is pending; `fn` is an optional reducer step."""

    sources: Sequence[str]
    fn: NodeFn[Any] | None = None


@dataclass(frozen=True)
class SubFlow:
    """Run another `Flow` as one node: parent state in through `input`, its result out by `into`.

    `input`: child field -> parent field, or `fn(parent_state)` -> the child's initial values;
    None copies fields of the same name (additive `append` / `sum` fields start empty).
    `into`: a parent field for the child's final state, parent field -> child field, or
    `fn(child_state)` -> the parent update; None writes back same-named fields it changed.
    """

    flow: Any  # a Flow (typed loosely: the builder module imports this one)
    _: KW_ONLY
    input: Mapping[str, str] | Callable[[Any], Mapping[str, Any]] | None = None
    into: str | Mapping[str, str] | Callable[[Any], Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class NodeCache:
    """Reuse a node's updates for equal inputs: the state, its `reads` fields, or `key(state)`."""

    ttl: float | None = None  # seconds; None keeps entries as long as the backend does
    _: KW_ONLY
    key: Callable[[Any], Any] | None = None
    reads: Sequence[str] | None = None
    scope: CacheScope = "run"  # who shares an entry: the run, the thread, or everyone


@dataclass(frozen=True)
class Route(Generic[S]):
    """Pick the next node(s): `fn(state)` returns a label or labels found in `targets`."""

    fn: Callable[[S], str | Sequence[str]]
    targets: Sequence[str] | Mapping[str, str]  # node names, or label -> node name


def node_prompt(ctx: ToolContext) -> str | None:
    """The running node's prompt, or None."""
    prompt = ctx.extras.get(PROMPT_KEY)
    return prompt if isinstance(prompt, str) else None


def node_model(ctx: ToolContext) -> LLMPort | None:
    """The flow runner's model, instrumented by its executor and bound to this node, or None."""
    return cast("LLMPort | None", ctx.extras.get(MODEL_KEY))
