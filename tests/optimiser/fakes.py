"""Offline fakes for the optimiser tests: a toy host, a tree-driven agent, a scripted proposer."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.types import ToolCall, Usage
from agent_adapter.hooks.base import Hook, ShortCircuit
from agent_adapter.models.callable import CallableModel
from agent_adapter.optimiser.model import (
    CaseRun,
    EvalCase,
    HarnessTree,
    Score,
    SurfaceCaps,
)
from agent_adapter.optimiser.surface import DEFAULT_CAPS

PRICED = "gpt-4.1"  # in the default price table: $2 / $8 per 1M input / output tokens
USAGE = Usage(input_tokens=1000, output_tokens=100)  # $0.0028 per call on PRICED
CALL_USD = 0.0028


def reply(content: str, *, parsed: Any = None, model: str = PRICED) -> LLMResponse:
    """A priced text reply."""
    return LLMResponse(
        provider="openai",
        model=model,
        text=content,
        parsed=parsed,
        usage=USAGE,
        message=Message(role="assistant", content=content),
    )


def tool_reply(name: str, args: dict[str, Any], call_id: str = "c1") -> LLMResponse:
    """A priced reply asking for one tool call."""
    call = ToolCall(name=name, arguments=args, id=call_id)
    return LLMResponse(
        provider="openai",
        model=PRICED,
        tool_calls=(call,),
        finish_reason="tool_calls",
        usage=USAGE,
        message=Message(role="assistant", tool_calls=(call,)),
    )


def system_of(request: LLMRequest) -> str:
    return next((m.content for m in request.messages if m.role == "system"), "")


def user_of(request: LLMRequest) -> str:
    return next((m.content for m in reversed(request.messages) if m.role == "user"), "")


# --- toy host -----------------------------------------------------------------------------

CASES = (
    EvalCase(id="q1", input="rent relief", expected="80C"),
    EvalCase(id="q2", input="tuition fees", expected="80C"),
    EvalCase(id="q3", input="home loan", expected="80C"),
)
BASE_FILES = {
    "prompt/system": "You answer tax questions.",
    "tool/search/description": "Search the archive.",
    "hook/budget/config.json": json.dumps({"max_usd": 5.0}),
}


class ToyAgent:
    """The agent model: its answer depends only on the harness text it is shown."""

    def __init__(self, reject: Callable[[LLMRequest], BaseException | None] | None = None) -> None:
        self.requests: list[LLMRequest] = []
        self.reject = reject
        self.port = CallableModel(self._answer, provider="openai", model=PRICED)

    async def _answer(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        system, user = system_of(request), user_of(request)
        error = self.reject(request) if self.reject else None
        if error is not None:
            raise error
        if "EMPTY" in system:
            return reply("")
        if "BROKEN" in system:
            raise RuntimeError("agent crashed")
        if "cite" in system or "Cite" in system:
            return reply(f"{user}: see section 80C")
        return reply(f"{user}: no idea")


async def judge_80c(case: EvalCase, run: CaseRun) -> Score | None:
    """Exact-match judge: 1.0 when the expected section is in the output."""
    if "JUDGE-FAILS" in run.output:
        raise RuntimeError("judge broke")
    hit = str(case.expected) in run.output
    return Score(value=1.0 if hit else 0.0, passed=hit, notes="cites it" if hit else "missing")


class FakeHost:
    """A minimal Host: one model call per case with the tree's text as the system prompt."""

    def __init__(
        self,
        *,
        cases: Sequence[EvalCase] = CASES,
        files: Mapping[str, str] | None = None,
        judge: Callable[[EvalCase, CaseRun], Any] | None = judge_80c,
        domain: str = "tax",
        problems: Callable[[HarnessTree], list[str]] | None = None,
    ) -> None:
        self.domain = domain
        self.judge = judge
        self._cases = tuple(cases)
        self._files = dict(BASE_FILES if files is None else files)
        self._problems = problems
        self.runs: list[tuple[str, str]] = []

    def surface(self) -> HarnessTree:
        return HarnessTree(files=dict(self._files))

    def caps(self) -> SurfaceCaps:
        return SurfaceCaps(
            max_chars=dict(DEFAULT_CAPS),
            kinds=frozenset({"tool", "hook", "prompt"}),
            hook_schemas={"budget": {"type": "object", "required": ["max_usd"]}},
        )

    def validate(self, tree: HarnessTree) -> list[str]:
        return self._problems(tree) if self._problems else []

    def cases(self) -> Sequence[EvalCase]:
        return self._cases

    async def run(
        self, tree: HarnessTree, case: EvalCase, *, run_id: str, model: LLMPort
    ) -> CaseRun:
        self.runs.append((run_id, case.id))
        files = tree.files
        system = "\n".join(
            files.get(p, "")
            for p in ("prompt/system", "tool/search/description", "tool/search/prompt_hint")
        )
        request = LLMRequest(
            messages=(
                Message(role="system", content=system),
                Message(role="user", content=str(case.input)),
            ),
            stage="agent",
        )
        try:
            response = await model.generate(request)
        except Exception as exc:
            return CaseRun(
                case_id=case.id,
                label="",
                ok=False,
                delivered=False,
                output="",
                transcript="",
                error=f"{getattr(exc, 'kind', type(exc).__name__)}: {exc}",
            )
        searched = "search" in files.get("tool/search/prompt_hint", "")
        return CaseRun(
            case_id=case.id,
            label="",
            ok=True,
            delivered=bool(response.text),
            output=response.text,
            transcript=f"[USER] {case.input}\n[ASSISTANT] {response.text}",
            metrics={"tool_calls.search": 1.0 if searched else 0.0, "steps": 1.0},
        )


# --- scripted proposer --------------------------------------------------------------------

Scripted = Mapping[str, Any] | str | BaseException


class ScriptedProposer:
    """Answers by stage label (`propose:proc1` -> "proc1"); a list per label is used in order."""

    def __init__(self, script: Mapping[str, Scripted | Sequence[Scripted]]) -> None:
        self.script = {
            label: list(items) if isinstance(items, list) else [items]
            for label, items in script.items()
        }
        self.requests: list[LLMRequest] = []
        self.port = CallableModel(self._answer, provider="openai", model=PRICED)

    def stages(self) -> list[str]:
        return [r.stage for r in self.requests]

    async def _answer(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        label = request.stage.split(":", 1)[-1]
        items = self.script.get(label) or self.script.get(request.stage.split(":", 1)[0])
        if not items:
            return reply(json.dumps(keep_base()), parsed=keep_base())
        item = items.pop(0) if len(items) > 1 else items[0]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, str):
            return reply(item)
        return reply(json.dumps(item), parsed=dict(item))


def proposal(*edits: tuple[str, str | None], notes: str = "n") -> dict[str, Any]:
    """A proposer reply editing `(path, content)` pairs (None deletes)."""
    return {
        "keep_base": False,
        "notes": notes,
        "edits": [
            {"path": p, "content": c or "", "delete": c is None, "reason": f"because {p}"}
            for p, c in edits
        ],
    }


def keep_base(notes: str = "nothing to change") -> dict[str, Any]:
    return {"keep_base": True, "notes": notes, "edits": []}


def make_registry() -> Registry:
    """Three tools: search, fetch and a confirm-gated delete."""
    registry = Registry()

    @registry.tool(parallel_safe=True)
    async def search(query: str, token: str = "") -> str:
        """Search the archive."""
        return f"Section 80C covers {query}"

    @registry.tool(prompt_hint="Use after search.")
    async def fetch(doc_id: str) -> str:
        """Fetch one document."""
        return f"doc {doc_id}"

    @registry.tool(confirm=True)
    async def delete(doc_id: str, api_key: str = "") -> str:
        """Delete a document."""
        return "deleted"

    return registry


class LoopAgent:
    """A tool-loop agent: searches when the harness tells it to, then answers from the result."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []
        self.port = CallableModel(self._answer, provider="openai", model=PRICED)

    async def _answer(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        last = request.messages[-1]
        if last.role == "tool":
            return reply(f"ANSWER: {last.content}")
        system = system_of(request)
        descriptions = " ".join(t.description for t in request.tools)
        if "BROKEN" in system:
            raise RuntimeError("agent crashed")
        if "DELETE" in system:
            return tool_reply("delete", {"doc_id": "d1", "api_key": "sk-secret"})
        if "always search" in system.lower() or "every question" in descriptions:
            return tool_reply("search", {"query": user_of(request), "token": "t0p"})
        return reply("no idea")


class BlockHook(Hook):
    """A tunable, non-safety hook: short-circuits calls to the listed tools."""

    name = "block"

    def __init__(self, tools: Sequence[str] = ()) -> None:
        self.tools = tuple(tools)

    async def before_tool(self, inv: Any) -> ShortCircuit | None:
        return ShortCircuit("blocked by policy") if inv.spec.name in self.tools else None

    def config(self) -> dict[str, Any]:
        return {"tools": list(self.tools)}

    @classmethod
    def config_schema(cls) -> dict[str, Any] | None:
        return {
            "type": "object",
            "properties": {"tools": {"type": "array", "items": {"type": "string"}}},
            "required": ["tools"],
            "additionalProperties": False,
        }

    def with_config(self, config: Mapping[str, Any]) -> BlockHook:
        unknown = sorted(set(config) - {"tools"})
        if unknown:
            raise ConfigError(f"unknown BlockHook config keys: {unknown}")
        return BlockHook(config["tools"])
