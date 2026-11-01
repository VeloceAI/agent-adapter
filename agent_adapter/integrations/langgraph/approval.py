"""LangGraph approvals: an ApprovalPort that pauses the graph with `interrupt()`.

Every interrupt value carries a `binding`: the SHA-256 of the request's run id, tool, kind,
`request_digest`, the call id (when stable) and the approval id (when a store holds the
record). A resume value must be `{"binding", "decision", "mac"}`, where `mac` is the
HMAC-SHA256 of the binding and the decision under the port's key
(`LangGraphApproval.resume_value` makes one). Anything else is LangGraph's "invalid resume":
nothing is saved and the interrupt can be answered again. Write access to the checkpointer
is therefore not approval authority, and each answer is used once per task (`_spent`).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict

from agent_adapter.approval.store import request_digest
from agent_adapter.core.errors import ApprovalPending, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest
from agent_adapter.flow.visit import runner_key
from agent_adapter.integrations.langgraph._context import task_namespace
from agent_adapter.integrations.langgraph._spent import Asked, ClaimKey, Owner, claim, scoped_owner

logger = logging.getLogger(__name__)

PAYLOAD_TYPE = "agent_adapter.approval"
MAC_DOMAIN = b"agent_adapter.langgraph.resume/1\n"  # separates resume MACs from other uses
SCRATCHPAD_KEY = "__pregel_scratchpad"  # LangGraph 1.x: a task's resume values, `configurable`
NOT_IN_NODE = "LangGraphApproval decides only inside a LangGraph node with a checkpointer"
UNSUPPORTED = "unsupported LangGraph version: the task's resume scratchpad has an unknown shape"

# The task namespace of a compiled flow node: its approvals defer to the node (see `defer_to_node`).
_DEFER: ContextVar[str | None] = ContextVar("aa_langgraph_defer", default=None)
# (request, key) pairs `LangGraphApproval` decided from a resume value, while `collect_asked` runs.
_ASKED: ContextVar[list[tuple[ApprovalRequest, bytes]] | None] = ContextVar(
    "aa_langgraph_asked", default=None
)


class ResumeValue(BaseModel):
    """What a resume value must be: the interrupt's binding, the decision and their MAC."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    binding: str
    decision: ApprovalDecision
    mac: str


def binding(
    request: ApprovalRequest, *, exact: bool = True, with_id: bool = False, scope: str = ""
) -> str:
    """The hex binding of `request`; `exact=False` leaves an unstable call id out, `with_id`
    adds the approval id (a store's record), `scope` names the graph (a compiled flow's).

    Raises:
        ToolUsageError: the arguments are not plain JSON (approvals fail closed).
    """
    body = {
        "v": 2,
        "run_id": request.run_id,
        "tool": request.tool,
        "kind": request.kind,
        "digest": request_digest(request),
        "call_id": request.call_id if exact else "",
        "approval_id": request.approval_id if with_id else "",
        "scope": scope,
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def resume_mac(key: bytes, bound_to: str, decision: ApprovalDecision) -> str:
    """HMAC-SHA256 hex of a binding and a decision (its JSON, canonical)."""
    data = json.dumps(
        decision.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    message = MAC_DOMAIN + f"{bound_to}\n{data}".encode("utf-8", "surrogatepass")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def resume_value(key: bytes, interrupt: Any, decision: ApprovalDecision) -> dict[str, Any]:
    """The resume value answering `interrupt` (an `Interrupt` or its value) with `decision`.

    Raises:
        ToolUsageError: `interrupt` is not an agent_adapter approval interrupt.
    """
    value = getattr(interrupt, "value", interrupt)
    if not isinstance(value, Mapping) or value.get("type") != PAYLOAD_TYPE:
        raise ToolUsageError("not an agent_adapter approval interrupt")
    bound_to = str(value["binding"])
    return {
        "binding": bound_to,
        "decision": decision.model_dump(mode="json"),
        "mac": resume_mac(key, bound_to, decision),
    }


def payload(
    request: ApprovalRequest, bound_to: str, extra: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """The interrupt value: the request the reviewer sees and the binding to answer with."""
    return {
        "type": PAYLOAD_TYPE,
        "binding": bound_to,
        "request": request.model_dump(mode="json"),
        **(extra or {}),
    }


def _valid(value: Any, expected: str, key: bytes) -> bool:
    if not isinstance(value, ResumeValue):
        return False
    mac = resume_mac(key, value.binding, value.decision)
    return hmac.compare_digest(value.binding, expected) and hmac.compare_digest(value.mac, mac)


def ask(
    request: ApprovalRequest,
    *,
    key: bytes,
    exact: bool = True,
    extra: Mapping[str, Any] | None = None,
    scope: str = "",
    started: bool = True,
) -> ApprovalDecision:
    """Interrupt for `request` (or take this position's resume value) and return the decision.

    Raises:
        GraphInterrupt: no resume value yet (the graph pauses; always propagates).
        pydantic.ValidationError: the value is not bound to `request` or its MAC is wrong
            (LangGraph's invalid resume: nothing is saved).
        ApprovalAlreadyDecided: the answer was already used in this task (`_spent`).
        ToolUsageError: called outside a LangGraph node or a task scope (`one_use`), an
            unsupported LangGraph version, or a value LangGraph let through unvalidated.
    """
    return ask_claimed(request, key=key, exact=exact, extra=extra, scope=scope, started=started)[0]


def ask_claimed(
    request: ApprovalRequest,
    *,
    key: bytes,
    exact: bool = True,
    extra: Mapping[str, Any] | None = None,
    scope: str = "",
    started: bool = True,
) -> tuple[ApprovalDecision, ClaimKey]:
    """`ask`, also returning the key of the claim it took on the answer (raises as `ask`)."""
    expected = binding(request, exact=exact, scope=scope)
    config = _node_config()
    namespace = task_namespace(config)
    owner = scoped_owner(request.run_id, namespace)  # one use per task run needs a task scope
    resume_count(config)  # a scratchpad of an unknown shape fails closed
    _catch_up(owner, namespace)
    if owner.replays:
        owner.replays.popleft()  # this ask's own position: the node asks for it again
    decision = _interrupt(request, key, expected, extra)
    claimed = claim(request.run_id, namespace, "resume", expected, started=started)
    owner.asked.append(Asked(request, key))
    return decision, claimed


def catch_up(owner: Owner) -> None:
    """Claim the answers of the finished calls a one_use node replays first (before it runs)."""
    if owner.replays:
        resume_count(_node_config())  # a scratchpad of an unknown shape fails closed
        _catch_up(owner, owner.namespace)


def _node_config() -> Any:
    """The running LangGraph node's config; ToolUsageError outside a node."""
    from langgraph.config import get_config

    try:
        return get_config()
    except RuntimeError as exc:
        raise ToolUsageError(NOT_IN_NODE) from exc


def _catch_up(owner: Owner, namespace: str) -> None:
    """Take the resume positions of the finished calls a one_use node replays next.

    Claimed started, so a failure before the node's next ask still spends them.
    """
    while owner.replays and owner.replays[0].finished:
        head = owner.replays.popleft()
        request = head.request
        expected = binding(request)
        _interrupt(request, head.key, expected, None)
        claim(request.run_id, namespace, "resume", expected)  # its call ran: spent on failure
        owner.asked.append(head)


def _interrupt(
    request: ApprovalRequest, key: bytes, expected: str, extra: Mapping[str, Any] | None
) -> ApprovalDecision:
    from langgraph.types import interrupt

    def check(value: ResumeValue) -> ResumeValue:
        if not _valid(value, expected, key):
            logger.warning(
                "resume value for tool %s (run %s, call %s) is not bound to it or not signed",
                request.tool,
                request.run_id,
                request.call_id,
            )
            raise ValueError("the resume value is not bound to this approval request")
        return value

    schema = Annotated[ResumeValue, AfterValidator(check)]
    try:
        value = interrupt(payload(request, expected, extra), response_schema=schema)
    except (RuntimeError, KeyError) as exc:
        raise ToolUsageError(NOT_IN_NODE) from exc
    if not _valid(value, expected, key):  # LangGraph returned a value it did not validate
        raise ToolUsageError("the resume value is not bound to this approval request")
    decision: ApprovalDecision = value.decision  # typed without langgraph installed
    return decision


def resume_count(config: Any) -> int:
    """How many resume values LangGraph holds for the running task.

    Raises:
        ToolUsageError: the scratchpad is missing or has an unknown shape (fails closed).
    """
    configurable = config.get("configurable") or {} if isinstance(config, Mapping) else {}
    resume = getattr(configurable.get(SCRATCHPAD_KEY), "resume", None)
    if not isinstance(resume, list):
        raise ToolUsageError(UNSUPPORTED)
    return len(resume)


class LangGraphApproval:
    """An ApprovalPort that pauses the graph with `interrupt()`; a signed resume value decides.

    `key` signs resume values (`resume_value`); None is a random per-process key, so a run
    resumes only in the process that paused it: give every process the same key otherwise.
    Inside a compiled flow node the node raises the interrupt itself.

    Raises:
        ToolUsageError: `key` is not bytes of at least 16 bytes.
    """

    def __init__(self, key: bytes | None = None) -> None:
        self.key = runner_key(key)

    async def request(self, request: ApprovalRequest) -> ApprovalDecision:
        """The reviewer's decision for `request`, from the resume value at this position.

        Raises:
            GraphInterrupt: the graph pauses until a resume value arrives.
            ApprovalPending: inside a compiled flow node (the node interrupts for it).
            ApprovalAlreadyDecided: the answer was already used in this task.
            ToolUsageError: outside a LangGraph node.
        """
        deferred = _DEFER.get()
        if deferred is not None and deferred == _current_namespace():
            raise ApprovalPending(
                request.approval_id,
                tool=request.tool,
                run_id=request.run_id,
                call_id=request.call_id,
                arguments=dict(request.arguments),
            )
        decision = ask(request, key=self.key)
        asked = _ASKED.get()
        if asked is not None:
            asked.append((request, self.key))
        return decision

    def resume_value(self, interrupt: Any, decision: ApprovalDecision) -> dict[str, Any]:
        """The signed value for `Command(resume={interrupt.id: ...})` answering `interrupt`."""
        return resume_value(self.key, interrupt, decision)

    def __repr__(self) -> str:
        return "LangGraphApproval()"


@contextmanager
def defer_to_node(namespace: str) -> Iterator[None]:
    """Within this block, `LangGraphApproval` in task `namespace` raises `ApprovalPending`."""
    token = _DEFER.set(namespace)
    try:
        yield
    finally:
        _DEFER.reset(token)


@contextmanager
def collect_asked() -> Iterator[list[tuple[ApprovalRequest, bytes]]]:
    """Collect the requests `LangGraphApproval` answers from resume values in this block."""
    asked: list[tuple[ApprovalRequest, bytes]] = []
    token = _ASKED.set(asked)
    try:
        yield asked
    finally:
        _ASKED.reset(token)


def _current_namespace() -> str:
    from langgraph.config import get_config

    try:
        return task_namespace(get_config())
    except RuntimeError:  # not inside a runnable
        return ""


__all__ = ["LangGraphApproval"]
