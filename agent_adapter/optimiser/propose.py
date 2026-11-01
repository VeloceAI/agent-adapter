"""One proposer stage: call, parse, validate, retry once with the problems, else fall back."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_adapter.core.errors import ModelError, ModelOutputInvalid
from agent_adapter.core.messages import LLMRequest, LLMResponse, Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import ErrorInfo
from agent_adapter.optimiser._fatal import raise_if_fatal
from agent_adapter.optimiser.model import Candidate, Edit, SurfaceCaps
from agent_adapter.optimiser.prompts import PROPOSAL_SCHEMA, retry_message
from agent_adapter.optimiser.surface import (
    Dimension,
    Validates,
    dimension_problems,
    validate_tree,
)

logger = logging.getLogger(__name__)

ATTEMPTS = 2  # the first reply plus one retry with the validation problems


class _EditReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    content: str = ""
    delete: bool = False
    reason: str = ""


class _ProposalReply(BaseModel):
    model_config = ConfigDict(extra="ignore")

    keep_base: bool = False
    notes: str = ""
    edits: list[_EditReply] = []


async def propose(
    llm: LLMPort,
    *,
    label: str,
    dim: Dimension,
    parent: Candidate,
    caps: SurfaceCaps,
    host: Validates,
    system: str,
    evidence: str,
    seed: int | None,
) -> Candidate:
    """Ask `llm` for edits to `parent.tree`; invalid twice (or a model error) -> the parent.

    Raises:
        BudgetExceeded: a budget hook stopped the call (the run stops).
        OptimiserError: a fatal model error (bad key, rejected request).
    """
    messages = [Message(role="system", content=system), Message(role="user", content=evidence)]
    problems: list[str] = []
    for attempt in range(1, ATTEMPTS + 1):
        request = LLMRequest(
            messages=tuple(messages),
            output_schema=PROPOSAL_SCHEMA,
            stage=f"propose:{label}",
            seed=seed,
        )
        try:
            response = await llm.generate(request)
        except ModelOutputInvalid as exc:
            problems = [f"the reply is not valid JSON for the schema ({exc})"]
            messages.append(Message(role="user", content=retry_message(problems)))
            continue
        except ModelError as exc:
            raise_if_fatal(exc, f"proposer {label}")
            info = ErrorInfo.from_exception(exc)
            logger.warning("proposer %s failed (%s); keeping the parent", label, info.kind)
            return _fallback(label, dim, parent, f"{info.kind}: {info.message}", attempt)
        outcome = _accept(response, label=label, dim=dim, parent=parent, caps=caps, host=host)
        if isinstance(outcome, Candidate):
            return outcome.model_copy(update={"attempts": attempt})
        problems = outcome
        logger.info("proposal %s attempt %d rejected: %d problems", label, attempt, len(problems))
        messages += [response.message, Message(role="user", content=retry_message(problems))]
    logger.warning("proposer %s fell back to %s after %d attempts", label, parent.label, ATTEMPTS)
    return _fallback(label, dim, parent, "; ".join(problems), ATTEMPTS)


def _accept(
    response: LLMResponse,
    *,
    label: str,
    dim: Dimension,
    parent: Candidate,
    caps: SurfaceCaps,
    host: Validates,
) -> Candidate | list[str]:
    """A candidate from a valid reply, else the list of problems to send back."""
    if not isinstance(response.parsed, dict):
        return [f"no JSON object in the reply (finish_reason={response.finish_reason})"]
    try:
        reply = _ProposalReply.model_validate(response.parsed)
    except ValidationError as exc:
        return [
            f"{'.'.join(map(str, e['loc']))}: {e['msg']}"
            for e in exc.errors(include_url=False, include_input=False)
        ]
    if reply.keep_base or not reply.edits:
        return _keep(label, dim, parent, reply.notes)
    edits = tuple(
        Edit(path=e.path, content=None if e.delete else e.content, reason=e.reason.strip())
        for e in reply.edits
    )
    problems = dimension_problems(edits, dim)
    paths = [e.path for e in edits]
    problems += [f"{p}: edited more than once" for p in sorted(set(paths)) if paths.count(p) > 1]
    problems += [f"{e.path}: an edit needs a reason" for e in edits if not e.reason]
    if problems:
        return problems
    changed = tuple(e for e in edits if parent.tree.files.get(e.path) != e.content)
    if not changed:
        return _keep(label, dim, parent, reply.notes)
    tree = parent.tree.with_edits({e.path: e.content for e in changed})
    problems = validate_tree(tree, caps, host)
    if problems:
        return problems
    return Candidate(
        label=label,
        dim=dim,
        edits=changed,
        tree=tree,
        parent_label=parent.label,
        notes=reply.notes,
    )


def _keep(label: str, dim: Dimension, parent: Candidate, notes: str) -> Candidate:
    return Candidate(
        label=label,
        dim=dim,
        edits=(),
        tree=parent.tree,
        parent_label=parent.label,
        notes=notes,
    )


def _fallback(
    label: str, dim: Dimension, parent: Candidate, error: str, attempts: int
) -> Candidate:
    return Candidate(
        label=label,
        dim=dim,
        edits=(),
        tree=parent.tree,
        parent_label=parent.label,
        fell_back=True,
        error=error,
        attempts=attempts,
    )
