"""Lessons: a JSON store of transferable cues (atomic rewrites) and the end-of-run reflection."""

from __future__ import annotations

import logging
import os
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_adapter.cache.keys import digest
from agent_adapter.core.errors import ModelError, OptimiserError
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.types import ErrorInfo
from agent_adapter.optimiser._fatal import raise_if_fatal
from agent_adapter.optimiser._files import write_atomic
from agent_adapter.optimiser.model import HarnessTree, Lesson
from agent_adapter.optimiser.prompts import REFLECT_SCHEMA, reflect_system
from agent_adapter.optimiser.surface import parse_path

logger = logging.getLogger(__name__)

STORE_VERSION = 1
Confidence = Literal["low", "medium", "high"]
_CONFIDENCE: dict[str, Confidence] = {"low": "low", "medium": "medium", "high": "high"}


class _StoreFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int
    lessons: list[Lesson]


class LessonsStore:
    """`lessons.json`: lessons of every run, rewritten atomically on each `add`."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def load(self) -> list[Lesson]:
        """Every stored lesson, oldest first; [] when the file does not exist.

        Raises:
            OptimiserError: the file is not a valid lessons file (it is never overwritten).
        """
        try:
            text = self.path.read_text("utf-8")
        except FileNotFoundError:
            return []
        try:
            data = _StoreFile.model_validate_json(text)
        except ValidationError as exc:
            logger.error("lessons file %s is invalid", self.path)
            raise OptimiserError(
                f"lessons file {self.path} is invalid: {exc.error_count()} problems"
            ) from exc
        if data.version != STORE_VERSION:
            raise OptimiserError(f"lessons file {self.path} has unsupported version {data.version}")
        return data.lessons

    def relevant(self, *, domain: str, tree: HarnessTree | None, limit: int) -> list[Lesson]:
        """Newest first: lessons of `domain` whose components exist in `tree` (or name none)."""
        found = [
            lesson
            for lesson in self.load()
            if lesson.domain == domain and (tree is None or _applies(lesson, tree))
        ]
        found.sort(key=lambda lesson: lesson.created_at, reverse=True)
        return found[: max(limit, 0)]

    def add(self, lessons: Sequence[Lesson]) -> None:
        """Append lessons whose id is new; re-reads the file first so other writers are kept."""
        current = self.load()
        known = {lesson.id for lesson in current}
        fresh = [lesson for lesson in lessons if lesson.id not in known]
        if not fresh:
            return
        data = _StoreFile(version=STORE_VERSION, lessons=[*current, *fresh])
        write_atomic(self.path, data.model_dump_json(indent=2) + "\n")
        logger.info("stored %d lesson(s) in %s", len(fresh), self.path)


def lessons_text(path: str, *, domain: str, tree: HarnessTree, limit: int) -> str:
    """The relevant lessons rendered for the proposers ("" when `path` is "" or none match)."""
    if not path:
        return ""
    found = LessonsStore(path).relevant(domain=domain, tree=tree, limit=limit)
    return "\n".join(
        f"- When {x.cue}: {x.lesson} (evidence: {x.evidence}) ({x.confidence})" for x in found
    )


def _anchor(path: str) -> str:
    """The prefix a tree must contain for an edit of `path` to be possible."""
    parsed = parse_path(path)
    if parsed.kind == "flow":
        return f"flow/{parsed.name}/{parsed.node}/"
    if parsed.kind == "prompt":
        return "prompt/"
    return f"{parsed.kind}/{parsed.name}/"


def _applies(lesson: Lesson, tree: HarnessTree) -> bool:
    if not lesson.component_paths:
        return True
    anchors = []
    for path in lesson.component_paths:
        try:
            anchors.append(_anchor(path))
        except OptimiserError:
            continue
    return any(p.startswith(a) for a in anchors for p in tree.files)


class _Row(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    candidates: str = ""
    change: str = ""
    observed: str = ""
    verdict: str


class _Reflection(BaseModel):
    model_config = ConfigDict(extra="ignore")

    rows: list[_Row]
    no_entry: bool = False
    cue: str = ""
    lesson: str = ""
    evidence: str = ""
    confidence: str = "low"
    component_paths: list[str] = []


async def reflect(
    composer: LLMPort,
    *,
    domain: str,
    run: str,
    evidence: str,
    adopted_paths: Collection[str],
    created_at: str,
) -> tuple[Lesson | None, str]:
    """One reflection call -> a lesson, or None and the reason it was not kept.

    Raises:
        BudgetExceeded: a budget hook stopped the call.
        OptimiserError: a fatal model error.
    """
    request = LLMRequest(
        messages=(
            Message(role="system", content=reflect_system()),
            Message(role="user", content=evidence),
        ),
        output_schema=REFLECT_SCHEMA,
        stage="reflect",
    )
    try:
        response = await composer.generate(request)
        reply = _Reflection.model_validate(response.parsed)
    except ModelError as exc:
        raise_if_fatal(exc, "reflection")
        info = ErrorInfo.from_exception(exc)
        logger.warning("reflection failed (%s); no lesson stored", info.kind)
        return None, info.kind
    except ValidationError:
        logger.warning("reflection returned an invalid reply; no lesson stored")
        return None, "invalid reply"
    return _lesson(reply, domain=domain, run=run, adopted=adopted_paths, created_at=created_at)


def _lesson(
    reply: _Reflection, *, domain: str, run: str, adopted: Collection[str], created_at: str
) -> tuple[Lesson | None, str]:
    if reply.no_entry:
        return None, "NO_ENTRY"
    cue, text, evidence = reply.cue.strip(), reply.lesson.strip(), reply.evidence.strip()
    if not (cue and text and evidence):
        return None, "incomplete lesson"
    paths = [p for p in reply.component_paths if _valid(p)]
    rows = [r for r in reply.rows if not paths or r.path in paths]
    if not rows:
        return None, "no evidence rows" + (" for the lesson's component paths" if paths else "")
    if all(r.verdict == "confounded" for r in rows):
        return None, "rests only on confounded rows"
    confidence: Confidence = _CONFIDENCE.get(reply.confidence, "low")
    return (
        Lesson(
            id=f"ls_{created_at[:10]}_{digest([domain, run, cue, text])[:8]}",
            domain=domain,
            created_at=created_at,
            cue=cue,
            lesson=text,
            evidence=evidence,
            confidence=confidence,
            component_paths=paths,
            run=run,
            adopted=any(p in adopted for p in paths),
        ),
        "",
    )


def _valid(path: str) -> bool:
    try:
        parse_path(path)
    except OptimiserError:
        return False
    return True
