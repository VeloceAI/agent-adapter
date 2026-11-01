"""Resumable run state: finished stages saved atomically to `<out_dir>/state.json`."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from agent_adapter.cache.keys import digest
from agent_adapter.optimiser._files import write_atomic
from agent_adapter.optimiser.model import (
    Candidate,
    DecisionLine,
    EvalCase,
    HarnessTree,
    Lesson,
    OptimiserConfig,
    ProbeReport,
    SurfaceCaps,
)

logger = logging.getLogger(__name__)

STATE_VERSION = 1
STATE_FILE = "state.json"

# Config fields that do not change what a stage produces (limits, paths, concurrency).
_NEUTRAL = {"budget", "out_dir", "rollout_concurrency", "lessons_path"}


class ComposeRecord(BaseModel):
    """A finished compose stage."""

    model_config = ConfigDict(frozen=True)

    decisions: list[DecisionLine]
    merge_calls: int
    assessed: dict[str, str]  # candidate label -> evidence line


class SelectionRecord(BaseModel):
    """A finished selection."""

    model_config = ConfigDict(frozen=True)

    chosen: str | None
    why: str
    calls: int


class ReflectionRecord(BaseModel):
    """A finished reflection: the lessons stored (if any) and why none was."""

    model_config = ConfigDict(frozen=True)

    lessons: list[Lesson]
    why: str
    calls: int = 0  # reflection calls made (0 or 1)


class RunState(BaseModel):
    """Everything a resumed run can reuse instead of recomputing."""

    version: int = STATE_VERSION
    fingerprint: str
    run_id: str
    candidates: dict[str, Candidate] = {}
    probes: dict[str, ProbeReport] = {}
    composes: dict[str, ComposeRecord] = {}
    selection: SelectionRecord | None = None
    reflection: ReflectionRecord | None = None
    case_runs: int = 0
    proposer_calls: int = 0
    merge_calls: int = 0
    usd: float = 0.0
    wall_s: float = 0.0


def fingerprint(
    *,
    tree: HarnessTree,
    caps: SurfaceCaps,
    cases: list[EvalCase],
    config: OptimiserConfig,
    domain: str,
    models: dict[str, str],
) -> str:
    """Digest of every input that shapes a stage's result; another digest starts afresh."""
    material: dict[str, Any] = {
        "tree": tree.files,
        "caps": caps.model_dump(mode="json"),
        "cases": _cases_material(cases),
        "config": config.model_dump(mode="json", exclude=_NEUTRAL),
        "domain": domain,
        "models": models,
    }
    return digest(material)


def _cases_material(cases: list[EvalCase]) -> list[str]:
    """Digests that are the same in every process.

    A value JSON cannot hold goes in as its `str`, or as its type when that text holds a memory
    address (one WARNING): such inputs then only resume by type, so prefer JSON inputs.
    """
    unstable: list[str] = []

    def stable(value: Any) -> str:
        text = str(value)
        if " at 0x" not in text:
            return text
        unstable.append(f"{type(value).__module__}.{type(value).__qualname__}")
        return unstable[-1]

    material = [digest(case.model_dump(mode="json", fallback=stable)) for case in cases]
    if unstable:
        logger.warning(
            "case input not JSON-stable; fingerprint uses its type (%s)",
            ", ".join(sorted(set(unstable))),
        )
    return material


def load_state(out_dir: str | os.PathLike[str], expected: str) -> RunState | None:
    """The saved state when its fingerprint matches; None when absent, stale or unreadable."""
    path = Path(out_dir) / STATE_FILE
    try:
        text = path.read_text("utf-8")
    except FileNotFoundError:
        return None
    try:
        state = RunState.model_validate_json(text)
    except ValidationError:
        logger.warning("ignoring unreadable optimiser state %s; starting afresh", path)
        return None
    if state.version != STATE_VERSION or state.fingerprint != expected:
        logger.info("optimiser state %s is for other inputs; starting afresh", path)
        return None
    logger.info(
        "resuming optimiser run %s: %d candidates, %d probes restored",
        state.run_id,
        len(state.candidates),
        len(state.probes),
    )
    return state


def save_state(out_dir: str | os.PathLike[str], state: RunState) -> None:
    """Write the state atomically."""
    write_atomic(Path(out_dir) / STATE_FILE, state.model_dump_json() + "\n")
