"""Model errors that end the run: a bad key or a rejected request will not fix itself."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NoReturn

from agent_adapter.core.errors import (
    ModelAuthError,
    ModelBadRequest,
    ModelError,
    ModelRateLimited,
    ModelTimeout,
    ModelUnavailable,
    OptimiserError,
)
from agent_adapter.core.types import ErrorInfo
from agent_adapter.optimiser.model import Candidate, CaseRun, ProbeReport

logger = logging.getLogger(__name__)

FATAL_ERRORS: tuple[type[ModelError], ...] = (ModelAuthError, ModelBadRequest)
# Kinds that, when every case run of a stage shows them, mean the model, not the harness, failed.
_SYSTEMIC: tuple[type[ModelError], ...] = (
    *FATAL_ERRORS,
    ModelUnavailable,
    ModelRateLimited,
    ModelTimeout,
)
SYSTEMIC_KINDS = frozenset(cls.kind for cls in _SYSTEMIC)


def raise_if_fatal(exc: ModelError, stage: str) -> None:
    """Re-raise a fatal model error as `OptimiserError`; other model errors return.

    Raises:
        OptimiserError: `exc` is a `ModelAuthError` or `ModelBadRequest`.
    """
    if isinstance(exc, FATAL_ERRORS):
        info = ErrorInfo.from_exception(exc)
        logger.error("%s: fatal model error (%s); stopping the run", stage, info.kind)
        raise OptimiserError(f"{stage}: {info.kind}: {info.message}") from exc


def error_kind(error: str) -> str:
    """The kind prefix of a `CaseRun.error` / `Candidate.error` ("kind: message")."""
    return error.split(":", 1)[0].strip()


def check_probe(report: ProbeReport) -> None:
    """Stop when one probe shows an unusable agent model.

    An auth failure in any case run is fatal. In the base probe a rejected request is fatal
    only when every base case failed with it (otherwise those cases are failed base runs),
    and a base probe with no successful case is fatal too.

    Raises:
        OptimiserError: naming the probe and the case.
    """
    auth = next((r for r in report.runs if error_kind(r.error) == ModelAuthError.kind), None)
    if auth is not None:
        _fatal_run(report, auth)
    if report.label != "base" or not report.runs:
        return
    if all(error_kind(r.error) == ModelBadRequest.kind for r in report.runs):
        _fatal_run(report, report.runs[0])
    if not any(r.ok for r in report.runs):
        kinds = ", ".join(sorted({error_kind(r.error) for r in report.runs}))
        logger.error("no base case run succeeded (%s); stopping the run", kinds)
        raise OptimiserError(f"no base case run succeeded: {report.runs[0].error}")


def _fatal_run(report: ProbeReport, run: CaseRun) -> NoReturn:
    logger.error("probe %s, case %s: fatal model error; stopping", report.label, run.case_id)
    raise OptimiserError(f"probe {report.label}, case {run.case_id}: {run.error}")


def check_stage(reports: Sequence[ProbeReport], stage: str) -> None:
    """Stop when every case run of every probe in a stage failed on a model error.

    Raises:
        OptimiserError: the agent model rejects every changed harness (systemic failure).
    """
    runs = [run for report in reports for run in report.runs]
    if runs and all(error_kind(r.error) in SYSTEMIC_KINDS for r in runs):
        kinds = ", ".join(sorted({error_kind(r.error) for r in runs}))
        logger.error("%s: every candidate case run failed (%s); stopping the run", stage, kinds)
        raise OptimiserError(
            f"{stage}: every candidate case run failed with a model error: {kinds}"
        )


TIMEOUT_KIND = "case-timeout"  # a case run (or its judging) passed `case_timeout_s`


def run_failures(report: ProbeReport) -> str:
    """`"know1: 3/3 case runs failed: model-bad-request"` for model errors and timeouts, else ""."""
    failed = [r for r in report.runs if _counted(error_kind(r.error))]
    if not failed:
        return ""
    kinds = ", ".join(sorted({error_kind(r.error) for r in failed}))
    return f"{report.label}: {len(failed)}/{len(report.runs)} case runs failed: {kinds}"


def _counted(kind: str) -> bool:
    return kind.startswith("model-") or kind == TIMEOUT_KIND


def check_proposers(candidates: Sequence[Candidate]) -> None:
    """Stop when every proposer of a stage fell back because its model call failed.

    Raises:
        OptimiserError: no proposer produced a reply.
    """
    failed = [c for c in candidates if c.fell_back and error_kind(c.error).startswith("model-")]
    if candidates and len(failed) == len(candidates):
        logger.error("every proposer failed on a model error; stopping the run")
        raise OptimiserError(f"every proposer failed on a model error: {failed[0].error}")
