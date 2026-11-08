"""Opt-in pytest plugin (`-p agent_adapter.evals.pytest_plugin`): one test per eval case."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

    from agent_adapter.evals.model import EvalCase

MARKER = "eval_cases"
FIXTURE = "eval_case"
_HELP = (
    f"{MARKER}(path, tags=None): run the test once per eval case in path (.jsonl, .json or "
    f".csv, relative to the test file), optionally only cases with one of tags; the test "
    f"takes the {FIXTURE} fixture"
)


def pytest_configure(config: pytest.Config) -> None:
    """Register the marker, so `--strict-markers` accepts it."""
    config.addinivalue_line("markers", _HELP)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize `eval_case` from the test's `eval_cases` marker, one test id per case.

    Raises:
        pytest.UsageError: the test takes `eval_case` without the marker, or the marker is bad.
        ConfigError: the case file cannot be loaded.
    """
    if FIXTURE not in metafunc.fixturenames:
        return
    marker = metafunc.definition.get_closest_marker(MARKER)
    if marker is None:
        import pytest

        raise pytest.UsageError(
            f"{metafunc.definition.nodeid}: {FIXTURE} needs @pytest.mark.{MARKER}(path)"
        )
    cases = _marked_cases(metafunc, marker.args, marker.kwargs)
    metafunc.parametrize(FIXTURE, cases, ids=[case.id for case in cases])


def _marked_cases(
    metafunc: pytest.Metafunc, args: Sequence[Any], kwargs: Mapping[str, Any]
) -> list[EvalCase]:
    import pytest

    from agent_adapter.evals.cases import load_cases

    unknown = sorted(set(kwargs) - {"path", "tags"})
    paths = [*args, *([kwargs["path"]] if "path" in kwargs else [])]
    if unknown or len(paths) != 1:
        raise pytest.UsageError(f"{metafunc.definition.nodeid}: use @pytest.mark.{MARKER}(path)")
    path = metafunc.definition.path.parent / paths[0]  # relative to the test file
    tags = kwargs.get("tags")
    wanted = None if tags is None else {tags} if isinstance(tags, str) else set(tags)
    cases = [case for case in load_cases(path) if wanted is None or wanted.intersection(case.tags)]
    if not cases:
        raise pytest.UsageError(f"{metafunc.definition.nodeid}: no cases with tags {tags!r}")
    return cases
