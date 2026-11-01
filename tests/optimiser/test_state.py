import logging
import os
import subprocess
import sys
from pathlib import Path

from agent_adapter.optimiser._state import fingerprint
from agent_adapter.optimiser.model import EvalCase, HarnessTree, OptimiserConfig, SurfaceCaps


class Thing:
    """A non-JSON case input whose default repr holds a memory address."""

    def __str__(self) -> str:
        return "thing"


def make() -> str:
    return fingerprint(
        tree=HarnessTree(files={"prompt/system": "s"}),
        caps=SurfaceCaps(max_chars={}, kinds=frozenset({"prompt"})),
        cases=[EvalCase(id="a", input=Thing())],
        config=OptimiserConfig(),
        domain="d",
        models={"model": "x:y"},
    )


def test_fingerprint_is_stable_for_inputs_that_are_not_json():
    assert make() == make()  # a fresh process builds fresh objects: the digest must not change


ROOT = Path(__file__).resolve().parents[2]


class Plain:
    """No `__str__`: its default text holds a memory address."""


def make_plain() -> str:
    return fingerprint(
        tree=HarnessTree(files={"prompt/system": "s"}),
        caps=SurfaceCaps(max_chars={}, kinds=frozenset({"prompt"})),
        cases=[EvalCase(id="a", input={"obj": Plain()})],
        config=OptimiserConfig(),
        domain="d",
        models={"model": "x:y"},
    )


def test_a_plain_object_input_uses_its_type_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_adapter.optimiser._state"):
        assert make_plain() == make_plain()
    warnings = [r for r in caplog.records if "not JSON-stable" in r.getMessage()]
    assert len(warnings) == 2  # one per fingerprint


def test_the_fingerprint_is_the_same_in_two_processes():
    script = "from tests.optimiser.test_state import make_plain; print(make_plain())"
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    outs = [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            cwd=ROOT,
            env=env,
            timeout=60,
        ).stdout.strip()
        for _ in range(2)
    ]
    assert outs[0] and outs[0] == outs[1]
