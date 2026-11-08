import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_adapter.evals import assert_eval, pytest_plugin
from agent_adapter.evals.graders import exact
from tests.evals.fakes import case

ROOT = Path(__file__).resolve().parents[2]
CASES = [
    {"id": "shipped", "input": "1042", "expected": "shipped", "tags": ["orders"]},
    {"id": "packed", "input": "1043", "expected": "packed", "tags": ["orders"]},
    {"id": "hello", "input": "hi", "expected": "hello", "tags": ["chat"]},
]


class FakeMetafunc:
    """The parts of pytest.Metafunc the plugin reads."""

    def __init__(self, folder, marker, fixtures=("eval_case",)):
        self.fixturenames = list(fixtures)
        self.definition = SimpleNamespace(
            nodeid="test_x.py::test_x",
            path=folder / "test_x.py",
            get_closest_marker=lambda name: marker,
        )
        self.parametrized = None

    def parametrize(self, name, values, ids):
        self.parametrized = (name, values, ids)


def mark(*args, **kwargs):
    return SimpleNamespace(args=args, kwargs=kwargs)


def write_cases(folder):
    path = folder / "cases.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in CASES), encoding="utf-8")
    return path


def test_the_marker_parametrizes_one_test_per_case(tmp_path):
    write_cases(tmp_path)
    meta = FakeMetafunc(tmp_path, mark("cases.jsonl"))
    pytest_plugin.pytest_generate_tests(meta)
    name, values, ids = meta.parametrized
    assert name == "eval_case" and ids == ["shipped", "packed", "hello"]
    assert values[0].expected == "shipped"


@pytest.mark.parametrize(
    ("tags", "ids"), [("chat", ["hello"]), (["orders"], ["shipped", "packed"])]
)
def test_the_marker_filters_by_tag(tmp_path, tags, ids):
    write_cases(tmp_path)
    meta = FakeMetafunc(tmp_path, mark(path="cases.jsonl", tags=tags))
    pytest_plugin.pytest_generate_tests(meta)
    assert meta.parametrized[2] == ids


@pytest.mark.parametrize(
    ("marker", "message"),
    [
        (None, "needs @pytest.mark.eval_cases"),
        (mark(), "use @pytest.mark.eval_cases"),
        (mark("cases.jsonl", other=1), "use @pytest.mark.eval_cases"),
        (mark("cases.jsonl", tags="none"), "no cases with tags"),
    ],
)
def test_bad_markers_are_usage_errors(tmp_path, marker, message):
    write_cases(tmp_path)
    with pytest.raises(pytest.UsageError, match=message):
        pytest_plugin.pytest_generate_tests(FakeMetafunc(tmp_path, marker))


def test_tests_without_the_fixture_are_left_alone(tmp_path):
    meta = FakeMetafunc(tmp_path, None, fixtures=("other",))
    pytest_plugin.pytest_generate_tests(meta)
    assert meta.parametrized is None


def test_configure_registers_the_marker():
    lines = []
    config = SimpleNamespace(addinivalue_line=lambda key, line: lines.append((key, line)))
    pytest_plugin.pytest_configure(config)
    assert lines[0][0] == "markers" and lines[0][1].startswith("eval_cases(path, tags=None)")


def test_assert_eval_returns_the_result_or_fails_with_reasons():
    passed = assert_eval(lambda q: "shipped", case(), [exact()])
    assert passed.passed and passed.run_id
    with pytest.raises(AssertionError, match=r"eval case c1 failed \(run run_\w+\): exact: not"):
        assert_eval(lambda q: "lost", case(), [exact()])

    def broken(q):
        raise RuntimeError("x")

    with pytest.raises(AssertionError, match="internal-error: RuntimeError"):
        assert_eval(broken, case(), [exact()])


def test_the_plugin_runs_a_case_file_in_a_real_pytest(tmp_path):
    write_cases(tmp_path)
    (tmp_path / "test_agent.py").write_text(
        textwrap.dedent(
            """
            import pytest

            from agent_adapter.evals.graders import exact
            from agent_adapter.evals import assert_eval

            ANSWERS = {"1042": "shipped", "1043": "packed", "hi": "hi there"}


            @pytest.mark.eval_cases("cases.jsonl")
            def test_agent(eval_case):
                assert_eval(ANSWERS.get, eval_case, [exact()])
            """
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(ROOT)}
    env.pop("COVERAGE_PROCESS_START", None)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "agent_adapter.evals.pytest_plugin",
            "-p",
            "no:cacheprovider",
            "--strict-markers",
            str(tmp_path / "test_agent.py"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "1 failed, 2 passed" in done.stdout, done.stdout + done.stderr
    assert "test_agent[hello]" in done.stdout and "exact: not the expected answer" in done.stdout
