import pytest

from agent_adapter.core import errors as e
from agent_adapter.server.problems import (
    PROBLEM_MEDIA_TYPE,
    Problem,
    problem,
    problem_for,
    status_for,
)


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (e.ToolNotFound("no tool named 'x'"), 404),
        (e.ApprovalNotFound("apr_1"), 404),
        (e.RunLimitExceeded("t", "once_per_run"), 409),
        (e.ApprovalAlreadyDecided("apr_1", status="approved"), 409),
        (e.ToolArgsInvalid("t", [{"type": "missing", "loc": ("a",), "msg": "required"}]), 422),
        (e.BudgetExceeded(1.0, 2.0), 429),
        (e.ToolUsageError("bad"), 400),
        (e.ToolTimeout("t", 1.5), 504),
        (e.Unauthorized("who?"), 401),
        (e.ToolFailed("t"), 502),
        (e.ToolError("nope", tool="t"), 502),
        (e.ModelRateLimited("slow", provider="openai"), 502),
        (e.ModelError("x", provider="openai"), 502),
        (e.Forbidden(), 403),
        (e.ApprovalDenied("t", reason="no"), 403),
        (e.GuardrailBlocked("pii", tool="t"), 403),
        (e.ToolDefinitionError("bad"), 500),
        (e.SchemaNotPortable(["x"]), 500),
        (e.ConfigError("bad"), 500),
        (e.AdapterError("boom"), 500),
        (e.PlanInvalid(["x"]), 500),
        (RuntimeError("password=hunter2"), 500),
    ],
)
def test_each_kind_maps_to_its_status(exc: BaseException, status: int):
    found = problem_for(exc)
    assert found.status == status
    assert found.type.startswith("urn:agent-adapter:error:")
    assert found.title


def test_unknown_exceptions_show_only_their_class():
    found = problem_for(RuntimeError("password=hunter2"))
    assert found.detail == "RuntimeError"
    assert "hunter2" not in found.model_dump_json()


def test_tool_args_invalid_carries_errors_without_input_values():
    exc = e.ToolArgsInvalid(
        "t", [{"type": "int_parsing", "loc": ("a",), "msg": "bad int", "input": "secret"}]
    )
    found = problem_for(exc)
    assert found.type == "urn:agent-adapter:error:tool-args-invalid"
    assert found.model_extra == {"tool": "t", "errors": exc.errors}
    assert "secret" not in found.model_dump_json()


def test_approval_errors_carry_the_approval_id():
    found = problem_for(e.ApprovalAlreadyDecided("apr_9", status="denied"))
    assert found.model_extra == {"approval_id": "apr_9"}


def test_route_problems_and_statuses():
    assert status_for("payload-too-large") == 413
    assert status_for("model-timeout") == 502
    assert status_for("whatever") == 500
    made = problem("not-found", "no route", path="/x")
    assert (made.status, made.title, made.model_extra) == (404, "Not found", {"path": "/x"})
    assert PROBLEM_MEDIA_TYPE == "application/problem+json"
    assert Problem(type="t", title="x", status=400).detail == ""
