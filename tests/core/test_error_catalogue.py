import pickle

import pytest

from agent_adapter.core import errors as E

KINDS = {
    E.AdapterError: "internal-error",
    E.ToolDefinitionError: "tool-definition-invalid",
    E.DuplicateTool: "duplicate-tool",
    E.SchemaNotPortable: "schema-not-portable",
    E.ToolNotFound: "tool-not-found",
    E.ToolArgsInvalid: "tool-args-invalid",
    E.ToolUsageError: "tool-usage-error",
    E.MissingDependency: "missing-dependency",
    E.ToolError: "tool-error",
    E.ToolTimeout: "tool-timeout",
    E.ToolFailed: "tool-failed",
    E.RunLimitExceeded: "run-limit-exceeded",
    E.GuardrailBlocked: "guardrail-blocked",
    E.ApprovalDenied: "approval-denied",
    E.ApprovalPending: "approval-pending",
    E.ApprovalNotFound: "approval-not-found",
    E.ApprovalAlreadyDecided: "approval-already-decided",
    E.BudgetExceeded: "budget-exceeded",
    E.PlanInvalid: "plan-invalid",
    E.FlowError: "flow-error",
    E.FlowDefinitionError: "flow-definition-invalid",
    E.FlowCompileError: "flow-compile-failed",
    E.FlowLoopExceeded: "flow-loop-exceeded",
    E.ModelError: "model-error",
    E.ModelRateLimited: "model-rate-limited",
    E.ModelUnavailable: "model-unavailable",
    E.ModelTimeout: "model-timeout",
    E.ModelAuthError: "model-auth-failed",
    E.ModelBadRequest: "model-bad-request",
    E.ModelContextOverflow: "model-context-overflow",
    E.ModelContentFiltered: "model-content-filtered",
    E.ModelOutputInvalid: "model-output-invalid",
    E.ConfigError: "config-invalid",
    E.IntegrationError: "integration-error",
    E.AuthError: "auth-error",
    E.Unauthorized: "unauthorized",
    E.Forbidden: "forbidden",
    E.OptimiserError: "optimiser-error",
}
RETRYABLE = {E.ToolTimeout, E.ModelRateLimited, E.ModelUnavailable, E.ModelTimeout}

SAMPLES = [
    E.AdapterError("boom"),
    E.ToolArgsInvalid("t", [{"type": "x", "loc": ("a",), "msg": "bad"}]),
    E.SchemaNotPortable(["p"], tool="t"),
    E.ToolError("no rows", tool="t", details={"rows": 0}),
    E.ToolTimeout("t", 1.5, call_id="call_1"),
    E.ToolFailed("t", call_id="call_1", attempts=3),
    E.RunLimitExceeded("t", "max_calls", run_id="run_1"),
    E.GuardrailBlocked("pii", tool="t", stage="result"),
    E.ApprovalDenied("t", call_id="c", reason="no"),
    E.ApprovalPending("apr_1", tool="t", run_id="r", call_id="c"),
    E.ApprovalNotFound("apr_1"),
    E.ApprovalAlreadyDecided("apr_1", status="approved"),
    E.BudgetExceeded(1.0, 1.25, scope="thread", run_id="r"),
    E.PlanInvalid(["cycle a -> b"]),
    E.FlowError("bad", flow="f", node="n"),
    E.FlowDefinitionError(["no entry"], flow="f"),
    E.FlowCompileError("langgraph", ["no Join"], flow="f"),
    E.FlowLoopExceeded("f", "n", 3),
    E.ModelRateLimited("slow down", provider="openai", model="m", status_code=429, retry_after=2.0),
    E.ConfigError("bad"),
    E.IntegrationError("x", integration="mcp"),
    E.Unauthorized("who"),
    E.OptimiserError("x"),
]


@pytest.mark.parametrize("cls", list(KINDS))
def test_every_error_has_its_kind_and_retry_flag(cls):
    assert cls.kind == KINDS[cls]
    assert cls.retryable is (cls in RETRYABLE)
    assert issubclass(cls, E.AdapterError)


def test_kinds_are_unique():
    assert len(set(KINDS.values())) == len(KINDS)


def test_mixin_bases():
    assert issubclass(E.ToolTimeout, TimeoutError)
    assert issubclass(E.ConfigError, ValueError)
    assert issubclass(E.FlowCompileError, E.FlowDefinitionError)
    assert issubclass(E.FlowLoopExceeded, E.FlowError)
    assert issubclass(E.Forbidden, E.AuthError)
    for cls in (E.ModelRateLimited, E.ModelOutputInvalid, E.ModelAuthError):
        assert issubclass(cls, E.ModelError)


@pytest.mark.parametrize("error", SAMPLES, ids=lambda e: type(e).__name__)
def test_errors_survive_pickling(error):
    copy = pickle.loads(pickle.dumps(error))
    assert type(copy) is type(error)
    assert str(copy) == str(error)
    assert copy.__dict__ == error.__dict__


def test_tool_failed_cause_type_names_the_cause():
    try:
        try:
            raise KeyError("k")
        except KeyError as cause:
            raise E.ToolFailed("t", attempts=2) from cause
    except E.ToolFailed as exc:
        assert exc.cause_type == "KeyError"
        assert exc.attempts == 2
        assert "2 attempts" in str(exc)
    assert E.ToolFailed("t").cause_type == ""
    assert "1 attempt" in str(E.ToolFailed("t"))


def test_qualified_name_keeps_module_for_non_builtins():
    assert E.qualified_name(E.ToolError) == "agent_adapter.core.errors.ToolError"
    assert E.qualified_name(ValueError) == "ValueError"


def test_messages_and_attributes():
    timeout = E.ToolTimeout("search", 2.0, call_id="c1")
    assert (timeout.tool, timeout.timeout, timeout.call_id) == ("search", 2.0, "c1")
    assert str(timeout) == "tool 'search' timed out after 2s"
    assert E.ToolError("x").details == {}
    assert str(E.ApprovalDenied("t")) == "approval denied for tool 't'"
    assert str(E.ApprovalDenied("t", reason="risky")).endswith(": risky")
    assert "for tool 't'" in str(E.ApprovalPending("a", tool="t"))
    assert "for tool" not in str(E.ApprovalPending("a"))
    assert str(E.GuardrailBlocked("pii")) == "guardrail blocked args: pii"
    assert str(E.BudgetExceeded(1, 2)) == "run budget $1.0000 exceeded: spent $2.0000"
    assert str(E.FlowDefinitionError(["a", "b"])) == "invalid flow: a; b"
    compile_error = E.FlowCompileError("adk", ["no Map"], flow="f")
    assert compile_error.problems == ["no Map"] and compile_error.target == "adk"
    assert "adk: no Map" in str(compile_error) and compile_error.flow == "f"
    loop = E.FlowLoopExceeded("f", "n", 2)
    assert (loop.flow, loop.node, loop.limit) == ("f", "n", 2)
    model = E.ModelError("x", provider="gemini")
    assert (model.model, model.status_code, model.retry_after, model.request_id) == (
        "",
        None,
        None,
        "",
    )
    assert str(E.AuthError()) == ""
    assert E.ApprovalAlreadyDecided("a", status="denied").status == "denied"
