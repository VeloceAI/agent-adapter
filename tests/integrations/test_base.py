import asyncio
import importlib.util
from typing import Any

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import (
    ApprovalDenied,
    ApprovalPending,
    GuardrailBlocked,
    MissingDependency,
    RunLimitExceeded,
    ToolArgsInvalid,
    ToolError,
    ToolTimeout,
)
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ErrorInfo
from agent_adapter.integrations.base import (
    MODEL_VISIBLE_ERRORS,
    FlowCompiler,
    Integration,
    IntegrationBase,
    ToolRunner,
    _check_requires,
    _error_payload,
    model_visible_error,
)


def lookup(section: str) -> str:
    """Look up a section."""
    return section


class Plain(IntegrationBase):
    name = "plain"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        return (spec.name, executor)


class WithIds(Plain):
    name = "with_ids"
    call_id_fields = ("call_id", "id")


class Needy(Plain):
    name = "needy"
    requires = ("json", "aa_missing_root.sub", "aa_missing_flat")
    extra = "needy"


def registry() -> Registry:
    reg = Registry()
    reg.add(build_spec(lookup))
    reg.add(build_spec(lookup, name="only_plain", targets="plain"))
    reg.add(build_spec(lookup, name="only_other", targets="other"))
    return reg


def test_export_tools_honours_targets_and_passes_the_executor():
    runner = object()
    exported = Plain().export_tools(registry(), executor=runner)  # type: ignore[arg-type]
    assert exported == [("lookup", runner), ("only_plain", runner)]


def test_export_tools_defaults_executor_to_none():
    assert Plain().export_tools(registry())[0] == ("lookup", None)


def test_default_bind_context_keeps_the_native_object_under_the_integration_name():
    native = {"id": "call_1"}
    ctx = Plain().bind_context(native)
    assert isinstance(ctx, ToolContext)
    assert ctx.extras == {"plain": native}
    assert ctx.call_id == ""


class Obj:
    def __init__(self, **attrs: Any) -> None:
        self.__dict__.update(attrs)


@pytest.mark.parametrize(
    ("native", "expected"),
    [
        ({"call_id": "c1", "id": "fc_1"}, "c1"),
        ({"id": "fc_1"}, "fc_1"),
        ({"call_id": "", "id": "fc_1"}, "fc_1"),
        (Obj(call_id="c2", id="x"), "c2"),
        (Obj(id="x"), "x"),
        ({"call_id": None}, ""),
        ({"call_id": 7}, ""),
        ("not a call", ""),
        (None, ""),
    ],
)
def test_bind_context_reads_the_first_non_empty_call_id_field(native, expected):
    ctx = WithIds().bind_context(native)
    assert ctx.call_id == expected
    assert ctx.extras == {"with_ids": native}


def test_default_bind_approval_returns_none():
    assert Plain().bind_approval(object()) is None  # type: ignore[arg-type]


def test_integration_base_cannot_be_instantiated_without_export_tool():
    class Incomplete(IntegrationBase):
        name = "incomplete"

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_subclasses_satisfy_the_integration_protocol():
    assert isinstance(Plain(), Integration)
    assert not isinstance(Plain(), FlowCompiler)
    assert not isinstance(object(), Integration)


def test_flow_compiler_protocol_checks_compile_flow():
    class Compiles(Plain):
        def compile_flow(self, flow: Any, *, executor: Any = None, **options: Any) -> Any:
            return flow

    assert isinstance(Compiles(), FlowCompiler)


def test_require_passes_when_every_module_is_importable():
    Plain().require()
    _check_requires(Plain())


def test_require_names_missing_modules_and_the_extra():
    with pytest.raises(MissingDependency) as info:
        Needy().require()
    assert str(info.value) == (
        "integration 'needy' needs aa_missing_root.sub, aa_missing_flat: "
        'pip install "agent-adapter[needy]"'
    )
    assert isinstance(info.value, ImportError)


def test_require_without_an_extra_says_to_install_the_module():
    class NoExtra(Plain):
        name = "no_extra"
        requires = ("aa_missing_flat",)

    with pytest.raises(MissingDependency, match="install aa_missing_flat"):
        _check_requires(NoExtra())


def test_require_counts_a_loaded_module_without_a_spec_as_present(monkeypatch):
    import sys
    import types

    module = types.ModuleType("aa_specless")
    module.__spec__ = None
    monkeypatch.setitem(sys.modules, "aa_specless", module)
    with pytest.raises(ValueError):
        importlib.util.find_spec("aa_specless")

    class Specless(Plain):
        name = "specless"
        requires = ("aa_specless",)

    Specless().require()


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (ToolArgsInvalid("t", [{"loc": ("a",), "msg": "bad", "type": "x"}]), "tool-args-invalid"),
        (ToolError("no stock", tool="t"), "tool-error"),
        (ApprovalDenied("t", reason="nope"), "approval-denied"),
        (GuardrailBlocked("pii"), "guardrail-blocked"),
        (RunLimitExceeded("t", "max_calls"), "run-limit-exceeded"),
        (ToolTimeout("t", 1.0), "tool-timeout"),
    ],
)
def test_model_visible_error_has_kind_and_message_only(exc, kind):
    payload = model_visible_error(exc)
    assert payload == {"error": {"kind": kind, "message": str(exc)}}


def test_model_visible_error_hides_foreign_exception_text():
    payload = model_visible_error(KeyError("secret-key"))
    assert payload == {"error": {"kind": "internal-error", "message": "KeyError"}}


def test_model_visible_errors_are_the_contract_list():
    assert (
        ToolArgsInvalid,
        ToolError,
        ApprovalDenied,
        GuardrailBlocked,
        RunLimitExceeded,
    ) == MODEL_VISIBLE_ERRORS
    assert not issubclass(ApprovalPending, MODEL_VISIBLE_ERRORS)
    assert not issubclass(ToolTimeout, MODEL_VISIBLE_ERRORS)


class RecordingRunner:
    """A ToolRunner that records its calls and returns or raises `outcome`."""

    def __init__(self, outcome: Any = "ok") -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, Any, ToolContext | None, str | None]] = []

    async def run_spec(self, spec, args=None, ctx=None, *, call_id=None):
        self.calls.append((spec.name, args, ctx, call_id))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


async def test_run_bound_binds_the_context_and_passes_the_native_call_id():
    runner, native = RecordingRunner("value"), {"call_id": "c1"}
    spec = build_spec(lookup)
    assert await WithIds()._run_bound(native, spec, {"section": "80C"}, runner) == "value"
    [(name, args, ctx, call_id)] = runner.calls
    assert (name, args, call_id) == ("lookup", {"section": "80C"}, "c1")
    assert ctx is not None and ctx.call_id == "c1" and ctx.extras == {"with_ids": native}


async def test_run_bound_sends_none_when_the_native_call_has_no_id():
    runner = RecordingRunner()
    await Plain()._run_bound({}, build_spec(lookup), {"section": "x"}, runner)
    assert runner.calls[0][3] is None


@pytest.mark.parametrize(
    "exc", [ToolError("no stock", tool="t"), ToolArgsInvalid("t", []), ApprovalDenied("t")]
)
async def test_run_bound_returns_model_visible_errors(exc):
    result = await Plain()._run_bound({}, build_spec(lookup), {}, RecordingRunner(exc))
    assert result == model_visible_error(exc)


async def test_run_bound_formats_model_visible_errors_with_the_visible_hook():
    exc = ToolError("no stock", tool="t")
    result = await Plain()._run_bound(
        {}, build_spec(lookup), {}, RecordingRunner(exc), visible=lambda e: f"text: {e}"
    )
    assert result == "text: no stock"


@pytest.mark.parametrize(
    "exc",
    [ToolTimeout("t", 1.0), ApprovalPending("apr_1"), KeyError("k"), asyncio.CancelledError()],
)
async def test_run_bound_propagates_every_other_exception(exc):
    with pytest.raises(type(exc)):
        await Plain()._run_bound({}, build_spec(lookup), {}, RecordingRunner(exc))


def test_error_payload_is_shared_by_model_visible_error():
    info = ErrorInfo.from_exception(ToolError("boom"))
    assert _error_payload(info) == {"error": {"kind": "tool-error", "message": "boom"}}
