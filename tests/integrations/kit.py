"""Contract kit: the checks C1-C8 every integration passes (contract §14.3).

Use it from `tests/integrations/test_<name>_contract.py`:

    globals().update((item.__name__, item) for item in contract_tests(MyHarness))

The list holds the module-scoped harness fixture, then the tests C1..C8. The kit is typed
against `ToolRunner` only. By default every check runs against `FakeRunner`; pass
`runner_factory=Executor` to run them through the real executor (its constructor takes the
`hooks=` and `approval=` keywords the kit uses).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import itertools
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any, Literal, Protocol

import pydantic
import pytest
import pytest_asyncio

from agent_adapter.core.errors import ToolTimeout
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.registry import Registry
from agent_adapter.core.tool import ToolSpec, build_spec
from agent_adapter.core.types import ApprovalDecision
from agent_adapter.integrations.base import Integration, IntegrationBase, ToolRunner
from agent_adapter.integrations.discovery import available_integrations, get_integration
from tests.fakes.approval import RecordingApproval, ScriptedApproval
from tests.fakes.hooks import RecordingHook
from tests.integrations._kit_support import WAIT_LIMIT_S, FakeRunner, Probe, wait_until

__all__ = ["CHECKS", "FakeRunner", "Harness", "Probe", "contract_tests", "harness_session"]

CANCEL_GRACE_S = 1.0  # how long a cancelled tool may take to observe its CancelledError


class Harness(Protocol):
    """Drives one integration's framework for the checks.

    Optional members, read with `getattr`:

    - `native_call_ids: bool = True` — False when the framework has no call id of its own
      (C5 then only asks for a non-empty `ctx.call_id`).
    - `cancels_tool: bool = True` — False for sync-only bridges (`run_sync` in a thread):
      C7 then asserts only that the awaiter sees `CancelledError` and the call ends; the
      tool's thread may run on until the kit releases it.
    - `approval_port() -> ApprovalPort | None` — in native mode, the port the runner gets
      (wrapped in a `RecordingApproval`).
    - `native_uses_port: bool = False` — in native mode, C6 asserts the port was asked
      with the original arguments (at least once: a resume-based gate asks again).
    - `port_asks: int | None = None` — with `native_uses_port`, the exact number of asks.
    - `export_all(registry, executor) -> list` — the module's `*_tools(registry, …)`;
      C1 checks it honours `targets`, as it does `IntegrationBase.export_tools`.
    - `approve_flow(..., *, port=None)` — receives the `RecordingApproval` when it accepts it.
    - `__aenter__` / `__aexit__` — entered once per `contract_tests` call (per module).
    """

    name: str  # integration name
    edits_supported: bool  # approval may edit args
    approval: Literal["native", "port"]

    def export(self, spec: ToolSpec, executor: ToolRunner) -> Any: ...

    async def call(self, native: Any, args: dict[str, Any], *, call_id: str) -> Any:
        """Run `native` through the framework's own invocation path."""

    def value(self, raw: Any) -> Any:
        """Framework result -> plain value."""

    def is_error(self, raw: Any) -> bool:
        """Whether `raw` (a result, or an exception the call raised) is a conventional error."""

    async def approve_flow(
        self, native: Any, args: dict[str, Any], decision: ApprovalDecision
    ) -> Any:
        """Run a `confirm` tool through the framework's approval path with `decision`."""

    def describe(self, native: Any) -> tuple[str, str, Mapping[str, Any]]:
        """The exported (name, description, parameter schema)."""

    def expected_schema(self, spec: ToolSpec) -> dict[str, Any]:
        """The target's transform of `spec.json_schema()`."""


RunnerFactory = Callable[..., ToolRunner]  # called with keyword `hooks=` and/or `approval=`
Check = Callable[[Harness, RunnerFactory], Awaitable[None]]


@contextlib.asynccontextmanager
async def harness_session(harness_factory: Callable[[], Harness]) -> AsyncIterator[Harness]:
    """One harness; entered and exited around the block when it is an async context manager."""
    harness = harness_factory()
    if hasattr(harness, "__aenter__") and hasattr(harness, "__aexit__"):
        async with harness as entered:  # type: ignore[attr-defined]
            yield entered if entered is not None else harness
    else:
        yield harness


async def _outcome(call: Awaitable[Any]) -> tuple[Any, Exception | None]:
    """(result, None) or (None, exception); BaseExceptions such as CancelledError propagate."""
    try:
        return await call, None
    except Exception as exc:
        return None, exc


def _is_error(harness: Harness, raw: Any, exc: Exception | None) -> bool:
    return harness.is_error(exc if exc is not None else raw)


def _chain(exc: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        current = current.__cause__ or current.__context__
    return seen


def _task_outcome(task: asyncio.Future[Any]) -> str:
    if task.cancelled():
        return "CancelledError"
    error = task.exception()
    return repr(error) if error is not None else f"result {task.result()!r}"


async def check_c1(harness: Harness, make_runner: RunnerFactory) -> None:
    """C1: name, description and the target's schema are exported; `targets` is honoured."""
    for spec in Probe().specs():
        name, description, schema = harness.describe(harness.export(spec, make_runner()))
        assert name == spec.name, f"C1: exported name {name!r} != {spec.name!r}"
        assert description == spec.description, f"C1: {spec.name}: description differs"
        expected = harness.expected_schema(spec)
        assert dict(schema) == expected, (
            f"C1: {spec.name}: schema differs: exported {dict(schema)!r}, expected {expected!r}"
        )
        properties = set(spec.json_schema().get("properties", {}))
        assert set(dict(schema).get("properties", {})) == properties, f"C1: {spec.name}: fields"
    await _check_targets(harness, make_runner)


async def _check_targets(harness: Harness, make_runner: RunnerFactory) -> None:
    """Registry exporters skip specs whose `targets` exclude this integration."""
    add = Probe()._add
    registry = Registry()
    registry.add_many(
        [
            build_spec(add, name="kit_everywhere"),
            build_spec(add, name="kit_only_here", targets=harness.name),
            build_spec(add, name="kit_elsewhere", targets="aa_kit_elsewhere"),
        ]
    )
    exporters: list[tuple[str, Callable[[Registry, ToolRunner], list[Any]]]] = []
    integration = available_integrations().get(harness.name)
    if isinstance(integration, IntegrationBase):
        exporters.append(("export_tools", lambda r, e: integration.export_tools(r, executor=e)))
    export_all = getattr(harness, "export_all", None)
    if export_all is not None:
        exporters.append(("export_all", export_all))
    for label, export in exporters:
        names = {harness.describe(native)[0] for native in export(registry, make_runner())}
        expected = {"kit_everywhere", "kit_only_here"}
        assert names == expected, f"C1: {label} ignored targets: exported {sorted(names)}"


async def check_c2(harness: Harness, make_runner: RunnerFactory) -> None:
    """C2: the happy path returns what `spec.invoke` returns."""
    probe = Probe()
    cases = [
        (probe.add, {"a": 1, "b": 3}),
        (probe.add, {"a": 5}),
        (probe.profile, {"name": "Asha", "tags": ["tax", "gst"]}),
    ]
    for i, (spec, args) in enumerate(cases):
        native = harness.export(spec, make_runner())
        raw = await harness.call(native, dict(args), call_id=f"call_kit_c2_{i}")
        assert not harness.is_error(raw), f"C2: {spec.name}{args} returned an error: {raw!r}"
        expected = await spec.invoke(args)
        assert harness.value(raw) == expected, f"C2: {spec.name}{args}: {raw!r} != {expected!r}"


async def check_c3(harness: Harness, make_runner: RunnerFactory) -> None:
    """C3: invalid args give the framework's error, never a raw pydantic.ValidationError."""
    probe = Probe()
    native = harness.export(probe.add, make_runner())
    for i, args in enumerate([{"a": "not a number"}, {"b": 1}]):
        raw, exc = await _outcome(harness.call(native, args, call_id=f"call_kit_c3_{i}"))
        assert not isinstance(exc, pydantic.ValidationError), f"C3: {args}: raw ValidationError"
        assert _is_error(harness, raw, exc), f"C3: {args} was not reported as an error: {raw!r}"
    assert probe.executed == [], f"C3: the tool ran with invalid args: {probe.executed}"


async def check_c4(harness: Harness, make_runner: RunnerFactory) -> None:
    """C4: executor policy applies: hooks see before/after; a timeout is `tool-timeout`."""
    probe, hook = Probe(), RecordingHook()
    native = harness.export(probe.add, make_runner(hooks=[hook]))
    raw = await harness.call(native, {"a": 1, "b": 1}, call_id="call_kit_c4")
    assert harness.value(raw) == 2, f"C4: unexpected value {raw!r}"
    assert hook.phases() == ["before_tool", "after_tool"], f"C4: hooks saw {hook.phases()}"
    assert hook.calls[0][1][0].spec.name == "kit_add", "C4: hooks saw another tool"
    native = harness.export(probe.slow, make_runner())
    try:
        raw, exc = await _outcome(harness.call(native, {"label": "x"}, call_id="call_kit_c4_t"))
    finally:
        probe.released.set()
    if exc is not None:
        timed_out = any(isinstance(e, ToolTimeout) for e in _chain(exc))
    else:
        text = json.dumps(harness.value(raw), default=str)
        timed_out = harness.is_error(raw) and ("tool-timeout" in text or "timed out" in text)
    assert timed_out, f"C4: timeout not reported as tool-timeout: {raw!r} / {exc!r}"


async def check_c5(harness: Harness, make_runner: RunnerFactory) -> None:
    """C5: the native call id reaches `ctx.call_id` when the framework has one."""
    native = harness.export(Probe().call_id, make_runner())
    value = harness.value(await harness.call(native, {"tag": "t"}, call_id="call_kit_c5"))
    if getattr(harness, "native_call_ids", True):
        assert value == "t:call_kit_c5", f"C5: tool saw {value!r}"
    else:
        assert isinstance(value, str) and len(value) > 2, f"C5: tool saw {value!r}"


async def check_c6(harness: Harness, make_runner: RunnerFactory) -> None:
    """C6: `confirm=True`: deny skips the tool, approve runs it, edits apply when supported."""
    cases: list[tuple[ApprovalDecision, str | None]] = [
        (ApprovalDecision(approved=False, reason="kit says no"), None),
        (ApprovalDecision(approved=True), "alpha"),
    ]
    if harness.edits_supported:
        cases.append((ApprovalDecision(approved=True, edited_args={"item": "beta"}), "beta"))
    for decision, ran_with in cases:
        probe, port = Probe(), _c6_port(harness, decision)
        runner = make_runner(approval=port) if port is not None else make_runner()
        native = harness.export(probe.remove, runner)
        raw, exc = await _outcome(_approve_flow(harness, native, decision, port))
        if ran_with is None:
            assert probe.executed == [], f"C6: denied call ran: {probe.executed}"
            assert _is_error(harness, raw, exc), f"C6: denial not reported: {raw!r} / {exc!r}"
        else:
            assert exc is None, f"C6: approved call failed: {exc!r}"
            assert probe.executed == [("kit_remove", {"item": ran_with})], f"C6: {probe.executed}"
            assert harness.value(raw) == f"removed {ran_with}", f"C6: {raw!r}"
        _c6_port_asked(harness, port)


def _c6_port(harness: Harness, decision: ApprovalDecision) -> RecordingApproval | None:
    """Port mode: a scripted port; native mode: the harness's `approval_port()`, if any."""
    if harness.approval == "port":
        return RecordingApproval(ScriptedApproval([decision]))
    make_port: Callable[[], ApprovalPort | None] | None = getattr(harness, "approval_port", None)
    base = make_port() if make_port is not None else None
    return RecordingApproval(base) if base is not None else None


async def _approve_flow(
    harness: Harness, native: Any, decision: ApprovalDecision, port: RecordingApproval | None
) -> Any:
    args = {"item": "alpha"}
    params = inspect.signature(harness.approve_flow).parameters.values()
    if any(p.name == "port" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return await harness.approve_flow(native, args, decision, port=port)  # type: ignore[call-arg]
    return await harness.approve_flow(native, args, decision)


def _c6_port_asked(harness: Harness, port: RecordingApproval | None) -> None:
    if harness.approval == "port":
        assert port is not None
        assert [r.tool for r in port.requests] == ["kit_remove"], "C6: port not asked once"
        assert port.requests[0].arguments == {"item": "alpha"}, "C6: port saw other args"
    elif getattr(harness, "native_uses_port", False):
        assert port is not None, "C6: native_uses_port needs approval_port() to return a port"
        # A resume-based gate (LangGraph `interrupt`) asks again when the call re-runs.
        assert port.requests, "C6: the native gate never asked the approval port"
        asks = getattr(harness, "port_asks", None)
        if asks is not None:
            assert len(port.requests) == asks, (
                f"C6: port asked {len(port.requests)} times, expected {asks}"
            )
        for request in port.requests:
            assert (request.tool, request.arguments) == ("kit_remove", {"item": "alpha"}), (
                f"C6: port saw {request.tool} {request.arguments}"
            )


async def check_c7(harness: Harness, make_runner: RunnerFactory) -> None:
    """C7: `asyncio.CancelledError` propagates, from outside and from inside the tool."""
    probe = Probe()
    try:
        await _cancel_from_outside(harness, make_runner, probe)
    finally:
        probe.released.set()  # lets a tool that kept running (sync bridges) finish
    native = harness.export(probe.cancel, make_runner())
    try:
        async with asyncio.timeout(WAIT_LIMIT_S):
            raw = await harness.call(native, {"label": "y"}, call_id="call_kit_c7_inner")
    except asyncio.CancelledError:
        return
    except TimeoutError:
        raise AssertionError("C7: a tool raising CancelledError never returned") from None
    except Exception as exc:
        raise AssertionError(f"C7: CancelledError from the tool became {exc!r}") from exc
    raise AssertionError(f"C7: CancelledError from the tool became {raw!r}")


async def _cancel_from_outside(harness: Harness, make_runner: RunnerFactory, probe: Probe) -> None:
    native = harness.export(probe.wait, make_runner())
    task = asyncio.ensure_future(harness.call(native, {"label": "x"}, call_id="call_kit_c7"))
    started = asyncio.ensure_future(wait_until(probe.started, WAIT_LIMIT_S))
    try:
        await _cancel_after_start(harness, probe, task, started)
    finally:
        await _drain(task, started)  # a failed check leaves nothing on the module loop


async def _cancel_after_start(
    harness: Harness, probe: Probe, task: asyncio.Future[Any], started: asyncio.Future[bool]
) -> None:
    done, _ = await asyncio.wait(
        {task, started}, timeout=WAIT_LIMIT_S, return_when=asyncio.FIRST_COMPLETED
    )
    if task in done:
        raise AssertionError(f"C7: the call ended before the tool started: {_task_outcome(task)}")
    if not (started in done and started.result()):
        raise AssertionError("C7: the tool never started")
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=WAIT_LIMIT_S)
    assert done, "C7: the cancelled call did not end"
    assert task.cancelled(), f"C7: cancelling the call gave {_task_outcome(task)}"
    if getattr(harness, "cancels_tool", True):
        cancelled = await wait_until(probe.cancelled, CANCEL_GRACE_S)
        assert cancelled, "C7: the call was cancelled but the tool kept running"


async def _drain(*futures: asyncio.Future[Any]) -> None:
    """Cancel what is still pending (again, if it shrugged a cancel off) and wait briefly."""
    pending = [future for future in futures if not future.done()]
    for future in pending:
        future.cancel()
    if pending:
        await asyncio.wait(pending, timeout=CANCEL_GRACE_S)


async def check_c8(harness: Harness, make_runner: RunnerFactory) -> None:
    """C8: `get_integration(name)` finds this integration through its entry point."""
    available_integrations(refresh=True)
    found = get_integration(harness.name)
    assert isinstance(found, Integration), f"C8: {found!r} is not an Integration"
    assert found.name == harness.name


CHECKS: dict[str, tuple[str, Check]] = {
    "C1": ("exports_name_description_and_schema", check_c1),
    "C2": ("happy_path_matches_invoke", check_c2),
    "C3": ("invalid_args_are_framework_errors", check_c3),
    "C4": ("executor_policy_applies", check_c4),
    "C5": ("native_call_id_reaches_the_context", check_c5),
    "C6": ("confirm_deny_approve_and_edit", check_c6),
    "C7": ("cancellation_propagates", check_c7),
    "C8": ("entry_point_is_discoverable", check_c8),
}

_fixture_ids = itertools.count()


def contract_tests(
    harness_factory: Callable[[], Harness], *, runner_factory: RunnerFactory | None = None
) -> list[Any]:
    """The harness fixture, then pytest tests C1..C8, all on one module-scoped event loop."""
    make_runner = runner_factory or FakeRunner
    fixture_name = f"aa_kit_harness_{next(_fixture_ids)}"

    async def harness() -> AsyncIterator[Harness]:
        async with harness_session(harness_factory) as entered:
            yield entered

    harness.__name__ = harness.__qualname__ = fixture_name
    fixture = pytest_asyncio.fixture(scope="module", loop_scope="module", name=fixture_name)(
        harness
    )
    tests = [
        _bind(check_id, slug, check, fixture_name, make_runner)
        for check_id, (slug, check) in CHECKS.items()
    ]
    return [fixture, *tests]


def _bind(
    check_id: str, slug: str, check: Check, fixture_name: str, make_runner: RunnerFactory
) -> Callable[..., Any]:
    async def test(**fixtures: Any) -> None:
        await check(fixtures[fixture_name], make_runner)

    test.__name__ = test.__qualname__ = f"test_{check_id.lower()}_{slug}"
    test.__doc__ = check.__doc__
    test.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(fixture_name, inspect.Parameter.KEYWORD_ONLY)]
    )
    marked: Callable[..., Any] = pytest.mark.asyncio(loop_scope="module")(test)
    return marked
