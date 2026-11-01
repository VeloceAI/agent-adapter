import asyncio

import pytest

from agent_adapter.core import control
from agent_adapter.core.control import control_flow_member, is_control_flow, register_control_flow
from agent_adapter.core.errors import ApprovalPending, ToolUsageError


class GraphBubbleUp(Exception):
    """Same name as LangGraph's interrupt base class."""


class GraphInterrupt(GraphBubbleUp):
    pass


class Signal(Exception):
    pass


class SignalChild(Signal):
    pass


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(control, "_registered", ())


def test_cancelled_error_is_control_flow():
    assert is_control_flow(asyncio.CancelledError())


def test_any_class_named_graph_bubble_up_in_the_mro_is_control_flow():
    assert is_control_flow(GraphBubbleUp())
    assert is_control_flow(GraphInterrupt())


def test_ordinary_errors_are_not_control_flow():
    assert not is_control_flow(ValueError("x"))
    assert not is_control_flow(ApprovalPending("apr_1"))
    assert not is_control_flow(KeyboardInterrupt())


def test_registered_types_and_their_subclasses_are_control_flow():
    assert not is_control_flow(SignalChild())
    register_control_flow(Signal)
    register_control_flow(Signal)  # idempotent
    assert is_control_flow(Signal()) and is_control_flow(SignalChild())
    assert control._registered == (Signal,)


def test_register_rejects_non_exception_types():
    with pytest.raises(ToolUsageError, match="BaseException subclasses"):
        register_control_flow(int)  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError):
        register_control_flow(ValueError("instance"))  # type: ignore[arg-type]
    assert control._registered == ()


def test_register_needs_at_least_one_type():
    with pytest.raises(ToolUsageError, match="at least one"):
        register_control_flow()


def test_groups_count_as_control_flow_when_a_member_is():
    leaf = GraphInterrupt()
    group = ExceptionGroup("outer", [ValueError("x"), ExceptionGroup("inner", [KeyError(), leaf])])
    assert is_control_flow(group) and control_flow_member(group) is leaf
    cancel = asyncio.CancelledError()
    assert control_flow_member(BaseExceptionGroup("b", [cancel])) is cancel


def test_groups_of_ordinary_errors_are_not_control_flow():
    group = ExceptionGroup("outer", [ValueError("x"), ExceptionGroup("inner", [KeyError()])])
    assert not is_control_flow(group) and control_flow_member(group) is None


def test_exec_control_re_exports_the_base_layer_registry():
    from agent_adapter import exec as exec_pkg
    from agent_adapter.exec import control as exec_control

    assert exec_control.register_control_flow is register_control_flow
    assert exec_control.is_control_flow is is_control_flow
    assert exec_pkg.register_control_flow is register_control_flow
    exec_pkg.register_control_flow(Signal)
    assert is_control_flow(Signal())
