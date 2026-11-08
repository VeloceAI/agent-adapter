"""Policy guards, budgets, rate limits, memory retention and sandbox checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import BudgetExceeded, ConfigError, GuardrailBlocked
from agent_adapter.core.messages import LLMRequest, Message
from agent_adapter.exec import Executor
from agent_adapter.hooks import CallableGuard, PIIGuard
from agent_adapter.hooks.builtin.budgets import MemoryBudgetStore
from agent_adapter.policy import PolicyError, load_policy
from tests.fakes.models import ScriptedModel, text
from tests.policy.helpers import Approvals, caller, shop

EMAIL_NOTE = {"text": "write to ann@example.com"}


def ex(data: dict[str, Any], **kwargs: Any) -> Executor:
    approval = kwargs.pop("approval", None)
    return load_policy(data, **kwargs).apply(Executor(shop(), approval=approval))


async def test_pii_guard_blocks_selected_tools_only() -> None:
    executor = ex({"guards": [{"guard": "pii", "tools": "send_note", "on": "args"}]})
    with pytest.raises(GuardrailBlocked):
        await executor.run("send_note", EMAIL_NOTE)
    assert await executor.run("lookup_order", {"order_id": "ann@example.com"})


async def test_pii_guard_redacts_results() -> None:
    from agent_adapter.core.registry import Registry

    reg = Registry()

    @reg.tool
    async def contact() -> str:
        """The shop's contact."""
        return "mail ann@example.com"

    policy = load_policy({"guards": [{"guard": "pii", "on": "result", "action": "redact"}]})
    assert "ann@example.com" not in await policy.apply(Executor(reg)).run("contact")


async def test_approve_action_sends_flagged_calls_to_approval() -> None:
    port = Approvals()
    data = {"guards": [{"guard": "pii", "tools": "send_note", "action": "approve"}]}
    executor = ex(data, approval=port)
    assert await executor.run("send_note", {"text": "hello"}) == "hello"
    assert port.asked == []
    assert await executor.run("send_note", EMAIL_NOTE) == EMAIL_NOTE["text"]
    assert [r.tool for r in port.asked] == ["send_note"]
    with pytest.raises(Exception, match="no approval port"):
        await ex(data).run("send_note", EMAIL_NOTE)


async def test_guard_on_model_calls_when_it_selects_every_tool() -> None:
    model = ScriptedModel([text("reply"), text("reply")])
    request = LLMRequest(messages=(Message(role="user", content="mail ann@example.com"),))
    everywhere = ex({"guards": [{"guard": "pii", "on": "llm_request"}]})
    with pytest.raises(GuardrailBlocked):
        await everywhere.model(model).bind(ToolContext(run_id="r")).generate(request)
    with pytest.raises(PolicyError, match="every model call") as caught:
        load_policy({"guards": [{"guard": "pii", "tools": "send_note", "on": "llm_request"}]})
    assert caught.value.key == "guards[0]"
    some = ex({"guards": [{"guard": "pii", "tools": "send_note"}]})  # tool stages only
    assert some.hooks == ()
    reply = await some.model(model).bind(ToolContext(run_id="r")).generate(request)
    assert reply.text == "reply"


async def test_guard_passed_in_code_and_scoped_to_a_team() -> None:
    guard = CallableGuard(lambda stage, payload: "no" not in str(payload), on=("args",))
    data = {"guards": [{"guard": "words", "tools": "send_note", "teams": "trainees"}]}
    executor = load_policy(data, guards={"words": guard}).apply(Executor(shop()))
    assert await executor.run("send_note", {"text": "no"}, caller(team="support")) == "no"
    with pytest.raises(GuardrailBlocked):
        await executor.run("send_note", {"text": "no"}, caller(team="trainees"))


@pytest.mark.parametrize(
    ("data", "guards", "match"),
    [
        ({"guard": "shout"}, {}, "unknown guard"),
        ({"guard": "words", "action": "block"}, {"words": PIIGuard()}, "built in code"),
        ({"guard": "words"}, {"words": object()}, "ContentGuard"),
        ({"guard": "pii", "on": "args", "action": "redact"}, {}, "redact"),
        ({"guard": "pii", "entities": ["shoe_size"]}, {}, "unknown PII kinds"),
    ],
)
def test_bad_guards_fail_at_load(data: dict[str, Any], guards: dict[str, Any], match: str) -> None:
    with pytest.raises(PolicyError, match=match) as caught:
        load_policy({"guards": [data]}, guards=guards)
    assert caught.value.key.startswith("guards[0]")


def test_presidio_without_the_library_fails_at_load() -> None:
    try:
        import presidio_analyzer  # noqa: F401
    except ImportError:
        with pytest.raises(PolicyError, match="presidio"):
            load_policy({"guards": [{"guard": "presidio"}]})
    else:  # pragma: no cover - only with the extra installed
        assert load_policy({"guards": [{"guard": "presidio"}]}).guards


async def test_budget_stops_calls_for_the_scope() -> None:
    executor = ex({"budgets": [{"max_usd": 0, "scope": "user", "per": "day"}]})
    with pytest.raises(BudgetExceeded):
        await executor.run("lookup_order", {"order_id": "o"}, caller(user="ann"))


async def test_team_budget_uses_the_policy_team_and_refuses_without_one() -> None:
    store = MemoryBudgetStore()
    data = {"budgets": [{"max_usd": 0, "scope": "team", "store": "shared"}]}
    executor = load_policy(data, stores={"shared": store}).apply(Executor(shop()))
    with pytest.raises(BudgetExceeded):
        await executor.run("lookup_order", {"order_id": "o"}, caller(team="support"))
    with pytest.raises(ConfigError, match="scope_key"):
        await executor.run("lookup_order", {"order_id": "o"}, caller())


def test_budget_stores(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(
        '{"budgets": [{"max_usd": 1, "scope": "user", "store": "sqlite:spend.db"},'
        ' {"max_usd": 2, "scope": "user", "store": "sqlite:spend.db", "name": "b"}]}',
        encoding="utf-8",
    )
    policy = load_policy(path)
    first, second = (hook for _, _, hook in policy.budgets)
    assert first.store is second.store
    assert Path(first.store.path).parent == tmp_path  # type: ignore[union-attr]
    with pytest.raises(PolicyError, match="unknown budget store"):
        load_policy({"budgets": [{"max_usd": 1, "store": "redis"}]})
    with pytest.raises(PolicyError, match="BudgetStores"):
        load_policy({"budgets": [{"max_usd": 1, "store": "x"}]}, stores={"x": object()})  # type: ignore[dict-item]
    with pytest.raises(PolicyError, match="file path"):
        load_policy({"budgets": [{"max_usd": 1, "store": "sqlite::memory:"}]})
    with pytest.raises(PolicyError, match=r"scope_key|team"):
        load_policy({"budgets": [{"max_usd": 1, "scope": "team"}]}, team_of="x")  # type: ignore[arg-type]


async def test_rate_limit_per_kind() -> None:
    policy = load_policy({"rate_limits": [{"kind": "tool", "calls": 1, "per": 60}]})
    assert policy.rates is not None
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    policy.rates._sleep = sleep
    executor = policy.apply(Executor(shop()))
    await executor.run("lookup_order", {"order_id": "a"})
    await executor.run("lookup_order", {"order_id": "b"})
    assert slept and slept[0] > 0


def test_two_rate_limits_for_one_kind_fail() -> None:
    data = {"rate_limits": [{"kind": "tool", "calls": 1, "per": 1}] * 2}
    with pytest.raises(PolicyError, match="second rate limit") as caught:
        load_policy(data)
    assert caught.value.key == "rate_limits[1]"
    by_env = {
        "environments": ["dev", "prod"],
        "rate_limits": [
            {"kind": "tool", "calls": 1, "per": 1, "envs": "dev"},
            {"kind": "tool", "calls": 9, "per": 1, "envs": "prod"},
        ],
    }
    assert load_policy(by_env, env="prod").rates.rates["tool"].calls == 9  # type: ignore[union-attr]


def test_memory_retention_only_shortens_unless_relaxed(tmp_path: Path) -> None:
    from agent_adapter.memory import SQLiteMemory

    policy = load_policy({"memory": {"retention_days": 30}})
    forever = SQLiteMemory(tmp_path / "a.db")
    assert policy.apply_memory(forever) is forever
    assert forever.retention_days == 30
    short = SQLiteMemory(tmp_path / "b.db", retention_days=7)
    assert policy.apply_memory(short).retention_days == 7
    relaxed = load_policy(
        {"memory": {"retention_days": 30}, "relax": ["memory.retention_days"]}, allow_relax=True
    )
    assert relaxed.apply_memory(short).retention_days == 30
    with pytest.raises(PolicyError, match="memory backend"):
        policy.apply_memory(object())
    nothing = object()
    assert load_policy({}).apply_memory(nothing) is nothing


def test_sandbox_requirements() -> None:
    from agent_adapter.sandbox import CallableSandbox, LocalSubprocessSandbox

    async def run(request: Any) -> Any:
        raise AssertionError("not run")

    strict = load_policy({"sandbox": {"require_isolated": True, "deny_network": True}})
    hosted = CallableSandbox(run)
    assert strict.check_sandbox(hosted) is hosted
    with pytest.raises(PolicyError, match="network"):
        strict.check_sandbox(CallableSandbox(run, network=True))
    with pytest.raises(PolicyError, match="not isolated"):
        strict.check_sandbox(LocalSubprocessSandbox(allow_unsafe_local=True))
    with pytest.raises(PolicyError, match="not isolated"):
        strict.check_sandbox(object())
    anything = object()
    assert load_policy({}).check_sandbox(anything) is anything


def test_retention_bounds_from_several_policies(tmp_path: Path) -> None:
    from agent_adapter.memory import SQLiteMemory

    memory = SQLiteMemory(tmp_path / "c.db", retention_days=7)
    load_policy({"memory": {"retention_days": 5}}).apply_memory(memory)
    assert memory.retention_days == 5
    relaxed = load_policy(
        {"memory": {"retention_days": 60}, "relax": ["memory.retention_days"]}, allow_relax=True
    )
    assert relaxed.apply_memory(memory).retention_days == 5  # the first policy still binds
    alone = SQLiteMemory(tmp_path / "d.db", retention_days=7)
    assert relaxed.apply_memory(alone).retention_days == 60
