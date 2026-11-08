"""The golden-path support desk runs offline: tools, approval, memory, guards, budgets, flows."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from agent_adapter.errors import (
    ApprovalAlreadyDecided,
    ApprovalNotFound,
    BudgetExceeded,
    FlowError,
    Forbidden,
    GuardrailBlocked,
)
from tests.docs.golden_path_support import golden

__all__ = ["golden"]


class PowerCut(BaseException):
    """The worker dies mid-call (a killed container)."""


def ask(desk: Any, user: str, chat: str, text: str) -> Any:
    """One turn, from sync test code."""
    return asyncio.run(desk.ask(user, chat, text))


def test_importing_the_agent_creates_nothing(golden: ModuleType) -> None:
    assert list(Path.cwd().iterdir()) == []  # no data folder, no runs folder, no store
    assert golden.default_desk.cache_info().currsize == 0
    with pytest.raises(AttributeError):
        _ = golden.not_a_name


def test_build_desk_defaults_to_the_env_folder_and_the_viewer_runs(golden: ModuleType) -> None:
    desk = golden.default_desk()
    assert golden.default_desk() is desk
    assert Path(".support_desk").is_dir()  # $SUPPORT_DESK_DATA
    assert desk.recorder.folder == Path(".agent_adapter", "runs").resolve()


def test_orders_and_help_answers_cite_their_source(golden: ModuleType, tmp_path: Path) -> None:
    desk = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    try:
        assert ask(desk, "ann", "c1", "Where is order 1042?").text == "Your order is shipped."
        assert ask(desk, "ann", "c1", "Order 9999?").text == "Your order is unknown."
        help_reply = ask(desk, "ann", "c1", "How many days does delivery take?")
        assert help_reply.text.endswith("[source: https://help.example.com/delivery]")
    finally:
        desk.close()
    assert list((tmp_path / "runs").glob("*.jsonl")), "the recorder wrote the runs"


def test_memory_is_recalled_across_chats_for_its_user_only(
    golden: ModuleType, tmp_path: Path
) -> None:
    desk = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    try:
        saved = ask(desk, "ann", "chat-1", "remember I prefer email updates at ann@example.com")
        assert saved.text == "I will remember that."
        recalled = ask(desk, "ann", "chat-2", "What do you know about me?").text
        assert recalled == "From earlier chats: I prefer email updates at [EMAIL]"
        assert ask(desk, "ben", "chat-3", "What do you know about me?").text == (
            "Nothing saved about you."
        )
    finally:
        desk.close()


def test_a_card_number_never_reaches_the_model_or_a_tool(
    golden: ModuleType, tmp_path: Path
) -> None:
    desk = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    try:
        with pytest.raises(GuardrailBlocked, match="found card"):
            ask(desk, "ann", "c1", "remember my card is 4111 1111 1111 1111")
        assert ask(desk, "ann", "c2", "What do you know about me?").text == (
            "Nothing saved about you."
        )
        with pytest.raises(FlowError, match="guardrail-blocked"):
            asyncio.run(desk.escalate("ann", "1042", "card 4111 1111 1111 1111"))
    finally:
        desk.close()


def test_a_refund_waits_for_one_use_approval_across_a_restart(
    golden: ModuleType, tmp_path: Path
) -> None:
    first = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    reply = ask(first, "ann", "c1", "Please refund order 1043")
    first.close()
    assert reply.text == "A reviewer must approve refund_order first."
    assert reply.approval_id.startswith("apr_")
    second = golden.build_desk(
        tmp_path / "d", runs_dir=tmp_path / "runs"
    )  # a restarted worker, same files
    try:
        done = asyncio.run(second.decide(reply.approval_id, True, "lena"))
        assert done == "refunded 18.50 EUR on order 1043"
        with pytest.raises(ApprovalAlreadyDecided):
            asyncio.run(second.decide(reply.approval_id, True, "lena"))
        with pytest.raises(ApprovalNotFound):
            asyncio.run(second.decide("apr_unknown", True, "lena"))
        denied = ask(second, "ann", "c1", "Please refund order 1042")
        refused = asyncio.run(second.decide(denied.approval_id, False, "lena"))
        assert refused == "not run (approval-denied)"
    finally:
        second.close()


def test_users_only_see_and_refund_their_own_orders(golden: ModuleType, tmp_path: Path) -> None:
    desk = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    try:
        assert ask(desk, "ben", "c1", "Where is order 1042?").text == "Your order is unknown."
        asked = ask(desk, "ben", "c1", "Please refund order 1042")
        refused = asyncio.run(desk.decide(asked.approval_id, True, "lena"))
        assert refused == "not run (tool-usage-error)"  # the tool runs as ben: not his order
        with pytest.raises(FlowError, match="no order 1042"):
            asyncio.run(desk.escalate("ben", "1042", "lost"))
    finally:
        desk.close()


def test_a_reviewer_never_decides_their_own_request(golden: ModuleType, tmp_path: Path) -> None:
    desk = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    try:
        asked = ask(desk, "lena", "c1", "Please refund order 3001")
        with pytest.raises(Forbidden):
            asyncio.run(desk.decide(asked.approval_id, True, "lena"))
        done = asyncio.run(desk.decide(asked.approval_id, True, "omar"))
        assert done == "refunded 40.00 EUR on order 3001"
    finally:
        desk.close()


def test_the_daily_budget_stops_one_user_not_the_others(golden: ModuleType, tmp_path: Path) -> None:
    desk = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs", daily_usd=0.01)
    answered = 0
    try:
        with pytest.raises(BudgetExceeded):
            for _ in range(20):
                ask(desk, "ann", "c1", "Where is order 1042?")
                answered += 1
        assert 1 <= answered < 20
        assert ask(desk, "ben", "c2", "Where is order 2001?").text == "Your order is shipped."
        status = asyncio.run(desk.budget.status(desk.context("ann", "")))
        assert 0 < status.spent_usd < 0.01  # refused before the model, on the estimate
    finally:
        desk.close()


def test_a_crashed_escalation_is_finished_by_the_next_worker(
    golden: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import golden_path.tools as tools

    pages: list[str] = []

    def flaky_pager(ticket_id: str) -> str:
        pages.append(ticket_id)
        if len(pages) == 1:
            raise PowerCut
        return f"team notified about {ticket_id}"

    monkeypatch.setattr(tools, "page_team", flaky_pager)
    first = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    with pytest.raises(PowerCut):
        asyncio.run(first.escalate("ann", "1042", "parcel lost"))
    first.close()
    second = golden.build_desk(tmp_path / "d", runs_dir=tmp_path / "runs")
    try:
        [held] = asyncio.run(second.recover()).values()
        assert held == "recovery-needs-decision"  # the page may have gone out: a person checks
        [status] = asyncio.run(second.recover(rerun_in_doubt=True)).values()
        assert status == "done"
        assert asyncio.run(second.recover()) == {}
        assert len(pages) == 2 and pages[0] == pages[1]  # one ticket id, opened once
    finally:
        second.close()


def test_the_checkpoint_key_comes_from_the_environment_or_one_key_file(
    golden: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "keys"
    folder.mkdir()
    made = golden.checkpoint_key(folder)
    assert len(made) == 32 and golden.checkpoint_key(folder) == made
    monkeypatch.setenv("SUPPORT_DESK_KEY", "from-the-vault-0123456789")
    assert golden.checkpoint_key(folder) == b"from-the-vault-0123456789"
