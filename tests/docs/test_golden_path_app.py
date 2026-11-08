"""The golden-path app over HTTP (FastAPI's TestClient), and its eval cases as the CI gate."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from types import ModuleType

import pytest

from agent_adapter.cli.main import main
from agent_adapter.tracing.sinks import members
from agent_adapter.viewer.recorder import Recorder
from tests.docs.golden_path_support import EXAMPLES, golden

__all__ = ["golden"]
ANN = {"Authorization": "Bearer ann-token"}
LENA = {"Authorization": "Bearer lena-token"}
CASES = EXAMPLES / "golden_path" / "cases.jsonl"


def run_eval(cases: Path, *extra: str) -> tuple[int, str]:
    """`agent-adapter eval` as CI runs it, from the project folder (the fixture's cwd)."""
    out = io.StringIO()
    argv = ["eval", "golden_path.agent:eval_chat", str(cases)]
    argv += ["--executor", "golden_path.agent:eval_executor", "--grader", "contains"]
    code = main([*argv, *extra], out=out)
    return code, out.getvalue()


def test_the_eval_cases_pass_as_the_ci_gate_on_temporary_stores(golden: ModuleType) -> None:
    code, text = run_eval(CASES, "--grader", "max-cost=0.01")
    assert code == 0, text
    assert text.startswith("eval eval_chat: 5/5 passed")
    assert golden.default_desk.cache_info().currsize == 0  # the live desk was never built
    assert not Path(".support_desk").exists()  # no live approval, budget or memory touched
    desk = golden.eval_desk()
    assert desk.memory.path.startswith(tempfile.gettempdir())
    recorders = [s for s in members(golden.eval_executor.sink) if isinstance(s, Recorder)]
    assert len(recorders) == 1  # the eval reused the desk's recorder: same default folder
    assert list(Path(".agent_adapter/runs").glob("*.jsonl"))


def test_the_eval_stores_are_deleted_at_exit(golden: ModuleType) -> None:
    desk = golden.eval_desk()
    folder = Path(desk.memory.path).parent
    assert folder.is_dir()
    golden._discard(desk, folder)
    assert not folder.exists()


def test_the_eval_gate_fails_on_a_wrong_answer(golden: ModuleType, tmp_path: Path) -> None:
    case = {"id": "wrong", "input": "Where is order 1042?", "expected": "delivered"}
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case) + "\n", encoding="utf-8")
    code, text = run_eval(cases)
    assert code == 1
    assert "FAIL wrong" in text


def chat_turn(golden: ModuleType, session_id: str, user: str, text: str) -> str:
    """One chat-page turn of the live agent."""
    from agent_adapter.chat import ChatMessage, ChatSession
    from agent_adapter.runtime import run_sync

    executor = golden.default_desk().executor
    session = ChatSession(
        id=session_id, history=(), state={}, executor=executor, run_id=f"run_{session_id}",
        user_id=user,
    )  # fmt: skip
    reply: str = run_sync(golden.chat(ChatMessage(text=text), session))
    return reply


def test_the_chat_agent_runs_for_its_user(golden: ModuleType) -> None:
    assert chat_turn(golden, "s1", "ann", "Where is order 1043?") == "Your order is packed."


def test_anonymous_chats_never_share_memory(golden: ModuleType) -> None:
    assert chat_turn(golden, "s1", "", "remember I like blue") == "I will remember that."
    assert chat_turn(golden, "s1", "", "What do you know about me?").endswith("I like blue")
    assert chat_turn(golden, "s2", "", "What do you know about me?") == "Nothing saved about you."


@pytest.fixture
def client(golden: ModuleType, tmp_path: Path):  # type: ignore[no-untyped-def]
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from golden_path.app import create_app

    with TestClient(
        create_app(golden.build_desk(tmp_path / "http", runs_dir=tmp_path / "runs"))
    ) as test_client:
        yield test_client


def test_http_needs_a_known_token(client) -> None:  # type: ignore[no-untyped-def]
    assert client.post("/chat", json={"chat_id": "c1", "text": "hi"}).status_code == 401
    bad = {"Authorization": "Bearer ann-tokenX"}
    assert client.get("/budget", headers=bad).status_code == 401


def test_http_chat_refund_and_one_use_approval(client) -> None:  # type: ignore[no-untyped-def]
    answer = client.post("/chat", headers=ANN, json={"chat_id": "c1", "text": "Order 1042?"})
    assert answer.json()["text"] == "Your order is shipped."
    asked = client.post("/chat", headers=ANN, json={"chat_id": "c1", "text": "refund 1043"})
    approval_id = asked.json()["approval_id"]
    path = f"/approvals/{approval_id}"
    assert client.post(path, headers=ANN, json={"approved": True}).status_code == 403
    done = client.post(path, headers=LENA, json={"approved": True})
    assert done.json() == {"result": "refunded 18.50 EUR on order 1043"}
    again = client.post(path, headers=LENA, json={"approved": True})
    assert again.status_code == 409 and again.json()["detail"]["kind"] == (
        "approval-already-decided"
    )
    unknown = client.post("/approvals/apr_x", headers=LENA, json={"approved": True})
    assert unknown.status_code == 404


def test_http_tickets_budget_and_guard(client) -> None:  # type: ignore[no-untyped-def]
    ticket = client.post("/tickets", headers=ANN, json={"order_id": "1042", "summary": "lost"})
    assert ticket.json()["note"].startswith("team notified about T-")
    blocked = client.post(
        "/tickets", headers=ANN, json={"order_id": "1042", "summary": "card 4111 1111 1111 1111"}
    )
    assert blocked.status_code == 422 and blocked.json()["detail"]["kind"] == "flow-error"
    budget = client.get("/budget", headers=ANN).json()
    assert budget["spent_usd"] == 0.0 and budget["remaining_usd"] == 0.5
    too_long = {"chat_id": "c1", "text": "x" * 4001}
    assert client.post("/chat", headers=ANN, json=too_long).status_code == 422


def test_create_app_without_a_desk_builds_the_default_one_at_startup(golden: ModuleType) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from golden_path.app import app

    assert not Path(".support_desk").exists()  # importing the app built nothing
    with TestClient(app) as client:
        answer = client.post("/chat", headers=ANN, json={"chat_id": "c1", "text": "Order 1043?"})
        assert answer.json()["text"] == "Your order is packed."
        assert Path(".support_desk").is_dir()
    assert golden.default_desk.cache_info().currsize == 1


def test_http_reviewers_never_approve_their_own_refund(client) -> None:  # type: ignore[no-untyped-def]
    asked = client.post("/chat", headers=LENA, json={"chat_id": "c1", "text": "refund 3001"})
    path = f"/approvals/{asked.json()['approval_id']}"
    own = client.post(path, headers=LENA, json={"approved": True})
    assert own.status_code == 403 and own.json()["detail"]["kind"] == "forbidden"


def test_http_users_never_see_another_users_order(client) -> None:  # type: ignore[no-untyped-def]
    ben = {"Authorization": "Bearer ben-token"}
    other = client.post("/chat", headers=ben, json={"chat_id": "c1", "text": "Order 1042?"})
    assert other.json()["text"] == "Your order is unknown."
    card = {"chat_id": "c1", "text": "my card is 4111 1111 1111 1111"}
    blocked = client.post("/chat", headers=ben, json=card)
    assert blocked.status_code == 422 and blocked.json()["detail"]["kind"] == "guardrail-blocked"
