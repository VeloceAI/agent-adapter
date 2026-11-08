"""The Memory panel's routes, the `--memory` flags, and memory events in the run viewer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.cli.ui import add_ui_parser, run_ui
from agent_adapter.core.errors import ConfigError
from agent_adapter.memory import CallableMemory, MemoryRecord, MemoryScope, SQLiteMemory
from agent_adapter.viewer._fold import Fold, summarise
from agent_adapter.viewer.chat.memory import clean_texts, open_memory
from agent_adapter.viewer.lanes import build_spans
from tests.viewer.chat.helpers import chat, chat_served, get, post


def echo(message: Any, session: Any) -> str:
    return "ok"


def seeded(tmp_path: Path) -> SQLiteMemory:
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.add_sync(
        [
            MemoryRecord(scope=MemoryScope(user_id="ann", session_id="s1"), text="first"),
            MemoryRecord(scope=MemoryScope(user_id="ann", session_id="s2"), text="second"),
            MemoryRecord(scope=MemoryScope(user_id="bob", session_id="s3"), text="bob's"),
        ]
    )
    return memory


def test_the_panel_lists_and_deletes_only_the_users_records(tmp_path: Path) -> None:
    memory = seeded(tmp_path)
    (bob,) = memory.export_sync(MemoryScope(user_id="bob"))
    with chat_served(tmp_path, agent=echo, user_id="ann", memory=memory) as server:
        listing = get(server, "/api/chat/memory")[1]
        assert listing["memory"] == {
            "enabled": True,
            "user": "ann",
            "backend": "sqlite",
            "save": "none",
            "every": 1,
        }
        assert [r["text"] for r in listing["records"]] == ["second", "first"]
        first = listing["records"][1]["id"]
        status, body = post(server, "/api/chat/memory/delete", {"ids": [first, bob.id]})
        assert (status, body) == (200, {"deleted": 1})
        assert [r["text"] for r in get(server, "/api/chat/memory")[1]["records"]] == ["second"]
    assert [r.text for r in memory.export_sync(MemoryScope(user_id="bob"))] == ["bob's"]


@pytest.mark.parametrize(
    ("headers", "reason"),
    [({"Origin": "http://evil.example"}, "Origin"), ({"X-AA-CSRF": ""}, "CSRF")],
)
def test_delete_needs_the_origin_and_csrf_checks(
    tmp_path: Path, headers: dict[str, str], reason: str
) -> None:
    memory = seeded(tmp_path)
    with chat_served(tmp_path, agent=echo, user_id="ann", memory=memory) as server:
        ids = [r["id"] for r in get(server, "/api/chat/memory")[1]["records"]]
        status, body = post(server, "/api/chat/memory/delete", {"ids": ids}, **headers)
    assert status == 403 and reason in body["error"]
    assert len(memory.export_sync(MemoryScope(user_id="ann"))) == 2


@pytest.mark.parametrize("body", [{}, {"ids": []}, {"ids": "mem_1"}, {"ids": [1]}, {"ids": [""]}])
def test_delete_refuses_bad_ids(tmp_path: Path, body: dict[str, Any]) -> None:
    with chat_served(tmp_path, agent=echo, user_id="ann", memory=seeded(tmp_path)) as server:
        status, reply = post(server, "/api/chat/memory/delete", body)
    assert status == 400 and reply["error"]


def test_with_memory_off_the_listing_says_so_and_delete_is_404(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=echo) as server:
        assert get(server, "/api/chat/memory")[1] == {"memory": {"enabled": False}, "records": []}
        status, _ = post(server, "/api/chat/memory/delete", {"ids": ["mem_1"]})
    assert status == 404


def test_a_failing_backend_answers_502(tmp_path: Path) -> None:
    def broken(*args: Any) -> Any:
        raise OSError("down")

    memory = CallableMemory(add=broken, search=broken, export=broken, delete=broken)
    with chat_served(tmp_path, agent=echo, user_id="ann", memory=memory) as server:
        assert get(server, "/api/chat/memory")[0] == 502
        assert post(server, "/api/chat/memory/delete", {"ids": ["mem_1"]})[0] == 502
        assert chat(server).memory is not None


def test_clean_texts_drops_blanks_repeats_and_none() -> None:
    assert clean_texts([" a ", "", "NONE", "none", "a", "b"]) == ["a", "b"]
    assert len(clean_texts([str(n) for n in range(50)])) == 20


def test_open_memory_reads_sqlite_paths_and_targets(tmp_path: Path) -> None:
    assert isinstance(open_memory(f"sqlite:{tmp_path / 'm.db'}"), SQLiteMemory)
    with pytest.raises(ConfigError, match="needs a file path"):
        open_memory("sqlite:")
    with pytest.raises(ConfigError):
        open_memory("no_such_module_here:memory")


def parse(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_ui_parser(parser.add_subparsers())
    return parser.parse_args(["ui", *argv])


def test_memory_flags_parse_and_are_chat_only(tmp_path: Path) -> None:
    args = parse("--memory", "sqlite:m.db", "--user", "ann", "--memory-save", "facts")
    assert (args.memory, args.user, args.memory_save, args.memory_every, args.recall) == (
        "sqlite:m.db",
        "ann",
        "facts",
        1,
        5,
    )
    for flags in (["--user", "ann"], ["--memory", "sqlite:m.db"], ["--recall", "3"]):
        with pytest.raises(ConfigError, match="is a chat option"):
            run_ui(parse(str(tmp_path), *flags), out=None)  # type: ignore[arg-type]
    with pytest.raises(SystemExit):
        parse("--memory-save", "all")


def test_the_cli_builds_a_chat_server_with_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: dict[str, Any] = {}
    monkeypatch.setattr(
        "agent_adapter.viewer.serving.make_chat_server", lambda **kw: built.update(kw) or "srv"
    )
    monkeypatch.setattr("agent_adapter.viewer.serving.run_server", lambda server, out: None)
    monkeypatch.setattr("agent_adapter.viewer.chat.targets.load_target", lambda t, o: echo)
    db = tmp_path / "m.db"
    flags = ("--agent", "x:y", "--memory", f"sqlite:{db}", "--user", "ann", "--recall", "3")
    assert run_ui(parse(str(tmp_path), *flags, "--memory-every", "2"), out=None) == 0  # type: ignore[arg-type]
    assert isinstance(built["memory"], SQLiteMemory)
    assert (built["user_id"], built["recall_k"], built["memory_every"]) == ("ann", 3, 2)
    assert built["memory_save"] == "none"


def memory_event(kind: str, ts: float, **data: Any) -> dict[str, Any]:
    return {"kind": "event", "type": kind, "ts": ts, "call_id": "c1", "data": data}


def test_memory_events_become_spans_and_are_counted_not_named_as_the_framework() -> None:
    events = [
        memory_event("memory.searched", 1.0, backend="sqlite", count=2, latency_ms=4.0, k=5),
        memory_event("memory.added", 2.0, backend="sqlite", count=1, latency_ms=1.0),
        memory_event("memory.deleted", 3.0, backend="sqlite", count=0, latency_ms=1.0),
        memory_event("memory.exported", 4.0, backend="sqlite", count=3, latency_ms=1.0),
    ]
    spans = build_spans(events)
    assert [s["name"] for s in spans] == [
        "recall (sqlite)",
        "remember (sqlite)",
        "forget (sqlite)",
        "export (sqlite)",
    ]
    assert [s["count"] for s in spans] == [2.0, 1.0, 0.0, 3.0]
    assert {s["family"] for s in spans} == {"memory"} and len({s["id"] for s in spans}) == 4
    assert spans[0]["start"] == pytest.approx(0.996)
    fold = Fold()
    for item in events:
        fold.add(item)
    summary = summarise("run_1", [fold], 0.0, set())
    assert summary["memory"] == 4 and summary["framework"] == ""
