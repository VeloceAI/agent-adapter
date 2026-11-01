"""The chat demo in the docs matches its generator, stays offline, replays every pattern.

Regenerate it with `python scripts/gen_docs_chat_demo.py`.
"""

from __future__ import annotations

import json
import posixpath
import re
from pathlib import Path
from typing import Any

import pytest

from agent_adapter.viewer.assets import CHAT_SCRIPTS, DEMO_SCRIPTS, SCRIPTS
from scripts import chat_demo_data, gen_docs_chat_demo
from tests.viewer.test_export import EXTERNAL, FETCHED

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
DEMO = DOCS / "assets" / "demo" / "chat-demo.html"
PAGES = (DOCS / "how-to" / "chat-with-your-agent.md", DOCS / "how-to" / "chat-flows.md")
IFRAME = re.compile(r'<iframe class="aa-demo" src="([^"]+)" title="([^"]+)"[^>]*loading="lazy"')
FLOWS = ["ask", "compare_prices", "booking", "park_search", "day_plan"]


@pytest.fixture(scope="module")
def built() -> str:
    return gen_docs_chat_demo.build()


def _data(html: str) -> dict[str, Any]:
    found = re.search(r'id="aa-data">(.*?)</script>', html, re.S)
    assert found is not None
    return dict(json.loads(found.group(1)))


def test_the_committed_demo_matches_its_generator(built: str) -> None:
    assert DEMO.read_text(encoding="utf-8") == built, "run python scripts/gen_docs_chat_demo.py"


def test_the_generator_is_byte_stable(built: str) -> None:
    assert gen_docs_chat_demo.build() == built


def test_check_mode_reports_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stale = tmp_path / "chat-demo.html"
    stale.write_text("old", encoding="utf-8")
    monkeypatch.setattr(gen_docs_chat_demo, "OUT", stale)
    monkeypatch.setattr(gen_docs_chat_demo, "ROOT", tmp_path)
    assert gen_docs_chat_demo.main(["--check"]) == 1
    assert gen_docs_chat_demo.main([]) == 0
    assert gen_docs_chat_demo.main(["--check"]) == 0


def test_the_demo_is_offline_and_small() -> None:
    html = DEMO.read_text(encoding="utf-8")
    assert "default-src 'none'" in html and "connect-src" not in html
    assert not EXTERNAL.search(html)
    assert set(FETCHED.findall(html.replace("http://www.w3.org/2000/svg", ""))) == set()
    assert "AA.chatApi" in html and "AA.chatDemo" in html, "the chat page and its stand-in API"
    assert 'name="aa-chat" content="demo"' in html
    assert "eval(" not in html and "new Function" not in html
    scripts = (*SCRIPTS, *CHAT_SCRIPTS, *DEMO_SCRIPTS)
    assert html.count("</script>") == len(scripts) + 1, "the data cannot close its element"
    local_path = re.compile(r"[A-Za-z]:\\\\|/home/|/Users/|\.agent_adapter[/\\]")
    assert not local_path.search(html), "no local paths"
    assert len(html.encode("utf-8")) < 1_500_000


def test_the_demo_replays_every_pattern_in_three_chats() -> None:
    data = _data(DEMO.read_text(encoding="utf-8"))
    chats = data["chat"]["chats"]
    assert [c["session"]["title"] for c in chats] == [
        "Museum visit",
        "Playground by the river",
        "A rainy day",
    ]
    turns = [t for c in chats for t in c["turns"]]
    assert [t["label"] for t in turns] == [
        "router",
        "fan-out, fan-in",
        "approval",
        "loop",
        "all together",
    ]
    assert [data["details"][t["turn"]["run_id"]]["summary"]["flows"] for t in turns] == [
        [f] for f in FLOWS
    ]
    assert data["chat"]["live"] == "python examples/chat_flows.py"
    assert data["chat"]["info"]["mode"] == "demo"
    for turn in turns:
        steps = turn["steps"]
        times = [s["at"] for s in steps]
        assert times == sorted(times) and steps[-1]["sse"]["type"] == "done"
        wait = sum(chat_demo_data.APPROVAL_WAIT_MS for s in steps if _decided(s))
        assert 3000 <= times[-1] - wait <= 6000, turn["label"]
        assert steps[-1]["sse"]["turn"] == turn["turn"], "the end event carries the stored turn"
        run = data["details"][turn["turn"]["run_id"]]
        shown = [s["run"] for s in steps if "run" in s]
        assert shown == sorted(shown) and shown[-1] == len(run["events"])
        assert run["graphs"], "the graph the inspector lights up"


def _decided(step: dict[str, Any]) -> bool:
    event = step.get("sse") or {}
    return bool(event.get("type") == "approval" and event.get("status") != "pending")


def test_the_booking_is_approved_through_the_one_use_decision() -> None:
    data = _data(DEMO.read_text(encoding="utf-8"))
    booking = data["chat"]["chats"][0]["turns"][2]
    statuses = [
        s["sse"]["status"] for s in booking["steps"] if s.get("sse", {}).get("type") == "approval"
    ]
    assert statuses == ["pending", "approved"]
    (approval,) = booking["turn"]["approvals"]
    assert (approval["decided_by"], approval["status"]) == ("demo", "approved")
    events = data["details"][booking["turn"]["run_id"]]["events"]
    assert [e["data"]["approved"] for e in events if e["type"] == "approval.decided"] == [True]
    assert booking["state"] == {"place": "City Museum", "questions": 3}


def test_the_loops_ran_and_state_was_traced() -> None:
    data = _data(DEMO.read_text(encoding="utf-8"))
    by_flow = {d["summary"]["flows"][0]: d for d in data["details"].values()}
    visits = [
        e["data"]["node"] for e in by_flow["park_search"]["events"] if e["type"] == "node.started"
    ]
    assert visits.count("search") == 3, "widened twice"
    day = [e["data"]["node"] for e in by_flow["day_plan"]["events"] if e["type"] == "node.started"]
    assert day.count("candidates") == 2 and day.count("check") == 2
    finished = [
        e for d in data["details"].values() for e in d["events"] if e["type"] == "node.finished"
    ]
    assert any("updates" in e["data"] for e in finished), "FlowRunner(trace_state=True)"
    assert any(e["type"] == "llm.finished" for e in by_flow["ask"]["events"]), "the Cost tab"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_the_page_embeds_the_demo_where_the_built_site_serves_it(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    match = IFRAME.search(text)
    assert match is not None, "an iframe with a title and lazy loading"
    assert "Agent Adapter" in match.group(2)
    directory_urls = "use_directory_urls: false" not in (ROOT / "mkdocs.yml").read_text("utf-8")
    base = f"how-to/{page.stem}/" if directory_urls else "how-to/"
    assert (DOCS / posixpath.normpath(posixpath.join(base, match.group(1)))).resolve() == DEMO
    assert "](../assets/demo/chat-demo.html)" in text, "a full-screen link checked by --strict"
    tag = re.search(r'<iframe class="aa-demo"[^>]*>', text)
    assert tag is not None
    sandbox = re.search(r'sandbox="([^"]*)"', tag.group(0))
    assert sandbox is not None and set(sandbox.group(1).split()) == {
        "allow-scripts",
        "allow-downloads",
    }
    assert "python examples/chat_flows.py" in text


def test_the_graph_page_points_at_the_chat_demo() -> None:
    text = (DOCS / "how-to" / "navigate-the-graph.md").read_text(encoding="utf-8")
    assert "chat-flows.md" in text


def test_the_timeline_reveals_run_events_before_the_stream_event_they_preceded() -> None:
    stream = [
        {"type": "state"},
        {"type": "tool", "status": "running"},
        {"type": "approval", "status": "pending"},
        {"type": "approval", "status": "approved"},
        {"type": "done"},
    ]
    types = ["run.started", "flow.started", "node.started", "tool.started", "node.finished"]
    steps = chat_demo_data.timeline(stream, [1, 1, 1, 2, 4], types)
    order = [s["sse"]["type"] if "sse" in s else s["run"] for s in steps]
    assert order == ["state", 1, 2, 3, "tool", 4, "approval", "approval", 5, "done"]
    times = [s["at"] for s in steps]
    assert times == sorted(times)
    pending, decided = (s["at"] for s in steps if s.get("sse", {}).get("type") == "approval")
    assert decided - pending == chat_demo_data.APPROVAL_WAIT_MS, "the wait is never scaled"
    assert times[-1] - chat_demo_data.APPROVAL_WAIT_MS >= chat_demo_data.MIN_TURN_MS


def test_the_timeline_caps_a_long_turn_and_refuses_marks_that_do_not_fit() -> None:
    stream = [{"type": "text"}] * 200
    assert chat_demo_data.timeline(stream, [], [])[-1]["at"] == chat_demo_data.MAX_TURN_MS
    with pytest.raises(RuntimeError, match="marks"):
        chat_demo_data.timeline(stream, [0], [])


def test_each_run_sits_on_its_turn_s_replay_clock() -> None:
    data = _data(DEMO.read_text(encoding="utf-8"))
    for chat in data["chat"]["chats"]:
        for turn in chat["turns"]:
            view = turn["turn"]
            run = data["details"][view["run_id"]]
            assert run["summary"]["started"] == view["started"]
            assert view["started"] < run["summary"]["ended"] <= view["ended"]
            shown = {s["run"]: s["at"] for s in turn["steps"] if "run" in s}
            stamps = [e["ts"] for e in run["events"]]
            assert stamps == sorted(stamps)
            for count, at in shown.items():  # the last event a step reveals: at that step's time
                assert abs(stamps[count - 1] - (view["started"] + at / 1000)) < 0.01


def _run_file(folder: Path, types: list[str]) -> Path:
    records = [{"kind": "meta", "run_id": "run_x", "started": 5.0, "ts": 5.0}]
    for n, kind in enumerate(types):
        data = {"latency_ms": 1.0} if kind.endswith(".finished") else {}
        records.append({"kind": "event", "type": kind, "ts": 5.0 + n, "call_id": "c", "data": data})
    path = folder / "0000000005000.run_x.1.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _one_turn(types: list[str]) -> tuple[chat_demo_data.Recorded, list[dict[str, Any]]]:
    recorded = chat_demo_data.Recorded(runs=Path("."), info={}, emitted={"orig": types})
    steps = [{"at": 0, "run": 1}, {"at": 1000, "sse": {"type": "text"}}, {"at": 2000, "run": 2}]
    return recorded, [{"turns": [{"turn": {"run_id": "orig", "started": 100.0}, "steps": steps}]}]


def test_retime_refuses_a_run_file_that_differs_from_what_was_sent(tmp_path: Path) -> None:
    _run_file(tmp_path, ["tool.finished", "tool.started"])
    recorded, chats = _one_turn(["tool.started", "tool.finished"])
    with pytest.raises(RuntimeError, match="not the ones sent"):
        chat_demo_data.retime(tmp_path, recorded, chats, {"orig": "run_x"})


def test_retime_moves_events_and_durations_onto_the_replay(tmp_path: Path) -> None:
    _run_file(tmp_path, ["tool.started", "tool.finished"])
    recorded, chats = _one_turn(["tool.started", "tool.finished"])
    chat_demo_data.retime(tmp_path, recorded, chats, {"orig": "run_x"})
    (path,) = tmp_path.glob("*.jsonl")
    assert path.name == "0000000100000.run_x.1.jsonl"
    meta, started, finished = (json.loads(line) for line in path.read_text("utf-8").splitlines())
    assert (meta["started"], started["ts"], finished["ts"]) == (100.0, 100.0, 102.0001)
    assert finished["data"]["latency_ms"] == 2000.1
