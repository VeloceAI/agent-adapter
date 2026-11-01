"""The chat examples run offline; their wrappers work with stand-ins; the graph page matches."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import threading
import types
from collections.abc import AsyncIterator, Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from agent_adapter import Executor, Message
from agent_adapter.viewer import ChatMessage, ChatSession

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
HOW_TO = ROOT / "docs" / "how-to"
SNIPPET = re.compile(r'--8<-- "examples/(chat_[a-z_]+\.py):([a-z_]+)"')


def _load(name: str) -> dict[str, Any]:
    import runpy

    return runpy.run_path(str(EXAMPLES / name), run_name="chat_example")


def _session(executor: Executor, history: tuple[Message, ...] = ()) -> ChatSession:
    return ChatSession(id="ses_1", history=history, state={}, executor=executor, run_id="run_1")


async def _collect(reply: Any) -> str:
    if isinstance(reply, str):
        return reply
    return "".join([chunk async for chunk in reply])


def test_builtin_check_runs_one_turn_through_the_http_api(tmp_path: Path) -> None:
    done = subprocess.run(
        [sys.executable, str(EXAMPLES / "chat_builtin.py"), "--check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == "done 2 tool calls $0.001080"
    assert "- **Booking:** booked Casa Azul for 2" in lines
    assert (tmp_path / ".agent_adapter" / "chats").is_dir()
    assert any((tmp_path / ".agent_adapter" / "runs").iterdir()), "the turn was recorded"


def test_minimal_agent_answers_on_its_own() -> None:
    module = _load("chat_minimal.py")
    reply = module["chat"](ChatMessage(text="hello"), _session(Executor()))
    assert asyncio.run(reply) == "You said: hello"
    text = (EXAMPLES / "chat_minimal.py").read_text(encoding="utf-8")
    section = text.split("# --8<-- [start:minimal]")[1].split("# --8<-- [end:minimal]")[0]
    assert "from agent_adapter.viewer import ChatMessage, ChatSession, serve" in section
    assert "serve(agent=chat)" in section, "the docs snippet runs on its own"


def test_serve_snippets_use_no_command_line_arguments() -> None:
    for name in ("chat_any_agent.py", "chat_minimal.py"):
        assert "args." not in (EXAMPLES / name).read_text(encoding="utf-8"), name


def test_plain_agent_runs_its_flow_in_the_turn_and_streams(tmp_path: Path) -> None:
    module = _load("chat_any_agent.py")
    session = _session(Executor(module["registry"]))
    reply = module["chat"](ChatMessage(text="When can I visit the museum?"), session)
    text = asyncio.run(_collect(reply))
    assert text.strip() == "**museum: 09:00-18:00**, and a ticket costs **12.50 EUR**."
    assert session.state == {"questions": 1}


def test_langgraph_wrapper_sends_history_and_returns_the_last_message() -> None:
    module = _load("chat_wrappers.py")
    seen: list[Any] = []

    class App:
        async def ainvoke(self, state: dict[str, Any]) -> dict[str, Any]:
            seen.append(state)
            return {"messages": [*state["messages"], types.SimpleNamespace(content="hi there")]}

    history = (Message(role="user", content="a"), Message(role="assistant", content="b"))
    agent = module["langgraph_agent"](App())
    answer = asyncio.run(agent(ChatMessage(text="c"), _session(Executor(), history)))
    assert answer == "hi there"
    assert seen == [{"messages": [("user", "a"), ("assistant", "b"), ("user", "c")]}]


def test_crewai_wrapper_passes_the_question() -> None:
    module = _load("chat_wrappers.py")

    class Crew:
        async def kickoff_async(self, inputs: dict[str, str]) -> Any:
            return types.SimpleNamespace(raw="crew says: " + inputs["question"])

    agent = module["crewai_agent"](Crew())
    assert asyncio.run(agent(ChatMessage(text="why?"), _session(Executor()))) == "crew says: why?"


@pytest.fixture
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stand-in `google.genai.types`, so the ADK wrapper runs without google-adk."""
    genai = types.ModuleType("google.genai")
    genai_types = types.ModuleType("google.genai.types")
    genai_types.Content = lambda role, parts: types.SimpleNamespace(role=role, parts=parts)  # type: ignore[attr-defined]
    genai_types.Part = lambda text: types.SimpleNamespace(text=text)  # type: ignore[attr-defined]
    genai.types = genai_types  # type: ignore[attr-defined]
    google = sys.modules.get("google") or types.ModuleType("google")
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)


def test_adk_wrapper_makes_the_session_once_and_yields_final_text(fake_genai: None) -> None:
    module = _load("chat_wrappers.py")
    made: list[str] = []

    class Sessions:
        async def get_session(self, **key: str) -> Any:
            return key["session_id"] if key["session_id"] in made else None

        async def create_session(self, **key: str) -> None:
            made.append(key["session_id"])

    def event(text: str, final: bool) -> Any:
        content = types.SimpleNamespace(parts=[types.SimpleNamespace(text=text)])
        return types.SimpleNamespace(content=content, is_final_response=lambda: final)

    class Runner:
        app_name = "demo"
        session_service = Sessions()

        async def run_async(self, **kwargs: Any) -> AsyncIterator[Any]:
            assert kwargs["new_message"].parts[0].text == "hello"
            yield event("thinking", False)
            yield event("hi", True)

    agent = module["adk_agent"](Runner())
    for _ in range(2):
        out = asyncio.run(_collect(agent(ChatMessage(text="hello"), _session(Executor()))))
        assert out == "hi"
    assert made == ["ses_1"]


@pytest.fixture
def backend() -> Iterator[str]:
    """A local HTTP backend answering {"answer"} for the HTTP wrapper."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps({"answer": f"{body['session_id']}: {body['message']}"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            return None

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/chat"
    finally:
        server.shutdown()
        server.server_close()


def test_http_wrapper_posts_the_message_and_reads_the_answer(backend: str) -> None:
    module = _load("chat_wrappers.py")
    agent = module["http_agent"](backend)
    assert agent(ChatMessage(text="ping"), _session(Executor())) == "ses_1: ping"


def test_langgraph_wrapper_with_a_real_graph() -> None:
    graph_module = pytest.importorskip("langgraph.graph")
    module = _load("chat_wrappers.py")

    def echo(state: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [("assistant", "echo: " + state["messages"][-1].content)]}

    graph = graph_module.StateGraph(graph_module.MessagesState)
    graph.add_node("echo", echo)
    graph.add_edge(graph_module.START, "echo")
    graph.add_edge("echo", graph_module.END)
    agent = module["langgraph_agent"](graph.compile())
    assert asyncio.run(agent(ChatMessage(text="hi"), _session(Executor()))) == "echo: hi"


def test_every_chat_snippet_the_docs_include_exists() -> None:
    pages = ("chat-with-your-agent.md", "chat-wrappers.md")
    page = "".join((HOW_TO / name).read_text(encoding="utf-8") for name in pages)
    found = SNIPPET.findall(page)
    assert {name for _, name in found} >= {
        "minimal",
        "plain",
        "langgraph",
        "adk",
        "crewai",
        "http",
        "serve",
    }
    for file, name in found:
        text = (EXAMPLES / file).read_text(encoding="utf-8")
        assert f"# --8<-- [start:{name}]" in text and f"# --8<-- [end:{name}]" in text


def test_the_graph_page_shows_the_sample_flow_mermaid() -> None:
    flow = _load("viewer_fan_out.py")["flow"]
    page = (HOW_TO / "navigate-the-graph.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```mermaid\n(.*?)```", page, re.S)
    assert flow.to_mermaid().strip() in [block.strip() for block in blocks]
