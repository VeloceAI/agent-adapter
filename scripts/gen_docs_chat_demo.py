"""Write (or `--check`) docs/assets/demo/chat-demo.html, byte-stable, from a real chat host run."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import runpy
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # the checkout's agent_adapter, not an installed copy

from agent_adapter import Executor, Ledger  # noqa: E402
from agent_adapter.core.errors import ApprovalNotFound  # noqa: E402  (needs the path above)
from agent_adapter.core.types import Event  # noqa: E402
from agent_adapter.viewer.assets import export_chat_demo_html  # noqa: E402
from agent_adapter.viewer.chat.host import ChatHost  # noqa: E402
from agent_adapter.viewer.chat.routes import session_detail  # noqa: E402
from agent_adapter.viewer.chat.setup import build_host  # noqa: E402
from agent_adapter.viewer.runs import RunFolder  # noqa: E402
from scripts import chat_demo_data, gen_docs_demo  # noqa: E402
from scripts.chat_demo_data import Recorded  # noqa: E402

logger = logging.getLogger(__name__)

EXAMPLE = ROOT / "examples" / "chat_flows.py"
OUT = ROOT / "docs" / "assets" / "demo" / "chat-demo.html"
REVIEWER = "demo"
REASON = "approved by the recording"
TURN_WAIT_S = 30.0
# Chats of the demo, each with the suggested questions it replays, in order.
# The pattern each suggested question shows, by its place in the example's QUESTIONS.
LABELS = ("router", "fan-out, fan-in", "approval", "loop", "all together")
CHATS = (
    ("Museum visit", (0, 1, 2)),
    ("Playground by the river", (3,)),
    ("A rainy day", (4,)),
)


@dataclass
class _Marks:
    """A sink noting each run event's place in its turn's stream; it approves on the host loop."""

    host: ChatHost
    marks: dict[str, list[int | None]] = field(default_factory=dict)
    emitted: dict[str, list[str]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: Event) -> None:
        try:
            live = self.host.watch.find(event.run_id)
            with self.lock:
                self.marks.setdefault(event.run_id, []).append(
                    len(live.events) if live is not None else None
                )
                self.emitted.setdefault(event.run_id, []).append(str(event.type))
            if event.type == "approval.requested":
                loop = asyncio.get_running_loop()
                loop.call_soon(self._decide, loop, str(event.data["approval_id"]), 0)
        except Exception:  # a sink never raises; the build checks the counts afterwards
            logger.exception("chat demo: could not note %s of run %s", event.type, event.run_id)

    def _decide(self, loop: asyncio.AbstractEventLoop, approval_id: str, tries: int) -> None:
        """The reviewer's one decision, on the loop, as soon as the desk has the request."""
        try:
            self.host.desk.decide(approval_id, approved=True, reason=REASON)
        except ApprovalNotFound:
            if tries >= 100:
                logger.error("chat demo: approval %s never reached the desk", approval_id)
                return
            loop.call_soon(self._decide, loop, approval_id, tries + 1)


def _wait_for_end(host: ChatHost, turn_id: str) -> None:
    deadline = time.monotonic() + TURN_WAIT_S
    live = host.live(turn_id)
    while not live.ended:
        if time.monotonic() > deadline:
            raise RuntimeError(f"turn {turn_id} did not end within {TURN_WAIT_S:g} s")
        time.sleep(0.005)


def record(work: Path) -> Recorded:
    """Run every chat of the demo through a real chat host; what it sent and recorded."""
    module = runpy.run_path(str(EXAMPLE), run_name="docs_chat_demo")
    questions: tuple[str, ...] = module["QUESTIONS"]
    executor = Executor(module["registry"], ledger=Ledger(prices=module["PRICES"]))
    host = build_host(
        runs=work / "runs",
        agent=module["chat"],
        executor=executor,
        chat_dir=work / "chats",
        reviewer=REVIEWER,
        agent_name="city help desk",
    )
    marks = _Marks(host)
    host.executor.add_sink(marks)
    recorded = Recorded(runs=work / "runs", info=host.info())
    try:
        with gen_docs_demo._no_waits(), contextlib.redirect_stdout(io.StringIO()):
            for title, picks in CHATS:
                session = host.store.create(title)
                for pick in picks:
                    started = host.start(session.id, questions[pick], [])
                    _wait_for_end(host, started["turn_id"])
                    live = host.live(started["turn_id"])
                    recorded.streams[started["turn_id"]] = [dict(e) for e in live.events]
                    recorded.labels[started["turn_id"]] = LABELS[pick]
                recorded.chats.append(session_detail(host, session.id))
    finally:
        host.close()
    recorded.marks = marks.marks
    recorded.emitted = marks.emitted
    return recorded


def build() -> str:
    """The demo page: the recorded chats, their streams and runs, inline and offline."""
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        recorded = record(work / "recorded")
        fixed = work / "fixed"
        fixed.mkdir()
        gen_docs_demo.normalise(recorded.runs, fixed)
        ids = chat_demo_data.id_map(recorded.runs, fixed)
        chats = chat_demo_data.plan(recorded)
        chat_demo_data.retime(fixed, recorded, chats, ids)
        folder = RunFolder(fixed)
        folder.refresh()
        summaries = folder.summaries()
        details = {s["key"]: folder.detail(s["key"], paged=False) for s in summaries}
        data = chat_demo_data.page_data(recorded, chats, ids, summaries, details)
    return export_chat_demo_html(data)


def main(argv: list[str]) -> int:
    """Write the demo, or with `--check` exit 1 when the committed one is out of date."""
    text = build()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            name = OUT.relative_to(ROOT).as_posix()
            sys.stderr.write(f"{name} is out of date: run python scripts/gen_docs_chat_demo.py\n")
            return 1
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(text.encode("utf-8"))
    sys.stdout.write(f"wrote {OUT.relative_to(ROOT).as_posix()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
