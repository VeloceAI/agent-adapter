"""Chat memory across chats: a ticket desk that remembers what a user told it last time.

Run it and open the printed link: tell it your seat preference in one chat, then start a new
chat and ask for a seat. `--check` instead starts on a free port, runs both chats through the
HTTP API, prints what the second chat recalled and exits (no browser, no network, no key).
"""

import argparse
import json
import re
import signal
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any

from agent_adapter import CallableModel, LLMRequest
from agent_adapter.memory import SQLiteMemory
from agent_adapter.viewer import serve


# --8<-- [start:model]
def scripted_model(request: LLMRequest) -> str:
    """Stands in for a real model: writes facts when asked, else answers with its notes."""
    system = " ".join(m.content for m in request.messages if m.role == "system")
    asked = [m.content for m in request.messages if m.role == "user"][-1]
    if system.startswith("List lasting facts"):
        said = " ".join(line for line in asked.splitlines() if line.startswith("user:"))
        return "\n".join(f"- {fact}" for fact in re.findall(r"I prefer [^.]+", said)) or "NONE"
    fenced = " ".join(
        m.content for m in request.messages if "<untrusted-memory-notes>" in m.content
    )
    notes = [line[2:] for line in fenced.splitlines() if line.startswith("- ")]
    if notes:
        return f"Booked for you, keeping in mind: {'; '.join(notes)}."
    return "Noted. How else can the ticket desk help?"


# --8<-- [end:model]

MODEL = CallableModel(scripted_model, provider="local", model="scripted")


def main(folder: Path, port: int = 8780, *, open_browser: bool = True, out: Any = None) -> None:
    """Serve the chat page for user ann until Ctrl+C."""
    # --8<-- [start:serve]
    serve(
        model=MODEL,
        memory=SQLiteMemory(folder / "memory.db", retention_days=90),
        user_id="ann",  # memory is kept per user; tools see ctx.user_id
        memory_save="facts",  # "summary", "facts" or "none" (the default)
        folder=folder / "runs",
        chat_dir=folder / "chats",
        port=port,
        open_browser=open_browser,
        out=out,
    )
    # --8<-- [end:serve]


class _Link:
    """Collects what `serve` prints and signals once the link line is there."""

    def __init__(self) -> None:
        self.text = ""
        self.ready = threading.Event()

    def write(self, text: str) -> int:
        self.text += text
        if "#token=" in self.text:
            self.ready.set()
        return len(text)

    def flush(self) -> None:
        return None


def _call(link: str, csrf: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """A GET, or a JSON POST with the headers the page sends."""
    url, token = link.split("#token=")
    headers = {"X-AA-Token": token}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        origin = url.rstrip("/")
        headers.update({"X-AA-CSRF": csrf, "Origin": origin, "Content-Type": "application/json"})
    request = urllib.request.Request(url + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
    return raw if path.endswith("stream?since=0") else json.loads(raw)


def chat_once(link: str, csrf: str, text: str) -> list[dict[str, Any]]:
    """Send `text` in a new chat and return the turn's stream events."""
    sid = _call(link, csrf, "api/chat/sessions", {"title": text})["session"]["id"]
    turn = _call(link, csrf, f"api/chat/sessions/{sid}/turns", {"text": text})
    stream = _call(link, csrf, f"api/chat/turns/{turn['turn_id']}/stream?since=0")
    return [json.loads(line[6:]) for line in stream.splitlines() if line.startswith("data: ")]


def check(folder: Path) -> None:
    """Two chats through the HTTP API: the second recalls what the first saved."""
    printed = _Link()
    result: dict[str, Any] = {}

    def drive() -> None:
        try:
            printed.ready.wait(30)
            link = re.search(r"open (\S+#token=\S+)", printed.text).group(1)  # type: ignore[union-attr]
            with urllib.request.urlopen(link.split("#")[0], timeout=30) as response:
                page = response.read().decode("utf-8")
            csrf = re.search(r'name="aa-csrf" content="([^"]+)"', page).group(1)  # type: ignore[union-attr]
            chat_once(link, csrf, "I prefer aisle seats. Any concerts on Friday?")
            result["events"] = chat_once(link, csrf, "Book two seats for Friday's concert.")
            result["records"] = _call(link, csrf, "api/chat/memory")["records"]
        except Exception as exc:  # reported by the main thread below
            result["failure"] = repr(exc)
        finally:
            signal.raise_signal(signal.SIGINT)  # serve stops on Ctrl+C

    threading.Thread(target=drive, daemon=True).start()
    main(folder, port=0, open_browser=False, out=printed)
    if "failure" in result:
        raise SystemExit(f"check failed: {result['failure']}")
    events = result["events"]
    recalled = next(e for e in events if e["type"] == "memory")["recalled"]
    print("recalled:", [r["text"] for r in recalled])
    print("answer:", "".join(e["text"] for e in events if e["type"] == "text"))
    print("stored:", len(result["records"]), "record(s) for ann")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--check", action="store_true", help="run two chats offline and exit")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if args.check:
            check(Path(tmp))
        else:
            main(Path(tmp), args.port)
