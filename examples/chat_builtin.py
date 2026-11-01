"""Chat with the built-in tool loop: a scripted model, a weather tool and a booking to approve.

Run it and open the printed link. `--check` instead starts the server on a free port, sends one
message through the HTTP API, approves the booking, prints the answer and exits (no browser,
no network, no API key).
"""

import argparse
import json
import re
import signal
import threading
import urllib.request
from collections.abc import Iterator
from typing import Any

from agent_adapter import (
    CallableModel,
    Executor,
    Ledger,
    LLMRequest,
    LLMResponse,
    Message,
    Prices,
    Registry,
    RunLimits,
    ToolCall,
    Usage,
)
from agent_adapter.viewer import serve

registry = Registry()


@registry.tool
def weather(city: str) -> dict[str, Any]:
    """Today's forecast for a city."""
    return {"city": city, "sky": "sunny", "high_c": 24, "low_c": 16}


@registry.tool(confirm=True, effects="additive")
def book_table(restaurant: str, people: int) -> str:
    """Book a restaurant table (needs approval)."""
    return f"booked {restaurant} for {people}"


# --8<-- [start:model]
def scripted_model(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: weather first, then the booking if asked, then the answer."""
    last = max(i for i, m in enumerate(request.messages) if m.role == "user")
    asked = request.messages[last].content
    results = [m for m in request.messages[last:] if m.role == "tool"]  # this turn's tool results
    usage = Usage(input_tokens=200 + 40 * len(results), output_tokens=30)
    wants: list[tuple[str, dict[str, Any]]] = [("weather", {"city": "Lisbon"})]
    if "book" in asked.lower():
        wants.append(("book_table", {"restaurant": "Casa Azul", "people": 2}))
    if len(results) < len(wants):
        name, args = wants[len(results)]
        call = ToolCall(name=name, arguments=args, id=f"call_{name}")
        return LLMResponse(
            provider="local",
            model="scripted",
            tool_calls=(call,),
            finish_reason="tool_calls",
            usage=usage,
            message=Message(role="assistant", tool_calls=(call,)),
        )
    lines = ["Here is what I found:", "", f"- **Weather:** `{results[0].content}`"]
    if len(results) > 1:
        lines.append(f"- **Booking:** {results[1].content}")
    text = "\n".join(lines)
    return LLMResponse(
        provider="local",
        model="scripted",
        text=text,
        usage=usage,
        message=Message(role="assistant", content=text),
    )


# --8<-- [end:model]


PRICES = Prices.empty().merged(
    {"version": "demo", "models": {"scripted": {"provider": "local", "input": 1.0, "output": 4.0}}}
)


def main(port: int = 8780, *, open_browser: bool = True, out: Any = None) -> None:
    """Serve the chat page until Ctrl+C."""
    # --8<-- [start:serve]
    model = CallableModel(scripted_model, provider="local", model="scripted")
    executor = Executor(registry, ledger=Ledger(prices=PRICES), limits=RunLimits(max_calls=20))
    serve(
        model=model,
        registry=registry,
        executor=executor,  # optional: here it adds prices, so the Cost tab shows dollars
        system="You help plan a day out.",
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


def _post(url: str, token: str, csrf: str, path: str, body: dict[str, Any]) -> Any:
    """A JSON POST with the headers the page sends: token, Origin and CSRF."""
    headers = {
        "X-AA-Token": token,
        "X-AA-CSRF": csrf,
        "Origin": url.rstrip("/"),
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url + path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def _events(url: str, token: str, path: str) -> Iterator[dict[str, Any]]:
    """A turn's stream events, read line by line as they arrive (`: ping` lines skipped)."""
    request = urllib.request.Request(url + path, headers={"X-AA-Token": token})
    with urllib.request.urlopen(request, timeout=30) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip()
            if line.startswith("data: "):
                yield json.loads(line[len("data: ") :])


def one_turn(link: str, message: str) -> dict[str, Any]:
    """Send `message` over the HTTP API, approve each pending call, return the finished turn."""
    url, token = link.split("#token=")
    with urllib.request.urlopen(url, timeout=30) as response:  # the page holds the CSRF token
        page = response.read().decode("utf-8")
    csrf = re.search(r'name="aa-csrf" content="([^"]+)"', page).group(1)  # type: ignore[union-attr]
    session = _post(url, token, csrf, "api/chat/sessions", {"title": "check"})["session"]["id"]
    turn = _post(url, token, csrf, f"api/chat/sessions/{session}/turns", {"text": message})
    for event in _events(url, token, f"api/chat/turns/{turn['turn_id']}/stream?since=0"):
        if event["type"] == "approval" and event["status"] == "pending":
            decide = f"api/chat/approvals/{event['approval_id']}/decide"
            _post(url, token, csrf, decide, {"approved": True, "reason": "check run"})
        if event["type"] in ("done", "error"):
            return dict(event["turn"])
    raise RuntimeError("the stream ended before the turn did")


def check() -> None:
    """Start on a free port, run one turn through the API, print its answer, stop."""
    link = _Link()
    result: dict[str, Any] = {}

    def drive() -> None:
        try:
            link.ready.wait(30)
            found = re.search(r"open (\S+#token=\S+)", link.text)
            if found is None:
                raise RuntimeError("serve printed no link")
            question = "What is the weather in Lisbon? Book a table too."
            result.update(one_turn(found.group(1), question))
        except Exception as exc:  # reported by the main thread below
            result["failure"] = repr(exc)
        finally:
            signal.raise_signal(signal.SIGINT)  # serve stops on Ctrl+C

    threading.Thread(target=drive, daemon=True).start()
    main(port=0, open_browser=False, out=link)
    if "failure" in result:
        raise SystemExit(f"check failed: {result['failure']}")
    print(result["status"], f"{len(result['tools'])} tool calls", f"${result['cost_usd']:.6f}")
    print(result["answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--check", action="store_true", help="run one turn offline and exit")
    parser.add_argument("--no-browser", action="store_true", help="only print the link")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        main(args.port, open_browser=not args.no_browser)
