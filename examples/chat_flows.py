"""A city visitor help desk to chat with: five scripted flows, one per agent pattern.

Run it and open the printed link; Ctrl+C stops it. `--check` instead runs every suggested
question once in this process and prints the answers (no browser, no network, no API key).
"""

import argparse
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any

from pydantic import BaseModel, Field

from agent_adapter import (
    AutoApprove,
    CallableModel,
    Executor,
    Flow,
    FlowRunner,
    Ledger,
    LLMRequest,
    Message,
    Prices,
    Registry,
    ToolContext,
)
from agent_adapter.errors import ConfigError
from agent_adapter.flow import Call, Join, Map, Merge, Route, node_model
from agent_adapter.viewer import ChatMessage, ChatSession, serve

registry = Registry()

MUSEUMS: dict[str, dict[str, Any]] = {
    "City Museum": {"closed": "Monday", "hours": "10:00-18:00", "adult": 14.0, "family": True},
    "Science Hall": {"closed": "", "hours": "09:00-17:00", "adult": 11.5, "family": True},
    "Harbour Gallery": {"closed": "Tuesday", "hours": "11:00-19:00", "adult": 9.0, "family": False},
    "Clock Tower": {"closed": "", "hours": "10:00-16:00", "adult": 6.0, "family": False},
}
PARKS: tuple[dict[str, Any], ...] = (
    {"name": "Willow Gardens", "km": 0.8, "playground": False, "river": True},
    {"name": "Hilltop Common", "km": 1.2, "playground": True, "river": False},
    {"name": "Riverside Park", "km": 1.6, "playground": True, "river": True},
    {"name": "Old Mill Green", "km": 3.1, "playground": True, "river": True},
)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
QUESTIONS = (
    "Is the City Museum open on Monday?",
    "Compare ticket prices at three museums",
    "Book two tickets for Saturday",
    "Find a park with a playground near the river",
    "Plan a rainy day for a family",
)


async def _lookup() -> None:
    await asyncio.sleep(0.2)  # a real lookup waits on a service; parallel calls overlap


@registry.tool(parallel_safe=True)
async def opening_hours(place: str, day: str) -> dict[str, Any]:
    """Whether a museum is open on a day, and its hours."""
    await _lookup()
    info = MUSEUMS[place]
    return {"place": place, "day": day, "open": info["closed"] != day, "hours": info["hours"]}


@registry.tool(parallel_safe=True)
async def ticket_price(place: str) -> dict[str, Any]:
    """An adult ticket at a museum, in EUR."""
    await _lookup()
    return {"place": place, "adult_eur": MUSEUMS[place]["adult"]}


@registry.tool(parallel_safe=True)
async def family_check(place: str) -> dict[str, Any]:
    """Whether a museum suits children."""
    await _lookup()
    return {"place": place, "family": MUSEUMS[place]["family"], "hours": MUSEUMS[place]["hours"]}


@registry.tool
async def find_parks(near: str, within_km: float) -> list[dict[str, Any]]:
    """Parks by the river (or away from it) within a distance of the centre."""
    await _lookup()
    return [p for p in PARKS if p["river"] == (near == "river") and p["km"] <= within_km]


@registry.tool
async def forecast(day: str) -> dict[str, Any]:
    """The weather for a day."""
    await _lookup()
    return {"day": day, "sky": "rain", "high_c": 14}


@registry.tool(confirm=True, effects="additive")
async def book_tickets(place: str, day: str, people: int) -> dict[str, Any]:
    """Book museum tickets (needs approval)."""
    await _lookup()
    return {"place": place, "day": day, "people": people, "ref": "CM-0042"}


# --8<-- [start:writer]
async def write(request: LLMRequest) -> dict[str, Any]:
    """Stands in for a model: it returns the draft it is given, with made-up token counts."""
    draft = request.messages[-1].content
    return {"text": draft, "usage": {"input_tokens": 60 + len(draft) // 4, "output_tokens": 40}}


WRITER = CallableModel(write, provider="local", model="writer")
PRICES = Prices.empty().merged(
    {"version": "demo", "models": {"writer": {"provider": "local", "input": 1.0, "output": 4.0}}}
)
Node = Callable[[Any, ToolContext], Awaitable[dict[str, Any]]]


def reply(draft: Callable[[Any], str]) -> Node:
    """The last node of every flow: the writer model turns the flow's draft into the answer."""

    async def write_answer(state: Any, ctx: ToolContext) -> dict[str, Any]:
        model = node_model(ctx)
        if model is None:
            raise ConfigError("the reply node needs FlowRunner(model=...)")
        message = Message(role="user", content=draft(state))
        return {"answer": (await model.generate(LLMRequest(messages=(message,)))).text}

    return write_answer


# --8<-- [end:writer]


# --8<-- [start:router]
class Ask(BaseModel):
    question: str
    place: str = ""
    day: str = ""
    topic: str = ""
    facts: dict[str, Any] = Field(default_factory=dict)
    answer: str = ""


def classify(state: Ask) -> dict[str, Any]:
    text = state.question.lower()
    place = next((name for name in MUSEUMS if name.lower() in text), "City Museum")
    day = next((d for d in DAYS if d.lower() in text), "Saturday")
    topic = "hours" if "open" in text else "prices" if "price" in text else "other"
    return {"topic": topic, "place": place, "day": day}


def ask_draft(s: Ask) -> str:
    f = s.facts
    if s.topic == "hours" and f["open"]:
        return f"Yes: the **{s.place}** is open on {s.day}, {f['hours']}."
    if s.topic == "hours":
        return f"No: the **{s.place}** is closed on {s.day}s. Other days: {f['hours']}."
    if s.topic == "prices":
        return f"An adult ticket at the {s.place} costs **{f['adult_eur']:.2f} EUR**."
    return "I can check hours and prices, compare museums, book tickets and find parks."


ask_flow = (
    Flow("ask", state=Ask)
    .node("classify", classify)
    .node("hours", Call("opening_hours", lambda s: {"place": s.place, "day": s.day}, "facts"))
    .node("prices", Call("ticket_price", lambda s: {"place": s.place}, "facts"))
    .node("help", lambda s: {"facts": {}})
    .node("reply", reply(ask_draft))
    .route(
        "classify",
        Route(lambda s: s.topic, {"hours": "hours", "prices": "prices", "other": "help"}),
    )
    .edge("hours", "reply")
    .edge("prices", "reply")
    .edge("help", "reply")
    .end("reply")
)
# --8<-- [end:router]


# --8<-- [start:fanout]
class Compare(BaseModel):
    question: str
    lookups: list[dict[str, Any]] = Field(default_factory=list)
    prices: Annotated[list[dict[str, Any]], Merge.append] = Field(default_factory=list)
    table: str = ""
    answer: str = ""


def pick_museums(state: Compare) -> dict[str, Any]:
    return {"lookups": [{"place": name} for name in list(MUSEUMS)[:3]]}


def cheapest_first(state: Compare) -> dict[str, Any]:
    rows = sorted(state.prices, key=lambda p: p["adult_eur"])
    lines = ["| Museum | Adult |", "|---|---|"]
    lines += [f"| {p['place']} | {p['adult_eur']:.2f} EUR |" for p in rows]
    return {"table": "\n".join(lines)}


compare_flow = (
    Flow("compare_prices", state=Compare)
    .node("plan", pick_museums)
    .node("prices", Map("lookups", tool="ticket_price", into="prices"), concurrency=3)
    .node("combine", Join(["prices"], fn=cheapest_first))
    .node("reply", reply(lambda s: f"Cheapest first:\n\n{s.table}"))
    .edge("plan", "prices")
    .edge("prices", "combine")
    .edge("combine", "reply")
    .end("reply")
)
# --8<-- [end:fanout]


# --8<-- [start:approval]
class Booking(BaseModel):
    question: str
    place: str = "City Museum"
    day: str = "Saturday"
    people: int = 2
    price: dict[str, Any] = Field(default_factory=dict)
    booking: dict[str, Any] = Field(default_factory=dict)
    answer: str = ""


def booking_details(state: Booking) -> dict[str, Any]:
    text = state.question.lower()
    day = next((d for d in DAYS if d.lower() in text), state.day)
    people = next((n for word, n in (("one", 1), ("two", 2), ("three", 3)) if word in text), 2)
    return {"day": day, "people": people}


def booking_draft(s: Booking) -> str:
    total = s.price["adult_eur"] * s.people
    b = s.booking
    trip = f"{b['place']}, {b['day']}"
    return f"Booked: **{b['people']} tickets**, {trip} ({total:.2f} EUR, ref `{b['ref']}`)."


booking_flow = (
    Flow("booking", state=Booking)
    .node("details", booking_details)
    .node("price", Call("ticket_price", lambda s: {"place": s.place}, "price"))
    .node(
        "book",
        Call(
            "book_tickets",
            lambda s: {"place": s.place, "day": s.day, "people": s.people},
            "booking",
        ),
    )
    .node("reply", reply(booking_draft))
    .edge("details", "price")
    .edge("price", "book")
    .edge("book", "reply")
    .end("reply")
)
# --8<-- [end:approval]


# --8<-- [start:loop]
class ParkSearch(BaseModel):
    question: str
    within_km: float = 0.5
    found: list[dict[str, Any]] = Field(default_factory=list)
    matches: list[str] = Field(default_factory=list)
    answer: str = ""


def enough_parks(state: ParkSearch) -> dict[str, Any]:
    matches = [p["name"] for p in state.found if p["playground"]]
    return {"matches": matches} if matches else {"within_km": state.within_km * 2}


def park_draft(s: ParkSearch) -> str:
    if not s.matches:
        return "No park with a playground by the river is close enough."
    return f"**{s.matches[0]}** has a playground by the river, within {s.within_km:g} km."


park_flow = (
    Flow("park_search", state=ParkSearch)
    .node(
        "search", Call("find_parks", lambda s: {"near": "river", "within_km": s.within_km}, "found")
    )
    .node("check", enough_parks)
    .node("reply", reply(park_draft))
    .edge("search", "check")
    .route(
        "check",
        Route(lambda s: "done" if s.matches else "widen", {"widen": "search", "done": "reply"}),
        max_loops=3,
        on_exhausted="reply",
    )
    .end("reply")
)
# --8<-- [end:loop]


# --8<-- [start:combined]
INDOOR = (("Harbour Gallery", "City Museum"), ("Science Hall", "Clock Tower"))


class DayPlan(BaseModel):
    question: str
    weather: dict[str, Any] = Field(default_factory=dict)
    round: int = 0
    lookups: list[dict[str, Any]] = Field(default_factory=list)
    checked: Annotated[list[dict[str, Any]], Merge.append] = Field(default_factory=list)
    picks: list[str] = Field(default_factory=list)
    answer: str = ""


def next_places(state: DayPlan) -> dict[str, Any]:
    batch = INDOOR[min(state.round, len(INDOOR) - 1)]
    return {"lookups": [{"place": name} for name in batch], "round": state.round + 1}


def shortlist(state: DayPlan) -> dict[str, Any]:
    return {"picks": [c["place"] for c in state.checked if c["family"]]}


def day_draft(s: DayPlan) -> str:
    if not s.picks:
        return f"It will be {s.weather['sky']}, up to {s.weather['high_c']} C: a day in the parks."
    stops = " then ".join(f"**{place}**" for place in s.picks)
    return f"Rain all day (up to {s.weather['high_c']} C), so stay inside: {stops}."


day_flow = (
    Flow("day_plan", state=DayPlan)
    .node("forecast", Call("forecast", {"day": "Sunday"}, "weather"))
    .node("outdoor", lambda s: {"picks": []})
    .node("candidates", next_places)
    .node("check", Map("lookups", tool="family_check", into="checked"), concurrency=2)
    .node("shortlist", Join(["check"], fn=shortlist))
    .node("reply", reply(day_draft))
    .route("forecast", Route(lambda s: s.weather["sky"], {"rain": "candidates", "sun": "outdoor"}))
    .edge("candidates", "check")
    .edge("check", "shortlist")
    .route(
        "shortlist",
        Route(
            lambda s: "done" if len(s.picks) >= 2 else "more",
            {"more": "candidates", "done": "reply"},
        ),
        max_loops=2,
        on_exhausted="reply",
    )
    .edge("outdoor", "reply")
    .end("reply")
)
# --8<-- [end:combined]


# --8<-- [start:agent]
def pick_flow(text: str) -> Flow[Any]:
    """The flow for a question, by its words."""
    words = text.lower()
    for keys, flow in (
        (("book",), booking_flow),
        (("compare", "prices at"), compare_flow),
        (("park", "playground"), park_flow),
        (("plan",), day_flow),
    ):
        if any(key in words for key in keys):
            return flow
    return ask_flow


async def chat(message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
    """Runs the question's flow in this turn's run, then streams the answer in chunks."""
    flow = pick_flow(message.text)
    start: dict[str, Any] = {"question": message.text}
    if flow is booking_flow and session.state.get("place"):
        start["place"] = session.state["place"]  # the museum asked about earlier in this chat
    runner = FlowRunner(session.executor, model=WRITER, trace_state=True)
    result = await runner.run(flow, start, ctx=session.context())
    place = getattr(result.state, "place", "")
    if place:
        session.state["place"] = place  # remembered for later turns; editable in the page
    session.state["questions"] = session.state.get("questions", 0) + 1
    words = result.state.answer.split(" ")
    for at in range(0, len(words), 3):
        yield " ".join(words[at : at + 3]) + (" " if at + 3 < len(words) else "")
        await asyncio.sleep(0.05)


# --8<-- [end:agent]


async def check() -> int:
    """Every suggested question once, in one chat; 0 when each one got an answer."""
    executor = Executor(
        registry, ledger=Ledger(prices=PRICES), approval=AutoApprove(reason="check")
    )
    state: dict[str, Any] = {}
    failed = 0
    for number, question in enumerate(QUESTIONS, 1):
        session = ChatSession("check", (), state, executor, run_id=f"check-{number}")
        answer = "".join([chunk async for chunk in chat(ChatMessage(question), session)])
        print(f"{pick_flow(question).name}: {answer}")
        failed += not answer.strip()
    return 1 if failed else 0


def main(port: int = 8780, *, open_browser: bool = True) -> None:
    """Serve the chat page until Ctrl+C."""
    # --8<-- [start:serve]
    executor = Executor(registry, ledger=Ledger(prices=PRICES))  # prices fill the Cost tab
    serve(agent=chat, executor=executor, port=port, open_browser=open_browser)
    # --8<-- [end:serve]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--check", action="store_true", help="run every question offline")
    parser.add_argument("--no-browser", action="store_true", help="only print the link")
    args = parser.parse_args()
    if args.check:
        raise SystemExit(asyncio.run(check()))
    main(args.port, open_browser=not args.no_browser)
