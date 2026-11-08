"""The chat demo's page data from a recorded chat: replay timelines, fixed ids and times."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts import gen_docs_demo

# The chat's own ids (sessions, turns): 16 hex characters.
CHAT_IDS = re.compile(r"\b(ses|turn)_[0-9a-f]{16}\b")
GAP_MS = {"state": 120, "tool": 220, "usage": 140, "text": 110, "done": 260, "error": 260}
RUN_GAP_MS = {"flow.started": 160, "node.started": 300, "node.finished": 160}
APPROVAL_SHOW_MS = 350  # the approval card appears
APPROVAL_WAIT_MS = 1800  # it waits this long before the recorded decision arrives
MIN_TURN_MS, MAX_TURN_MS = 3000.0, 6000.0
TURN_SPACING_S = 120.0  # recorded turns start this far apart
EVENT_SPACING_S = 0.0001  # run events revealed in one step stay in order
LIVE_COMMAND = "python examples/chat_flows.py"


@dataclass
class Recorded:
    """What the chat host sent and wrote while the demo's chats ran."""

    runs: Path
    info: dict[str, Any]
    chats: list[dict[str, Any]] = field(default_factory=list)  # session details, in order
    streams: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # by turn id
    marks: dict[str, list[int | None]] = field(default_factory=dict)  # by run id
    emitted: dict[str, list[str]] = field(default_factory=dict)  # event types, by run id
    labels: dict[str, str] = field(default_factory=dict)  # the pattern a turn shows, by turn id


@dataclass
class _Step:
    gap: float
    fixed: bool  # an approval's wait: never scaled
    sse: dict[str, Any] | None = None
    run: int = 0


def id_map(runs: Path, fixed: Path) -> dict[str, str]:
    """Each recorded run, call and approval id with the fixed id `normalise` gave it."""
    originals = sorted(runs.glob("*.jsonl"), key=gen_docs_demo._order)
    copies = sorted(fixed.glob("*.jsonl"))
    if len(originals) != len(copies):
        raise RuntimeError(f"{len(originals)} recorded runs, {len(copies)} fixed")
    found: dict[str, str] = {}
    for original, copied in zip(originals, copies, strict=True):
        before = [m.group(0) for m in gen_docs_demo.IDS.finditer(original.read_text("utf-8"))]
        after = [m.group(0) for m in gen_docs_demo.IDS.finditer(copied.read_text("utf-8"))]
        if len(before) != len(after):
            raise RuntimeError(f"{copied.name} does not line up with {original.name}")
        for old, new in zip(before, after, strict=True):
            if found.setdefault(old, new) != new:
                raise RuntimeError(f"{old} was fixed twice, differently")
    return found


def timeline(
    stream: list[dict[str, Any]], marks: list[int | None], run_types: list[str]
) -> list[dict[str, Any]]:
    """Replay steps, `{"at": ms, "sse": event}` or `{"at": ms, "run": events shown}`, in order."""
    if len(marks) != len(run_types):
        raise RuntimeError(f"{len(marks)} marks for {len(run_types)} run events")
    steps: list[_Step] = []
    shown = 0

    def reveal(upto: int) -> None:
        nonlocal shown
        while shown < upto:
            gap = RUN_GAP_MS.get(run_types[shown])
            if gap is not None or not steps or steps[-1].sse is not None:
                steps.append(_Step(float(gap or 0), False))
            shown += 1
            steps[-1].run = shown

    for index, event in enumerate(stream):
        # a mark counts the stream events sent before its run event
        reveal(sum(1 for mark in marks if mark is not None and mark <= index))
        steps.append(_Step(*_gap(event), sse=event))
    reveal(len(run_types))
    return _paced(steps)


def _gap(event: dict[str, Any]) -> tuple[float, bool]:
    if event["type"] == "approval":
        if event.get("status") == "pending":
            return APPROVAL_SHOW_MS, False
        return APPROVAL_WAIT_MS, True
    return float(GAP_MS.get(str(event["type"]), 100)), False


def _paced(steps: list[_Step]) -> list[dict[str, Any]]:
    """Steps at absolute times, scaled so a turn takes 3-6 s besides its approval waits."""
    if steps:
        steps[0].gap = 0.0
    if sum(s.gap for s in steps if s.fixed) > MAX_TURN_MS:
        raise RuntimeError("an approval wait is longer than a whole turn")
    scalable = sum(s.gap for s in steps if not s.fixed) or 1.0
    factor = min(max(MIN_TURN_MS / scalable, 1.0), MAX_TURN_MS / scalable)
    out: list[dict[str, Any]] = []
    clock = 0.0
    for step in steps:
        clock += step.gap if step.fixed else step.gap * factor
        at = round(clock)
        out.append(
            {"at": at, "sse": step.sse} if step.sse is not None else {"at": at, "run": step.run}
        )
    return out


def _fix_times(turn: dict[str, Any], steps: list[dict[str, Any]], start: float) -> None:
    """Times of a turn and of its stream as the replay shows them, from a fixed start."""
    first: dict[str, int] = {}
    decided: dict[str, int] = {}
    took: dict[str, float] = {}
    for step in steps:
        event = step.get("sse")
        if not event:
            continue
        if event["type"] == "tool":
            first.setdefault(event["call_id"], step["at"])
            if event.get("ms") is not None:
                took[event["call_id"]] = float(step["at"] - first[event["call_id"]])
                event["ms"] = took[event["call_id"]]
        elif event["type"] == "approval":
            event["created_at"] = round(
                start + first.setdefault(event["approval_id"], step["at"]) / 1000, 3
            )
            if event.get("decided_at"):
                decided[event["approval_id"]] = step["at"]
                event["decided_at"] = round(start + step["at"] / 1000, 3)
    turn["started"] = start
    turn["ended"] = round(start + (steps[-1]["at"] if steps else 0) / 1000, 3)
    for tool in turn["tools"]:
        tool["ms"] = took.get(tool["call_id"], tool["ms"])
    for approval in turn["approvals"]:
        approval["created_at"] = round(start + first.get(approval["approval_id"], 0) / 1000, 3)
        if approval.get("decided_at"):
            approval["decided_at"] = round(
                start + decided.get(approval["approval_id"], 0) / 1000, 3
            )
    for step in steps:
        event = step.get("sse")
        if event and event["type"] in ("done", "error"):
            event["turn"] = copy.deepcopy(turn)


def plan(recorded: Recorded) -> list[dict[str, Any]]:
    """Every chat with its turns' replay steps and fixed times; ids still as recorded."""
    clock = gen_docs_demo.BASE_TS
    chats = []
    for detail in recorded.chats:
        turns = []
        for view in detail["turns"]:
            run_id = view["run_id"]
            stream = copy.deepcopy(recorded.streams[view["id"]])
            marks, types = recorded.marks.get(run_id, []), recorded.emitted.get(run_id, [])
            steps = timeline(stream, marks, types)
            turn = copy.deepcopy(view)
            _fix_times(turn, steps, clock)
            clock += TURN_SPACING_S
            states = [s["sse"]["state"] for s in steps if s.get("sse", {}).get("type") == "state"]
            label = recorded.labels.get(view["id"], "")
            state = states[-1] if states else {}
            turns.append({"label": label, "turn": turn, "steps": steps, "state": state})
        session = dict(detail["session"], running=None)
        session["created"] = turns[0]["turn"]["started"] - 1 if turns else clock
        session["updated"] = turns[-1]["turn"]["ended"] if turns else clock
        chats.append({"session": session, "turns": turns})
    return chats


def retime(
    fixed: Path, recorded: Recorded, chats: list[dict[str, Any]], ids: dict[str, str]
) -> None:
    """Move each fixed run onto its turn's replay clock, after checking it holds what was sent."""
    for chat in chats:
        for turn in chat["turns"]:
            original = turn["turn"]["run_id"]
            (path,) = fixed.glob(f"*.{ids[original]}.*.jsonl")
            records = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
            sent = recorded.emitted.get(original, [])
            if [r["type"] for r in records if r.get("kind") == "event"] != sent:
                raise RuntimeError(f"run {original}: the recorded events are not the ones sent")
            times = _event_times(turn["steps"], turn["turn"]["started"], len(sent))
            _move(records, times)
            path.unlink()
            name = f"{int(times[0] * 1000):013d}.{ids[original]}.1.jsonl" if times else path.name
            lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in records]
            (fixed / name).write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _event_times(steps: list[dict[str, Any]], start: float, count: int) -> list[float]:
    """When the replay shows each run event, as a timestamp from the turn's start."""
    times: list[float] = []
    end = steps[-1]["at"] if steps else 0
    for index in range(count):
        at = next((s["at"] for s in steps if s.get("run", 0) > index), end)
        times.append(round(start + at / 1000 + index * EVENT_SPACING_S, 6))
    return times


def _move(records: list[dict[str, Any]], times: list[float]) -> None:
    """Records at the new times; durations (`*_ms`) follow their calls' new start times."""
    index = 0
    began: dict[str, float] = {}
    for record in records:
        if record.get("kind") == "meta" and times:
            record["started"] = times[0]
        if "ts" not in record or not times:
            continue
        record["ts"] = times[min(index, len(times) - 1)]
        if record.get("kind") != "event":
            continue
        index += 1
        if str(record["type"]).endswith(".started"):
            began[record["call_id"]] = record["ts"]
        for key, value in record["data"].items():
            if key.endswith("_ms") and isinstance(value, float):
                since = began.get(record["call_id"], record["ts"])
                record["data"][key] = round((record["ts"] - since) * 1000, 3)


def _renumbered(data: Any, ids: dict[str, str]) -> Any:
    """`data` with run, call and approval ids as `normalise` fixed them, chat ids renumbered."""
    text = json.dumps(data, ensure_ascii=False)
    known = dict(ids)

    def run_id(match: re.Match[str]) -> str:
        found = match.group(0)
        if found not in known:
            known[found] = f"{match.group(1)}_{len(known) + 1:032x}"
        return known[found]

    chat_ids: dict[str, str] = {}

    def chat_id(match: re.Match[str]) -> str:
        found = match.group(0)
        if found not in chat_ids:
            chat_ids[found] = f"{match.group(1)}_{len(chat_ids) + 1:016x}"
        return chat_ids[found]

    return json.loads(CHAT_IDS.sub(chat_id, gen_docs_demo.IDS.sub(run_id, text)))


def page_data(
    recorded: Recorded,
    chats: list[dict[str, Any]],
    ids: dict[str, str],
    summaries: list[dict[str, Any]],
    details: dict[str, Any],
) -> dict[str, Any]:
    """The page's data: the Runs view's runs and details, plus the chats to replay."""
    fixed_chats = _renumbered(chats, ids)
    for chat in fixed_chats:
        for turn in chat["turns"]:
            if turn["turn"]["run_id"] not in details:
                raise RuntimeError(f"turn {turn['turn']['id']} has no recorded run")
    info = dict(recorded.info, mode="demo")
    return {
        "runs": summaries,
        "details": details,
        "exported_at": gen_docs_demo.BASE_TS,
        "chat": {"info": info, "live": LIVE_COMMAND, "chats": fixed_chats},
    }
