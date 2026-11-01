"""Chat sessions, turns and stored files as plain records that round-trip through JSON."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

TurnStatus = Literal["running", "done", "failed", "stopped", "timed-out"]
ENDED = frozenset({"done", "failed", "stopped", "timed-out"})


@dataclass(frozen=True)
class ChatOptions:
    """How turns run: their time limits, the reviewer name and the mode shown in the page."""

    mode: str  # "builtin" or "agent"
    agent_name: str
    turn_timeout: float = 300.0
    reviewer: str = "local"
    heartbeat_s: float = 5.0


class Refused(Exception):
    """A chat request the server refuses: `status` is the HTTP status, the text says why."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class FileRef:
    """An uploaded file: `path` is relative to the chat folder; `name` is display text only."""

    id: str
    name: str
    media_type: str
    size: int
    path: str
    owned: bool = True  # False: inherited from the session a branch came from

    def view(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "media_type": self.media_type, "size": self.size}


@dataclass
class Turn:
    """One user message and the agent's answer, with what the run did."""

    id: str
    session_id: str
    run_id: str
    text: str
    attachments: list[str] = field(default_factory=list)  # file ids
    state_before: dict[str, Any] = field(default_factory=dict)
    status: TurnStatus = "running"
    answer: str = ""
    error: dict[str, str] | None = None
    started: float = field(default_factory=time.time)
    ended: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    tools: list[dict[str, Any]] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    replay_of: str = ""
    edit_of: str = ""

    def view(self) -> dict[str, Any]:
        """What the page shows of a turn (`state_before` stays on the server)."""
        data = asdict(self)
        data.pop("state_before")
        return data


@dataclass
class Session:
    """A conversation: its turns, live state and files; `parent` names the branch origin."""

    id: str
    title: str
    state: dict[str, Any] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    files: dict[str, FileRef] = field(default_factory=dict)
    parent: dict[str, str] | None = None  # {"session_id", "turn_id"}
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    uploaded_bytes: int = 0
    running: str = ""  # the id of the running turn; never saved
    # JSON copy of `state` taken between turns: what is saved and shown. Only the loop
    # thread touches `state` while a turn runs, so other threads never read it then.
    saved_state: dict[str, Any] = field(default_factory=dict)

    def turn(self, turn_id: str) -> Turn | None:
        return next((turn for turn in self.turns if turn.id == turn_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "state": copy.deepcopy(self.saved_state),
            "turns": [asdict(turn) for turn in self.turns],
            "files": {key: asdict(ref) for key, ref in self.files.items()},
            "parent": copy.deepcopy(self.parent),
            "created": self.created,
            "updated": self.updated,
            "uploaded_bytes": self.uploaded_bytes,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "turns": len(self.turns),
            "running": self.running or None,
            "parent": copy.deepcopy(self.parent),
        }


def session_from_dict(data: dict[str, Any]) -> tuple[Session, bool]:
    """A session read back from its JSON file, and whether a turn was cut off by a restart.

    Raises:
        KeyError, TypeError, ValueError: the data is not a saved session.
    """
    turns = [Turn(**turn) for turn in data.get("turns", [])]
    files = {str(key): FileRef(**ref) for key, ref in dict(data.get("files", {})).items()}
    state = dict(data.get("state", {}))
    session = Session(
        id=str(data["id"]),
        title=str(data.get("title", "")),
        state=copy.deepcopy(state),
        saved_state=state,
        turns=turns,
        files=files,
        parent=dict(data["parent"]) if data.get("parent") else None,
        created=float(data.get("created", 0.0)),
        updated=float(data.get("updated", 0.0)),
        uploaded_bytes=int(data.get("uploaded_bytes", 0)),
    )
    interrupted = False
    for turn in turns:
        if turn.status == "running":
            interrupted = True
            turn.status = "stopped"
            turn.error = {"kind": "interrupted", "message": "the server stopped during this turn"}
            turn.ended = turn.ended or turn.started
        elif turn.status not in ENDED:
            raise ValueError(f"turn {turn.id} has status {turn.status!r}")
    return session, interrupted
