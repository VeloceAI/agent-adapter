"""Wire a chat host: the agent, its executor (approvals, watch, recorder) and the stores."""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import cast

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.exec.executor import Executor
from agent_adapter.exec.guards import RunLimits
from agent_adapter.memory import GuardedMemory
from agent_adapter.memory.records import MAX_ID_CHARS
from agent_adapter.tracing.sinks import members
from agent_adapter.viewer.chat.approvals import ChatApprovals
from agent_adapter.viewer.chat.builtin import BuiltinAgent
from agent_adapter.viewer.chat.contract import ChatAgent
from agent_adapter.viewer.chat.host import ChatHost
from agent_adapter.viewer.chat.memory import ChatMemory, SavePolicy
from agent_adapter.viewer.chat.records import ChatOptions
from agent_adapter.viewer.chat.sessions import SessionStore
from agent_adapter.viewer.chat.watch import TurnWatch
from agent_adapter.viewer.recorder import Recorder, record

logger = logging.getLogger(__name__)

DEFAULT_CHAT_DIR = ".agent_adapter/chats"
DEFAULT_MAX_CALLS = 50  # tool calls per turn when the host makes the executor
MAX_REVIEWER = 120


def build_host(
    *,
    runs: str | os.PathLike[str],
    agent: ChatAgent | None = None,
    model: LLMPort | None = None,
    registry: Registry | None = None,
    executor: Executor | None = None,
    system: str = "",
    max_steps: int = 8,
    turn_timeout: float = 300.0,
    approval_timeout: float = 600.0,
    reviewer: str = "local",
    chat_dir: str | os.PathLike[str] = DEFAULT_CHAT_DIR,
    agent_name: str = "",
    stream: bool = True,
    user_id: str = "",
    memory: GuardedMemory | None = None,
    memory_save: str = "none",
    memory_every: int = 1,
    recall_k: int = 5,
) -> ChatHost:
    """A chat host for `agent`, or for the built-in loop over `model` and `registry`.

    Turns are recorded into the runs folder `runs`, so the Runs view shows them too. With
    `registry` and `executor` both given, the executor runs that registry's tools. The
    built-in loop streams (`stream=True`), which skips an LLM response cache. `user_id`
    owns the chats (`ctx.user_id` in tools); with `memory`, each turn recalls that user's
    records and, unless `memory_save="none"`, saves a summary or facts every
    `memory_every` turns (through the model, or the agent's `save_memories`).

    Raises:
        ConfigError: both or neither of `agent` and `model`, a bad option, a runs target that
            is a file, or a folder that cannot be made.
    """
    if (agent is None) == (model is None):
        raise ConfigError("chat needs either an agent or a model (with tools), not both")
    if agent is not None and not callable(agent):
        raise ConfigError(f"the chat agent must be callable, got {type(agent).__name__}")
    if model is not None and not isinstance(model, LLMPort):
        raise ConfigError(f"the model must be an LLMPort, got {type(model).__name__}")
    if executor is not None and not isinstance(executor, Executor):
        raise ConfigError(f"executor must be an Executor, got {type(executor).__name__}")
    if registry is not None and not isinstance(registry, Registry):
        raise ConfigError(f"tools must be a Registry, got {type(registry).__name__}")
    _check_options(max_steps, turn_timeout, approval_timeout, reviewer)
    user = _check_user(user_id)
    chat_memory = _memory(memory, user, memory_save, memory_every, recall_k)
    saves = chat_memory is not None and chat_memory.save != "none"
    if saves and agent is not None and not callable(getattr(agent, "save_memories", None)):
        raise ConfigError(
            f"memory_save={memory_save!r} needs an agent with save_memories(transcript, "
            "session, policy), or use the built-in loop (model=)"
        )
    folder = Path(runs)
    if folder.is_file():
        raise ConfigError(f"chat records turns into a runs folder; {folder} is a file")
    store = SessionStore(chat_dir, user_id=user)
    watch = TurnWatch()
    desk = ChatApprovals(
        watch, reviewer=reviewer.strip(), timeout=approval_timeout, history=store.approval_views
    )
    host_executor = _executor(executor, registry, watch, desk)
    builtin = None
    if model is not None:
        builtin = BuiltinAgent(
            model,
            registry or host_executor.registry,
            system=system,
            max_steps=max_steps,
            stream=stream,
        )
    chat_agent = builtin if builtin is not None else cast(ChatAgent, agent)
    options = ChatOptions(
        mode="builtin" if builtin is not None else "agent",
        agent_name=agent_name or (builtin.name if builtin is not None else _name(chat_agent)),
        turn_timeout=float(turn_timeout),
        reviewer=reviewer.strip(),
    )
    host = ChatHost(
        chat_agent,
        host_executor,
        options=options,
        store=store,
        watch=watch,
        desk=desk,
        refusal=builtin.refusal if builtin is not None else None,
        memory=chat_memory,
    )
    recorder = _recorder(host_executor, folder)
    if recorder is not None:
        host.on_close.append(recorder.stop)
    logger.info("chat: %s mode (%s)", options.mode, options.agent_name)
    return host


def _executor(
    given: Executor | None, registry: Registry | None, watch: TurnWatch, desk: ChatApprovals
) -> Executor:
    """The host's executor: the given one (its own approval port kept) plus the watch."""
    if given is None:
        return Executor(
            registry if registry is not None else Registry(),
            approval=desk.port(),
            hooks=(watch,),
            sink=watch,
            limits=RunLimits(max_calls=DEFAULT_MAX_CALLS),
        )
    changes: dict[str, object] = {"hooks": (*given.hooks, watch)}
    if registry is not None:
        changes["registry"] = registry
    if given.approval is None:
        changes["approval"] = desk.port()
    else:
        logger.info("chat: the executor keeps its own approval port")
    made = given.replace(**changes)
    made.add_sink(watch)
    return made


def _recorder(executor: Executor, folder: Path) -> Recorder | None:
    """A new chat recorder, or None when one already writes `folder` for this executor."""
    resolved = folder.resolve()
    for sink in members(executor.sink):
        if isinstance(sink, Recorder) and sink.folder == resolved:
            logger.info("chat: turns go to the recorder the executor already has")
            return None
    return record(executor, folder=folder, source="chat")


def _check_user(user_id: str) -> str:
    """The stripped user id: printable, at most `MAX_ID_CHARS` characters ("" = no user)."""
    user = user_id.strip() if isinstance(user_id, str) else None
    if user is None or len(user) > MAX_ID_CHARS or not user.isprintable():
        raise ConfigError(f"user_id must be printable text of at most {MAX_ID_CHARS} characters")
    return user


def _memory(
    memory: GuardedMemory | None, user: str, save: str, every: int, k: int
) -> ChatMemory | None:
    if memory is None:
        if save != "none":
            raise ConfigError(f"memory_save={save!r} needs memory=")
        return None
    if not user:
        raise ConfigError("memory= needs user_id=: memory is kept per user")
    return ChatMemory(memory, user, cast(SavePolicy, save), every, k)


def _check_options(max_steps: int, turn: float, approval: float, reviewer: str) -> None:
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1:
        raise ConfigError(f"max_steps must be an int >= 1, got {max_steps!r}")
    for label, value in (("turn_timeout", turn), ("approval_timeout", approval)):
        number = isinstance(value, int | float) and not isinstance(value, bool)
        if not number or not math.isfinite(value) or not value > 0:
            raise ConfigError(f"{label} must be a number of seconds > 0, got {value!r}")
    name = str(reviewer or "").strip()
    if not name or len(name) > MAX_REVIEWER:
        raise ConfigError(f"reviewer must name the reviewer (1-{MAX_REVIEWER} characters)")


def _name(agent: object) -> str:
    qualname = getattr(agent, "__qualname__", None) or type(agent).__qualname__
    return f"{getattr(agent, '__module__', '') or type(agent).__module__}.{qualname}"
