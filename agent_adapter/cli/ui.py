"""`agent-adapter ui`: serve the run viewer (and chat) on 127.0.0.1, or export one HTML file."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, TextIO

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

EPILOG = """examples:
  agent-adapter ui                                         # browse recorded runs
  agent-adapter ui --model app:model --tools app:registry  # chat via the built-in loop
  agent-adapter ui --agent app:chat                        # chat with any agent

Targets are module.path:attribute, imported from the current folder. With chat on, this
process runs the agent and its tools; anyone with the printed link can chat and decide
approvals as --reviewer."""

# Chat-only options and their defaults: set without --model/--agent they are refused.
_CHAT_DEFAULTS: dict[str, Any] = {
    "tools": None,
    "executor": None,
    "system": "",
    "max_steps": 8,
    "turn_timeout": 300.0,
    "approval_timeout": 600.0,
    "chat_dir": None,
}


def add_ui_parser(commands: Any) -> None:
    """Register the `ui` subcommand on an argparse subparsers object."""
    from agent_adapter.viewer import DEFAULT_FOLDER, LOOPBACK
    from agent_adapter.viewer.chat.setup import DEFAULT_CHAT_DIR
    from agent_adapter.viewer.server import DEFAULT_PORT

    ui = commands.add_parser(
        "ui",
        # the agent guide quotes this line
        help="browse recorded runs, or chat with an agent, in a local web page",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ui.add_argument(
        "target", nargs="?", default=DEFAULT_FOLDER, help=f"runs folder or file ({DEFAULT_FOLDER})"
    )
    ui.add_argument("--port", type=_port, default=DEFAULT_PORT, help=f"port ({DEFAULT_PORT})")
    ui.add_argument("--host", default=LOOPBACK, help=f"bind address; only {LOOPBACK} is allowed")
    ui.add_argument("--approvals", metavar="PATH.db", help="a SQLite approval store to decide")
    ui.add_argument(
        "--reviewer",
        default="",
        metavar="NAME",
        help="name stored as decided_by (needed with --approvals; chat default: local)",
    )
    ui.add_argument("--export", metavar="report.html", help="write one offline HTML file and exit")
    ui.add_argument("--run", metavar="KEY", help="with --export: only this run")
    chat = ui.add_argument_group("chat (off unless --model or --agent is given)")
    chat.add_argument("--model", metavar="MOD:ATTR", help="an LLMPort: chat via the built-in loop")
    chat.add_argument(
        "--tools",
        metavar="MOD:ATTR",
        help="a Registry: the built-in loop's tools, or session.executor's with --agent",
    )
    chat.add_argument("--agent", metavar="MOD:ATTR", help="any chat agent: fn(message, session)")
    chat.add_argument("--executor", metavar="MOD:ATTR", help="an Executor to run tools with")
    chat.add_argument("--system", default="", metavar="TEXT", help="with --model: system prompt")
    chat.add_argument(
        "--max-steps", type=_positive_int, default=8, help="with --model: model calls per turn (8)"
    )
    chat.add_argument(
        "--turn-timeout", type=_seconds, default=300.0, help="seconds before a turn ends (300)"
    )
    chat.add_argument(
        "--approval-timeout",
        type=_seconds,
        default=600.0,
        help="seconds an approval waits before it is denied (600)",
    )
    chat.add_argument(
        "--chat-dir",
        metavar="DIR",
        help=f"sessions, uploads and eval_cases.jsonl ({DEFAULT_CHAT_DIR})",
    )


def _port(text: str) -> int:
    """An argparse type: a TCP port, 0 (any free port) to 65535."""
    try:
        port = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}") from None
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 0-65535, got {port}")
    return port


def _positive_int(text: str) -> int:
    """An argparse type: a whole number >= 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _seconds(text: str) -> float:
    """An argparse type: a number of seconds > 0."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number of seconds: {text!r}") from None
    if not value > 0 or value == float("inf"):
        raise argparse.ArgumentTypeError(f"must be > 0 seconds, got {text}")
    return value


def run_ui(args: argparse.Namespace, out: TextIO) -> int:
    """Export, or serve until interrupted.

    Raises:
        ConfigError: a bad host, target, approvals database, reviewer or chat option.
        OSError: the port is taken or a file cannot be read or written.
    """
    from agent_adapter.viewer import export, make_server

    chat_on = bool(args.model or args.agent)
    if not chat_on:
        _refuse_chat_options(args)
    if args.export:
        if args.approvals:
            raise ConfigError("--export writes a read-only file: drop --approvals")
        if chat_on:
            raise ConfigError("--export writes a read-only file: drop --model/--agent")
        path = export(args.target, args.export, run=args.run)
        out.write(f"wrote {path}\n")
        return 0
    if args.run:
        raise ConfigError("--run needs --export")
    if args.approvals and not args.reviewer:
        raise ConfigError("--approvals needs --reviewer NAME (stored as decided_by)")
    if chat_on:
        server = _chat_server(args)
    else:
        server = make_server(
            args.target,
            host=args.host,
            port=args.port,
            approvals=args.approvals,
            reviewer=args.reviewer,
        )
    from agent_adapter.viewer.serving import run_server

    run_server(server, out)
    return 0


def _refuse_chat_options(args: argparse.Namespace) -> None:
    for name, default in _CHAT_DEFAULTS.items():
        if getattr(args, name) != default:
            flag = "--" + name.replace("_", "-")
            raise ConfigError(f"{flag} is a chat option: add --model app:model or --agent app:chat")


def _import_from_here() -> None:
    """Put the current folder on sys.path, so targets import from it."""
    here = os.getcwd()
    if here not in sys.path:
        sys.path.append(here)  # last: never shadows the standard library or installed code
        logger.debug("ui: added %s to sys.path for chat targets", here)


def _chat_server(args: argparse.Namespace) -> Any:
    """The viewer with chat on, from the `module:attribute` targets on the command line."""
    from agent_adapter.viewer.chat.setup import DEFAULT_CHAT_DIR
    from agent_adapter.viewer.chat.targets import load_target
    from agent_adapter.viewer.serving import make_chat_server

    if args.model and args.agent:
        raise ConfigError("use --model (built-in loop) or --agent, not both")
    _import_from_here()
    if args.agent:
        for flag, value, default in (
            ("--system", args.system, ""),
            ("--max-steps", args.max_steps, 8),
        ):
            if value != default:
                raise ConfigError(f"{flag} goes with --model; an --agent runs its own loop")
    return make_chat_server(
        agent=load_target(args.agent, "--agent") if args.agent else None,
        model=load_target(args.model, "--model") if args.model else None,
        registry=load_target(args.tools, "--tools") if args.tools else None,
        executor=load_target(args.executor, "--executor") if args.executor else None,
        folder=args.target,
        host=args.host,
        port=args.port,
        system=args.system,
        max_steps=args.max_steps,
        turn_timeout=args.turn_timeout,
        approval_timeout=args.approval_timeout,
        reviewer=args.reviewer or "local",
        chat_dir=args.chat_dir or DEFAULT_CHAT_DIR,
        approvals=args.approvals,
        approvals_reviewer=args.reviewer,
        agent_name=args.agent or args.model,
    )
