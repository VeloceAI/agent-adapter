"""`serve(...)`: the viewer with chat, from Python; blocks until Ctrl+C."""

from __future__ import annotations

import logging
import os
import sys
import webbrowser
from pathlib import Path
from typing import TextIO

from agent_adapter.core.errors import ConfigError
from agent_adapter.core.ports import LLMPort
from agent_adapter.core.registry import Registry
from agent_adapter.exec.executor import Executor
from agent_adapter.viewer._files import DEFAULT_FOLDER
from agent_adapter.viewer.chat.contract import ChatAgent
from agent_adapter.viewer.chat.setup import DEFAULT_CHAT_DIR, build_host
from agent_adapter.viewer.server import (
    DEFAULT_PORT,
    LOOPBACK,
    ViewerServer,
    check_host,
    make_server,
)

logger = logging.getLogger(__name__)


def make_chat_server(
    *,
    agent: ChatAgent | None = None,
    model: LLMPort | None = None,
    registry: Registry | None = None,
    executor: Executor | None = None,
    folder: str | os.PathLike[str] = DEFAULT_FOLDER,
    host: str = LOOPBACK,
    port: int = DEFAULT_PORT,
    system: str = "",
    max_steps: int = 8,
    turn_timeout: float = 300.0,
    approval_timeout: float = 600.0,
    reviewer: str = "local",
    chat_dir: str | os.PathLike[str] = DEFAULT_CHAT_DIR,
    approvals: str | os.PathLike[str] | None = None,
    approvals_reviewer: str = "",
    agent_name: str = "",
    stream: bool = True,
) -> ViewerServer:
    """A bound viewer server with chat on; logs the start-up WARNING chat mode needs.

    Raises:
        ConfigError: a bad option (see `build_host` and `make_server`).
        OSError: the port is taken.
    """
    check_host(host)
    chat = build_host(
        runs=folder,
        agent=agent,
        model=model,
        registry=registry,
        executor=executor,
        system=system,
        max_steps=max_steps,
        turn_timeout=turn_timeout,
        approval_timeout=approval_timeout,
        reviewer=reviewer,
        chat_dir=chat_dir,
        agent_name=agent_name,
        stream=stream,
    )
    try:
        server = make_server(
            Path(folder),
            host=host,
            port=port,
            approvals=Path(approvals) if approvals is not None else None,
            reviewer=approvals_reviewer,
            chat=chat,
        )
    except BaseException:
        chat.close()
        raise
    logger.warning(
        "chat is on: this server runs %s and its tools in this process; anyone with the "
        "printed link can chat and decide approvals as %r",
        chat.options.agent_name,
        chat.options.reviewer,
    )
    return server


def run_server(server: ViewerServer, out: TextIO, *, open_browser: bool = False) -> None:
    """Print the link, optionally open it, and serve until Ctrl+C; then close the server."""
    out.write(f"agent-adapter ui: open {server.url}\n")
    out.write("  the address holds an access token: keep it private (Ctrl+C to stop)\n")
    if server.chat is not None:
        out.write(
            f"  chat is on ({server.chat.options.agent_name}): it runs your agent and tools here; "
            f"anyone with the link decides approvals as {server.chat.options.reviewer!r}\n"
        )
    out.flush()
    if open_browser:
        open_in_browser(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        out.write("stopped\n")
    finally:
        server.server_close()


def open_in_browser(server: ViewerServer) -> None:
    """Open a one-use launch link: the token itself never reaches a command line or a file."""
    if not webbrowser.open(f"{server.base_url}launch/{server.launch.new()}"):
        logger.info("viewer: no browser opened; open the printed link")


def serve(
    agent: ChatAgent | None = None,
    *,
    model: LLMPort | None = None,
    registry: Registry | None = None,
    executor: Executor | None = None,
    folder: str | os.PathLike[str] = DEFAULT_FOLDER,
    port: int = DEFAULT_PORT,
    open_browser: bool = True,
    system: str = "",
    max_steps: int = 8,
    turn_timeout: float = 300.0,
    approval_timeout: float = 600.0,
    reviewer: str = "local",
    chat_dir: str | os.PathLike[str] = DEFAULT_CHAT_DIR,
    stream: bool = True,
    out: TextIO | None = None,
) -> None:
    """Chat with `agent` (or the built-in loop over `model` and `registry`) in the viewer.

    Blocks until Ctrl+C. Turns are recorded into `folder`, so the Runs view shows them.
    `open_browser` opens a one-use launch link (valid 60 s); `stream=False`
    makes the built-in loop call `generate` (so an LLM response cache applies).

    Raises:
        ConfigError: neither or both of `agent` and `model`, or a bad option.
        OSError: the port is taken.
    """
    if agent is None and model is None:
        raise ConfigError("serve needs agent= or model= (with registry=)")
    server = make_chat_server(
        agent=agent,
        model=model,
        registry=registry,
        executor=executor,
        folder=folder,
        port=port,
        system=system,
        max_steps=max_steps,
        turn_timeout=turn_timeout,
        approval_timeout=approval_timeout,
        reviewer=reviewer,
        chat_dir=chat_dir,
        stream=stream,
    )
    run_server(server, out or sys.stdout, open_browser=open_browser)
