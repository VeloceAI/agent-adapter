"""`agent-adapter voice check|serve FILE`: a VoxCat call from one config file."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from pathlib import Path
from typing import Any, TextIO

from agent_adapter.core.errors import MissingDependency

EPILOG = """examples:
  agent-adapter voice check voxcat.yaml
  agent-adapter voice serve voxcat.yaml            # the microphone and speakers

Keys come from the environment (DEEPGRAM_API_KEY, OPENAI_API_KEY, ...), never the file."""


def add_voice_parser(commands: Any) -> None:
    """Register the `voice` subcommand on an argparse subparsers object."""
    cmd = commands.add_parser(
        "voice",
        help="check or run a VoxCat voice call from one config file",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = cmd.add_subparsers(dest="voice_command", required=True, metavar="ACTION")
    check = actions.add_parser("check", help="validate FILE and print the call it sets up")
    check.add_argument("file", type=Path, help="a .yaml, .yml or .json voice config")
    serve = actions.add_parser("serve", help="run the call on this computer's mic and speakers")
    serve.add_argument("file", type=Path, help="a .yaml, .yml or .json voice config")


def run_voice(args: argparse.Namespace, out: TextIO) -> int:
    """Check or serve `args.file`; return the exit code (2 for a bad config)."""
    from agent_adapter.voice.config import VoiceConfig

    try:
        config = VoiceConfig.load(args.file)
        out.write("".join(f"{line}\n" for line in config.describe()))
        if args.voice_command == "check":
            return 0
        from agent_adapter.voice.local import LocalAudio

        session, agent = config.build(LocalAudio())
    except MissingDependency as exc:
        sys.stderr.write(f"agent-adapter voice: {exc}\n")
        return 2
    out.write("listening; press Ctrl+C to end the call\n")
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(session.run(agent))
    return 0
