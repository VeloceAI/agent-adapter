"""`agent-adapter audit verify|export DIR`: check a signed audit log, or export a window of it."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, TextIO

from agent_adapter.core.errors import ConfigError

logger = logging.getLogger(__name__)

EPILOG = """The signing key is read from the environment (AA_AUDIT_KEY, or --key-env NAME), never
from the command line, where other users could see it in the process list.

examples:
  agent-adapter audit verify .agent_adapter/audit
  agent-adapter audit export .agent_adapter/audit --since 2026-11-01 --format csv --out audit.csv

verify exits 1 when anything was edited, removed, reordered or cut; export refuses a log
that does not verify."""


def add_audit_parser(commands: Any) -> None:
    """Register `audit verify DIR` and `audit export DIR` on an argparse subparsers object."""
    cmd = commands.add_parser(
        "audit",
        help="verify a signed audit log, or export it as JSONL or CSV",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = cmd.add_subparsers(dest="audit_command", required=True, metavar="ACTION")
    check = actions.add_parser("verify", help="check every record, the chain and the seals")
    check.add_argument(
        "--anchor", metavar="LOG:SEQ:HASH", help="an anchor printed by an earlier verify"
    )
    out = actions.add_parser("export", help="write the verified records as JSONL or CSV")
    for sub in (check, out):
        sub.add_argument("folder", type=Path, metavar="DIR", help="the audit folder")
        sub.add_argument("--key-env", default="AA_AUDIT_KEY", metavar="NAME", help="key variable")
    out.add_argument("--since", metavar="TIME", help="ISO date or time, inclusive (UTC)")
    out.add_argument("--until", metavar="TIME", help="ISO date or time, exclusive (UTC)")
    out.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    out.add_argument("--out", type=Path, metavar="PATH", help="write here instead of stdout")
    out.add_argument("--force", action="store_true", help="replace PATH if it exists")


def run_audit(args: argparse.Namespace, out: TextIO) -> int:
    """Run `audit verify` (exit 1 when not intact) or `audit export` (exit 0).

    Raises:
        ConfigError: no key in the environment, a bad folder or time, a log that does not
            verify (export), or an `--out` that exists without `--force`.
    """
    from agent_adapter.audit import export_to, verify

    key = _key(args.key_env)
    if args.audit_command == "verify":
        result = verify(args.folder, key, anchor=args.anchor)
        out.write(result.summary() + "\n")
        out.write("".join(f"  {problem}\n" for problem in result.problems))
        if result.ok:  # keep this line outside the host: a later --anchor catches a rollback
            out.write(f"anchor {result.anchor}\n")
        return 0 if result.ok else 1
    window = {"since": args.since, "until": args.until}
    if args.out is None:
        export_to(out, args.folder, **window, format=args.format, key=key)
        return 0
    reserved = _reserve(args.out, args.force)
    partial = args.out.with_name(args.out.name + ".part")
    done = False
    try:
        with partial.open("w", encoding="utf-8", newline="") as target:
            count = export_to(target, args.folder, **window, format=args.format, key=key)
            target.flush()
            os.fsync(target.fileno())  # durable before it takes the target's name
        os.replace(partial, args.out)
        done = True
    finally:
        partial.unlink(missing_ok=True)
        if reserved and not done:
            args.out.unlink(missing_ok=True)
    out.write(f"wrote {count} record(s) to {args.out}\n")
    return 0


def _reserve(path: Path, force: bool) -> bool:
    """Claim `path` atomically unless `force`; True when this call created it.

    Raises:
        ConfigError: `path` exists and `force` is off.
    """
    if force:
        return False
    try:
        path.open("x", encoding="utf-8").close()
    except FileExistsError:
        raise ConfigError(f"{path} exists; pass --force to replace it") from None
    return True


def _key(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise ConfigError(f"set {name} to the audit signing key")
    return value
