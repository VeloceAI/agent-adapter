"""`agent-adapter policy check FILE`: validate a policy file and print the rules it sets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from agent_adapter.core.errors import MissingDependency

if TYPE_CHECKING:
    from agent_adapter.policy import Policy

logger = logging.getLogger(__name__)

EPILOG = """examples:
  agent-adapter policy check policy.yaml
  agent-adapter policy check policy.toml --env prod --tool refund --tool send_email

Exit 0 when the file is valid, 2 when it is not (the message names the line and key)."""


def add_policy_parser(commands: Any) -> None:
    """Register the `policy` subcommand on an argparse subparsers object."""
    cmd = commands.add_parser(
        "policy",
        help="validate a policy file and print the rules it sets",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = cmd.add_subparsers(dest="policy_command", required=True, metavar="ACTION")
    check = actions.add_parser("check", help="validate FILE and print its effective rules")
    check.add_argument("file", type=Path, help="a .yaml, .yml, .toml or .json policy")
    check.add_argument("--env", help="the environment (default: AA_ENV)")
    check.add_argument(
        "--tool", action="append", default=[], metavar="NAME", help="also explain this tool"
    )
    check.add_argument(
        "--allow-relax", action="store_true", help="accept the relaxations under `relax`"
    )


def run_policy(args: argparse.Namespace, out: TextIO) -> int:
    """Validate `args.file`; print its rules (and each `--tool`) and return the exit code.

    Raises:
        PolicyError: the file is invalid (the CLI prints it and exits 2).
    """
    from agent_adapter.policy import load_policy

    try:
        policy = load_policy(args.file, env=args.env, allow_relax=args.allow_relax)
    except MissingDependency as exc:
        sys.stderr.write(f"agent-adapter policy: {exc}\n")
        return 2
    out.write("".join(f"{line}\n" for line in summary(policy)))
    for name in args.tool:
        out.write(f"tool {name}:\n")
        out.write("".join(f"  {line}\n" for line in policy.explain(name).lines))
    return 0


def summary(policy: Policy) -> list[str]:
    """The effective rules of `policy` in its environment, one line each."""
    spec = policy.spec
    lines = [
        f"{policy.source}: valid (environment {policy.env or '(none)'}, default {policy.default})"
    ]
    for where, rule in policy.rules:
        parts = [f"tools {', '.join(rule.tools)}"]
        if rule.allow is not None:
            parts.append("allow" if rule.allow else "deny")
        if rule.confirm is not None:
            parts.append(
                "confirm always" if rule.confirm is True else f"confirm when {rule.confirm.shown()}"
            )
        scope = [
            f"{k} {', '.join(getattr(rule, k))}"
            for k in ("kinds", "tags", "teams")
            if getattr(rule, k)
        ]
        if rule.users:
            scope.append(f"{len(rule.users)} listed user(s)")
        lines.append(f"  {where}: {'; '.join(parts + scope)}")
    for where, rule, hook in policy.guards:
        stages = ", ".join(sorted(hook.guard.stages))
        lines.append(f"  {where}: guard {rule.guard} ({hook.guard.action} on {stages})")
    for where, rule, _ in policy.budgets:
        window = f" per {rule.per}" if rule.per else ""
        lines.append(
            f"  {where}: budget {rule.name} {rule.max_usd:g} USD per {rule.scope}{window}, "
            f"{rule.on_exceed}, store {_store(rule.store)}"
        )
    if policy.rates is not None:
        for kind, rate in policy.rates.rates.items():
            lines.append(f"  rate limit {kind}: {rate.calls} per {rate.per:g} s")
    if spec.limits and spec.limits.max_calls:
        lines.append(f"  limits: {spec.limits.max_calls} calls per run")
    if spec.memory:
        days = spec.memory.retention_days
        kept = f"{days:g} days" if days else "unchanged"
        lines.append(f"  memory: save {spec.memory.save}, retention {kept}")
    if spec.sandbox:
        sandbox = spec.sandbox
        lines.append(
            f"  sandbox: isolated {'required' if sandbox.require_isolated else 'optional'}, "
            f"network {'denied' if sandbox.deny_network else 'as configured'}"
        )
    if spec.relax:
        lines.append(f"  relaxations: {', '.join(spec.relax)}")
    return lines


def _store(name: str) -> str:
    return "sqlite" if name.startswith("sqlite:") else name  # paths stay off the screen
