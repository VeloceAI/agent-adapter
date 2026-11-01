"""CLI: `python -m agent_adapter.optimiser --host mod:fn --model provider:model ...`."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Sequence
from typing import Any, NoReturn, cast

from agent_adapter.core.errors import AdapterError, ConfigError, OptimiserError, log_failure
from agent_adapter.core.ports import LLMPort
from agent_adapter.optimiser._budget import check_config
from agent_adapter.optimiser.callable_host import NullModel
from agent_adapter.optimiser.hosts import Host
from agent_adapter.optimiser.loop import optimise
from agent_adapter.optimiser.model import OptimiserBudget, OptimiserConfig
from agent_adapter.optimiser.surface import cap_key, validate_tree
from agent_adapter.runtime.sync import run_sync

logger = logging.getLogger(__name__)

EXIT_PATCH, EXIT_ERROR, EXIT_KEEP_BASE, EXIT_STOPPED = 0, 1, 2, 3


class _UsageError(ConfigError):
    """Bad command-line arguments (exit 1, so exit 2 stays KEEP_BASE)."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise _UsageError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="python -m agent_adapter.optimiser",
        description="Propose, probe and compose harness edits; writes a patch, applies nothing.",
    )
    parser.add_argument("--host", required=True, help="module:callable returning a Host")
    parser.add_argument(
        "--model",
        help="the agent model, provider:model (openai, gemini, anthropic), or none when "
        "the host's agent brings its own model",
    )
    parser.add_argument("--proposer-model", help="the proposer model (default: --model)")
    parser.add_argument("--k", type=int, default=1, help="candidates per dimension (default 1)")
    parser.add_argument("--mode", choices=("parallel", "sequential"), default="parallel")
    parser.add_argument("--compose", choices=("greedy", "llm"), default="greedy")
    parser.add_argument("--out", default="runs/opt", help="output directory")
    parser.add_argument("--budget-usd", type=float, help="stop once this much is spent")
    parser.add_argument("--max-case-runs", type=int, help="stop before exceeding this many runs")
    parser.add_argument("--lessons", default=".agent_adapter/lessons.json", help="lessons file")
    parser.add_argument("--dry-run", action="store_true", help="show the plan; call no model")
    return parser


def load_host(target: str) -> Host:
    """Import `module:callable` and call it.

    Raises:
        ConfigError: the target is malformed, missing, or does not return a Host.
    """
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ConfigError(f"--host must be module:callable, got {target!r}")
    try:
        found: Any = importlib.import_module(module_name)
        for part in attr.split("."):
            found = getattr(found, part)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(f"cannot load host {target!r}: {exc}") from exc
    host = found() if callable(found) else found
    if not isinstance(host, Host):
        raise ConfigError(f"{target} returned {type(host).__name__}, not a Host")
    return host


def model_from_spec(spec: str) -> LLMPort:
    """`openai:gpt-5` -> OpenAIModel; `gemini:`, `anthropic:` likewise; `none` -> NullModel.

    Raises:
        ConfigError: not `provider:model`, or an unknown provider.
        MissingDependency: the provider's SDK is not installed.
    """
    if spec == "none":
        return cast("LLMPort", NullModel())
    provider, _, name = spec.partition(":")
    if not name:
        raise ConfigError(f"model must be provider:model, got {spec!r}")
    if provider == "openai":
        from agent_adapter.models.openai import OpenAIModel

        return cast("LLMPort", OpenAIModel(name))
    if provider == "gemini":
        from agent_adapter.models.gemini import GeminiModel

        return cast("LLMPort", GeminiModel(name))
    if provider == "anthropic":
        from agent_adapter.models.anthropic import AnthropicModel

        return cast("LLMPort", AnthropicModel(name))
    raise ConfigError(f"unknown model provider {provider!r}; use openai, gemini or anthropic")


def _pattern(path: str) -> str:
    """The caps pattern of a path, or "" for a path outside the grammar."""
    try:
        return cap_key(path)
    except OptimiserError:
        return ""


def dry_run_text(host: Host, config: OptimiserConfig) -> str:
    """The surface, the cases and the budget identity; no model is called."""
    tree, caps, cases = host.surface(), host.caps(), list(host.cases())
    k, n = config.k, len(cases)
    problems = validate_tree(tree, caps, host)
    lines = [f"Surface ({len(tree.files)} paths, domain {host.domain}):"]
    for path, text in sorted(tree.files.items()):
        cap = caps.max_chars.get(_pattern(path))
        lines.append(f"  {path}  {len(text)} chars" + (f" (cap {cap})" if cap else ""))
    lines += [f"  PROBLEM: {p}" for p in problems]
    lines.append(f"Cases ({n}):")
    lines += [f"  {c.id}" + (f"  [{', '.join(c.tags)}]" if c.tags else "") for c in cases]
    merge = "0 (greedy compose)" if config.compose_mode == "greedy" else f"at most {k} (llm)"
    judge = "0 (the host has a judge)" if host.judge is not None else "at most 1 (no judge)"
    lines += [
        f"Budget identity (k={k}, mode {config.mode}, N={n}):",
        f"  case runs       (1 + 3k) x N = {(1 + 3 * k) * n}",
        f"  proposer calls  2k = {2 * k} (a retry is an attempt, not a call)",
        f"  merge calls     {merge}",
        "  reflection      at most 1"
        if config.lessons_path
        else "  reflection      0 (no lessons file)",
        f"  selection       {judge}",
        f"Limits: usd {config.budget.max_usd}, case runs {config.budget.max_case_runs}",
        "Dry run: no model was called.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI: 0 a patch, 1 an error, 2 KEEP_BASE, 3 a budget stop before any choice."""
    try:
        args = _parser().parse_args(argv)
        host = load_host(args.host)
        config = OptimiserConfig(
            k=args.k,
            mode=args.mode,
            compose_mode=args.compose,
            out_dir=args.out,
            lessons_path=args.lessons,
            budget=OptimiserBudget(max_usd=args.budget_usd, max_case_runs=args.max_case_runs),
        )
        check_config(config)
        if args.dry_run:
            sys.stdout.write(dry_run_text(host, config))
            return EXIT_PATCH
        if not args.model:
            raise _UsageError("--model is required unless --dry-run is given")
        if args.model == "none" and not args.proposer_model:
            raise _UsageError("--model none needs --proposer-model: the proposer is a real model")
        model = model_from_spec(args.model)
        proposer = model_from_spec(args.proposer_model) if args.proposer_model else model
        result = run_sync(optimise(host, config, proposer=proposer, model=model))
    except _UsageError as exc:
        sys.stderr.write(f"usage error: {exc}\n")
        return EXIT_ERROR
    except AdapterError as exc:
        log_failure(logger, logging.ERROR, exc, "optimiser failed")
        sys.stderr.write(f"error: {exc}\n")
        return EXIT_ERROR
    except Exception as exc:
        log_failure(logger, logging.ERROR, exc, "optimiser failed")
        sys.stderr.write(f"error: {type(exc).__name__}: {exc}\n")
        return EXIT_ERROR
    used = result.budget
    sys.stdout.write(
        f"chosen: {result.chosen or 'KEEP_BASE'}; case runs {used.case_runs}; "
        f"proposer calls {used.proposer_calls}; cost ${used.usd:.4f}"
        + (f"; stopped: {used.stopped}" if used.stopped else "")
        + f"\noutputs: {config.out_dir}\n"
    )
    if result.chosen:
        return EXIT_PATCH
    return EXIT_STOPPED if used.stopped else EXIT_KEEP_BASE


if __name__ == "__main__":
    raise SystemExit(main())
