"""The prose sections of the guide: what it is, import rules, safety, errors and the CLI."""

from __future__ import annotations

import posixpath
import re
from pathlib import Path

from agent_adapter._version import __version__
from agent_adapter.cli.main import command_lines

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
SITE = re.search(r"^site_url:\s*(\S+)\s*$", (ROOT / "mkdocs.yml").read_text(encoding="utf-8"), re.M)
SITE_URL = SITE.group(1) if SITE else "https://agent-adapter.veloceai.in/"
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
CATALOGUE = ("<!-- catalogue:start -->", "<!-- catalogue:end -->")
NL = "\n"


def title(*, full: bool) -> str:
    """The H1 and the generated-file note; says which tier this is."""
    tier = (
        "This is the full guide: the concise one plus the API index."
        if full
        else "This is the concise guide; `agent-adapter guide --full` adds every signature."
    )
    return NL.join(
        [
            "# Agent Adapter: guide for AI coding agents",
            "",
            f"Version {__version__}. Generated from the package's own code, so it matches",
            f"this install. Print it with `agent-adapter guide`. {tier}",
            "",
        ]
    )


def what_it_is() -> str:
    """Three lines on what the SDK is."""
    return NL.join(
        [
            "## What it is",
            "",
            "A Python SDK: write a tool once as a plain function, then run it from any model",
            "API, agent framework, MCP, A2A or HTTP. One executor adds validation, approvals,",
            "budgets, cache, retries and tracing. It works with no agent framework; it can be",
            "your agent loop.",
            "",
        ]
    )


def import_rules() -> str:
    """The import rules, copied from the top of `docs/llms.txt`."""
    llms = (DOCS / "llms.txt").read_text(encoding="utf-8")
    rule = next(line for line in llms.splitlines() if line.startswith("Import rules:"))
    return f"## Import rules{NL}{NL}{rule}{NL}"


def safety() -> str:
    """Dos and don'ts for agents writing code with the SDK."""
    return NL.join(
        [
            "## Safety rules for agents writing code with it",
            "",
            "Do:",
            "",
            "- Mark every tool with a side effect (write, pay, send, delete) `confirm=True`,",
            "  imported ones too: `import_openapi(..., operations=[...], confirm=True)` for the",
            "  writes, and the same for `import_mcp` and `import_a2a` tools that change data.",
            "- Give the executor an approval port: `CallbackApproval`, `DeferredApproval` or",
            "  `AutoDeny`. With none, a `confirm=True` call is denied (it fails closed).",
            "- Approve only through the executor: its approval port, or",
            '  `executor.resume(approval_id, decision)`. A framework\'s own "approved" never',
            "  counts alone; only a one-use decision issued by the adapter does.",
            "- Run tools through the executor (`executor.run`, `run_sync`, `aa.call`, or",
            "  `attach(..., executor=executor)`). Calling the function directly skips validation,",
            "  approval, limits and the ledger.",
            "- Pass `executor=executor` to every server and app (`create_app`, MCP, A2A,",
            "  `attach`); without it they use the default executor, not your approval port.",
            "- Set limits: `RunLimits(max_calls=)`, `BudgetHook(max_usd)`, and `timeout=` on",
            "  slow tools.",
            '- Raise `ToolError("...")` for a message the model should see; other exceptions',
            "  reach the model only as `tool-failed`.",
            "- Require auth on every server: `bearer_auth` (HTTP), `static_bearer` (A2A),",
            "  `auth=` with `token_verifier=` (MCP). Without auth an app serves loopback only.",
            "- Keep `run_id` and the call id with an approval id, so a resume matches its call.",
            "",
            "Don't:",
            "",
            "- Never pass an approval to `attach()`; approval belongs to the executor",
            "  (`attach(..., approval=...)` raises `ToolUsageError`).",
            "- Never log, print or return secrets, tokens, passwords or full personal data.",
            "  `LoggingHook` redacts secret-named keys; `agent_adapter.hooks.redact` does it",
            "  by hand.",
            "- Never make a secret a model-visible argument: bind it with `fixed=` or read it",
            "  from `ToolContext`.",
            '- Never use `cache_scope="global"` for per-user, secret or personal data: global',
            "  means every caller of the cache gets the same result.",
            "- Never set `retries=` on a tool with a side effect unless it is idempotent: a",
            "  timed-out sync tool keeps running in its thread, so a retry can repeat it.",
            "- Never import from `agent_adapter.core` or a `_private` module.",
            "- Never use `AutoApprove` for side-effect tools outside tests.",
            "- Never add `--extra-index-url` for this package (dependency confusion).",
            "- Never serve with `allow_anonymous` on a public host.",
            "",
        ]
    )


def _absolute(text: str, page: str) -> str:
    """Rewrite relative Markdown links on `page` (relative to docs/) as site URLs."""

    def url(match: re.Match[str]) -> str:
        label, target = match.groups()
        if "://" in target or target.startswith("#"):
            return match.group(0)
        path, _, anchor = target.partition("#")
        joined = posixpath.normpath(posixpath.join(posixpath.dirname(page), path))
        joined = joined.removesuffix(".md").removesuffix("index")
        joined = joined + "/" if joined and not joined.endswith("/") else joined
        return f"[{label}]({SITE_URL}{joined}{'#' + anchor if anchor else ''})"

    return LINK.sub(url, text)


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", row.strip())[1:-1]]


def _tables(text: str) -> list[list[list[str]]]:
    """Every Markdown table in `text` as rows of cells (header first, no separator)."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.startswith("|"):
            if not re.fullmatch(r"\|[-:| ]+\|", line.strip()):
                current.append(_cells(line))
        elif current:
            tables.append(current)
            current = []
    return tables


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def errors(*, full: bool) -> str:
    """The error catalogue; the full tier adds every troubleshooting table."""
    page = (DOCS / "reference" / "errors.md").read_text(encoding="utf-8")
    block = page.split(CATALOGUE[0], 1)[1].split(CATALOGUE[1], 1)[0]
    (catalogue,) = _tables(block)
    names = ("Error", "Kind", "Retryable", "Fix") if full else ("Error", "Fix")
    pick = [catalogue[0].index(name) for name in names]
    rows = [_row([row[i] for i in pick]) for row in catalogue[1:]]
    if not full:  # concise: link labels only; the full guide keeps the URLs
        rows = [LINK.sub(r"\1", row) for row in rows]
    out = [
        "## Common errors → fixes",
        "",
        "Every error subclasses `AdapterError` and carries a stable `kind`.",
        "",
        _row(list(names)),
        "|" + "---|" * len(names),
        _absolute(NL.join(rows), "reference/errors.md"),
        "",
    ]
    if not full:
        out += [f"Symptoms and their fixes: <{SITE_URL}troubleshooting/>.", ""]
        return NL.join(out)
    out += ["| Symptom or message | Cause | Fix |", "|---|---|---|"]
    trouble = (DOCS / "troubleshooting.md").read_text(encoding="utf-8")
    for table in _tables(trouble):
        if len(table[0]) == 3 and table[0][-1] == "Fix":
            out.append(_absolute(NL.join(_row(row) for row in table[1:]), "troubleshooting.md"))
    out.append("")
    return NL.join(out)


def cli() -> str:
    """Every command as one line."""
    lines = [f"- {line}" for line in command_lines()]
    lines += [
        "- `agent-adapter guide --full`: the guide plus every public signature",
        "- `python -m agent_adapter guide --section recipes`: the same CLI without the script",
        "- `python -m agent_adapter.server --registry mypkg.tools:registry`: serve a registry"
        " over HTTP (tokens from `AGENT_ADAPTER_TOKENS`)",
        "- `python -m agent_adapter.optimiser --host mypkg.evals:build_host --dry-run`: plan an"
        " optimiser run without calling a model",
    ]
    return NL.join(["## CLI commands", "", *lines, ""])


def full_api_pointer() -> str:
    """Where the concise guide sends an agent for every signature."""
    return NL.join(
        [
            "## Full API index",
            "",
            "Every public name with its signature and first docstring line, by module:",
            "`agent-adapter guide --section api`, `agent-adapter guide --full`, or",
            f"<{SITE_URL}llms-full.txt>.",
            "",
        ]
    )
