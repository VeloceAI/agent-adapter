"""The support matrix page is generated from pyproject.toml and noxfile.py, so it cannot drift.

Regenerate the tables with `python -m tests.docs.test_support_matrix` and paste the output
between the markers in docs/support-matrix.md.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "support-matrix.md"
START, END = "<!-- support-matrix:start -->", "<!-- support-matrix:end -->"
GATE = re.compile(r"^_?SUPPORTED(?:_SDK)? = \(\(([\d, ]+)\), \(([\d, ]+)\)\)", re.M)
OWN_ENV = re.compile(r"([a-z][a-z0-9-]*) \(([^)]+)\)")
SKIPPED = {"dev", "docs", "all"}


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _nox() -> ast.Module:
    return ast.parse((ROOT / "noxfile.py").read_text(encoding="utf-8"))


def _constant(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        if isinstance(node, ast.Assign):
            target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name and node.value is not None:
            return ast.literal_eval(node.value)
    raise AssertionError(f"noxfile.py has no constant {name}")


def _session_extras(tree: ast.Module) -> dict[str, set[str]]:
    """Fixed nox sessions -> the extras their `-e .[...]` installs name (without dev)."""
    found: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
            continue
        name = node.name
        for decorator in node.decorator_list:
            for keyword in getattr(decorator, "keywords", []):
                if keyword.arg == "name" and isinstance(keyword.value, ast.Constant):
                    name = keyword.value.value
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                match = re.fullmatch(r"\.\[([\w,-]+)\]", sub.value)
                if match:
                    found.setdefault(name, set()).update(match.group(1).split(","))
    return {name: extras - {"dev"} for name, extras in found.items()}


def _all_members(extras: dict[str, list[str]]) -> set[str]:
    members: set[str] = set()
    for requirement in extras["all"]:
        match = re.fullmatch(r"agent-adapter\[([\w,-]+)\]", requirement)
        assert match, requirement
        members.update(match.group(1).split(","))
    return members


def _own_env_reasons() -> dict[str, str]:
    """`extra -> reason` from the pyproject comment above `all = [`."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    comment = text.split("Own environment:", 1)[1].split("all = [", 1)[0]
    comment = " ".join(line.lstrip("# ").strip() for line in comment.splitlines())
    return dict(OWN_ENV.findall(comment))


def _pins(requirements: list[str]) -> str:
    own = [r for r in requirements if not r.startswith("agent-adapter[")]
    via = [r.split("[", 1)[1].rstrip("]") for r in requirements if r.startswith("agent-adapter[")]
    parts = [f"`{r}`" for r in own] + [f"extra `{v}`" for v in via]
    return ", ".join(parts)


def _title(target: str) -> tuple[str, str]:
    page = f"integrations/{target.replace('_', '-')}.md"
    heading = (ROOT / "docs" / page).read_text(encoding="utf-8").splitlines()[0]
    return heading.removeprefix("# ").strip(), page


def _gate(target: str) -> str:
    base = ROOT / "agent_adapter" / "integrations"
    for path in (base / f"{target}.py", base / target / "_support.py"):
        if path.is_file() and (match := GATE.search(path.read_text(encoding="utf-8"))):
            low, high = (".".join(p.strip() for p in g.split(",")) for g in match.groups())
            return f"{low} to below {high}"
    return "none"


def render() -> str:
    """Both tables, exactly as they appear between the page's markers."""
    extras: dict[str, list[str]] = _pyproject()["project"]["optional-dependencies"]
    tree = _nox()
    targets: dict[str, str] = _constant(tree, "EXTRAS")
    pythons = ", ".join(_constant(tree, "PYTHONS"))
    in_all, reasons = _all_members(extras), _own_env_reasons()
    rows = [
        "| Integration | Module | Extra | Pinned range | Environment | Contract session "
        f"(Python {pythons}) | Checked at use |",
        "|---|---|---|---|---|---|---|",
    ]
    for target, extra in targets.items():
        title, page = _title(target)
        if not extra:
            env, pins, shown = "core only", "none (plain dicts)", "none"
        else:
            shown, pins = f"`{extra}`", _pins(extras[extra])
            env = "in `all`" if extra in in_all else f"own: {reasons[extra]}"
        session = f"`contract-{target}`"
        if not extra or extra in in_all:
            session += ", `contract-all-compatible`"
        rows.append(
            f"| [{title}]({page}) | `{target}` | {shown} | {pins} | {env} | {session} | "
            f"{_gate(target)} |"
        )
    sessions = _session_extras(tree)
    contract = {extra: f"contract-{target}" for target, extra in targets.items() if extra}
    rows += [
        "",
        "| Extra | Pinned range | In `all` | Installed by nox sessions |",
        "|---|---|---|---|",
    ]
    for extra, requirements in extras.items():
        if extra in SKIPPED:
            continue
        names = sorted(
            {n for n, used in sessions.items() if extra in used}
            | ({contract[extra]} if extra in contract else set())
        )
        if extra in in_all:
            names.append("contract-all-compatible")
        listed = ", ".join(f"`{n}`" for n in names) or "none"
        shared = "yes" if extra in in_all else "no"
        rows.append(f"| `{extra}` | {_pins(requirements)} | {shared} | {listed} |")
    return "\n".join(rows)


def _block() -> str:
    text = PAGE.read_text(encoding="utf-8")
    return text.split(START, 1)[1].split(END, 1)[0].strip()


def test_support_matrix_matches_pyproject_and_noxfile() -> None:
    assert _block() == render(), "regenerate: python -m tests.docs.test_support_matrix"


def test_every_contract_target_and_extra_is_listed() -> None:
    block = _block()
    for target in _constant(_nox(), "EXTRAS"):
        assert f"`contract-{target}`" in block
    for extra in set(_pyproject()["project"]["optional-dependencies"]) - SKIPPED:
        assert f"| `{extra}` |" in block


def test_own_environment_reasons_cover_every_extra_outside_all() -> None:
    extras = _pyproject()["project"]["optional-dependencies"]
    targets = _constant(_nox(), "EXTRAS").values()
    outside = {e for e in targets if e and e not in _all_members(extras)}
    assert outside == set(_own_env_reasons())


if __name__ == "__main__":
    print(render())
