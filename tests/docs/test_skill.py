"""The agent skill and its Claude Code plugin files stay valid and match the API."""

import json
import re
from pathlib import Path

import agent_adapter

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "skills" / "agent-adapter" / "SKILL.md"
PLUGIN = ROOT / ".claude-plugin"


def _frontmatter() -> dict[str, str]:
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n"), "frontmatter must open on the first line"
    block = text.split("---\n", 2)[1]
    return dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)


def test_frontmatter_follows_the_agent_skills_spec() -> None:
    meta = _frontmatter()
    assert meta["name"] == SKILL.parent.name
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"])
    assert 0 < len(meta["description"]) <= 1024
    assert set(meta) <= {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}


def test_skill_is_short() -> None:
    assert len(SKILL.read_text(encoding="utf-8").splitlines()) < 150


def test_named_api_exists() -> None:
    names = ["Executor", "attach", "CallbackApproval", "DeferredApproval", "AutoDeny",
             "RunLimits", "BudgetHook", "ToolContext", "run_sync"]
    text = SKILL.read_text(encoding="utf-8")
    for name in names:
        assert name in text
        assert hasattr(agent_adapter, name), name


def test_plugin_files_point_at_the_skill() -> None:
    market = json.loads((PLUGIN / "marketplace.json").read_text(encoding="utf-8"))
    plugin = json.loads((PLUGIN / "plugin.json").read_text(encoding="utf-8"))
    assert market["owner"]["name"] and market["name"] == "agent-adapter"
    [entry] = market["plugins"]
    assert entry["source"] == "./" and entry["name"] == plugin["name"] == "agent-adapter"
    assert "skills" not in entry, "an explicit list would stop the default skills/ scan"
