import asyncio
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import mypy.api

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_anthropic_results_answers_each_tool_use(capsys):
    runpy.run_path(str(EXAMPLES / "anthropic_results.py"), run_name="__main__")
    reply = json.loads(capsys.readouterr().out)
    assert reply["role"] == "user"
    ok, failed = reply["content"]
    assert ok == {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": "18.0",
        "is_error": False,
    }
    assert failed["tool_use_id"] == "toolu_2" and failed["is_error"] is True
    assert json.loads(failed["content"]) == {
        "error": {"kind": "tool-error", "message": "no rate for 'mars'"}
    }


def test_discover_integrations_finds_and_exports_the_built_ins(capsys):
    runpy.run_path(str(EXAMPLES / "discover_integrations.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "shipping_rate",
        "['anthropic', 'gemini', 'openai']",
    ]


def _providers_with_wipe() -> tuple[dict, list[str]]:
    """The providers example plus a confirm tool that records whether it ran."""
    ns = runpy.run_path(str(EXAMPLES / "providers.py"), run_name="providers_example")
    ran: list[str] = []

    @ns["registry"].tool(confirm=True, effects="destructive")
    def wipe(target: str) -> str:
        """Delete everything in a target."""
        ran.append(target)
        return "wiped"

    return ns, ran


def test_providers_openai_round_trip_denies_a_confirm_tool_without_running_it():
    ns, ran = _providers_with_wipe()
    call = SimpleNamespace(
        id="call_9", function=SimpleNamespace(name="wipe", arguments='{"target": "db"}')
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[call]))]
    )
    _, replies = asyncio.run(ns["openai_round_trip"](response))
    assert ran == []
    assert replies[0]["tool_call_id"] == "call_9"
    assert json.loads(replies[0]["content"])["error"]["kind"] == "approval-denied"


def test_providers_anthropic_round_trip_denies_a_confirm_tool_without_running_it():
    ns, ran = _providers_with_wipe()
    block = SimpleNamespace(type="tool_use", id="toolu_9", name="wipe", input={"target": "db"})
    _, reply = asyncio.run(ns["anthropic_round_trip"](SimpleNamespace(content=[block])))
    assert ran == []
    (answer,) = reply["content"]
    assert answer["tool_use_id"] == "toolu_9" and answer["is_error"] is True
    assert json.loads(answer["content"])["error"]["kind"] == "approval-denied"


def test_provider_pages_never_invoke_specs_directly():
    docs = EXAMPLES.parent / "docs" / "integrations"
    for path in (EXAMPLES / "providers.py", docs / "openai.md", docs / "anthropic.md"):
        assert ".invoke(" not in path.read_text(encoding="utf-8"), path


def test_integration_examples_type_check_strictly():
    names = ("discover_integrations.py", "anthropic_results.py", "providers.py")
    files = [str(EXAMPLES / name) for name in names]
    out, err, status = mypy.api.run(["--strict", "--no-error-summary", *files])
    assert status == 0, out + err
