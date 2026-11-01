import json
from types import SimpleNamespace

import pytest

from agent_adapter.core.errors import ApprovalPending, ToolArgsInvalid, ToolError
from agent_adapter.core.types import ErrorInfo, ToolResult
from agent_adapter.integrations import anthropic, gemini, openai
from agent_adapter.integrations.base import Integration, IntegrationBase
from tests.integrations.sample_tools import targeted_registry as registry


@pytest.mark.parametrize(
    ("module", "name", "requires", "extra"),
    [
        (openai, "openai", (), ""),
        (gemini, "gemini", ("google.genai",), "gemini"),
        (anthropic, "anthropic", (), ""),
    ],
)
def test_each_renderer_module_exposes_an_integration(module, name, requires, extra):
    integration = module.INTEGRATION
    assert isinstance(integration, IntegrationBase)
    assert isinstance(integration, Integration)
    assert (integration.name, integration.requires, integration.extra) == (name, requires, extra)
    assert integration.bind_approval(object()) is None


def test_openai_integration_exports_chat_tools_by_default():
    reg = registry()
    exported = openai.INTEGRATION.export_tools(reg)
    assert exported == openai.chat_tools(reg)
    assert [t["function"]["name"] for t in exported] == ["lookup", "only_openai"]


def test_openai_integration_can_export_responses_tools_without_strict():
    reg = registry()
    integration = openai.OpenAIIntegration(api="responses", strict=False)
    assert integration.export_tools(reg) == openai.responses_tools(reg, strict=False)
    assert integration.export_tool(reg.get("lookup"))["strict"] is False


def test_openai_integration_rejects_an_unknown_api():
    with pytest.raises(ValueError, match="api must be"):
        openai.OpenAIIntegration(api="assistants")  # type: ignore[arg-type]


def test_gemini_and_anthropic_integrations_export_their_renderer_dicts():
    reg = registry()
    assert gemini.INTEGRATION.export_tools(reg) == gemini.function_declarations(reg)
    assert anthropic.INTEGRATION.export_tools(reg) == anthropic.anthropic_tools(reg)


@pytest.mark.parametrize(
    ("integration", "native", "call_id"),
    [
        (openai.INTEGRATION, {"id": "call_chat", "type": "function"}, "call_chat"),
        (openai.INTEGRATION, {"id": "fc_1", "call_id": "call_resp"}, "call_resp"),
        (openai.INTEGRATION, SimpleNamespace(id="fc_2", call_id="call_obj"), "call_obj"),
        (gemini.INTEGRATION, SimpleNamespace(id="g_1", name="lookup", args={}), "g_1"),
        (gemini.INTEGRATION, {"name": "lookup", "args": {}}, ""),
        (anthropic.INTEGRATION, SimpleNamespace(id="toolu_1", type="tool_use"), "toolu_1"),
        (anthropic.INTEGRATION, {"id": "toolu_2", "type": "tool_use"}, "toolu_2"),
    ],
)
def test_bind_context_carries_the_provider_call_id(integration, native, call_id):
    ctx = integration.bind_context(native)
    assert ctx.call_id == call_id
    assert ctx.extras == {integration.name: native}


def ok(call_id: str, value: object) -> ToolResult:
    return ToolResult(call_id=call_id, name="lookup", status="ok", value=value)


def failed(call_id: str, exc: Exception) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        name="lookup",
        status="pending" if isinstance(exc, ApprovalPending) else "error",
        error=ErrorInfo.from_exception(exc),
        exception=exc,
    )


def test_tool_result_blocks_render_ok_values_as_text_or_json():
    blocks = anthropic.tool_result_blocks([ok("toolu_1", "80C"), ok("toolu_2", {"rate": 18.0})])
    assert blocks == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "80C", "is_error": False},
        {
            "type": "tool_result",
            "tool_use_id": "toolu_2",
            "content": '{"rate": 18.0}',
            "is_error": False,
        },
    ]


def test_tool_result_blocks_render_none_and_non_json_values():
    blocks = anthropic.tool_result_blocks([ok("a", None), ok("b", {1, 2})])
    assert blocks[0]["content"] == "null"
    assert sorted(json.loads(blocks[1]["content"])) == [1, 2]


def test_tool_result_blocks_render_errors_as_model_visible_json():
    args_error = ToolArgsInvalid("lookup", [{"loc": ("section",), "msg": "x", "type": "missing"}])
    blocks = anthropic.tool_result_blocks(
        [failed("t1", ToolError("no such section", tool="lookup")), failed("t2", args_error)]
    )
    assert blocks[0] == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": json.dumps({"error": {"kind": "tool-error", "message": "no such section"}}),
        "is_error": True,
    }
    assert json.loads(blocks[1]["content"])["error"]["kind"] == "tool-args-invalid"
    assert blocks[1]["is_error"] is True


def test_tool_result_blocks_hide_foreign_exception_text():
    block = anthropic.tool_result_blocks([failed("t1", KeyError("db-password"))])[0]
    assert "db-password" not in block["content"]
    assert json.loads(block["content"]) == {
        "error": {"kind": "internal-error", "message": "KeyError"}
    }


def test_tool_result_blocks_render_an_error_without_error_info():
    result = ToolResult(call_id="t1", name="lookup", status="error")
    assert json.loads(anthropic.tool_result_blocks([result])[0]["content"]) == {
        "error": {"kind": "internal-error", "message": "tool 'lookup' failed"}
    }


def test_tool_result_blocks_render_pending_approval():
    block = anthropic.tool_result_blocks([failed("t1", ApprovalPending("apr_1", tool="lookup"))])[0]
    assert block == {
        "type": "tool_result",
        "tool_use_id": "t1",
        "content": "approval required: apr_1",
        "is_error": True,
    }


def test_tool_result_blocks_of_nothing_is_empty():
    assert anthropic.tool_result_blocks([]) == []
