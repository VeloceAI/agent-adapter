"""Native conversations in every dict format, each with a system prompt, tool rounds and padding."""

from __future__ import annotations

import json

PAD = "x" * 400  # makes an old turn worth removing
DENIED = json.dumps({"error": {"kind": "approval-denied", "message": "no"}})


def openai_chat() -> list[dict]:
    return [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": f"old question {PAD}", "name": "ann"},
        {"role": "assistant", "content": f"old answer {PAD}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "a"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": f"result {PAD}"},
        {"role": "user", "content": [{"type": "text", "text": "new"}, {"type": "image_url"}]},
    ]


def openai_responses() -> list[dict]:
    return [
        {"role": "developer", "content": "Be brief."},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": PAD}]},
        {"type": "reasoning", "id": "rs_1", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": PAD}]},
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": '{"q":"a"}'},
        {"type": "function_call", "call_id": "c2", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": f"one {PAD}"},
        {"type": "function_call_output", "call_id": "c2", "output": DENIED},
        {"role": "user", "content": "new"},
    ]


def anthropic() -> list[dict]:
    return [
        {"role": "user", "content": f"old question {PAD}"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hm", "signature": "sig"},
                {"type": "text", "text": f"let me look {PAD}"},
                {"type": "tool_use", "id": "t1", "name": "lookup", "input": {"q": "a"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": f"found {PAD}"},
                {"type": "text", "text": "and then?", "cache_control": {"type": "ephemeral"}},
            ],
        },
        {"role": "assistant", "content": f"done {PAD}"},
        {"role": "user", "content": "new"},
    ]


def gemini() -> list[dict]:
    return [
        {"role": "user", "parts": [{"text": f"old question {PAD}"}]},
        {
            "role": "model",
            "parts": [
                {"text": "thinking", "thought": True},
                {"text": f"looking {PAD}"},
                {"functionCall": {"name": "lookup", "args": {"q": "a"}}},
            ],
        },
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "lookup", "response": {"rows": PAD}}}],
        },
        {"role": "model", "parts": [{"text": f"done {PAD}"}]},
        {"role": "user", "parts": [{"text": "new"}]},
    ]


CASES = {
    "openai": openai_chat,
    "openai_responses": openai_responses,
    "anthropic": anthropic,
    "gemini": gemini,
}
FORMAT_OF = {"openai_responses": "openai"}


# --- approval state in every format: each case's approval natives must survive any step ------


def openai_responses_approval() -> list[dict]:
    return [
        {"role": "developer", "content": "Be brief."},
        {"role": "user", "content": f"old {PAD}"},
        {
            "type": "mcp_approval_request",
            "id": "mcpr_1",
            "server_label": "crm",
            "name": "delete_contact",
            "arguments": '{"id": 7}',
        },
        {"type": "mcp_approval_response", "approval_request_id": "mcpr_1", "approve": False},
        {"role": "assistant", "content": f"old answer {PAD}"},
        {"role": "user", "content": f"middle {PAD}"},
        {"role": "assistant", "content": f"middle answer {PAD}"},
        {"role": "user", "content": "new"},
    ]


def openai_chat_approval() -> list[dict]:
    chat = openai_chat()
    chat[4] = {"role": "tool", "tool_call_id": "c1", "content": DENIED}
    return [*chat[:-1], {"role": "assistant", "content": f"later {PAD}"}, chat[-1]]


def anthropic_approval() -> list[dict]:
    return [
        {"role": "user", "content": f"old question {PAD}"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"asking {PAD}"},
                {"type": "tool_use", "id": "t1", "name": "refund", "input": {"amount": 5}},
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": DENIED}],
        },
        {"role": "user", "content": [{"type": "tool_confirmation", "text": f"no {PAD}"}]},
        {"role": "assistant", "content": f"done {PAD}"},
        {"role": "user", "content": "new"},
    ]


def gemini_confirmation() -> list[dict]:
    ask = {"name": "adk_request_confirmation", "args": {"tool": "refund"}}
    return [
        {"role": "user", "parts": [{"text": f"old question {PAD}"}]},
        {"role": "model", "parts": [{"text": f"asking {PAD}"}, {"functionCall": ask}]},
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "adk_request_confirmation",
                        "response": {"confirmed": False, "note": PAD},
                    }
                }
            ],
        },
        {"role": "model", "parts": [{"text": f"done {PAD}"}]},
        {"role": "user", "parts": [{"text": "new"}]},
    ]


APPROVAL_CASES = {
    "openai_responses": openai_responses_approval,
    "openai": openai_chat_approval,
    "anthropic": anthropic_approval,
    "gemini": gemini_confirmation,
}
