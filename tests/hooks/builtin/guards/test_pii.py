import re

import pytest

from agent_adapter.core.errors import ConfigError, GuardrailBlocked
from agent_adapter.hooks.builtin.guards import (
    PII_KINDS,
    PIIGuard,
    find_pii,
    iban_ok,
    luhn_ok,
    redact_pii,
)
from tests.hooks.builtin.helpers import llm_inv, llm_response, ok, tool_inv


def kinds(text, **kw):
    return [m.kind for m in find_pii(text, **kw)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mail ana@example.com now", ["email"]),
        ("ana.b+shop@mail.example.co.uk", ["email"]),
        ("card 4111 1111 1111 1111", ["card"]),
        ("card 4111-1111-1111-1111", ["card"]),
        ("card 4111111111111111", ["card"]),
        ("call +1 (415) 555-0123", ["phone"]),
        ("call 415-555-0123, or 415-555-0124", ["phone", "phone"]),
        ("ring +441632960961", ["phone"]),
        ("iban GB82 WEST 1234 5698 7654 32", ["iban"]),
        ("iban GB82WEST12345698765432", ["iban"]),
        ("key AKIAIOSFODNN7EXAMPLE", ["secret"]),
        ("token ghp_" + "a" * 36, ["secret"]),
        ("sk-" + "b" * 24, ["secret"]),
        ("sk_live_" + "c" * 20, ["secret"]),
        ("api_key = 'abcd1234efgh'", ["secret"]),
        ("PASSWORD: hunter2hunter2", ["secret"]),
        ("-----BEGIN RSA PRIVATE KEY-----", ["secret"]),
        ("jwt eyJhbGciOiJI.eyJzdWIiOiIx.c2lnbmF0dXJl", ["secret"]),
    ],
)
def test_find_pii_finds_each_kind(text, expected):
    assert kinds(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "card 4111 1111 1111 1112",  # fails Luhn
        "order 4155550123",  # bare digits are not a phone
        "server 192.168.100.200",
        "total 12345678.90",
        "date 2026-11-05 at 10:30",
        "iban GB00WEST12345698765432",  # fails mod 97
        "token: short",
        "",
        "user@localhost",
    ],
)
def test_find_pii_ignores_lookalikes(text):
    assert kinds(text) == []


def test_find_pii_respects_kinds_and_custom_patterns():
    text = "ana@example.com ticket TK-12345"
    assert kinds(text, kinds=["phone"]) == []
    found = kinds(text, kinds=["email"], patterns={"ticket": re.compile(r"TK-\d+")})
    assert found == ["email", "ticket"]


def test_overlapping_findings_keep_the_first():
    # the secret rule spans the whole assignment, including the email-looking value
    assert kinds("token=ana@example.com") == ["secret"]


def test_checksums():
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")
    assert not luhn_ok("")
    assert iban_ok("GB82 WEST 1234 5698 7654 32")
    assert not iban_ok("GB83WEST12345698765432")


def test_redact_pii_uses_the_placeholder():
    text = "ana@example.com / 4111 1111 1111 1111"
    assert redact_pii(text, find_pii(text)) == "[EMAIL] / [CARD]"
    assert redact_pii(text, find_pii(text), "<{kind}>") == "<EMAIL> / <CARD>"
    assert redact_pii("plain", []) == "plain"


async def test_blocks_args_naming_kinds_never_the_value(caplog):
    guard = PIIGuard()
    inv = tool_inv(arguments={"user": "ana@example.com", "note": "x"})
    with caplog.at_level("DEBUG"), pytest.raises(GuardrailBlocked) as info:
        await guard.before_tool(inv)
    assert info.value.reason == "found email"
    assert info.value.stage == "args" and info.value.tool == "login"
    assert "ana@example.com" not in caplog.text
    assert "ana@example.com" not in str(info.value)


async def test_clean_payloads_pass():
    guard = PIIGuard(on=("args", "result", "llm_request", "llm_response"))
    assert await guard.before_tool(tool_inv(arguments={"user": "ana", "note": "hi"})) is None
    assert await guard.after_tool(tool_inv(), ok("fine")) is None
    assert await guard.before_llm(llm_inv()) is None
    assert await guard.after_llm(llm_inv(), llm_response()) is None


async def test_redacts_results_and_model_replies():
    guard = PIIGuard(action="redact", on=("result", "llm_response"))
    value = {"rows": [{"email": "ana@example.com", "n": 4111111111111111, "ok": True}]}
    replaced = await guard.after_tool(tool_inv(), ok(value))
    assert replaced.value == {"rows": [{"email": "[EMAIL]", "n": "[CARD]", "ok": True}]}
    assert value["rows"][0]["email"] == "ana@example.com"  # the original is untouched
    reply = await guard.after_llm(llm_inv(), llm_response("write to ana@example.com"))
    assert reply.text == reply.message.content == "write to [EMAIL]"


async def test_request_stage_blocks():
    guard = PIIGuard(on="llm_request", kinds=["secret"])
    inv = llm_inv()
    request = inv.request.model_copy(
        update={
            "messages": (inv.request.messages[0].model_copy(update={"content": "sk-" + "z" * 30}),)
        }
    )
    with pytest.raises(GuardrailBlocked, match="found secret"):
        await guard.before_llm(type(inv)(request, inv.provider, inv.model, inv.ctx))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"kinds": ["passport"]}, "unknown PII kinds"),
        ({"kinds": []}, "at least one kind"),
        ({"patterns": {"x": "("}}, "does not compile"),
        ({"placeholder": "[{other}]"}, "placeholder"),
        ({"on": ()}, "non-empty subset"),
        ({"on": ["args"], "action": "redact"}, "redact rewrites"),
        ({"on": ["result"], "action": "approve"}, "approve applies"),
        ({"action": "warn"}, "action must be one of"),
    ],
)
def test_bad_options_raise_config_error(options, message):
    with pytest.raises(ConfigError, match=message):
        PIIGuard(**options)


def test_only_patterns_is_enough():
    assert PIIGuard(kinds=[], patterns={"ticket": r"TK-\d+"}).kinds == ()
    assert set(PII_KINDS) == {"email", "phone", "card", "iban", "secret"}


# Review fix round 1: the reviewer's scenarios.


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("card 4111 1111 1111 1111 123", "4111 1111 1111 1111"),  # a card, then its CVV
        ("id 7 4111111111111111", "4111111111111111"),
        ("IBAN DE89 3704 0044 0532 0130 00 EUR", "DE89 3704 0044 0532 0130 00"),
        ("DE89370400440532013000 BIC", "DE89370400440532013000"),
        ("de89370400440532013000", "de89370400440532013000"),
        ("DE89370400440532013000BIC", "DE89370400440532013000"),
    ],
)
def test_cards_and_ibans_inside_longer_runs_are_found(text, found):
    matches = find_pii(text)
    assert [text[m.start : m.end] for m in matches] == [found]


def test_iban_ok_rejects_non_alphanumeric_text():
    assert not iban_ok("DE89-3704")
    assert not iban_ok("DÉ89370400440532013000")


@pytest.mark.parametrize(
    "payload",
    [{"password": "Tr0ub4dor&3xyz"}, {"api_key": "a8f3k2m9q7w1z5x0"}, {"ana@example.com": "vip"}],
)
async def test_secrets_by_key_and_pii_in_keys_are_blocked(payload):
    with pytest.raises(GuardrailBlocked):
        await PIIGuard(on="result").after_tool(tool_inv(), ok(payload))


async def test_secrets_by_key_and_pii_in_keys_are_redacted():
    guard = PIIGuard(on="result", action="redact")
    value = {"password": "Tr0ub4dor&3xyz", "ana@example.com": "vip", "max_tokens": "800"}
    redacted = (await guard.after_tool(tool_inv(), ok(value))).value
    assert redacted == {"password": "[SECRET]", "[EMAIL]": "vip", "max_tokens": "800"}


async def test_secret_names_are_ignored_without_the_secret_kind():
    assert (
        await PIIGuard(kinds=["email"], on="result").after_tool(
            tool_inv(), ok({"password": "Tr0ub4dor&3xyz"})
        )
        is None
    )


async def test_redacting_a_reply_rewrites_raw_items_and_tool_call_values():
    from agent_adapter.core.messages import Message, ProviderRaw
    from agent_adapter.core.types import ToolCall

    call = ToolCall(name="send", arguments={"to": "bob@shop.example"}, id="c1")
    thinking = {"type": "thinking", "thinking": "look up the shop", "signature": "sig-1"}
    raw = ProviderRaw(
        provider="anthropic", items=[thinking, {"type": "text", "text": "mail bob@shop.example"}]
    )
    message = Message(
        role="assistant", content="mail bob@shop.example", tool_calls=(call,), raw=raw
    )
    reply = llm_response("mail bob@shop.example").model_copy(
        update={"tool_calls": (call,), "message": message}
    )
    out = await PIIGuard(on="llm_response", action="redact").after_llm(llm_inv(), reply)
    assert "bob@shop.example" not in out.model_dump_json()
    assert out.message.raw.items == [thinking, {"type": "text", "text": "mail [EMAIL]"}]


async def test_a_request_guard_reads_raw_items_and_tool_calls():
    from agent_adapter.core.messages import LLMRequest, Message, ProviderRaw
    from agent_adapter.core.types import ToolCall

    hidden = Message(
        role="assistant",
        raw=ProviderRaw(provider="openai", items=[{"type": "text", "text": "bob@shop.example"}]),
    )
    called = Message(
        role="assistant", tool_calls=(ToolCall(name="s", arguments={"to": "bob@shop.example"}),)
    )
    guard = PIIGuard(on="llm_request")
    for message in (hidden, called):
        inv = llm_inv()
        request = LLMRequest(messages=(message,))
        with pytest.raises(GuardrailBlocked):
            await guard.before_llm(type(inv)(request, inv.provider, inv.model, inv.ctx))


def test_an_iban_ends_at_punctuation():
    text = "pay DE89370400440532013000, thanks; or 2: x"
    assert [text[m.start : m.end] for m in find_pii(text, ["iban"])] == ["DE89370400440532013000"]
    assert find_pii({1: "x"}.get(1), ["iban"]) == []


# Review fix round 2: the reviewer's payloads.


@pytest.mark.parametrize(
    "payload",
    [
        {"password": {"value": "Tr0ub4dor&3xyz"}},
        {"auth": {"type": "bearer", "value": "a8f3k2m9q7w1z5x0"}},
        {"headers": [["Authorization", "a8f3k2m9q7w1z5x0"]]},
        {"headers": [{"name": "X-Api-Key", "value": "a8f3k2m9q7w1z5x0"}]},
        {"passcode": "4321"},
        {"user": {"pin": "0000"}},
    ],
)
async def test_secret_context_reaches_every_level(payload):
    with pytest.raises(GuardrailBlocked, match="found secret"):
        await PIIGuard(on="result").after_tool(tool_inv(), ok(payload))


def test_passcode_assignments_in_text_are_found():
    assert kinds("passcode = 'abcd1234efgh'") == ["secret"]


async def test_redaction_is_by_place_never_by_equal_text():
    guard = PIIGuard(on="result", action="redact")
    out = await guard.after_tool(tool_inv(), ok({"token": "1", "count": "1", "n": 1}))
    assert out.value == {"token": "[SECRET]", "count": "1", "n": 1}


async def test_redacted_keys_never_drop_an_entry():
    guard = PIIGuard(on="result", action="redact")
    value = {"ana@example.com": "vip", "bob@example.com": "gold"}
    out = await guard.after_tool(tool_inv(), ok(value))
    assert out.value == {"[EMAIL]": "vip", "[EMAIL]#2": "gold"}


async def test_nested_secrets_are_redacted_in_place():
    guard = PIIGuard(on="result", action="redact")
    value = {"headers": [["Authorization", "a8f3k2m9q7w1z5x0"], ["Accept", "a8f3k2m9q7w1z5x0"]]}
    out = await guard.after_tool(tool_inv(), ok(value))
    assert out.value == {"headers": [["Authorization", "[SECRET]"], ["Accept", "a8f3k2m9q7w1z5x0"]]}
