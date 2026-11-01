"""Redaction before summarising hides secret values only; ordinary settings and words stay."""

import json

from agent_adapter.context import Summarize, compress
from agent_adapter.context.redaction import redact_text
from tests.context.ai_fakes import FakeSummariser
from tests.context.conversations import assistant, policy, tool_round, user

API_KEY = "sk-proj-abcdefghijklmnop1234"
JWT = "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.c2lnbmF0dXJlMTIz"


def chars(messages):
    return sum(len(m.content) for m in messages)


def test_text_keeps_ordinary_settings_and_hides_secret_values():
    text = (
        f"max_tokens=800 tokens: 12 token_count=3 keyboard: qwerty api_key={API_KEY} password: x1"
    )
    masked = redact_text(text)
    for kept in ("max_tokens=800", "tokens: 12", "token_count=3", "keyboard: qwerty"):
        assert kept in masked, kept
    assert API_KEY not in masked and "x1" not in masked
    assert "api_key=***" in masked and "password: ***" in masked


def test_json_keeps_ordinary_keys_and_hides_secret_values():
    body = json.dumps(
        {
            "max_tokens": 800,
            "usage": {"output_tokens": 5, "total_tokens": 9},
            "cache_key": "aa:llm:1",
            "access_token": "t-1",
            "headers": {"Authorization": "Bearer abcdefgh", "Accept": "json"},
        }
    )
    masked = json.loads(redact_text(body))
    assert masked["max_tokens"] == 800 and masked["usage"] == {
        "output_tokens": 5,
        "total_tokens": 9,
    }
    assert masked["cache_key"] == "aa:llm:1"
    assert masked["access_token"] == "***" and masked["headers"]["Authorization"] == "***"
    assert masked["headers"]["Accept"] == "json"


def test_secret_shaped_values_are_hidden_under_any_key():
    masked = redact_text(json.dumps({"note": f"use {API_KEY}", "session": JWT, "id": "row-42"}))
    assert API_KEY not in masked and JWT not in masked and "row-42" in masked


async def test_a_summary_keeps_max_tokens_800_but_never_sees_the_api_key():
    config = f"settings: max_tokens=800, model=gpt-5, api_key={API_KEY}"
    messages = [
        user("configure the client " + "u" * 300),
        *tool_round("c1", config + " " + "r" * 300, args={"max_tokens": 800, "api_key": API_KEY}),
        assistant("done " + "a" * 300),
        user("next"),
    ]
    model = FakeSummariser()
    p = policy(300, 100, [Summarize(model, keep_recent_messages=1)], count_tokens=chars)
    await compress(messages, p)
    sent = model.requests[0].messages[-1].content
    assert "max_tokens=800" in sent and '"max_tokens": 800' in sent
    assert API_KEY not in sent and "api_key=***" in sent


def test_plural_and_bare_secret_names_are_hidden_in_text_and_json():
    assert json.loads(redact_text('{"secrets": "s3", "cookies": "c"}')) == {
        "secrets": "***",
        "cookies": "***",
    }
    assert redact_text("my secrets=topsecret max_tokens=5") == "my secrets=*** max_tokens=5"
    assert json.loads(redact_text('{"private_key_pem": "abc"}')) == {"private_key_pem": "***"}
    assert redact_text("key=plainsecret key=id") == "key=*** key=id"


async def test_the_review_repro_never_reaches_the_summariser():
    arguments = {"api_keys": ["abc123secretvalue"], "passwords": "hunter2", "key": "plainsecret"}
    before = [
        user("u" * 300),
        *tool_round("s1", "done", args=arguments),
        assistant("a" * 300),
        user("now"),
    ]
    model = FakeSummariser()
    await compress(
        before, policy(200, 50, [Summarize(model, keep_recent_messages=1)], count_tokens=chars)
    )
    sent = model.requests[0].messages[-1].content
    for secret in ("abc123secretvalue", "hunter2", "plainsecret"):
        assert secret not in sent, secret
