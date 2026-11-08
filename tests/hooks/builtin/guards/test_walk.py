import dataclasses
from collections import namedtuple

from pydantic import BaseModel

from agent_adapter.core.messages import LLMRequest, Message, ProviderRaw
from agent_adapter.core.types import ToolCall
from agent_adapter.hooks.builtin.guards._walk import MAX_DEPTH, Field, fields, rewrite
from tests.hooks.builtin.helpers import llm_response


class Row(BaseModel):
    email: str
    tags: list[str] = []


@dataclasses.dataclass
class Box:
    label: str
    hidden: str = dataclasses.field(default="h", init=False)


Pair = namedtuple("Pair", "left right")


class Opaque:
    def __str__(self):
        return "opaque-text"


def upper(path, text):
    return text.upper()


def same(path, text):
    return text


def texts(stage, payload):
    return list(dict.fromkeys(field.text for field in fields(stage, payload)))


def test_texts_walks_every_container_in_order():
    value = {
        "a": ["x", ("y", 7)],
        "b": {"x"},
        "c": Row(email="r", tags=["t"]),
        "d": Box("box"),
        "e": None,
        "f": True,
        "g": 1.5,
        "h": Opaque(),
    }
    assert texts("result", value) == [
        *("a", "x", "y", "7", "b", "c", "r", "t", "d", "box", "h", "e", "f", "g", "opaque-text")
    ]


def test_texts_of_model_messages():
    request = LLMRequest(messages=(Message(role="user", content="hi"), Message(role="user")))
    assert texts("llm_request", request) == ["hi"]
    response = llm_response("ok").model_copy(update={"parsed": {"k": "v"}})
    assert texts("llm_response", response) == ["ok", "k", "v"]


def test_rewrite_keeps_types_and_untouched_values():
    value = {
        "list": ["a"],
        "tuple": ("b",),
        "named": Pair("c", 1),
        "set": frozenset({"d"}),
        "model": Row(email="e", tags=["f"]),
        "box": Box("g"),
        "n": 12,
        "flag": False,
        "obj": Opaque(),
        "none": None,
    }
    out = rewrite("result", value, upper)
    assert out["LIST"] == ["A"] and out["TUPLE"] == ("B",)
    assert out["NAMED"] == Pair("C", 1) and isinstance(out["NAMED"], Pair)
    assert out["SET"] == frozenset({"D"})
    assert out["MODEL"] == Row(email="E", tags=["F"])
    assert out["BOX"].label == "G"
    assert out["N"] == 12 and out["FLAG"] is False and out["NONE"] is None
    assert out["OBJ"] == "OPAQUE-TEXT"
    unchanged = Opaque()
    assert rewrite("result", unchanged, same) is unchanged


def test_rewrite_changes_ints_only_when_their_text_changes():
    assert rewrite("result", 42, same) == 42
    assert rewrite("result", 42, lambda path, text: "[N]") == "[N]"


def test_very_deep_values_are_scanned_as_text():
    value: object = "leaf"
    for _ in range(MAX_DEPTH + 3):
        value = [value]
    found = texts("result", value)
    assert len(found) == 1 and "leaf" in found[0]
    assert "LEAF" in str(rewrite("result", value, upper))


def test_fields_carry_every_key_above_them():
    value = {"login": {"password": {"value": "pw"}}, "rows": [Row(email="e")], "box": Box("b")}
    found = fields("result", value)
    assert Field("pw", ("value", 0, "value", 0, "value", 0), ("login", "password", "value")) in (
        found
    )
    assert Field("e", ("value", 1, 0, "email"), ("rows", "email")) in found
    assert Field("login", ("key", 0)) in found  # a key is a text, with no keys of its own


def test_pairs_name_their_value():
    found = fields("args", {"headers": [["Authorization", "abc"], {"name": "X-Pin", "value": "1"}]})
    keyed = {field.text: field.keys for field in found}
    assert keyed["abc"] == ("headers", "Authorization")
    assert keyed["1"] == ("headers", "value", "X-Pin")
    assert keyed["Authorization"] == ("headers",)


def test_non_text_keys_keep_the_outer_key():
    assert fields("result", {"password": {7: "pw"}}) == [
        Field("password", ("key", 0)),
        Field("pw", ("value", 0, "value", 0), ("password",)),
    ]


def test_rewrite_is_by_place_not_by_text():
    value = {"token": "1", "count": "1", "n": 1}
    out = rewrite("result", value, lambda path, text: "[S]" if path == ("value", 0) else text)
    assert out == {"token": "[S]", "count": "1", "n": 1}


def test_rewritten_keys_never_collapse():
    value = {"a@x.example": "vip", "b@x.example": "gold", "[EMAIL]": "kept", 3: "n"}
    out = rewrite("result", value, lambda path, text: "[EMAIL]" if "@" in text else text)
    assert out == {"[EMAIL]#2": "vip", "[EMAIL]#3": "gold", "[EMAIL]": "kept", 3: "n"}


def raw_reply(text, args):
    call = ToolCall(name="send", arguments=args, id="c1")
    raw = ProviderRaw(
        provider="anthropic",
        items=[
            {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
            {"type": "text", "text": text},
        ],
    )
    message = Message(role="assistant", content=text, tool_calls=(call,), raw=raw)
    return llm_response(text).model_copy(update={"tool_calls": (call,), "message": message})


def test_model_texts_include_raw_items_and_tool_call_arguments():
    reply = raw_reply("hello", {"to": "bob"})
    expected = ["hello", "to", "bob", "type", "thinking", "plan", "signature", "sig-1", "text"]
    assert texts("llm_response", reply) == expected
    request = LLMRequest(messages=(reply.message,))
    assert texts("llm_request", request) == expected


def test_a_rewritten_reply_rewrites_raw_in_place_and_keeps_the_rest():
    def only_bob(path, text):
        return "[EMAIL]" if text == "bob" else text

    out = rewrite("llm_response", raw_reply("hi bob", {"to": "bob"}), only_bob)
    assert out.tool_calls[0].arguments == {"to": "[EMAIL]"}
    assert out.message.tool_calls[0].arguments == {"to": "[EMAIL]"}
    items = out.message.raw.items
    assert items[0] == {"type": "thinking", "thinking": "plan", "signature": "sig-1"}
    assert items[1] == {"type": "text", "text": "hi bob"}  # "hi bob" is not the text "bob"
    whole = rewrite("llm_response", raw_reply("hi bob", {}), upper)
    assert whole.message.raw.items[1] == {"TYPE": "TEXT", "TEXT": "HI BOB"}
    assert (whole.text, whole.message.content) == ("HI BOB", "HI BOB")
