"""The tagged canonical encoding never confuses types and round-trips what it encodes."""

import datetime as dt
import enum
import uuid
from decimal import Decimal

import pytest
from pydantic import BaseModel, ConfigDict

from agent_adapter.exec._canon import (
    NotSealable,
    canonical,
    check_plain,
    decode,
    digest,
    encode,
    reachable_types,
    type_name,
)


class Color(enum.Enum):
    RED = "red"


class Level(enum.IntEnum):
    HIGH = 1


class Inner(BaseModel):
    n: int = 0


class Outer(BaseModel):
    model_config = ConfigDict(extra="allow")

    inner: Inner = Inner()
    tags: tuple[str, ...] = ()


VALUES = [
    None,
    True,
    1,
    1.0,
    float("nan"),
    float("-inf"),
    "text",
    "\udc80",
    b"\xff\x00",
    bytearray(b"ab"),
    [1, (2, 3)],
    (1, [2]),
    {3, 1, 2},
    frozenset({"a"}),
    {"plain": 1},
    {"$bytes": "looks tagged"},
    {1: "int key", "1": "str key"},
    dt.datetime(2026, 10, 29, 1, 2, 3, tzinfo=dt.UTC),
    dt.date(2026, 10, 29),
    dt.time(1, 2),
    dt.timedelta(days=1, seconds=2, microseconds=3),
    Decimal("1.10"),
    uuid.UUID(int=7),
    Color.RED,
    Level.HIGH,
    Outer(inner=Inner(n=2), tags=("x",), more=b"x"),
]


@pytest.mark.parametrize("value", VALUES, ids=repr)
def test_every_supported_value_round_trips(value):
    encoded = encode(value)
    check_plain(encoded)
    back = decode(encoded, known=reachable_types(Color, Level, Outer))
    if isinstance(value, float) and value != value:
        assert back != back
    else:
        assert back == value and type(back) is type(value)
    assert canonical(encode(back)) == canonical(encoded)


def test_values_that_json_would_merge_get_different_digests():
    groups = [
        [b"abc", "YWJj", "abc", bytearray(b"abc")],
        [1, 1.0, True, "1", Decimal(1)],
        [(1, 2), [1, 2], {1, 2}, frozenset({1, 2})],
        [{1: "a"}, {"1": "a"}],
        [Color.RED, "red"],
        [Level.HIGH, 1],
        [dt.date(2026, 1, 1), "2026-01-01"],
    ]
    for group in groups:
        digests = {digest(value) for value in group}
        assert len(digests) == len(group), group


def test_set_and_dict_order_do_not_change_the_encoding():
    assert digest({3, 1, 2}) == digest({1, 2, 3})
    assert digest({2: "b", 1: "a"}) == digest({1: "a", 2: "b"})
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})


def test_unknown_types_name_the_path_and_type():
    with pytest.raises(NotSealable, match=r"\$\.a\[1\] of type object"):
        encode({"a": [1, object()]})
    with pytest.raises(NotSealable, match="subclass"):
        encode({"k": type("subclass", (str,), {})("x")})  # exact types only


def test_an_encoder_replaces_custom_values():
    class Point:
        def __init__(self, x):
            self.x = x

    encoded = encode([Point(1)], lambda p: {"x": p.x})
    assert encoded == [{"$custom": type_name(Point), "value": {"x": 1}}]
    assert decode(encoded, lambda name, value: ("point", value)) == [("point", {"x": 1})]
    with pytest.raises(NotSealable, match="seal_decoder"):
        decode(encoded)
    with pytest.raises(NotSealable, match="raised ValueError"):
        encode(Point(1), lambda p: int("x"))
    with pytest.raises(NotSealable, match="same type"):
        encode(Point(1), lambda p: p)


def test_too_deep_values_are_not_sealable():
    deep: list = []
    for _ in range(5000):
        deep = [deep]
    with pytest.raises(NotSealable, match="nested too deeply"):
        encode(deep)


def test_decode_refuses_unknown_tags_and_classes():
    with pytest.raises(NotSealable, match="unknown tag"):
        decode({"$nope": 1})
    with pytest.raises(NotSealable, match="not allowed"):
        decode({"$enum": "no.such.module:Thing", "value": 1})
    with pytest.raises(NotSealable, match="not a BaseModel"):
        decode({"$model": "builtins:dict", "fields": {}}, known={"builtins:dict": dict})
    known = {"elsewhere:Inner": Inner}
    assert decode({"$model": "elsewhere:Inner", "fields": {"n": 4}}, known=known) == Inner(n=4)


def test_check_plain_refuses_anything_but_json():
    check_plain({"a": [1, 2.5, None, True, "x"]})
    for bad in ((1,), {1: "a"}, float("inf"), b"x"):
        with pytest.raises(NotSealable):
            check_plain({"v": bad})
