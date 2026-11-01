"""Canonical encoding edges: huge ints, JSON failures, time zones, the class allow-list."""

import datetime as dt
import enum
from importlib.resources import files
from typing import Annotated, Any, Optional, Union
from zoneinfo import ZoneInfo

import pytest
import tzdata
from pydantic import BaseModel

from agent_adapter.exec._canon import (
    NotSealable,
    canonical,
    check_plain,
    decode,
    digest,
    encode,
    reachable_types,
    tagged_classes,
    type_name,
)

HUGE = 10**5000
NEW_YORK = ZoneInfo("America/New_York")


class Kind(enum.Enum):
    A = "a"


class Leaf(BaseModel):
    kind: Kind = Kind.A


class Branch(BaseModel):
    leaves: list[Leaf] = []
    by_name: dict[str, Optional["Branch"]] = {}
    pair: tuple[Leaf, int] | None = None
    either: Union[Leaf, str] = ""  # noqa: UP007 - the typing form must be followed too
    marked: Annotated[Leaf | None, "meta"] = None
    anything: Any = None


def test_huge_ints_round_trip_and_ordinary_ints_stay_plain():
    for value in (HUGE, -HUGE, 2**8000, -(2**8000) + 1):
        encoded = encode({"n": value})
        check_plain(encoded)
        canonical(encoded)  # no int-to-text limit applies to the tagged form
        assert decode(encoded) == {"n": value}
        assert digest(value) != digest(value + 1)
    assert encode([0, -7, 2**64]) == [0, -7, 2**64]


def test_an_int_beyond_one_mebibyte_of_hex_is_not_sealable():
    with pytest.raises(NotSealable, match=r"\$\.n of type int"):
        encode({"n": 1 << (4 * 1024 * 1024 + 4)})


def test_json_failures_become_not_sealable():
    with pytest.raises(NotSealable):
        canonical({"raw": HUGE})  # not encoded first: json.dumps refuses the int
    deep: list = []
    for _ in range(100_000):
        deep = [deep]
    with pytest.raises(NotSealable):
        canonical(deep)


MOMENTS = [
    dt.datetime(2026, 3, 1, 12, 0),
    dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.UTC),
    dt.datetime(2026, 3, 1, 12, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30))),
    dt.datetime(2026, 3, 1, tzinfo=dt.timezone(dt.timedelta(hours=-3), "BRT")),
    dt.datetime(2026, 3, 1, tzinfo=dt.timezone(dt.timedelta(seconds=1, microseconds=5))),
    dt.datetime(2026, 7, 1, 9, 0, tzinfo=NEW_YORK),
    dt.datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK, fold=1),
    dt.time(1, 30, fold=1),
    dt.time(1, 30, tzinfo=dt.timezone(dt.timedelta(hours=2))),
]


def test_moments_keep_zone_offset_name_and_fold():
    for value in MOMENTS:
        back = decode(encode(value))
        assert back == value and type(back) is type(value), value
        assert back.fold == value.fold and back.tzname() == value.tzname()
        assert type(back.tzinfo) is type(value.tzinfo)
        if isinstance(value.tzinfo, ZoneInfo):
            assert back.tzinfo is value.tzinfo
        assert back.utcoffset() == value.utcoffset()


def test_fold_and_zone_change_the_encoding():
    first = dt.datetime(2026, 11, 1, 1, 30, tzinfo=NEW_YORK)
    assert digest(first) != digest(first.replace(fold=1))
    summer = dt.datetime(2026, 7, 1, 9, 0, tzinfo=NEW_YORK)
    fixed = summer.astimezone(dt.timezone(dt.timedelta(hours=-4)))
    assert summer == fixed and digest(summer) != digest(fixed)


def test_other_tzinfo_types_are_not_sealable():
    class Custom(dt.tzinfo):
        def utcoffset(self, when):
            return dt.timedelta(0)

    with pytest.raises(NotSealable, match="tzinfo"):
        encode(dt.datetime(2026, 1, 1, tzinfo=Custom()))
    with open(_zone_file(), "rb") as handle:
        keyless = ZoneInfo.from_file(handle)
    with pytest.raises(NotSealable, match="key"):
        encode(dt.datetime(2026, 1, 1, tzinfo=keyless))


def _zone_file():
    return str(files(tzdata).joinpath("zoneinfo", "UTC"))


def test_reachable_types_follow_annotations_recursively():
    names = set(reachable_types(Branch))
    assert names == {type_name(Branch), type_name(Leaf), type_name(Kind)}
    assert reachable_types(None) == {} and reachable_types(Any) == {}
    assert set(reachable_types(list[Kind])) == {type_name(Kind)}


def test_decode_builds_only_allowed_classes():
    value = Branch(leaves=[Leaf()], by_name={"x": None})
    encoded = encode(value)
    assert tagged_classes(encoded) == {type_name(Branch), type_name(Leaf), type_name(Kind)}
    assert decode(encoded, known=reachable_types(Branch)) == value
    with pytest.raises(NotSealable, match="not allowed"):
        decode(encoded)
    with pytest.raises(NotSealable, match="not allowed"):
        decode(encoded, known=reachable_types(Leaf))


def test_a_decoder_handles_classes_outside_the_allow_list_or_refuses():
    seen = []

    def decoder(name, value):
        seen.append(name)
        return Kind(value) if name == type_name(Kind) else Leaf(**value)

    assert decode(encode(Leaf()), decoder) == Leaf()
    assert seen == [type_name(Kind), type_name(Leaf)]
    with pytest.raises(NotSealable, match="seal_decoder refused"):
        decode(encode(Kind.A), lambda name, value: int("x"))


def test_malformed_tagged_ints_and_moments_are_refused():
    with pytest.raises(NotSealable, match="hex text"):
        decode({"$int": 12})
    with pytest.raises(NotSealable, match="malformed"):
        decode({"$datetime": ["2026-01-01T00:00:00", None, 2]})
