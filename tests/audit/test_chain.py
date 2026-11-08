"""Keys, canonical form, MACs and the chain."""

from __future__ import annotations

import pytest

from agent_adapter.audit._chain import (
    GENESIS,
    KEY_ENV,
    Chain,
    canonical,
    chain_hash,
    check_mac,
    key_id,
    keyed_digest,
    resolve_key,
)
from agent_adapter.errors import ConfigError
from tests.audit.helpers import KEY


def test_resolve_key_prefers_the_argument_then_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, "e" * 20)
    assert resolve_key(KEY) == KEY.encode()
    assert resolve_key(b"b" * 16) == b"b" * 16
    assert resolve_key(None) == b"e" * 20


@pytest.mark.parametrize("bad", ["", "short", b"x" * 15])
def test_resolve_key_refuses_a_missing_or_short_key(
    bad: str | bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(ConfigError):
        resolve_key(bad)


def test_resolve_key_without_any_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(ConfigError, match=KEY_ENV):
        resolve_key(None)


def test_canonical_is_sorted_compact_and_refuses_nan() -> None:
    assert canonical({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode()
    with pytest.raises(ValueError):
        canonical({"x": float("nan")})


def test_chain_signs_in_order_and_links_each_record() -> None:
    chain = Chain(KEY.encode())
    first, line1 = chain.sign({"kind": "event"})
    assert (first["seq"], first["prev"]) == (0, GENESIS)
    chain.advance(line1)
    second, _ = chain.sign({"kind": "event"})
    assert (second["seq"], second["prev"]) == (1, chain_hash(line1))
    assert check_mac(KEY.encode(), second)
    assert not check_mac(b"another-key-0123456789", second)
    assert not check_mac(KEY.encode(), {**second, "kind": "seal"})
    assert not check_mac(KEY.encode(), {"seq": 1})


def test_digests_and_key_ids_depend_on_the_key() -> None:
    one, two = KEY.encode(), b"another-key-0123456789"
    assert keyed_digest(one, {"a": 1}) == keyed_digest(one, {"a": 1})
    assert keyed_digest(one, {"a": 1}) != keyed_digest(two, {"a": 1})
    assert len(key_id(one)) == 16 and key_id(one) != key_id(two)
