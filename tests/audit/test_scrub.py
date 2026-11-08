"""What may land in a record: scalars, masked secrets and PII, digests for identities."""

from __future__ import annotations

from agent_adapter.audit._scrub import MAX_ITEMS, MAX_TEXT, clean_data, clean_value


def digest(value: object) -> str:
    return f"d({value})"


def test_scalars_pass_and_other_values_drop() -> None:
    assert clean_value(None) is None
    assert clean_value(True) is True
    assert clean_value(3) == 3
    assert clean_value(1.5) == 1.5
    assert clean_value(float("inf")) is None
    assert clean_value({"a": 1}) is None
    assert clean_value(object()) is None


def test_strings_are_clipped_and_pii_masked() -> None:
    assert clean_value("x" * 500) == "x" * MAX_TEXT
    assert clean_value("mail ann@example.com") == "mail [EMAIL]"
    assert clean_value("key sk-abcdefghijklmnopqrstuvwxyz") == "key [SECRET]"


def test_names_keep_their_digits_but_never_secrets() -> None:
    assert clean_value("apr_1234567890abcdef1234", name=True) == "apr_1234567890abcdef1234"
    assert clean_value("sk-abcdefghijklmnopqrstuvwxyz", name=True) == "[SECRET]"
    assert clean_value("ann@example.com", name=True) == "[EMAIL]"


def test_lists_are_bounded_and_flat() -> None:
    assert clean_value(list(range(MAX_ITEMS + 10))) == list(range(MAX_ITEMS))
    assert clean_value([1, [2], {"a": 3}]) == [1, None]


def test_clean_data_drops_content_masks_secrets_and_digests_identities() -> None:
    data = {
        "tool": "refund",
        "arguments": {"card": "4111"},
        "result": "x",
        "api_key": "abc",
        "user_id": "ann",
        "scope": "user",
        "team": "front-desk",
    }
    out = clean_data(data, None, digest)
    assert out == {
        "tool": "refund",
        "api_key": "***",
        "user_id_digest": "d(ann)",
        "scope": "user",
        "team_digest": "d(front-desk)",
    }


def test_clean_data_keeps_only_listed_keys() -> None:
    assert clean_data({"tool": "t", "other": 1}, ("tool", "missing"), digest) == {"tool": "t"}
