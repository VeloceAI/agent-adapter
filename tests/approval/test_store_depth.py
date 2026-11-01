"""The approval JSON rule caps nesting depth instead of overflowing the stack."""

from typing import Any

import pytest

from agent_adapter.approval.store import MAX_JSON_DEPTH, check_json, request_digest
from agent_adapter.core.errors import ToolUsageError
from agent_adapter.core.types import ApprovalRequest


def nested(depth: int, *, kind: str = "list") -> Any:
    value: Any = 0
    for _ in range(depth):
        value = [value] if kind == "list" else {"k": value}
    return value


def req(arguments: dict[str, Any]) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="apr_1", run_id="run_1", call_id="c1", tool="pay", arguments=arguments
    )


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_check_json_rejects_deep_nesting_without_recursion_error(kind):
    with pytest.raises(ToolUsageError, match="nested too deeply"):
        check_json(nested(3000, kind=kind))


@pytest.mark.parametrize("kind", ["list", "dict"])
def test_check_json_accepts_nesting_up_to_the_limit(kind):
    check_json(nested(MAX_JSON_DEPTH, kind=kind))
    with pytest.raises(ToolUsageError, match="nested too deeply"):
        check_json(nested(MAX_JSON_DEPTH + 1, kind=kind))


def test_request_digest_rejects_deep_arguments():
    with pytest.raises(ToolUsageError, match="nested too deeply"):
        request_digest(req({"a": nested(3000)}))


def test_request_digest_accepts_arguments_at_the_limit():
    # The arguments object itself is one level.
    digest = request_digest(req({"a": nested(MAX_JSON_DEPTH - 1)}))
    assert digest == request_digest(req({"a": nested(MAX_JSON_DEPTH - 1)}))
    assert digest != request_digest(req({"a": nested(MAX_JSON_DEPTH - 2)}))
    with pytest.raises(ToolUsageError, match="nested too deeply"):
        request_digest(req({"a": nested(MAX_JSON_DEPTH)}))


def test_the_first_non_json_value_is_still_reported_with_its_path():
    with pytest.raises(ToolUsageError, match=r"not JSON at \$\.a\[1\]\.b: tuple"):
        check_json({"a": [1, {"b": (1,)}], "z": b"x"})
