import pytest

from agent_adapter.approval import _records, persist, store
from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest


def _request(approval_id: str = "", created_at: float = 0.0) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=approval_id, run_id="r", call_id="c", tool="t", created_at=created_at
    )


def test_filled_sets_an_empty_id_and_created_at():
    request = _records.filled(_request())
    assert request.approval_id.startswith("apr_") and len(request.approval_id) == 36
    assert request.created_at > 0


def test_filled_returns_a_complete_request_unchanged():
    request = _request(approval_id="apr_x", created_at=5.0)
    assert _records.filled(request) is request


def test_stamped_fills_only_an_empty_time():
    assert _records.stamped(ApprovalDecision(approved=True)).decided_at > 0
    decided = ApprovalDecision(approved=False, decided_at=7.0)
    assert _records.stamped(decided) is decided


def test_stamped_rejects_edits_that_are_not_plain_json():
    with pytest.raises(ToolUsageError, match="not JSON"):
        _records.stamped(ApprovalDecision(approved=True, edited_args={"a": (1,)}))
    edited = ApprovalDecision(approved=True, edited_args={"a": [1]}, decided_at=3.0)
    assert _records.stamped(edited) is edited


@pytest.mark.parametrize(("status", "limit"), [(None, 0), ("pending", 1), ("denied", 100)])
def test_check_list_accepts_known_statuses_and_limits(status, limit):
    _records.check_list(status, limit)


@pytest.mark.parametrize(
    ("status", "limit", "match"),
    [("open", 1, "status must be one of"), (None, -1, "limit must be >= 0")],
)
def test_check_list_rejects_bad_arguments(status, limit, match):
    with pytest.raises(ConfigError, match=match):
        _records.check_list(status, limit)


def test_every_store_shares_one_copy_of_the_record_rules():
    assert persist.filled is _records.filled
    assert persist.stamped is _records.stamped
    assert persist.check_list is _records.check_list
    assert store.check_json is _records.check_json
    assert store.STATUSES is _records.STATUSES
