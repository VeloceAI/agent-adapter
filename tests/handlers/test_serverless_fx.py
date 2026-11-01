"""tool_handler review fixes: one shared loop, stable ids for Lambda direct-invoke retries."""

from agent_adapter.exec import _sugar
from agent_adapter.handlers import tool_handler

FILING = {"tool": "file_return", "arguments": {"pan": "P", "year": 2026}}


class LambdaContext:
    """A Lambda context stand-in with a request id (stable across async retries)."""

    aws_request_id = "req-1"

    def get_remaining_time_in_millis(self) -> int:
        return 30_000


def test_a_direct_invoke_retry_parks_the_same_approval(executor, store):
    handler = tool_handler(executor, timeout=5)
    first = handler(FILING, LambdaContext())
    again = handler(FILING, LambdaContext())  # Lambda's async retry: the same request id
    assert first["status"] == again["status"] == "pending_approval"
    assert first["approval_id"] == again["approval_id"]
    assert first["call_id"] == again["call_id"]


def test_a_direct_invoke_without_a_request_id_gets_fresh_ids(executor):
    handler = tool_handler(executor, timeout=5)
    assert handler(FILING)["call_id"] != handler(FILING)["call_id"]


def test_handlers_share_the_sync_loop_of_the_executor(executor):
    handler = tool_handler(executor, timeout=5)
    assert handler({"tool": "echo", "arguments": {"text": "a"}})["status"] == "ok"
    runner = _sugar.shared_runner()
    assert runner.running and runner is _sugar.shared_runner()
