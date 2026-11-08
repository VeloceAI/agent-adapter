"""A refused body is read and discarded, up to a limit, so the client gets the answer."""

import io

from agent_adapter.viewer._drain import DRAIN_LIMIT, drain


def test_drain_reads_the_whole_body() -> None:
    body = io.BytesIO(b"x" * 200_000)
    assert drain(body, 200_000)
    assert body.read() == b""


def test_drain_refuses_bodies_over_the_limit_without_reading() -> None:
    body = io.BytesIO(b"x" * 10)
    assert not drain(body, DRAIN_LIMIT + 1)
    assert body.tell() == 0


def test_drain_reports_a_body_that_ends_early() -> None:
    assert not drain(io.BytesIO(b"abc"), 10)
