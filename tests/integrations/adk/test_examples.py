"""The ADK examples run offline and print what the docs show."""

import pytest

pytest.importorskip("google.adk")

from tests.test_examples import _run

pytestmark = pytest.mark.contract


def test_the_tools_example_runs(capsys):
    assert _run("adk_tools.py", capsys).splitlines() == [
        "('shipping_rate', {'result': 5.0})",
        "('send_invoice', {'result': 'invoiced cus_1024 for 49.50 USD'})",
        "('send_invoice', {'error': 'This tool call is rejected.'})",
    ]


def test_the_flow_example_runs(capsys):
    assert _run("adk_flow.py", capsys).splitlines() == [
        "paused node:notice.send {'draft': 'reply for cus_1024'}",
        "done edited reply {'draft': 1, 'send': 1}",
    ]
