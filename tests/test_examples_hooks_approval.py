"""Runs the hooks and approval examples (kept apart from test_examples.py: one owner per file)."""

from tests.test_examples import _run


def test_hooks_redact_cap_spend_block_and_short_circuit(capsys):
    lines = _run("hooks.py", capsys).splitlines()
    assert lines == [
        "{'query': 'returns', 'api_key': '***', 'headers': {'Authorization': '***'}}",
        "thread 0.5 0.6",
        "args looks like a password",
        "ShortCircuit(value=['returns: free within 30 days'])",
    ]


def test_approval_rule_ports_and_deferred_round_trip(capsys):
    lines = _run("approval.py", capsys).splitlines()
    assert lines == [
        "20 False",
        "500 True",
        "True",
        "read-only mode",
        "pending",
        "True {'amount': 450} True",
    ]
