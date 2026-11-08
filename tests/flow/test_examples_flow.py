import runpy
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _run(name: str, capsys) -> list[str]:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
    return capsys.readouterr().out.splitlines()


def test_flow_research_pauses_resumes_and_dedupes_evidence(capsys):
    lines = _run("flow_research.py", capsys)
    assert lines == [
        "paused node:research.approve {'answer': 'returns answer from 3 sources'}",
        "done 8 {'classify': 1, 'retrieve': 2, 'merge': 2, 'reason': 2, 'approve': 1}",
        "['article:returns', 'ticket:returns', 'article:shipping']",
        "returns answer from 3 sources (approved)",
    ]


def test_first_flow_routes_each_ticket(capsys):
    lines = _run("first_flow.py", capsys)
    assert lines == [
        "done | Our refund policy: full refund within 30 days.",
        "{'classify': 1, 'policy': 1}",
        "other | A person will contact you today.",
    ]


def test_flow_subflows_pauses_inside_the_child_and_caches_the_quote(capsys):
    lines = _run("flow_subflows.py", capsys)
    assert lines == [
        "paused node:packing.label {'boxes': 2}",
        "done ['quoted mug', 'shipped 2 box(es) of mug for 16']",
        "paused node:packing.label {'boxes': 2}",
        "done ['quoted mug', 'shipped 2 box(es) of mug for 16']",
        "quote ran 1 time(s)",
    ]
