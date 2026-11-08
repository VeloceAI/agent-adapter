import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "retrieval_rag.py"


def test_retrieval_example_answers_with_citations_per_scope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "city -> A ticket refund is possible up to 24 hours before the visit. "
        "[source: https://help.example.com/city/refunds]",
        "harbour -> Tickets bought online get no refund. "
        "[source: https://help.example.com/harbour/refunds]",
        "['city-refunds', 'city-audio-guide'] ['harbour-refunds']",
        "faq-7 Parking is free after 17:00.",
    ]


def test_concept_page_snippets_come_from_the_example() -> None:
    page = (ROOT / "docs" / "concepts" / "retrieval.md").read_text(encoding="utf-8")
    example = EXAMPLE.read_text(encoding="utf-8")
    for part in ("corpus", "tool", "loop", "callable"):
        assert f'--8<-- "examples/retrieval_rag.py:{part}"' in page
        assert f"# --8<-- [start:{part}]" in example
