import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_embeddings_example_batches_caches_bills_and_reports_bad_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    runpy.run_path(str(ROOT / "examples" / "embeddings.py"), run_name="__main__")
    assert capsys.readouterr().out.splitlines() == [
        "3 (11.0, 2.0, 2.0) 6 0",
        "1 2",
        "api embed letters-v1 6 0.00000012 False",
        "api embed letters-v1 2 0.00000004 False",
        "model-output-invalid callable broken: fn returned 1 vectors for 2 texts",
    ]


def test_concept_page_snippets_come_from_the_example() -> None:
    page = (ROOT / "docs" / "concepts" / "embeddings.md").read_text(encoding="utf-8")
    example = (ROOT / "examples" / "embeddings.py").read_text(encoding="utf-8")
    for part in ("fn", "wire", "errors"):
        assert f'--8<-- "examples/embeddings.py:{part}"' in page
        assert f"# --8<-- [start:{part}]" in example
