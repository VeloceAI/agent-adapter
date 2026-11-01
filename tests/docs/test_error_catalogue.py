"""The error catalogue in docs/reference/errors.md matches `agent_adapter.errors`.

Regenerate the table after changing an error (keeps each row's Fix text):
    python -m tests.docs.test_error_catalogue
"""

import inspect
from pathlib import Path

from agent_adapter import errors
from agent_adapter.core import error_base, model_errors
from agent_adapter.core import errors as core_errors
from agent_adapter.server.problems import status_for

PAGE = Path(__file__).resolve().parents[2] / "docs" / "reference" / "errors.md"
START, END = "<!-- catalogue:start -->", "<!-- catalogue:end -->"
HEADER = [
    "| Error | Kind | Bases | Retryable | HTTP | Cause | Fix |",
    "|---|---|---|---|---|---|---|",
]
GENERATED = 6  # every column but Fix comes from the code


def error_classes() -> list[type[core_errors.AdapterError]]:
    """The classes in `agent_adapter.errors.__all__`, in definition order, module by module."""
    exported = {getattr(errors, name) for name in errors.__all__}
    found: list[type[core_errors.AdapterError]] = []
    for module in (error_base, core_errors, model_errors):
        for value in vars(module).values():
            if not (inspect.isclass(value) and issubclass(value, core_errors.AdapterError)):
                continue
            if module is core_errors and value in vars(model_errors).values():
                continue  # re-exported: listed with its own module
            if value in exported and value not in found:
                found.append(value)
    return found


def _http(cls: type[core_errors.AdapterError]) -> str:
    if cls is core_errors.ApprovalPending:
        return "202, not a problem"  # the server answers with the approval id
    return str(status_for(cls.kind))


def generated_cells(cls: type[core_errors.AdapterError]) -> list[str]:
    """The code-derived cells of one row."""
    bases = ", ".join(f"`{base.__name__}`" for base in cls.__bases__)
    cause = (cls.__doc__ or "").strip().splitlines()[0].replace("|", "\\|")
    retry = "yes" if cls.retryable else "no"
    return [f"`{cls.__name__}`", f"`{cls.kind}`", bases, retry, _http(cls), cause]


def _split(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split(" | ")]


def page_rows(text: str) -> list[list[str]]:
    """The catalogue rows of the page, as cells, without the header."""
    body = text.split(START, 1)[1].split(END, 1)[0].strip().splitlines()
    assert body[: len(HEADER)] == HEADER
    return [_split(row) for row in body[len(HEADER) :]]


def render(fixes: dict[str, str]) -> str:
    """The catalogue table, with `fixes` keyed by the error's name cell."""
    rows = []
    for cls in error_classes():
        cells = generated_cells(cls)
        rows.append("| " + " | ".join([*cells, fixes.get(cells[0], "TODO")]) + " |")
    return "\n".join([*HEADER, *rows])


def test_every_error_has_one_catalogue_row_in_definition_order() -> None:
    names = [row[0] for row in page_rows(PAGE.read_text(encoding="utf-8"))]
    expected = [f"`{cls.__name__}`" for cls in error_classes()]
    assert names == expected, "run `python -m tests.docs.test_error_catalogue`"


def test_catalogue_cells_match_the_code() -> None:
    rows = {row[0]: row for row in page_rows(PAGE.read_text(encoding="utf-8"))}
    for cls in error_classes():
        cells = generated_cells(cls)
        assert rows[cells[0]][:GENERATED] == cells


def test_every_row_says_how_to_fix_it() -> None:
    for row in page_rows(PAGE.read_text(encoding="utf-8")):
        assert len(row) == GENERATED + 1
        assert row[-1] and row[-1] != "TODO", row[0]


def test_catalogue_covers_every_exported_error() -> None:
    exported = {name for name in errors.__all__ if name != "ErrorInfo"}
    assert {cls.__name__ for cls in error_classes()} == exported


def main() -> None:
    """Rewrite the catalogue table from the code, keeping each row's Fix text."""
    text = PAGE.read_text(encoding="utf-8")
    fixes = {row[0]: row[-1] for row in page_rows(text) if len(row) > GENERATED}
    head, rest = text.split(START, 1)
    tail = rest.split(END, 1)[1]
    PAGE.write_text(f"{head}{START}\n{render(fixes)}\n{END}{tail}", encoding="utf-8")


if __name__ == "__main__":
    main()
