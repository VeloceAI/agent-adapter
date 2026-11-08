"""Load eval cases from a JSONL, JSON or CSV file."""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from agent_adapter.core.errors import ConfigError
from agent_adapter.evals.model import EvalCase

logger = logging.getLogger(__name__)

_TAG_SPLIT = re.compile(r"[;,]")


def load_cases(
    path: str | os.PathLike[str],
    *,
    id_field: str = "id",
    input_field: str = "input",
    expected_field: str = "expected",
    tags_field: str = "tags",
) -> tuple[EvalCase, ...]:
    """Cases from JSONL (one object per line), JSON (a list of objects) or CSV (a header row).

    `id` and `input` are required; `expected` and `tags` are optional. Tags are a list, or a
    string split on `,` or `;`. CSV cells are text; an empty `expected` cell means none.
    Other fields are ignored.

    Raises:
        ConfigError: unreadable file, unknown suffix, bad JSON, a missing field, duplicate ids,
            or no cases.
    """
    source = Path(path)
    fields = (id_field, input_field, expected_field, tags_field)
    is_csv = source.suffix.lower() == ".csv"
    cases = tuple(_case(row, where, fields, is_csv) for where, row in _rows(source))
    if not cases:
        raise ConfigError(f"{source}: no cases")
    duplicates = sorted(i for i, n in Counter(case.id for case in cases).items() if n > 1)
    if duplicates:
        raise ConfigError(f"{source}: duplicate case ids: {duplicates}")
    logger.info("loaded %d eval cases from %s", len(cases), source.name)
    return cases


def _rows(source: Path) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """(location, row) pairs; the location names the file and line or item for errors."""
    suffix = source.suffix.lower()
    if suffix not in (".jsonl", ".json", ".csv"):
        raise ConfigError(f"{source}: use a .jsonl, .json or .csv file")
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{source}: cannot read it ({exc})") from exc
    if suffix == ".csv":
        reader = csv.DictReader(io.StringIO(text, newline=""))
        for row in reader:
            yield f"{source}:{reader.line_num}", row
        return
    if suffix == ".json":
        items = _json(text, str(source))
        if not isinstance(items, list):
            raise ConfigError(f"{source}: a .json file must hold a list of case objects")
        for index, item in enumerate(items):
            yield f"{source}[{index}]", _object(item, f"{source}[{index}]")
        return
    # JSON may hold U+2028 and other separators raw; only "\n" ends a JSONL record.
    for number, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r")
        if line.strip():
            where = f"{source}:{number}"
            yield where, _object(_json(line, where), where)


def _json(text: str, where: str) -> Any:
    try:
        return json.loads(text)
    except ValueError as exc:
        raise ConfigError(f"{where}: not valid JSON ({exc})") from exc


def _object(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: a case must be a JSON object")
    return value


def _case(
    row: Mapping[str, Any], where: str, fields: tuple[str, str, str, str], is_csv: bool
) -> EvalCase:
    id_field, input_field, expected_field, tags_field = fields
    for name in (id_field, input_field):
        if row.get(name) in (None, ""):
            raise ConfigError(f"{where}: missing field {name!r}")
    case_id = row[id_field]
    if isinstance(case_id, bool) or not isinstance(case_id, str | int):
        raise ConfigError(f"{where}: {id_field!r} must be text or a whole number")
    expected = row.get(expected_field)
    tags = row.get(tags_field) or ()
    if isinstance(tags, str):
        tags = tuple(t.strip() for t in _TAG_SPLIT.split(tags) if t.strip())
    try:
        return EvalCase(
            id=str(case_id),
            input=row[input_field],
            expected=None if is_csv and expected == "" else expected,
            tags=tags,
        )
    except ValidationError as exc:
        problems = "; ".join(e["msg"] for e in exc.errors(include_url=False, include_input=False))
        raise ConfigError(f"{where}: invalid case ({problems})") from exc
