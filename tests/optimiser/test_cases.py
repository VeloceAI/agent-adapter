import json

import pytest

from agent_adapter.core.errors import ConfigError
from agent_adapter.optimiser import EvalCase, load_cases


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_jsonl_skips_blank_lines_and_keeps_non_text_input(tmp_path):
    rows = [
        {"id": "a", "input": "rent", "expected": "10(13A)", "tags": ["x", "y"]},
        {"id": 2, "input": {"topic": "tuition"}, "tags": "p; q,r", "extra": 1},
    ]
    path = write(tmp_path, "c.jsonl", "\n".join(json.dumps(r) for r in rows[:1]) + "\n\n")
    path.write_text(path.read_text() + json.dumps(rows[1]) + "\n", encoding="utf-8")
    assert load_cases(path) == (
        EvalCase(id="a", input="rent", expected="10(13A)", tags=("x", "y")),
        EvalCase(id="2", input={"topic": "tuition"}, tags=("p", "q", "r")),
    )


def test_json_list_and_custom_field_names(tmp_path):
    path = write(tmp_path, "c.json", json.dumps([{"key": "k1", "q": "tuition", "a": "80C"}]))
    cases = load_cases(path, id_field="key", input_field="q", expected_field="a")
    assert cases == (EvalCase(id="k1", input="tuition", expected="80C"),)


def test_csv_with_bom_quoted_newlines_and_empty_expected(tmp_path):
    text = '﻿id,input,expected,tags\nt,"tuition\nfees",80C,a;b\nr,rent,,\n'
    path = tmp_path / "c.csv"
    path.write_text(text, encoding="utf-8")
    assert load_cases(path) == (
        EvalCase(id="t", input="tuition\nfees", expected="80C", tags=("a", "b")),
        EvalCase(id="r", input="rent"),
    )


@pytest.mark.parametrize(
    ("name", "text", "message"),
    [
        ("c.jsonl", '{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}\n', "duplicate"),
        ("c.jsonl", '{"id": "a"}\n', "c.jsonl:1: missing field 'input'"),
        ("c.jsonl", '{"input": "x"}\n', "missing field 'id'"),
        ("c.jsonl", "{not json\n", "c.jsonl:1: not valid JSON"),
        ("c.jsonl", "[1]\n", "must be a JSON object"),
        ("c.jsonl", '{"id": true, "input": "x"}\n', "text or a whole number"),
        ("c.jsonl", '{"id": "a", "input": "x", "tags": [1]}\n', "invalid case"),
        ("c.jsonl", "\n", "no cases"),
        ("c.json", '{"id": "a"}', "must hold a list"),
        ("c.csv", "id,input\na,\n", "c.csv:2: missing field 'input'"),
        ("c.txt", "x", "use a .jsonl"),
    ],
)
def test_bad_files_raise_config_error(tmp_path, name, text, message):
    with pytest.raises(ConfigError, match=message.replace("(", r"\(")):
        load_cases(write(tmp_path, name, text))


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_cases(tmp_path / "none.jsonl")


def test_jsonl_keeps_unicode_line_separators_inside_strings(tmp_path):
    rows = [{"id": "u", "input": "a\u2028b\u2029c\x85d\x0ce"}, {"id": "w", "input": "crlf"}]
    text = json.dumps(rows[0], ensure_ascii=False) + "\r\n" + json.dumps(rows[1]) + "\r\n"
    path = tmp_path / "c.jsonl"
    path.write_bytes(text.encode("utf-8"))
    assert load_cases(path) == (
        EvalCase(id="u", input="a\u2028b\u2029c\x85d\x0ce"),
        EvalCase(id="w", input="crlf"),
    )


def test_an_empty_expected_in_json_stays_an_empty_string(tmp_path):
    path = write(tmp_path, "c.json", json.dumps([{"id": "e", "input": "x", "expected": ""}]))
    assert load_cases(path) == (EvalCase(id="e", input="x", expected=""),)


def test_duplicate_ids_are_listed_once_each(tmp_path):
    rows = [{"id": i, "input": "x"} for i in ("b", "a", "b", "a", "c")]
    path = write(tmp_path, "c.json", json.dumps(rows))
    with pytest.raises(ConfigError, match=r"duplicate case ids: \['a', 'b'\]"):
        load_cases(path)
