import math

import pytest

from agent_adapter.core.errors import ConfigError, ToolUsageError
from agent_adapter.sandbox import Limits
from agent_adapter.sandbox._checks import (
    MAX_CODE_BYTES,
    MAX_ENV,
    MAX_INPUT_BYTES,
    MAX_INPUT_FILES,
    check_code,
    check_env,
    check_files,
    check_language,
    check_name,
    check_timeout,
    is_safe_name,
    positive_int,
)


@pytest.mark.parametrize("name", ["a.txt", "data/in.csv", "x-y_z.1", "a" * 255])
def test_check_name_accepts_plain_relative_names(name: str) -> None:
    assert check_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/etc/passwd",
        "../up",
        "a/../b",
        "./a",
        "a//b",
        "a/",
        "C:x",
        "a\b",
        "a b",
        "é",
        3,
        "a" * 256,
        "..",
    ],
)
def test_check_name_refuses_unsafe_names(name: object) -> None:
    with pytest.raises(ToolUsageError):
        check_name(name)


def test_check_files_encodes_str_and_keeps_bytes() -> None:
    assert check_files({"a.txt": "é", "b.bin": b"\x00"}, "main.py") == {
        "a.txt": "é".encode(),
        "b.bin": b"\x00",
    }
    assert check_files(None, "main.py") == {}


@pytest.mark.parametrize(
    ("files", "message"),
    [
        ({"main.py": "x"}, "overwrite the code"),
        ({"a": 3}, "bytes or str"),
        ({"a": "x", "a/b": "y"}, "folder"),
        ({f"f{i}": "" for i in range(MAX_INPUT_FILES + 1)}, "at most"),
        ({"big": b"x" * (MAX_INPUT_BYTES + 1)}, "larger"),
        (["a"], "mapping"),
    ],
)
def test_check_files_refuses_bad_inputs(files: object, message: str) -> None:
    with pytest.raises(ToolUsageError, match=message):
        check_files(files, "main.py")  # type: ignore[arg-type]


def test_check_env_accepts_plain_names_and_refuses_the_rest() -> None:
    assert check_env({"A_1": "x"}) == {"A_1": "x"}
    assert check_env(None) == {}
    for bad in ({"1A": "x"}, {"A-B": "x"}, {"A": 1}, {"A": "a\x00"}, {3: "x"}, ["A"]):
        with pytest.raises(ToolUsageError):
            check_env(bad)  # type: ignore[arg-type]
    with pytest.raises(ToolUsageError, match="at most"):
        check_env({f"K{i}": "" for i in range(MAX_ENV + 1)})


@pytest.mark.parametrize("code", ["", "   ", 3, "a\x00b", "x" * (MAX_CODE_BYTES + 1)])
def test_check_code_refuses_blank_nul_long_and_non_str(code: object) -> None:
    with pytest.raises(ToolUsageError):
        check_code(code)


def test_check_code_counts_bytes_not_characters() -> None:
    assert check_code("x" * MAX_CODE_BYTES)
    with pytest.raises(ToolUsageError):
        check_code("é" * (MAX_CODE_BYTES // 2 + 1))


@pytest.mark.parametrize("value", [0, -1, math.inf, math.nan, 601, True, "5", None])
def test_check_timeout_refuses_out_of_range(value: object) -> None:
    with pytest.raises(ConfigError):
        check_timeout(value)


def test_check_timeout_and_language_accept_good_values() -> None:
    assert check_timeout(1) == 1.0
    assert check_timeout(600) == 600.0
    assert check_language("python", ("python",)) == "python"
    with pytest.raises(ToolUsageError, match="one of"):
        check_language("ruby", ("python",))


def test_positive_int_bounds() -> None:
    assert positive_int("n", 5, 1, 5) == 5
    for bad in (0, 6, True, 1.0):
        with pytest.raises(ConfigError):
            positive_int("n", bad, 1, 5)


def test_limits_refuse_negative_and_non_int() -> None:
    assert Limits(max_files=0, max_file_bytes=0).max_files == 0
    with pytest.raises(ConfigError):
        Limits(max_files=-1)
    with pytest.raises(ConfigError):
        Limits(max_file_bytes=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "name", ["CON", "nul.txt", "a/COM1.log", "lpt9", "aux.tar.gz", "a.", "d./x"]
)
def test_check_name_refuses_device_names_and_trailing_dots(name: str) -> None:
    assert not is_safe_name(name)
    with pytest.raises(ToolUsageError):
        check_name(name)


def test_is_safe_name_accepts_lookalikes() -> None:
    assert is_safe_name("console.txt") and is_safe_name("com10") and is_safe_name("a.b")
