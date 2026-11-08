"""verify: every kind of tampering shows, and a clean log passes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from agent_adapter.audit import FileAuditStore, verify
from agent_adapter.audit._files import HEAD, segments
from agent_adapter.errors import ConfigError
from tests.audit.helpers import KEY, entry, write_log

Rows = list[bytes]


def two_segments(folder: Path) -> list[Path]:
    write_log(folder, 2)
    store = FileAuditStore(folder, key=KEY, fsync=False)
    store.write(entry())
    store.close()
    return [path for _, path in segments(folder)]


def rewrite(path: Path, change: Callable[[Rows], Rows]) -> None:
    rows = change(path.read_bytes().split(b"\n")[:-1])
    path.write_bytes(b"".join(row + b"\n" for row in rows))


def test_a_clean_log_verifies(tmp_path: Path) -> None:
    two_segments(tmp_path)
    result = verify(tmp_path, KEY)
    assert result.ok and result.segments == 2 and result.events == 3 and result.problems == ()


def test_an_edited_value_shows(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: [rows[0], rows[1].replace(b"refund", b"refunx"), *rows[2:]])
    assert any("MAC" in p for p in verify(tmp_path, KEY).problems)


def test_a_reformatted_record_shows(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: [rows[0], json.dumps(json.loads(rows[1])).encode(), *rows[2:]])
    assert any("not a signed record" in p for p in verify(tmp_path, KEY).problems)


def test_a_deleted_record_shows(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: [rows[0], *rows[2:]])
    problems = verify(tmp_path, KEY).problems
    assert any("missing or reordered" in p for p in problems)
    assert any("chain broken" in p for p in problems)


def test_reordered_records_show(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: [rows[0], rows[2], rows[1], *rows[3:]])
    assert any("missing or reordered" in p for p in verify(tmp_path, KEY).problems)


def test_a_cut_tail_shows(tmp_path: Path) -> None:
    _, last = two_segments(tmp_path)
    rewrite(last, lambda rows: rows[:-1])
    assert any("audit.head says" in p for p in verify(tmp_path, KEY).problems)


def test_a_partial_last_line_shows_as_a_cut(tmp_path: Path) -> None:
    _, last = two_segments(tmp_path)
    last.write_bytes(last.read_bytes()[:-10])
    assert any("(cut)" in p for p in verify(tmp_path, KEY).problems)


def test_a_partial_line_in_an_earlier_segment_shows(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    first.write_bytes(first.read_bytes() + b'{"kind":"ev')
    assert any("partial record" in p for p in verify(tmp_path, KEY).problems)


@pytest.mark.parametrize("which", [0, 1])
def test_a_removed_segment_shows(tmp_path: Path, which: int) -> None:
    paths = two_segments(tmp_path)
    paths[which].unlink()
    assert not verify(tmp_path, KEY).ok


def test_swapped_segment_contents_show(tmp_path: Path) -> None:
    first, last = two_segments(tmp_path)
    a, b = first.read_bytes(), last.read_bytes()
    first.write_bytes(b)
    last.write_bytes(a)
    assert any("own open record" in p for p in verify(tmp_path, KEY).problems)


def test_an_unsealed_segment_before_a_later_one_shows(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: rows[:-1])
    assert any("not sealed" in p for p in verify(tmp_path, KEY).problems)


def test_a_record_after_the_seal_and_a_second_open_show(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: [*rows, rows[1], rows[0]])
    problems = verify(tmp_path, KEY).problems
    assert any("after the segment's seal" in p for p in problems)
    assert any("second open" in p for p in problems)


def test_a_seal_with_the_wrong_count_shows(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: [rows[0], rows[1], rows[-1]])
    assert any("seal does not match" in p for p in verify(tmp_path, KEY).problems)


def test_an_empty_segment_shows(tmp_path: Path) -> None:
    _, last = two_segments(tmp_path)
    last.write_bytes(b"")
    assert any("empty segment" in p for p in verify(tmp_path, KEY).problems)


def test_a_missing_or_edited_head_shows(tmp_path: Path) -> None:
    two_segments(tmp_path)
    head = tmp_path / HEAD
    head.write_bytes(head.read_bytes().replace(b'"sealed":true', b'"sealed":false'))
    assert any("does not match this key" in p for p in verify(tmp_path, KEY).problems)
    head.unlink()
    assert any("audit.head is missing" in p for p in verify(tmp_path, KEY).problems)


def test_a_malformed_or_unreadable_head_shows(tmp_path: Path) -> None:
    write_log(tmp_path, 1)
    head = tmp_path / HEAD
    head.write_bytes(b"{")
    assert any("unreadable" in p for p in verify(tmp_path, KEY).problems)
    from agent_adapter.audit._chain import mac

    body = {"v": 1, "seq": "x"}
    head.write_text(json.dumps({**body, "mac": mac(KEY.encode(), body)}))
    assert any("malformed" in p for p in verify(tmp_path, KEY).problems)


def test_an_empty_folder_or_a_head_alone_is_not_intact(tmp_path: Path) -> None:
    assert any("no audit segments" in p for p in verify(tmp_path, KEY).problems)
    write_log(tmp_path / "x", 1)
    for _, path in segments(tmp_path / "x"):
        path.unlink()
    assert any("there are none" in p for p in verify(tmp_path / "x", KEY).problems)


def test_garbage_lines_show(tmp_path: Path) -> None:
    first, _ = two_segments(tmp_path)
    rewrite(first, lambda rows: [rows[0], b"not json", b"[1]", *rows[1:]])
    assert any("not a signed record" in p for p in verify(tmp_path, KEY).problems)


def test_the_wrong_key_fails(tmp_path: Path) -> None:
    write_log(tmp_path, 1)
    result = verify(tmp_path, "the-wrong-key-0123456789")
    assert not result.ok and "NOT intact" in result.summary()


def test_verify_needs_a_folder_and_a_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigError, match="not an audit folder"):
        verify(tmp_path / "missing", KEY)
    monkeypatch.delenv("AA_AUDIT_KEY", raising=False)
    with pytest.raises(ConfigError):
        verify(tmp_path)
