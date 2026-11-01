"""Uploads: sniffed types, size caps, safe names, and the download headers."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_adapter.viewer.chat import files as files_module
from agent_adapter.viewer.chat.files import FileStore, display_name, sniff
from agent_adapter.viewer.chat.records import FileRef, Refused
from agent_adapter.viewer.chat.sessions import SessionStore
from tests.viewer.chat.helpers import chat, chat_served, get, new_session
from tests.viewer.chat.test_builtin import PNG, upload
from tests.viewer.chat.test_contract import async_text
from tests.viewer.helpers import request


@pytest.mark.parametrize(
    ("data", "name", "media_type"),
    [
        (PNG, "x.txt", "image/png"),
        (b"\xff\xd8\xff\xe0rest", "a.jpg", "image/jpeg"),
        (b"GIF89a....", "a.gif", "image/gif"),
        (b"GIF87a....", "a.gif", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "a.webp", "image/webp"),
        (b"%PDF-1.7\n", "doc.pdf", "application/pdf"),
        (b'{"a": 1}', "data.txt", "application/json"),
        (b"a,b\n1,2\n", "t.csv", "text/csv"),
        (b"plain words", "notes", "text/plain"),
    ],
)
def test_the_type_comes_from_the_bytes(data: bytes, name: str, media_type: str) -> None:
    assert sniff(data, name) == media_type


@pytest.mark.parametrize("data", [b"MZ\x90\x00binary", b"\xff\xfe\x00bad utf8", b"\xc3\x28"])
def test_other_bytes_are_not_allowed(data: bytes) -> None:
    assert sniff(data, "a.txt") is None


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("../../etc/passwd", "passwd"),
        ("..\\..\\boot.ini", "boot.ini"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("  .hidden\x00name\x1f ", "hiddenname"),
        ("", "file"),
        ("..", "file"),
        ("x" * 300, "x" * 120),
    ],
)
def test_names_are_display_text_only(raw: str, shown: str) -> None:
    assert display_name(raw) == shown


def test_an_upload_is_stored_under_a_random_name_and_served_as_a_download(
    tmp_path: Path,
) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        status, body = upload(server, sid, "..%2F..%2Fsecret%20%C3%A9.txt", b"hello")
        assert status == 201
        assert body["name"] == "secret \u00e9.txt" and body["size"] == 5
        stored = list((tmp_path / "chats" / sid / "files").iterdir())
        assert [p.name for p in stored] == [body["attachment_id"]]
        status, text, headers = request(server, "GET", f"/api/chat/files/{body['attachment_id']}")
        assert status == 200 and text == "hello"
        assert headers["content-type"] == "text/plain; charset=utf-8"
        assert headers["content-disposition"] == "attachment; filename*=UTF-8''secret%20%C3%A9.txt"
        assert headers["x-content-type-options"] == "nosniff"
        detail = get(server, f"/api/chat/sessions/{sid}")[1]
        assert detail["files"][0]["id"] == body["attachment_id"]
        assert get(server, "/api/chat/files/fil_0000000000000000")[0] == 404
        assert get(server, "/api/chat/files/..%2F..%2Fx")[0] == 404


def test_uploads_are_refused_when_wrong(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files_module, "MAX_SESSION_FILES", 8)
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        assert upload(server, sid, "a.exe", b"MZ\x90\x00")[0] == 415
        assert upload(server, sid, "e.txt", b"")[0] == 400
        assert upload(server, "ses_0000000000000000", "a.txt", b"x")[0] == 404
        assert upload(server, sid, "a.txt", b"12345")[0] == 201
        status, body = upload(server, sid, "b.txt", b"67890")
        assert status == 413 and "session" in body["error"]
        headers = {
            "Origin": f"http://127.0.0.1:{server.port}",
            "X-AA-CSRF": server.csrf,
            "Content-Type": "application/json",
        }
        path = f"/api/chat/files?session={sid}"
        assert request(server, "POST", path, body=b"x", headers=headers)[0] == 403


def test_a_file_over_the_cap_is_refused_before_it_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_adapter.viewer.chat import routes

    monkeypatch.setattr(routes, "MAX_FILE", 4)
    with chat_served(tmp_path, agent=async_text) as server:
        status, body = upload(server, new_session(server), "big.txt", b"12345")
    assert status == 413 and "4 bytes" in body["error"]


def test_the_store_refuses_files_too_big_and_paths_outside_the_folder(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "chats")
    files = FileStore(store)
    session = store.create()
    with pytest.raises(Refused) as too_big:
        files.add(session, "a.txt", b"x" * (files_module.MAX_FILE + 1))
    assert too_big.value.status == 413
    outside = FileRef("fil_1", "x", "text/plain", 1, "../../outside.txt")
    (tmp_path / "outside.txt").write_text("x")
    with pytest.raises(Refused):
        files.path(outside)


def test_the_upload_counts_back_when_the_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SessionStore(tmp_path / "chats")
    files = FileStore(store)
    session = store.create()

    def broken(self: Path, data: bytes) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_bytes", broken)
    with pytest.raises(OSError):
        files.add(session, "a.txt", b"abc")
    assert session.uploaded_bytes == 0 and session.files == {}


def test_a_stored_file_that_vanished_is_404(tmp_path: Path) -> None:
    with chat_served(tmp_path, agent=async_text) as server:
        sid = new_session(server)
        file_id = upload(server, sid, "a.txt", b"abc")[1]["attachment_id"]
        (tmp_path / "chats" / sid / "files" / file_id).unlink()
        assert get(server, f"/api/chat/files/{file_id}")[0] == 404
        assert chat(server).store.find_file(file_id).id == file_id
