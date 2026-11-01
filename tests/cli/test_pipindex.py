"""pip index checks: which index pip uses, extra indexes warn, never a credential in output."""

from __future__ import annotations

from pathlib import Path

from agent_adapter.cli.pipindex import config_files, index_findings, read_settings, redact

INTERNAL = "https://user:s3cret@pkgs.example.com/simple?token=abc"


def _ini(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_redact_drops_credentials_query_and_fragment():
    assert redact(INTERNAL) == "https://***@pkgs.example.com/simple"
    assert redact("https://pkgs.example.com/simple#x") == "https://pkgs.example.com/simple"


def test_redact_hides_token_like_path_segments():
    token = "AbCdEf0123456789xyz"
    assert redact(f"https://pypi.fury.io/{token}/acct/") == "https://pypi.fury.io/***/acct/"
    assert redact("https://pkgs.example.com/simple/") == "https://pkgs.example.com/simple/"
    assert redact("https://pkgs.example.com/api/pypi/pypi-local-repository/simple") == (
        "https://pkgs.example.com/api/pypi/pypi-local-repository/simple"
    )


def test_config_files_follow_pips_order_on_windows_and_posix():
    windows = config_files(
        {"USERPROFILE": "C:/u", "APPDATA": "C:/u/ad", "PROGRAMDATA": "C:/pd"},
        windows=True,
        prefix="C:/venv",
    )
    assert [p.as_posix() for p in windows] == [
        "C:/pd/pip/pip.ini",
        "C:/u/pip/pip.ini",
        "C:/u/ad/pip/pip.ini",
        "C:/venv/pip.ini",
    ]
    posix = config_files(
        {"HOME": "/h", "XDG_CONFIG_DIRS": "/etc/xdg", "PIP_CONFIG_FILE": "/c/pip.conf"},
        windows=False,
        prefix="/venv",
    )
    assert [p.as_posix() for p in posix] == [
        "/etc/xdg/pip/pip.conf",
        "/etc/pip.conf",
        "/h/.pip/pip.conf",
        "/h/.config/pip/pip.conf",
        "/venv/pip.conf",
        "/c/pip.conf",
    ]


def test_settings_come_from_files_then_the_environment(tmp_path):
    first = _ini(tmp_path / "a.ini", "[global]\nindex-url = https://a/simple\n")
    second = _ini(
        tmp_path / "b.ini",
        "[install]\nextra-index-url =\n    https://x/simple\n    https://y/simple\n",
    )
    settings, problems = read_settings(
        [first, tmp_path / "absent.ini", second], {"PIP_INDEX_URL": "https://env/simple"}
    )
    assert problems == []
    assert [(s.key, s.url) for s in settings] == [
        ("index-url", "https://a/simple"),
        ("extra-index-url", "https://x/simple"),
        ("extra-index-url", "https://y/simple"),
        ("index-url", "https://env/simple"),
    ]
    assert settings[-1].source == "$PIP_INDEX_URL"


def test_an_unreadable_file_is_reported_without_its_content(tmp_path):
    broken = _ini(tmp_path / "pip.ini", "index-url = https://u:TOPSECRET@h/simple\n")
    findings = index_findings({}, [broken])
    assert findings[0].level == "warn"
    assert "MissingSectionHeaderError" in findings[0].message
    assert all("TOPSECRET" not in f.message for f in findings)


def test_no_index_means_public_pypi():
    (finding,) = index_findings({}, [])
    assert (finding.level, finding.topic) == ("ok", "pip index")
    assert "public PyPI" in finding.message


def test_a_mirror_index_is_ok_and_redacted():
    (finding,) = index_findings({"PIP_INDEX_URL": INTERNAL}, [])
    assert finding.level == "ok"
    assert finding.message == "https://***@pkgs.example.com/simple (from $PIP_INDEX_URL)"


def test_an_extra_index_warns_naming_dependency_confusion(tmp_path):
    conf = _ini(tmp_path / "pip.ini", f"[global]\nextra-index-url = {INTERNAL}\n")
    findings = index_findings({"PIP_INDEX_URL": "https://pkgs.example.com/simple"}, [conf])
    warnings = [f for f in findings if f.level == "warn"]
    assert len(warnings) == 1
    assert "dependency confusion" in warnings[0].message
    assert "s3cret" not in warnings[0].message and "token" not in warnings[0].message


def test_uv_settings_are_reported():
    findings = index_findings(
        {"PIP_INDEX_URL": "https://i/simple", "UV_DEFAULT_INDEX": "https://i/simple"}, []
    )
    assert findings[-1].line() == "[ok   ] uv index: set by UV_DEFAULT_INDEX"
    findings = index_findings({"UV_EXTRA_INDEX_URL": "https://x/simple"}, [])
    assert findings[-1].level == "warn"
