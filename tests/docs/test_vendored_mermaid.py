"""The docs site draws diagrams with its own copy of Mermaid: no CDN, licence kept, not packaged."""

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / "docs" / "assets" / "javascripts" / "vendor"
VERSION = "11.17.2"
SCRIPT = f"assets/javascripts/vendor/mermaid-{VERSION}.min.js"
# sha256 of dist/mermaid.min.js in the npm tarball mermaid-11.17.2.tgz, whose sha512
# integrity matched the registry's (V6K3C8EBdEsPFZXSKMJe6ppQOENxuHARr9GvHX4hh47lAbhMRD9qf4oE...).
SHA256 = "581ed7d74bd9048d0e3a91363927d72ef22942d7722546b27f7cc29e35390eb8"
CDN = re.compile(r"unpkg\.com|jsdelivr\.net|cdnjs\.")


def test_the_vendored_file_is_the_verified_release_with_its_licence() -> None:
    script = VENDOR / f"mermaid-{VERSION}.min.js"
    assert hashlib.sha256(script.read_bytes()).hexdigest() == SHA256
    assert (
        script.read_text(encoding="utf-8")
        .rstrip()
        .endswith('globalThis["mermaid"] = globalThis.__esbuild_esm_mermaid_nm["mermaid"].default;')
    )
    licence = (VENDOR / f"mermaid-{VERSION}.LICENSE.txt").read_text(encoding="utf-8")
    assert licence.startswith("The MIT License (MIT)") and "Knut Sveidqvist" in licence
    assert {p.name for p in VENDOR.iterdir()} == {script.name, f"mermaid-{VERSION}.LICENSE.txt"}


def test_the_built_config_loads_it_before_the_narrow_screen_tweak() -> None:
    config_module = pytest.importorskip("mkdocs.config")
    pytest.importorskip("material")
    pytest.importorskip("mkdocstrings")
    config = config_module.load_config(str(ROOT / "mkdocs.yml"))
    scripts = [str(getattr(item, "path", item)) for item in config.extra_javascript]
    assert scripts[:2] == [SCRIPT, "assets/javascripts/diagrams.js"]
    assert all(getattr(item, "defer", False) for item in config.extra_javascript)
    assert (ROOT / "docs" / SCRIPT).is_file()


def test_no_page_or_override_loads_a_script_from_a_cdn() -> None:
    sources = [ROOT / "mkdocs.yml", ROOT / "overrides" / "main.html"]
    sources += [
        p for p in (ROOT / "docs").rglob("*") if p.suffix in {".md", ".html", ".js", ".css"}
    ]
    for path in sources:
        rel = path.relative_to(ROOT)
        if "superpowers" in rel.parts or VENDOR in path.parents:
            continue
        assert not CDN.search(path.read_text(encoding="utf-8")), rel


def test_docs_assets_stay_out_of_the_wheel_and_the_sdist() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    targets = config["tool"]["hatch"]["build"]["targets"]
    assert targets["wheel"]["packages"] == ["agent_adapter"]
    assert "docs" not in targets["sdist"]["only-include"]
    assert not any(
        p.suffix == ".js" and "mermaid" in p.name for p in (ROOT / "agent_adapter").rglob("*")
    )
