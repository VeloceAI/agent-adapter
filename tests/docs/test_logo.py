"""The logo variants share mark.svg's geometry, and every place that shows the logo uses them."""

import re
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
CANONICAL = "mark.svg"
# favicon.svg and app-icon.svg are drawn heavier for small tiles, so they have their own geometry.
VARIANTS = [
    "mark-dark.svg",
    "mark-on-primary.svg",
    "logo.svg",
    "logo-dark.svg",
    "logo-outlined.svg",
    "logo-dark-outlined.svg",
    "logo-card-outlined.svg",  # the light logo on a white card: readable on any README theme
]
# docs/assets holds the whole final set under its stable names, so a new version drops in.
FULL_SET = {"favicon.svg", "app-icon.svg", CANONICAL, *VARIANTS}
# Raster fallbacks rendered from the SVGs: browsers without SVG favicons, iOS, link previews.
PNGS = {
    "favicon-32.png": (32, 32),
    "apple-touch-icon.png": (180, 180),
    "social-card.png": (1200, 630),
}
OVERRIDES = ROOT / "overrides" / "main.html"
SVG = "{http://www.w3.org/2000/svg}"


def _mark(name: str) -> list[tuple[str, dict[str, str]]]:
    """The stroked shapes of the mark, without colours."""
    root = ET.parse(ASSETS / name).getroot()
    shapes = []
    for node in root:
        if node.get("fill") != "none":
            continue  # wordmark glyphs and tiles
        attrs = {k: v for k, v in node.attrib.items() if k != "stroke"}
        shapes.append((node.tag.removeprefix(SVG), attrs))
    return shapes


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_draws_the_canonical_mark(name: str) -> None:
    assert _mark(name) == _mark(CANONICAL)


def test_assets_are_exactly_the_logo_set() -> None:
    assert {p.name for p in ASSETS.glob("*.svg")} == FULL_SET


def test_every_reference_exists() -> None:
    texts = {
        "mkdocs.yml": (ROOT / "mkdocs.yml").read_text(encoding="utf-8"),
        "README.md": (ROOT / "README.md").read_text(encoding="utf-8"),
        "docs/index.md": (ROOT / "docs" / "index.md").read_text(encoding="utf-8"),
        "overrides/main.html": OVERRIDES.read_text(encoding="utf-8"),
    }
    joined = "\n".join(texts.values())
    refs = re.findall(r"assets/([\w.-]+\.(?:svg|png))", joined)
    assert set(PNGS) <= set(refs), "the site links every raster icon"
    for ref in refs:
        assert (ASSETS / ref).is_file(), ref


def test_the_site_uses_the_overrides_with_its_icons() -> None:
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^  custom_dir: overrides$", config, re.M)
    assert re.search(r"^  favicon: assets/favicon\.svg$", config, re.M)
    html = OVERRIDES.read_text(encoding="utf-8")
    assert 'rel="apple-touch-icon"' in html and 'type="image/svg+xml"' in html


@pytest.mark.parametrize(("name", "size"), sorted(PNGS.items()))
def test_raster_icons_are_pngs_of_their_size(name: str, size: tuple[int, int]) -> None:
    head = (ASSETS / name).read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR", name
    assert struct.unpack(">II", head[16:24]) == size


def test_every_logo_names_the_product() -> None:
    for path in ASSETS.glob("*.svg"):
        root = ET.parse(path).getroot()
        assert root.get("aria-label") == "Agent Adapter", path.name
        assert root.findtext(f"{SVG}title") == "Agent Adapter", path.name


def test_outlined_logos_need_no_font() -> None:
    for name in ("logo-outlined.svg", "logo-dark-outlined.svg", "logo-card-outlined.svg"):
        text = (ASSETS / name).read_text(encoding="utf-8")
        assert "<text" not in text and "font-family" not in text, name
