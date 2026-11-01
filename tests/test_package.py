import agent_adapter


def test_package_exposes_version():
    assert agent_adapter.__version__ == "1.0.0"


def test_all_extra_never_pulls_in_litellm():
    import re
    import tomllib
    from pathlib import Path

    extras = tomllib.loads(Path(__file__).parents[1].joinpath("pyproject.toml").read_text())[
        "project"
    ]["optional-dependencies"]
    named = {n for spec in extras["all"] for n in re.findall(r"[\w-]+", spec.split("[", 1)[1])}
    assert "dspy" not in named  # dspy depends on litellm
    assert not any("litellm" in dep for name in named for dep in extras.get(name, []))


def test_built_wheel_and_sdist_ship_both_agent_guides(tmp_path):
    import tarfile
    import zipfile
    from pathlib import Path

    # hatchling is in the dev extra: a missing one must fail here, not skip.
    from hatchling.builders.sdist import SdistBuilder
    from hatchling.builders.wheel import WheelBuilder

    root = Path(__file__).parents[1]
    (wheel,) = WheelBuilder(str(root)).build(directory=str(tmp_path), versions=["standard"])
    (sdist,) = SdistBuilder(str(root)).build(directory=str(tmp_path), versions=["standard"])
    with zipfile.ZipFile(wheel) as built:
        concise = built.read("agent_adapter/_guide/AGENTS.md").decode("utf-8")
        full = built.read("agent_adapter/_guide/llms-full.txt").decode("utf-8")
    with tarfile.open(sdist) as source:
        names = set(source.getnames())
    assert concise == (root / "AGENTS.md").read_text(encoding="utf-8")
    assert full == (root / "docs" / "llms-full.txt").read_text(encoding="utf-8")
    for name in ("AGENTS.md", "llms-full.txt"):
        assert f"agent_adapter-1.0.0/agent_adapter/_guide/{name}" in names


def test_built_wheel_and_sdist_ship_the_viewer_page(tmp_path):
    import tarfile
    import zipfile
    from pathlib import Path

    from hatchling.builders.sdist import SdistBuilder
    from hatchling.builders.wheel import WheelBuilder

    root = Path(__file__).parents[1]
    static = sorted(p.name for p in (root / "agent_adapter" / "viewer" / "static").iterdir())
    (wheel,) = WheelBuilder(str(root)).build(directory=str(tmp_path), versions=["standard"])
    (sdist,) = SdistBuilder(str(root)).build(directory=str(tmp_path), versions=["standard"])
    with zipfile.ZipFile(wheel) as built:
        in_wheel = set(built.namelist())
    with tarfile.open(sdist) as source:
        in_sdist = set(source.getnames())
    for name in static:
        assert f"agent_adapter/viewer/static/{name}" in in_wheel
        assert f"agent_adapter-1.0.0/agent_adapter/viewer/static/{name}" in in_sdist
