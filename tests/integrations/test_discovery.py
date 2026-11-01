import importlib.metadata
import logging
import sys
import threading
from dataclasses import dataclass
from typing import Any

import pytest

from agent_adapter.core.errors import IntegrationError, MissingDependency
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations import anthropic, discovery, gemini, openai
from agent_adapter.integrations.base import IntegrationBase, ToolRunner
from agent_adapter.integrations.discovery import GROUP, available_integrations, get_integration

DEMO_DIST = "agent-adapter-demo-integration"
LOGGER = "agent_adapter.integrations.discovery"


class Good(IntegrationBase):
    name = "good"

    def export_tool(self, spec: ToolSpec, *, executor: ToolRunner | None = None) -> Any:
        return spec.name


class Needy(Good):
    name = "needy"
    requires = ("aa_missing_module",)
    extra = "needy"


class Exploding(Good):
    name = "exploding"

    def __init__(self) -> None:
        raise RuntimeError("constructor failed")


class NoMethods:
    name = "no_methods"
    requires = ()
    extra = ""


class BadTypes:
    name = "bad_types"
    requires = ["not", "a", "tuple"]  # noqa: RUF012 - the wrong type is the point
    extra = ""

    def export_tool(self, spec: Any, *, executor: Any = None) -> Any: ...

    def bind_context(self, native: Any) -> Any: ...

    def bind_approval(self, port: Any) -> Any: ...


GOOD = Good()
NEEDY = Needy()


@dataclass
class FakeDist:
    name: str


@dataclass
class FakeEntryPoint:
    """Duck-typed entry point whose `load` raises."""

    name: str
    error: BaseException
    value: str = "fake.module:OBJ"
    dist: FakeDist | None = None

    def load(self) -> Any:
        raise self.error


def ep(name: str, attr: str) -> importlib.metadata.EntryPoint:
    return importlib.metadata.EntryPoint(name, f"{__name__}:{attr}", GROUP)


@pytest.fixture(autouse=True)
def fresh_cache():
    available_integrations(refresh=True)
    yield
    available_integrations(refresh=True)


@pytest.fixture
def entry_points(monkeypatch):
    """Install the given entry points; returns the list of groups requested."""
    calls: list[str] = []

    def install(*points: Any) -> list[str]:
        def fake(*, group: str) -> list[Any]:
            calls.append(group)
            return list(points)

        monkeypatch.setattr(importlib.metadata, "entry_points", fake)
        return calls

    return install


def warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelname == "WARNING"]


def tracebacks(caplog) -> list[BaseException]:
    """The exceptions logged with a traceback: only at DEBUG (their text can hold secrets)."""
    logged = [r for r in caplog.records if r.name == LOGGER and r.exc_info]
    assert all(r.levelno == logging.DEBUG for r in logged)
    return [r.exc_info[1] for r in logged]


def test_loads_module_attributes_and_instantiates_classes(entry_points):
    calls = entry_points(ep("good", "GOOD"), ep("good_class", "GoodClassAlias"))
    found = available_integrations(refresh=True)
    assert found["good"] is GOOD
    assert isinstance(found["good_class"], Good)
    assert calls == [GROUP]


class GoodClassAlias(Good):
    name = "good_class"


def test_missing_module_is_skipped_with_a_warning(entry_points, caplog):
    entry_points(
        importlib.metadata.EntryPoint("gone", "aa_no_such_module:INTEGRATION", GROUP),
        ep("good", "GOOD"),
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        found = available_integrations(refresh=True)
    assert list(found) == ["good"]
    [message] = warnings(caplog)
    assert "gone" in message and "aa_no_such_module" in message and "not installed" in message


def test_missing_attribute_is_skipped_with_a_warning(entry_points, caplog):
    entry_points(ep("absent", "NO_SUCH_ATTRIBUTE"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert available_integrations(refresh=True) == {}
    [message] = warnings(caplog)
    assert "absent" in message and "failed to load" in message


def test_broken_plugin_is_skipped_with_its_traceback_at_debug(entry_points, caplog):
    entry_points(
        FakeEntryPoint("broken", RuntimeError("boom"), dist=FakeDist("broken-dist")),
        ep("good", "GOOD"),
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        found = available_integrations(refresh=True)
    assert list(found) == ["good"]
    [message] = warnings(caplog)
    assert "broken" in message and "broken-dist" in message and "RuntimeError" in message
    assert [str(exc) for exc in tracebacks(caplog)] == ["boom"]


def test_failing_constructor_is_skipped(entry_points, caplog):
    entry_points(ep("exploding", "Exploding"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert available_integrations(refresh=True) == {}
    [message] = warnings(caplog)
    assert "exploding" in message and "could not be created" in message


@pytest.mark.parametrize(
    ("name", "attr", "reason"),
    [
        ("no_methods", "NoMethods", "does not implement the Integration protocol"),
        ("bad_types", "BadTypes", "requires must be a tuple of str"),
        ("good_alias", "GOOD", "name 'good' does not match the entry-point name 'good_alias'"),
    ],
)
def test_non_conforming_objects_are_skipped(entry_points, caplog, name, attr, reason):
    entry_points(ep(name, attr))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert available_integrations(refresh=True) == {}
    [message] = warnings(caplog)
    assert name in message and reason in message


def test_non_str_name_and_extra_are_rejected(entry_points, caplog, monkeypatch):
    class BadName(Good):
        name = 7  # type: ignore[assignment]

    class BadExtra(Good):
        name = "bad_extra"
        extra = None  # type: ignore[assignment]

    monkeypatch.setitem(globals(), "BadName", BadName)
    monkeypatch.setitem(globals(), "BadExtra", BadExtra)
    entry_points(ep("bad_name", "BadName"), ep("bad_extra", "BadExtra"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert available_integrations(refresh=True) == {}
    messages = warnings(caplog)
    assert "name must be a str" in messages[0]
    assert "extra must be a str" in messages[1]


def test_duplicate_names_keep_the_first(entry_points, caplog, monkeypatch):
    monkeypatch.setitem(globals(), "SECOND", Good())
    entry_points(ep("good", "GOOD"), ep("good", "SECOND"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        found = available_integrations(refresh=True)
    assert found["good"] is GOOD
    [message] = warnings(caplog)
    assert "registered more than once" in message


def test_base_exceptions_from_a_plugin_propagate(entry_points):
    entry_points(FakeEntryPoint("interrupt", KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        available_integrations(refresh=True)


def test_results_are_cached_until_refresh(entry_points):
    calls = entry_points(ep("good", "GOOD"))
    available_integrations(refresh=True)
    first = available_integrations()
    first["intruder"] = GOOD
    assert "intruder" not in available_integrations()
    assert calls == [GROUP]
    available_integrations(refresh=True)
    assert calls == [GROUP, GROUP]


def test_get_integration_returns_a_known_integration(entry_points):
    entry_points(ep("good", "GOOD"))
    available_integrations(refresh=True)
    assert get_integration("good") is GOOD


def test_get_integration_unknown_name_lists_the_available_ones(entry_points):
    entry_points(ep("good", "GOOD"))
    available_integrations(refresh=True)
    with pytest.raises(IntegrationError) as info:
        get_integration("nope")
    assert info.value.integration == "nope"
    assert "available: ['good']" in str(info.value)


def test_get_integration_checks_requirements(entry_points):
    entry_points(ep("needy", "NEEDY"))
    available_integrations(refresh=True)
    assert "needy" in available_integrations()
    with pytest.raises(MissingDependency, match=r'pip install "agent-adapter\[needy\]"'):
        get_integration("needy")


def test_the_renderer_integrations_are_discovered_through_our_entry_points():
    assert get_integration("openai") is openai.INTEGRATION
    assert get_integration("anthropic") is anthropic.INTEGRATION
    assert available_integrations()["gemini"] is gemini.INTEGRATION


def test_built_in_entry_points_point_at_module_integrations():
    points = {p.name: p.value for p in importlib.metadata.entry_points(group=GROUP)}
    assert points["openai"] == "agent_adapter.integrations.openai:INTEGRATION"
    assert discovery.GROUP == "agent_adapter.integrations"


def _demo_installed() -> bool:
    try:
        importlib.metadata.distribution(DEMO_DIST)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


needs_demo = pytest.mark.skipif(not _demo_installed(), reason="demo plugin not installed")


@needs_demo
def test_installed_demo_plugin_is_found():
    import demo_integration

    assert get_integration("demo") is demo_integration.INTEGRATION


@needs_demo
def test_installed_broken_plugins_are_skipped_with_warnings(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        found = available_integrations(refresh=True)
    assert "demo" in found
    for name in ("demo_broken", "demo_missing", "demo_not_an_integration"):
        assert name not in found
        assert any(name in message for message in warnings(caplog))


REENTRANT_MODULE = """
from agent_adapter.integrations.discovery import get_integration
get_integration("good")  # a plugin must not look up integrations while being discovered
INTEGRATION = None
"""

REENTRANT_CLASS = """
from agent_adapter.integrations.discovery import available_integrations
from tests.integrations.test_discovery import Good


class Reentrant(Good):
    name = "reentrant_class"

    def __init__(self) -> None:
        available_integrations()
"""


def _scratch_module(tmp_path, monkeypatch, name: str, source: str) -> None:
    (tmp_path / f"{name}.py").write_text(source, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)


def _discover_in_thread() -> dict[str, Any]:
    """Run discovery in a worker thread so a deadlock fails the test instead of hanging it."""
    result: dict[str, Any] = {}

    def work() -> None:
        try:
            result["found"] = available_integrations(refresh=True)
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), "discovery deadlocked on a re-entrant plugin"
    return result


def test_plugin_calling_get_integration_at_import_is_skipped_not_deadlocked(
    entry_points, caplog, tmp_path, monkeypatch
):
    _scratch_module(tmp_path, monkeypatch, "aa_reentrant_plugin", REENTRANT_MODULE)
    entry_points(
        importlib.metadata.EntryPoint("reentrant", "aa_reentrant_plugin:INTEGRATION", GROUP),
        ep("good", "GOOD"),
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        result = _discover_in_thread()
    assert list(result["found"]) == ["good"]
    [message] = warnings(caplog)
    assert "reentrant" in message
    [exc] = tracebacks(caplog)
    assert "called during discovery" in str(exc)
    assert available_integrations()["good"] is GOOD  # the guard is released afterwards


def test_plugin_constructor_calling_discovery_is_skipped(
    entry_points, caplog, tmp_path, monkeypatch
):
    _scratch_module(tmp_path, monkeypatch, "aa_reentrant_class", REENTRANT_CLASS)
    entry_points(
        importlib.metadata.EntryPoint("reentrant_class", "aa_reentrant_class:Reentrant", GROUP)
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        result = _discover_in_thread()
    assert result["found"] == {}
    assert "could not be created" in warnings(caplog)[0]


def test_discovery_error_releases_the_reentrancy_guard(entry_points):
    entry_points(FakeEntryPoint("interrupt", KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        available_integrations(refresh=True)
    entry_points(ep("good", "GOOD"))
    assert available_integrations(refresh=True) == {"good": GOOD}


def test_concurrent_discovery_from_two_threads_does_not_block(entry_points):
    entry_points(ep("good", "GOOD"))
    results = [_discover_in_thread(), _discover_in_thread()]
    assert all(r["found"] == {"good": GOOD} for r in results)


def test_a_plugin_whose_own_dependency_is_missing_logs_the_traceback_at_debug(
    entry_points, caplog, tmp_path, monkeypatch
):
    _scratch_module(tmp_path, monkeypatch, "aa_needs_dep", "import aa_dep_not_installed\n")
    entry_points(importlib.metadata.EntryPoint("needs_dep", "aa_needs_dep:INTEGRATION", GROUP))
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        assert available_integrations(refresh=True) == {}
    [message] = warnings(caplog)
    assert "failed to load" in message and "ModuleNotFoundError" in message
    [exc] = tracebacks(caplog)
    assert isinstance(exc, ModuleNotFoundError) and exc.name == "aa_dep_not_installed"


def test_a_missing_parent_package_is_logged_without_traceback(entry_points, caplog):
    entry_points(importlib.metadata.EntryPoint("deep", "aa_no_pkg.sub.mod:INTEGRATION", GROUP))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert available_integrations(refresh=True) == {}
    [record] = [r for r in caplog.records if r.name == LOGGER]
    assert "aa_no_pkg is not installed" in record.getMessage()
    assert record.exc_info is None


class SlowLoad:
    """An entry point whose load blocks until released, counting loads and instances."""

    name = "slow"
    value = "aa_slow:Slow"
    dist = None

    def __init__(self) -> None:
        self.loads = 0
        self.loading = threading.Event()
        self.proceed = threading.Event()

    def load(self) -> Any:
        self.loads += 1
        self.loading.set()
        assert self.proceed.wait(timeout=5)

        class Slow(Good):
            name = "slow"

        return Slow


@pytest.mark.parametrize("refresh", [False, True])
def test_concurrent_discoveries_run_once(entry_points, caplog, monkeypatch, refresh):
    slow = SlowLoad()
    calls = entry_points(slow, FakeEntryPoint("broken", RuntimeError("boom")))
    if not refresh:
        monkeypatch.setattr(discovery, "_found", None)  # a cold cache: both threads are "first"
    results: list[dict[str, Any]] = []

    def work() -> None:
        results.append(available_integrations(refresh=refresh))

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        first = threading.Thread(target=work, daemon=True)
        first.start()
        assert slow.loading.wait(timeout=5)
        second = threading.Thread(target=work, daemon=True)
        second.start()
        second.join(timeout=0.05)  # let it reach the discovery lock
        slow.proceed.set()
        first.join(timeout=10)
        second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert slow.loads == 1 and len(calls) == 1
    assert results[0]["slow"] is results[1]["slow"]
    assert sum("broken" in m for m in warnings(caplog)) == 1


def test_reentrant_guard_names_the_function_called(entry_points, caplog, tmp_path, monkeypatch):
    source = "from agent_adapter.integrations.discovery import available_integrations\n"
    source += "available_integrations()\n"
    _scratch_module(tmp_path, monkeypatch, "aa_reentrant_available", source)
    _scratch_module(tmp_path, monkeypatch, "aa_reentrant_plugin", REENTRANT_MODULE)
    entry_points(
        importlib.metadata.EntryPoint("by_available", "aa_reentrant_available:X", GROUP),
        importlib.metadata.EntryPoint("by_get", "aa_reentrant_plugin:INTEGRATION", GROUP),
    )
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        _discover_in_thread()
    errors = [str(exc) for exc in tracebacks(caplog)]
    assert errors[0].startswith("available_integrations called during discovery")
    assert errors[1].startswith("get_integration called during discovery")
