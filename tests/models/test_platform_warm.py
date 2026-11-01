"""`warm_platform_cache`: fills `platform`'s caches once, on Windows without any WMI query."""

import logging
import platform
import sys
import sysconfig
import threading

import pytest

import agent_adapter.models._port as port_module
from agent_adapter.models._port import warm_platform_cache

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="WMI exists only on Windows")


@pytest.fixture
def cold(monkeypatch, wmi_query):
    """Empty platform caches and an unwarmed helper; monkeypatch restores both afterwards."""
    monkeypatch.setattr(port_module, "_platform_warmed", False, raising=False)
    monkeypatch.setattr(platform, "_uname_cache", None)
    monkeypatch.setattr(platform, "_platform_cache", {})


@pytest.fixture
def wmi_calls(monkeypatch):
    """Every `_wmi.exec_query` call; each one fails so the test itself never queries WMI."""
    wmi = pytest.importorskip("_wmi")
    calls = []

    def spy(query):
        calls.append(query)
        raise OSError("spy")

    monkeypatch.setattr(wmi, "exec_query", spy)
    return calls


def test_warm_platform_cache_calls_platform_only_on_windows(cold, monkeypatch):
    seen = []
    monkeypatch.setattr(platform, "platform", lambda **kw: seen.append(kw))
    monkeypatch.setattr(sys, "platform", "linux")
    warm_platform_cache()
    assert seen == []
    monkeypatch.setattr(sys, "platform", "win32")
    warm_platform_cache()
    assert seen == [
        {"aliased": aliased, "terse": terse}
        for aliased, terse in ((False, False), (True, False), (False, True), (True, True))
    ]


def test_warm_platform_cache_runs_once_per_process(cold, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    warm_platform_cache()
    seen = []
    monkeypatch.setattr(platform, "platform", lambda **kw: seen.append(kw))
    warm_platform_cache()
    assert seen == []


def test_warm_platform_cache_is_a_no_op_without_wmi_query(cold, monkeypatch):
    seen = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delattr(platform, "_wmi_query")
    monkeypatch.setattr(platform, "platform", lambda **kw: seen.append(kw))
    warm_platform_cache()
    assert seen == []


def test_warm_platform_cache_restores_wmi_query_when_platform_fails(cold, monkeypatch):
    original = platform._wmi_query

    def boom(**_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(platform, "platform", boom)
    with pytest.raises(RuntimeError, match="boom"):
        warm_platform_cache()
    assert platform._wmi_query is original
    assert port_module._platform_warmed is False


@windows_only
def test_warm_platform_cache_never_queries_wmi(cold, wmi_calls):
    original = platform._wmi_query
    warm_platform_cache()
    assert wmi_calls == []
    assert platform._wmi_query is original
    assert platform._uname_cache is not None
    assert "processor" in vars(platform._uname_cache)  # the lazy processor is cached too


@windows_only
def test_a_worker_thread_reads_platform_without_wmi_after_warming(cold, wmi_calls):
    warm_platform_cache()
    seen = []
    worker = threading.Thread(
        target=lambda: seen.append((platform.platform(), platform.uname().processor))
    )
    worker.start()
    worker.join()
    assert wmi_calls == []
    assert seen == [(platform.platform(), platform.uname().processor)]


@windows_only
def test_concurrent_warming_never_queries_wmi(cold, wmi_calls):
    original = platform._wmi_query
    barrier = threading.Barrier(8)

    def warm():
        barrier.wait()
        warm_platform_cache()

    workers = [threading.Thread(target=warm) for _ in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert wmi_calls == []
    assert platform._wmi_query is original


@windows_only
def test_warming_fills_every_platform_string_variant(cold, wmi_calls):
    warm_platform_cache()
    assert wmi_calls == []
    assert set(platform._platform_cache) == {
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    }


@pytest.fixture
def no_arch_env(monkeypatch):
    """No PROCESSOR_ARCHITECTURE variables: without WMI, CPython leaves the machine blank."""
    monkeypatch.delenv("PROCESSOR_ARCHITECTURE", raising=False)
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)


@windows_only
def test_a_blank_machine_is_taken_from_the_build_platform(
    cold, wmi_calls, no_arch_env, monkeypatch
):
    monkeypatch.setattr(sysconfig, "get_platform", lambda: "win-arm64")
    warm_platform_cache()
    assert wmi_calls == []
    assert platform.machine() == "ARM64" and platform.uname().machine == "ARM64"
    assert "processor" in vars(platform._uname_cache)
    assert len(platform._platform_cache) == 4


@windows_only
def test_an_unknown_build_platform_leaves_the_machine_blank(
    cold, wmi_calls, no_arch_env, monkeypatch, caplog
):
    caplog.set_level(logging.WARNING, logger="agent_adapter.models._port")
    monkeypatch.setattr(sysconfig, "get_platform", lambda: "mingw_x86_64")
    warm_platform_cache()
    assert wmi_calls == [] and platform.uname().machine == ""
    assert "platform machine unknown (build mingw_x86_64)" in caplog.text


def test_a_known_machine_is_kept(cold, monkeypatch):
    info = platform.uname_result("Windows", "host", "11", "10.0", "AMD64")
    monkeypatch.setattr(sysconfig, "get_platform", lambda: "win-arm64")
    port_module._fill_machine(info)
    assert platform._uname_cache is None  # nothing replaced


def test_filling_the_machine_drops_platform_strings_built_without_it(cold, monkeypatch):
    platform._platform_cache[(False, False)] = "stale"
    monkeypatch.setattr(sysconfig, "get_platform", lambda: "win-amd64")
    port_module._fill_machine(platform.uname_result("Windows", "host", "11", "10.0", ""))
    assert platform._uname_cache.machine == "AMD64" and platform._platform_cache == {}
