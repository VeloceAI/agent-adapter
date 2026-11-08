"""`agent-adapter doctor` feature checks: guards, OTel exporters and the pluggable list."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_adapter.cli import doctor, pipindex
from agent_adapter.cli.doctor_features import FEATURES, Feature, check_feature, feature_findings

GUARD = Feature("guard demo", ("demo-core", "demo-extra"), extra="demo", purpose="demo checks")
OTEL = next(feature for feature in FEATURES if feature.topic == "otel export")


def versions(**found: str):
    """A version lookup that knows only `found` (underscores stand for dashes)."""
    table = {name.replace("_", "-"): value for name, value in found.items()}
    return table.get


def test_the_default_list_checks_the_guard_extras_and_otel():
    topics = [feature.topic for feature in FEATURES]
    assert topics[:3] == ["guard presidio", "guard nemo", "otel export"]
    assert topics[3:6] == ["memory mem0", "memory zep", "memory graphiti"]
    presidio = FEATURES[0]
    assert presidio.needs == ("presidio-analyzer", "presidio-anonymizer")
    assert FEATURES[1].needs == ("nemoguardrails",)


def test_a_ready_feature_is_ok_with_versions():
    finding = check_feature(GUARD, {}, versions(demo_core="1.0", demo_extra="2.0"))
    assert finding.line() == "[ok   ] guard demo: ready demo-core 1.0, demo-extra 2.0"


def test_an_absent_feature_is_info_with_the_extra_when_the_build_has_it():
    with_extra = check_feature(GUARD, {"demo": []}, versions())
    assert with_extra.level == "info"
    assert with_extra.message == 'not installed (demo checks): pip install "agent-adapter[demo]"'
    without = check_feature(GUARD, {}, versions())
    assert without.message.endswith("pip install demo-core demo-extra")


def test_a_half_installed_feature_warns_and_names_what_is_missing():
    finding = check_feature(GUARD, {"demo": []}, versions(demo_core="1.0"))
    assert finding.level == "warn"
    assert finding.message.startswith("demo-extra missing (demo checks): ")


def test_otel_needs_the_sdk_and_one_exporter():
    http = "opentelemetry-exporter-otlp-proto-http"
    base = {"opentelemetry-api": "1.39", "opentelemetry-sdk": "1.39"}
    ready = check_feature(OTEL, {"otel": []}, {**base, http: "1.39"}.get)
    assert ready.level == "ok" and http in ready.message
    grpc = "opentelemetry-exporter-otlp-proto-grpc"
    assert check_feature(OTEL, {}, {**base, grpc: "1.39"}.get).level == "ok"
    no_exporter = check_feature(OTEL, {"otel": []}, base.get)
    assert no_exporter.level == "info"  # opentelemetry-api often arrives with other libraries
    assert no_exporter.message.startswith(f"{http} missing")
    assert no_exporter.message.endswith(f"pip install {http}")  # the extra brings no exporter
    api_only = check_feature(OTEL, {"otel": []}, {"opentelemetry-api": "1.39"}.get)
    assert api_only.message.startswith(f"{http}, opentelemetry-sdk missing")
    absent = check_feature(OTEL, {"otel": []}, versions())
    assert absent.message.startswith("not installed")


@pytest.mark.parametrize("partial", ["info", "warn", "error"])
def test_a_feature_chooses_its_partial_level(partial):
    feature = Feature("x", ("a", "b"), partial=partial)
    assert check_feature(feature, {}, versions(a="1")).level == partial


def test_feature_findings_keep_table_order():
    found = feature_findings({}, [GUARD, OTEL], versions())
    assert [f.topic for f in found] == ["guard demo", "otel export"]
    assert feature_findings({}, [], versions()) == []


def test_diagnose_runs_the_features_it_is_given(monkeypatch):
    monkeypatch.setattr(pipindex, "config_files", lambda environ: [])
    monkeypatch.setattr(doctor, "_version", versions(demo_core="1.0", demo_extra="2.0"))
    findings = doctor.diagnose({}, Path.cwd(), features=[GUARD])
    topics = [f.topic for f in findings]
    assert topics[:4] == ["python", "agent-adapter", "extras", "guard demo"]
    assert findings[3].level == "ok"
    default = [f.topic for f in doctor.diagnose({}, Path.cwd())]
    assert default[3:6] == ["guard presidio", "guard nemo", "otel export"]


def test_docker_finding_reports_versions_or_the_reason():
    from agent_adapter.cli.doctor_features import docker_finding
    from agent_adapter.sandbox.docker import DockerStatus

    ready = docker_finding(DockerStatus(True, client_version="27.1", server_version="27.1"))
    assert (ready.level, ready.topic) == ("ok", "sandbox docker")
    assert "27.1" in ready.message
    missing = docker_finding(DockerStatus(False, reason="'docker' is not on PATH"))
    assert missing.level == "info" and "not on PATH" in missing.message


def test_diagnose_adds_the_docker_check_after_the_features(monkeypatch):
    from agent_adapter.cli.findings import Finding

    monkeypatch.setattr(pipindex, "config_files", lambda environ: [])
    stub = Finding("info", "sandbox docker", "not available (stub)")
    topics = [
        f.topic for f in doctor.diagnose({}, Path.cwd(), features=[GUARD], docker=lambda: stub)
    ]
    assert topics[3:5] == ["guard demo", "sandbox docker"]
