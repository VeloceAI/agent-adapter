"""The LoopRunner example and the deploy files (app, Dockerfile, Kubernetes) run offline."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
DEPLOY = EXAMPLES / "deploy"


def test_loop_runner_example_keeps_one_client_warm(capsys):
    runpy.run_path(str(EXAMPLES / "backend_loop_runner.py"), run_name="__main__")
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "{'status': 'ok', 'rate': 18.0}",
        "{'status': 'ok', 'rate': 5.0}",
        "1 client",
        "[True]",
    ]


def _dockerfile() -> dict[str, str]:
    """Each Dockerfile instruction (continuation lines joined), keyed by its last use."""
    text = DEPLOY.joinpath("Dockerfile").read_text(encoding="utf-8").replace("\\\n", " ")
    found: dict[str, str] = {}
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            word, _, rest = line.strip().partition(" ")
            found[word] = rest.strip()
    return found


def _command() -> list[str]:
    command: list[str] = json.loads(_dockerfile()["CMD"])  # the exec form is a JSON array
    return command


def _flag(name: str) -> str:
    command = _command()
    return command[command.index(name) + 1]


def _manifests() -> list[dict[str, Any]]:
    yaml = pytest.importorskip("yaml")
    return list(yaml.safe_load_all(DEPLOY.joinpath("k8s.yaml").read_text(encoding="utf-8")))


def test_dockerfile_runs_non_root_with_a_graceful_shutdown():
    docker = _dockerfile()
    assert docker["FROM"].startswith("python:3.1") and docker["FROM"].endswith("-slim")
    assert docker["USER"] not in ("root", "0")
    command = _command()
    assert command[:2] == ["uvicorn", "app:app"]
    assert "--preload" not in command and "gunicorn" not in command
    assert docker["EXPOSE"] == _flag("--port") == "8000"
    assert int(_flag("--timeout-graceful-shutdown")) == 30


def test_kubernetes_probes_drain_inside_the_grace_period():
    deployment = next(m for m in _manifests() if m["kind"] == "Deployment")
    pod = deployment["spec"]["template"]["spec"]
    container = pod["containers"][0]
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    sleep = int(container["lifecycle"]["preStop"]["exec"]["command"][-1])
    graceful = int(_flag("--timeout-graceful-shutdown"))
    assert pod["terminationGracePeriodSeconds"] > sleep + graceful
    env = {item["name"]: item for item in container["env"]}
    assert "secretKeyRef" in env["TOOLS_TOKEN"]["valueFrom"]  # never a literal token
    hosts = env["ALLOWED_HOSTS"]["value"].split(",")
    for probe in ("livenessProbe", "readinessProbe"):
        headers = {h["name"]: h["value"] for h in container[probe]["httpGet"]["httpHeaders"]}
        assert headers["Host"] in hosts  # else the probe gets 421


async def test_deploy_app_serves_probes_and_drains():
    pytest.importorskip("starlette")
    httpx2 = pytest.importorskip("httpx2")
    app_module = runpy.run_path(str(DEPLOY / "app.py"))
    assert app_module["app"] is None  # no TOOLS_TOKEN in the test environment
    app = app_module["build_app"]({"TOOLS_TOKEN": "t-1", "ALLOWED_HOSTS": "tools, tools.svc"})
    transport = httpx2.ASGITransport(app=app)
    async with httpx2.AsyncClient(transport=transport, base_url="http://tools") as http:
        async with app.router.lifespan_context(app):
            assert (await http.get("/healthz")).status_code == 200
            assert (await http.get("/readyz")).status_code == 200
            call = await http.post(
                "/tools/shipping_rate/invoke",
                json={"arguments": {"region": "us"}},
                headers={"Authorization": "Bearer t-1"},
            )
            assert call.status_code == 200 and call.json()["result"] == 18.0
            assert (await http.get("/readyz", headers={"Host": "evil"})).status_code == 421
        assert (await http.get("/readyz")).status_code == 503


def test_deploy_app_uses_shared_stores_when_redis_is_configured():
    pytest.importorskip("redis")
    app_module = runpy.run_path(str(DEPLOY / "app.py"))
    state = app_module["shared_state"]("redis://localhost:6379/0")  # no connection is made
    assert set(state) == {"approvals", "idempotency"}
    assert app_module["shared_state"](None) == {}
