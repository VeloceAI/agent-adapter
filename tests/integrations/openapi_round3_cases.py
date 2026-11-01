"""OpenAPI importer cases from review round 3: cost relative to size, gzip members, waits.

Collected through `test_openapi_contract.py` (the `contract-openapi` session runs that file).
"""

from __future__ import annotations

import json
import tracemalloc
import zlib
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent_adapter.core import _attempt
from agent_adapter.core.context import ToolContext
from agent_adapter.core.errors import ToolError
from agent_adapter.core.registry import Registry
from agent_adapter.exec import Executor
from agent_adapter.exec.pipeline import current_attempt
from agent_adapter.integrations.openapi import _decode, _refs, import_openapi
from agent_adapter.integrations.openapi import _http as http_module
from tests.integrations.openapi_support import PETS, Wire, pets, raw
from tests.integrations.openapi_work import run_counted


def fan_out_cycle(depth: int = 8, operations: int = 20) -> dict[str, Any]:
    """S_i refers to S_{i+1} twice and back to S0; every operation sends and returns S0."""
    schemas: dict[str, Any] = {}
    for i in range(depth):
        nxt = {"$ref": f"#/components/schemas/S{(i + 1) % depth}"}
        schemas[f"S{i}"] = {
            "type": "object",
            "properties": {"a": nxt, "b": nxt, "back": {"$ref": "#/components/schemas/S0"}},
        }
    s0 = {"schema": {"$ref": "#/components/schemas/S0"}}
    paths = {
        f"/p{n}": {
            "post": {
                "operationId": f"op{n}",
                "requestBody": {"content": {"application/json": s0}},
                "responses": {"200": {"description": "ok", "content": {"application/json": s0}}},
            }
        }
        for n in range(operations)
    }
    return {**PETS, "components": {"schemas": schemas}, "paths": paths}


# Lines of agent_adapter run (deterministic, unlike seconds): the 20-operation import runs
# ~0.6 M; a 500-operation one stops at its budget in ~0.3 M. Without a size-relative budget
# and shared $ref values they run 2.3 M and over 50 M.
FAN_OUT_LINES = 1_200_000
LARGE_FAN_OUT_LINES = 1_000_000


# --- 1 + 2: cyclic fan-out specs do bounded work, relative to their size ---------------------


def test_a_fan_out_cyclic_spec_imports_in_bounded_work(tmp_path: Path):
    result = run_counted(fan_out_cycle(), tmp_path, cap=FAN_OUT_LINES)
    assert result["outcome"] == "ok" and len(result["names"]) == 20, result["outcome"]


def test_a_large_fan_out_cyclic_spec_fails_on_its_budget_in_bounded_work(tmp_path: Path):
    document = fan_out_cycle(depth=8, operations=500)
    result = run_counted(document, tmp_path, cap=LARGE_FAN_OUT_LINES, memory=True)
    assert result["peak_mb"] < 50, result["peak_mb"]
    assert result["outcome"].startswith("ConfigError"), result["outcome"]
    assert "budget" in result["outcome"], result["outcome"]


@pytest.mark.parametrize(
    ("values", "required"),
    [(10, 50_000), (1_000, 64_000), (20_000, 1_000_000), (10**6, 1_000_000)],
)
def test_the_inlining_budget_is_relative_to_the_document_size(values: int, required: int):
    assert _refs.budgets(values) == (required, required // 2)


def test_ordinary_specs_stay_well_within_their_budget():
    registry = Registry()
    specs = import_openapi(registry, PETS, client_factory=Wire().factory)
    assert {s.name for s in specs} >= {"getPet", "createPet", "setTags"}
    petstore = {
        **PETS,
        "paths": {f"/r{i}/{{id}}": PETS["paths"]["/pets/{petId}"] for i in range(40)},
    }
    petstore = json.loads(json.dumps(petstore).replace("petId", "id"))
    assert len(import_openapi(Registry(), petstore, client_factory=Wire().factory)) == 80


# --- 3: concatenated gzip members --------------------------------------------------------------


def gzip_member(data: bytes) -> bytes:
    packer = zlib.compressobj(9, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
    return packer.compress(data) + packer.flush()


async def test_concatenated_gzip_members_are_all_decoded():
    data = gzip_member(b'{"a": ') + gzip_member(b"[1, 2]") + gzip_member(b"}")
    reply = raw(200, data, content_encoding="gzip", content_type="application/json")
    registry, _ = pets(wire=Wire(reply))
    assert await registry.get("getPet").invoke({"petId": 1}) == {"a": [1, 2]}


@pytest.mark.parametrize(
    ("encoding", "data"),
    [
        ("gzip", gzip_member(b"ok") + b"garbage"),
        ("deflate", zlib.compress(b"ok") + b"garbage"),
        ("gzip", gzip_member(b"ok")[:-6]),  # truncated
    ],
)
async def test_trailing_or_missing_data_is_a_tool_error(encoding: str, data: bytes):
    registry, _ = pets(wire=Wire(raw(200, data, content_encoding=encoding)))
    with pytest.raises(ToolError, match="could not be decoded"):
        await registry.get("getPet").invoke({"petId": 1})


async def test_a_multi_member_gzip_bomb_is_bounded_in_memory():
    member = gzip_member(b"\0" * (4 << 20))
    payload = member * 16  # 64 MiB in 16 members
    registry, _ = pets(wire=Wire(lambda _: raw(200, payload, content_encoding="gzip")))
    tracemalloc.start()
    try:
        with pytest.raises(ToolError, match="larger than"):
            await registry.get("getPet").invoke({"petId": 1})
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 3 * _decode.MAX_RESPONSE_BYTES


# --- 4: Retry-After waits only when the executor will retry ---------------------------------


async def test_a_direct_invoke_with_a_call_id_never_waits(monkeypatch: pytest.MonkeyPatch):
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(http_module, "_sleep", record)
    registry, _ = pets(wire=Wire(httpx.Response(429, headers={"retry-after": "4"})), retries=3)
    ctx = ToolContext(run_id="run_direct", call_id="call_direct")
    for _ in range(2):
        with pytest.raises(httpx.HTTPStatusError):
            await registry.get("getPet").invoke({"petId": 1}, ctx)
    assert waits == []


async def test_the_executor_tells_each_attempt_whether_another_follows():
    seen: list[Any] = []

    async def flaky(n: int, ctx: ToolContext) -> int:
        """Fail until the last attempt."""
        seen.append(current_attempt())
        if len(seen) < 3:
            raise RuntimeError("again")
        return n

    from agent_adapter.core.tool import build_spec

    async def no_sleep(_: float) -> None:
        return None

    registry = Registry()
    registry.add(build_spec(flaky, retries=2))
    assert await Executor(registry, sleep=no_sleep).run("flaky", {"n": 5}) == 5
    assert [(a.number, a.total, a.last) for a in seen] == [
        (1, 3, False),
        (2, 3, False),
        (3, 3, True),
    ]
    assert len({a.call_id for a in seen}) == 1


async def test_copied_attempt_info_of_another_call_does_not_count(monkeypatch: pytest.MonkeyPatch):
    waits: list[float] = []

    async def record(seconds: float) -> None:
        waits.append(seconds)

    monkeypatch.setattr(http_module, "_sleep", record)
    registry, _ = pets(wire=Wire(httpx.Response(429, headers={"retry-after": "4"})))
    ctx = ToolContext(run_id="r", call_id="call_child")
    token = _attempt.ATTEMPT.set(_attempt.Attempt("call_parent", 1, 3))  # a parent's attempt
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await registry.get("getPet").invoke({"petId": 1}, ctx)
    finally:
        _attempt.ATTEMPT.reset(token)
    assert waits == []


async def test_an_empty_compressed_body_is_none():
    registry, _ = pets(wire=Wire(raw(204, b"", content_encoding="gzip")))
    assert await registry.get("deletePet").invoke({"petId": 1}) is None
