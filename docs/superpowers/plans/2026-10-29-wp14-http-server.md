# WP14 — HTTP tool server and WSGI: TDD plan

Contract §15 (produces), §7, §8, §9.4, §10.3, §14.4 `mcp` (consumes). Exit gate: every
Server row in §20.3 including the §15.5 security tests.

## Design decisions (security, §15.5)

| Topic | Rule |
|---|---|
| call id | `Idempotency-Key: k` → `scoped_id("call_", principal.id, k)`; else `new_id("call_")` |
| run id | the client sees its run key (`run_id` / `X-Run-Id`, else `new_id("run_")`); the executor sees `scoped_id("run_", owner, key)`, so a lost owner entry or another worker never merges two principals' runs |
| thread id | executor sees `scoped_id("thr_", principal, key)` (thread-scoped tool cache never shared across principals) |
| owner table | bounded LRU, per-principal quota; another principal's run key → 403 |
| approvals | metadata `principal`, `thread_id`, `run_key`, `origin`, `idempotency` set by a `DeferredApproval` subclass; read = owner or `APPROVER_SCOPE` (else 404); decide = `authorize_decision` (else 403) |
| idempotency | `(principal, key)` → cached response + fingerprint (tool, args digest, run key, thread key); other fingerprint → 400; concurrent duplicate → 409 |
| anonymous | only with `allow_anonymous=True`; all anonymous callers share one identity (`None` in scoped ids); authenticated principals keep every check |
| logs / problems | principal ids logged as a short hash; no tokens, bodies or arguments; unknown exceptions show their class only |
| executor | `.replace(approval=…, sink=FanoutSink(…), ledger=…)`, never mutated |

## Tasks

1. `runtime/ids.py`: `scoped_id` moved from `integrations/mcp/_scope.py` (re-exported there); tests.
2. `server/auth.py`: `Principal`, `RequestInfo`, `bearer_auth` (sha256-indexed tokens), defaults; tests.
3. `server/problems.py`: `Problem`, `problem_for`, kind → status/title table; tests per row.
4. `server/events.py`: `MemoryEventStream` (bounded, seq per run, thread-safe wake-ups); tests.
5. `server/openapi.py`: `openapi_document`; tests (one `invoke_<tool>` op, components).
6. `server/_runs.py` (owner table, scoping), `server/_idempotency.py` (replay cache, in-flight),
   `server/_approval.py` (metadata port), `server/_payloads.py` (bodies, descriptors),
   `server/_sse.py` (frames, keep-alive, idle close); unit tests.
7. `server/service.py`: `ToolService`, `ServiceResponse`, routing table §15.2; route tests
   for every status code; idempotent replay; approval round trip; SSE replay.
8. Security suite `tests/server/test_security*.py`: cross-principal invoke / events / cost /
   list / read / decide; key reuse → other call id; replay with other args; concurrent
   duplicates; non-owner decide; SSE isolation; oversized / malformed bodies; problem bodies
   never echo secrets; host executor never mutated; anonymous rules.
9. `server/asgi.py` (`create_app`, `serve`, MCP mount behind the same authenticator and
   `authorize`), `server/wsgi.py`, `server/__main__.py`; tests via `httpx.ASGITransport`,
   `a2wsgi`, CLI argument parsing.
10. `tests/server/test_ts_client.py`: uvicorn on an ephemeral port + `node --test
    clients/typescript/test` with `AA_BASE_URL` / tokens (skips without Node or the client).
11. Docs `docs/cross-language/{index,http-server,openapi}.md`, `examples/serve_tools.py` +
    example test; SHARED-EDITS (nav, features, CHANGELOG, exports).

## Exit gate

`pytest -q` 0 warnings, ≥ 90 % on new modules (server ODMs measured with
`--cov=agent_adapter.server` in a non-omitting config); ruff check / format; mypy strict;
mkdocs strict; `nox -s server` and `ts` (where Node and the client exist).
