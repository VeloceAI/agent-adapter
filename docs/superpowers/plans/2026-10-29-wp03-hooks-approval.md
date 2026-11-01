# WP03 — Hooks and approval: TDD plan

Contract: `docs/superpowers/specs/2026-10-29-interface-contract.md` rev 2, §6.2, §7, §21.3 WP03.
Consumes WP01 only (base, runtime, tracing events, hooks base/dispatch, `tests/fakes`).

## Tasks

| # | Task | Files | Tests first |
|---|---|---|---|
| 1 | Plan | this file | — |
| 2 | Confirm rule + built-in ports | `approval/builtin.py` | `tests/approval/test_builtin.py`: `evaluate_confirm` bool / sync / async / non-bool / raising (logged, re-raised); `AutoApprove`, `AutoDeny`, `CallbackApproval` (bool, decision, async, bad type, `ApprovalPending` passes through) |
| 3 | Store | `approval/store.py` | `tests/approval/test_store.py`: create (fills id, `created_at`), duplicate id, get / not found, find newest unconsumed, decide / already decided, consume, list filters / order / limit, eviction order, 100 concurrent `decide` (one winner), same across two `run_sync` loops in two threads |
| 4 | `DeferredApproval` | `approval/deferred.py` | `tests/approval/test_deferred.py`: create → pending → pending again (same id) → decide → consume → second request creates a new record; denied decision returned; emits no events |
| 5 | Package exports | `approval/__init__.py` | import test |
| 6 | `LoggingHook` | `hooks/builtin/logging.py` | `tests/hooks/builtin/test_logging.py`: redaction table (nested, case-insensitive, lists), confirm tools never log args, one line per start / finish / error for tool, LLM, node; no message content |
| 7 | `BudgetHook` | `hooks/builtin/budget.py` | `test_budget.py`: stop / warn per scope run, thread (and empty thread → run), global with `RecordingLedger`; warn once per run + `budget.warning` event vs `REQUIRED_DATA`; no ledger → one WARNING; config round trip; bad config |
| 8 | `GuardrailHook` | `hooks/builtin/guardrail.py` | `test_guardrail.py`: each stage, sync / async check, reason string, `False`, bad return, unknown stage |
| 9 | `RateLimitHook` | `hooks/builtin/rate_limit.py` | `test_rate_limit.py`: bucket refill with `FakeClock`, per kind, `llm` kind, unlimited kinds, cancel refunds, shared by two loops in two threads |
| 10 | Hooks exports | `hooks/builtin/__init__.py`, `hooks/__init__.py` | import test |
| 11 | Docs + examples | `docs/concepts/{hooks,approval}.md`, `examples/{hooks,approval}.py`, `tests/test_examples.py` | example tests |
| 12 | Shared edits | `SHARED-EDITS.md` | — |

## Exit gate

`pytest -q` (0 warnings, ≥ 90 % line + branch on new modules), `ruff check .`,
`ruff format --check .`, `mypy` (strict), `mkdocs build --strict`, boundary test, no file
over 500 lines.
