---
title: Resume contract
description: Six resume properties under injected crashes, checked on FlowRunner, run_tool_loop and LangGraph.
---

# Resume contract

A durable run must survive a crash at any moment without running an effect twice or
continuing from the wrong place. An audit of agent frameworks names six properties a
resume must keep:
[Resume Means Resume](https://arxiv.org/abs/2608.03836) (S. Khan, arXiv 2608.03836,
2026). It found LangGraph at-least-once across crashes and CrewAI re-running finished
effects. We run the same six properties, under injected crashes, on our
[durable runs](durable-runs.md), and on LangGraph for comparison.

## The six properties

| Property | What must hold |
|---|---|
| Prefix continuation | A recovery goes on from the last durable position and its state, not from the start |
| Effect exactly-once | A side effect lands once across pauses, crashes and resumes |
| Fork determinism | Resumes of one paused checkpoint with different answers each end as that answer says |
| Checkpoint validity | A state that breaks the schema is refused with an error, never stored |
| Consume-once | A pause is consumed by one resume; a rival, late or repeated resume changes nothing |
| Recovery determinism | What replays and what runs again depends on the stored data alone |

We add one check of our own: a store that lost a record is refused, not run on.

## How it is tested

The workload: charge, then ship and a quote in parallel, a refund that waits for an
approval, then a node that mails a receipt and writes an audit row. The tool loop runs
the same tools from a scripted model whose reply depends only on the history.

- **Crash points.** Every store write is two crash points (before and after), and every
  tool or model call is one (just after its effect). A crash kills the worker: nothing
  more runs and the store is unreachable from it. Depth 2 crashes again at every point
  of the recovery.
- **Effects** land in a fake outside system that survives crashes and drops a repeat
  carrying the same `idempotency_key()`.
- **Calls in doubt.** When recovery raises `RecoveryNeedsDecision`, the test plays the
  operator: it reruns with `rerun_in_doubt=True`, and checks that only the reported calls
  ran again.
- **Forks** copy the stores byte for byte: one store has no fork of its own.
- **Recovery determinism** redoes every recovery on two copies of the store in fresh
  workers, one with different scheduling, and compares what ran.
- **Rival resumes** race in one event loop and in two threads on one SQLite file.

Everything is offline and deterministic: no model, no network, no clock.

## Results

Depth 2 for us, depth 1 for LangGraph 1.2.22 (`InMemorySaver`, `durability="sync"`, its
strongest setting).

| Property | FlowRunner | run_tool_loop | LangGraph |
|---|---|---|---|
| Prefix continuation | holds, 1449 cases | holds, 1214 cases | holds, 38 cases |
| Effect exactly-once | holds, 1449 cases | holds, 1214 cases | **fails**, 9 of 38 |
| Fork determinism | holds, 9 cases | holds, 6 cases | **fails** |
| Checkpoint validity | holds, 3 cases | holds, 12 cases | **fails**, 2 of 3 |
| Consume-once | holds, 6 cases | holds, 4 cases | **fails** |
| Recovery determinism | holds, 50 cases | holds, 46 cases | holds, 38 cases |
| Lost record refused (ours) | holds, 5 cases | holds, 5 cases | **fails**, 2 of 2 |

What LangGraph showed, in line with the paper:

- **Effects.** A node is the unit of recovery, so a crash in the audit call runs the whole
  node again and mails the receipt twice. A call cut short runs again without asking.
  A parallel node whose writes were already stored did not run again.
- **Forks.** Resuming the interrupt's checkpoint id with a new answer returns the first
  answer again (the paper files it as LangGraph issue #6663).
- **Validity.** With a pydantic state, a node update and `update_state` both store
  `order=7` for a `str` field, with no error (the paper's #6491 class).
- **Consume-once.** Two resumes of one interrupt at the same time both run the refund.
- **Lost records.** A dropped pending write is not noticed: its node runs again.

What the suite found in our code, now fixed:

- A resume stored with its checkpoint was checked against a memory of spent approvals
  kept by the process, so a recovery could fail in one process and pass in another.
  The stored decision now decides.
- A node that changed the state in place and returned nothing skipped validation, so an
  invalid state could reach the store. The state is now validated after every node and
  before every save.

Each run of ours either finishes with every effect once, or raises
`RecoveryNeedsDecision` before anything runs again. A tool that sends its
`idempotency_key()` to the outside system finishes exactly once even when the operator
reruns a call in doubt.

## Rerun it

```bash
python scripts/resume_contract.py
python scripts/resume_contract.py --langgraph
pytest tests/resume_contract
```

The script prints the table and exits 1 if a property fails for us. Tests use depth 1;
`AA_RESUME_DEPTH=2` adds every pair of crashes. The LangGraph test is a contract test
(`pytest -m contract`): it asserts what LangGraph does today.

## Limits

- Crashes are exceptions that freeze the store, not killed processes; the
  [kill test](durable-runs.md) covers a real process exit.
- Our runs use the memory store; consume-once also uses SQLite. LangGraph uses
  `InMemorySaver`: its SQLite saver writes on threads, so crash points are not repeatable.
- One workload. The paper's own harness, proofs and its other frameworks are not rerun here.
