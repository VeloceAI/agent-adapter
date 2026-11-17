---
title: See how parallel agents went
description: One report from a run's events shows how long parallel branches waited, which calls repeated, what the cache and shared calls saved, contested claims, and tokens per agent.
---

# See how parallel agents went

`analyse_run(events)` turns a run's events into one report: how long each parallel branch
waited for the slowest, which calls ran twice, what the cache and in-flight sharing saved,
which board claims were contested, and the calls, tokens and cost of every agent. Check it
in tests and CI, or open the **Parallel** tab of the run viewer.

## Get the report

Record the events with any sink, then pass them in. This is the end of
`examples/parallel_research.py`, which fans out to three researchers:

```python
--8<-- "examples/parallel_research.py:run"
```

```text
parallel branches: 3, slowest waited 0 ms: True
contested claims: 1
input tokens: alpha 409, beta 554, gamma 195, main 51, writer 111
```

It takes recorded dicts (from `agent-adapter ui` run files) or `Event`s alike.

## What it holds

| Key | What it holds |
|---|---|
| `joins` | each parallel section (flow superstep or agent fan-out): every branch's time, how long it waited for the slowest, and the slowest |
| `parallel_ms`, `waited_ms`, `critical_path` | totals, and the slowest branch of each section |
| `duplicates`, `duplicate_calls` | the same tool with the same arguments (needs arguments in the events: `record(executor, capture_content=True)`) |
| `saved` | calls answered by the cache, and by an identical call already running (`inflight.shared`) |
| `tokens` | model tokens used (`input`, `output`), and the tokens of replies the cache or a shared call gave (`saved_by_cache`, `saved_by_inflight`) |
| `claims` | contested claims: who held the key and who also asked |
| `agents` | model and tool calls, input and output tokens, cost, busy time, posts and claims per agent or node |
| `board` | every post and claim, in order |

Calls belong to the innermost agent tool around them, else to their flow node, else to
`main`. Input tokens count cached and cache-write input too.

## Check it in CI

```python
report = analyse_run(sink.events)
assert report["duplicate_calls"] == 0, report["duplicates"]
assert report["tokens"]["input"] < 20_000
assert all(join["waited_ms"] < 5_000 for join in report["joins"])
```

## In the run viewer

`agent-adapter ui` shows board posts and claims as points on the timeline, and the
**Parallel** tab shows the same report: a sentence for each finding (waits, repeats,
tokens saved, contested claims), one table per parallel section, repeated calls,
contested claims, a per-agent table with tokens in and out, and the board history.

## Fewer tokens for parallel agents

- Share a board and let agents claim work: a claimed search runs once.
- `LoopAgent(board="auto")` sends each agent only the news, capped at `board_chars`
  ([Share a board between agents](shared-board.md)).
- Cache model replies (`executor.model(model, cache_ttl=)`). Identical calls that run at
  the same moment can share one reply (`inflight.shared` events), counted under
  `saved_by_inflight`.
