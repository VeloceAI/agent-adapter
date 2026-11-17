---
title: Share a board between agents
description: Agents that run in parallel or one after another see what the others are doing and have found, claim work so nobody does it twice, and pay few tokens for it.
---

# Share a board between agents

Agents of one run can share a board. Each one posts what it is doing and what it found,
claims work before doing it, and reads what the others posted. Parallel agents stop
repeating each other, and a later agent picks up where an earlier one stopped. Two
examples run offline with scripted models: `examples/parallel_research.py` (fan-out, claims,
fan-in) and `examples/board_handoff.py` (one agent after another).

## Every run has a board

`Executor.context()` puts one board in `ctx.extras["aa.board"]`. Nested calls, agent
tools, tool loops and flow nodes copy the context, so they all get the same board.
`board_of(ctx)` returns it (with no argument: the board of the call running now).

```python
from agent_adapter.board import board_of

board = board_of(ctx)
await board.post("finding", "Oslo: 3 hotels under 100 EUR", key="hotels:oslo")
claim = await board.claim("search:oslo hotels")  # the first caller wins
if claim.won:
    ...  # do the work
else:
    print(claim.holder, "already does it")
latest = await board.read(kind="finding", since=0, limit=10)
```

| Entry kind | What it says |
|---|---|
| `status` | "working on X": a summary keeps only each agent's latest status |
| `finding` | "found Y" |
| `claim` | "I'm doing X": `claim(key)` posts it for the winner |
| `handoff` | a summary for the next agent |

`by` (who posted) defaults to the agent tool or flow node that is running. Set it with
`by=`, or wrap your own code in `posting_as("name")`. `watch()` streams new entries as
they arrive, from any process the store reaches.

## Fan out, claim, fan in

Three researchers run side by side and claim their topic first; the third asks for a
topic already taken, so its search never runs. A writer then reads their findings.

```python
--8<-- "examples/parallel_research.py:agents"
```

- Agent tools of one run take turns. `agent_tool(..., parallel_safe=True)` lets them run
  side by side, and the tool calls inside them too.
- **`board_tools()`** gives the model `board_read`, `board_post` and `board_claim`. They
  only write to the run's board, so they need no approval. They still run through the
  executor, so guards, redaction and limits apply.
- Agent tools post `started: <task>` and `finished (ok)` statuses on their own, and every
  flow node posts `finished; set <fields>` when it ends. These posts never fail a run: a
  full board or a store error is logged and skipped.

## Agents see the news, not the whole board

`LoopAgent(board="auto")` (or `run_tool_loop(..., board="auto")`) adds a short note
before a model call with what the *other* agents posted since that agent's last turn:

- only new entries, newest first when they do not all fit, then a pointer to the rest
  (`+35 older: board_read(since=0)`) and the claims the others still hold;
- at most `board_chars` characters (600 by default); long texts end with
  `[cut: board_read(entry=n)]`, so the model reads one in full only if it needs it;
- nothing at all when nothing is new.

The note is a user message of fenced data, kept in the history, so each entry reaches a
model once. Its first line records how far the board was read, which also survives a
checkpoint. Measured in `tests/board/test_tokens.py`: three agents next to 50 long
entries got notes of at most 600 characters each, under a fifth of the board's text in all.

## Agents one after another

The board outlives each step of a run. A planner posts a `handoff`; the writer that runs
next finds it in its note.

```python
--8<-- "examples/board_handoff.py:handoff"
```

```text
Plan for 2 days in Oslo, museums first, 300 EUR in all
board: 45 entries, 1554 characters
writer's note: 559 characters (cap 600), 1 note
```

A flow node can read the board itself: `await board_of(ctx).read(kind="handoff", limit=1)`.

## Short reads

- `board.summary(1_500)` leaves out the running agent's own entries (`include_own=True`
  keeps them), keeps each agent's latest status only, and cuts long texts with a pointer.
- `board_read` returns the 10 newest entries by default (`limit`, up to 50), filtered by
  `kind`, `by` and `since`; `entry=n` returns one entry in full.

## Stores

| Store | Shared by |
|---|---|
| `MemoryBoardStore()` (the default) | this process; keeps the boards of the last 256 runs |
| `SQLiteBoardStore("board.db")` | every process on one host (WAL) |
| `RedisBoardStore(lambda: redis.asyncio.Redis())` | every worker and host; boards expire after `ttl_s` (7 days) |

```python
from agent_adapter.board import BoardSettings, SQLiteBoardStore

executor = aa.Executor(registry, board=SQLiteBoardStore("board.db"))
# every run of one thread shares a board; caps per board
settings = BoardSettings(store=SQLiteBoardStore("board.db"), scope="thread", max_entries=200)
executor = aa.Executor(registry, board=settings)
```

Claims and caps are atomic in every store: two processes claiming one key get one winner.

## Durable runs keep the board

A paused or checkpointed flow seals the board's entries into its checkpoint, and so does a
durable tool loop (`run_tool_loop(checkpoints=...)`). `FlowRunner.resume`, `recover` and
`recover_tool_loop` put them back first, so a run resumed in a fresh process still sees
what its agents posted before the pause or crash.

## Safety

- **Caps.** A board takes at most `max_entries` (500) entries and `max_bytes` (256 KiB); a
  post past that raises `ToolError`, so a model reads why. Text past 2,000 characters is cut.
- **Data, not instructions.** Notes, summaries and `board_read` show each entry on one line
  inside `<shared-board>` under a header saying it is data; fence tags inside an entry are
  neutralised.
- **One board per run.** Runs never see each other's boards, unless you choose
  `scope="thread"`.
- **Events carry no text.** `board.posted` has the entry id, kind, key, author and length;
  `board.claimed` has the key, who asked, who holds it and `won`.

To see how the parallel work went (waits, repeats, claims, tokens per agent), read
[See how parallel agents went](parallel-runs.md).
