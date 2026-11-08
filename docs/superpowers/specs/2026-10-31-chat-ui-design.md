# Chat UI, better graph, and docs — design

Status: approved by the user 2026-10-31 ("do what is right and best"), including the
security change in §6. Builds on the run viewer (`agent_adapter/viewer`, see
`docs/how-to/see-your-runs.md`).

## 1. Goal

A local dev UI like ADK's `adk web`, and better, for any framework or none:

- chat with an agent, see its tools, graph, timeline, state and cost live;
- a graph you can navigate (zoom, pan, minimap, click through, replay);
- docs that show the sample flow and how to use and navigate it.

"Better than ADK" means these specific gaps (from research of adk-web, 2026-10):
no cost view, no parallel timeline, no click-through from graph to events, state view
that goes stale (adk-web #312), no seeding state (#8), dev server without auth (CVE-2026-4810),
ADK-only. Out of scope: live audio/video, a visual agent builder, generative UI (A2UI).

## 2. Hard rules (unchanged from the viewer)

- Standard library only on the Python side; no new runtime dependency. No external URL in
  any page (CSP `default-src 'none'`; only `blob:` is added to `img-src`).
- Binds 127.0.0.1 only; `#token=` link; `X-AA-Token` on every `/api/` call; every POST
  checks Origin, `X-AA-CSRF` and `application/json` (uploads: see §5.4).
- Never render agent, tool or user text as HTML. Build DOM nodes; links only `http(s):`
  with `rel="noopener noreferrer"`, `target="_blank"`.
- 500-line cap per file (Python, JS, CSS); split by responsibility. Short docstrings
  (project CLAUDE.md §15). Errors from `agent_adapter.errors`; structured logging; no print.
- Export (`--export`) keeps working offline; chat is not part of an export.
- Boundary rules: `agent_adapter.viewer` may import exec, flow, models, tracing, approval,
  optimiser (cases only); nothing imports viewer except cli.

## 3. How it starts

```bash
agent-adapter ui                                         # viewer only, as today
agent-adapter ui --model app:model --tools app:registry  # chat via the built-in loop
agent-adapter ui --agent app:chat                        # chat with any agent
```

Optional: `--executor app:executor` (built-in mode uses it instead of making one),
`--system "text"`, `--max-steps 8`, `--turn-timeout 300`, `--approval-timeout 600`,
`--reviewer NAME` (default `local`), `--chat-dir DIR` (default `.agent_adapter/chats`; holds
sessions, uploads and `eval_cases.jsonl`). Targets are `module.path:attribute`, checked by regex
`^[A-Za-z_][\w.]*:[A-Za-z_]\w*$`; never a file path. A bad target is a `ConfigError` naming it.

The same from Python, so no import strings are needed:

```python
from agent_adapter.viewer import serve

serve(agent=chat)  # or serve(model=model, registry=registry)
```

`serve(...)` blocks, prints the link, and accepts the same options as the CLI (`registry=`
is the CLI's `--tools`)
(`folder=".agent_adapter/runs"`, `port=8780`, `open_browser=True`). It records the chat
executor's runs to `folder` (via `record()`), so the Runs view shows them too.

## 4. The agent contract (`--agent`)

```python
async def chat(message: ChatMessage, session: ChatSession) -> str | AsyncIterator[str]: ...
```

- A sync function is accepted (run in a thread). Returning an async iterator streams text
  chunks. Any exception ends the turn as `failed` with the error kind and message
  (an `AdapterError`'s text; other exceptions show only the type name, like `tool-failed`).
- `ChatMessage`: `text`, `attachments: tuple[Attachment, ...]`.
  `Attachment`: `name`, `media_type`, `size`, `path` (a local file), `read_bytes()`, `read_text()`.
- `ChatSession`: `id`, `history: tuple[Message, ...]` (earlier user/assistant turns),
  `state: dict[str, Any]` (mutable; the UI shows it and can edit it before a turn),
  `executor: Executor` (the host's; approvals reach the UI), `run_id` of this turn.
- Tool calls show in the UI only when tools run through an executor that records
  (`session.executor`, or one the user passed to `record()`). Docs say so plainly.
- Public names live in `agent_adapter.viewer` (`serve`, `ChatMessage`, `ChatSession`,
  `Attachment`, `ChatAgent` protocol). Docs show 5-line wrappers for a plain function,
  LangGraph, ADK, CrewAI and a FastAPI/HTTP backend.

Built-in mode wraps `aa.run_tool_loop(model, registry, history + message, executor=...)`.
Text attachments (UTF-8, ≤ 64 KB) are inlined into the user message; other types are
refused with a message pointing at `--agent` (messages are text-only; no
provider-specific code). Text streams when the model port has `stream()`; otherwise the
answer arrives as one chunk. Adding streaming to the loop is allowed as an optional,
provider-neutral parameter of `run_tool_loop` built on the existing `StreamEvent`.

## 5. Server side

### 5.1 Layout

`agent_adapter/viewer/chat/` package: `contract.py` (public types), `targets.py`
(load `module:attr`), `builtin.py` (built-in agent), `host.py` (turns on a background
`LoopRunner` loop: start, stream queue, stop, timeout), `sessions.py` (store),
`approvals.py` (interactive desk), `files.py` (uploads), `routes.py` (HTTP handlers).
`server.py` dispatches `/api/chat/` to `routes.py`; it must stay under 500 lines.

### 5.2 Endpoints (all need the token; POSTs need the §2 guards)

| Method, path | Does |
|---|---|
| `GET /api/chat` | `{enabled, mode: "builtin"|"agent", agent_name}` |
| `GET/POST /api/chat/sessions` | list; create `{title?, state?}` |
| `GET /api/chat/sessions/<sid>` | messages, turns, state, branches |
| `POST /api/chat/sessions/<sid>/state` | replace state before the next turn (JSON object ≤ 64 KB) |
| `POST /api/chat/sessions/<sid>/turns` | `{text, attachment_ids?, edit_of?}` → `{turn_id, run_id}`; `edit_of` forks a branch session |
| `GET /api/chat/turns/<tid>/stream?since=N` | SSE: `text`, `tool`, `approval`, `state`, `usage`, `done`, `error`; heartbeat comment every 5 s |
| `POST /api/chat/turns/<tid>/stop` | cancel the turn; pending approvals in it are denied |
| `POST /api/chat/turns/<tid>/replay` | run the same input on the same prior history in a new branch; returns the new turn |
| `POST /api/chat/approvals/<aid>/decide` | `{approved, reason?}`; one use, bound to its turn and call |
| `POST /api/chat/turns/<tid>/eval-case` | `{id, expected, tags?}` → appends to `<chat-dir>/eval_cases.jsonl` (loadable by `optimiser.cases.load_cases`) |
| `POST /api/chat/files` | upload (§5.4) → `{attachment_id, name, size}` |
| `GET /api/chat/files/<fid>` | the file, `Content-Disposition: attachment`, `nosniff` |

The page reads SSE with `fetch()` + a stream reader (EventSource cannot send headers).
Turns: one running turn per session (409 otherwise); `--turn-timeout` ends a turn as
`timed-out`. Body caps: JSON 64 KB; text ≤ 32 KB.

### 5.3 Sessions

In memory, saved as `<chat-dir>/<sid>.json` (atomic replace), loaded on start.
Each turn keeps: input, answer, status, run_id, started/ended, tokens, cost, and tool
call summaries (name, status, ms; args/output cut to 2 KB with secret-named keys masked by
the existing scrubber). The docs say chats are stored locally in the chat dir.

### 5.4 Uploads

`POST /api/chat/files` takes `application/octet-stream` with `X-AA-Filename` (Origin,
CSRF, token still checked); ≤ 10 MB per file, ≤ 50 MB per session. Allowed media types:
text/* , application/json, image/png|jpeg|gif|webp, application/pdf (sniffed from the
first bytes, not trusted from the header). Stored under `<chat-dir>/<sid>/files/<random>`;
the original name is kept only as display text (sanitised). No path from the client is
ever used to build a file path.

### 5.5 Approvals in chat

The host's executor uses an interactive `CallbackApproval`: the call waits on a future the
UI resolves. The request shows inline in the chat (tool, args with secrets masked, reason).
A decision is one-use, keyed by approval id and bound to session, turn and call id; a
second decision gets 409. Stop, turn timeout, or `--approval-timeout` (default 600 s)
deny it. The decision is recorded with `--reviewer`. A framework's own "approved" never
counts (contract §14.4); `--executor` with its own approval port keeps that port.

### 5.6 Flow state for the State view

`node.finished` gains `changed`: the list of state field names the node updated (names
only, always). Values are sent only when the run asks for them: `FlowRunner(trace_state=True)`
adds `updates` (the changed values, JSON-safe, each cut to 2 KB). Chat mode turns it on for
its own runs. The recorder masks secret-named keys and keeps `updates` only with
`capture_content=True` or in chat mode. OTel and logging sinks get only `changed`.

## 6. Security change (approved)

The viewer never ran a tool. With chat on, the server runs the user's agent and tools in
its process. Mitigations: chat is off unless `--model/--tools` or `--agent`; start-up logs
a WARNING saying so; every turn runs through the executor (validation, approvals, limits,
ledger); `RunLimits(max_calls=50)` by default in built-in mode; the same token, Origin,
CSRF and loopback rules; uploads sniffed and size-capped; no HTML rendering. The docs say
anyone with the printed link can chat and decide approvals as `--reviewer`.

## 7. Page

Top tabs: **Chat** (when enabled) and **Runs** (today's view). Chat layout: sessions list
(left), conversation (centre), inspector (right) with tabs Graph, Timeline, Events, State,
Cost for the selected turn's run, updating live.

Conversation: streaming text with a tiny safe Markdown subset (paragraphs, bold, italic,
inline code, code blocks with copy, lists, links, tables); tool call cards (name, status,
time, expandable args/output); approval cards (Approve/Deny); Stop; edit-and-resend
(branch); per-turn menu: Replay, Compare, Save as eval case, Copy. State editor (JSON,
validated) before sending. Attach button and drag-and-drop.

Runs view gains Events (filter by node, tool, kind, text; raw JSON per event; keyboard
↑/↓), State (per step, diff of `changed`/`updates`) and Compare (pick two runs or turns:
answer, tool calls, tokens, cost, duration, and which steps differ).

## 8. Graph

- Navigation: wheel zoom at the pointer, drag to pan, `+`/`−`/`0` (fit) buttons and keys,
  minimap with the viewport box, arrow keys move focus along edges, Enter opens the node,
  Esc closes, `/` searches nodes.
- Click a node: side panel with its visits, calls, changed state, errors, cost; links to
  the same items in Timeline and Events. Hover highlights incoming and outgoing paths.
- Live status colour per node: pending, running, done, failed, paused (with a legend that
  also uses icons, not colour alone). Edge labels show times taken; Map nodes show item
  progress (`3/3`); loop edges are drawn on their own lane with the loop count.
- Replay: a step slider and play/pause replays the run's node order on the graph.
- Layout: barycentre crossing reduction in `layout.py`; curved edges; works for 100+ nodes
  (no layout over 50 ms for 200 nodes in a test).
- Export the graph as SVG and copy Mermaid (`Flow.to_mermaid()`).
- Accessible: focusable nodes with `aria-label`, works in light and dark themes.

## 9. Docs and examples

- `docs/how-to/chat-with-your-agent.md`: start (CLI and `serve`), the contract, wrappers
  for plain function, LangGraph, ADK, CrewAI, HTTP backend, approvals, files, state,
  replay/compare, eval cases, security note. A mermaid sequence diagram.
- `docs/how-to/navigate-the-graph.md`: the sample flow (`examples/viewer_fan_out.py`) with
  its Mermaid, a table of mouse/keyboard controls, what each colour/shape means, replay.
- `examples/chat_builtin.py` (fake `CallableModel`, runs offline) and
  `examples/chat_any_agent.py` (a plain function agent plus wrapper snippets).
- Update `see-your-runs.md`, mkdocs nav, the examples gallery, the AI guide (recipes for
  `serve`), README feature list, and the 1.0.0 changelog entry (unpublished).

## 10. Tests

- Python: every endpoint's happy path and refusals (token, Host, Origin, CSRF, content
  type, caps, unknown ids, 409s); agent contract (str, async iterator, sync, exception);
  stop/timeout deny pending approvals; one-use decisions; uploads (sniffing, size, names
  with `../`); sessions survive restart; eval case loads with `load_cases`; `changed`/
  `updates` on `node.finished`; built-in mode end-to-end with a fake model; layout speed.
- JS: pure logic (Markdown subset, SSE parser, pan/zoom maths, replay order, diff) in
  `tests/viewer/js/*.test.mjs` run by `node --test`, wrapped by a pytest test that skips
  without `node`. A test checks no static file contains an external URL.
- All gates: `scripts/check.py`, nox matrix, `mkdocs build --strict`.
