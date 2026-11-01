# Flow pause and resume

A flow pauses at a `pause` gate or when a tool inside a node needs an approval that is not
decided yet. `FlowRunner.resume(flow, checkpoint, decision)` continues the same run.

```mermaid
sequenceDiagram
  participant N as paused node
  participant E as Executor
  participant P as approval port
  N->>E: call A (finished, journaled)
  N->>E: call B (confirm)
  E->>P: request B
  P-->>E: ApprovalPending
  Note over N: status="paused", checkpoint sealed
  Note over N: resume(decision for B)
  N->>E: call A again
  E-->>N: replayed result (A does not run)
  N->>E: call B again
  E-->>N: decision answers B once; B runs
```

## What a paused result holds

| Field | Holds |
|---|---|
| `status` | `"paused"` |
| `pending` | the `ApprovalRequest`: tool (or `node:<flow>.<node>` for a gate), run and call ids; `pending.arguments` are the requested arguments (reprs redact them) |
| `checkpoint` | where to continue: state, frontier, counters, and sealed private parts |

## What resume does

| On re-run | Behaviour |
|---|---|
| executor calls the node already finished | **replayed** from the node's journal, keyed by tool, the canonical form of the validated arguments the caller passed (unset fields left out at every depth, so a `default_factory` id, even in a nested model, still replays) and order: no execution, run guards, ledger entries or events |
| finished or skipped `Map` items | kept; not run again |
| calls inside a `Map(fn=...)` item | replayed from that item's own journal (keyed by item index) while the item is unfinished |
| a nested flow run inside the node | its node visits get child journals inside the outer node's journal: their calls replay too |
| a replayed failure | raises an exception rebuilt from the result's `ErrorInfo` (same `AdapterError` class, message and attributes) |
| approval requests | `decision`, and every earlier answer for the node, each answer once a request with the same run, tool and arguments (and call id for `Call` / `Map`); anything else asks the port |
| a denied tool decision | `ApprovalDenied` in the node: a `FlowError`, or a skipped `Map` item |
| tasks that outlive the node | get no answer and no journal |
| other pending records of the pause | a `Call` / `Map(tool=)` item record stays pending: its call id repeats, so the port finds it again; any other (a node function's or `Map(fn=)` item's call, which gets a new call id) is denied and consumed with reason `superseded` and asked again on resume |

A node needing two approvals finishes after two resumes. LangGraph re-runs a resumed
node's calls instead of replaying them; the parity case `replay_two_confirms` records this
as an intended difference.

## Safety

| Rule | Effect |
|---|---|
| sealed | every field, in a strict type-tagged canonical form, is covered by HMAC-SHA256 with the runner's `checkpoint_key`; a changed or foreign checkpoint (or a MAC that is not text) raises `FlowError` (`checkpoint-tampered`) and runs nothing |
| graph | the checkpoint holds a digest of the graph: name, state model, start, ends, edges, and per node its name, kind, gate and action (`module:qualname` of a function, gate payload or route function; a `Call`'s tool, `into` and static `args` in canonical form; a `Map`'s tool and options); a changed graph is refused before the approval is spent |
| key | default: a random per-process key, so checkpoints resume only in that process; pass the same `checkpoint_key` (bytes, 16+) to every runner that should resume them |
| once | a second resume raises `ApprovalAlreadyDecided` and runs nothing |
| store | when the port has a `store` (e.g. `DeferredApproval`), the pending record is decided and consumed; every earlier answer that store holds must be a consumed record with that decision (one it does not hold belongs to another port) |
| own store | a node that runs tools on its own executor with its own `DeferredApproval`: the answer's record in that store is decided and consumed there; a record decided otherwise denies the call (`ApprovalAlreadyDecided`) |
| memory | other ports rely on a process-wide memory of resumed approvals (bounded, 24 h); after an entry is evicted only a store can still refuse |
| no tool | a pending that names no tool is refused |

A function counts by its `module:qualname` only: a changed body under the same name, or
another lambda in the same scope (same `<lambda>` qualname), is not detected. Answers still
fail closed: one answers only a request with the same run, tool and arguments.

A refused settle (the store holds another decision, or it is unreachable) does not spend
the checkpoint: resume again with the store's decision. `ToolResult.to_dict()` and pickled
`ApprovalPending` drop the arguments, so a pending rebuilt from them asks the port again.
A pending result pickles and deep-copies; the port's store stays in the process.

## Sealable state

Only these values can be sealed; the encoding never confuses types.

| Value | Canonical form |
|---|---|
| `None`, `bool`, `int`, finite `float`, `str`, `list` | as is; an int past ~600 digits is tagged as hex (up to 1 MiB) |
| `dict` with str keys not starting with `$` | as is; other dicts become a sorted pair list |
| `bytes`, `bytearray` | base64, tagged (invalid UTF-8 is fine) |
| `tuple`, `set`, `frozenset`, NaN / infinity | tagged (sets sorted) |
| `date`, `timedelta`, `Decimal`, `UUID`, `Enum` | tagged |
| `datetime`, `time` | tagged with `fold` and the zone: `datetime.timezone` as offset seconds (and its name), `zoneinfo.ZoneInfo` by key; other `tzinfo` types are not sealable |
| pydantic models | tagged class name and fields (recursively); private attributes (`_x`) are **not** sealed |
| anything else | `FlowRunner(seal_encoder=fn)`: `fn(value)` returns a sealable stand-in |

A flow that never pauses may hold anything. When a run would pause, everything the
checkpoint needs is sealed first; if any part cannot be, the pause fails closed with
`FlowError` (`state-not-sealable`, naming the path):

| Part | Check |
|---|---|
| gate | the state and the gate payload are checked **before** the port is asked: no record is made |
| tool approval | every pending record the paused visits made (concurrent calls, `Map` items) is denied and consumed in the store its port reaches (the runner's executor or the node's own); with no store, an ERROR log names the approval id and an `approval.decided` event (`withdrawn=False`) is emitted |
| sibling updates, `Map` item values | not dropped: they fail the pause the same way (nothing runs again) |
| call results | a result that cannot be journaled leaves a marker (the call ran); the pause fails naming the call. Pass `seal_encoder` and `seal_decoder` to replay custom results |

## Storing checkpoints

| Form | Use |
|---|---|
| in memory | pass the checkpoint back to `resume` |
| `runner.dump_checkpoint(cp)` / `runner.load_checkpoint(flow, text)` | JSON text for any storage; the MAC and graph are checked before anything is built; custom values need `seal_decoder=fn(type_name, value)` to load |
| pickle | trusted storage only: unpickling runs code before the MAC is checked |

Loading builds only the classes the state model's field annotations reach (through
`Optional`, unions, lists, dicts, tuples and nested models) and their subclasses already
loaded (a `Dog` in an `Animal` field; a bare `BaseModel` or `Enum` annotation opens
nothing), plus `ApprovalRequest` and `ApprovalDecision`; a replayed result may also hold classes its tool's return annotation
reaches. Any other class name goes to `seal_decoder` (the value is the decoded fields or
enum value), so the decoder **must refuse names it does not know**; without one it is
refused before any object is built, and `dump_checkpoint` refuses such a checkpoint up front.

## Limits

| Limit | Use instead |
|---|---|
| a port without a store, resumed in another process or after the memory evicted the entry, can resume twice | `DeferredApproval` (its store refuses) |
| pickled checkpoints from untrusted storage | `dump_checkpoint` / `load_checkpoint` |
| checkpoints grow with journaled results (measured: 200 results of 10 KB, about 2 MB as pickle or JSON) | keep large data out of tool results, or in external storage |
| custom types in state or results | `seal_encoder` (and `seal_decoder` to load JSON or replay them) |
