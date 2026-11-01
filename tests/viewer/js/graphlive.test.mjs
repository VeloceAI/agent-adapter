// graphlive.js: node status, visits, calls, state, edge counts, Map progress, replay, paths, Mermaid.
import assert from "node:assert/strict";
import { test } from "node:test";

import { events, graph, run } from "./fixtures.mjs";
import { load, plain } from "./load.mjs";

const AA = load("graphlive.js");
const live = AA.graphLive;

test("replay steps are the flow's node events in order", () => {
  const steps = live.replaySteps(events, "help");
  assert.equal(steps.length, 16);
  assert.deepEqual(steps.slice(0, 3), [1, 2, 3]);
  assert.deepEqual(live.replaySteps(events, "other"), []);
  assert.deepEqual(live.replaySteps(undefined, "help"), []);
});

test("a live snapshot counts visits, calls, errors, cost and changed state per node", () => {
  const snap = live.snapshot(graph, run());
  const search = snap.nodes.search;
  assert.equal(search.status, "ok");
  assert.equal(search.visits, 2);
  assert.equal(search.calls, 3);
  assert.equal(search.errors, 1);
  assert.deepEqual(plain(search.errorKinds), ["tool-timeout"]);
  assert.deepEqual(plain(search.items), { done: 1, total: 1 }); // the latest visit's items
  assert.equal(snap.nodes.answer.cost, 0.002);
  assert.deepEqual(plain(snap.nodes.answer.changed), ["answer", "rounds"]);
  assert.deepEqual(plain(snap.nodes.answer.updates), { answer: "second" });
  assert.equal(snap.nodes.answer.visitList.length, 2);
  assert.equal(snap.nodes.plan.items, null);
  assert.equal(snap.ts, events[events.length - 1].ts);
});

test("edge counts follow the order nodes ran in, loops included", () => {
  const counts = live.snapshot(graph, run()).edges;
  const key = live.edgeKey;
  assert.deepEqual(plain(counts), {
    [key("plan", "search")]: 1,
    [key("search", "merge")]: 2,
    [key("merge", "answer")]: 2,
    [key("answer", "search")]: 1,
    [key("answer", "finish")]: 1,
  });
});

test("a snapshot at a replay step shows only what had happened by then", () => {
  const snap = live.snapshot(graph, run(), 5); // second item started, nothing finished
  assert.equal(snap.nodes.plan.status, "ok");
  assert.equal(snap.nodes.search.status, "running");
  assert.equal(snap.nodes.merge.status, "pending");
  assert.deepEqual(plain(snap.nodes.search.items), { done: 0, total: 2 });
  assert.equal(snap.nodes.search.errors, 0);
  assert.deepEqual(plain(snap.edges), { [live.edgeKey("plan", "search")]: 1 });
  const before = live.snapshot(graph, run(), -1);
  assert.ok(Object.values(before.nodes).every((n) => n.status === "pending" && n.visits === 0));
});

test("failed and paused node events set those statuses", () => {
  const extra = [
    { type: "node.started", ts: 1, data: { flow: "help", node: "plan", visit: 1 } },
    { type: "node.failed", ts: 2, data: { flow: "help", node: "plan", visit: 1, error_kind: "flow-error" } },
    { type: "node.paused", ts: 3, data: { flow: "help", node: "answer", approval_id: "apr_1" } },
    { type: "node.finished", ts: 4, data: { flow: "elsewhere", node: "merge", visit: 1 } },
  ];
  const snap = live.snapshot(graph, { events: extra, spans: [] });
  assert.equal(snap.nodes.plan.status, "error");
  assert.deepEqual(plain(snap.nodes.plan.errorKinds), ["flow-error"]);
  assert.equal(snap.nodes.answer.status, "paused");
  assert.equal(snap.nodes.merge.status, "pending"); // another flow's node of the same name
  assert.equal(live.snapshot(graph, null).nodes.plan.status, "pending");
});

test("paths reach ancestors and descendants over forward edges only", () => {
  const found = live.paths(graph, "merge");
  assert.deepEqual(plain(found.nodes), ["answer", "finish", "merge", "plan", "search"]);
  assert.ok(!found.edges.has(live.edgeKey("answer", "search")));
  assert.ok(found.edges.has(live.edgeKey("plan", "search")));
  assert.deepEqual(plain(live.paths(graph, "finish").nodes), ["answer", "finish", "merge", "plan", "search"]);
});

test("search matches node names, tools and kinds, case-insensitively", () => {
  assert.deepEqual(live.search(graph, "ANS"), ["answer"]);
  assert.deepEqual(live.search(graph, "item.tool"), ["search"]);
  assert.deepEqual(live.search(graph, "join"), ["merge"]);
  assert.deepEqual(live.search(graph, "  "), []);
});

test("mermaid text marks shapes, loops, gates and escapes labels", () => {
  const text = live.mermaid(graph);
  assert.ok(text.startsWith("flowchart TD\n  aa_start((start)) --> n_plan\n"));
  assert.ok(text.includes('n_search[["search<br/>fan-out over plan_steps<br/>tool: item.tool"]]'));
  assert.ok(text.includes('n_merge{{"merge<br/>fan-in: search"}}'));
  assert.ok(text.includes('n_answer["answer<br/>pause gate"]:::pause'));
  assert.ok(text.includes('n_answer -.->|"again (max 1)"| n_search'));
  assert.ok(text.includes('n_answer -->|"done"| n_finish'));
  assert.ok(text.endsWith("n_finish --> aa_end((end))\n  classDef pause stroke-dasharray: 5 5\n"));
  const odd = { flow: "q", start: "a", nodes: [{ id: "a", kind: "call", tool: 'say "<hi>" & bye', end: true }], edges: [] };
  assert.ok(live.mermaid(odd).includes('tool: say #quot;#lt;hi#gt;#quot; #amp; bye"'));
});

test("every status has an icon, so the legend never relies on colour alone", () => {
  assert.deepEqual(live.STATUSES.map((s) => s.key), ["pending", "running", "ok", "error", "paused", "stopped"]);
  assert.ok(live.STATUSES.every((s) => s.icon && s.label));
});

test("nodes still running when their flow or run ended show as stopped", () => {
  const ev = (type, data) => ({ type, ts: 1, data });
  const cancelled = [
    ev("node.started", { flow: "help", node: "plan", visit: 1 }),
    ev("node.started", { flow: "help", node: "merge", visit: 1 }),
    ev("node.failed", { flow: "help", node: "merge", visit: 1, error_kind: "flow-error" }),
    ev("flow.finished", { flow: "help", status: "failed", steps: 1 }),
  ];
  const snap = live.snapshot(graph, { events: cancelled, spans: [] });
  assert.equal(snap.nodes.plan.status, "stopped");
  assert.equal(snap.nodes.merge.status, "error");
  assert.equal(live.snapshot(graph, { events: cancelled, spans: [] }, 2).nodes.plan.status, "running"); // replay before the end
  const killed = { events: cancelled.slice(0, 1), spans: [], summary: { key: "k", status: "unfinished" } };
  assert.equal(live.snapshot(graph, killed).nodes.plan.status, "stopped");
  assert.equal(live.snapshot(graph, { ...killed, summary: { key: "k", status: "running" } }).nodes.plan.status, "running");
  assert.ok(live.STATUSES.some((s) => s.key === "stopped" && s.icon));
});

test("node and tool names like Object members are plain keys", () => {
  const odd = {
    flow: "f", start: "constructor",
    nodes: [{ id: "constructor", kind: "fn" }, { id: "__proto__", kind: "fn" }],
    edges: [{ source: "constructor", target: "__proto__" }],
  };
  const events = [
    { type: "node.started", ts: 1, data: { flow: "f", node: "toString", visit: 1 } }, // not in this graph
    { type: "node.started", ts: 2, data: { flow: "f", node: "constructor", visit: 1 } },
    { type: "node.finished", ts: 3, data: { flow: "f", node: "constructor", visit: 1 } },
    { type: "node.started", ts: 4, data: { flow: "f", node: "__proto__", visit: 1 } },
  ];
  const snap = live.snapshot(odd, { events: events, spans: [{ family: "tool", node: "hasOwnProperty", events: [1], status: "ok" }] });
  assert.deepEqual(Object.keys(snap.nodes).sort(), ["__proto__", "constructor"]);
  assert.equal(snap.nodes.constructor.status, "ok");
  assert.equal(snap.nodes["__proto__"].status, "running");
  assert.equal(snap.edges[live.edgeKey("constructor", "__proto__")], 1);
});
