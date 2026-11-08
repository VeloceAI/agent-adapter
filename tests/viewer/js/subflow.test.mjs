// Sub-flow nodes: events kept to their scope, Mermaid subgraphs, and opening a sub-flow in the view.
import assert from "node:assert/strict";
import { test } from "node:test";

import { event, loadDom } from "./fakedom.mjs";

globalThis.ResizeObserver = class {
  observe() {}
  disconnect() {}
};

const { AA, document } = loadDom("data.js", "graphlive.js", "graphnav.js", "graph.js", "graphbar.js");
const live = AA.graphLive;

const child = {
  flow: "pack", start: "count", layers: 2,
  nodes: [
    { id: "count", kind: "fn", layer: 0, x: 0, start: true },
    { id: "check", kind: "fn", layer: 1, x: 0, end: true, pause: "always" },
  ],
  edges: [{ source: "count", target: "check", kind: "edge", label: "" }],
};
const graph = {
  flow: "shop", start: "start", layers: 2,
  nodes: [
    { id: "start", kind: "fn", layer: 0, x: 0, start: true },
    { id: "sub", kind: "subflow", layer: 1, x: 0, end: true, subflow: child },
  ],
  edges: [{ source: "start", target: "sub", kind: "edge", label: "" }],
};
const ev = (type, call, parent, data) => ({ kind: "event", type: type, ts: 1, call_id: call, parent_id: parent, data: data });
const events = [
  ev("flow.started", "", "", { flow: "shop" }),
  ev("node.started", "shop:start:1", "", { flow: "shop", node: "start", visit: 1 }),
  ev("node.finished", "shop:start:1", "", { flow: "shop", node: "start", visit: 1, changed: [] }),
  ev("node.started", "shop:sub:1", "", { flow: "shop", node: "sub", visit: 1 }),
  ev("flow.started", "shop:sub:1/pack", "shop:sub:1", { flow: "pack", parent_flow: "shop", parent_node: "sub" }),
  ev("node.started", "shop:sub:1/pack:count:1", "shop:sub:1", { flow: "pack", node: "count", visit: 1 }),
  ev("node.finished", "shop:sub:1/pack:count:1", "shop:sub:1", { flow: "pack", node: "count", visit: 1, changed: ["qty"] }),
  // the same child flow under another parent node: never shown in the "sub" view
  ev("node.started", "shop:other:1/pack:count:1", "shop:other:1", { flow: "pack", node: "count", visit: 1 }),
  ev("node.failed", "shop:other:1/pack:count:1", "shop:other:1", { flow: "pack", node: "count", visit: 1, error_kind: "x" }),
  ev("node.paused", "shop:sub:1/pack:check:1", "shop:sub:1", { flow: "pack", node: "check", approval_id: "a" }),
];
const run = (extra) => Object.assign({ key: "r1", events: events, spans: [], summary: { status: "paused" } }, extra || {});
const opened = Object.assign({}, child, { parent: { flow: "shop", node: "sub" } });

test("ids belong to the top level or to one SubFlow node's visits", () => {
  assert.equal(live.inScope(graph, "shop:start:1"), true);
  assert.equal(live.inScope(graph, "shop:sub:1/pack:count:1"), false);
  assert.equal(live.inScope(opened, "shop:sub:1/pack:count:1"), true);
  assert.equal(live.inScope(opened, "shop:other:1/pack:count:1"), false);
  const deeper = { flow: "box", parent: { flow: "pack", node: "inner" } };
  assert.equal(live.inScope(deeper, "shop:sub:1/pack:inner:2/box:x:1"), true);
  assert.equal(live.inScope(deeper, "shop:sub:1/box:x:1"), false);
});

test("an opened sub-flow shows only its own node events", () => {
  const snap = live.snapshot(opened, run());
  assert.equal(snap.nodes.count.status, "ok");
  assert.equal(snap.nodes.count.errors, 0); // the other node's failed run is not this one
  assert.equal(snap.nodes.check.status, "paused");
  assert.deepEqual(live.replaySteps(events, "pack", opened), [5, 6, 9]);
  assert.equal(live.replaySteps(events, "pack").length, 5); // no graph: by flow name only
  const top = live.snapshot(graph, run());
  assert.equal(top.nodes.sub.status, "running");
});

test("a sub-flow's flow.finished stops only its own view's running nodes", () => {
  const ended = events.slice(0, 6).concat([
    ev("flow.finished", "shop:sub:1/pack", "shop:sub:1", { flow: "pack", status: "cancelled", steps: 0, parent_flow: "shop", parent_node: "sub" }),
  ]);
  assert.equal(live.snapshot(opened, run({ events: ended })).nodes.count.status, "stopped");
  assert.equal(live.snapshot(graph, run({ events: ended })).nodes.sub.status, "running");
});

test("calls are counted only under the sub-flow visit they ran in", () => {
  const spans = [
    { id: "t1", family: "tool", node: "count", call_id: "shop:sub:1/pack:count:1:0", parent_id: "shop:sub:1/pack:count:1", events: [6], status: "ok" },
    { id: "t2", family: "tool", node: "count", call_id: "shop:other:1/pack:count:1:0", parent_id: "shop:other:1/pack:count:1", events: [7], status: "ok" },
  ];
  assert.equal(live.snapshot(opened, run({ spans: spans })).nodes.count.calls, 1);
});

test("Mermaid draws a sub-flow as a subgraph with its own ids", () => {
  const lines = live.mermaid(graph).split("\n");
  assert.deepEqual(lines.slice(2, 9), [
    '  n_start["start"]',
    '  subgraph n_sub["sub<br/>sub-flow: pack"]',
    '    s1__count["count"]',
    '    s1__check["check<br/>pause gate"]:::pause',
    "    s1__count --> s1__check",
    "  end",
    "  n_start --> n_sub",
  ]);
});

function view(onOpen, onEscape) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const opts = { onOpen: onOpen || (() => {}), onEscape: onEscape || (() => {}) };
  const graphView = AA.createGraph(container, opts);
  graphView.update(graph, run());
  return { container, graphView, stage: container.querySelector(".gv-stage") };
}

const names = (container) => container.querySelectorAll(".g-name").map((n) => n.textContent);

test("the open glyph shows a sub-flow in place and up goes back", () => {
  let escaped = 0;
  const { container, graphView } = view(null, () => { escaped += 1; });
  assert.deepEqual(names(container), ["start", "sub"]);
  const crumb = container.querySelector(".gv-crumb");
  assert.equal(crumb.hidden, true);
  container.querySelector(".g-open").click();
  assert.deepEqual(names(container), ["count", "check"]);
  assert.equal(graphView.shown().flow, "pack");
  assert.deepEqual(graphView.shown().parent, { flow: "shop", node: "sub" });
  assert.equal(crumb.hidden, false);
  assert.match(crumb.textContent, /shop › sub \(pack\)/);
  assert.equal(graphView.snapshot().nodes.check.status, "paused");
  assert.equal(escaped, 1); // a node panel of the other graph is closed
  crumb.querySelector("button").click();
  assert.deepEqual(names(container), ["start", "sub"]);
  assert.equal(graphView.shown().flow, "shop");
});

test("o opens the focused sub-flow and u goes back up", () => {
  const { container, graphView, stage } = view();
  const sub = container.querySelectorAll(".g-node").find((n) => n.getAttribute("data-id") === "sub");
  sub.dispatchEvent(event("focus"));
  stage.dispatchEvent(event("keydown", { key: "o" }));
  assert.equal(graphView.shown().flow, "pack");
  stage.dispatchEvent(event("keydown", { key: "o" })); // "count" is no sub-flow: nothing happens
  assert.equal(graphView.shown().flow, "pack");
  stage.dispatchEvent(event("keydown", { key: "u" }));
  assert.equal(graphView.shown().flow, "shop");
});

test("a new graph without that sub-flow node falls back to the top", () => {
  const { container, graphView } = view();
  container.querySelector(".g-open").click();
  const plain = { flow: "shop", start: "start", layers: 1, nodes: [{ id: "start", kind: "fn", layer: 0, x: 0 }], edges: [] };
  graphView.update(plain, run());
  assert.equal(graphView.shown().flow, "shop");
  assert.equal(container.querySelector(".gv-crumb").hidden, true);
});

test("a sub-flow node says what it holds", () => {
  const { container } = view();
  const sub = container.querySelectorAll(".g-node").find((n) => n.getAttribute("data-id") === "sub");
  assert.match(sub.querySelector(".sub").textContent, /sub-flow: pack/);
  assert.ok(sub.querySelector(".subframe"));
  assert.match(sub.getAttribute("aria-label"), /sub-flow pack \(o opens it\)/);
});
