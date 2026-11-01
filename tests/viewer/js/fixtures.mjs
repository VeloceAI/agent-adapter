// A small recorded run of a help-search flow: plan -> search (Map) -> merge -> answer -> loop or finish.
export const graph = {
  flow: "help",
  start: "plan",
  layers: 5,
  width: 1,
  nodes: [
    { id: "plan", kind: "fn", start: true, end: false, fan_out: false, fan_in: false, pause: null, tool: null, over: null, sources: [], layer: 0, order: 0, x: 0 },
    { id: "search", kind: "map", start: false, end: false, fan_out: true, fan_in: false, pause: null, tool: "item.tool", over: "plan_steps", sources: [], layer: 1, order: 0, x: 0 },
    { id: "merge", kind: "join", start: false, end: false, fan_out: false, fan_in: true, pause: null, tool: null, over: null, sources: ["search"], layer: 2, order: 0, x: 0 },
    { id: "answer", kind: "fn", start: false, end: false, fan_out: false, fan_in: false, pause: "always", tool: null, over: null, sources: [], layer: 3, order: 0, x: 0 },
    { id: "finish", kind: "fn", start: false, end: true, fan_out: false, fan_in: false, pause: null, tool: null, over: null, sources: [], layer: 4, order: 0, x: 0 },
  ],
  edges: [
    { source: "plan", target: "search", kind: "edge", label: "", parallel: false, loop: false, max_loops: null, on_exhausted: false, via: [] },
    { source: "search", target: "merge", kind: "edge", label: "", parallel: false, loop: false, max_loops: null, on_exhausted: false, via: [] },
    { source: "merge", target: "answer", kind: "edge", label: "", parallel: false, loop: false, max_loops: null, on_exhausted: false, via: [] },
    { source: "answer", target: "search", kind: "route", label: "again", parallel: false, loop: true, max_loops: 1, on_exhausted: false, via: [] },
    { source: "answer", target: "finish", kind: "route", label: "done", parallel: false, loop: false, max_loops: 1, on_exhausted: true, via: [] },
  ],
};

let ts = 100;
function ev(type, data, call_id = "", parent_id = "") {
  ts += 0.01;
  return { kind: "event", type, run_id: "run_1", call_id, parent_id, ts, data };
}
const node = (type, name, visit, extra = {}) => ev(type, { flow: "help", node: name, visit, ...extra }, `help:${name}:${visit}`);

export const events = [
  ev("flow.started", { flow: "help" }), // 0
  node("node.started", "plan", 1), // 1
  node("node.finished", "plan", 1, { changed: ["plan_steps"], updates: { plan_steps: [1, 2] } }), // 2
  node("node.started", "search", 1), // 3
  ev("tool.started", { tool: "faq", kind: "retrieval" }, "help:search:1:0", "help:search:1"), // 4
  ev("tool.started", { tool: "forum", kind: "retrieval" }, "help:search:1:1", "help:search:1"), // 5
  ev("tool.finished", { tool: "faq", latency_ms: 5, attempts: 1, cached: false }, "help:search:1:0", "help:search:1"), // 6
  ev("tool.failed", { tool: "forum", error_kind: "tool-timeout", attempts: 1, latency_ms: 9 }, "help:search:1:1", "help:search:1"), // 7
  node("node.finished", "search", 1, { changed: ["hits"] }), // 8
  node("node.started", "merge", 1), // 9
  node("node.finished", "merge", 1, { changed: [] }), // 10
  node("node.started", "answer", 1), // 11
  ev("llm.started", { provider: "local", model: "m", stage: "llm", stream: false }, "help:answer:1:llm", "help:answer:1"), // 12
  ev("llm.finished", { provider: "local", model: "m", stage: "llm", latency_ms: 2, usage: { input_tokens: 120, output_tokens: 12 }, cost_usd: 0.002, finish_reason: "stop" }, "help:answer:1:llm", "help:answer:1"), // 13
  node("node.finished", "answer", 1, { changed: ["answer", "rounds"], updates: { answer: "first", rounds: 1 } }), // 14
  node("node.started", "search", 2), // 15
  ev("tool.started", { tool: "faq", kind: "retrieval" }, "help:search:2:0", "help:search:2"), // 16
  ev("tool.finished", { tool: "faq", latency_ms: 1, attempts: 1, cached: true }, "help:search:2:0", "help:search:2"), // 17
  node("node.finished", "search", 2, { changed: ["hits"] }), // 18
  node("node.started", "merge", 2), // 19
  node("node.finished", "merge", 2, { changed: [] }), // 20
  node("node.started", "answer", 2), // 21
  node("node.finished", "answer", 2, { changed: ["answer"], updates: { answer: "second" } }), // 22
  node("node.started", "finish", 1), // 23
  node("node.finished", "finish", 1, { changed: ["answer"], updates: { answer: "second" } }), // 24
  ev("flow.finished", { flow: "help", status: "done", steps: 8 }), // 25
];

function span(id, family, name, nodeName, first, last, status, extra = {}) {
  return { id, family, name, node: nodeName, call_id: id.split(":").slice(1).join(":"), parent_id: extra.parent_id || "", events: [first, last], status, cost_usd: extra.cost_usd ?? null, error_kind: extra.error_kind || "", start: events[first].ts, end: events[last].ts, lane: 0 };
}

export const spans = [
  span("tool:help:search:1:0", "tool", "faq", "search", 4, 6, "ok", { parent_id: "help:search:1" }),
  span("tool:help:search:1:1", "tool", "forum", "search", 5, 7, "error", { parent_id: "help:search:1", error_kind: "tool-timeout" }),
  span("llm:help:answer:1:llm", "llm", "local/m", "answer", 12, 13, "ok", { parent_id: "help:answer:1", cost_usd: 0.002 }),
  span("tool:help:search:2:0", "tool", "faq", "search", 16, 17, "cached", { parent_id: "help:search:2" }),
  span("node:help:plan:1", "node", "plan", "plan", 1, 2, "ok"),
];

export function run(extra = {}) {
  return { summary: { key: "k1", run_id: "run_1", status: "ok", duration_ms: 250, cost_usd: 0.002, errors: 1, events: events.length }, events, spans, graphs: { help: graph }, ...extra };
}
