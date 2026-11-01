// events.js, statepanel.js and compare.js logic: event filters, state diffs, run comparison.
import assert from "node:assert/strict";
import { test } from "node:test";

import { events, run, spans } from "./fixtures.mjs";
import { load, plain } from "./load.mjs";

const AA = load("data.js", "events.js", "statepanel.js", "compare.js");

test("events are tied to nodes through their call ids and parents", () => {
  const byCall = AA.eventFilter.nodeIndex(spans);
  assert.equal(AA.eventFilter.nodeOf(events[6], byCall), "search");
  assert.equal(AA.eventFilter.nodeOf(events[1], byCall), "plan");
  assert.equal(AA.eventFilter.nodeOf(events[0], byCall), "");
  const unknownChild = { type: "tool.started", call_id: "x", parent_id: "help:search:1:0", data: {} };
  assert.equal(AA.eventFilter.nodeOf(unknownChild, byCall), "search");
});

test("filter options list the nodes, tools and kinds present", () => {
  const found = AA.eventFilter.options(events, AA.eventFilter.nodeIndex(spans));
  assert.deepEqual(plain(found.tools), ["faq", "forum"]);
  assert.deepEqual(plain(found.kinds), ["flow", "llm", "node", "tool"]);
  assert.ok(found.nodes.includes("answer") && found.nodes.includes("search"));
});

test("filters combine node, tool, kind and text", () => {
  const byCall = AA.eventFilter.nodeIndex(spans);
  const apply = (filter) => plain(AA.eventFilter.apply(events, filter, byCall));
  assert.deepEqual(apply({ tool: "forum" }), [5, 7]);
  assert.deepEqual(apply({ node: "answer", kind: "llm" }), [12, 13]);
  assert.deepEqual(apply({ text: "TOOL-TIMEOUT" }), [7]);
  assert.deepEqual(apply({ kind: "flow" }), [0, 25]);
  assert.equal(apply({}).length, events.length);
  assert.deepEqual(apply({ text: "nothing like this" }), []);
});

test("an event's name is its tool, node, flow or model", () => {
  assert.equal(AA.eventFilter.nameOf(events[4]), "faq");
  assert.equal(AA.eventFilter.nameOf(events[1]), "plan");
  assert.equal(AA.eventFilter.nameOf(events[0]), "help");
  assert.equal(AA.eventFilter.nameOf({ data: { provider: "p", model: "m" } }), "p/m");
});

test("state steps show before and after for traced fields, names only otherwise", () => {
  const steps = AA.stateSteps.steps(events);
  assert.equal(steps.length, 8);
  const [plan, search] = steps;
  assert.deepEqual(plain(plan.fields), [{ name: "plan_steps", before: { missing: true }, after: [1, 2], kind: "added" }]);
  assert.equal(plan.traced, true);
  assert.equal(search.traced, false);
  assert.equal(search.fields[0].kind, "named");
  const answers = steps.filter((s) => s.node === "answer");
  assert.equal(answers[0].fields[0].kind, "added");
  assert.deepEqual(plain(answers[1].fields[0]), { name: "answer", before: "first", after: "second", kind: "changed" });
  const finish = steps[steps.length - 1];
  assert.equal(finish.fields[0].kind, "same");
  assert.deepEqual(plain(AA.stateSteps.knownAfter(events, "help", finish.index)), { plan_steps: [1, 2], answer: "second", rounds: 1 });
});

test("state steps ignore malformed updates", () => {
  const odd = [{ type: "node.finished", ts: 1, data: { node: "a", changed: "x", updates: [1] } }];
  const [step] = AA.stateSteps.steps(odd);
  assert.deepEqual(plain(step.fields), []);
  assert.equal(step.traced, false);
});

test("run facts add up tokens, tool calls and node steps", () => {
  const facts = AA.compareRuns.facts(run(), "the answer");
  assert.equal(facts.input_tokens, 120);
  assert.equal(facts.output_tokens, 12);
  assert.equal(facts.tool_calls, 3);
  assert.deepEqual(plain(facts.tools), { faq: 2, forum: 1 });
  assert.equal(facts.llm_calls, 1);
  assert.equal(facts.steps.length, 8);
  assert.deepEqual(plain(facts.steps[0]), { node: "help.plan", status: "ok", changed: ["plan_steps"] });
  assert.equal(facts.answer, "the answer");
  assert.equal(AA.compareRuns.facts(null).answer, null);
});

test("step diffs align the same nodes and mark changed, added and removed", () => {
  const s = (node, status = "ok", changed = []) => ({ node, status, changed });
  const diff = AA.compareRuns.diffSteps([s("a"), s("b"), s("c")], [s("a"), s("b", "error"), s("x"), s("c")]);
  assert.deepEqual(diff.map((d) => d.op), ["same", "changed", "added", "same"]);
  const gone = AA.compareRuns.diffSteps([s("a"), s("b")], [s("b")]);
  assert.deepEqual(gone.map((d) => d.op), ["removed", "same"]);
  assert.deepEqual(AA.compareRuns.diffSteps([], []), []);
});

test("comparing a run with itself finds no difference", () => {
  const facts = AA.compareRuns.facts(run());
  const result = AA.compareRuns.compare(facts, facts);
  assert.ok(result.rows.every((r) => r.same));
  assert.ok(result.steps.every((s) => s.op === "same"));
  assert.ok(!result.rows.some((r) => r.label === "answer"));
  const other = AA.compareRuns.facts(run({ summary: { status: "failed", duration_ms: 10, cost_usd: null, errors: 0 } }), "x");
  const rows = AA.compareRuns.compare(facts, other).rows;
  assert.equal(rows[0].label, "answer");
  assert.equal(rows.find((r) => r.label === "status").same, false);
});

test("each flow keeps its own before values", () => {
  const two = [
    { type: "node.finished", ts: 1, data: { flow: "outer", node: "a", changed: ["x"], updates: { x: 1 } } },
    { type: "node.finished", ts: 2, data: { flow: "inner", node: "b", changed: ["x"], updates: { x: 2 } } },
    { type: "node.finished", ts: 3, data: { flow: "outer", node: "c", changed: ["x"], updates: { x: 3 } } },
  ];
  const [, inner, outer] = AA.stateSteps.steps(two);
  assert.equal(inner.fields[0].kind, "added"); // the outer flow's x is not the inner one's
  assert.equal(outer.fields[0].before, 1);
  assert.deepEqual(plain(AA.stateSteps.knownAfter(two, "outer", 2)), { x: 3 });
});

test("state fields named like Object members are plain fields", () => {
  const odd = [{ type: "node.finished", ts: 1, data: { flow: "f", node: "a", changed: ["constructor", "__proto__"], updates: JSON.parse('{"constructor": 1, "__proto__": 2}') } }];
  const [step] = AA.stateSteps.steps(odd);
  assert.deepEqual(step.fields.map((f) => f.kind), ["added", "added"]);
  const known = AA.stateSteps.knownAfter(odd, "f", 0);
  assert.deepEqual(Object.keys(known).sort(), ["__proto__", "constructor"]);
});

test("nested model calls are counted once, cached input tokens included", () => {
  const ev = (type, data) => ({ type, ts: 1, data });
  const usage = { input_tokens: 10, cached_input_tokens: 5, cache_write_input_tokens: 2, output_tokens: 3 };
  const facts = AA.compareRuns.facts({ events: [
    ev("llm.started", {}), ev("llm.started", {}),
    ev("llm.finished", { usage: usage }), // the inner call
    ev("llm.finished", { usage: usage, nested: true }), // the outer one repeats it
    ev("llm.started", {}), ev("llm.cached", {}),
  ] });
  assert.equal(facts.llm_calls, 2);
  assert.equal(facts.input_tokens, 17);
  assert.equal(facts.output_tokens, 3);
});

test("tools named like Object members are counted", () => {
  const ev = (tool) => ({ type: "tool.started", ts: 1, data: { tool } });
  const facts = AA.compareRuns.facts({ events: [ev("constructor"), ev("constructor"), ev("__proto__")] });
  assert.deepEqual(Object.entries(facts.tools).sort(), [["__proto__", 1], ["constructor", 2]]);
});

test("a long step diff trims the common ends and stays fast", () => {
  const s = (node) => ({ node, status: "ok", changed: [] });
  const long = Array.from({ length: 20000 }, (_, i) => s("n" + (i % 7)));
  const other = long.slice();
  other.splice(10000, 1, s("x"));
  const started = Date.now();
  const diff = AA.compareRuns.diffSteps(long, other);
  assert.ok(Date.now() - started < 1000);
  assert.equal(diff.filter((d) => d.op !== "same").length, 2); // one removed, one added
  assert.ok(!diff.approximate);
});

test("a huge differing middle is paired by position and marked approximate", () => {
  const s = (node) => ({ node, status: "ok", changed: [] });
  const a = Array.from({ length: 3000 }, (_, i) => s("a" + i));
  const b = Array.from({ length: 3000 }, (_, i) => s("b" + i));
  const diff = AA.compareRuns.diffSteps(a, b);
  assert.equal(diff.approximate, true);
  assert.equal(diff.length, 3000);
  assert.ok(AA.compareRuns.compare({ ...AA.compareRuns.facts(null), steps: a }, { ...AA.compareRuns.facts(null), steps: b }).approximate);
});
