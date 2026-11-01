// inspector.js: incremental run loading, no redraw when nothing changed, polling stops and pauses.
import assert from "node:assert/strict";
import { test } from "node:test";

import { run as fixtureRun } from "./fixtures.mjs";
import { loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "graphlive.js", "graphnav.js", "graph.js", "graphbar.js", "timeline.js", "events.js", "statepanel.js", "compare.js", "panels.js", "views.js", "inspector.js");

// Timers that only run when a test says so.
const timers = { intervals: [], timeouts: [] };
globalThis.setInterval = (fn) => { timers.intervals.push(fn); return timers.intervals.length; };
globalThis.clearInterval = (id) => { timers.intervals[id - 1] = null; };
globalThis.setTimeout = (fn) => { timers.timeouts.push(fn); return 0; };
const settle = () => new Promise((resolve) => setImmediate(resolve));
async function tickAll() {
  timers.timeouts.splice(0).forEach((fn) => fn());
  timers.intervals.filter(Boolean).forEach((fn) => fn());
  await settle();
  await settle();
}

function server(pages) {
  timers.intervals.length = 0; // inspectors of earlier tests stop ticking
  timers.timeouts.length = 0;
  const calls = [];
  AA.runsView = {
    loadMore: async (key, events) => {
      calls.push({ key, from: events.length });
      const next = pages.shift() || { add: 0, status: "ok" };
      for (let i = 0; i < next.add; i += 1) events.push({ type: "custom.x", data: {}, ts: i });
      return { summary: { key, status: next.status, events: events.length }, events, events_total: events.length, spans: [], spans_total: 0, graphs: {}, tools: [] };
    },
  };
  return calls;
}

const turn = (status) => ({ id: "turn_1", run_id: "run_1", status, input_tokens: 5, output_tokens: 1, cost_usd: 0 });

test("a running turn's run is fetched from where the last load stopped", async () => {
  const calls = server([{ add: 3, status: "running" }, { add: 2, status: "running" }, { add: 0, status: "running" }]);
  const box = document.createElement("div");
  const insp = AA.createInspector(box, () => {}, () => true);
  insp.show(turn("running"), null);
  await tickAll();
  await tickAll();
  await tickAll();
  assert.deepEqual(calls.map((c) => c.from), [0, 3, 5], "only new events are asked for");
  assert.equal(insp.current().run.events.length, 5);
});

test("an unchanged run is not drawn again", async () => {
  server([{ add: 2, status: "running" }, { add: 0, status: "running" }]);
  const box = document.createElement("div");
  const insp = AA.createInspector(box, () => {}, () => true);
  insp.showTab("cost");
  insp.show(turn("running"), null);
  await tickAll();
  const drawn = box.querySelector("#insp-cost").firstChild;
  await tickAll();
  assert.equal(box.querySelector("#insp-cost").firstChild, drawn, "same nodes: nothing changed");
});

test("polling stops once the turn ended and the run is no longer running", async () => {
  server([{ add: 1, status: "running" }, { add: 1, status: "ok" }]);
  const box = document.createElement("div");
  const insp = AA.createInspector(box, () => {}, () => true);
  insp.show(turn("running"), null);
  await tickAll();
  assert.equal(insp.current().polling, true);
  insp.show(turn("done"), null);
  await tickAll();
  assert.equal(insp.current().polling, false);
});

test("polling gives up a few loads after the turn ended, even if the run never settles", async () => {
  server(Array.from({ length: 20 }, () => ({ add: 0, status: "running" })));
  const box = document.createElement("div");
  const insp = AA.createInspector(box, () => {}, () => true);
  insp.show(turn("done"), null);
  for (let i = 0; i < 8; i += 1) await tickAll();
  assert.equal(insp.current().polling, false);
});

test("nothing is fetched while the chat view is hidden", async () => {
  const calls = server([{ add: 1, status: "running" }]);
  let shown = false;
  const box = document.createElement("div");
  const insp = AA.createInspector(box, () => {}, () => shown);
  insp.show(turn("running"), null);
  await tickAll();
  await tickAll();
  assert.equal(calls.length, 0);
  shown = true;
  await tickAll();
  assert.equal(calls.length, 1);
});

test("the Cost tab shows the turn's running usage and the run's model calls", async () => {
  server([]);
  AA.runsView = { loadMore: async () => fixtureRun() };
  const box = document.createElement("div");
  const insp = AA.createInspector(box, () => {}, () => true);
  insp.showTab("cost");
  insp.show(Object.assign(turn("running"), { input_tokens: 120, output_tokens: 12, cost_usd: 0.002 }), null);
  await tickAll();
  const text = box.querySelector("#insp-cost").textContent;
  assert.match(text, /120 in \/ 12 out/);
  assert.match(text, /local\/m/);
});
