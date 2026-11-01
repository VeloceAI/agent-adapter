// The DOM controllers on a small fake DOM: graph view, camera, minimap, tabs, Events, Compare.
import assert from "node:assert/strict";
import { test } from "node:test";

import { event, loadDom } from "./fakedom.mjs";
import { graph, run } from "./fixtures.mjs";

let resized = null;
globalThis.ResizeObserver = class {
  constructor(callback) { resized = callback; }
  observe() {}
  disconnect() {}
};

const { AA, document } = loadDom("data.js", "graphlive.js", "graphnav.js", "graph.js", "graphbar.js", "events.js", "compare.js", "views.js");

function mount() {
  const el = document.createElement("div");
  document.body.appendChild(el);
  return el;
}

function graphView(options) {
  const container = mount();
  const view = AA.createGraph(container, options || {});
  view.update(graph, run());
  const stage = container.querySelector(".gv-stage");
  const world = container.querySelector(".gv-world");
  const transform = () => world.getAttribute("transform");
  return { container, view, stage, world, transform, mini: container.querySelector(".gv-mini") };
}

test("the snapshot is not worked out again on hover", () => {
  const real = AA.graphLive.snapshot;
  let calls = 0;
  AA.graphLive.snapshot = (...args) => { calls += 1; return real(...args); };
  try {
    const { container, view } = graphView();
    const before = calls;
    const node = container.querySelector('[data-id="merge"]');
    for (let i = 0; i < 5; i += 1) {
      node.dispatchEvent(event("pointerenter"));
      node.dispatchEvent(event("pointerleave"));
    }
    assert.equal(calls, before);
    view.update(graph, run()); // the same run again
    assert.equal(calls, before);
    view.update(graph, run({ events: run().events.slice(0, 5) }));
    assert.equal(calls, before + 1);
  } finally {
    AA.graphLive.snapshot = real;
  }
});

test("moving the replay tells the page, so the node panel can follow", () => {
  let steps = 0;
  const { container } = graphView({ onStep: () => { steps += 1; } });
  const slider = container.querySelector(".gv-slider");
  slider.value = "3";
  slider.dispatchEvent(event("input"));
  assert.equal(steps, 1);
  assert.ok(container.querySelector(".gv-step").textContent.startsWith("step 3/16"));
});

test("the stage is a named group and a redraw keeps the focused node focused", () => {
  const { container, view, stage } = graphView();
  assert.equal(stage.getAttribute("role"), "group");
  container.querySelector('[data-id="answer"]').dispatchEvent(event("focus"));
  container.querySelector('[data-id="answer"]').focus();
  const moved = JSON.parse(JSON.stringify(graph));
  moved.nodes[3].x = 1; // a new drawing
  view.update(moved, run());
  const now = container.querySelector('[data-id="answer"]');
  assert.equal(document.activeElement, now);
});

test("dragging far away stops with part of the graph still on screen", () => {
  const { stage, transform } = graphView();
  stage.dispatchEvent(event("pointerdown", { button: 0, pointerId: 1, clientX: 0, clientY: 0, target: stage }));
  stage.dispatchEvent(event("pointermove", { pointerId: 1, clientX: -100000, clientY: -100000 }));
  const [x, y, k] = transform().match(/-?[\d.]+/g).map(Number);
  assert.ok(x > -100000 && y > -100000);
  assert.ok(k > 0);
  stage.dispatchEvent(event("lostpointercapture", { pointerId: 1 }));
  const ended = transform();
  stage.dispatchEvent(event("pointermove", { pointerId: 1, clientX: 50, clientY: 50 }));
  assert.equal(transform(), ended); // no drag after the capture was lost
});

test("a resize refits until the reader moves the view", () => {
  const { stage, transform } = graphView();
  stage.clientWidth = 1600;
  resized();
  const refit = transform();
  stage.clientWidth = 400;
  resized();
  assert.notEqual(transform(), refit);
  stage.dispatchEvent(event("keydown", { key: "+", target: stage }));
  const zoomed = transform();
  stage.clientWidth = 1200;
  resized();
  assert.equal(transform(), zoomed); // kept where the reader put it
});

test("a cancelled minimap drag no longer moves the view", () => {
  const { stage, mini, transform } = graphView();
  stage.dispatchEvent(event("keydown", { key: "+", target: stage }));
  mini.dispatchEvent(event("pointerdown", { pointerId: 2, clientX: 10, clientY: 10 }));
  mini.dispatchEvent(event("pointercancel", { pointerId: 2 }));
  const after = transform();
  mini.dispatchEvent(event("pointermove", { pointerId: 2, clientX: 150, clientY: 100 }));
  assert.equal(transform(), after);
});

test("tabs rove: one is in the Tab order, arrows move along and pick", () => {
  const list = mount();
  const panels = ["a", "b", "c"].map((name) => {
    const button = document.createElement("button");
    button.setAttribute("role", "tab");
    button.setAttribute("data-tab", name);
    list.appendChild(button);
    const panel = mount();
    panel.id = "panel-" + name;
    AA.tabs.linkPanel(button, panel);
    return { button, panel };
  });
  const buttons = panels.map((p) => p.button);
  let picked = null;
  AA.tabs.arrowKeys(list, (b) => { picked = b.getAttribute("data-tab"); AA.tabs.select(buttons, b); });
  AA.tabs.select(buttons, buttons[0]);
  assert.deepEqual(buttons.map((b) => b.getAttribute("tabindex")), ["0", "-1", "-1"]);
  assert.equal(buttons[1].getAttribute("aria-controls"), "panel-b");
  assert.equal(panels[1].panel.getAttribute("role"), "tabpanel");
  assert.equal(panels[1].panel.getAttribute("aria-labelledby"), buttons[1].id);
  buttons[0].focus();
  list.dispatchEvent(event("keydown", { key: "ArrowLeft" }));
  assert.equal(picked, "c");
  assert.equal(document.activeElement, buttons[2]);
  assert.equal(buttons[2].getAttribute("aria-selected"), "true");
});

test("event filter selects stay the same elements and a focused one keeps its options", () => {
  const container = mount();
  const events = AA.createEvents(container);
  const small = run({ events: run().events.slice(0, 5) });
  events.update(small);
  const toolSelect = container.querySelector('select[aria-label="filter events by tool"]');
  assert.deepEqual(toolSelect.options.map((o) => o.value), ["", "faq"]);
  toolSelect.focus();
  events.update(run());
  assert.equal(container.querySelector('select[aria-label="filter events by tool"]'), toolSelect);
  assert.deepEqual(toolSelect.options.map((o) => o.value), ["", "faq"]); // not changed while in use
  toolSelect.blur();
  events.update(run());
  assert.deepEqual(toolSelect.options.map((o) => o.value), ["", "faq", "forum"]);
});

test("a picked event row is marked current, not selected", () => {
  const container = mount();
  const events = AA.createEvents(container);
  events.update(run());
  const body = container.querySelector(".ev-body");
  body.dispatchEvent(event("keydown", { key: "ArrowDown" }));
  const row = container.querySelector("tr.selected");
  assert.equal(row.getAttribute("aria-current"), "true");
  assert.equal(row.getAttribute("aria-selected"), null);
});

test("compare reloads the other run when it changes and keeps the table otherwise", async () => {
  const container = mount();
  const loads = [];
  const other = run({ summary: { key: "k2", run_id: "run_2", status: "running", events: 3 }, events: run().events.slice(0, 3) });
  const compare = AA.createCompare(container, async (key) => { loads.push(key); return other; });
  const listed = [{ key: "k1", status: "ok", events: 26, started: 1, run_id: "run_1" }, { key: "k2", status: "running", events: 3, started: 2, run_id: "run_2" }];
  compare.update(run(), listed);
  const picker = container.querySelector("select");
  picker.value = "k2";
  picker.dispatchEvent(event("change"));
  await new Promise((r) => setTimeout(r, 0));
  const table = container.querySelector(".cmp-table");
  assert.ok(table);
  compare.update(run(), listed);
  assert.equal(container.querySelector(".cmp-table"), table); // nothing changed: same DOM
  assert.deepEqual(loads, ["k2"]);
  compare.update(run(), [listed[0], { ...listed[1], events: 9, status: "ok" }]);
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(loads, ["k2", "k2"]);
});

test("compare shows a status that changed without new events", async () => {
  const container = mount();
  const other = run({ summary: { key: "k2", run_id: "run_2", status: "ok", events: 3 }, events: run().events.slice(0, 3) });
  const compare = AA.createCompare(container, async () => other);
  const listed = [{ key: "k1", status: "running", events: 26, started: 1, run_id: "run_1" }, { key: "k2", status: "ok", events: 3, started: 2, run_id: "run_2" }];
  const live = run({ summary: { key: "k1", run_id: "run_1", status: "running", duration_ms: 250, events: 26 } });
  compare.update(live, listed);
  const picker = container.querySelector("select");
  picker.value = "k2";
  picker.dispatchEvent(event("change"));
  await new Promise((r) => setTimeout(r, 0));
  const status = () => container.querySelector(".cmp-table").textContent;
  assert.ok(status().includes("running"));
  compare.update(run({ summary: { ...live.summary, status: "unfinished", duration_ms: 900 } }), listed);
  assert.ok(status().includes("unfinished") && !status().includes("running"));
});
