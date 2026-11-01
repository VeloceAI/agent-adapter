// Touch on the graph (one-finger pan, pinch zoom, tap opens a node), the join badge, and the
// node panel scrolling into view on a stacked layout.
import assert from "node:assert/strict";
import { test } from "node:test";

import { event, loadDom } from "./fakedom.mjs";
import { graph, run } from "./fixtures.mjs";

globalThis.ResizeObserver = class {
  observe() {}
  disconnect() {}
};

const { AA, document } = loadDom("data.js", "graphlive.js", "graphnav.js", "graph.js", "graphbar.js", "panels.js", "views.js");
const nav = AA.graphNav;
const close = (a, b) => Math.abs(a - b) < 1e-9;
const scaleOf = (transform) => Number(/scale\(([\d.]+)\)/.exec(transform)[1]);
const touch = (type, id, x, y, target) => event(type, { button: 0, pointerType: "touch", pointerId: id, clientX: x, clientY: y, target: target });

function graphView(onOpen) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const view = AA.createGraph(container, { onOpen: onOpen || (() => {}) });
  view.update(graph, run());
  const stage = container.querySelector(".gv-stage");
  const world = container.querySelector(".gv-world");
  return { container, stage, transform: () => world.getAttribute("transform") };
}

test("a pinch scales by the finger spread and keeps the midpoint's world point still", () => {
  const view = { k: 1, x: 10, y: 20 };
  const next = nav.pinchView(view, { x: 100, y: 100 }, { x: 200, y: 100 }, { x: 50, y: 100 }, { x: 250, y: 100 });
  assert.equal(next.k, 2);
  const worldX = (150 - view.x) / view.k, worldY = (100 - view.y) / view.k;
  assert.ok(close(worldX * next.k + next.x, 150) && close(worldY * next.k + next.y, 100));
  const moved = nav.pinchView(view, { x: 0, y: 0 }, { x: 100, y: 0 }, { x: 30, y: 40 }, { x: 130, y: 40 });
  assert.deepEqual(moved, { k: 1, x: 40, y: 60 }); // same spread: a two-finger pan
  assert.equal(nav.pinchView(view, { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 0 }, { x: 1000, y: 0 }).k, nav.MAX_K);
});

test("one finger on the background pans the graph", () => {
  const { stage, transform } = graphView();
  const before = transform();
  stage.dispatchEvent(touch("pointerdown", 7, 100, 100, stage));
  stage.dispatchEvent(touch("pointermove", 7, 130, 90, stage));
  assert.notEqual(transform(), before);
  assert.ok(stage.classList.contains("panning"));
  stage.dispatchEvent(touch("pointerup", 7, 130, 90, stage));
  assert.ok(!stage.classList.contains("panning"));
});

test("two fingers zoom, even when one starts on a node, and lifting one ends the pinch", () => {
  const { container, stage, transform } = graphView();
  const node = container.querySelector('[data-id="merge"]');
  const k0 = scaleOf(transform());
  stage.dispatchEvent(touch("pointerdown", 1, 100, 100, node));
  stage.dispatchEvent(touch("pointerdown", 2, 200, 100, stage));
  stage.dispatchEvent(touch("pointermove", 2, 300, 100, stage));
  const k1 = scaleOf(transform());
  assert.ok(Math.abs(k1 - nav.clampK(k0 * 2)) < 1e-3, `${k0} -> ${k1}`); // the transform rounds k
  stage.dispatchEvent(touch("pointerup", 2, 300, 100, stage));
  stage.dispatchEvent(touch("pointermove", 1, 140, 140, node));
  assert.equal(scaleOf(transform()), k1, "one finger left: no zoom, and no pan from a node");
  stage.dispatchEvent(touch("pointerup", 1, 140, 140, node));
  assert.ok(!stage.classList.contains("panning"));
});

test("a tap on a node opens it, without starting a pan", () => {
  const opened = [];
  const { container, stage, transform } = graphView((id) => opened.push(id));
  const node = container.querySelector('[data-id="merge"]');
  const before = transform();
  stage.dispatchEvent(touch("pointerdown", 3, 50, 50, node));
  stage.dispatchEvent(touch("pointermove", 3, 52, 51, node));
  stage.dispatchEvent(touch("pointerup", 3, 52, 51, node));
  node.click(); // the browser's click after a tap
  assert.equal(transform(), before);
  assert.deepEqual(opened, ["merge"]);
});

test("a join's visit badge sits inside its hexagon, not on the slanted side", () => {
  const { container } = graphView();
  const scene = AA.graphScene.build(graph);
  const join = container.querySelector('[data-id="merge"] .g-badge');
  const fn = container.querySelector('[data-id="plan"] .g-badge');
  const W = AA.graphScene.W;
  assert.equal(Number(join.getAttribute("x")), scene.pos.merge.x + W - 18);
  assert.equal(Number(fn.getAttribute("x")), scene.pos.plan.x + W - 8);
});

function panel(top, height) {
  const scrolled = [];
  return {
    scrolled,
    getBoundingClientRect: () => ({ top: top, bottom: top + height }),
    scrollIntoView: (options) => scrolled.push(options),
  };
}

test("an opened node panel below the fold scrolls into view; one on screen stays put", () => {
  const view = { innerHeight: 673 };
  const below = panel(974, 300);
  assert.equal(AA.revealPanel(below, view), true);
  assert.deepEqual(below.scrolled, [{ block: "nearest" }]);
  const tall = panel(995, 900);
  assert.equal(AA.revealPanel(tall, view), true);
  assert.deepEqual(tall.scrolled, [{ block: "start" }], "a panel taller than the screen shows its top");
  const above = panel(-500, 300);
  assert.equal(AA.revealPanel(above, view), true);
  const seen = panel(400, 600);
  assert.equal(AA.revealPanel(seen, view), false);
  assert.deepEqual(seen.scrolled, []);
  assert.equal(AA.revealPanel(null, view), false);
  assert.equal(AA.revealPanel(panel(974, 10), { innerHeight: 0 }), false, "no layout yet: leave it");
});
