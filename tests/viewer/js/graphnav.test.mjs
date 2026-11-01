// graphnav.js and graph.js maths: zoom at the pointer, fit, visible box, focus moves, scene layout.
import assert from "node:assert/strict";
import { test } from "node:test";

import { graph } from "./fixtures.mjs";
import { load } from "./load.mjs";

const AA = load("graphlive.js", "graphnav.js", "graph.js");
const nav = AA.graphNav;

const close = (a, b) => Math.abs(a - b) < 1e-9;

test("zooming keeps the world point under the pointer still", () => {
  const view = { k: 1, x: 30, y: -20 };
  const px = 200, py = 120;
  const worldX = (px - view.x) / view.k, worldY = (py - view.y) / view.k;
  const next = nav.zoomAt(view, 1.5, px, py);
  assert.equal(next.k, 1.5);
  assert.ok(close(worldX * next.k + next.x, px) && close(worldY * next.k + next.y, py));
});

test("zoom is clamped at both ends", () => {
  assert.equal(nav.zoomAt({ k: 3.9, x: 0, y: 0 }, 2, 0, 0).k, nav.MAX_K);
  assert.equal(nav.zoomAt({ k: 0.2, x: 0, y: 0 }, 0.1, 0, 0).k, nav.MIN_K);
  assert.equal(nav.clampK(1), 1);
});

test("fit centres the graph and never blows a small one up past 1.5x", () => {
  const big = nav.fit({ x: 0, y: 0, w: 2000, h: 500 }, { w: 1000, h: 400 }, 0);
  assert.equal(big.k, 0.5);
  assert.ok(close(big.x, 0) && close(big.y, (400 - 250) / 2));
  const small = nav.fit({ x: 10, y: 0, w: 100, h: 50 }, { w: 1000, h: 400 });
  assert.equal(small.k, 1.5);
  assert.ok(close(small.x + (10 + 50) * small.k, 500)); // the centre lands in the middle
});

test("the visible box and centring are inverses", () => {
  const view = nav.centerOn({ k: 2, x: 0, y: 0 }, 100, 50, { w: 400, h: 200 });
  const seen = nav.visible(view, { w: 400, h: 200 });
  assert.ok(close(seen.x + seen.w / 2, 100) && close(seen.y + seen.h / 2, 50));
  assert.ok(close(seen.w, 200));
});

test("wheel deltas in lines and pages zoom like pixels", () => {
  assert.ok(nav.wheelFactor({ deltaY: -100, deltaMode: 0 }) > 1);
  assert.ok(close(nav.wheelFactor({ deltaY: 3, deltaMode: 1 }), nav.wheelFactor({ deltaY: 48, deltaMode: 0 })));
  assert.ok(nav.wheelFactor({ deltaY: 1, deltaMode: 2 }) < 1);
});

const wide = {
  flow: "w", start: "top", layers: 2,
  nodes: [
    { id: "top", kind: "fn", layer: 0, x: 1 },
    { id: "l", kind: "fn", layer: 1, x: 0 },
    { id: "m", kind: "fn", layer: 1, x: 1 },
    { id: "r", kind: "fn", layer: 1, x: 2 },
  ],
  edges: [{ source: "top", target: "l" }, { source: "top", target: "m" }, { source: "top", target: "r" }, { source: "r", target: "top", loop: true }],
};

test("arrow keys follow edges down and up, and siblings left and right", () => {
  const scene = AA.graphScene.build(wide);
  assert.equal(nav.neighbour("top", "ArrowDown", scene), "m"); // the closest child
  assert.equal(nav.neighbour("l", "ArrowUp", scene), "top");
  assert.equal(nav.neighbour("m", "ArrowRight", scene), "r");
  assert.equal(nav.neighbour("m", "ArrowLeft", scene), "l");
  assert.equal(nav.neighbour("l", "ArrowLeft", scene), null);
  assert.equal(nav.neighbour("top", "ArrowUp", scene), null); // the loop edge is not a way up
  assert.equal(nav.neighbour("ghost", "ArrowDown", scene), null);
});

test("the scene places nodes by slot and layer and leaves room for loop lanes", () => {
  const scene = AA.graphScene.build(wide);
  assert.equal(scene.pos.m.x - scene.pos.l.x, scene.pos.r.x - scene.pos.m.x);
  assert.equal(scene.pos.top.cx, scene.pos.m.cx);
  assert.ok(scene.pos.l.y > scene.pos.top.y);
  assert.ok(scene.bounds.w > scene.right && scene.bounds.h > scene.pos.l.y);
  const old = AA.graphScene.build({ ...wide, nodes: wide.nodes.map(({ x, ...n }) => ({ ...n, order: 0 })) });
  assert.equal(old.pos.l.x, old.pos.m.x); // files without x fall back to the order
});

test("the drawing key changes with the shape of the graph, not with live state", () => {
  const key = AA.graphScene.signature(graph);
  assert.equal(AA.graphScene.signature(JSON.parse(JSON.stringify(graph))), key);
  const moved = JSON.parse(JSON.stringify(graph));
  moved.nodes[1].x = 2;
  assert.notEqual(AA.graphScene.signature(moved), key);
});

test("curves pass through every bend with vertical tangents", () => {
  const d = AA.graphScene.curve([{ x: 0, y: 0 }, { x: 10, y: 100 }, { x: 10, y: 200 }]);
  assert.equal(d, "M0,0 C0,50 10,50 10,100 C10,150 10,150 10,200");
});

test("the view is kept so part of the graph stays on screen", () => {
  const bounds = { x: 0, y: 0, w: 500, h: 300 };
  const viewport = { w: 400, h: 200 };
  const far = nav.clampView({ k: 1, x: -2000, y: 5000 }, bounds, viewport, 40);
  assert.equal(far.x + bounds.w, 40); // the right edge 40 px in from the left
  assert.equal(far.y, viewport.h - 40); // the top edge 40 px up from the bottom
  const fine = { k: 1, x: -50, y: -20 };
  assert.deepEqual(nav.clampView(fine, bounds, viewport, 40), fine);
});
