// panels.js Tools tab: the effective policy lines of each tool, shown as plain text.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "panels.js");

const tool = (extra) => Object.assign({ name: "refund", description: "Refund an order.", kind: "tool", schema: {}, confirm: "rule", cache_ttl: null, cache_scope: "run", effects: null, idempotent: null, timeout: null, retries: 0, parallel_safe: false, once_per_run: false, tags: [] }, extra || {});

test("a tool under a policy lists its rules", () => {
  const box = document.createElement("div");
  AA.renderTools(box, [tool({ policy: ["allowed (default allow)", "approval: when amount > 100 (rules[0])"] })]);
  const items = box.all("li").map((li) => li.textContent);
  assert.deepEqual(items, ["allowed (default allow)", "approval: when amount > 100 (rules[0])"]);
  assert.ok(box.all("p").some((p) => p.textContent === "Policy"));
});

test("a tool without a policy shows no policy block", () => {
  const box = document.createElement("div");
  AA.renderTools(box, [tool()]);
  assert.equal(box.all("li").length, 0);
  assert.equal(AA.toolPolicy([]), null);
  assert.equal(AA.toolPolicy("not a list"), null);
});

test("policy lines are text, never markup", () => {
  const box = document.createElement("div");
  AA.renderTools(box, [tool({ policy: ["<img src=x onerror=alert(1)>"] })]);
  assert.equal(box.all("img").length, 0);
  assert.equal(box.all("li")[0].textContent, "<img src=x onerror=alert(1)>");
});
