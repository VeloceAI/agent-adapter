// runsearch.js and panels.js: the search box, the tag filter, tag chips and the tag editor.
import assert from "node:assert/strict";
import { test } from "node:test";

import { event, loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "panels.js", "runsearch.js");

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test("typing searches after a pause; the tag filter lists known tags", async () => {
  const box = document.createElement("div");
  let changes = 0;
  const search = AA.runSearch.createSearch(box, () => { changes += 1; });
  const input = box.querySelector("input");
  const pick = box.querySelector("select");
  assert.deepEqual(search.params(), { q: "", tag: "" });
  assert.equal(search.active(), false);
  input.value = "  stock ";
  input.dispatchEvent(event("input"));
  input.dispatchEvent(event("input"));
  assert.equal(changes, 0);
  await wait(300);
  assert.equal(changes, 1);
  assert.deepEqual(search.params(), { q: "stock", tag: "" });
  search.paint({ runs: [{}], tags: ["vip", "<b>x</b>"], total: 3 });
  assert.equal(box.querySelector(".run-count").textContent, "1 of 3 runs");
  const options = pick.querySelectorAll("option");
  assert.deepEqual(options.map((o) => o.textContent), ["all tags", "vip", "<b>x</b>"]);
  assert.equal(pick.querySelectorAll("b").length, 0);
  pick.value = "vip";
  pick.dispatchEvent(event("change"));
  assert.equal(changes, 2);
  assert.equal(search.active(), true);
  search.paint({ runs: [], tags: [], total: 3 }); // the picked tag stays listed
  assert.deepEqual(pick.querySelectorAll("option").map((o) => o.textContent), ["all tags", "vip"]);
});

test("the runs list shows tag chips as text and says when nothing matches", () => {
  const list = document.createElement("div");
  AA.renderRuns(list, [{ key: "r1", run_id: "r1", started: 1, status: "ok", duration_ms: 1, tags: ["vip", "<i>t</i>"] }], "r1", () => {});
  assert.deepEqual(list.querySelectorAll(".chip.tag").map((c) => c.textContent), ["vip", "<i>t</i>"]);
  assert.equal(list.querySelectorAll("i").length, 0);
  AA.renderRuns(list, [], null, () => {}, true);
  assert.equal(list.textContent, "No run matches this search.");
  AA.renderRuns(list, [], null, () => {});
  assert.match(list.textContent, /No runs yet/);
});

test("the tag editor adds and removes tags and shows a failed save", async () => {
  const head = document.createElement("div");
  const changes = [];
  let fail = false;
  AA.runSearch.renderTags(head, ["vip"], (change) => {
    changes.push(change);
    if (fail) return Promise.reject(new Error("Origin must be this viewer"));
    return Promise.resolve(change.add ? ["vip"].concat(change.add) : []);
  });
  const form = head.querySelector("form");
  head.querySelector("input").value = "  ";
  form.dispatchEvent(event("submit"));
  assert.equal(changes.length, 0);
  head.querySelector("input").value = " slow ";
  form.dispatchEvent(event("submit"));
  await wait(0);
  assert.deepEqual(changes[0], { add: ["slow"] });
  assert.deepEqual(head.querySelectorAll(".chip.tag").map((c) => c.firstChild.textContent), ["vip", "slow"]);
  head.querySelector(".chip.tag button").click();
  await wait(0);
  assert.deepEqual(changes[1], { remove: ["vip"] });
  assert.equal(head.querySelectorAll(".chip.tag").length, 0);
  fail = true;
  head.querySelector("input").value = "x";
  head.querySelector("form").dispatchEvent(event("submit"));
  await wait(0);
  assert.match(head.querySelector(".notice").textContent, /Not saved: Origin/);
  assert.equal(head.querySelector("input").disabled, false);
});

test("embedded exports list runs without asking the server", async () => {
  AA.embedded = { runs: [{ key: "r1" }] };
  try {
    assert.deepEqual(await AA.api.runList({ q: "x" }), { runs: [{ key: "r1" }], tags: [], total: 1 });
  } finally {
    delete AA.embedded;
  }
});

test("runList sends the search; setTags posts JSON with the CSRF header", async () => {
  const sent = [];
  const saved = globalThis.fetch;
  globalThis.fetch = async (url, options) => {
    sent.push({ url, options });
    const ok = !String(url).includes("bad");
    return { ok, status: ok ? 200 : 403, json: async () => (ok ? { runs: [], tags: ["a"], total: 0 } : { error: "Origin must be this viewer" }) };
  };
  try {
    await AA.api.runList({ q: "a b", tag: "vip" });
    await AA.api.runList({});
    assert.equal(sent[0].url, "api/runs?q=a+b&tag=vip");
    assert.equal(sent[1].url, "api/runs");
    assert.deepEqual(await AA.api.setTags("r1", { add: ["x"] }), ["a"]);
    assert.equal(sent[2].url, "api/runs/r1/tags");
    assert.equal(sent[2].options.method, "POST");
    assert.equal(sent[2].options.headers["Content-Type"], "application/json");
    assert.ok("X-AA-CSRF" in sent[2].options.headers);
    assert.equal(sent[2].options.body, '{"add":["x"]}');
    await assert.rejects(AA.api.setTags("bad", { add: ["x"] }), /Origin/);
  } finally {
    globalThis.fetch = saved;
  }
});

test("the tag editor survives a live re-render of the same run while someone types", async () => {
  const head = document.createElement("div");
  document.body.appendChild(head);
  const save = (change) => Promise.resolve(change.add ? ["vip"].concat(change.add) : []);
  const first = AA.runSearch.attachTags(head, "r1", ["vip"], save);
  const input = first.querySelector("input");
  input.value = "half-typ";
  input.focus();
  AA.clear(head);
  assert.equal(AA.runSearch.attachTags(head, "r1", ["vip", "new"], save), first); // typing: kept
  assert.equal(head.querySelector("input").value, "half-typ");
  input.blur();
  AA.clear(head);
  assert.equal(AA.runSearch.attachTags(head, "r1", ["vip"], save), first); // same tags: kept
  const rebuilt = AA.runSearch.attachTags(head, "r1", ["vip", "new"], save); // changed elsewhere
  assert.notEqual(rebuilt, first);
  rebuilt.querySelector("input").value = "slow";
  rebuilt.querySelector("form").dispatchEvent(event("submit"));
  await wait(0);
  AA.clear(head);
  assert.equal(AA.runSearch.attachTags(head, "r1", ["vip", "slow"], save), rebuilt); // saved here
  assert.notEqual(AA.runSearch.attachTags(head, "r2", ["vip", "slow"], save), rebuilt); // other run
});
