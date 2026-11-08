// runcases.js: "Save as eval case" and "Turn into regression case" in the run header.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "runcases.js");
AA.csrf = "csrf-1";

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
const buttons = (node) => node.querySelectorAll("button").map((b) => b.textContent);

test("an ok run offers a case; the dialog sends id, tags, expected_from and an expected value", async () => {
  const head = document.createElement("div");
  const sent = [];
  const save = (body) => { sent.push(body); return Promise.resolve({ case: { id: body.id }, file: "eval_cases.jsonl" }); };
  AA.runCases.attach(head, "r1", "ok", save);
  assert.deepEqual(buttons(head), ["Save as eval case"]);
  head.querySelector("button").click();
  const dialog = document.body.querySelector("dialog");
  assert.ok(dialog && dialog.attrs.has("open"));
  const [id, useOutput, tags] = dialog.querySelectorAll("input");
  assert.equal(id.value, "run-r1");
  assert.equal(useOutput.checked, true);
  tags.value = "vip, <b>x</b>";
  dialog.querySelector("textarea").value = '{"ok": true}';
  dialog.querySelector("button.approve").click();
  await settle();
  assert.deepEqual(sent, [{ id: "run-r1", tags: ["vip", "<b>x</b>"], expected_from: "output", expected: { ok: true } }]);
  assert.equal(dialog.querySelector(".notice").textContent, "Saved run-r1 to eval_cases.jsonl.");
  assert.equal(dialog.querySelectorAll("b").length, 0);
  dialog.close();
  assert.equal(document.body.querySelector("dialog"), null);
});

test("the dialog refuses an empty id and shows a refused save as text", async () => {
  const head = document.createElement("div");
  const save = () => Promise.reject(new Error("<i>not saved</i>: recorded without capture_content=True"));
  AA.runCases.attach(head, "r2", "unfinished", save);
  const dialog = AA.runCases.caseDialog("r2", "unfinished", save);
  const [id, useOutput] = dialog.querySelectorAll("input");
  assert.equal(useOutput.checked, false);
  id.value = " ";
  const button = dialog.querySelector("button.approve");
  button.click();
  assert.match(dialog.querySelector(".notice").textContent, /give the case an id/);
  id.value = "mine";
  button.click();
  await settle();
  assert.match(dialog.querySelector(".notice").textContent, /^Not saved: <i>not saved<\/i>/);
  assert.equal(dialog.querySelectorAll("i").length, 0);
  assert.equal(button.disabled, false);
  dialog.close();
});

test("a failed run gets a one-click regression case; the notice survives a re-render", async () => {
  const head = document.createElement("div");
  const sent = [];
  let fail = true;
  const save = (body) => {
    sent.push(body);
    return fail ? Promise.reject(new Error("a case with id 'regression-r3' exists")) : Promise.resolve({ case: { id: "regression-r3" }, file: "eval_cases.jsonl" });
  };
  const box = AA.runCases.attach(head, "r3", "failed", save);
  assert.deepEqual(buttons(head), ["Save as eval case", "Turn into regression case"]);
  const quick = head.querySelectorAll("button")[1];
  quick.click();
  await settle();
  assert.match(head.querySelector(".notice").textContent, /^Not saved: a case/);
  fail = false;
  quick.click();
  await settle();
  assert.deepEqual(sent, [{ regression: true }, { regression: true }]);
  assert.equal(head.querySelector(".notice").textContent, "Saved regression-r3 to eval_cases.jsonl.");
  const again = document.createElement("div");
  assert.equal(AA.runCases.attach(again, "r3", "failed", save), box); // same run: kept
  assert.notEqual(AA.runCases.attach(again, "r4", "failed", save), box);
});

test("saveCase posts JSON with the CSRF header and rejects with the server's reason", async () => {
  const saved = globalThis.fetch;
  const calls = [];
  let reply = { ok: true, status: 201, body: { case: { id: "run-k" }, file: "eval_cases.jsonl" } };
  globalThis.fetch = async (url, options) => {
    calls.push([url, options]);
    return { ok: reply.ok, status: reply.status, json: async () => reply.body };
  };
  try {
    assert.equal((await AA.runCases.saveCase("a/b", { regression: true })).case.id, "run-k");
    const [url, options] = calls[0];
    assert.equal(url, "api/runs/a%2Fb/eval-case");
    assert.equal(options.method, "POST");
    assert.equal(options.headers["Content-Type"], "application/json");
    assert.equal(options.headers["X-AA-CSRF"], "csrf-1");
    assert.equal(options.body, '{"regression":true}');
    reply = { ok: false, status: 422, body: { error: "not saved: no input was recorded" } };
    await assert.rejects(AA.runCases.saveCase("k", {}), /no input was recorded/);
    reply = { ok: false, status: 500, body: {} };
    await assert.rejects(AA.runCases.saveCase("k", {}), /HTTP 500/);
  } finally {
    globalThis.fetch = saved;
  }
});

test("parseExpected keeps JSON, text, or nothing", () => {
  assert.deepEqual(AA.runCases.parseExpected(' {"a": 1} '), { a: 1 });
  assert.equal(AA.runCases.parseExpected("shipped"), "shipped");
  assert.equal(AA.runCases.parseExpected("  "), undefined);
});
