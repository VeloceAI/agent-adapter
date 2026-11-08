// The Evals view: its pure logic, and the view on the fake DOM against a stubbed server.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDom } from "./fakedom.mjs";

const PAGE = '<header><nav id="top-tabs"><button id="view-chat" data-view="chat" role="tab">Chat</button>'
  + '<button id="view-runs" data-view="runs" role="tab">Runs</button></nav></header>'
  + '<section id="chat-view"><div id="chat-off"></div></section><section id="runs-view"></section>';

function grade(grader, score, passed, notes) {
  return { grader: grader, score: score, passed: passed, notes: notes || "" };
}

const BASE = {
  key: "r1", name: "shop", started_at: "2026-11-05T10:00:00+00:00", wall_s: 2, n_cases: 2, n_passed: 2, n_errors: 0,
  mean_score: 1, pass_rate: 1, cost_usd: 0.002, by_tag: { orders: { n: 2, n_passed: 2, mean_score: 1 } }, by_grader: {},
  cases: [
    { case_id: "a", tags: ["orders"], passed: true, score: 1, grades: [grade("answer", 1, true)], run_id: "run_a0", input: "i", expected: "e", output: "o", error: "", cost_usd: 0.001, latency_s: 0.5, tool_calls: [] },
    { case_id: "b", tags: ["orders"], passed: true, score: 1, grades: [grade("answer", 1, true)], run_id: "run_b0", input: "", expected: "", output: "", error: "", cost_usd: 0.001, latency_s: 0.5, tool_calls: [] },
  ],
};

const HEAD = Object.assign({}, BASE, {
  key: "r2", n_passed: 1, mean_score: 0.5, pass_rate: 0.5,
  by_grader: { answer: { n: 2, n_passed: 1, mean_score: 0.5 } },
  cases: [
    { case_id: "a", tags: ["orders"], passed: false, score: 0, grades: [grade("answer", 0, false, "<img src=x onerror=alert(1)>")], run_id: "run_a1", input: "<b>hi</b>", expected: "shipped", output: "unknown", error: "", cost_usd: 0.001, latency_s: 0.25, tool_calls: ["order_status"] },
    { case_id: "c", tags: ["chat"], passed: true, score: 1, grades: [grade("answer", 1, true)], run_id: "", input: "x", expected: "", output: "y", error: "", cost_usd: 0, latency_s: 0.1, tool_calls: [] },
  ],
});

const COMPARISON = {
  score_delta: -0.5, regressed: true, added: ["c"], removed: ["b"], improvements: [],
  regressions: [{ case_id: "a", before_passed: true, after_passed: false, before_score: 1, after_score: 0 }],
};

function summary(report) {
  const out = Object.assign({ problem: "" }, report);
  delete out.cases;
  return out;
}

const calls = [];
let responses = {};
globalThis.fetch = async (path, options) => {
  calls.push({ path: path, token: options.headers["X-AA-Token"] });
  const found = responses[path];
  const status = found === undefined ? 404 : found.status || 200;
  return { ok: status < 400, status: status, json: async () => (found && found.body) || { error: "no" } };
};

function serve() {
  responses = {
    "api/evals": { body: { reports: [summary(HEAD), summary(BASE), { key: "bad", problem: "not an eval report" }], folder: "evals" } },
    "api/evals/r1": { body: BASE },
    "api/evals/r2": { body: HEAD },
    "api/evals/compare/r1/r2": { body: COMPARISON },
  };
}

serve();
globalThis.sessionStorage = { setItem() {}, getItem: () => "" };
const { AA, document } = loadDom({ html: PAGE, hash: "#token=t0k" }, "data.js", "views.js", "evals.js", "evalscards.js", "evalsview.js");
const settle = async () => { for (let i = 0; i < 10; i += 1) await new Promise((r) => setTimeout(r, 0)); };
const $ = (sel) => document.querySelector(sel);

test("verdicts, formats and filters", () => {
  const E = AA.evals;
  assert.deepEqual(E.verdict(summary(BASE)), { text: "all passed", status: "ok" });
  assert.deepEqual(E.verdict(summary(HEAD)), { text: "1 failed", status: "failed" });
  assert.equal(E.verdict({ n_cases: 3, n_passed: 0, n_errors: 2 }).text, "2 errors");
  assert.equal(E.verdict({ problem: "x" }).text, "unreadable");
  assert.equal(E.verdict({ n_cases: 0, n_passed: 0, n_errors: 0 }).text, "no cases");
  assert.equal(E.fmt.score(0.5), "0.500");
  assert.equal(E.fmt.score(null), "—");
  assert.equal(E.fmt.score(NaN), "—");
  assert.equal(E.fmt.delta(-0.25), "−0.250");
  assert.equal(E.fmt.delta(0.25), "+0.250");
  assert.equal(E.fmt.delta(0), "±0.000");
  assert.equal(E.fmt.percent(0.5), "50%");
  assert.equal(E.fmt.when("not a time"), "not a time");
  assert.equal(E.caseStatus({ error: "timeout: x", passed: false }).text, "error");
  assert.deepEqual(E.filterCases(HEAD.cases, "failed", "").map((c) => c.case_id), ["a"]);
  assert.deepEqual(E.filterCases(HEAD.cases, "all", "chat").map((c) => c.case_id), ["c"]);
  assert.deepEqual(E.filterCases(null, "all", ""), []);
  assert.deepEqual(E.groups({ b: { n: 1, n_passed: 0, mean_score: 0 }, a: {} }).map((g) => [g.name, g.n]), [["a", 0], ["b", 1]]);
});

test("side by side puts regressions first and marks added and removed cases", () => {
  const rows = AA.evals.sideBySide(BASE, HEAD, COMPARISON);
  assert.deepEqual(rows.map((r) => [r.id, r.change]), [["a", "regressed"], ["c", "added"], ["b", "removed"]]);
  assert.equal(rows[1].base, null);
  assert.equal(rows[2].head, null);
  assert.deepEqual(AA.evals.sideBySide(null, null, null), []);
});

test("a case's run key comes from the runs list", () => {
  const runs = [{ key: "k1", run_id: "run_a1" }, { key: "run_x", run_id: "other" }];
  assert.equal(AA.evals.runKey(runs, "run_a1"), "k1");
  assert.equal(AA.evals.runKey(runs, "run_x"), "run_x");
  assert.equal(AA.evals.runKey(runs, "missing"), null);
  assert.equal(AA.evals.runKey(runs, ""), null);
});

test("the Evals tab joins the top tabs and lists the reports with the token", async () => {
  const button = $("#view-evals");
  assert.ok(button && button.getAttribute("aria-controls") === "evals-view");
  assert.ok($("#evals-view").hidden);
  button.click();
  await settle();
  assert.equal($("#evals-view").hidden, false);
  assert.ok(calls.every((c) => c.token === "t0k"));
  const items = $("#evl-list").querySelectorAll(".run");
  assert.equal(items.length, 3);
  assert.ok(items[0].classList.contains("selected"), "the newest readable report opens");
  assert.ok(items[2].disabled && items[2].textContent.includes("not an eval report"));
  const head = $("#evl-head").textContent;
  assert.ok(head.includes("0.500") && head.includes("1 failed") && head.includes("1 / 2"));
});

test("cases show input, expected, output and each grade as text, never as markup", () => {
  const card = $('#evl-cases [data-case="a"]');
  const text = card.textContent;
  assert.ok(text.includes("<b>hi</b>") && text.includes("shipped") && text.includes("unknown"));
  assert.ok(text.includes("<img src=x onerror=alert(1)>"), "a grader's reason stays text");
  assert.equal(card.querySelectorAll("img").length + card.querySelectorAll("b").length, 0);
  assert.ok(text.includes("order_status") && text.includes("run_a1"));
  assert.ok($("#evl-groups").textContent.includes("By tag"));
});

test("the failing filter and a tag narrow the cases", () => {
  const select = $("#evl-cases select");
  select.value = "failed";
  select.dispatch("change");
  assert.deepEqual($("#evl-cases").querySelectorAll(".evl-case").map((c) => c.getAttribute("data-case")), ["a"]);
  select.value = "all";
  $("#evl-cases select").value = "all";
  $("#evl-cases select").dispatch("change");
  $("#evl-groups button").click(); // the "orders" tag
  assert.deepEqual($("#evl-cases").querySelectorAll(".evl-case").map((c) => c.getAttribute("data-case")), ["a"]);
  $("#evl-cases .ev-filters button").click(); // clear the tag
  assert.equal($("#evl-cases").querySelectorAll(".evl-case").length, 2);
});

test("comparing with a baseline highlights the regression", async () => {
  const picker = $("#evl-head select");
  assert.deepEqual(picker.options.map((o) => o.value), ["", "r1"], "unreadable files are not offered");
  picker.value = "r1";
  picker.dispatch("change");
  await settle();
  const section = $("#evl-compare");
  assert.equal(section.hidden, false);
  assert.ok(section.textContent.includes("1 case(s) regressed"));
  const rows = section.querySelectorAll("tbody tr");
  assert.deepEqual(rows.map((r) => r.className), ["evl-regressed", "evl-added", "evl-removed"]);
  assert.ok(section.textContent.includes("1.000 → 0.500 (−0.500)"));
});

test("Open run switches to the Runs view at the case's run, or says it is missing", async () => {
  const picked = [];
  AA.runsView = { pick: (key) => picked.push(key) };
  AA.api.runs = async () => [{ key: "k-a1", run_id: "run_a1" }];
  const open = () => $('#evl-cases [data-case="a"] .actions button');
  open().click();
  await settle();
  assert.deepEqual(picked, ["k-a1"]);
  assert.equal($("#runs-view").hidden, false);
  assert.ok($("#evals-view").hidden);
  AA.views.show("evals");
  await settle();
  AA.api.runs = async () => [];
  open().click();
  await settle();
  assert.ok($('#evl-cases [data-case="a"] .actions').textContent.includes("not in the runs folder"));
  assert.equal($('#evl-cases [data-case="c"] .actions button'), null, "no run id, no button");
  AA.api.runs = async () => { throw new Error("down"); };
  open().click();
  await settle();
  assert.ok($('#evl-cases [data-case="a"] .actions').textContent.includes("Could not list the runs: down"));
});

test("a failed report load says so; a 401 locks the view and stops polling", async () => {
  responses["api/evals/r1"] = { status: 500, body: { error: "the viewer could not read it" } };
  const quiet = console.error;
  console.error = () => {};
  try {
    await AA.evalsView.pick("r1");
    assert.ok($("#evl-head").textContent.includes("Could not load that report: the viewer could not read it"));
    responses["api/evals"] = { status: 401, body: { error: "token" } };
    await AA.evalsView.refresh();
    assert.ok($("#evl-head").textContent.includes("refused the access token"));
    calls.length = 0;
    await AA.evalsView.refresh();
    assert.equal(calls.length, 0, "a locked view asks nothing more");
    assert.equal(AA.evalsView.state().timer, null);
  } finally {
    console.error = quiet;
    AA.evalsView.stop();
    AA.views.show("runs");
  }
});
