// chatcards.js, chatextras.js, inspector.js: the turn model, cards, upload checks, state and cost.
import assert from "node:assert/strict";
import { test } from "node:test";

import { events } from "./fixtures.mjs";
import { html, loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "chatapi.js", "markdown.js", "chatcards.js", "chatextras.js", "inspector.js");

const turn = (extra) => Object.assign({ id: "turn_1", session_id: "ses_1", run_id: "run_1", text: "hi", attachments: [], status: "running", answer: "", error: null, started: 100, ended: null, input_tokens: 0, output_tokens: 0, cost_usd: 0, tools: [], approvals: [] }, extra || {});
const tool = (extra) => Object.assign({ type: "tool", call_id: "c1", name: "add", status: "running", ms: null }, extra || {});
const approval = (extra) => Object.assign({ type: "approval", approval_id: "apr_1", call_id: "c1", tool: "send", arguments: '{"to": "a"}', hint: "", status: "pending", reason: "", decided_by: "" }, extra || {});

test("stream events fold into text segments around tool and approval cards", () => {
  const model = AA.chatTurn.create(turn());
  const evs = [
    { type: "state", state: { n: 1 } }, { type: "text", text: "Let me " }, { type: "text", text: "check." },
    tool(), approval(), tool({ status: "ok", ms: 3, output: "4" }), approval({ status: "approved", decided_by: "local" }),
    { type: "usage", input_tokens: 10, output_tokens: 2, cost_usd: 0.001 }, { type: "text", text: "It is 4." },
    { type: "done", status: "done", turn: turn({ status: "done", answer: "Let me check.It is 4.", ended: 101, input_tokens: 10, output_tokens: 2, cost_usd: 0.001 }) },
  ];
  evs.forEach((ev) => assert.equal(AA.chatTurn.reduce(model, ev), true, ev.type));
  assert.deepEqual(model.segments.map((s) => s.kind + ":" + (s.text || s.id)), ["text:Let me check.", "tool:c1", "approval:apr_1", "text:It is 4."]);
  assert.equal(model.tools.c1.status, "ok");
  assert.equal(model.approvals.apr_1.status, "approved");
  assert.deepEqual(model.state, { n: 1 });
  assert.equal(model.status, "done");
  assert.equal(model.turn.ended, 101);
  assert.equal(model.error, null);
  assert.equal(AA.chatTurn.answerText(model), "Let me check.It is 4.");
  assert.equal(AA.chatTurn.reduce(model, { type: "text", text: "" }), false);
  assert.equal(AA.chatTurn.reduce(model, { type: "unknown" }), false);
});

test("an error event keeps its kind and message; a stored turn rebuilds its cards", () => {
  const model = AA.chatTurn.create(turn());
  AA.chatTurn.reduce(model, { type: "error", status: "stopped", kind: "stopped", message: "the turn was stopped", turn: turn({ status: "stopped" }) });
  assert.deepEqual(model.error, { kind: "stopped", message: "the turn was stopped" });
  assert.equal(model.status, "stopped");
  const stored = AA.chatTurn.fromStored(turn({
    status: "done", answer: "done.",
    tools: [tool({ status: "ok" }), tool({ call_id: "c2", name: "send", status: "ok" })],
    approvals: [approval({ call_id: "c2", status: "approved" }), approval({ approval_id: "apr_9", call_id: "zz", status: "denied" })],
  }));
  assert.deepEqual(stored.segments.map((s) => s.kind + ":" + (s.text || s.id)), ["tool:c1", "tool:c2", "approval:apr_1", "approval:apr_9", "text:done."]);
});

function view(model, overrides) {
  const calls = [];
  const h = Object.assign({
    copy: () => {}, onFile: (ref) => calls.push(["file", ref.id]), onSelect: () => calls.push(["select"]),
    onEdit: () => calls.push(["edit"]), onAction: (name, _view, button) => calls.push(["action", name, button.textContent]),
    onDecide: (a, approved, reason) => { calls.push(["decide", a.approval_id, approved, reason]); return Promise.resolve(Object.assign({}, a, { status: approved ? "approved" : "denied", decided_by: "local", reason })); },
  }, overrides || {});
  return { v: AA.chatTurn.createView(model, h, { fil_1: { id: "fil_1", name: "notes.txt", size: 3 } }), calls };
}

const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test("a turn view paints cards in place and never renders agent text as HTML", () => {
  const model = AA.chatTurn.create(turn({ text: "<img src=x onerror=alert(1)>", attachments: ["fil_1"] }));
  const { v, calls } = view(model);
  assert.ok(!v.root.all("img").length);
  assert.equal(v.root.all("div").find((d) => d.className === "msg-text").textContent, "<img src=x onerror=alert(1)>");
  AA.chatTurn.reduce(model, { type: "text", text: "**hi** <script>x</script>" });
  AA.chatTurn.reduce(model, tool());
  v.render(101000);
  const card = v.root.all("details")[0];
  assert.match(card.className, /tc-running/);
  AA.chatTurn.reduce(model, tool({ status: "failed", error_kind: "tool-timeout", arguments: '{"a":1}' }));
  v.render(101000);
  assert.equal(v.root.all("details")[0], card, "the same card node, updated in place");
  assert.match(card.className, /tc-failed/);
  assert.match(card.textContent, /tool-timeout/);
  assert.match(card.textContent, /"a": 1/, "arguments pretty-printed");
  assert.equal(v.root.all("script").length, 0);
  assert.equal(v.root.all("strong").filter((s) => s.textContent === "hi").length, 1);
  const buttons = v.root.all("button");
  const byText = (t) => buttons.find((b) => b.textContent === t);
  assert.equal(byText("Replay").disabled, true, "menu waits for the turn to end");
  assert.equal(byText("edit").disabled, true, "a running turn cannot be edited yet");
  assert.equal(byText("Inspect").disabled, false);
  byText("notes.txt").click();
  byText("edit").click();
  byText("Inspect").click();
  assert.deepEqual(calls, [["file", "fil_1"], ["action", "inspect", "Inspect"]]);
  assert.equal(v.root.getAttribute("aria-busy"), "true");
  AA.chatTurn.reduce(model, { type: "done", status: "done", turn: turn({ status: "done", ended: 102 }) });
  v.render();
  assert.equal(v.root.getAttribute("aria-busy"), "false");
  assert.equal(byText("edit").disabled, false);
  byText("Copy").click();
  assert.deepEqual(calls[2], ["action", "copy", "Copy"], "the menu button is passed for feedback");
});

test("the footer is built once: the typing dots and the alert are updated, not re-created", () => {
  const model = AA.chatTurn.create(turn());
  const { v } = view(model);
  const foot = v.root.querySelector(".msg-foot");
  const dots = foot.querySelector(".typing");
  const alert = foot.querySelector(".msg-error");
  assert.equal(dots.hidden, false);
  assert.equal(alert.hidden, true);
  v.render(101500);
  v.render(102500);
  assert.equal(foot.querySelector(".typing"), dots, "the animation is not restarted every second");
  assert.match(foot.textContent, /2\.50 s/);
  AA.chatTurn.reduce(model, { type: "error", status: "failed", kind: "tool-failed", message: "boom", turn: turn({ status: "failed" }) });
  v.render();
  v.render();
  assert.equal(foot.querySelector(".msg-error"), alert);
  assert.equal(alert.hidden, false);
  assert.equal(alert.textContent, "tool-failed: boom");
  assert.equal(dots.hidden, true);
});

function approvalView(onDecide) {
  const model = AA.chatTurn.create(turn());
  AA.chatTurn.reduce(model, approval());
  const out = view(model, onDecide ? { onDecide } : null);
  const card = out.v.root.all("div").find((d) => /^ac /.test(d.className));
  const get = (t) => card.all("button").find((b) => b.textContent === t);
  return Object.assign(out, { model, card, reason: card.all("input")[0], approve: get("Approve"), deny: get("Deny") });
}

test("an approval decides once: both buttons and the reason lock, the answer paints the card", async () => {
  const { v, calls, model, card, reason, approve, deny } = approvalView();
  reason.value = "looks fine";
  approve.click();
  assert.equal(approve.disabled && deny.disabled && reason.disabled, true);
  deny.click();
  assert.deepEqual(calls, [["decide", "apr_1", true, "looks fine"]], "a second click sends nothing");
  await tick();
  assert.match(card.className, /ac-approved/, "decided even without the stream event");
  assert.match(card.textContent, /Approved by local: looks fine/);
  assert.equal(model.approvals.apr_1.status, "approved");
  v.render();
  assert.match(card.className, /ac-approved/);
  AA.chatTurn.reduce(model, approval({ status: "approved", decided_by: "local", reason: "looks fine" }));
  v.render();
  assert.equal(card.all("div").find((d) => d.className === "actions").hidden, true);
});

test("an approval that failed to save unlocks; one already decided stays locked", async () => {
  const flaky = approvalView(() => Promise.reject(Object.assign(new Error("the viewer is not reachable"), { status: 0 })));
  flaky.deny.click();
  await tick();
  assert.equal(flaky.deny.disabled || flaky.approve.disabled || flaky.reason.disabled, false);
  assert.match(flaky.card.textContent, /Not saved: the viewer is not reachable/);
  const taken = approvalView(() => Promise.reject(Object.assign(new Error("already denied"), { status: 409 })));
  taken.approve.click();
  await tick();
  assert.equal(taken.approve.disabled && taken.deny.disabled, true);
  assert.match(taken.card.textContent, /Already decided \(already denied\)/);
});

test("upload checks: sizes always; types only where the built-in loop cannot send them", () => {
  const f = (name, size, type) => ({ name, size, type });
  const MB = 1024 * 1024;
  assert.equal(AA.chatApi.uploadProblem(f("a.txt", 10, "text/plain"), 0, "builtin"), "");
  assert.match(AA.chatApi.uploadProblem(f("a.txt", 0, "text/plain"), 0, "agent"), /empty/);
  assert.match(AA.chatApi.uploadProblem(f("big.bin", 10 * MB + 1, ""), 0, "agent"), /over 10 MB/);
  assert.match(AA.chatApi.uploadProblem(f("b.png", 2 * MB, "image/png"), 49 * MB, "agent"), /50 MB/);
  assert.equal(AA.chatApi.uploadProblem(f("p.pdf", 5, "application/pdf"), 0, "agent"), "");
  assert.match(AA.chatApi.uploadProblem(f("p.pdf", 5, "application/pdf"), 0, "builtin"), /text files only/);
  assert.match(AA.chatApi.uploadProblem(f("b.png", 5, "image/png"), 0, "builtin"), /text files only/);
  for (const [name, type] of [["a.ts", "video/mp2t"], ["a.svg", "image/svg+xml"], ["a.xml", "application/xml"], ["a.yaml", "application/x-yaml"], ["a.json", ""]]) {
    if (type !== "image/svg+xml") assert.equal(AA.chatApi.uploadProblem(f(name, 5, type), 0, "builtin"), "", name + " is left to the server");
    assert.equal(AA.chatApi.uploadProblem(f(name, 5, type), 0, "agent"), "", name);
  }
});

test("state editor input: a JSON object whose request fits in 64 KB", () => {
  assert.deepEqual(AA.chatExtras.parseState('{"a": 1}'), { value: { a: 1 } });
  assert.deepEqual(AA.chatExtras.parseState("  "), { value: {} });
  assert.match(AA.chatExtras.parseState("[1]").error, /JSON object/);
  assert.match(AA.chatExtras.parseState("null").error, /JSON object/);
  assert.match(AA.chatExtras.parseState("{a:1}").error, /not JSON/);
  assert.match(AA.chatExtras.parseState(JSON.stringify({ k: "x".repeat(64 * 1024) })).error, /64 KB/);
  const near = { k: "x".repeat(64 * 1024 - 30) }; // fits compact, not pretty-printed with indents
  assert.ok(JSON.stringify({ state: near }).length <= 64 * 1024);
  assert.deepEqual(AA.chatExtras.parseState(JSON.stringify(near, null, 40)), { value: near });
});

test("the state editor keeps typed text when a turn ends; reset and save refill it", async () => {
  const box = document.createElement("div");
  let state = { a: 1 };
  const editor = AA.chatExtras.createStateEditor(box, { state: () => state, running: () => false, save: async (s) => { state = s; return s; } });
  const area = box.querySelector("textarea");
  editor.fill(false);
  assert.equal(area.value, '{\n  "a": 1\n}');
  area.value = '{"a": 2}';
  area.dispatch("input");
  state = { a: 3 };
  editor.fill(false);
  assert.equal(area.value, '{"a": 2}', "unsaved typing survives a refresh");
  box.all("button").find((b) => b.textContent === "Save state").click();
  await tick();
  assert.equal(editor.dirty(), false);
  state = { a: 4 };
  editor.fill(false);
  assert.equal(area.value, '{\n  "a": 4\n}');
});

test("eval case fields: expected as JSON or text, tags split", () => {
  assert.deepEqual(AA.chatExtras.parseExpected('{"answer": 4}'), { answer: 4 });
  assert.equal(AA.chatExtras.parseExpected("It is 4."), "It is 4.");
  assert.equal(AA.chatExtras.parseExpected("   "), undefined);
  assert.deepEqual(AA.chatExtras.parseTags(" a, ,b ,c"), ["a", "b", "c"]);
  assert.equal(AA.chatExtras.parseTags(Array.from({ length: 30 }, (_, i) => "t" + i).join(",")).length, 20);
});

test("cost rows list model calls and priced tools with totals", () => {
  const run = { events, spans: [{ family: "tool", name: "faq", cost_usd: 0.01, start: 1, end: 1.5 }, { family: "tool", name: "free", cost_usd: null, start: 1, end: 2 }] };
  const found = AA.inspectorCost.rows(run);
  const models = found.rows.filter((r) => r.kind === "model");
  assert.ok(models.length >= 1);
  assert.deepEqual(models[0], { kind: "model", name: "local/m", input: 120, output: 12, cost: 0.002, ms: 2 });
  assert.deepEqual(found.rows.filter((r) => r.kind === "tool").map((r) => [r.name, r.cost, r.ms]), [["faq", 0.01, 500]]);
  assert.equal(found.total.calls, models.length);
  assert.ok(Math.abs(found.total.cost - (0.01 + models.reduce((n, r) => n + (r.cost || 0), 0))) < 1e-12);
  assert.deepEqual(AA.inspectorCost.rows(null), { rows: [], total: { input: 0, output: 0, cost: 0, calls: 0 } });
});

test("pretty prints JSON text and leaves cut or plain text alone", () => {
  assert.equal(AA.chatTurn.pretty('{"a":[1]}'), '{\n  "a": [\n    1\n  ]\n}');
  assert.equal(AA.chatTurn.pretty('{"a": "xx... [+20 chars]'), '{"a": "xx... [+20 chars]');
  assert.equal(html(AA.el("p", { text: AA.chatTurn.pretty("<b>") })), "<p>&lt;b&gt;</p>");
});

test("the state editor never saves a draft into another chat", async () => {
  const box = document.createElement("div");
  let owner = "ses_a";
  const states = { ses_a: { a: 1 }, ses_b: { b: 2 } };
  const saved = [];
  const editor = AA.chatExtras.createStateEditor(box, {
    owner: () => owner, state: () => states[owner], running: () => false,
    save: async (s) => { saved.push([owner, s]); return s; },
  });
  const area = box.querySelector("textarea");
  editor.fill(true);
  area.value = '{"draft": 1}';
  area.dispatch("input");
  owner = "ses_b"; // another chat opened without a refill
  box.all("button").find((b) => b.textContent === "Save state").click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(saved, []);
  assert.deepEqual(JSON.parse(area.value), { b: 2 });
  assert.match(box.textContent, /Another chat is open/);
  editor.fill(false);
  area.value = '{"b": 3}';
  area.dispatch("input");
  editor.fill(false);
  assert.equal(area.value, '{"b": 3}', "a draft for the open chat survives a refresh");
});

test("a draft typed before the first message belongs to the chat it creates", async () => {
  const box = document.createElement("div");
  let owner = "";
  const saved = [];
  const editor = AA.chatExtras.createStateEditor(box, {
    owner: () => owner, state: () => ({}), running: () => false,
    save: async (s) => { owner = "ses_new"; saved.push(s); return s; },
  });
  editor.fill(true);
  const area = box.querySelector("textarea");
  area.value = '{"seed": 1}';
  area.dispatch("input");
  box.all("button").find((b) => b.textContent === "Save state").click();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.deepEqual(saved, [{ seed: 1 }]);
  assert.equal(editor.dirty(), false);
});
