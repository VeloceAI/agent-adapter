// chatmemory.js, approval memory lines and memory spans: recall, delete, advice that never decides.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "graphlive.js", "graphnav.js", "graph.js", "graphbar.js", "timeline.js", "events.js", "statepanel.js", "compare.js", "panels.js", "views.js", "chatapi.js", "markdown.js", "chatcards.js", "chatmemory.js", "inspector.js");

const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise((r) => setImmediate(r)); };
const turn = (extra) => Object.assign({ id: "turn_1", session_id: "ses_1", run_id: "run_1", text: "hi", attachments: [], status: "done", answer: "", error: null, started: 1, ended: 2, input_tokens: 0, output_tokens: 0, cost_usd: 0, tools: [], approvals: [] }, extra || {});
const record = (id, text, extra) => Object.assign({ id, text, kind: "fact", chat: "ses_0", created_at: 10 }, extra || {});
const CONTEXT = { similar: 3, approved: 2, denied: 1, summary: "3 similar: 2 approved, 1 denied (by ann)" };

function fakeApi(listing) {
  const calls = [];
  AA.chatApi.memory = async () => { calls.push(["memory"]); return JSON.parse(JSON.stringify(listing.now)); };
  AA.chatApi.forget = async (ids) => {
    calls.push(["forget", ids]);
    if (listing.fail) throw Object.assign(new Error("backend down"), { status: 502 });
    listing.now.records = listing.now.records.filter((r) => ids.indexOf(r.id) < 0);
    return { deleted: ids.length };
  };
  return calls;
}

test("an approval context line says it is history only, and is absent without similar decisions", () => {
  assert.equal(AA.approvalContext(null), null);
  assert.equal(AA.approvalContext({ similar: 0, summary: "no similar decisions yet" }), null);
  const line = AA.approvalContext(CONTEXT);
  assert.equal(line.textContent, "Seen before: 3 similar: 2 approved, 1 denied (by ann). History only: this call still needs a decision.");
});

test("a pending chat approval card shows the context and still offers both buttons", () => {
  const model = AA.chatTurn.create(turn({ status: "running" }));
  AA.chatTurn.reduce(model, { type: "approval", approval_id: "apr_1", call_id: "c1", tool: "refund", arguments: "{}", hint: "", status: "pending", reason: "", decided_by: "", context: CONTEXT });
  const view = AA.chatTurn.createView(model, { onDecide: () => Promise.resolve(null), onSelect() {}, onAction() {}, onEdit() {}, onFile() {}, copy() {} }, {});
  const card = view.root.all("div").find((d) => /^ac /.test(d.className));
  assert.match(card.textContent, /Seen before: 3 similar/);
  assert.ok(/ac-pending/.test(card.className), "the call still waits for a person");
  const buttons = card.all("button").map((b) => b.textContent);
  assert.deepEqual(buttons, ["Approve", "Deny"]);
  assert.ok(card.all("button").every((b) => !b.disabled));
});

test("the memory event folds the recalled records and the save result into the turn model", () => {
  const model = AA.chatTurn.create(turn({ status: "running" }));
  assert.equal(AA.chatTurn.reduce(model, { type: "memory", recalled: [record("mem_1", "likes tea")] }), true);
  assert.equal(AA.chatTurn.reduce(model, { type: "memory", saved: 2 }), true);
  assert.deepEqual(model.memories.map((r) => r.id), ["mem_1"]);
  assert.deepEqual(model.memorySaved, { saved: 2, error: "" });
});

test("with memory off the panel says how to turn it on", async () => {
  fakeApi({ now: { memory: { enabled: false }, records: [] } });
  const box = document.createElement("div");
  const panel = AA.createMemoryPanel(box);
  panel.show(turn());
  await settle();
  assert.match(box.textContent, /Memory is off\. Start the viewer with --memory/);
});

test("the panel lists what a turn recalled, as text, and the user's records", async () => {
  const listing = { now: { memory: { enabled: true, user: "ann", backend: "sqlite", save: "facts" }, records: [record("mem_1", "<b>likes tea</b>"), record("mem_2", "lives in Lyon", { kind: "summary" })] } };
  fakeApi(listing);
  const box = document.createElement("div");
  const panel = AA.createMemoryPanel(box);
  panel.show(turn({ memories: [{ id: "mem_1", score: 0.5 }, { id: "mem_gone", score: 0.2 }], memory_saved: 1 }));
  await settle();
  assert.ok(!box.all("b").length, "record text is never parsed as HTML");
  assert.match(box.textContent, /Recalled for this turn/);
  assert.match(box.textContent, /<b>likes tea<\/b>/);
  assert.match(box.textContent, /\(deleted or expired\)/);
  assert.match(box.textContent, /score 0\.50/);
  assert.match(box.textContent, /Saved 1 record\(s\) after this turn\./);
  assert.match(box.textContent, /Memory of ann/);
  assert.equal(box.all("button").filter((b) => b.textContent === "Delete").length, 2);
});

test("live recalled texts win over stored ids, and a failed save is shown", async () => {
  fakeApi({ now: { memory: { enabled: true, user: "ann" }, records: [] } });
  const box = document.createElement("div");
  const panel = AA.createMemoryPanel(box);
  panel.show(turn({ memories: [{ id: "mem_x" }], live_memories: [record("mem_x", "live text")], memory_result: { saved: 0, error: "IntegrationError" } }));
  await settle();
  assert.match(box.textContent, /live text/);
  assert.match(box.textContent, /Saving memories failed \(IntegrationError\)\./);
  assert.match(box.textContent, /No records yet\./);
});

test("Delete forgets one record and lists again; a failure says so and keeps the button", async () => {
  const listing = { now: { memory: { enabled: true, user: "ann" }, records: [record("mem_1", "a"), record("mem_2", "b")] } };
  const calls = fakeApi(listing);
  const box = document.createElement("div");
  const panel = AA.createMemoryPanel(box);
  panel.show(null);
  await settle();
  assert.match(box.textContent, /Pick a turn/);
  box.all("button").filter((b) => b.textContent === "Delete")[0].click();
  await settle();
  assert.deepEqual(calls.filter((c) => c[0] === "forget"), [["forget", ["mem_1"]]]);
  assert.equal(box.all("button").filter((b) => b.textContent === "Delete").length, 1);
  assert.match(box.textContent, /Deleted\./);
  listing.fail = true;
  const last = box.all("button").find((b) => b.textContent === "Delete");
  last.click();
  await settle();
  assert.match(box.textContent, /Not deleted: backend down/);
  box.all("button").find((b) => b.textContent === "Refresh").click();
  await settle();
  assert.equal(calls.filter((c) => c[0] === "memory").length, 3);
});

test("a failed listing shows an error", async () => {
  AA.chatApi.memory = async () => { throw Object.assign(new Error("nope"), { status: 502 }); };
  const box = document.createElement("div");
  const errors = [];
  const original = console.error;
  console.error = (...args) => errors.push(args);
  try {
    AA.createMemoryPanel(box).show(turn());
    await settle();
  } finally {
    console.error = original;
  }
  assert.match(box.textContent, /Could not load the memory: nope/);
  assert.equal(errors.length, 1);
});

test("the inspector has a Memory tab that shows the panel", async () => {
  fakeApi({ now: { memory: { enabled: false }, records: [] } });
  AA.runsView = { loadMore: async () => ({ summary: { status: "ok" }, events: [], spans: [], graphs: {} }) };
  globalThis.setInterval = () => 1;
  globalThis.clearInterval = () => {};
  const box = document.createElement("div");
  const insp = AA.createInspector(box, () => {}, () => true);
  assert.ok(box.all("button").some((b) => b.textContent === "Memory"));
  insp.showTab("memory");
  insp.show(turn(), null);
  await settle();
  assert.match(box.querySelector("#insp-memory").textContent, /Memory is off/);
});

test("memory spans show how many records they touched", () => {
  assert.deepEqual(AA.spanChips({ family: "memory", count: 3 }), [["3 record(s)", ""]]);
  assert.deepEqual(AA.spanChips({ family: "tool", count: 3 }), []);
});

test("the Runs view's approval list shows the context next to the decision buttons", () => {
  const box = document.createElement("div");
  const pending = { approval_id: "apr_1", tool: "refund", run_id: "run_1", created_at: 1, hint: "", arguments: { order: "o-1" }, decidable: true, context: CONTEXT };
  AA.renderApprovals(box, [pending, Object.assign({}, pending, { approval_id: "apr_2", context: { similar: 0 } })], () => {});
  assert.equal(box.all("p").filter((p) => /Seen before/.test(p.textContent)).length, 1);
  assert.equal(box.all("button").filter((b) => b.textContent === "Approve").length, 2);
});
