// chatdemo.js: the recorded chat demo replays turns on the real chat page, with a fake clock.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { event, installDom, runScripts } from "./fakedom.mjs";
import { STATIC } from "./load.mjs";

const page = readFileSync(join(STATIC, "index.html"), "utf8");
const section = page.slice(page.indexOf('<section id="chat-view"'), page.indexOf('<section id="runs-view"'));

// ---- a small recording: chat A (a lookup, then a booking to approve), chat B (one turn) ----
const view = (id, sid, text, extra) => Object.assign({ id, session_id: sid, run_id: "run_" + id, text, attachments: [], status: "done", answer: "", error: null, started: 100, ended: 103, input_tokens: 5, output_tokens: 3, cost_usd: 0.001, tools: [], approvals: [], replay_of: "", edit_of: "" }, extra || {});
const tool = (callId, name, status, extra) => Object.assign({ type: "tool", call_id: callId, name, status, ms: null }, extra || {});
const approval = (status, extra) => Object.assign({ type: "approval", approval_id: "apr_1", turn_id: "turn_2", session_id: "ses_a", call_id: "c2", tool: "book", arguments: "{}", hint: "", status, reason: "", decided_by: "", created_at: 1 }, extra || {});
const lookedUp = tool("c1", "lookup", "ok", { ms: 300, arguments: "{}", output: "1" });
const T1 = view("turn_1", "ses_a", "Look it up", { answer: "Hello **there**.", tools: [lookedUp] });
const decided = approval("approved", { reason: "ok", decided_by: "demo" });
const T2 = view("turn_2", "ses_a", "Book it", { answer: "Booked.", tools: [tool("c2", "book", "ok", { ms: 900 })], approvals: [decided] });
const T3 = view("turn_3", "ses_b", "Other question", { answer: "Other answer." });
const steps1 = [
  { at: 0, sse: { type: "state", state: {} } },
  { at: 0, run: 1 },
  { at: 200, run: 2 },
  { at: 400, sse: tool("c1", "lookup", "running") },
  { at: 700, sse: lookedUp },
  { at: 700, run: 4 },
  { at: 900, sse: { type: "text", text: "Hello " } },
  { at: 1000, sse: { type: "text", text: "**there**." } },
  { at: 1100, sse: { type: "state", state: { n: 1 } } },
  { at: 1200, sse: { type: "done", status: "done", turn: T1 } },
];
const steps2 = [
  { at: 0, sse: { type: "state", state: { n: 1 } } },
  { at: 100, sse: tool("c2", "book", "running") },
  { at: 300, sse: approval("pending") },
  { at: 2300, sse: decided },
  { at: 2500, sse: tool("c2", "book", "ok", { ms: 900 }) },
  { at: 2600, sse: { type: "text", text: "Booked." } },
  { at: 2700, sse: { type: "done", status: "done", turn: T2 } },
];
const steps3 = [{ at: 0, sse: { type: "text", text: "Other answer." } }, { at: 3000, sse: { type: "done", status: "done", turn: T3 } }];
const session = (id, title) => ({ id, title, created: 99, updated: 99, turns: 0, running: null, parent: null });
const ev = (type, ts, data) => ({ kind: "event", type, ts, run_id: "run_turn_1", call_id: "", data: data || {} });
const run1 = {
  summary: { key: "run_turn_1", run_id: "run_turn_1", status: "ok", started: 100, ended: 101, events: 4 },
  graphs: { f: { flow: "f", nodes: [], edges: [] } },
  spans: [{ id: "s1", family: "tool", events: [2, 3], status: "ok", start: 100.2, end: 100.3, marks: [], cost_usd: 0.5 }],
  events: [ev("run.started", 100), ev("flow.started", 100.1, { flow: "f" }), ev("tool.started", 100.2), ev("tool.finished", 100.3)],
  events_from: 0,
  events_total: 4,
};
const data = {
  runs: [run1.summary],
  details: { run_turn_1: run1 },
  chat: {
    info: { enabled: true, mode: "demo", agent_name: "help desk" },
    live: "python examples/chat_flows.py",
    chats: [
      { session: session("ses_a", "Chat A"), turns: [{ label: "router", turn: T1, steps: steps1, state: { n: 1 } }, { label: "approval", turn: T2, steps: steps2, state: { n: 2 } }] },
      { session: session("ses_b", "Chat B"), turns: [{ label: "loop", turn: T3, steps: steps3, state: {} }] },
    ],
  },
};

const document = installDom({ html: '<meta name="aa-chat" content="demo"><span id="exported">exported file</span>' + section });
const holder = document.createElement("script");
holder.setAttribute("id", "aa-data");
holder.textContent = JSON.stringify(data);
document.body.appendChild(holder);
globalThis.requestAnimationFrame = (fn) => { setImmediate(fn); return 1; };
globalThis.setInterval = () => 0; // the list refresh and the clock: not needed here
const errors = [];
console.error = (...args) => errors.push(args.map(String).join(" "));
const AA = runScripts("data.js");
const shownViews = [];
AA.views = { register() {}, show: (name) => shownViews.push(name), has: () => true };
// The Runs view: shows the run it was last asked for, as an export loads it (once per pick).
const runsShown = { run: null };
AA.runsView = { pick: async (key) => { runsShown.run = await AA.api.run(key, 0); }, current: () => runsShown.run, loadAll: (key) => AA.api.run(key, 0) };
AA.createInspector = () => ({ show() {}, showTab() {} });
runScripts("chatapi.js", "chatdemo.js", "markdown.js", "chatcards.js", "chatextras.js", "chatsessions.js", "chatcomposer.js", "chat.js");

// A fake clock for the replay: `advance(ms)` runs every timer due by then, in order.
const clock = { now: 0, queue: [], id: 0, reduced: false };
Object.assign(AA.chatDemo.env, {
  setTimeout: (fn, ms) => { clock.id += 1; clock.queue.push({ id: clock.id, at: clock.now + ms, fn }); return clock.id; },
  clearTimeout: (id) => { clock.queue = clock.queue.filter((t) => t.id !== id); },
  reducedMotion: () => clock.reduced,
});
const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((r) => setImmediate(r)); };
async function advance(ms) {
  const end = clock.now + ms;
  for (;;) {
    clock.queue.sort((a, b) => a.at - b.at || a.id - b.id);
    const next = clock.queue[0];
    if (!next || next.at > end) break;
    clock.queue.shift();
    clock.now = next.at;
    next.fn();
    await settle();
  }
  clock.now = end;
  await settle();
}
await AA.chatView.ready;
await settle();

const $ = (id) => document.getElementById(id);
const questions = () => $("chat-main").querySelectorAll(".demo-q");
const log = () => $("chat-log").textContent;
const ask = async (n) => { questions()[n].click(); await settle(); };
const reset = async () => { await AA.chatDemo.startOver(); clock.reduced = false; await settle(); };

test("the page says it is a recorded demo and shows the chat, not the 'no agent' panel", () => {
  assert.equal($("chat-off").hidden, true);
  assert.equal($("chat-shell").hidden, false);
  assert.equal($("chat-composer").hidden, true, "suggested questions replace the composer");
  assert.equal($("chat-new").hidden, true);
  assert.equal($("exported").textContent, "recorded demo");
  assert.match($("chat-main").querySelector(".demo-note").textContent, /Recorded demo\..*python examples\/chat_flows\.py/);
  assert.deepEqual(questions().map((b) => b.tagName), ["BUTTON", "BUTTON", "BUTTON"], "real buttons");
  assert.deepEqual(questions().map((b) => b.getAttribute("aria-label")), ["router: Look it up", "approval: Book it", "loop: Other question"]);
  assert.equal($("chat-log").getAttribute("aria-live"), "polite");
  assert.ok(shownViews.includes("chat"));
  assert.match($("chat-head").textContent, /Chat A/);
});

test("a question replays its turn in order: user bubble, tool card, streamed text, then done", async () => {
  await ask(0);
  assert.match(log(), /Look it up/);
  assert.doesNotMatch(log(), /lookup/);
  assert.equal(AA.chatDemo.state.running.ses_a, "turn_1");
  assert.equal(questions()[1].getAttribute("aria-disabled"), "true", "one replay at a time");
  await advance(450);
  assert.match(log(), /lookup/);
  assert.match(log(), /running/);
  let run = await AA.api.run("run_turn_1", 0);
  assert.equal(run.summary.status, "running");
  assert.deepEqual(run.events.map((e) => e.type), ["run.started", "flow.started"], "the run so far");
  assert.deepEqual(Object.keys(run.graphs), ["f"], "the graph shows once its flow started");
  assert.deepEqual(run.spans, [], "no call has started yet");
  await advance(500);
  assert.match(log(), /Hello/);
  assert.doesNotMatch(log(), /there/, "the answer streams in chunks");
  run = await AA.api.run("run_turn_1", 0);
  assert.equal(run.spans[0].status, "ok");
  await advance(300);
  assert.match(log(), /Hello there\./);
  assert.match(log(), /done/);
  assert.equal(AA.chatDemo.state.played.turn_1, true);
  assert.equal(questions()[0].getAttribute("aria-disabled"), "true");
  assert.match(questions()[0].getAttribute("aria-label"), /\(played\)/);
  assert.equal(questions()[1].getAttribute("aria-disabled"), "false");
  assert.equal(document.activeElement, questions()[0], "focus stays on the question");
  run = await AA.api.run("run_turn_1", 0);
  assert.equal(run.summary.status, "ok", "a finished turn's run is the whole recording");
  assert.equal(run.events.length, 4);
  assert.deepEqual(errors, []);
});

test("finished turns stay browsable: Inspect, Compare and Copy; never Replay, edit or eval", async () => {
  const menu = $("chat-log").querySelector(".msg-menu");
  assert.deepEqual(menu.querySelectorAll("button").map((b) => b.textContent), ["Inspect", "Compare", "Copy"]);
  const edit = $("chat-log").querySelectorAll("button").find((b) => b.textContent === "edit");
  assert.equal(edit.hidden, true);
});

test("the approval card waits, then resolves as the recording decided, by itself", async () => {
  await ask(1);
  await advance(350);
  const card = $("chat-log").querySelector(".ac");
  assert.match(card.textContent, /Approve this call\?/);
  assert.equal(card.querySelector(".actions").hidden, false);
  assert.match(card.querySelector(".actions").textContent, /Recorded demo: the recorded decision arrives by itself/);
  await advance(1000);
  assert.match($("chat-log").querySelector(".ac").textContent, /Approve this call\?/, "still waiting");
  await advance(1000);
  assert.match($("chat-log").querySelector(".ac").textContent, /Approved by demo: ok/);
  await advance(500);
  assert.match(log(), /Booked\./);
  assert.equal(AA.chatDemo.state.played.turn_2, true);
  assert.deepEqual(errors, []);
});

test("Approve and Deny are both inert in the demo: neither click changes the card", async () => {
  await reset();
  await ask(1); // turn 1 shows as finished first: the chat's history
  assert.equal(AA.chatDemo.state.played.turn_1, true);
  assert.match(log(), /Hello there\./);
  await advance(350);
  const card = () => $("chat-log").querySelectorAll(".ac")[0];
  const buttons = card().querySelectorAll("button");
  assert.deepEqual(buttons.map((b) => [b.textContent, b.disabled]), [["Approve", true], ["Deny", true]]);
  assert.equal(card().querySelector("input").disabled, true);
  buttons.forEach((b) => b.click());
  await settle();
  assert.doesNotMatch(card().textContent, /Already decided|Not saved|saving/);
  assert.match(card().textContent, /Approve this call\?/, "still waiting for the recording");
  await assert.rejects(AA.chatApi.decide("apr_1", true), (err) => err.status === 409);
  await advance(2500);
  assert.match(card().textContent, /Approved by demo: ok/);
  assert.match(log(), /Booked\./);
  assert.deepEqual(errors, []);
});

test("with reduced motion a question shows its whole turn at once", async () => {
  await reset();
  clock.reduced = true;
  await ask(2);
  await advance(0);
  assert.match(log(), /Other answer\./);
  assert.match(log(), /done/);
  assert.equal(AA.chatDemo.state.played.turn_3, true);
  assert.equal(clock.queue.length, 0, "no timer left");
  clock.reduced = false;
});

test("actions the demo has no server for are refused cleanly, never thrown", async () => {
  const api = AA.chatApi;
  for (const call of [() => api.replay("turn_1"), () => api.evalCase("turn_1", {}), () => api.createSession("x"), () => api.setState("ses_a", {}), () => api.upload("ses_a", {}), () => api.fileBlob("fil_1"), () => api.stop("turn_1")]) {
    await assert.rejects(call(), (err) => err.status === 409);
  }
  await assert.rejects(api.startTurn("ses_a", { text: "anything else" }), (err) => err.status === 400 || err.status === 409);
  await assert.rejects(api.session("ses_nope"), (err) => err.status === 404);
  assert.match(api.uploadProblem({ size: 1 }), /not part of the recorded demo/);
  assert.equal(api.supports("replay") || api.supports("eval") || api.supports("edit"), false);
  assert.equal(api.supports("compare"), true);
  await reset();
  await ask(0);
  $("chat-input").dispatchEvent(event("keydown", { key: "Escape", target: $("chat-input") })); // Esc stops a live turn
  await settle();
  assert.equal($("chat-error").textContent, "");
  await advance(5000);
  assert.match(log(), /Hello there\./, "the turn plays to its end");
  assert.deepEqual(errors, []);
});

test("switching chats mid-replay cancels it cleanly; the question can play again", async () => {
  await reset();
  await ask(0);
  await advance(450);
  assert.ok(AA.chatDemo.state.player, "replaying");
  $("chat-session-list").querySelectorAll("button").find((b) => b.dataset.sid === "ses_b").click();
  await settle();
  await advance(0);
  assert.equal(AA.chatDemo.state.player, null);
  assert.deepEqual(AA.chatDemo.state.running, {});
  assert.equal(AA.chatDemo.state.played.turn_1, undefined, "a cut-short turn is not played");
  assert.equal(clock.queue.length, 0, "its timers are gone");
  assert.equal((await AA.api.run("run_turn_1", 0)).summary.status, "ok", "its run is the recording again");
  assert.match($("chat-head").textContent, /Chat B/);
  assert.equal(questions()[0].getAttribute("aria-disabled"), "false");
  await advance(5000);
  $("chat-session-list").querySelectorAll("button").find((b) => b.dataset.sid === "ses_a").click();
  await settle();
  assert.doesNotMatch(log(), /Look it up/, "chat A holds no half turn");
  await ask(0);
  await advance(2000);
  assert.match(log(), /Hello there\./);
  assert.deepEqual(errors, []);
});

const chatButton = (sid) => $("chat-session-list").querySelectorAll("button").find((b) => b.dataset.sid === sid);
const turnsListed = (sid) => chatButton(sid).querySelector(".meta").textContent.split(" · ")[0];

test("reopening the playing chat keeps its replay: same turn, buttons busy, a short clock", async () => {
  await reset();
  await ask(0);
  await advance(450);
  const player = AA.chatDemo.state.player;
  chatButton("ses_a").click();
  await settle();
  await advance(0);
  assert.equal(AA.chatDemo.state.player, player, "the replay goes on");
  assert.deepEqual(questions().map((b) => b.getAttribute("aria-disabled")), ["true", "true", "true"]);
  assert.match(questions()[0].getAttribute("aria-label"), /\(playing\)/);
  assert.match(log(), /lookup/, "what played so far is shown again");
  const foot = $("chat-log").querySelector(".msg-foot").textContent;
  assert.match(foot, /running/);
  assert.doesNotMatch(foot, /min/, "the clock counts from the pick, not the recording's date");
  assert.equal($("chat-log").querySelectorAll(".turn").length, 1);
  await advance(2000);
  assert.match(log(), /Hello there\./);
  assert.equal(($("chat-log").textContent.match(/Hello there\./g) || []).length, 1, "the answer once");
  assert.equal(AA.chatDemo.state.played.turn_1, true);
  assert.deepEqual(errors, []);
});

test("the Runs view showing a replaying run follows it and ends on the whole recording", async () => {
  await reset();
  await ask(0);
  await advance(450);
  await AA.runsView.pick("run_turn_1");
  assert.equal(runsShown.run.summary.status, "running");
  assert.equal(runsShown.run.events.length, 2);
  await advance(300);
  assert.equal(runsShown.run.events.length, 4, "it loads again as events arrive");
  await advance(2000);
  assert.equal(runsShown.run.summary.status, "ok");
  assert.equal(runsShown.run.events.length, 4);
  runsShown.run = null;
});

test("chat lists count the playing turn, and Start over mid-replay clears turns, timers and counts", async () => {
  await reset();
  assert.equal(turnsListed("ses_a"), "0 turns");
  await ask(1); // turn 1 shows as played, turn 2 plays
  await advance(100);
  assert.equal(turnsListed("ses_a"), "2 turns", "the playing turn counts, as on a live server");
  assert.match(chatButton("ses_a").textContent, /running/);
  await AA.chatDemo.startOver();
  await settle();
  await advance(0);
  assert.equal(AA.chatDemo.state.player, null);
  assert.deepEqual(AA.chatDemo.state.played, {});
  assert.deepEqual(AA.chatDemo.state.running, {});
  assert.equal(clock.queue.length, 0, "no replay timer left");
  assert.equal(turnsListed("ses_a"), "0 turns", "the list is read again at once");
  assert.doesNotMatch(chatButton("ses_a").textContent, /running/);
  assert.doesNotMatch(log(), /Look it up|Book it/);
  assert.deepEqual(questions().map((b) => b.getAttribute("aria-disabled")), ["false", "false", "false"]);
  assert.deepEqual(errors, []);
});

test("errors stay visible: a dropped file is refused in the demo bar, not the hidden composer", async () => {
  const drop = event("drop", { dataTransfer: { types: ["Files"], files: [{ name: "a.txt", size: 3, type: "text/plain" }] } });
  $("chat-main").dispatchEvent(drop);
  await settle();
  const error = $("chat-error");
  assert.match(error.textContent, /attaching files is not part of the recorded demo/);
  assert.ok(error.closest(".demo-bar"), "shown in the demo bar");
  assert.equal(error.closest("#chat-composer"), null);
  $("chat-error").textContent = "";
});

test("Compare in a one-turn chat says there is nothing to compare, never to replay", async () => {
  clock.reduced = true;
  await ask(2);
  await advance(0);
  clock.reduced = false;
  $("chat-log").querySelector(".msg-menu").querySelectorAll("button").find((b) => b.textContent === "Compare").click();
  await settle();
  const dialog = document.body.querySelector("dialog");
  const picker = dialog.querySelector("select");
  assert.match(picker.textContent, /no other finished turn in this chat to compare with/);
  assert.doesNotMatch(dialog.textContent, /replay/i);
  assert.equal(picker.disabled, true);
  dialog.close();
  assert.deepEqual(errors, []);
});
