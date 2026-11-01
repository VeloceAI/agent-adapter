// chat.js, chatcomposer.js and chatsessions.js on the page's real markup, with a fake chat API.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { event, installDom, runScripts } from "./fakedom.mjs";
import { STATIC } from "./load.mjs";

const page = readFileSync(join(STATIC, "index.html"), "utf8");
const section = page.slice(page.indexOf('<section id="chat-view"'), page.indexOf('<section id="runs-view"'));
const document = installDom({
  hash: "#token=t1",
  html: '<meta name="aa-chat" content="builtin"><meta name="aa-csrf" content="c1">' + section.replace(" hidden>", ">"),
});
globalThis.requestAnimationFrame = (fn) => { setImmediate(fn); return 1; };
globalThis.sessionStorage = { getItem: () => null, setItem() {} };
globalThis.setInterval = () => 0; // the 5 s list refresh and the clock: driven by hand here
const AA = runScripts("data.js");
AA.views = { register() {}, show() {} };
AA.runsView = { pick() {} };
const inspected = [];
AA.createInspector = () => ({ show: (turn) => inspected.push(turn), showTab() {} });
runScripts("chatapi.js", "markdown.js", "chatcards.js", "chatextras.js", "chatsessions.js", "chatcomposer.js");

const settle = async () => { for (let i = 0; i < 6; i += 1) await new Promise((r) => setImmediate(r)); };
const $ = (id) => document.getElementById(id);
const err = (status, message) => Object.assign(new Error(message), { status });
const turn = (id, extra) => Object.assign({ id, session_id: "ses_a", run_id: "run_" + id, text: "q " + id, attachments: [], status: "done", answer: "a " + id, error: null, started: 1, ended: 2, input_tokens: 0, output_tokens: 0, cost_usd: 0, tools: [], approvals: [], replay_of: "", edit_of: "" }, extra || {});
const summary = (id, extra) => Object.assign({ id, title: "chat " + id, created: 1, updated: 1, turns: 1, running: null, parent: null }, extra || {});

// The fake server: what each call answers, and what it was sent.
const api = { calls: [], streams: {}, details: {}, list: [], next: 0 };
api.details.ses_a = { session: summary("ses_a"), state: {}, turns: [turn("turn_1", { attachments: ["fil_1"] })], files: [{ id: "fil_1", name: "notes.txt", media_type: "text/plain", size: 9 }], branches: [] };
api.list = [summary("ses_a"), summary("ses_b")];
api.details.ses_b = { session: summary("ses_b"), state: { b: 2 }, turns: [turn("turn_b1", { session_id: "ses_b" })], files: [], branches: [] };
Object.assign(AA.chatApi, {
  info: async () => ({ enabled: true, mode: "builtin", agent_name: "local:fake" }),
  sessions: async () => { if (api.sessionsFail) throw api.sessionsFail; return api.list.slice(); },
  session: async (id) => api.details[id],
  createSession: async () => assert.fail("no new session in these tests"),
  startTurn: async (sid, body) => {
    api.calls.push(["start", sid, body]);
    if (api.startFail) throw api.startFail;
    api.next += 1;
    const id = "turn_n" + api.next;
    const target = body.edit_of ? "ses_branch" : sid;
    if (body.edit_of) api.details.ses_branch = { session: summary("ses_branch", { parent: { session_id: "ses_a", turn_id: body.edit_of } }), state: {}, turns: [turn(id, { session_id: "ses_branch", status: "running", text: body.text })], files: api.details.ses_a.files, branches: [] };
    return { turn_id: id, run_id: "run_" + id, session_id: target };
  },
  stop: async (id) => { api.calls.push(["stop", id]); return { turn_id: id, stopping: true }; },
  upload: async (sid, file) => {
    if (!api.uploads) assert.fail("nothing over the caps is uploaded");
    api.uploads.push([sid, file.name]);
    return { attachment_id: "fil_" + file.name.replace(/\W/g, ""), name: file.name, size: file.size, media_type: "text/plain" };
  },
  setState: async (sid, state) => { api.calls.push(["state", sid, state]); return { state }; },
  streamTurn: (id, since, handlers) => { api.streams[id] = handlers; return { close() { api.calls.push(["close", id]); } }; },
});
runScripts("chat.js");
await AA.chatView.ready;
await settle();

function type(text) { $("chat-input").value = text; }
function key(target, name, extra) { const ev = event("keydown", Object.assign({ key: name, target }, extra || {})); target.dispatchEvent(ev); return ev; }
const lastStart = () => api.calls.filter((c) => c[0] === "start").pop();

test("the newest chat opens with its turns", () => {
  assert.match($("chat-log").textContent, /a turn_1/);
  assert.match($("chat-head").textContent, /chat ses_a/);
  assert.equal($("chat-send").hidden, false);
});

test("Enter sends; Shift+Enter and IME composition do not", async () => {
  type("hello");
  key($("chat-input"), "Enter", { shiftKey: true });
  key($("chat-input"), "Enter", { isComposing: true });
  await settle();
  assert.equal(lastStart(), undefined);
  key($("chat-input"), "Enter");
  await settle();
  assert.deepEqual(lastStart(), ["start", "ses_a", { text: "hello" }]);
  assert.equal($("chat-input").value, "");
  assert.equal($("chat-send").hidden, true, "Stop replaces Send while the turn runs");
  assert.equal($("chat-stop").hidden, false);
});

test("Esc typed in an approval's reason does not stop the turn; Esc elsewhere does", async () => {
  const handlers = api.streams.turn_n1;
  handlers.onEvent({ seq: 0, type: "approval", approval_id: "apr_1", call_id: "c", tool: "send", arguments: "{}", status: "pending", reason: "", decided_by: "", hint: "" });
  await settle();
  const reason = $("chat-log").querySelector(".ac").querySelector("input");
  key(reason, "Escape");
  await settle();
  assert.equal(api.calls.filter((c) => c[0] === "stop").length, 0);
  key($("chat-input"), "Escape");
  await settle();
  assert.deepEqual(api.calls.filter((c) => c[0] === "stop"), [["stop", "turn_n1"]]);
  assert.equal($("chat-stop").textContent, "Stopping...");
});

test("a reconnect notice clears when the stream reopens, other messages stay", async () => {
  const handlers = api.streams.turn_n1;
  handlers.onStatus("retrying");
  assert.match($("chat-error").textContent, /reconnecting/);
  handlers.onStatus("open");
  assert.equal($("chat-error").textContent, "");
  $("chat-error").textContent = "Not sent: busy";
  handlers.onStatus("open");
  assert.equal($("chat-error").textContent, "Not sent: busy");
});

test("a lost stream frees the composer: the turn shows as stored, never stuck running", async () => {
  api.details.ses_a.turns.push(turn("turn_n1", { status: "stopped", answer: "", error: { kind: "stopped", message: "the turn was stopped" } }));
  api.streams.turn_n1.onFatal(err(404, "no such turn"));
  await settle();
  assert.equal($("chat-send").hidden, false);
  assert.equal($("chat-stop").hidden, true);
  assert.match($("chat-log").textContent, /stopped: the turn was stopped/);
  type("again");
  key($("chat-input"), "Enter");
  await settle();
  assert.deepEqual(lastStart()[2], { text: "again" }, "a new turn can be sent");
});

test("a lost stream of a turn the server still runs shows as lost", async () => {
  api.streams.turn_n2.onFatal(err(0, "lost the connection to the viewer"));
  await settle();
  assert.match($("chat-log").textContent, /connection lost/);
  assert.match($("chat-log").textContent, /stream-lost/);
  assert.equal($("chat-send").hidden, false);
});

test("edit resends the text and the turn's files in a new branch", async () => {
  const edit = $("chat-log").querySelectorAll("button").find((b) => b.textContent === "edit");
  edit.click();
  assert.equal($("chat-input").value, "q turn_1");
  assert.match($("chat-editing").textContent, /files attached again/);
  assert.match($("chat-attachments").textContent, /notes\.txt · 9 B/);
  type("q turn_1, edited");
  key($("chat-input"), "Enter");
  await settle();
  assert.deepEqual(lastStart(), ["start", "ses_a", { text: "q turn_1, edited", attachment_ids: ["fil_1"], edit_of: "turn_1" }]);
  assert.match($("chat-head").textContent, /Branch of/);
  assert.equal($("chat-attachments").textContent, "");
});

test("a file over the size cap is never sent", async () => {
  $("chat-file").files = [{ name: "huge.txt", size: 11 * 1024 * 1024, type: "text/plain" }];
  $("chat-file").dispatchEvent(event("change"));
  await settle();
  assert.match($("chat-error").textContent, /huge\.txt is over 10 MB/);
});

test("the sessions list keeps its buttons and focus when nothing changed", () => {
  const box = document.createElement("div");
  document.body.appendChild(box);
  const opened = [];
  const list = AA.createSessionList(box, (sid) => opened.push(sid));
  assert.equal(list.update([summary("s1"), summary("s2")], "s1"), true);
  const second = box.querySelectorAll("button")[1];
  second.focus();
  assert.equal(list.update([summary("s1"), summary("s2")], "s1"), false, "same data: nothing redrawn");
  assert.equal(box.querySelectorAll("button")[1], second);
  assert.equal(list.update([summary("s0"), summary("s1"), summary("s2", { turns: 2 })], "s1"), true);
  assert.notEqual(box.querySelectorAll("button")[2], second);
  assert.equal(document.activeElement, box.querySelectorAll("button")[2], "focus follows the chat");
  document.activeElement.click();
  assert.deepEqual(opened, ["s2"]);
});

const openChat = async (sid) => {
  $("chat-session-list").querySelectorAll("button").find((b) => b.dataset.sid === sid).click();
  await settle();
};

test("a state draft stays with its chat: another chat shows its own state, Save never carries it over", async () => {
  await openChat("ses_a");
  $("chat-state-toggle").click();
  const area = $("chat-state").querySelector("textarea");
  area.value = '{"draft": 1}';
  area.dispatchEvent(event("input"));
  await openChat("ses_b");
  assert.deepEqual(JSON.parse(area.value), { b: 2 }, "chat B's state, not chat A's draft");
  $("chat-state").querySelectorAll("button").find((b) => b.textContent === "Save state").click();
  await settle();
  const saved = api.calls.filter((c) => c[0] === "state");
  assert.ok(saved.every((c) => !("draft" in c[2])), "the draft is never written");
  assert.ok(saved.every((c) => c[1] !== "ses_b" || c[2].b === 2));
  $("chat-state-toggle").click();
});

test("edit keeps files attached before it, counts stored files once, and cancel drops only its own", async () => {
  await openChat("ses_a");
  api.uploads = [];
  const pick = async (files) => {
    $("chat-file").files = files;
    $("chat-file").dispatchEvent(event("change"));
    await settle();
  };
  await pick([{ name: "extra.txt", size: 5, type: "text/plain" }]);
  const edit = () => $("chat-log").querySelectorAll("button").find((b) => b.textContent === "edit").click();
  edit();
  assert.match($("chat-attachments").textContent, /extra\.txt.*notes\.txt/);
  $("chat-editing").querySelector("button").click(); // cancel
  assert.match($("chat-attachments").textContent, /extra\.txt/);
  assert.doesNotMatch($("chat-attachments").textContent, /notes\.txt/);
  edit();
  edit(); // twice: still one of each
  assert.equal($("chat-attachments").textContent.match(/notes\.txt/g).length, 1);
  // 50 MB minus the 9 B stored file and the 5 B new one: fits only if notes.txt counts once.
  const MB = 1024 * 1024;
  const big = [1, 2, 3, 4].map((n) => ({ name: "big" + n + ".txt", size: 10 * MB, type: "text/plain" }));
  big.push({ name: "big5.txt", size: 10 * MB - 14, type: "text/plain" });
  await pick(big);
  assert.equal($("chat-error").textContent, "");
  assert.deepEqual(api.uploads.map((u) => u[1]), ["extra.txt"].concat(big.map((f) => f.name)));
  key($("chat-input"), "Enter");
  await settle();
  assert.deepEqual(lastStart()[2].attachment_ids, ["fil_extratxt", "fil_1"].concat(big.map((f) => "fil_" + f.name.replace(".", ""))));
  api.uploads = null;
});

test("a refused token locks the page with a clear message", async () => {
  await openChat("ses_a");
  api.startFail = err(401, "bad token");
  type("locked out");
  key($("chat-input"), "Enter");
  await settle();
  assert.match($("chat-error").textContent, /refused the access token/);
});
