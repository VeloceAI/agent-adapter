// chatcards.js: tool arguments streamed into a draft card, which becomes the call's card.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDom } from "./fakedom.mjs";

const { AA } = loadDom("data.js", "chatapi.js", "markdown.js", "chatcards.js");

const turn = () => ({ id: "turn_1", session_id: "ses_1", run_id: "run_1", text: "hi", attachments: [], status: "running", answer: "", error: null, started: 100, ended: null, tools: [], approvals: [] });
const noop = () => {};
const handlers = { onEdit: noop, onAction: noop, onSelect: noop, onDecide: noop, onFile: noop, copy: noop };

test("a draft card fills as arguments arrive, then becomes the call's card in place", () => {
  const model = AA.chatTurn.create(turn());
  AA.chatTurn.reduce(model, { type: "text", text: "Booking." });
  assert.equal(AA.chatTurn.reduce(model, { type: "tool_draft", index: 0, call_id: "c1", name: "book", arguments: "" }), true);
  AA.chatTurn.reduce(model, { type: "tool_draft", index: 0, call_id: "c1", name: "book", arguments: '{"seats": 2}' });
  assert.deepEqual(model.segments.map((s) => s.id || s.kind), ["text", "draft:0"]);
  assert.equal(model.tools["draft:0"].status, "streaming");
  assert.equal(model.tools["draft:0"].arguments, '{"seats": 2}');
  const view = AA.chatTurn.createView(model, handlers, {});
  const card = view.root.querySelector(".tc");
  assert.equal(card.className, "tc tc-streaming");
  assert.match(card.textContent, /seats/);
  AA.chatTurn.reduce(model, { type: "tool_draft", index: 0, call_id: "c1", closed: true });
  AA.chatTurn.reduce(model, { type: "tool", call_id: "c1", name: "book", status: "running", ms: null, arguments: '{"seats": 2}' });
  assert.deepEqual(model.segments.map((s) => s.id || s.kind), ["text", "c1"]);
  assert.equal(model.tools["draft:0"], undefined);
  view.render();
  assert.equal(view.root.querySelectorAll(".tc").length, 1);
});

test("an unmatched draft is removed when closed or when the turn ends", () => {
  const model = AA.chatTurn.create(turn());
  AA.chatTurn.reduce(model, { type: "tool", call_id: "c1", name: "book", status: "ok" });
  AA.chatTurn.reduce(model, { type: "tool_draft", index: 0, call_id: "", name: "book", arguments: "" });
  AA.chatTurn.reduce(model, { type: "tool_draft", index: 0, call_id: "c1", closed: true }); // c1 already shown
  assert.deepEqual(model.segments.map((s) => s.id), ["c1"]);
  assert.equal(AA.chatTurn.reduce(model, { type: "tool_draft", index: 5, closed: true }), false);
  AA.chatTurn.reduce(model, { type: "tool_draft", index: 1, call_id: "", name: "x", arguments: "{}" });
  AA.chatTurn.reduce(model, { type: "done", status: "done" });
  assert.deepEqual(model.segments.map((s) => s.id), ["c1"]);
  assert.deepEqual(Object.keys(model.tools), ["c1"]);
});

test("draft names and arguments are shown as text, never as markup", () => {
  const model = AA.chatTurn.create(turn());
  AA.chatTurn.reduce(model, { type: "tool_draft", index: 0, call_id: "", name: "<img src=x onerror=alert(1)>", arguments: '{"q": "<b>x</b>"}' });
  const view = AA.chatTurn.createView(model, handlers, {});
  const card = view.root.querySelector(".tc");
  assert.equal(card.querySelector("code").textContent, "<img src=x onerror=alert(1)>");
  assert.equal(card.querySelectorAll("img").length, 0);
  assert.equal(card.querySelectorAll("b").length, 0);
});

test("a draft card says it has not run yet", () => {
  const model = AA.chatTurn.create(turn());
  AA.chatTurn.reduce(model, { type: "tool_draft", index: 0, call_id: "c1", name: "book", arguments: "" });
  const card = AA.chatTurn.createView(model, handlers, {}).root.querySelector(".tc");
  assert.equal(card.querySelector(".status").textContent, "draft: not run yet");
  assert.match(card.textContent, /arguments arriving/);
});
