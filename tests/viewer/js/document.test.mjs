// document.js: the Document card for template documents (compose.* events), live and safe.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "markdown.js", "document.js");

const started = (blank, kind) => ({ type: "compose.blank_started", data: { document: "report", blank: blank, kind: kind } });
const filled = (blank, kind, extra) => ({
  type: "compose.blank_filled",
  data: Object.assign({ document: "report", blank: blank, kind: kind, ok: true, attempts: 1, cached: false, cost_usd: 0.0021, problems: 0 }, extra),
});
const run = (events, key, captured) => ({ summary: { key: key || "run_a" }, meta: [{ capture_content: captured !== false }], events: events });

function card() {
  const box = document.createElement("section");
  box.hidden = true;
  return { box: box, panel: AA.createDocument(box) };
}

const rows = (box) => box.querySelectorAll(".doc-blank").map((li) => ({
  name: li.querySelector("strong").textContent,
  status: li.querySelector(".status").textContent,
}));

test("hidden for a run without compose events", () => {
  const { box, panel } = card();
  panel.update(run([{ type: "run.started", data: {} }]));
  assert.equal(box.hidden, true);
  panel.update(null);
  assert.equal(box.hidden, true);
});

test("blanks are listed in first-seen order and their status follows the run live", () => {
  const { box, panel } = card();
  const events = [started("summary", "text"), started("sales", "table")];
  panel.update(run(events));
  assert.equal(box.hidden, false);
  assert.deepEqual(rows(box), [{ name: "summary", status: "filling…" }, { name: "sales", status: "filling…" }]);
  assert.equal(box.querySelectorAll("p.muted").length, 0, "content is captured: no 'not recorded' note while filling");
  events.push(filled("sales", "table", { ok: false, attempts: 3, problems: 2 }));
  events.push(filled("summary", "text"));
  events.push(filled("funnel", "diagram", { cached: true, cost_usd: 0 }));
  panel.update(run(events.slice()));
  assert.deepEqual(rows(box), [
    { name: "summary", status: "ok" },
    { name: "sales", status: "problems 2" },
    { name: "funnel", status: "cached" },
  ]);
  const sales = box.querySelectorAll(".doc-blank")[1];
  assert.ok(sales.textContent.includes("3 attempts"));
  assert.ok(!box.querySelectorAll(".doc-blank")[0].textContent.includes("attempts"), "one attempt is not shown");
});

test("each blank shows its cost and the document its total", () => {
  const { box, panel } = card();
  panel.update(run([filled("summary", "text"), filled("sales", "table", { cost_usd: 0.5 })]));
  const costs = box.querySelectorAll(".doc-cost").map((n) => n.textContent);
  assert.deepEqual(costs, [AA.fmt.usd(0.0021), AA.fmt.usd(0.5)]);
  const head = box.querySelector(".doc-head").textContent;
  assert.ok(head.includes("report") && head.includes("2 of 2 blanks filled") && head.includes(AA.fmt.usd(0.5021)), head);
});

test("captured Markdown is drawn: text, a table, and a mermaid diagram as its source", () => {
  const { box, panel } = card();
  panel.update(run([
    filled("summary", "text", { markdown: "Sales **grew** in Q3.", problems: [] }),
    filled("sales", "table", { markdown: "| Region | Sales |\n| --- | --- |\n| EU | 10 |", problems: [] }),
    filled("funnel", "diagram", { markdown: "```mermaid\nflowchart TD\n  a --> b\n```", problems: [] }),
  ]));
  const [summary, sales, funnel] = box.querySelectorAll(".doc-blank");
  assert.equal(summary.querySelector(".doc-md strong").textContent, "grew");
  assert.deepEqual(sales.querySelectorAll("td").map((n) => n.textContent), ["EU", "10"]);
  assert.equal(funnel.querySelector(".md-code-head").textContent, "mermaid");
  assert.equal(funnel.querySelector("code").textContent, "flowchart TD\n  a --> b");
  assert.equal(box.querySelectorAll("p.muted").length, 0, "no 'not recorded' note");
});

test("without captured content the card says how to record the text", () => {
  const { box, panel } = card();
  panel.update(run([filled("summary", "text", { ok: false, problems: 1 })], "run_a", false));
  assert.equal(box.querySelectorAll(".doc-md").length, 0);
  assert.equal(rows(box)[0].status, "problems 1");
  assert.ok(box.querySelector("p.muted").textContent.includes("capture_content=True"));
});

test("captured problems are listed under their blank", () => {
  const { box, panel } = card();
  panel.update(run([filled("sales", "table", { ok: false, problems: ["missing column Sales", "too short"], markdown: "x" })]));
  assert.equal(rows(box)[0].status, "problems 2");
  assert.deepEqual(box.querySelectorAll(".doc-problems li").map((n) => n.textContent), ["missing column Sales", "too short"]);
});

test("an accepted revision replaces the text, marks the blank edited and can show the old text", () => {
  const { box, panel } = card();
  const edited = {
    type: "compose.edited",
    data: {
      document: "report", revised: 1, rejected: 1,
      revisions: [
        { blank: "summary", before: "Sales grew in Q3.", after: "Sales rose in Q3.", reason: "tone", accepted: true, problems: [] },
        { blank: "sales", before: "a", after: "b", reason: "no", accepted: false, problems: ["bad table"] },
      ],
    },
  };
  const events = [filled("summary", "text", { markdown: "Sales grew in Q3.", problems: [] }), filled("sales", "table", { markdown: "a", problems: [] }), edited];
  panel.update(run(events));
  assert.deepEqual(rows(box), [{ name: "summary", status: "edited" }, { name: "sales", status: "ok" }]);
  const summary = () => box.querySelectorAll(".doc-blank")[0];
  assert.equal(summary().querySelector(".doc-md").textContent, "Sales rose in Q3.");
  assert.ok(summary().textContent.includes("why: tone"));
  assert.equal(box.querySelectorAll(".doc-blank")[1].querySelector(".doc-md").textContent, "a", "a rejected revision changes nothing");
  assert.ok(box.querySelector(".doc-head").textContent.includes("editor: 1 changed, 1 rejected"));
  summary().querySelector("button").click();
  assert.equal(summary().querySelector(".doc-md").textContent, "Sales grew in Q3.");
  summary().querySelector("button").click();
  assert.equal(summary().querySelector(".doc-md").textContent, "Sales rose in Q3.");
});

test("a cached blank reported after the editor keeps the edited mark and shows the latest text", () => {
  const { box, panel } = card();
  panel.update(run([
    filled("summary", "text", { markdown: "old", problems: [] }),
    { type: "compose.edited", data: { document: "report", revised: 1, rejected: 0, revisions: [{ blank: "summary", before: "old", after: "new", reason: "", accepted: true }] } },
    filled("summary", "text", { cached: true, cost_usd: 0, markdown: "new", problems: [] }),
  ]));
  assert.deepEqual(rows(box), [{ name: "summary", status: "edited" }]);
  assert.equal(box.querySelector(".doc-md").textContent, "new");
});

test("model Markdown never becomes markup", () => {
  const { box, panel } = card();
  const hostile = '<img src=x onerror=alert(1)><script>alert(1)</script> [x](javascript:alert(1))';
  panel.update(run([filled("summary", "text", { markdown: hostile, problems: ['<b onclick="x">'] })]));
  const tags = box.querySelectorAll("*").map((n) => n.tagName.toLowerCase());
  assert.ok(!tags.includes("img") && !tags.includes("script") && !tags.includes("b"), tags.join(","));
  assert.equal(box.querySelectorAll("a").length, 0, "an unsafe link keeps only its label");
  assert.ok(box.querySelector(".doc-md").textContent.includes("<script>alert(1)</script>"));
});

test("several documents in one run each get a heading; another run starts fresh", () => {
  const { box, panel } = card();
  const other = { type: "compose.blank_started", data: { document: "memo", blank: "intro", kind: "text" } };
  panel.update(run([started("summary", "text"), other]));
  assert.deepEqual(box.querySelectorAll(".doc-head strong").map((n) => n.textContent), ["report", "memo"]);
  panel.update(run([started("x", "list")], "run_b"));
  assert.deepEqual(rows(box), [{ name: "x", status: "filling…" }]);
});
