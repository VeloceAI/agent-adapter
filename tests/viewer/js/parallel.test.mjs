// parallel.js: the Parallel tab's summary sentences and tables.
import assert from "node:assert/strict";
import { test } from "node:test";

import { loadDom } from "./fakedom.mjs";

const { AA, document } = loadDom("data.js", "parallel.js");

const REPORT = {
  calls: { tool: 5, llm: 3 },
  duplicates: [{ tool: "lookup", count: 2, call_ids: ["a", "b"], agents: ["mid", "slow"] }],
  duplicate_calls: 1,
  arguments_recorded: true,
  saved: { cache: 2, inflight: 1, inflight_waited_ms: 40 },
  tokens: { input: 900, output: 120, saved_by_cache: 300, saved_by_inflight: 150 },
  joins: [{ kind: "agents", where: "main", start: 0, wall_ms: 150, slowest: "slow", waited_ms: 240,
    branches: [{ name: "fast", duration_ms: 10, waited_ms: 140, status: "ok" }, { name: "slow", duration_ms: 150, waited_ms: 0, status: "ok" }] }],
  parallel_ms: 150,
  waited_ms: 240,
  critical_path: ["slow"],
  claims: { total: 3, contested: [{ key: "lookup:oslo", holder: "fast", lost: ["mid", "slow"] }] },
  agents: [{ agent: "fast", tool_calls: 2, llm_calls: 1, cost_usd: 0.01, input_tokens: 900, output_tokens: 120, busy_ms: 10, posts: 1, claims_won: 1, claims_lost: 0 }],
  board: [{ type: "board.claimed", ts: 1, by: "mid", key: "lookup:oslo", won: false, holder: "fast" }],
};

test("the facts name waits, repeats, savings and contested claims", () => {
  const facts = AA.parallelFacts(REPORT);
  assert.equal(facts.length, 6);
  assert.match(facts[0], /1 parallel section/);
  assert.match(facts[1], /1 repeated call/);
  assert.match(facts[2], /shared an identical call/);
  assert.match(facts[4], /450 model tokens saved/);
  assert.match(facts[5], /contested/);
  const bare = AA.parallelFacts({ ...REPORT, joins: [], duplicate_calls: 0, arguments_recorded: false, saved: {}, tokens: {}, claims: { contested: [] } });
  assert.deepEqual(bare.length, 1);
  assert.match(bare[0], /capture_content/);
});

test("the tab draws one table per join and the per-agent table", () => {
  const box = document.createElement("div");
  AA.renderParallel(box, REPORT);
  const text = box.textContent;
  assert.match(text, /slowest: slow/);
  assert.match(text, /Contested claims/);
  assert.match(text, /lost claim to fast/);
  assert.match(text, /900 \/ 120/);
  assert.equal(box.querySelectorAll("table").length, 5);
  AA.renderParallel(box, null);
  assert.match(box.textContent, /Pick a run/);
});
