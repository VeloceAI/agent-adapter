// The docs' chat demo file runs whole: every script on the page's markup, its real recorded
// data, in a sandboxed iframe where storage throws. A question replays into the inspector.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { installDom, runScripts } from "./fakedom.mjs";
import { STATIC } from "./load.mjs";

const ROOT = join(STATIC, "..", "..", "..");
const demo = readFileSync(join(ROOT, "docs", "assets", "demo", "chat-demo.html"), "utf8");
const data = /<script type="application\/json" id="aa-data">([\s\S]*?)<\/script>/.exec(demo)[1];
const page = readFileSync(join(STATIC, "index.html"), "utf8");
const body = page.slice(page.indexOf("<body>") + "<body>".length, page.indexOf("<!-- AA_DATA -->"));
const scripts = Array.from(page.matchAll(/<script src="static\/([\w.]+)"><\/script>/g), (m) => m[1]);
scripts.splice(scripts.indexOf("chatapi.js") + 1, 0, "chatdemo.js"); // where the demo export puts it

const denied = () => { throw Object.assign(new Error("The document is sandboxed"), { name: "SecurityError" }); };
const document = installDom({ html: '<meta name="aa-chat" content="demo">' + body });
const holder = document.createElement("script");
holder.setAttribute("id", "aa-data");
holder.textContent = data;
document.body.appendChild(holder);
for (const name of ["localStorage", "sessionStorage"]) Object.defineProperty(globalThis, name, { get: denied, configurable: true });
globalThis.requestAnimationFrame = (fn) => setImmediate(fn);
const intervals = [];
globalThis.setInterval = (fn) => { intervals.push(fn); return intervals.length; }; // run by hand: the process must end
globalThis.clearInterval = (id) => { intervals[id - 1] = null; };
globalThis.matchMedia = () => ({ matches: true }); // reduced motion: the turn shows at once
const errors = [];
console.error = (...args) => errors.push(args.map(String).join(" "));
const AA = runScripts(...scripts);
const settle = async () => { for (let i = 0; i < 10; i += 1) await new Promise((r) => setTimeout(r, 1)); };
await AA.chatView.ready;
await settle();
const $ = (id) => document.getElementById(id);

test("the demo file opens on the chat with one button per suggested question", () => {
  assert.equal($("chat-view").hidden, false);
  assert.equal($("chat-shell").hidden, false);
  const labels = $("chat-main").querySelectorAll(".demo-q").map((b) => b.getAttribute("aria-label"));
  assert.equal(labels.length, 5);
  assert.deepEqual(labels.map((l) => l.split(":")[0]), ["router", "fan-out, fan-in", "approval", "loop", "all together"]);
  assert.deepEqual(errors, []);
});

test("the combined question plays into the conversation and the inspector's graph", async () => {
  $("chat-main").querySelectorAll(".demo-q")[4].click();
  await settle();
  for (let i = 0; i < 4; i += 1) {
    intervals.filter(Boolean).forEach((fn) => fn());
    await settle();
  }
  assert.match($("chat-log").textContent, /Rain all day/);
  assert.match($("chat-log").textContent, /family_check/);
  const nodes = $("chat-inspector").querySelectorAll(".g-node");
  assert.ok(nodes.length >= 6, "the day plan's graph");
  assert.match($("chat-inspector").textContent, /ok/);
  assert.equal($("chat-session-list").querySelectorAll("button").length, 3);
  assert.deepEqual(errors, []);
});
