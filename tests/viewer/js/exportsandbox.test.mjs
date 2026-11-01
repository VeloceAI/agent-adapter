// The docs demo runs in a sandboxed iframe (opaque origin): every storage access throws there.
// The export page must still draw its runs, graph and tabs, and its theme button must work.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";

import { event, installDom, runScripts } from "./fakedom.mjs";
import { STATIC } from "./load.mjs";

const ROOT = join(STATIC, "..", "..", "..");
const demo = readFileSync(join(ROOT, "docs", "assets", "demo", "sample-run.html"), "utf8");
const data = /<script type="application\/json" id="aa-data">([\s\S]*?)<\/script>/.exec(demo)[1];
const page = readFileSync(join(STATIC, "index.html"), "utf8");
const body = page.slice(page.indexOf("<body>") + "<body>".length, page.indexOf("<!-- AA_DATA -->"));
const scripts = Array.from(page.matchAll(/<script src="static\/([\w.]+)"><\/script>/g), (m) => m[1]).filter((name) => !name.startsWith("chat") && !["markdown.js", "inspector.js"].includes(name));

const denied = () => { throw Object.assign(new Error("The document is sandboxed and lacks the 'allow-same-origin' flag."), { name: "SecurityError" }); };
const document = installDom({ hash: "#run=run_" + "0".repeat(31) + "1", html: '<meta name="aa-chat" content="">' + body }); // the iframe opens the flow run
const script = document.createElement("script");
script.setAttribute("id", "aa-data");
script.textContent = data;
document.body.appendChild(script);
for (const name of ["localStorage", "sessionStorage"]) Object.defineProperty(globalThis, name, { get: denied, configurable: true });
globalThis.requestAnimationFrame = (fn) => setImmediate(fn);
const errors = [];
const logged = console.error;
console.error = (...args) => errors.push(args.join(" "));
const AA = runScripts(...scripts);
await new Promise((resolve) => setTimeout(resolve, 50));

test("the export draws its runs with every storage access refused", () => {
  assert.ok(AA.embedded, "the page reads its embedded data");
  assert.equal(document.getElementById("run-list").querySelectorAll("button").length, 3);
  assert.ok(document.getElementById("graph").querySelectorAll(".g-node").length >= 5, "the flow run's graph");
  assert.deepEqual(errors, []);
});

test("the theme button still cycles when the choice cannot be stored", () => {
  const button = document.getElementById("theme");
  const before = button.textContent;
  button.dispatchEvent(event("click"));
  assert.notEqual(button.textContent, before);
  console.error = logged;
});
