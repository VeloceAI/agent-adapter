// Loads the viewer's static scripts into this Node process (no browser): `load("graphlive.js")`.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

export const STATIC = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "agent_adapter", "viewer", "static");

// Just enough of a page for data.js (no embedded data, no token, no meta tags).
function stubPage() {
  if (globalThis.window) return;
  globalThis.window = globalThis;
  globalThis.document = { getElementById: () => null, querySelector: () => null };
  globalThis.location = { hash: "", pathname: "/" };
  globalThis.history = { replaceState() {} };
}

export function load(...names) {
  stubPage();
  for (const name of names) vm.runInThisContext(readFileSync(join(STATIC, name), "utf8"), { filename: name });
  return globalThis.AA;
}

// Plain JSON copies, so values built by the scripts compare with deepStrictEqual.
export function plain(value) {
  return JSON.parse(JSON.stringify(value, (key, item) => (item instanceof Set ? Array.from(item).sort() : item)));
}
