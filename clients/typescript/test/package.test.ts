import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describe, test } from "node:test";

import * as api from "../src/index.ts";

describe("package surface", () => {
  test("exports the client, adapters and error classes", () => {
    for (const name of ["ToolClient", "toolDefinitions", "asVercelTools", "ToolServerError", "ToolArgsInvalidError"]) {
      assert.ok(name in api, name);
    }
  });

  test("keeps parser and mapping helpers internal", () => {
    for (const name of ["SseParser", "readEventStream", "DEFAULT_MAX_EVENT_SIZE", "errorFromProblem"]) {
      assert.ok(!(name in api), name);
    }
  });

  test("is private until a publishing decision is made", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as Record<
      string,
      unknown
    >;
    assert.equal(pkg["private"], true);
    assert.equal(pkg["license"], "Apache-2.0");
  });
});
