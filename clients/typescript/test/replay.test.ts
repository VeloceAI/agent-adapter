import assert from "node:assert/strict";
import { describe, test } from "node:test";

import { MAX_REMEMBERED_ID_LENGTH, YieldedEvents } from "../src/replay.ts";
import type { SseMessage } from "../src/sse.ts";

const message = (id: string, data = "{}"): SseMessage => ({ id, ownId: true, event: "tool.started", data });

describe("YieldedEvents", () => {
  test("remembers a normal id with its content", () => {
    const yielded = new YieldedEvents();
    yielded.add(message("5"));
    assert.equal(yielded.has(message("5")), true);
    assert.equal(yielded.has(message("5", '{"n":2}')), false);
    assert.equal(yielded.has(message("6")), false);
  });

  test("ids longer than the limit are never remembered", () => {
    const yielded = new YieldedEvents();
    const long = "x".repeat(MAX_REMEMBERED_ID_LENGTH + 1);
    yielded.add(message(long));
    assert.equal(yielded.size, 0);
    assert.equal(yielded.has(message(long)), false);
  });

  test("an id at the limit is remembered", () => {
    const yielded = new YieldedEvents();
    const edge = "x".repeat(MAX_REMEMBERED_ID_LENGTH);
    yielded.add(message(edge));
    assert.equal(yielded.has(message(edge)), true);
  });

  test("ids of the same hash input but different length do not collide", () => {
    const yielded = new YieldedEvents();
    yielded.add(message("a"));
    assert.equal(yielded.has(message("a\u0000")), false);
  });

  test("memory stays bounded: at most 1024 keys, each of fixed size", () => {
    const yielded = new YieldedEvents();
    for (let i = 0; i < 3000; i++) yielded.add(message(`${i}`.padStart(MAX_REMEMBERED_ID_LENGTH, "z")));
    assert.equal(yielded.size, 1024);
    for (const key of yielded.keys()) assert.ok(key.length <= 20, `key ${key} has a fixed small size`);
    assert.equal(yielded.has(message("2999".padStart(MAX_REMEMBERED_ID_LENGTH, "z"))), true);
    assert.equal(yielded.has(message("0".padStart(MAX_REMEMBERED_ID_LENGTH, "z"))), false);
  });

  test("an empty id is never remembered", () => {
    const yielded = new YieldedEvents();
    yielded.add(message(""));
    assert.equal(yielded.size, 0);
    assert.equal(yielded.has(message("")), false);
  });
});
