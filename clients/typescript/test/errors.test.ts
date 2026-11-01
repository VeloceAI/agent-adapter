import assert from "node:assert/strict";
import { describe, test } from "node:test";

import {
  ApprovalAlreadyDecidedError,
  ApprovalDeniedError,
  ApprovalNotFoundError,
  BadRequestError,
  BudgetExceededError,
  ForbiddenError,
  GuardrailBlockedError,
  InternalServerError,
  ModelError,
  NotFoundError,
  RunLimitExceededError,
  ToolArgsInvalidError,
  ToolError,
  ToolFailedError,
  ToolNotFoundError,
  ToolServerError,
  ToolTimeoutError,
  ToolUsageError,
  UnauthorizedError,
} from "../src/index.ts";
import { errorFromProblem } from "../src/errors.ts";
import type { Problem } from "../src/index.ts";

function urn(kind: string, status: number, extra: Record<string, unknown> = {}): Problem {
  return { type: `urn:agent-adapter:error:${kind}`, title: `title ${kind}`, status, detail: "d", ...extra };
}

// Every row of contract §15.4.
const KINDS: Array<[string, number, new (...args: never[]) => ToolServerError]> = [
  ["tool-not-found", 404, ToolNotFoundError],
  ["approval-not-found", 404, ApprovalNotFoundError],
  ["not-found", 404, NotFoundError],
  ["tool-args-invalid", 422, ToolArgsInvalidError],
  ["bad-request", 400, BadRequestError],
  ["tool-usage-error", 400, ToolUsageError],
  ["unauthorized", 401, UnauthorizedError],
  ["forbidden", 403, ForbiddenError],
  ["approval-denied", 403, ApprovalDeniedError],
  ["guardrail-blocked", 403, GuardrailBlockedError],
  ["run-limit-exceeded", 409, RunLimitExceededError],
  ["approval-already-decided", 409, ApprovalAlreadyDecidedError],
  ["budget-exceeded", 429, BudgetExceededError],
  ["tool-timeout", 504, ToolTimeoutError],
  ["tool-failed", 502, ToolFailedError],
  ["tool-error", 502, ToolError],
  ["model-rate-limited", 502, ModelError],
  ["model-auth-failed", 502, ModelError],
  ["tool-definition-invalid", 500, InternalServerError],
  ["schema-not-portable", 500, InternalServerError],
  ["config-invalid", 500, InternalServerError],
  ["internal-error", 500, InternalServerError],
];

describe("errorFromProblem", () => {
  for (const [kind, status, cls] of KINDS) {
    test(`maps ${kind} to ${cls.name}`, () => {
      const err = errorFromProblem(urn(kind, status));
      assert.ok(err instanceof cls);
      assert.ok(err instanceof ToolServerError);
      assert.ok(err instanceof Error);
      assert.equal(err.kind, kind);
      assert.equal(err.status, status);
      assert.equal(err.name, cls.name);
    });
  }

  test("not-found family shares NotFoundError", () => {
    assert.ok(errorFromProblem(urn("tool-not-found", 404)) instanceof NotFoundError);
    assert.ok(errorFromProblem(urn("approval-not-found", 404)) instanceof NotFoundError);
    assert.ok(errorFromProblem(urn("tool-usage-error", 400)) instanceof BadRequestError);
  });

  test("unknown kinds fall back to the status family", () => {
    assert.ok(errorFromProblem(urn("brand-new", 403)) instanceof ForbiddenError);
    assert.ok(errorFromProblem(urn("brand-new", 404)) instanceof NotFoundError);
    const other = errorFromProblem(urn("brand-new", 418));
    assert.equal(other.constructor, ToolServerError);
    assert.equal(other.kind, "brand-new");
  });

  test("about:blank problems use the status and have no kind", () => {
    const err = errorFromProblem({ type: "about:blank", title: "Unauthorized", status: 401 });
    assert.ok(err instanceof UnauthorizedError);
    assert.equal(err.kind, "");
    const gateway = errorFromProblem({ type: "about:blank", title: "Bad Gateway", status: 502 });
    assert.equal(gateway.constructor, ToolServerError);
  });

  test("a kind named like an Object prototype key is not a lookup hit", () => {
    const err = errorFromProblem(urn("constructor", 418));
    assert.equal(err.constructor, ToolServerError);
  });

  test("message carries status, title and detail; extensions are readable", () => {
    const err = errorFromProblem(urn("tool-args-invalid", 422, { tool: "add", errors: [{ loc: ["a"] }] }));
    assert.equal(err.message, "422 title tool-args-invalid: d");
    assert.equal(err.tool, "add");
    assert.deepEqual((err as ToolArgsInvalidError).errors, [{ loc: ["a"] }]);
    assert.equal(err.problem["tool"], "add");
  });

  test("missing extensions read as empty", () => {
    const err = new ToolArgsInvalidError({ type: "about:blank", title: "x", status: 422 });
    assert.deepEqual(err.errors, []);
    assert.equal(err.tool, undefined);
    assert.equal(err.message, "422 x");
  });

  test("the response status and Retry-After override the body", () => {
    const err = errorFromProblem(urn("budget-exceeded", 400), { status: 429, retryAfterMs: 60_000 });
    assert.ok(err instanceof BudgetExceededError);
    assert.equal(err.status, 429);
    assert.equal(err.retryAfterMs, 60_000);
  });
});
