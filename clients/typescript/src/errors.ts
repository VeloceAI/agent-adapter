/** Typed errors: one class per problem kind of the server (contract §15.4), plus transport errors. */
import type { Problem } from "./types.ts";

const TYPE_PREFIX = "urn:agent-adapter:error:";

/** A non-2xx response; `problem` is the RFC 9457 body (`type: "about:blank"` when the body was not one). */
export class ToolServerError extends Error {
  override name = "ToolServerError";
  readonly problem: Problem;
  readonly status: number;
  /** Server-requested wait before a retry (`Retry-After`), in milliseconds. */
  readonly retryAfterMs: number | undefined;

  constructor(problem: Problem, options: { status?: number; retryAfterMs?: number } = {}) {
    const status = options.status ?? problem.status;
    const detail = typeof problem.detail === "string" && problem.detail ? `: ${problem.detail}` : "";
    super(`${status} ${problem.title}${detail}`);
    this.problem = problem;
    this.status = status;
    this.retryAfterMs = options.retryAfterMs;
  }

  /** The agent-adapter error kind (`tool-not-found`, ...); `""` for other problem types. */
  get kind(): string {
    return this.problem.type.startsWith(TYPE_PREFIX) ? this.problem.type.slice(TYPE_PREFIX.length) : "";
  }

  /** The `tool` extension, when the server named one. */
  get tool(): string | undefined {
    return typeof this.problem["tool"] === "string" ? this.problem["tool"] : undefined;
  }
}

/** 400: malformed JSON, query or request. */
export class BadRequestError extends ToolServerError {
  override name = "BadRequestError";
}
/** 400: the tool was called the wrong way. */
export class ToolUsageError extends BadRequestError {
  override name = "ToolUsageError";
}
/** 401: missing or rejected credentials. */
export class UnauthorizedError extends ToolServerError {
  override name = "UnauthorizedError";
}
/** 403: the principal may not do this (tool, run or approval). */
export class ForbiddenError extends ToolServerError {
  override name = "ForbiddenError";
}
/** 403: a reviewer denied the call. */
export class ApprovalDeniedError extends ToolServerError {
  override name = "ApprovalDeniedError";
}
/** 403: a guardrail hook refused the arguments or the result. */
export class GuardrailBlockedError extends ToolServerError {
  override name = "GuardrailBlockedError";
}
/** 404: unknown route or resource. */
export class NotFoundError extends ToolServerError {
  override name = "NotFoundError";
}
/** 404: no tool with that name (or not visible to the principal). */
export class ToolNotFoundError extends NotFoundError {
  override name = "ToolNotFoundError";
}
/** 404: no approval with that id. */
export class ApprovalNotFoundError extends NotFoundError {
  override name = "ApprovalNotFoundError";
}
/** 409: `once_per_run` or `max_calls` reached for this run. */
export class RunLimitExceededError extends ToolServerError {
  override name = "RunLimitExceededError";
}
/** 409: the approval was already decided. */
export class ApprovalAlreadyDecidedError extends ToolServerError {
  override name = "ApprovalAlreadyDecidedError";
}
/** 422: arguments failed the tool's schema; `errors` lists the problems (no input values). */
export class ToolArgsInvalidError extends ToolServerError {
  override name = "ToolArgsInvalidError";
  get errors(): unknown[] {
    const errors = this.problem["errors"];
    return Array.isArray(errors) ? errors : [];
  }
}
/** 429: the run's budget is spent. */
export class BudgetExceededError extends ToolServerError {
  override name = "BudgetExceededError";
}
/** 502: the tool raised an unexpected exception. */
export class ToolFailedError extends ToolServerError {
  override name = "ToolFailedError";
}
/** 502: the tool raised `ToolError`; the detail is the tool's own message. */
export class ToolError extends ToolServerError {
  override name = "ToolError";
}
/** 502: a model call inside the tool failed; `kind` is the exact `model-*` kind. */
export class ModelError extends ToolServerError {
  override name = "ModelError";
}
/** 504: the tool ran past its timeout on the server. */
export class ToolTimeoutError extends ToolServerError {
  override name = "ToolTimeoutError";
}
/** 500: server misconfiguration or an internal error. */
export class InternalServerError extends ToolServerError {
  override name = "InternalServerError";
}

/** No HTTP response: the connection failed (`reason: "network"`) or the attempt timed out. */
export class ToolTransportError extends Error {
  override name = "ToolTransportError";
  readonly reason: "network" | "timeout";
  /** The platform's error code when it gave one (`ECONNREFUSED`, ...), else `""`. */
  readonly code: string;

  constructor(message: string, reason: "network" | "timeout", options: { cause?: unknown; code?: string } = {}) {
    super(message, { cause: options.cause });
    this.reason = reason;
    this.code = options.code ?? "";
  }
}

/** The server answered, but not with what the protocol promises. */
export class ToolProtocolError extends Error {
  override name = "ToolProtocolError";
  readonly status: number;

  constructor(message: string, status: number, options?: { cause?: unknown }) {
    super(message, options);
    this.status = status;
  }
}

/** A malformed or oversized server-sent event stream. */
export class EventStreamError extends ToolProtocolError {
  override name = "EventStreamError";
}

type ServerErrorClass = new (problem: Problem, options?: { status?: number; retryAfterMs?: number }) => ToolServerError;

const BY_KIND: Record<string, ServerErrorClass> = {
  "bad-request": BadRequestError,
  "tool-usage-error": ToolUsageError,
  unauthorized: UnauthorizedError,
  forbidden: ForbiddenError,
  "approval-denied": ApprovalDeniedError,
  "guardrail-blocked": GuardrailBlockedError,
  "not-found": NotFoundError,
  "tool-not-found": ToolNotFoundError,
  "approval-not-found": ApprovalNotFoundError,
  "run-limit-exceeded": RunLimitExceededError,
  "approval-already-decided": ApprovalAlreadyDecidedError,
  "tool-args-invalid": ToolArgsInvalidError,
  "budget-exceeded": BudgetExceededError,
  "tool-failed": ToolFailedError,
  "tool-error": ToolError,
  "tool-timeout": ToolTimeoutError,
  "tool-definition-invalid": InternalServerError,
  "schema-not-portable": InternalServerError,
  "config-invalid": InternalServerError,
  "internal-error": InternalServerError,
};

const BY_STATUS: Record<number, ServerErrorClass> = {
  400: BadRequestError,
  401: UnauthorizedError,
  403: ForbiddenError,
  404: NotFoundError,
  500: InternalServerError,
};

/** The error class for a problem: by agent-adapter kind first, then by HTTP status. */
export function errorFromProblem(
  problem: Problem,
  options: { status?: number; retryAfterMs?: number } = {},
): ToolServerError {
  const status = options.status ?? problem.status;
  const kind = problem.type.startsWith(TYPE_PREFIX) ? problem.type.slice(TYPE_PREFIX.length) : "";
  const cls = Object.hasOwn(BY_KIND, kind)
    ? BY_KIND[kind]
    : kind.startsWith("model-")
      ? ModelError
      : Object.hasOwn(BY_STATUS, status)
        ? BY_STATUS[status]
        : ToolServerError;
  return new (cls ?? ToolServerError)(problem, options);
}
