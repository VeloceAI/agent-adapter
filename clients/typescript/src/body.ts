/** Response bodies: size-bounded reading and RFC 9457 problem parsing. */
import { errorFromProblem, ToolProtocolError } from "./errors.ts";
import type { ToolServerError } from "./errors.ts";
import type { Problem } from "./types.ts";

const PROBLEM_MEDIA_TYPE = "application/problem+json";

/** The typed error for a non-2xx response, from its problem body or `about:blank`. */
export async function responseError(
  response: Response,
  retryAfterMs: number | undefined,
  maxBytes: number,
  what: string,
): Promise<ToolServerError> {
  const text = await readText(response, maxBytes, what);
  const media = (response.headers.get("content-type") ?? "").split(";")[0]?.trim().toLowerCase();
  const options = retryAfterMs === undefined ? { status: response.status } : { status: response.status, retryAfterMs };
  const parsed = media === PROBLEM_MEDIA_TYPE ? asProblem(text, response.status) : undefined;
  const problem = parsed ?? {
    type: "about:blank",
    title: response.statusText || `HTTP ${response.status}`,
    status: response.status,
  };
  return errorFromProblem(problem, options);
}

/** The body as a problem document, or `undefined` when it is not one. */
function asProblem(text: string, status: number): Problem | undefined {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return undefined; // not a problem document after all; the caller falls back to about:blank
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) return undefined;
  const record = value as Record<string, unknown>;
  if (typeof record["type"] !== "string" || typeof record["title"] !== "string") return undefined;
  return {
    ...record,
    type: record["type"],
    title: record["title"],
    status: typeof record["status"] === "number" ? record["status"] : status,
  };
}

/** The body as UTF-8 text, refusing more than `maxBytes` (declared or actually sent). */
export async function readText(response: Response, maxBytes: number, what: string): Promise<string> {
  const tooLarge = (): ToolProtocolError =>
    new ToolProtocolError(`${what}: response body exceeds ${maxBytes} bytes`, response.status);
  if (Number(response.headers.get("content-length")) > maxBytes) {
    await release(response);
    throw tooLarge();
  }
  if (response.body === null) return "";
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parts: string[] = [];
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maxBytes) {
        try {
          await reader.cancel();
        } catch {
          // the body already failed; the size error is what the caller needs
        }
        throw tooLarge();
      }
      parts.push(decoder.decode(value, { stream: true }));
    }
  } finally {
    reader.releaseLock();
  }
  parts.push(decoder.decode());
  return parts.join("");
}

/** Discards a body that will not be read (frees the connection). */
export async function release(response: Response): Promise<void> {
  try {
    await response.body?.cancel();
  } catch {
    // the body already failed; the response is being discarded for a retry either way
  }
}
