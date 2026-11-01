/** Server-sent events parser (WHATWG HTML "event stream interpretation"), with a bounded buffer. */
import { EventStreamError, ToolTransportError } from "./errors.ts";

export const DEFAULT_MAX_EVENT_SIZE = 1_048_576;

/** One dispatched event; `id` is the last event id seen so far (`""` when none). */
export interface SseMessage {
  id: string;
  event: string;
  data: string;
  /** The event had its own `id` field (else `id` is inherited from an earlier one). */
  ownId: boolean;
}

const LF = 10;
const CR = 13;
const COLON = 58;
const SPACE = 32;

/** Incremental parser: feed decoded text in any split, get complete events back. */
export class SseParser {
  /** Longest line and longest event data, in characters. */
  readonly maxSize: number;
  #started = false;
  #skipLF = false;
  #pending: string[] = [];
  #pendingSize = 0;
  #data = "";
  #event = "";
  #idBuffer: string;
  #sawId = false;
  #lastId: string;
  #retryMs: number | undefined;

  /** `lastEventId` carries the id of an earlier connection, as an EventSource does across reconnects. */
  constructor(options: { maxSize?: number; lastEventId?: string } = {}) {
    const maxSize = options.maxSize ?? DEFAULT_MAX_EVENT_SIZE;
    if (!Number.isInteger(maxSize) || maxSize < 1) throw new RangeError("maxSize must be a positive integer");
    this.maxSize = maxSize;
    this.#lastId = this.#idBuffer = options.lastEventId ?? "";
  }

  /** Reconnection delay sent by the server (`retry:`), in milliseconds. */
  get retryMs(): number | undefined {
    return this.#retryMs;
  }

  /** The id to send as `Last-Event-ID`: set at every dispatch (a blank line), even one without data. */
  get lastEventId(): string {
    return this.#lastId;
  }

  /** Parses the next piece of the stream; returns the events it completes. */
  feed(chunk: string): SseMessage[] {
    if (!this.#started) {
      if (chunk === "") return [];
      this.#started = true;
      if (chunk.charCodeAt(0) === 0xfeff) chunk = chunk.slice(1);
    }
    const out: SseMessage[] = [];
    let start = 0;
    if (this.#skipLF && chunk.length > 0) {
      // a CR ended the previous chunk; a LF right after it belongs to the same line end
      if (chunk.charCodeAt(0) === LF) start = 1;
      this.#skipLF = false;
    }
    for (let i = start; i < chunk.length; i++) {
      const code = chunk.charCodeAt(i);
      if (code !== LF && code !== CR) continue;
      let line = chunk.slice(start, i);
      if (this.#pending.length > 0) {
        this.#pending.push(line);
        line = this.#pending.join("");
        this.#pending = [];
        this.#pendingSize = 0;
      }
      if (code === CR) {
        if (i + 1 === chunk.length) this.#skipLF = true;
        else if (chunk.charCodeAt(i + 1) === LF) i++;
      }
      start = i + 1;
      this.#checkSize(line.length, "line");
      this.#line(line, out);
    }
    if (start < chunk.length) {
      const rest = chunk.slice(start);
      this.#pending.push(rest);
      this.#pendingSize += rest.length;
      this.#checkSize(this.#pendingSize, "line");
    }
    return out;
  }

  #checkSize(size: number, what: string): void {
    if (size > this.maxSize) {
      throw new EventStreamError(`event stream ${what} exceeds ${this.maxSize} characters`, 200);
    }
  }

  #line(line: string, out: SseMessage[]): void {
    if (line === "") {
      this.#dispatch(out);
      return;
    }
    if (line.charCodeAt(0) === COLON) return;
    const colon = line.indexOf(":");
    let field = line;
    let value = "";
    if (colon !== -1) {
      field = line.slice(0, colon);
      value = line.slice(colon + 1);
      if (value.charCodeAt(0) === SPACE) value = value.slice(1);
    }
    switch (field) {
      case "event":
        this.#event = value;
        break;
      case "data":
        this.#data += `${value}\n`;
        this.#checkSize(this.#data.length - 1, "event");
        break;
      case "id":
        if (!value.includes("\u0000")) {
          this.#idBuffer = value;
          this.#sawId = true;
        }
        break;
      case "retry":
        if (/^[0-9]+$/.test(value)) this.#retryMs = Number(value);
        break;
      default:
        break;
    }
  }

  #dispatch(out: SseMessage[]): void {
    this.#lastId = this.#idBuffer;
    const ownId = this.#sawId;
    this.#sawId = false;
    if (this.#data === "") {
      this.#event = "";
      return;
    }
    out.push({ id: this.#lastId, event: this.#event || "message", data: this.#data.slice(0, -1), ownId });
    this.#data = "";
    this.#event = "";
  }
}

export interface ReadEventStreamOptions {
  /** Aborting cancels a pending read and rejects with the signal's reason. */
  signal?: AbortSignal | undefined;
  /** No bytes (keep-alives included) for this long cancels the body and raises a timeout. */
  idleTimeoutMs?: number | undefined;
}

/** Reads a UTF-8 event stream body through `parser`; cancels the body when the consumer stops early. */
export async function* readEventStream(
  body: ReadableStream<Uint8Array>,
  parser: SseParser,
  options: ReadEventStreamOptions = {},
): AsyncGenerator<SseMessage, void, undefined> {
  const { signal, idleTimeoutMs } = options;
  signal?.throwIfAborted();
  const reader = body.getReader();
  // ignoreBOM: the parser strips exactly one BOM, as the SSE spec says
  const decoder = new TextDecoder("utf-8", { ignoreBOM: true });
  let finished = false;
  let idle = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const cancel = (reason?: unknown): void => {
    reader.cancel(reason).catch(() => {
      // cancelling a body that already failed rejects; the abort or idle error is what propagates
    });
  };
  const onAbort = (): void => cancel(signal?.reason);
  signal?.addEventListener("abort", onAbort, { once: true });
  try {
    for (;;) {
      // the idle clock runs only while waiting for the server, never while the consumer works
      if (idleTimeoutMs !== undefined) {
        timer = setTimeout(() => {
          idle = true;
          cancel();
        }, idleTimeoutMs);
      }
      const { done, value } = await reader.read();
      clearTimeout(timer);
      signal?.throwIfAborted();
      if (idle) throw new ToolTransportError(`event stream idle for ${idleTimeoutMs} ms`, "timeout");
      if (done) {
        finished = true;
        yield* parser.feed(decoder.decode());
        return;
      }
      yield* parser.feed(decoder.decode(value, { stream: true }));
    }
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
    if (!finished && !idle && !signal?.aborted) {
      try {
        await reader.cancel();
      } catch {
        // the body already failed; that failure is what propagates, cancelling only frees it
      }
    }
    reader.releaseLock();
  }
}
