/** The record of yielded events that lets `events()` skip what a server replays after a reconnect. */
import type { SseMessage } from "./sse.ts";

const RING_SIZE = 1024;

/** Longer ids are never remembered (so never skipped): ids may be up to `maxEventSize` long. */
export const MAX_REMEMBERED_ID_LENGTH = 256;

/** The ids yielded so far (bounded, oldest dropped first) with a hash of each event's type and data. */
export class YieldedEvents {
  /** Fixed-size id key (length + 64-bit hash) → 32-bit hash of the event's type and data. */
  readonly #byId = new Map<string, number>();

  /** True when this exact event (same id, type and data) was yielded before. */
  has(message: SseMessage): boolean {
    const key = idKey(message.id);
    return key !== undefined && this.#byId.get(key) === contentHash(message);
  }

  add(message: SseMessage): void {
    const key = idKey(message.id);
    if (key === undefined) return;
    this.#byId.delete(key);
    this.#byId.set(key, contentHash(message));
    if (this.#byId.size > RING_SIZE) this.#byId.delete(this.#byId.keys().next().value as string);
  }

  get size(): number {
    return this.#byId.size;
  }

  keys(): IterableIterator<string> {
    return this.#byId.keys();
  }
}

function idKey(id: string): string | undefined {
  if (id === "" || id.length > MAX_REMEMBERED_ID_LENGTH) return undefined;
  return `${id.length}:${fnv1a64(id).toString(16)}`;
}

function contentHash(message: SseMessage): number {
  return fnv1a32(`${message.event}\n${message.data}`);
}

/** 32-bit FNV-1a over UTF-16 code units: cheap, and only compared against the same client's own hashes. */
function fnv1a32(text: string): number {
  let hash = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
}

const FNV64_OFFSET = 0xcbf29ce484222325n;
const FNV64_PRIME = 0x100000001b3n;

/** 64-bit FNV-1a over UTF-16 code units (ids are at most 256 of them). */
function fnv1a64(text: string): bigint {
  let hash = FNV64_OFFSET;
  for (let i = 0; i < text.length; i++) {
    hash = BigInt.asUintN(64, (hash ^ BigInt(text.charCodeAt(i))) * FNV64_PRIME);
  }
  return hash;
}
