// Chat HTTP API: JSON calls, uploads, and each turn's Server-Sent Events read with fetch().
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const TERMINAL = { done: true, error: true };
  const MAX_FILE = 10 * 1024 * 1024;
  const MAX_SESSION_FILES = 50 * 1024 * 1024;
  // Types the built-in loop cannot send (it inlines text files only). Any other type is left to
  // the server, which sniffs the bytes: browsers name text files oddly (.ts is video/mp2t).
  const BINARY_TYPES = /^(image\/|audio\/|video\/(?!mp2t$)|application\/(pdf|zip|octet-stream|x-msdownload)$)/;

  const token = () => AA.token(); // data.js owns the access token

  // An Error carrying the HTTP status and the server's `{"error": ...}` text.
  function failure(status, body, fallback) {
    const err = new Error((body && body.error) || fallback || "HTTP " + status);
    err.status = status;
    return err;
  }

  function headers(extra) {
    return Object.assign({ Accept: "application/json", "X-AA-Token": token() }, extra || {});
  }

  async function call(method, path, body) {
    const options = { method: method, cache: "no-store", headers: headers() };
    if (method === "POST") {
      options.headers["Content-Type"] = "application/json";
      options.headers["X-AA-CSRF"] = AA.csrf || "";
      options.body = JSON.stringify(body || {});
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch (err) {
      throw failure(0, null, "the viewer is not reachable (" + err.message + ")");
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw failure(response.status, data);
    return data;
  }

  const seg = encodeURIComponent;

  // Why this file cannot be attached, or "" (the server sniffs the bytes and decides again).
  function uploadProblem(file, used, mode) {
    if (!file.size) return file.name + " is empty";
    if (file.size > MAX_FILE) return file.name + " is over 10 MB";
    if ((used || 0) + file.size > MAX_SESSION_FILES) return "a chat may hold 50 MB of files; " + file.name + " does not fit";
    const type = (file.type || "").toLowerCase();
    if (mode === "builtin" && BINARY_TYPES.test(type)) return file.name + ": the built-in agent takes text files only";
    return "";
  }

  async function upload(sessionId, file) {
    let response;
    try {
      response = await fetch("api/chat/files?session=" + seg(sessionId), {
        method: "POST",
        cache: "no-store",
        headers: headers({ "Content-Type": "application/octet-stream", "X-AA-CSRF": AA.csrf || "", "X-AA-Filename": encodeURIComponent(file.name) }),
        body: file,
      });
    } catch (err) {
      throw failure(0, null, file.name + " was not sent (" + err.message + ")");
    }
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw failure(response.status, data);
    return data;
  }

  // A stored file as a Blob (the request needs the token header, so no plain link).
  async function fileBlob(fileId) {
    const response = await fetch("api/chat/files/" + seg(fileId), { headers: headers({ Accept: "*/*" }), cache: "no-store" });
    if (!response.ok) throw failure(response.status, await response.json().catch(() => ({})));
    return response.blob();
  }

  // SSE parser: feed text chunks (split anywhere); calls onEvent({id, event, data}) per frame
  // and onComment(text) per comment line (the server's heartbeat).
  function createParser(onEvent, onComment) {
    let buffer = "";
    let frame = { id: null, event: "", data: [] };
    function line(text) {
      if (text === "") {
        if (frame.data.length) onEvent({ id: frame.id, event: frame.event || "message", data: frame.data.join("\n") });
        frame = { id: null, event: "", data: [] };
        return;
      }
      if (text[0] === ":") { if (onComment) onComment(text.slice(1).trim()); return; }
      const colon = text.indexOf(":");
      const field = colon < 0 ? text : text.slice(0, colon);
      let value = colon < 0 ? "" : text.slice(colon + 1);
      if (value[0] === " ") value = value.slice(1);
      if (field === "data") frame.data.push(value);
      else if (field === "event") frame.event = value;
      else if (field === "id" && value.indexOf("\u0000") < 0) frame.id = value;
    }
    return {
      feed(chunk) {
        buffer += chunk;
        for (;;) {
          const at = buffer.search(/[\r\n]/);
          if (at < 0) return;
          if (buffer[at] === "\r" && at === buffer.length - 1) return; // a \n may follow in the next chunk
          const width = buffer[at] === "\r" && buffer[at + 1] === "\n" ? 2 : 1;
          const text = buffer.slice(0, at);
          buffer = buffer.slice(at + width);
          line(text);
        }
      },
    };
  }

  // Reads a turn's events from `since`, reconnecting after a drop until its done/error event.
  // handlers: {onEvent(data), onStatus("open"|"retrying"), onFatal(err)}; returns {close()}.
  function streamTurn(turnId, since, handlers, deps) {
    const env = Object.assign({ fetch: (...a) => window.fetch(...a), setTimeout: (...a) => window.setTimeout(...a), clearTimeout: (...a) => window.clearTimeout(...a), idleMs: 16000, maxRetries: 8 }, deps || {});
    let next = since || 0;
    let closed = false;
    let ended = false;
    let failures = 0;
    let controller = null;
    let retryTimer = null;

    function finish() {
      closed = true;
      if (controller) controller.abort();
      if (retryTimer) env.clearTimeout(retryTimer);
    }

    function deliver(frame) {
      let data;
      try {
        data = JSON.parse(frame.data);
      } catch (err) {
        console.error("Agent Adapter chat: a stream event is not JSON; skipped", err);
        return;
      }
      if (typeof data.seq === "number") {
        if (data.seq < next) return; // already seen before a reconnect
        next = data.seq + 1;
      }
      failures = 0;
      if (TERMINAL[data.type]) ended = true;
      try {
        handlers.onEvent(data);
      } catch (err) { // a drawing bug must not lose the event's place in the stream
        console.error("Agent Adapter chat: handling " + data.type + " event " + data.seq + " of " + turnId + " failed", err);
      }
    }

    async function connect() {
      if (closed) return;
      controller = typeof AbortController === "function" ? new AbortController() : null;
      let idle = null;
      const bump = () => {
        if (idle) env.clearTimeout(idle);
        idle = env.setTimeout(() => { if (controller) controller.abort(); }, env.idleMs); // no bytes, not even a ping
      };
      try {
        const response = await env.fetch("api/chat/turns/" + seg(turnId) + "/stream?since=" + next, {
          headers: headers({ Accept: "text/event-stream" }), cache: "no-store", signal: controller ? controller.signal : undefined,
        });
        if (response.status === 401 || response.status === 404) {
          const body = await response.json().catch(() => ({}));
          throw Object.assign(failure(response.status, body), { fatal: true });
        }
        if (!response.ok || !response.body) throw failure(response.status, null, "the stream did not open");
        if (handlers.onStatus) handlers.onStatus("open");
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        const parser = createParser(deliver, null);
        bump();
        while (!ended && !closed) {
          const part = await reader.read();
          if (part.done) break;
          bump();
          parser.feed(decoder.decode(part.value, { stream: true }));
        }
        parser.feed(decoder.decode());
        if (ended || closed) {
          if (reader.cancel) reader.cancel().catch(() => {});
          closed = true;
          return;
        }
        throw failure(0, null, "the stream closed early");
      } catch (err) {
        if (closed || ended) return;
        if (err.fatal) { closed = true; handlers.onFatal(err); return; }
        failures += 1;
        if (failures > env.maxRetries) {
          closed = true;
          handlers.onFatal(failure(0, null, "lost the connection to the viewer"));
          return;
        }
        console.warn("Agent Adapter chat: stream of " + turnId + " dropped; reconnecting from " + next, err);
        if (handlers.onStatus) handlers.onStatus("retrying");
        retryTimer = env.setTimeout(connect, Math.min(250 * Math.pow(2, failures - 1), 5000));
      } finally {
        if (idle) env.clearTimeout(idle);
      }
    }

    connect();
    return { close: finish, next: () => next };
  }

  AA.chatApi = {
    info: () => call("GET", "api/chat"),
    sessions: async () => (await call("GET", "api/chat/sessions")).sessions || [],
    createSession: (title, state) => call("POST", "api/chat/sessions", state ? { title: title || "", state: state } : { title: title || "" }),
    session: (id) => call("GET", "api/chat/sessions/" + seg(id)),
    setState: (id, state) => call("POST", "api/chat/sessions/" + seg(id) + "/state", { state: state }),
    startTurn: (id, body) => call("POST", "api/chat/sessions/" + seg(id) + "/turns", body),
    stop: (turnId) => call("POST", "api/chat/turns/" + seg(turnId) + "/stop", {}),
    replay: (turnId) => call("POST", "api/chat/turns/" + seg(turnId) + "/replay", {}),
    decide: (approvalId, approved, reason) => call("POST", "api/chat/approvals/" + seg(approvalId) + "/decide", { approved: approved, reason: reason || "" }),
    evalCase: (turnId, body) => call("POST", "api/chat/turns/" + seg(turnId) + "/eval-case", body),
    upload: upload,
    uploadProblem: uploadProblem,
    fileBlob: fileBlob,
    streamTurn: streamTurn,
  };
  AA.sse = { createParser: createParser };
})();
