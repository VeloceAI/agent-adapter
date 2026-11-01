// Data access (live API or embedded export), DOM builders and formatters.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});

  function metaContent(name) {
    const node = document.querySelector('meta[name="' + name + '"]');
    return node ? node.getAttribute("content") || "" : "";
  }

  function readEmbedded() {
    const node = document.getElementById("aa-data");
    if (!node) return null;
    try {
      return JSON.parse(node.textContent || "null");
    } catch (err) {
      console.error("Agent Adapter viewer: the embedded data is not valid JSON", err);
      return null;
    }
  }

  const TOKEN_KEY = "aa-viewer-token";
  let memoryToken = "";

  // The access token comes in the address fragment (#token=...), which no server, log or
  // Referer ever sees. Keep it for this tab (sessionStorage is per origin, port included),
  // then remove it from the address bar and its history entry.
  function takeToken() {
    const hash = window.location.hash.replace(/^#/, "");
    const parts = hash ? hash.split("&") : [];
    const found = parts.find((part) => part.indexOf("token=") === 0);
    if (found) {
      memoryToken = decodeURIComponent(found.slice("token=".length));
      try {
        window.sessionStorage.setItem(TOKEN_KEY, memoryToken);
      } catch (err) {
        console.warn("Agent Adapter viewer: sessionStorage is blocked; the token lives in memory", err);
      }
      const rest = parts.filter((part) => part !== found).join("&");
      try {
        window.history.replaceState(null, "", window.location.pathname + (rest ? "#" + rest : ""));
      } catch (err) {
        console.warn("Agent Adapter viewer: cannot clean the address bar", err);
      }
      return;
    }
    try {
      memoryToken = window.sessionStorage.getItem(TOKEN_KEY) || "";
    } catch (err) {
      memoryToken = "";
    }
  }

  AA.embedded = readEmbedded();
  AA.csrf = metaContent("aa-csrf");
  AA.approvalsOn = !AA.embedded && metaContent("aa-approvals") === "on";
  AA.chatMode = AA.embedded ? "" : metaContent("aa-chat"); // "builtin", "agent" or "" (no chat)
  if (!AA.embedded) takeToken();
  AA.hasToken = () => Boolean(memoryToken);
  AA.token = () => memoryToken;

  // A 401 means a wrong or old token (the viewer restarted): callers stop polling.
  function failure(response, body) {
    const err = new Error(body.error || "HTTP " + response.status);
    err.status = response.status;
    return err;
  }

  async function getJSON(path) {
    const headers = { Accept: "application/json", "X-AA-Token": memoryToken };
    const response = await fetch(path, { headers: headers, cache: "no-store" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw failure(response, body);
    return body;
  }

  AA.api = {
    async runs() {
      if (AA.embedded) return AA.embedded.runs || [];
      return (await getJSON("api/runs")).runs || [];
    },
    async run(key, since) {
      const from = since || 0;
      if (AA.embedded) {
        const found = (AA.embedded.details || {})[key];
        return found ? Object.assign({}, found, { events: found.events.slice(from), events_from: from }) : null;
      }
      return getJSON("api/runs/" + encodeURIComponent(key) + "?since=" + from);
    },
    async approvals() {
      return (await getJSON("api/approvals")).approvals || [];
    },
    async decide(id, approved, reason) {
      const response = await fetch("api/approvals/" + encodeURIComponent(id) + "/decide", {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-AA-CSRF": AA.csrf, "X-AA-Token": memoryToken },
        body: JSON.stringify({ approved: approved, reason: reason || "" }),
      });
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw failure(response, body);
      return body.approval;
    },
  };

  // Every text goes in through textContent: recorded data is never parsed as HTML.
  AA.el = function (tag, attrs, children) {
    const node = document.createElement(tag);
    applyAttrs(node, attrs);
    appendAll(node, children);
    return node;
  };

  AA.svg = function (tag, attrs, children) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    applyAttrs(node, attrs);
    appendAll(node, children);
    return node;
  };

  function applyAttrs(node, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach((key) => {
      const value = attrs[key];
      if (value === undefined || value === null || value === false) return;
      if (key === "text") node.textContent = String(value);
      else if (key === "on") Object.keys(value).forEach((ev) => node.addEventListener(ev, value[ev]));
      else node.setAttribute(key, value === true ? "" : String(value));
    });
  }

  function appendAll(node, children) {
    if (children === undefined || children === null) return;
    (Array.isArray(children) ? children : [children]).forEach((child) => {
      if (child === null || child === undefined || child === false) return;
      node.appendChild(typeof child === "object" ? child : document.createTextNode(String(child)));
    });
  }

  // A run detail's key, or "" (a detail can come before its summary is listed).
  AA.runKey = (run) => (run && run.summary && run.summary.key) || "";

  AA.clear = function (node) {
    while (node.firstChild) node.removeChild(node.firstChild);
    return node;
  };

  AA.fmt = {
    time(ts) {
      if (!ts) return "";
      const date = new Date(ts * 1000);
      return date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" });
    },
    ms(ms) {
      if (ms === null || ms === undefined || isNaN(ms)) return "";
      if (ms < 1) return ms.toFixed(2) + " ms";
      if (ms < 1000) return Math.round(ms) + " ms";
      if (ms < 60000) return (ms / 1000).toFixed(2) + " s";
      return (ms / 60000).toFixed(1) + " min";
    },
    usd(value) {
      if (value === null || value === undefined) return "";
      if (value === 0) return "$0";
      return value < 0.01 ? "$" + value.toFixed(5) : "$" + value.toFixed(3);
    },
    json(value) {
      try {
        return JSON.stringify(value, null, 2);
      } catch (err) {
        return String(value);
      }
    },
  };

  AA.storage = {
    get(key) {
      try {
        return window.localStorage.getItem(key);
      } catch (err) {
        return null;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch (err) {
        // Storage can be blocked (private window); the theme then resets on reload.
      }
    },
  };
})();
