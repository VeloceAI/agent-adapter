// Runs list, run header, detail and node panels, tools catalogue and approvals.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const el = (...args) => AA.el(...args);

  function status(value) {
    return el("span", { class: "status " + value, text: value });
  }

  AA.renderRuns = function (container, runs, selected, onSelect) {
    AA.clear(container);
    if (!runs.length) {
      container.appendChild(el("p", { class: "muted", text: "No runs yet. Call record() in your app, then run it." }));
      return;
    }
    runs.forEach((run) => {
      const meta = [AA.fmt.ms(run.duration_ms), AA.fmt.usd(run.cost_usd), run.framework, (run.flows || []).join(", ")]
        .filter(Boolean).join(" · ");
      container.appendChild(el("button", {
        type: "button",
        class: "run" + (run.key === selected ? " selected" : ""),
        title: run.run_id,
        on: { click: () => onSelect(run.key) },
      }, [el("span", { class: "when", text: AA.fmt.time(run.started) }), status(run.status), el("span", { class: "meta", text: meta })]));
    });
  };

  function fact(name, value) {
    return value === "" || value === null || value === undefined ? null : el("div", null, [el("span", { text: name }), String(value)]);
  }

  AA.renderHead = function (container, run) {
    AA.clear(container);
    if (!run) {
      container.appendChild(el("p", { class: "muted", text: "Pick a run." }));
      return;
    }
    const s = run.summary;
    const meta = (run.meta || [])[0] || {};
    container.appendChild(el("div", { class: "card-head" }, [el("h2", { text: "Run" }), el("code", { text: s.run_id }), status(s.status)]));
    container.appendChild(el("div", { class: "facts" }, [
      fact("started", AA.fmt.time(s.started)),
      fact("duration", AA.fmt.ms(s.duration_ms)),
      fact("cost", AA.fmt.usd(s.cost_usd)),
      fact("framework / backend", s.framework),
      fact("events", s.events),
      fact("errors", s.errors || ""),
      fact("processes", s.processes > 1 ? s.processes : ""),
      fact("content captured", meta.capture_content ? "yes" : "no"),
    ]));
    if (run.truncated) container.appendChild(el("p", { class: "notice bad", text: "This run hit the recorder's event cap; later events were not written." }));
  };

  // A span (with its events) or one event index.
  AA.renderDetail = function (container, run, span, eventIndex) {
    AA.clear(container);
    container.appendChild(el("h2", { text: "Detail" }));
    if (!run || (!span && (eventIndex === null || eventIndex === undefined))) {
      container.appendChild(el("p", { class: "muted", text: "Click a node in the graph or a call in the timeline." }));
      return;
    }
    if (span) {
      container.appendChild(el("h3", { text: span.family + ": " + (span.name || "") }));
      container.appendChild(el("div", { class: "chips" }, [status(span.status)].concat(AA.spanChips(span).map((c) => el("span", { class: "chip " + c[1], text: c[0] })))));
      container.appendChild(el("div", { class: "facts" }, [
        fact("duration", AA.fmt.ms((span.end - span.start) * 1000)),
        fact("lane", span.lane + 1),
        fact("node", span.node),
        fact("call", span.call_id),
        fact("parent", span.parent_id),
      ]));
    }
    const indexes = span ? span.events.concat(span.marks.map((m) => m.event)) : [eventIndex];
    Array.from(new Set(indexes)).sort((a, b) => a - b).forEach((index) => {
      const ev = run.events[index];
      if (!ev) return;
      container.appendChild(el("h3", null, el("code", { text: ev.type })));
      const shown = { ts: ev.ts, run_id: ev.run_id, call_id: ev.call_id, parent_id: ev.parent_id, data: ev.data };
      if (ev.omitted) shown.omitted = ev.omitted;
      container.appendChild(el("pre", { text: AA.fmt.json(shown) }));
    });
  };

  // Scrolls a just-opened panel into view when it is wholly off screen (a stacked, narrow layout
  // puts it below the graph). True when it scrolled.
  AA.revealPanel = function (container, view) {
    const win = view || window;
    if (!container || typeof container.getBoundingClientRect !== "function") return false;
    const box = container.getBoundingClientRect();
    const height = win.innerHeight || 0;
    if (!height || (box.top < height && box.bottom > 0)) return false;
    container.scrollIntoView({ block: box.bottom - box.top > height ? "start" : "nearest" });
    return true;
  };

  // The node panel: what one graph node did. info comes from AA.graphLive.snapshot.
  AA.renderNodePanel = function (container, node, info, handlers) {
    AA.clear(container);
    const status = AA.graphLive.STATUSES.find((s) => s.key === info.status) || AA.graphLive.STATUSES[0];
    container.appendChild(el("div", { class: "card-head" }, [
      el("h2", { text: "Node" }),
      el("span", { class: "spacer" }),
      el("button", { type: "button", class: "ghost small", text: "close", title: "close (Esc)", on: { click: handlers.onClose } }),
    ]));
    container.appendChild(el("h3", null, [el("span", { class: "node-status st-" + status.key, text: status.icon + " " + status.label }), " ", el("code", { text: node.id })]));
    const kind = node.fan_out ? "fan-out over " + node.over : node.fan_in ? "fan-in of " + (node.sources || []).join(", ") : node.kind;
    container.appendChild(el("div", { class: "facts" }, [
      fact("kind", kind),
      fact("tool", node.tool),
      fact("visits", info.visits),
      fact("calls", info.calls),
      fact("errors", info.errors || ""),
      fact("cost", AA.fmt.usd(info.cost)),
      fact("items", info.items && info.items.total ? info.items.done + "/" + info.items.total + " done" : ""),
      fact("pause gate", node.pause),
      fact("timeout", node.timeout ? node.timeout + " s" : ""),
      fact("retries", node.retries || ""),
    ]));
    if (info.errorKinds.length) {
      container.appendChild(el("div", { class: "chips" }, Array.from(new Set(info.errorKinds)).map((k) => el("span", { class: "chip bad", text: k }))));
    }
    container.appendChild(el("h3", { text: "Changed state" }));
    container.appendChild(info.changed.length
      ? el("div", { class: "chips" }, info.changed.map((name) => el("span", { class: "chip", text: name })))
      : el("p", { class: "muted", text: info.visits ? "No state field changed." : "Not run yet." }));
    if (info.updates) container.appendChild(el("pre", { text: AA.fmt.json(info.updates) }));
    else if (info.changed.length) container.appendChild(el("p", { class: "muted", text: "Values are recorded with FlowRunner(trace_state=True)." }));
    if (info.visitList.length > 1) {
      container.appendChild(el("h3", { text: "Visits" }));
      container.appendChild(el("ol", { class: "visits" }, info.visitList.map((v) => el("li", { text: "visit " + v.visit + ": " + (v.changed.join(", ") || "no change") }))));
    }
    container.appendChild(el("div", { class: "actions" }, [
      el("button", { type: "button", class: "ghost", text: "Show in Timeline", on: { click: () => handlers.onTimeline(node.id) } }),
      el("button", { type: "button", class: "ghost", text: "Show in Events", on: { click: () => handlers.onEvents(node.id) } }),
      el("button", { type: "button", class: "ghost", text: "Show in State", on: { click: () => handlers.onState(node.id) } }),
    ]));
  };

  function setting(name, value) {
    if (value === null || value === undefined || value === "" || value === false) return null;
    return el("span", { class: "chip", text: name + (value === true ? "" : ": " + value) });
  }

  AA.renderTools = function (container, tools) {
    AA.clear(container);
    if (!tools) {
      container.appendChild(el("p", { class: "muted", text: "This run has no tool catalogue (record(executor) writes one when a run starts)." }));
      return;
    }
    if (!tools.length) {
      container.appendChild(el("p", { class: "muted", text: "The registry had no tools." }));
      return;
    }
    tools.forEach((tool) => {
      container.appendChild(el("details", { class: "tool-card" }, [
        el("summary", { text: tool.name }),
        el("p", { class: "muted", text: tool.description }),
        el("div", { class: "chips" }, [
          setting("kind", tool.kind),
          setting("confirm", tool.confirm === "never" ? null : tool.confirm),
          setting("cache ttl", tool.cache_ttl === null ? null : tool.cache_ttl + " s"),
          setting("cache scope", tool.cache_ttl === null ? null : tool.cache_scope),
          setting("effects", tool.effects),
          setting("idempotent", tool.idempotent === null ? null : String(tool.idempotent)),
          setting("timeout", tool.timeout === null ? null : tool.timeout + " s"),
          setting("retries", tool.retries || null),
          setting("parallel safe", tool.parallel_safe),
          setting("once per run", tool.once_per_run),
          setting("tags", (tool.tags || []).join(", ")),
        ]),
        el("pre", { text: AA.fmt.json(tool.schema) }),
      ]));
    });
  };

  AA.renderApprovals = function (container, approvals, onDecide) {
    AA.clear(container);
    container.appendChild(el("p", { class: "muted", text: "Decisions are stored for the app; its own executor.resume(approval_id) then runs the call. This page never runs a tool." }));
    if (!approvals.length) {
      container.appendChild(el("p", { class: "muted", text: "Nothing is waiting." }));
      return;
    }
    approvals.forEach((record) => {
      const reason = el("input", { type: "text", placeholder: "reason (optional)", maxlength: 500, "aria-label": "reason" });
      const notice = el("div", { class: "notice" });
      const decide = (approved) => onDecide(record, approved, reason.value, notice);
      const actions = record.decidable
        ? el("div", { class: "actions" }, [
            reason,
            el("button", { type: "button", class: "approve", text: "Approve", on: { click: () => decide(true) } }),
            el("button", { type: "button", class: "deny", text: "Deny", on: { click: () => decide(false) } }),
          ])
        : el("p", { class: "notice", text: "Not decided here: " + record.why_not });
      container.appendChild(el("div", { class: "approval" }, [
        el("div", { class: "card-head" }, [el("strong", { text: record.tool }), el("code", { text: record.approval_id })]),
        el("div", { class: "facts" }, [fact("run", record.run_id), fact("created", AA.fmt.time(record.created_at)), fact("hint", record.hint)]),
        el("pre", { text: AA.fmt.json(record.arguments) }),
        actions,
        notice,
      ]));
    });
  };
})();
