// What the run did on a flow graph: node status, visits, calls, cost, state, edge counts, replay.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});

  // Status of a node; the icon keeps the legend readable without colour.
  const STATUSES = [
    { key: "pending", icon: "○", label: "not run" },
    { key: "running", icon: "▶", label: "running" },
    { key: "ok", icon: "✓", label: "done" },
    { key: "error", icon: "✕", label: "failed" },
    { key: "paused", icon: "⏸", label: "paused" },
    { key: "stopped", icon: "■", label: "stopped (cancelled)" },
  ];
  const NODE_STATUS = dict({ "node.started": "running", "node.finished": "ok", "node.failed": "error", "node.paused": "paused" });
  const ICONS = dict();
  STATUSES.forEach((s) => { ICONS[s.key] = s.icon; });
  const ENDED = new Set(["ok", "failed", "unfinished"]); // run summary statuses of a run that is over

  // Lookup tables keyed by names: no prototype, so `constructor` or `__proto__` are plain keys.
  function dict(entries) {
    return Object.assign(Object.create(null), entries || {});
  }

  function has(table, key) {
    return Object.prototype.hasOwnProperty.call(table, key);
  }

  function edgeKey(source, target) {
    return source + "→" + target;
  }

  function links(graph) {
    const succ = dict(), pred = dict();
    (graph.nodes || []).forEach((n) => { succ[n.id] = []; pred[n.id] = []; });
    (graph.edges || []).forEach((e) => {
      if (!has(succ, e.source) || !has(pred, e.target)) return;
      if (succ[e.source].indexOf(e.target) < 0) succ[e.source].push(e.target);
      if (pred[e.target].indexOf(e.source) < 0) pred[e.target].push(e.source);
    });
    return { succ: succ, pred: pred };
  }

  // Node events of one flow, in order: what a replay steps through.
  function replaySteps(events, flow) {
    const steps = [];
    (events || []).forEach((ev, index) => {
      const data = ev.data || {};
      if (has(NODE_STATUS, ev.type) && data.flow === flow && data.node) steps.push(index);
    });
    return steps;
  }

  function blankInfo(node) {
    return {
      status: "pending", visits: 0, calls: 0, errors: 0, cost: null, changed: [], updates: null,
      visitList: [], items: node.fan_out ? { done: 0, total: 0 } : null, errorKinds: [], lastVisit: 0,
    };
  }

  // Everything known at event index `upto` (inclusive); live when `upto` is omitted.
  function snapshot(graph, run, upto) {
    const limit = upto === undefined || upto === null ? Infinity : upto;
    const events = (run && run.events) || [];
    const nodes = dict();
    (graph.nodes || []).forEach((n) => { nodes[n.id] = blankInfo(n); });
    const edges = dict();
    const fresh = dict(); // node -> successors not yet credited since it last finished
    const net = links(graph);
    let lastTs = null;
    for (let index = 0; index < events.length && index <= limit; index += 1) {
      const ev = events[index];
      const data = ev.data || {};
      if (typeof ev.ts === "number") lastTs = ev.ts;
      if ((ev.type === "flow.finished" && data.flow === graph.flow) || ev.type === "run.finished") stopRunning(nodes);
      if (!has(NODE_STATUS, ev.type) || data.flow !== graph.flow || !has(nodes, data.node)) continue;
      const status = NODE_STATUS[ev.type];
      applyNodeEvent(nodes[data.node], ev, status);
      if (ev.type === "node.started") {
        net.pred[data.node].forEach((source) => {
          if (has(fresh, source) && fresh[source].has(data.node)) {
            const key = edgeKey(source, data.node);
            edges[key] = (edges[key] || 0) + 1;
            fresh[source].delete(data.node);
          }
        });
      } else if (ev.type === "node.finished") {
        fresh[data.node] = new Set(net.succ[data.node]);
      }
    }
    if (limit === Infinity && run && run.summary && ENDED.has(run.summary.status)) stopRunning(nodes);
    addSpans(graph, run, nodes, limit);
    return { nodes: nodes, edges: edges, ts: lastTs, upto: limit };
  }

  // A node still "running" when its flow or run ended was cancelled (no node event says so).
  function stopRunning(nodes) {
    Object.keys(nodes).forEach((id) => { if (nodes[id].status === "running") nodes[id].status = "stopped"; });
  }

  function applyNodeEvent(info, ev, status) {
    const data = ev.data || {};
    info.status = status;
    if (ev.type === "node.started") {
      info.visits += 1;
      info.lastVisit = typeof data.visit === "number" ? data.visit : info.visits;
    } else if (ev.type === "node.finished") {
      const changed = Array.isArray(data.changed) ? data.changed.map(String) : [];
      changed.forEach((name) => { if (info.changed.indexOf(name) < 0) info.changed.push(name); });
      const updates = data.updates && typeof data.updates === "object" ? data.updates : null;
      if (updates) info.updates = updates;
      info.visitList.push({ visit: data.visit, changed: changed, updates: updates, ts: ev.ts });
    } else if (ev.type === "node.failed") {
      info.errors += 1;
      if (data.error_kind) info.errorKinds.push(String(data.error_kind));
    }
  }

  function spanEnded(span, limit) {
    if (span.status === "running") return false;
    const last = span.events && span.events.length ? span.events[span.events.length - 1] : -1;
    return last <= limit;
  }

  // Calls under each node, their cost and errors; a Map's items for its latest visit.
  function addSpans(graph, run, nodes, limit) {
    const byId = dict();
    (graph.nodes || []).forEach((n) => { byId[n.id] = n; });
    ((run && run.spans) || []).forEach((span) => {
      if (span.family === "node" || span.family === "flow") return;
      const info = has(nodes, span.node) ? nodes[span.node] : null;
      if (!info || !span.events || span.events[0] > limit) return;
      const ended = spanEnded(span, limit);
      info.calls += 1;
      if (ended && span.status === "error") {
        info.errors += 1;
        if (span.error_kind) info.errorKinds.push(span.error_kind);
      }
      if (ended && typeof span.cost_usd === "number") info.cost = (info.cost || 0) + span.cost_usd;
      const visitCall = graph.flow + ":" + span.node + ":" + info.lastVisit;
      if (info.items && byId[span.node].fan_out && span.parent_id === visitCall) {
        info.items.total += 1;
        if (ended) info.items.done += 1;
      }
    });
  }

  // Ancestors and descendants of `id` over forward edges (loops left out so a cycle stays readable).
  function paths(graph, id) {
    const forward = (graph.edges || []).filter((e) => !e.loop);
    const nodes = new Set([id]);
    const edges = new Set();
    [["source", "target"], ["target", "source"]].forEach((dir) => {
      const queue = [id];
      const seen = new Set([id]);
      while (queue.length) {
        const at = queue.shift();
        forward.forEach((e) => {
          if (e[dir[0]] !== at) return;
          edges.add(edgeKey(e.source, e.target));
          const next = e[dir[1]];
          nodes.add(next);
          if (!seen.has(next)) { seen.add(next); queue.push(next); }
        });
      }
    });
    return { nodes: nodes, edges: edges };
  }

  function search(graph, query) {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return [];
    return (graph.nodes || []).filter((n) => [n.id, n.tool, n.kind, n.over].some((v) => v && String(v).toLowerCase().indexOf(q) >= 0))
      .map((n) => n.id);
  }

  // The same text as Flow.to_mermaid(), built from the recorded graph.
  const SHAPES = { fn: ["[", "]"], call: ["([", "])"], map: ["[[", "]]"], join: ["{{", "}}"] };

  function quote(text) {
    const escaped = text.replace(/&/g, "#amp;").replace(/"/g, "#quot;").split("<br/>")
      .map((part) => part.replace(/</g, "#lt;").replace(/>/g, "#gt;")).join("<br/>");
    return '"' + escaped + '"';
  }

  function nodeLines(n) {
    const out = [String(n.id)];
    if (n.fan_out) out.push("fan-out over " + n.over);
    if (n.fan_in) out.push("fan-in: " + (n.sources || []).join(", "));
    if (n.tool && (n.kind === "call" || n.kind === "map")) out.push("tool: " + n.tool);
    if (n.pause) out.push("pause gate");
    return out;
  }

  function arrow(e) {
    let text = e.label || "";
    if (e.max_loops !== null && e.max_loops !== undefined && e.loop) text = (text + " (max " + e.max_loops + ")").trim();
    const shape = e.loop ? "-.->" : e.parallel ? "==>" : "-->";
    return text ? shape + "|" + quote(text) + "|" : shape;
  }

  function mermaid(graph) {
    const id = (name) => "n_" + name;
    const lines = ["flowchart TD", "  aa_start((start)) --> " + id(graph.start)];
    (graph.nodes || []).forEach((n) => {
      const shape = SHAPES[n.kind] || SHAPES.fn;
      lines.push("  " + id(n.id) + shape[0] + quote(nodeLines(n).join("<br/>")) + shape[1] + (n.pause ? ":::pause" : ""));
    });
    (graph.edges || []).forEach((e) => lines.push("  " + id(e.source) + " " + arrow(e) + " " + id(e.target)));
    (graph.nodes || []).filter((n) => n.end).forEach((n) => lines.push("  " + id(n.id) + " --> aa_end((end))"));
    lines.push("  classDef pause stroke-dasharray: 5 5");
    return lines.join("\n") + "\n";
  }

  AA.graphLive = {
    STATUSES: STATUSES, ICONS: ICONS, dict: dict, has: has, edgeKey: edgeKey, links: links, replaySteps: replaySteps,
    snapshot: snapshot, paths: paths, search: search, mermaid: mermaid,
  };
})();
