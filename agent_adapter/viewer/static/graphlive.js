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

  // A sub-flow's ids sit under its SubFlow node's visit: "<flow>:<node>:<n>/<child>:<node>:<n>".
  function scopeOf(id) {
    const text = String(id || "");
    const at = text.lastIndexOf("/");
    return at < 0 ? "" : text.slice(0, at);
  }

  // True when an id belongs to `graph`: top-level ids have no scope; a sub-flow view (with
  // `parent: {flow, node}`) takes ids under a visit of that parent node.
  function inScope(graph, id) {
    const scope = scopeOf(id);
    const parent = graph && graph.parent;
    if (!parent) return scope === "";
    return scope.slice(scope.lastIndexOf("/") + 1).indexOf(parent.flow + ":" + parent.node + ":") === 0;
  }

  // Node events of one flow, in order: what a replay steps through (`graph` scopes a sub-flow).
  function replaySteps(events, flow, graph) {
    const steps = [];
    (events || []).forEach((ev, index) => {
      const data = ev.data || {};
      if (has(NODE_STATUS, ev.type) && data.flow === flow && data.node && (!graph || inScope(graph, ev.call_id))) steps.push(index);
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
      if ((ev.type === "flow.finished" && data.flow === graph.flow && ownFlowEvent(graph, ev)) || ev.type === "run.finished") stopRunning(nodes);
      if (!has(NODE_STATUS, ev.type) || data.flow !== graph.flow || !has(nodes, data.node) || !inScope(graph, ev.call_id)) continue;
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

  // A sub-flow's flow.* events name the SubFlow node they ran under; a top-level flow's do not.
  function ownFlowEvent(graph, ev) {
    const data = ev.data || {};
    const parent = graph.parent;
    if (!parent) return !data.parent_flow;
    return data.parent_flow === parent.flow && data.parent_node === parent.node && inScope(graph, ev.call_id);
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
      if (!info || !span.events || span.events[0] > limit || !inScope(graph, span.parent_id || span.call_id)) return;
      const ended = spanEnded(span, limit);
      info.calls += 1;
      if (ended && span.status === "error") {
        info.errors += 1;
        if (span.error_kind) info.errorKinds.push(span.error_kind);
      }
      if (ended && typeof span.cost_usd === "number") info.cost = (info.cost || 0) + span.cost_usd;
      const visitCall = graph.flow + ":" + span.node + ":" + info.lastVisit;
      const parentVisit = String(span.parent_id || "").slice(scopeOf(span.parent_id).length).replace(/^\//, "");
      if (info.items && byId[span.node].fan_out && parentVisit === visitCall) {
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
    if (n.subflow) out.push("sub-flow: " + n.subflow.flow);
    if (n.pause) out.push("pause gate");
    return out;
  }

  function arrow(e) {
    let text = e.label || "";
    if (e.max_loops !== null && e.max_loops !== undefined && e.loop) text = (text + " (max " + e.max_loops + ")").trim();
    const shape = e.loop ? "-.->" : e.parallel ? "==>" : "-->";
    return text ? shape + "|" + quote(text) + "|" : shape;
  }

  // Mermaid ids: `n_<name>` at the top, `s<index path>__<name>` inside sub-flows (as Python).
  function mermaidId(name, path) {
    return path.length ? "s" + path.join("_") + "__" + name : "n_" + name;
  }

  function mermaidBody(graph, path, indent) {
    const lines = [];
    (graph.nodes || []).forEach((n, index) => {
      const text = quote(nodeLines(n).join("<br/>"));
      if (n.subflow) {
        lines.push(indent + "subgraph " + mermaidId(n.id, path) + "[" + text + "]");
        Array.prototype.push.apply(lines, mermaidBody(n.subflow, path.concat([index]), indent + "  "));
        lines.push(indent + "end");
        return;
      }
      const shape = SHAPES[n.kind] || SHAPES.fn;
      lines.push(indent + mermaidId(n.id, path) + shape[0] + text + shape[1] + (n.pause ? ":::pause" : ""));
    });
    (graph.edges || []).forEach((e) => lines.push(indent + mermaidId(e.source, path) + " " + arrow(e) + " " + mermaidId(e.target, path)));
    return lines;
  }

  function mermaid(graph) {
    const lines = ["flowchart TD", "  aa_start((start)) --> " + mermaidId(graph.start, [])];
    Array.prototype.push.apply(lines, mermaidBody(graph, [], "  "));
    (graph.nodes || []).filter((n) => n.end).forEach((n) => lines.push("  " + mermaidId(n.id, []) + " --> aa_end((end))"));
    lines.push("  classDef pause stroke-dasharray: 5 5");
    return lines.join("\n") + "\n";
  }

  AA.graphLive = {
    STATUSES: STATUSES, ICONS: ICONS, dict: dict, has: has, edgeKey: edgeKey, links: links, replaySteps: replaySteps,
    snapshot: snapshot, paths: paths, search: search, mermaid: mermaid, inScope: inScope,
  };
})();
