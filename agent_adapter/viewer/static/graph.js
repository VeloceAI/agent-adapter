// The flow graph scene: SVG nodes and curved edges over the server's layout, painted with live state.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const W = 156, H = 46, SLOT = W + 36, ROW = H + 64, PAD = 40, LOOP_LANE = 26, CORNER = 8;
  const JOIN_INSET = 14; // how far a join hexagon's top and bottom edges stop short of its sides

  // Pixel positions; `x` is in node slots (layout.py), `order` is the fallback for older files.
  function build(graph) {
    const pos = AA.graphLive.dict();
    let right = 0;
    (graph.nodes || []).forEach((n) => {
      const slot = typeof n.x === "number" ? n.x : n.order || 0;
      const x = PAD + slot * SLOT, y = PAD + (n.layer || 0) * ROW;
      pos[n.id] = { x: x, y: y, cx: x + W / 2, cy: y + H / 2, layer: n.layer || 0, node: n };
      right = Math.max(right, x + W);
    });
    const net = AA.graphLive.links({ nodes: graph.nodes, edges: (graph.edges || []).filter((e) => !e.loop) });
    const loops = (graph.edges || []).filter((e) => e.loop && pos[e.source] && pos[e.target]).length;
    const layers = graph.layers || 1;
    const bounds = { x: 0, y: 0, w: right + 16 + LOOP_LANE * (loops + 1) + 90, h: PAD * 2 + layers * ROW - (ROW - H) + 30 };
    return { pos: pos, succ: net.succ, pred: net.pred, start: graph.start, right: right + 16, bounds: bounds };
  }

  function curve(points) {
    let d = "M" + points[0].x + "," + points[0].y;
    for (let i = 1; i < points.length; i += 1) {
      const a = points[i - 1], b = points[i], mid = (a.y + b.y) / 2;
      d += " C" + a.x + "," + mid + " " + b.x + "," + mid + " " + b.x + "," + b.y;
    }
    return d;
  }

  function edgePoints(e, a, b, dx) {
    const points = [{ x: a.cx + dx, y: a.y + H }];
    (e.via || []).forEach((vx, i) => {
      points.push({ x: PAD + vx * SLOT + W / 2, y: PAD + (a.layer + 1 + i) * ROW + H / 2 });
    });
    points.push({ x: b.cx, y: b.y - 2 });
    return points;
  }

  function loopPath(a, b, lane) {
    const y1 = a.cy, y2 = b.cy, sign = y2 < y1 ? -1 : 1, r = CORNER;
    return "M" + (a.x + W) + "," + y1 + " H" + (lane - r) + " Q" + lane + "," + y1 + " " + lane + "," + (y1 + sign * r) +
      " V" + (y2 - sign * r) + " Q" + lane + "," + y2 + " " + (lane - r) + "," + y2 + " H" + (b.x + W + 4);
  }

  function baseLabel(e) {
    let text = e.label || "";
    if (e.on_exhausted) text = (text + " [on exhausted]").trim();
    return text || (e.parallel ? "parallel" : "");
  }

  function subtitle(n) {
    if (n.fan_out) return "fan-out over " + (n.over || "items");
    if (n.fan_in) return "fan-in: " + (n.sources || []).join(", ");
    if (n.subflow) return "sub-flow: " + n.subflow.flow;
    if (n.tool) return "tool: " + n.tool;
    return n.kind === "fn" ? "function" : n.kind;
  }

  function shape(n, p) {
    const parts = [];
    if (n.kind === "join") {
      const c = JOIN_INSET, x = p.x, y = p.y;
      const pts = [[x + c, y], [x + W - c, y], [x + W, y + H / 2], [x + W - c, y + H], [x + c, y + H], [x, y + H / 2]];
      parts.push(AA.svg("polygon", { class: "main", points: pts.map((q) => q.join(",")).join(" ") }));
      return parts;
    }
    if (n.fan_out) { // stacked cards: one call per item
      parts.push(AA.svg("rect", { class: "stack", x: p.x + 8, y: p.y - 8, width: W, height: H, rx: 6 }));
      parts.push(AA.svg("rect", { class: "stack", x: p.x + 4, y: p.y - 4, width: W, height: H, rx: 6 }));
    }
    parts.push(AA.svg("rect", { class: "main", x: p.x, y: p.y, width: W, height: H, rx: n.kind === "call" ? H / 2 : 6 }));
    if (n.subflow) parts.push(AA.svg("rect", { class: "subframe", x: p.x + 3, y: p.y + 3, width: W - 6, height: H - 6, rx: 4 })); // a graph inside
    return parts;
  }

  function arrowDefs() {
    const marker = (id, cls) => AA.svg("marker", { id: id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      AA.svg("path", { class: cls, d: "M0,0 L10,5 L0,10 z" }));
    return AA.svg("defs", null, [marker("aa-arrow", "g-arrow"), marker("aa-arrow-hl", "g-arrow hl")]);
  }

  function drawEdges(layer, labels, graph, scene, refs) {
    let loops = 0;
    (graph.edges || []).forEach((e) => {
      const a = scene.pos[e.source], b = scene.pos[e.target];
      if (!a || !b) return;
      const key = AA.graphLive.edgeKey(e.source, e.target);
      const classes = ["g-edge", e.kind === "route" ? "route" : "", e.parallel ? "parallel" : "", e.loop ? "loop" : ""].join(" ");
      const paths = [];
      let at;
      if (e.loop) {
        const lane = scene.right + LOOP_LANE * (loops + 1);
        paths.push(AA.svg("path", { class: classes, d: loopPath(a, b, lane), "marker-end": "url(#aa-arrow)" }));
        at = { x: lane + 5, y: (a.cy + b.cy) / 2 };
        loops += 1;
      } else {
        const strands = a.node.fan_out ? [-14, 0, 14] : [0];
        strands.forEach((dx) => {
          const points = edgePoints(e, a, b, dx);
          paths.push(AA.svg("path", { class: classes + (strands.length > 1 ? " fan" : ""), d: curve(points), "marker-end": dx === 0 ? "url(#aa-arrow)" : null }));
        });
        const points = edgePoints(e, a, b, 0);
        const p = points.length > 2 ? points[1] : { x: (points[0].x + points[1].x) / 2, y: (points[0].y + points[1].y) / 2 };
        at = { x: p.x + 8, y: p.y + 4 };
      }
      paths.forEach((path) => layer.appendChild(path));
      const label = AA.svg("text", { class: "g-label", x: at.x, y: at.y });
      labels.appendChild(label);
      (refs.edges[key] = refs.edges[key] || []).push({ edge: e, paths: paths, label: label });
    });
  }

  function drawTerminals(layer, graph, scene) {
    const start = scene.pos[graph.start];
    if (start) {
      const cy = start.y - 20;
      layer.appendChild(AA.svg("circle", { class: "g-terminal", cx: start.cx, cy: cy, r: 8 }));
      layer.appendChild(AA.svg("path", { class: "g-edge", d: "M" + start.cx + "," + (cy + 8) + " V" + (start.y - 2), "marker-end": "url(#aa-arrow)" }));
    }
    (graph.nodes || []).filter((n) => n.end && scene.pos[n.id]).forEach((n) => {
      const p = scene.pos[n.id], cy = p.y + H + 22;
      layer.appendChild(AA.svg("path", { class: "g-edge", d: "M" + p.cx + "," + (p.y + H) + " V" + (cy - 9), "marker-end": "url(#aa-arrow)" }));
      layer.appendChild(AA.svg("circle", { class: "g-terminal", cx: p.cx, cy: cy, r: 8 }));
      layer.appendChild(AA.svg("circle", { class: "g-terminal-dot", cx: p.cx, cy: cy, r: 4 }));
    });
  }

  function drawNode(layer, n, p, handlers, refs) {
    const title = AA.svg("title");
    const group = AA.svg("g", {
      class: "g-node", tabindex: -1, role: "button", "data-id": n.id,
      on: {
        click: () => handlers.open(n.id),
        focus: () => handlers.focused(n.id),
        pointerenter: () => handlers.hover(n.id),
        pointerleave: () => handlers.hover(null),
      },
    }, [title].concat(shape(n, p)));
    const icon = AA.svg("text", { class: "g-icon", x: p.x + 10, y: p.y + 19 });
    const name = AA.svg("text", { class: "g-name", x: p.x + 26, y: p.y + 19, text: clip(String(n.id), 17) });
    const sub = AA.svg("text", { class: "sub", x: p.x + 10, y: p.y + 36, text: clip(subtitle(n), 24) });
    const right = p.x + W - (n.kind === "join" ? JOIN_INSET + 4 : 8); // inside a hexagon's slanted side
    const badge = AA.svg("text", { class: "g-badge", x: right, y: p.y + 15, "text-anchor": "end" });
    const progress = AA.svg("text", { class: "g-progress", x: right, y: p.y + 37, "text-anchor": "end" });
    [icon, name, sub, badge, progress].forEach((t) => group.appendChild(t));
    if (n.pause) group.appendChild(AA.svg("text", { class: "g-gate", x: p.x + W - 8, y: p.y - 4, "text-anchor": "end", text: "⏸ gate" }));
    layer.appendChild(group);
    if (n.subflow && handlers.expand) { // outside the group, so a click opens the sub-flow, not the panel
      layer.appendChild(AA.svg("text", {
        class: "g-open", x: p.x + 8, y: p.y - 4, role: "button", "aria-label": "open sub-flow " + n.subflow.flow,
        text: "⊞ open", on: { click: () => handlers.expand(n.id) },
      }));
    }
    refs.nodes[n.id] = { group: group, title: title, icon: icon, badge: badge, progress: progress, node: n };
  }

  function clip(text, size) {
    return text.length > size ? text.slice(0, size - 1) + "…" : text;
  }

  // Draws the whole graph once into `world`; `paint` then updates it in place.
  function draw(world, graph, scene, handlers) {
    AA.clear(world);
    const refs = { nodes: AA.graphLive.dict(), edges: AA.graphLive.dict() };
    const edges = AA.svg("g", { class: "g-edges" });
    const labels = AA.svg("g", { class: "g-labels" });
    const nodes = AA.svg("g", { class: "g-nodes" });
    world.appendChild(arrowDefs());
    [edges, labels, nodes].forEach((layer) => world.appendChild(layer));
    drawEdges(edges, labels, graph, scene, refs);
    drawTerminals(edges, graph, scene);
    (graph.nodes || []).forEach((n) => drawNode(nodes, n, scene.pos[n.id], handlers, refs));
    return refs;
  }

  function setClass(node, name, on) {
    node.classList.toggle(name, Boolean(on));
  }

  function nodeText(n, info) {
    const status = AA.graphLive.STATUSES.find((s) => s.key === info.status) || AA.graphLive.STATUSES[0];
    const parts = [n.id + " (" + n.kind + ")", status.label];
    if (info.visits) parts.push(info.visits + (info.visits === 1 ? " visit" : " visits"));
    if (info.calls) parts.push(info.calls + (info.calls === 1 ? " call" : " calls"));
    if (info.errors) parts.push(info.errors + (info.errors === 1 ? " error" : " errors"));
    if (n.pause) parts.push("pause gate: " + n.pause);
    if (n.subflow) parts.push("sub-flow " + n.subflow.flow + " (o opens it)");
    return parts.join(", ");
  }

  // Live state onto the drawn graph. opts: {selected, focused, hover: {nodes, edges}|null, matches: Set|null}
  function paint(refs, world, snap, opts) {
    const ran = Object.keys(snap.nodes).some((id) => snap.nodes[id].visits > 0);
    setClass(world, "has-hl", opts.hover);
    setClass(world, "has-search", opts.matches);
    Object.keys(refs.nodes).forEach((id) => {
      const ref = refs.nodes[id];
      const info = snap.nodes[id];
      const group = ref.group;
      AA.graphLive.STATUSES.forEach((s) => setClass(group, "st-" + s.key, info.status === s.key));
      setClass(group, "pause-gate", ref.node.pause);
      setClass(group, "selected", opts.selected === id);
      setClass(group, "hl", opts.hover && opts.hover.nodes.has(id));
      setClass(group, "match", opts.matches && opts.matches.has(id));
      ref.icon.textContent = AA.graphLive.ICONS[info.status] || "";
      ref.badge.textContent = info.visits > 1 ? "×" + info.visits : "";
      ref.progress.textContent = info.items && info.items.total ? info.items.done + "/" + info.items.total : "";
      const text = nodeText(ref.node, info);
      ref.title.textContent = text;
      group.setAttribute("aria-label", text + ". Enter opens details.");
    });
    Object.keys(refs.edges).forEach((key) => {
      const count = snap.edges[key] || 0;
      refs.edges[key].forEach((item) => {
        const e = item.edge;
        let text = baseLabel(e);
        if (e.loop) {
          const max = e.max_loops === null || e.max_loops === undefined ? "" : "/" + e.max_loops;
          text = (text + " " + count + max + (count === 1 && !max ? " loop" : " loops")).trim();
        } else if (count) text = (text + " ×" + count).trim();
        item.label.textContent = text;
        setClass(item.label, "taken", count > 0);
        const lit = opts.hover && opts.hover.edges.has(key);
        item.paths.forEach((path) => {
          setClass(path, "taken", count > 0);
          setClass(path, "idle", ran && count === 0);
          setClass(path, "hl", lit);
          if (path.getAttribute("marker-end")) path.setAttribute("marker-end", lit ? "url(#aa-arrow-hl)" : "url(#aa-arrow)");
        });
      });
    });
  }

  // The minimap's content: one small box per node, coloured by status.
  function drawMini(mini, graph, scene) {
    AA.clear(mini);
    const b = scene.bounds;
    mini.setAttribute("viewBox", b.x + " " + b.y + " " + b.w + " " + b.h);
    const boxes = AA.graphLive.dict();
    (graph.nodes || []).forEach((n) => {
      const p = scene.pos[n.id];
      boxes[n.id] = AA.svg("rect", { class: "mini-node", x: p.x, y: p.y, width: W, height: H, rx: 6 });
      mini.appendChild(boxes[n.id]);
    });
    const view = AA.svg("rect", { class: "mini-view", x: 0, y: 0, width: 1, height: 1 });
    mini.appendChild(view);
    return { boxes: boxes, view: view };
  }

  function paintMini(mini, snap) {
    Object.keys(mini.boxes).forEach((id) => {
      const status = (snap.nodes[id] || {}).status;
      mini.boxes[id].setAttribute("class", "mini-node st-" + status);
    });
  }

  // A key that changes only when the drawing must change (not on live state).
  function signature(graph) {
    const nodes = (graph.nodes || []).map((n) => [n.id, n.kind, n.layer, n.x, n.order, n.end, n.pause]);
    const edges = (graph.edges || []).map((e) => [e.source, e.target, e.kind, e.label, e.loop, e.via]);
    const parent = graph.parent ? [graph.parent.flow, graph.parent.node] : null;
    return JSON.stringify([graph.flow, parent, graph.start, nodes, edges]);
  }

  AA.graphScene = { W: W, H: H, build: build, draw: draw, paint: paint, drawMini: drawMini, paintMini: paintMini, signature: signature, curve: curve };
})();
