// The graph view: toolbar (zoom, search, replay, export), stage, minimap and legend, kept in sync.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const PLAY_MS = 650;
  const STYLE_PROPS = ["fill", "fill-opacity", "stroke", "stroke-width", "stroke-opacity", "stroke-dasharray", "opacity", "paint-order", "font-size", "font-weight", "font-family"];

  function typing(target) {
    const tag = target && target.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (target && target.isContentEditable);
  }

  function stepText(run, index) {
    const ev = run && run.events ? run.events[index] : null;
    if (!ev) return "";
    const data = ev.data || {};
    return data.node + " " + ev.type.split(".")[1] + (data.visit ? " (visit " + data.visit + ")" : "");
  }

  function legend() {
    const statuses = AA.graphLive.STATUSES.map((s) => AA.el("span", { class: "lg st-" + s.key }, [AA.el("b", { text: s.icon }), s.label]));
    const shapes = [["fn", "function"], ["call", "tool call"], ["map", "fan-out (Map)"], ["join", "fan-in (Join)"], ["gate", "pause gate"]]
      .map((pair) => AA.el("span", { class: "lg" }, [AA.el("i", { class: "lg-shape " + pair[0] }), pair[1]]));
    const edges = [["route", "route"], ["parallel", "parallel"], ["loop", "loop: taken/max loops"], ["taken", "×N times taken"]]
      .map((pair) => AA.el("span", { class: "lg" }, [AA.el("i", { class: "lg-edge " + pair[0] }), pair[1]]));
    return AA.el("div", { class: "gv-legend", "aria-label": "graph legend" }, statuses.concat(shapes, edges));
  }

  // A standalone SVG of the whole graph with the current theme's colours written in.
  function exportSvg(svg, world, scene, name) {
    const copy = world.cloneNode(true);
    copy.removeAttribute("transform");
    const live = [world].concat(Array.from(world.querySelectorAll("*")));
    const cloned = [copy].concat(Array.from(copy.querySelectorAll("*")));
    live.forEach((node, i) => {
      const style = window.getComputedStyle(node);
      STYLE_PROPS.forEach((prop) => {
        const value = style.getPropertyValue(prop);
        if (value) cloned[i].setAttribute(prop, value);
      });
      cloned[i].removeAttribute("class");
      cloned[i].removeAttribute("tabindex");
    });
    const b = scene.bounds;
    const background = window.getComputedStyle(svg).getPropertyValue("--card") || "#ffffff";
    const out = AA.svg("svg", { width: b.w, height: b.h, viewBox: "0 0 " + b.w + " " + b.h },
      [AA.svg("rect", { x: 0, y: 0, width: b.w, height: b.h, fill: background.trim() || "#ffffff" }), copy]);
    const text = new XMLSerializer().serializeToString(out);
    const url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
    const link = AA.el("a", { href: url, download: "flow-" + String(name || "graph").replace(/[^A-Za-z0-9_.-]+/g, "_") + ".svg" });
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // container gets the view; options: {onOpen(id), onEscape(), onStep()} (onStep: replay moved).
  AA.createGraph = function (container, options) {
    const st = { graph: null, run: null, sig: "", refs: null, mini: null, selected: null, focused: null, hover: null, matches: null, step: null, steps: [], timer: null, snap: null, snapKey: "", sized: false };
    const scene = {};
    const button = (text, title, onClick) => AA.el("button", { type: "button", class: "ghost small", title: title, "aria-label": title, text: text, on: { click: onClick } });
    const search = AA.el("input", { type: "search", class: "gv-search", placeholder: "/ search nodes", "aria-label": "search nodes" });
    const play = button("▶", "play the run's node order", () => togglePlay());
    const slider = AA.el("input", { type: "range", min: 0, max: 0, value: 0, step: 1, class: "gv-slider", "aria-label": "replay step" });
    const where = AA.el("span", { class: "gv-step muted" });
    const notice = AA.el("span", { class: "gv-notice muted", role: "status" });
    const copyBox = AA.el("textarea", { class: "gv-copy", readonly: true, rows: 6, "aria-label": "Mermaid text", hidden: true });
    const bar = AA.el("div", { class: "gv-bar" }, [
      button("−", "zoom out (-)", () => camera.zoomBy(1 / 1.25)),
      button("+", "zoom in (+)", () => camera.zoomBy(1.25)),
      button("fit", "fit the graph (0)", () => camera.fit()),
      search,
      AA.el("span", { class: "gv-replay" }, [play, slider, where]),
      AA.el("span", { class: "spacer" }),
      button("SVG", "download the graph as SVG", () => exportSvg(svg, world, scene, st.graph && st.graph.flow)),
      button("Mermaid", "copy the graph as Mermaid", () => copyMermaid()),
      notice,
    ]);
    const world = AA.svg("g", { class: "gv-world" });
    const svg = AA.svg("svg", { class: "gv-svg", role: "group", "aria-label": "flow graph" }, world);
    const mini = AA.svg("svg", { class: "gv-mini", "aria-hidden": "true", preserveAspectRatio: "xMidYMid meet" });
    const stage = AA.el("div", { class: "gv-stage", tabindex: 0, role: "group", "aria-roledescription": "graph", "aria-label": "Flow graph. Arrow keys move between nodes, Enter opens one, + and - zoom, 0 fits, / searches." }, [svg, mini]);
    AA.clear(container);
    [bar, copyBox, stage, legend()].forEach((part) => container.appendChild(part));

    let showView = null; // the minimap's viewport box, once a graph is drawn
    const handlers = {
      open: (id) => { st.selected = id; paint(); if (options.onOpen) options.onOpen(id); },
      focused: (id) => { if (id !== undefined) st.focused = id; return st.focused; },
      focus: (id) => focusNode(id),
      hover: (id) => { st.hover = id && st.graph ? AA.graphLive.paths(st.graph, id) : null; paint(); },
      escape: () => { st.selected = null; st.hover = null; paint(); if (options.onEscape) options.onEscape(); },
    };
    const camera = AA.graphNav.attach(stage, world, scene, {
      focused: () => st.focused, focus: handlers.focus, open: handlers.open, escape: handlers.escape,
      onView: (view, seen) => {
        if (showView) showView(seen);
        const b = scene.bounds;
        // The minimap only helps when part of the graph is off screen.
        if (b) mini.classList.toggle("whole", seen.x <= b.x && seen.y <= b.y && seen.x + seen.w >= b.x + b.w && seen.y + seen.h >= b.y + b.h);
      },
    });
    showView = AA.graphNav.minimap(mini, scene, camera);

    function focusNode(id) {
      const ref = st.refs && st.refs.nodes[id];
      if (!ref) return;
      Object.keys(st.refs.nodes).forEach((other) => st.refs.nodes[other].group.setAttribute("tabindex", other === id ? 0 : -1));
      st.focused = id;
      ref.group.focus({ preventScroll: true });
      camera.reveal(id);
    }

    // Worked out again only when the run, the graph or the replay step changed (not on hover).
    function snapshot() {
      const live = st.step === null || st.step >= st.steps.length;
      const upto = live ? undefined : st.step === 0 ? -1 : st.steps[st.step - 1];
      const run = st.run || {};
      const key = [AA.runKey(run), (run.events || []).length, (run.spans || []).length, (run.summary || {}).status, upto, st.sig].join("|");
      if (key !== st.snapKey || !st.snap) {
        st.snapKey = key;
        st.snap = AA.graphLive.snapshot(st.graph, st.run, upto);
      }
      return st.snap;
    }

    function paint() {
      if (!st.refs) return;
      snapshot();
      AA.graphScene.paint(st.refs, world, st.snap, { selected: st.selected, hover: st.hover, matches: st.matches });
      AA.graphScene.paintMini(st.mini, st.snap);
      const at = st.step === null ? st.steps.length : st.step;
      slider.max = String(st.steps.length);
      slider.value = String(at);
      slider.disabled = !st.steps.length;
      play.disabled = !st.steps.length;
      where.textContent = !st.steps.length ? "no steps yet" : at >= st.steps.length ? "live · " + st.steps.length + " steps"
        : "step " + at + "/" + st.steps.length + (at ? " · " + stepText(st.run, st.steps[at - 1]) : " · start");
    }

    function setStep(value) {
      st.step = value >= st.steps.length ? null : Math.max(0, value);
      paint();
      if (options.onStep) options.onStep();
    }

    function togglePlay() {
      if (st.timer) { stopPlay(); return; }
      if (st.step === null) st.step = 0;
      play.textContent = "⏸";
      play.setAttribute("aria-label", "pause the replay");
      st.timer = setInterval(() => {
        const next = (st.step === null ? st.steps.length : st.step) + 1;
        setStep(next);
        if (next >= st.steps.length) stopPlay();
      }, PLAY_MS);
      paint();
    }

    function stopPlay() {
      clearInterval(st.timer);
      st.timer = null;
      play.textContent = "▶";
      play.setAttribute("aria-label", "play the run's node order");
    }

    async function copyMermaid() {
      if (!st.graph) return;
      const text = AA.graphLive.mermaid(st.graph);
      try {
        await navigator.clipboard.writeText(text);
        notice.textContent = "Mermaid copied";
        copyBox.hidden = true;
      } catch (err) {
        console.warn("Agent Adapter viewer: clipboard refused; showing the Mermaid text to copy by hand", err);
        copyBox.value = text;
        copyBox.hidden = false;
        copyBox.select();
        notice.textContent = "press Ctrl+C to copy";
      }
      setTimeout(() => { notice.textContent = ""; }, 3000);
    }

    slider.addEventListener("input", () => { stopPlay(); setStep(Number(slider.value)); });
    search.addEventListener("input", () => {
      const found = st.graph ? AA.graphLive.search(st.graph, search.value) : [];
      st.matches = search.value.trim() ? new Set(found) : null;
      where.title = st.matches ? found.length + " match(es)" : "";
      paint();
    });
    search.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && st.matches && st.matches.size) {
        ev.preventDefault();
        const first = st.graph.nodes.find((n) => st.matches.has(n.id));
        focusNode(first.id);
        handlers.open(first.id);
      } else if (ev.key === "Escape") {
        search.value = "";
        st.matches = null;
        paint();
        stage.focus();
      }
    });
    const onSlash = (ev) => {
      if (ev.key !== "/" || typing(ev.target) || container.offsetParent === null) return;
      ev.preventDefault();
      search.focus();
    };
    document.addEventListener("keydown", onSlash);

    return {
      // Draw again only when the graph changed; live state is painted in place.
      update(graph, run) {
        if (AA.runKey(run) !== AA.runKey(st.run)) { // another run: back to live, nothing picked
          stopPlay();
          st.step = null;
          st.selected = null;
          st.hover = null;
        }
        st.run = run;
        const sig = AA.graphScene.signature(graph);
        const fresh = sig !== st.sig;
        st.graph = graph;
        st.steps = AA.graphLive.replaySteps(run && run.events, graph.flow);
        if (fresh) {
          const active = document.activeElement;
          const hadFocus = Boolean(active) && active !== stage && stage.contains(active); // a node had it
          st.sig = sig;
          Object.keys(scene).forEach((key) => delete scene[key]);
          Object.assign(scene, AA.graphScene.build(graph));
          st.refs = AA.graphScene.draw(world, graph, scene, handlers);
          st.mini = AA.graphScene.drawMini(mini, graph, scene);
          st.focused = st.graph.nodes.some((n) => n.id === st.focused) ? st.focused : null;
          if (!st.sized) { // a first height that suits the graph; the user may resize it after
            st.sized = true;
            stage.style.height = Math.round(Math.min(Math.max(scene.bounds.h + 24, 260), 640)) + "px";
          }
          camera.fit();
          if (hadFocus && st.focused) focusNode(st.focused); // the redraw replaced the focused node
          else if (hadFocus) stage.focus({ preventScroll: true });
        }
        paint();
      },
      select(id) { st.selected = id; paint(); },
      snapshot: () => st.snap,
      fit: () => camera.fit(),
      destroy() { stopPlay(); camera.destroy(); document.removeEventListener("keydown", onSlash); AA.clear(container); },
    };
  };
})();
