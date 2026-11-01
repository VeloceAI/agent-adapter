// The graph camera: wheel or pinch zoom, mouse or one-finger drag to pan, fit, minimap, keyboard focus moves.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const MIN_K = 0.15, MAX_K = 4, STEP = 1.25;

  function clampK(k) {
    return Math.min(MAX_K, Math.max(MIN_K, k));
  }

  // A view maps world to screen: screen = world * k + (x, y).
  function zoomAt(view, factor, px, py) {
    const k = clampK(view.k * factor);
    const ratio = k / view.k;
    return { k: k, x: px - (px - view.x) * ratio, y: py - (py - view.y) * ratio };
  }

  function fit(bounds, viewport, pad) {
    const margin = pad === undefined ? 16 : pad;
    const w = Math.max(bounds.w, 1), h = Math.max(bounds.h, 1);
    const k = clampK(Math.min((viewport.w - 2 * margin) / w, (viewport.h - 2 * margin) / h, 1.5));
    return { k: k, x: (viewport.w - w * k) / 2 - bounds.x * k, y: (viewport.h - h * k) / 2 - bounds.y * k };
  }

  // Two fingers moved from (a0, b0) to (a1, b1): scale by their spread, keep the midpoint's world point under it.
  function pinchView(view, a0, b0, a1, b1) {
    const d0 = Math.hypot(b0.x - a0.x, b0.y - a0.y) || 1;
    const k = clampK(view.k * Math.hypot(b1.x - a1.x, b1.y - a1.y) / d0);
    const m0x = (a0.x + b0.x) / 2, m0y = (a0.y + b0.y) / 2;
    const wx = (m0x - view.x) / view.k, wy = (m0y - view.y) / view.k;
    return { k: k, x: (a1.x + b1.x) / 2 - wx * k, y: (a1.y + b1.y) / 2 - wy * k };
  }

  function centerOn(view, wx, wy, viewport) {
    return { k: view.k, x: viewport.w / 2 - wx * view.k, y: viewport.h / 2 - wy * view.k };
  }

  // The part of the world the viewport shows, in world units.
  function visible(view, viewport) {
    return { x: -view.x / view.k, y: -view.y / view.k, w: viewport.w / view.k, h: viewport.h / view.k };
  }

  // Arrow keys: down/up follow edges, left/right go to the nearest node of the same layer.
  function neighbour(id, key, scene) {
    const here = scene.pos[id];
    if (!here) return null;
    let options;
    if (key === "ArrowDown") options = scene.succ[id] || [];
    else if (key === "ArrowUp") options = scene.pred[id] || [];
    else {
      const sign = key === "ArrowRight" ? 1 : -1;
      options = Object.keys(scene.pos).filter((other) => {
        const p = scene.pos[other];
        return other !== id && p.layer === here.layer && (p.x - here.x) * sign > 0;
      });
    }
    let best = null, bestDistance = Infinity;
    options.forEach((other) => {
      const p = scene.pos[other];
      if (!p) return;
      const distance = Math.abs(p.x - here.x) + Math.abs(p.y - here.y) / 1000;
      if (distance < bestDistance) { best = other; bestDistance = distance; }
    });
    return best;
  }

  // Keep at least `margin` px of the graph's bounds on screen, so it can't be lost off an edge.
  function clampView(view, bounds, viewport, margin) {
    const m = margin === undefined ? 40 : margin;
    let x = view.x, y = view.y;
    const left = bounds.x * view.k + x, right = (bounds.x + bounds.w) * view.k + x;
    const top = bounds.y * view.k + y, bottom = (bounds.y + bounds.h) * view.k + y;
    if (right < m) x += m - right;
    else if (left > viewport.w - m) x -= left - (viewport.w - m);
    if (bottom < m) y += m - bottom;
    else if (top > viewport.h - m) y -= top - (viewport.h - m);
    return { k: view.k, x: x, y: y };
  }

  function wheelFactor(ev) {
    const unit = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1;
    return Math.exp(-ev.deltaY * unit * 0.0015);
  }

  // Wires the camera to `stage` (the element that gets the events) and `world` (the <g> moved).
  function attach(stage, world, scene, handlers) {
    let view = { k: 1, x: 0, y: 0 };
    let drag = null;
    let pinch = null; // two touch pointers down: { a, b, view } at the start
    const touches = new Map(); // touch pointer id -> stage-local point
    let moved = false; // the reader moved the view since the last fit: a resize keeps it
    const viewport = () => ({ w: stage.clientWidth || 600, h: stage.clientHeight || 360 });

    function set(next, byReader) {
      if (byReader !== false) moved = true;
      view = scene.bounds ? clampView(next, scene.bounds, viewport()) : next;
      world.setAttribute("transform", "translate(" + view.x.toFixed(2) + "," + view.y.toFixed(2) + ") scale(" + view.k.toFixed(4) + ")");
      if (handlers.onView) handlers.onView(view, visible(view, viewport()));
    }

    function localPoint(ev) {
      const box = stage.getBoundingClientRect();
      return { x: ev.clientX - box.left, y: ev.clientY - box.top };
    }

    stage.addEventListener("wheel", (ev) => {
      ev.preventDefault();
      const p = localPoint(ev);
      set(zoomAt(view, wheelFactor(ev), p.x, p.y));
    }, { passive: false });

    // A tap on a node stays a click (no capture); a second finger anywhere starts a pinch.
    stage.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0 || (ev.target.closest && ev.target.closest(".gv-mini"))) return;
      if (ev.pointerType === "touch") touches.set(ev.pointerId, localPoint(ev));
      if (touches.size === 2) {
        const points = Array.from(touches.values());
        pinch = { a: points[0], b: points[1], view: view };
        drag = null;
        touches.forEach((point, id) => stage.setPointerCapture(id));
        stage.classList.add("panning");
        return;
      }
      if (pinch || touches.size > 2 || (ev.target.closest && ev.target.closest(".g-node"))) return;
      drag = { id: ev.pointerId, x: ev.clientX, y: ev.clientY, view: view };
      stage.setPointerCapture(ev.pointerId);
      stage.classList.add("panning");
    });
    stage.addEventListener("pointermove", (ev) => {
      if (touches.has(ev.pointerId)) touches.set(ev.pointerId, localPoint(ev));
      if (pinch) {
        if (!touches.has(ev.pointerId) || touches.size < 2) return;
        const points = Array.from(touches.values());
        set(pinchView(pinch.view, pinch.a, pinch.b, points[0], points[1]));
        return;
      }
      if (!drag || drag.id !== ev.pointerId) return;
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      set({ k: drag.view.k, x: drag.view.x + dx, y: drag.view.y + dy });
    });
    const endDrag = (ev) => {
      touches.delete(ev.pointerId);
      if (pinch) {
        if (touches.size < 2) { pinch = null; stage.classList.remove("panning"); }
        return;
      }
      if (!drag || drag.id !== ev.pointerId) return;
      drag = null;
      stage.classList.remove("panning");
    };
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);
    stage.addEventListener("lostpointercapture", endDrag);

    // A resized stage (or one shown after being hidden) refits, unless the reader moved the view.
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => {
      if (!stage.clientWidth || !stage.clientHeight || !scene.bounds) return;
      if (moved) set(view, false);
      else camera.fit();
    }) : null;
    if (observer) observer.observe(stage);

    stage.addEventListener("keydown", (ev) => {
      if (ev.target.tagName === "INPUT") return;
      const vp = viewport();
      if (ev.key === "+" || ev.key === "=") set(zoomAt(view, STEP, vp.w / 2, vp.h / 2));
      else if (ev.key === "-" || ev.key === "_") set(zoomAt(view, 1 / STEP, vp.w / 2, vp.h / 2));
      else if (ev.key === "0") camera.fit();
      else if (ev.key.indexOf("Arrow") === 0) {
        const from = handlers.focused() || scene.start;
        const next = from === handlers.focused() ? neighbour(from, ev.key, scene) : from;
        if (next) handlers.focus(next);
      } else if (ev.key === "Enter" && handlers.focused()) handlers.open(handlers.focused());
      else if (ev.key === "Escape") handlers.escape();
      else return;
      ev.preventDefault();
    });

    const camera = {
      get: () => view,
      set: set,
      fit() { set(fit(scene.bounds, viewport()), false); moved = false; },
      zoomBy(factor) { const vp = viewport(); set(zoomAt(view, factor, vp.w / 2, vp.h / 2)); },
      // Keep a node on screen: move only when it is outside the visible part.
      reveal(id) {
        const p = scene.pos[id];
        if (!p) return;
        const seen = visible(view, viewport());
        const inside = p.cx > seen.x + 20 && p.cx < seen.x + seen.w - 20 && p.cy > seen.y + 20 && p.cy < seen.y + seen.h - 20;
        if (!inside) set(centerOn(view, p.cx, p.cy, viewport()));
      },
      centerWorld(wx, wy) { set(centerOn(view, wx, wy, viewport())); },
      viewport: viewport,
      destroy() { if (observer) observer.disconnect(); },
    };
    return camera;
  }

  // The minimap: the whole graph small, a box for what the viewport shows; click or drag to move.
  function minimap(mini, scene, camera) {
    let dragging = false;
    function toWorld(ev) {
      const rect = mini.getBoundingClientRect();
      const b = scene.bounds;
      const scale = Math.max(b.w / Math.max(rect.width, 1), b.h / Math.max(rect.height, 1));
      const offX = (rect.width - b.w / scale) / 2, offY = (rect.height - b.h / scale) / 2;
      return { x: b.x + (ev.clientX - rect.left - offX) * scale, y: b.y + (ev.clientY - rect.top - offY) * scale };
    }
    mini.addEventListener("pointerdown", (ev) => {
      dragging = true;
      mini.setPointerCapture(ev.pointerId);
      const p = toWorld(ev);
      camera.centerWorld(p.x, p.y);
    });
    mini.addEventListener("pointermove", (ev) => {
      if (!dragging) return;
      const p = toWorld(ev);
      camera.centerWorld(p.x, p.y);
    });
    const stop = () => { dragging = false; };
    ["pointerup", "pointercancel", "lostpointercapture"].forEach((name) => mini.addEventListener(name, stop));
    return function show(seen) {
      const box = mini.querySelector(".mini-view");
      if (!box) return;
      box.setAttribute("x", seen.x.toFixed(1));
      box.setAttribute("y", seen.y.toFixed(1));
      box.setAttribute("width", Math.max(seen.w, 1).toFixed(1));
      box.setAttribute("height", Math.max(seen.h, 1).toFixed(1));
    };
  }

  AA.graphNav = {
    MIN_K: MIN_K, MAX_K: MAX_K, clampK: clampK, zoomAt: zoomAt, pinchView: pinchView, fit: fit, centerOn: centerOn,
    visible: visible, clampView: clampView, neighbour: neighbour, wheelFactor: wheelFactor, attach: attach, minimap: minimap,
  };
})();
