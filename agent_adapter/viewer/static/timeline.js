// The timeline: one lane per concurrent call, lanes computed by the server from start/end times.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const FAMILIES = ["flow", "node", "tool", "llm", "context", "embedding"];

  function matches(span, filter) {
    if (!filter) return true;
    return span.node === filter;
  }

  function chips(span) {
    const out = [];
    if (span.retries) out.push(["retry ×" + span.retries, "warn"]);
    if (span.timeout) out.push(["timeout", "bad"]);
    if (span.cached) out.push(["cache hit", ""]);
    if (span.tokens) out.push([span.tokens.before + " → " + span.tokens.after + " tokens", ""]);
    if (span.cost_usd) out.push([AA.fmt.usd(span.cost_usd), ""]);
    if (span.error_kind) out.push([span.error_kind, "bad"]);
    if (span.status === "pending" || span.status === "paused") out.push(["waiting for approval", "warn"]);
    return out;
  }
  AA.spanChips = chips;

  function label(span) {
    const parts = [span.family === "context" ? "compress " + span.name : span.name || span.family];
    chips(span).forEach((c) => parts.push(c[0]));
    return parts.join(" · ");
  }

  function axis(t0, t1) {
    const node = AA.el("div", { class: "axis" });
    const total = Math.max(t1 - t0, 0.000001);
    for (let i = 0; i <= 4; i += 1) {
      const tick = AA.el("span", { text: AA.fmt.ms(((total * i) / 4) * 1000) });
      tick.style.left = i * 25 + "%";
      node.appendChild(tick);
    }
    return node;
  }

  function laneLabel(spans) {
    const families = Array.from(new Set(spans.map((s) => s.family)));
    return families.join(", ");
  }

  function legend() {
    return AA.el("div", { class: "legend" }, FAMILIES.map((f) => AA.el("span", null, [AA.el("i", { class: f }), f])));
  }

  // options: {filter: node name or null, selected: span id, onSelect(span)}
  AA.renderTimeline = function (container, run, options) {
    AA.clear(container);
    const spans = (run && run.spans) || [];
    if (!spans.length) {
      container.appendChild(AA.el("p", { class: "muted", text: "No calls recorded in this run." }));
      return;
    }
    const shown = spans.filter((s) => matches(s, options.filter));
    const t0 = spans.reduce((low, s) => Math.min(low, s.start), Infinity); // no spread: huge runs
    const t1 = spans.reduce((high, s) => Math.max(high, s.end), -Infinity);
    const total = Math.max(t1 - t0, 0.000001);
    const lanes = {};
    shown.forEach((s) => { (lanes[s.lane] = lanes[s.lane] || []).push(s); });
    const order = Object.keys(lanes).map(Number).sort((a, b) => a - b);
    if (run.spans_total > spans.length) {
      container.appendChild(AA.el("p", { class: "notice", text: "Showing " + spans.length + " of " + run.spans_total + " calls: every flow and node, then the latest calls." }));
    }
    container.appendChild(axis(t0, t1));
    const body = AA.el("div", { class: "lanes" });
    order.forEach((lane) => {
      const track = AA.el("div", { class: "lane-track" });
      lanes[lane].forEach((span) => {
        const classes = ["span", span.family, span.status, span.cached ? "cached" : "", options.selected === span.id ? "selected" : ""];
        const bar = AA.el("button", {
          type: "button",
          class: classes.join(" "),
          title: label(span) + " · " + AA.fmt.ms((span.end - span.start) * 1000),
          text: label(span),
          on: { click: () => options.onSelect(span) },
        });
        const left = ((span.start - t0) / total) * 100;
        const width = Math.max(((span.end - span.start) / total) * 100, 0.4);
        bar.style.left = left + "%";
        bar.style.width = Math.min(width, 100 - left) + "%";
        track.appendChild(bar);
      });
      body.appendChild(AA.el("div", { class: "lane" }, [
        AA.el("div", { class: "lane-label", text: "lane " + (lane + 1) + " · " + laneLabel(lanes[lane]) }),
        track,
      ]));
    });
    if (!order.length) body.appendChild(AA.el("p", { class: "muted", text: "No calls ran under this node." }));
    container.appendChild(body);
    container.appendChild(legend());
  };
})();
