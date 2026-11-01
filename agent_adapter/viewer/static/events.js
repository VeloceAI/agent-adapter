// The Events tab: every recorded event, filtered by node, tool, kind and text, with its raw JSON.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const MAX_ROWS = 2000;

  function nameOf(ev) {
    const d = ev.data || {};
    return String(d.tool || d.node || d.flow || (d.model ? (d.provider || "") + "/" + d.model : ""));
  }

  // call id -> flow node, from the spans (which carry the node through their parent chain).
  function nodeIndex(spans) {
    const byCall = Object.create(null);
    (spans || []).forEach((s) => { if (s.call_id && s.node) byCall[s.call_id] = s.node; });
    return byCall;
  }

  function nodeOf(ev, byCall) {
    const d = ev.data || {};
    if (ev.type.indexOf("node.") === 0 && d.node) return String(d.node);
    return byCall[ev.call_id] || byCall[ev.parent_id] || "";
  }

  function options(events, byCall) {
    const nodes = new Set(), tools = new Set(), kinds = new Set();
    (events || []).forEach((ev) => {
      const node = nodeOf(ev, byCall);
      if (node) nodes.add(node);
      if (ev.data && ev.data.tool) tools.add(String(ev.data.tool));
      kinds.add(ev.type.split(".")[0]);
    });
    const sorted = (set) => Array.from(set).sort();
    return { nodes: sorted(nodes), tools: sorted(tools), kinds: sorted(kinds) };
  }

  // Indexes of the events that pass `filter` ({node, tool, kind, text}; empty means any).
  function apply(events, filter, byCall) {
    const text = String(filter.text || "").trim().toLowerCase();
    const out = [];
    (events || []).forEach((ev, index) => {
      if (filter.node && nodeOf(ev, byCall) !== filter.node) return;
      if (filter.tool && !(ev.data && String(ev.data.tool) === filter.tool)) return;
      if (filter.kind && ev.type.split(".")[0] !== filter.kind) return;
      if (text && JSON.stringify(ev).toLowerCase().indexOf(text) < 0) return;
      out.push(index);
    });
    return out;
  }

  function notes(ev) {
    const d = ev.data || {};
    const parts = [d.error_kind, d.cached ? "cached" : "", d.approval_id ? "approval " + d.approval_id : ""];
    if (Array.isArray(d.changed) && d.changed.length) parts.push("changed: " + d.changed.join(", "));
    return parts.filter(Boolean).join(" · ");
  }

  // Fills a filter <select> in place; one being used keeps its options until it loses focus.
  function fillSelect(node, label, values, current) {
    if (document.activeElement === node) return;
    const wanted = values.indexOf(current) >= 0 || !current ? values : values.concat([current]);
    const have = Array.from(node.options).slice(1).map((o) => o.value);
    if (JSON.stringify(have) !== JSON.stringify(wanted)) {
      AA.clear(node);
      node.appendChild(AA.el("option", { value: "", text: "any " + label }));
      wanted.forEach((v) => node.appendChild(AA.el("option", { value: v, text: v })));
    }
    node.value = current || "";
  }

  // A controller: the filter bar stays put (focus kept) while the rows follow the run.
  AA.createEvents = function (container) {
    const st = { run: null, filter: { node: "", tool: "", kind: "", text: "" }, selected: null, shown: [] };
    const text = AA.el("input", { type: "search", placeholder: "filter text", "aria-label": "filter events by text" });
    const pickers = {};
    const selects = AA.el("span", { class: "ev-selects" }, ["node", "tool", "kind"].map((name) => {
      const node = (pickers[name] = AA.el("select", { "aria-label": "filter events by " + name }));
      node.appendChild(AA.el("option", { value: "", text: "any " + name }));
      node.addEventListener("change", () => { st.filter[name] = node.value; renderRows(); });
      return node;
    }));
    const count = AA.el("span", { class: "muted" });
    const body = AA.el("div", { class: "ev-body", tabindex: 0, "aria-label": "events; arrow keys move, Enter shows the JSON" });
    AA.clear(container);
    container.appendChild(AA.el("div", { class: "ev-filters" }, [selects, text, count]));
    container.appendChild(body);
    text.addEventListener("input", () => { st.filter.text = text.value; renderRows(); });
    body.addEventListener("keydown", (ev) => {
      if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
      ev.preventDefault();
      const at = st.shown.indexOf(st.selected);
      const next = ev.key === "ArrowDown" ? Math.min(st.shown.length - 1, at + 1) : Math.max(0, at - 1);
      if (st.shown.length) choose(st.shown[next < 0 ? 0 : next]);
    });

    function choose(index) {
      st.selected = index;
      renderRows();
      const row = body.querySelector("tr.selected");
      if (row && row.scrollIntoView) row.scrollIntoView({ block: "nearest" });
    }

    function renderSelects(byCall) {
      const found = options(st.run ? st.run.events : [], byCall);
      fillSelect(pickers.node, "node", found.nodes, st.filter.node);
      fillSelect(pickers.tool, "tool", found.tools, st.filter.tool);
      fillSelect(pickers.kind, "kind", found.kinds, st.filter.kind);
    }

    function renderRows() {
      const events = (st.run && st.run.events) || [];
      const byCall = nodeIndex(st.run && st.run.spans);
      renderSelects(byCall);
      st.shown = apply(events, st.filter, byCall);
      count.textContent = st.shown.length + " of " + events.length + " events" + (st.shown.length > MAX_ROWS ? " (first " + MAX_ROWS + " shown)" : "");
      AA.clear(body);
      if (!events.length) {
        body.appendChild(AA.el("p", { class: "muted", text: "No events recorded in this run." }));
        return;
      }
      const t0 = events[0].ts;
      const tbody = AA.el("tbody");
      st.shown.slice(0, MAX_ROWS).forEach((index) => {
        const ev = events[index];
        const selected = index === st.selected;
        tbody.appendChild(AA.el("tr", { class: "ev" + (selected ? " selected" : ""), "aria-current": selected ? "true" : null, on: { click: () => choose(selected ? null : index) } }, [
          AA.el("td", { text: "#" + index }),
          AA.el("td", { text: AA.fmt.ms((ev.ts - t0) * 1000) }),
          AA.el("td", null, AA.el("code", { text: ev.type })),
          AA.el("td", { text: nameOf(ev) }),
          AA.el("td", { text: nodeOf(ev, byCall) }),
          AA.el("td", { text: notes(ev) }),
        ]));
        if (selected) tbody.appendChild(AA.el("tr", { class: "ev-json" }, AA.el("td", { colspan: 6 }, AA.el("pre", { text: AA.fmt.json(ev) }))));
      });
      body.appendChild(AA.el("table", null, [
        AA.el("thead", null, AA.el("tr", null, ["#", "+time", "event", "name", "node", "notes"].map((h) => AA.el("th", { text: h })))),
        tbody,
      ]));
    }

    return {
      update(run) {
        if (AA.runKey(st.run) !== AA.runKey(run)) st.selected = null;
        st.run = run;
        renderRows();
      },
      // Show one node's events (from the graph's node panel).
      showNode(node) {
        st.filter = { node: node || "", tool: "", kind: "", text: "" };
        text.value = "";
        renderRows();
      },
      focus: () => body.focus(),
    };
  };

  AA.eventFilter = { nameOf: nameOf, nodeIndex: nodeIndex, nodeOf: nodeOf, options: options, apply: apply };
})();
