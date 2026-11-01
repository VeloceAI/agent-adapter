// The State tab: each node step's changed fields and, for traced runs, before/after values.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const MISSING = { missing: true }; // a field no earlier step set

  function same(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  function updatesOf(d) {
    return d.updates && typeof d.updates === "object" && !Array.isArray(d.updates) ? d.updates : null;
  }

  // One entry per node.finished: what it changed, and the values before and after when traced.
  // "Before" is the last value set in the same flow: flows of one run keep separate state.
  function steps(events) {
    const known = new Map(); // flow -> Map(field -> last value set)
    const out = [];
    (events || []).forEach((ev, index) => {
      if (ev.type !== "node.finished") return;
      const d = ev.data || {};
      const flow = String(d.flow || "");
      if (!known.has(flow)) known.set(flow, new Map());
      const seen = known.get(flow);
      const changed = Array.isArray(d.changed) ? d.changed.map(String) : [];
      const updates = updatesOf(d);
      const fields = changed.map((name) => {
        const before = seen.has(name) ? seen.get(name) : MISSING;
        const hasAfter = updates !== null && Object.prototype.hasOwnProperty.call(updates, name);
        const after = hasAfter ? updates[name] : MISSING;
        return { name: name, before: before, after: after, kind: diffKind(before, after) };
      });
      if (updates) Object.keys(updates).forEach((name) => seen.set(name, updates[name]));
      out.push({ index: index, ts: ev.ts, flow: flow, node: String(d.node || ""), visit: d.visit, fields: fields, traced: updates !== null });
    });
    return out;
  }

  // The values a flow's steps had set by event `index` (worked out only for the step shown).
  function knownAfter(events, flow, index) {
    const seen = new Map();
    for (let i = 0; i <= index && i < (events || []).length; i += 1) {
      const ev = events[i];
      const d = ev.data || {};
      const updates = ev.type === "node.finished" && String(d.flow || "") === flow ? updatesOf(d) : null;
      if (updates) Object.keys(updates).forEach((name) => seen.set(name, updates[name]));
    }
    return Object.fromEntries(seen);
  }

  function diffKind(before, after) {
    if (after === MISSING) return "named";
    if (before === MISSING) return "added";
    return same(before, after) ? "same" : "changed";
  }

  function show(value) {
    if (value === MISSING) return "—";
    return typeof value === "string" ? value : AA.fmt.json(value);
  }

  AA.createStatePanel = function (container) {
    const st = { run: null, selected: null, list: [] };
    const list = AA.el("ol", { class: "st-steps", tabindex: 0, "aria-label": "node steps; arrow keys move" });
    const detail = AA.el("div", { class: "st-detail" });
    const note = AA.el("p", { class: "muted" });
    AA.clear(container);
    container.appendChild(note);
    container.appendChild(AA.el("div", { class: "st-grid" }, [list, detail]));
    list.addEventListener("keydown", (ev) => {
      if (ev.key !== "ArrowDown" && ev.key !== "ArrowUp") return;
      ev.preventDefault();
      const at = Math.max(0, st.list.findIndex((s) => s.index === st.selected));
      const next = ev.key === "ArrowDown" ? Math.min(st.list.length - 1, at + 1) : Math.max(0, at - 1);
      if (st.list.length) pick(st.list[next].index);
    });

    function pick(index) {
      st.selected = index;
      render();
    }

    function renderList() {
      AA.clear(list);
      st.list.forEach((step, i) => {
        const chips = step.fields.map((f) => AA.el("span", { class: "chip " + f.kind, text: f.name }));
        const item = AA.el("li", { class: step.index === st.selected ? "selected" : "", on: { click: () => pick(step.index) } }, [
          AA.el("span", { class: "st-n", text: String(i + 1) }),
          AA.el("strong", { text: step.node }),
          AA.el("span", { class: "muted", text: (step.visit ? " #" + step.visit : "") + (step.flow ? " · " + step.flow : "") }),
          AA.el("div", { class: "chips" }, chips.length ? chips : AA.el("span", { class: "muted", text: "no change" })),
        ]);
        list.appendChild(item);
      });
    }

    function renderDetail() {
      AA.clear(detail);
      const step = st.list.find((s) => s.index === st.selected);
      if (!step) {
        detail.appendChild(AA.el("p", { class: "muted", text: "Pick a step." }));
        return;
      }
      detail.appendChild(AA.el("h3", { text: step.node + (step.visit ? " · visit " + step.visit : "") }));
      if (!step.fields.length) detail.appendChild(AA.el("p", { class: "muted", text: "This step changed no state field." }));
      else {
        detail.appendChild(AA.el("table", { class: "st-table" }, [
          AA.el("thead", null, AA.el("tr", null, ["field", "before (last set)", "set by this step"].map((h) => AA.el("th", { text: h })))),
          AA.el("tbody", null, step.fields.map((f) => AA.el("tr", { class: "st-" + f.kind }, [
            AA.el("td", null, AA.el("code", { text: f.name })),
            AA.el("td", null, AA.el("pre", { text: show(f.before) })),
            AA.el("td", null, AA.el("pre", { text: f.kind === "named" ? "(value not recorded)" : show(f.after) })),
          ]))),
        ]));
      }
      const known = knownAfter(st.run ? st.run.events : [], step.flow, step.index);
      const fields = Object.keys(known);
      if (fields.length) {
        detail.appendChild(AA.el("details", null, [
          AA.el("summary", { text: "values known after this step (" + fields.length + ")" }),
          AA.el("pre", { text: AA.fmt.json(known) }),
        ]));
      }
    }

    function render() {
      const traced = st.list.some((s) => s.traced);
      note.textContent = !st.list.length ? "No node steps in this run (state shows for flows run by FlowRunner)."
        : traced ? "Values are what each node returned; a merged field (sum, append, keyed) shows the node's own update."
        : "Only field names are recorded. Run the flow with FlowRunner(trace_state=True) to record the values.";
      renderList();
      renderDetail();
    }

    return {
      update(run) {
        const fresh = AA.runKey(st.run) !== AA.runKey(run);
        const followed = !st.list.length || st.selected === st.list[st.list.length - 1].index; // on the latest step
        st.run = run;
        st.list = steps(run ? run.events : []);
        if (fresh || followed || !st.list.some((s) => s.index === st.selected)) st.selected = st.list.length ? st.list[st.list.length - 1].index : null;
        render();
      },
      showNode(node) {
        const last = st.list.filter((s) => s.node === node).pop();
        if (last) pick(last.index);
      },
    };
  };

  AA.stateSteps = { steps: steps, knownAfter: knownAfter, diffKind: diffKind, MISSING: MISSING };
})();
