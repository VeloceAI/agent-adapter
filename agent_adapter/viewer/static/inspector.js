// Chat inspector: Graph, Timeline, Events, State, Cost and Memory of the selected turn, live.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const TABS = [["graph", "Graph"], ["timeline", "Timeline"], ["events", "Events"], ["state", "State"], ["cost", "Cost"], ["memory", "Memory"]];
  const POLL_MS = 1000;
  const SETTLE_POLLS = 5; // polls after the turn ended, until the recorder wrote the run's end

  // Model calls and priced tool calls of a run, with totals (pure; tested).
  function costRows(run) {
    const rows = [];
    const total = { input: 0, output: 0, cost: 0, calls: 0 };
    ((run && run.events) || []).forEach((ev) => {
      const d = ev.data || {};
      if (ev.type !== "llm.finished") return;
      const usage = d.usage && typeof d.usage === "object" ? d.usage : {};
      const row = { kind: "model", name: [d.provider, d.model].filter(Boolean).join("/") || "model", input: Number(usage.input_tokens) || 0, output: Number(usage.output_tokens) || 0, cost: typeof d.cost_usd === "number" ? d.cost_usd : null, ms: d.latency_ms };
      rows.push(row);
      total.input += row.input;
      total.output += row.output;
      total.cost += row.cost || 0;
      total.calls += 1;
    });
    ((run && run.spans) || []).forEach((span) => {
      if (span.family !== "tool" || typeof span.cost_usd !== "number") return;
      rows.push({ kind: "tool", name: span.name, input: 0, output: 0, cost: span.cost_usd, ms: (span.end - span.start) * 1000 });
      total.cost += span.cost_usd;
    });
    return { rows: rows, total: total };
  }

  function renderCost(container, run, turn) {
    AA.clear(container);
    if (turn) {
      container.appendChild(AA.el("div", { class: "facts" }, [
        AA.el("div", null, [AA.el("span", { text: "turn tokens" }), (turn.input_tokens || 0) + " in / " + (turn.output_tokens || 0) + " out"]),
        AA.el("div", null, [AA.el("span", { text: "turn cost" }), AA.fmt.usd(turn.cost_usd || 0)]),
      ]));
    }
    const found = costRows(run);
    if (!found.rows.length) {
      container.appendChild(AA.el("p", { class: "muted", text: run ? "No model calls recorded in this run." : "The run is not recorded yet." }));
      return;
    }
    const cell = (v) => AA.el("td", { text: v });
    container.appendChild(AA.el("table", { class: "cost-table" }, [
      AA.el("thead", null, AA.el("tr", null, ["call", "in", "out", "cost", "time"].map((h) => AA.el("th", { text: h })))),
      AA.el("tbody", null, found.rows.map((r) => AA.el("tr", null, [
        AA.el("td", null, [AA.el("span", { class: "chip", text: r.kind }), " ", AA.el("code", { text: r.name })]),
        cell(r.kind === "model" ? String(r.input) : ""), cell(r.kind === "model" ? String(r.output) : ""),
        cell(r.cost === null ? "unpriced" : AA.fmt.usd(r.cost)), cell(AA.fmt.ms(r.ms)),
      ]))),
      AA.el("tfoot", null, AA.el("tr", null, [AA.el("th", { text: found.total.calls + " model call(s)" }), cell(String(found.total.input)), cell(String(found.total.output)), cell(AA.fmt.usd(found.total.cost)), cell("")])),
    ]));
  }

  // What tells two loads of a run apart: drawing again is skipped when it did not change.
  function signature(run) {
    const summary = (run && run.summary) || {};
    return run ? [run.events.length, run.events_total, run.spans_total, summary.status, summary.events].join("|") : "";
  }

  // `openRun(key)` shows a run in the Runs view; polling waits while `shown()` is false.
  AA.createInspector = function (container, openRun, shown) {
    const visible = shown || (() => true);
    const st = { tab: "graph", key: "", turn: null, run: null, node: null, filter: null, span: null, seq: 0, timer: null, settle: 0, busy: false, error: "", sessionState: null };
    const bodies = {};
    const buttons = {};
    const head = AA.el("div", { class: "insp-head" });
    const nav = AA.el("nav", { class: "tabs", role: "tablist", "aria-label": "turn inspector" });
    AA.clear(container);
    container.appendChild(head);
    container.appendChild(nav);
    TABS.forEach((pair) => {
      buttons[pair[0]] = AA.el("button", { type: "button", role: "tab", class: "tab", text: pair[1], "aria-selected": "false", on: { click: () => showTab(pair[0]) } });
      nav.appendChild(buttons[pair[0]]);
      bodies[pair[0]] = AA.el("div", { class: "tab-body insp-body", id: "insp-" + pair[0], hidden: true });
      container.appendChild(bodies[pair[0]]);
      AA.tabs.linkPanel(buttons[pair[0]], bodies[pair[0]]);
    });
    AA.tabs.arrowKeys(nav, (button) => showTab(TABS.find((pair) => buttons[pair[0]] === button)[0]));
    const graphNote = AA.el("p", { class: "muted" });
    const graphBox = AA.el("div", { class: "graph" });
    const nodeBox = AA.el("div", { class: "insp-node" });
    bodies.graph.appendChild(graphNote);
    bodies.graph.appendChild(graphBox);
    bodies.graph.appendChild(nodeBox);
    const timelineNote = AA.el("p", { class: "notice", hidden: true });
    const timelineBox = AA.el("div");
    bodies.timeline.appendChild(timelineNote);
    bodies.timeline.appendChild(timelineBox);
    const stateBox = AA.el("div", { class: "insp-session-state" });
    const stepsBox = AA.el("div");
    bodies.state.appendChild(stateBox);
    bodies.state.appendChild(stepsBox);
    const graph = AA.createGraph(graphBox, { onOpen: openNode, onEscape: closeNode });
    const events = AA.createEvents(bodies.events);
    const steps = AA.createStatePanel(stepsBox);
    const memory = AA.createMemoryPanel ? AA.createMemoryPanel(bodies.memory) : null;

    function showTab(name) {
      st.tab = name;
      AA.tabs.select(TABS.map((pair) => buttons[pair[0]]), buttons[name]);
      TABS.forEach((pair) => { bodies[pair[0]].hidden = pair[0] !== name; });
      render();
    }

    function openNode(id) { st.node = id; renderNode(); AA.revealPanel(nodeBox); }
    function closeNode() { st.node = null; graph.select(null); renderNode(); }

    function currentGraph() {
      const graphs = (st.run && st.run.graphs) || {};
      const names = Object.keys(graphs);
      return names.length ? graphs[names[0]] : null;
    }

    function renderNode() {
      AA.clear(nodeBox);
      const g = graph.shown() || currentGraph(); // an opened sub-flow, else the flow
      const node = st.node && g ? g.nodes.find((n) => n.id === st.node) : null;
      const snap = node ? graph.snapshot() : null;
      if (!node || !snap || !snap.nodes[node.id]) return;
      AA.renderNodePanel(nodeBox, node, snap.nodes[node.id], {
        onClose: closeNode,
        onTimeline: (id) => { st.filter = id; showTab("timeline"); },
        onEvents: (id) => { showTab("events"); events.showNode(id); events.focus(); },
        onState: (id) => { showTab("state"); steps.showNode(id); },
      });
    }

    function renderHead() {
      AA.clear(head);
      if (!st.turn) { head.appendChild(AA.el("p", { class: "muted", text: "Send a message, then pick a turn to inspect its run." })); return; }
      const summary = st.run && st.run.summary;
      head.appendChild(AA.el("div", { class: "card-head" }, [
        AA.el("h2", { text: "Run" }),
        AA.el("code", { class: "muted", text: st.key }),
        summary ? AA.el("span", { class: "status " + summary.status, text: summary.status }) : null,
        AA.el("span", { class: "spacer" }),
        st.run ? AA.el("button", { type: "button", class: "ghost small", text: "open in Runs", on: { click: () => openRun(st.key) } }) : null,
      ]));
      if (st.error) head.appendChild(AA.el("p", { class: "notice bad", text: st.error }));
    }

    function render() {
      renderHead();
      const run = st.run;
      if (st.tab === "graph") {
        const g = currentGraph();
        graphNote.textContent = g ? "" : run ? "This turn ran no Flow: its calls are in Timeline and Events." : "Waiting for the run's first events...";
        graphNote.hidden = Boolean(g);
        graphBox.hidden = !g;
        if (g) graph.update(g, run);
        renderNode();
      } else if (st.tab === "timeline") {
        timelineNote.hidden = !st.filter;
        AA.clear(timelineNote);
        if (st.filter) {
          timelineNote.appendChild(AA.el("span", { text: "Calls under node " + st.filter + ". " }));
          timelineNote.appendChild(AA.el("button", { type: "button", class: "ghost small", text: "show all", on: { click: () => { st.filter = null; render(); } } }));
        }
        if (run) AA.renderTimeline(timelineBox, run, { filter: st.filter, selected: st.span, onSelect: (span) => { st.span = span.id; render(); } });
        else AA.clear(timelineBox).appendChild(AA.el("p", { class: "muted", text: "Waiting for the run's first events..." }));
      } else if (st.tab === "events") events.update(run);
      else if (st.tab === "state") {
        AA.clear(stateBox);
        if (st.sessionState) {
          stateBox.appendChild(AA.el("h3", { text: "Session state after this turn" }));
          stateBox.appendChild(AA.el("pre", { text: AA.fmt.json(st.sessionState) }));
          stateBox.appendChild(AA.el("h3", { text: "Flow steps" }));
        }
        steps.update(run);
      } else if (st.tab === "memory") {
        if (memory) memory.show(st.turn);
      } else renderCost(bodies.cost, run, st.turn);
    }

    // Fetches only the events after those already held, and draws only when something changed.
    async function load() {
      if (st.busy || !st.key || !visible()) return;
      st.busy = true;
      const seq = st.seq;
      const before = signature(st.run) + "|" + st.error;
      try {
        const run = await AA.runsView.loadMore(st.key, st.run ? st.run.events : []);
        if (seq !== st.seq) return; // another turn was picked meanwhile
        st.run = run;
        st.error = "";
      } catch (err) {
        if (seq !== st.seq) return;
        if (err.status !== 404) { // 404: the recorder has not written the run yet
          console.error("Agent Adapter chat: loading run " + st.key + " failed", err);
          st.error = "Could not load the run: " + err.message;
        }
      } finally {
        st.busy = false;
      }
      if (seq !== st.seq) return;
      if (signature(st.run) + "|" + st.error !== before) render();
      const ended = st.turn && st.turn.status !== "running";
      if (ended) st.settle += 1;
      const settled = ended && ((st.run && st.run.summary && st.run.summary.status !== "running") || st.settle > SETTLE_POLLS);
      if (settled) stopPolling();
    }

    function stopPolling() {
      if (st.timer) clearInterval(st.timer);
      st.timer = null;
    }

    showTab(st.tab);

    return {
      // Show this turn's run; call again with the updated turn as it streams.
      show(turn, sessionState) {
        const key = turn ? turn.run_id : "";
        st.sessionState = sessionState || null;
        if (key !== st.key) {
          st.seq += 1;
          st.key = key;
          st.run = null;
          st.node = st.filter = st.span = null;
          st.error = "";
          st.settle = 0;
          stopPolling();
        }
        const wasRunning = st.turn && st.turn.status === "running";
        st.turn = turn;
        if (!key) { render(); return; }
        if (!st.timer && (turn.status === "running" || wasRunning || !st.run)) {
          st.settle = 0;
          const every = (AA.chatApi && AA.chatApi.pollMs) || POLL_MS; // a recorded demo reads no disk: faster
          st.timer = setInterval(load, every);
          setTimeout(load, turn.status === "running" ? Math.min(400, every) : 0); // give the recorder a moment to write the run
        }
        render();
      },
      showTab: showTab,
      current: () => ({ key: st.key, run: st.run, polling: Boolean(st.timer) }),
    };
  };

  AA.inspectorCost = { rows: costRows };
})();
