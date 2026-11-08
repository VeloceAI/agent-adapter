// Runs view: page state, polling (live mode) and wiring between the graph, tabs and panels.
"use strict";

(function () {
  const AA = window.AA;
  const POLL_MS = 1000;
  const THEMES = ["auto", "light", "dark"];
  const TABS = ["timeline", "tools", "events", "state", "compare", "approvals"];
  const state = { runs: [], key: null, run: null, loaded: null, filter: null, span: null, node: null, tab: "timeline", flow: null, approvalIds: null, locked: false, tags: {}, search: null, again: false };
  const busy = { runs: false, run: false, approvals: false }; // one request of each kind at a time
  const timers = [];
  const $ = (id) => document.getElementById(id);
  const views = {};

  function applyTheme(theme) {
    if (theme === "auto") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", theme);
    $("theme").textContent = "theme: " + theme;
  }

  function setupTheme() {
    let theme = AA.storage.get("aa-viewer-theme") || "auto";
    if (THEMES.indexOf(theme) < 0) theme = "auto";
    applyTheme(theme);
    $("theme").addEventListener("click", () => {
      theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
      AA.storage.set("aa-viewer-theme", theme);
      applyTheme(theme);
    });
  }

  function showTab(name) {
    state.tab = name;
    const buttons = Array.from(document.querySelectorAll("#runs-view .tab"));
    AA.tabs.select(buttons, buttons.find((b) => b.getAttribute("data-tab") === name));
    TABS.forEach((tab) => { $("tab-" + tab).hidden = tab !== name; });
    if (name === "approvals") refreshApprovals(true);
    else renderTab();
  }

  function setupTabs() {
    document.querySelectorAll("#runs-view .tab").forEach((button) => {
      const name = button.getAttribute("data-tab");
      AA.tabs.linkPanel(button, $("tab-" + name));
      button.addEventListener("click", () => showTab(name));
    });
    AA.tabs.arrowKeys(document.querySelector("#runs-view .tabs"), (button) => showTab(button.getAttribute("data-tab")));
    const tabButtons = Array.from(document.querySelectorAll("#runs-view .tab"));
    AA.tabs.select(tabButtons, tabButtons[0]);
    $("flow-pick").addEventListener("change", () => {
      state.flow = $("flow-pick").value;
      state.node = null;
      renderRun();
    });
    views.graph = AA.createGraph($("graph"), {
      onOpen: openNode,
      onEscape: closeNode,
      onStep: () => { if (state.node) renderDetail(); }, // the panel follows the replay
    });
    views.events = AA.createEvents($("tab-events"));
    views.state = AA.createStatePanel($("tab-state"));
    views.compare = AA.createCompare($("tab-compare"), loadAll);
    document.addEventListener("keydown", (ev) => { // Esc closes the node panel (the graph handles its own)
      const tag = ev.target && ev.target.tagName;
      if (ev.key !== "Escape" || ev.defaultPrevented || !state.node || $("runs-view").hidden) return;
      if (tag !== "INPUT" && tag !== "TEXTAREA" && tag !== "SELECT") closeNode();
    });
  }

  function openNode(id) {
    state.node = id;
    state.span = null;
    renderDetail();
    AA.revealPanel($("detail"));
  }

  function closeNode() {
    state.node = null;
    views.graph.select(null);
    renderDetail();
  }

  function selectSpan(span) {
    state.span = span.id;
    state.node = null;
    views.graph.select(null);
    renderRun();
  }

  function currentGraph() {
    const graphs = (state.run && state.run.graphs) || {};
    const names = Object.keys(graphs);
    if (!names.length) return null;
    if (!graphs[state.flow]) state.flow = names[0];
    return graphs[state.flow];
  }

  function renderFlowPicker() {
    const names = Object.keys((state.run && state.run.graphs) || {});
    const pick = $("flow-pick");
    pick.hidden = names.length < 2;
    if (pick.dataset.names !== names.join(",")) {
      pick.dataset.names = names.join(",");
      AA.clear(pick);
      names.forEach((name) => pick.appendChild(AA.el("option", { value: name, text: name })));
    }
    pick.value = state.flow || "";
  }

  function renderTimeline() {
    const note = $("timeline-note");
    note.hidden = !state.filter;
    AA.clear(note);
    if (state.filter) {
      note.appendChild(AA.el("span", { text: "Showing calls under node " + state.filter + ". " }));
      note.appendChild(AA.el("button", { type: "button", class: "ghost small", text: "show all", on: { click: () => { state.filter = null; renderTimeline(); } } }));
    }
    AA.renderTimeline($("timeline-body"), state.run, { filter: state.filter, selected: state.span, onSelect: selectSpan });
  }

  // Only the visible tab is drawn on each update; the others catch up when shown.
  function renderTab() {
    const run = state.run;
    if (state.tab === "timeline") renderTimeline();
    else if (state.tab === "tools") AA.renderTools($("tab-tools"), run ? run.tools : null);
    else if (state.tab === "events") views.events.update(run);
    else if (state.tab === "state") views.state.update(run);
    else if (state.tab === "compare") views.compare.update(run, state.runs);
  }

  function renderDetail() {
    const graph = (views.graph && views.graph.shown()) || currentGraph(); // an opened sub-flow, else the flow
    const node = state.node && graph ? graph.nodes.find((n) => n.id === state.node) : null;
    const snap = node ? views.graph.snapshot() : null;
    if (node && snap && snap.nodes[node.id]) {
      AA.renderNodePanel($("detail"), node, snap.nodes[node.id], {
        onClose: closeNode,
        onTimeline: (id) => { state.filter = id; showTab("timeline"); },
        onEvents: (id) => { showTab("events"); views.events.showNode(id); views.events.focus(); },
        onState: (id) => { showTab("state"); views.state.showNode(id); },
      });
      return;
    }
    const run = state.run;
    const span = run && state.span ? run.spans.find((s) => s.id === state.span) : null;
    AA.renderDetail($("detail"), run, span, null);
  }

  function renderRun() {
    const run = state.run;
    AA.renderHead($("run-head"), run);
    if (run && !AA.embedded) {
      const key = AA.runKey(run);
      AA.runSearch.attachTags($("run-head"), key, state.tags[key] || [], (change) => AA.api.setTags(key, change).then((tags) => {
        state.tags[key] = tags;
        refreshRuns();
        return tags;
      }));
      if (AA.runCases) AA.runCases.attach($("run-head"), key, run.summary.status);
    }
    const graph = currentGraph();
    $("graph-card").hidden = !graph;
    renderFlowPicker();
    if (graph) views.graph.update(graph, run);
    renderTab();
    renderDetail();
  }

  // All of a run's events, page by page, onto `events` (already-held events are kept).
  async function fetchAll(key, events, stillWanted) {
    let detail = null;
    for (;;) {
      detail = await AA.api.run(key, events.length);
      if (!detail || !stillWanted()) return null;
      if (detail.events_total < events.length) {
        events.length = 0; // the run was read again (a file rotated): start over
        continue;
      }
      Array.prototype.push.apply(events, detail.events);
      if (!detail.events.length || events.length >= detail.events_total) break;
    }
    detail.events = events;
    return detail;
  }

  function loadAll(key) {
    return fetchAll(key, [], () => true);
  }

  // The run with only the events after `events` fetched; `events` (already held) is extended.
  function loadMore(key, events) {
    return fetchAll(key, events, () => true);
  }

  async function loadRun(key) {
    if (busy.run) return; // a slow response must not stack requests
    busy.run = true;
    try {
      const events = state.run && state.loaded === key ? state.run.events : [];
      const detail = await fetchAll(key, events, () => key === state.key);
      if (!detail) return; // another run was picked meanwhile
      state.run = detail;
      state.loaded = key;
      renderRun();
    } catch (err) {
      if (!unauthorized(err)) console.error("Agent Adapter viewer: loading run " + key + " failed", err);
    } finally {
      busy.run = false;
    }
  }

  function hashKey() {
    const match = /(?:^#|&)run=([A-Za-z0-9_-]+)/.exec(window.location.hash || "");
    return match ? match[1] : null;
  }

  function pick(key) {
    if (key !== state.key) {
      state.key = key;
      state.filter = state.span = state.node = state.flow = null;
      state.run = null;
      try {
        window.history.replaceState(null, "", "#run=" + key); // a link to this run
      } catch (err) {
        console.warn("Agent Adapter viewer: cannot update the address", err);
      }
    }
    AA.renderRuns($("run-list"), state.runs, state.key, pick, Boolean(state.search && state.search.active()));
    loadRun(key);
  }

  // Without a valid token nothing loads: say so once and stop polling.
  function locked(message) {
    state.locked = true;
    timers.forEach((timer) => clearInterval(timer));
    $("live").hidden = true;
    AA.renderHead($("run-head"), null);
    $("run-head").firstChild.textContent = message;
    $("run-head").firstChild.className = "notice bad";
  }

  function unauthorized(err) {
    if (err && err.status === 401) {
      locked("The viewer refused the access token (it may have restarted): open the address agent-adapter ui printed.");
      return true;
    }
    return false;
  }

  async function refreshRuns() {
    if (state.locked) return;
    if (busy.runs) {
      state.again = true; // the search changed mid-request: ask again when it ends
      return;
    }
    busy.runs = true;
    try {
      const body = await AA.api.runList(state.search ? state.search.params() : {});
      state.runs = body.runs || [];
      state.runs.forEach((r) => { state.tags[r.key] = r.tags || []; });
      if (state.search) state.search.paint(body);
    } catch (err) {
      if (!unauthorized(err)) console.error("Agent Adapter viewer: listing runs failed", err);
      return;
    } finally {
      busy.runs = false;
      if (state.again) {
        state.again = false;
        setTimeout(refreshRuns, 0);
      }
    }
    AA.renderRuns($("run-list"), state.runs, state.key, pick, Boolean(state.search && state.search.active()));
    if (!state.key && state.runs.length) {
      const wanted = hashKey();
      pick(state.runs.some((r) => r.key === wanted) ? wanted : state.runs[0].key);
      return;
    }
    const current = state.runs.find((r) => r.key === state.key);
    if (current && state.run && (current.events !== state.run.summary.events || current.status !== state.run.summary.status)) loadRun(state.key);
  }

  async function refreshApprovals(force) {
    if (!AA.approvalsOn || busy.approvals) return;
    const tab = $("tab-approvals");
    // Never wipe a reason being typed or a decision notice being read.
    if (!force && tab.contains(document.activeElement) && document.activeElement.tagName === "INPUT") return;
    busy.approvals = true;
    try {
      const approvals = await AA.api.approvals();
      const ids = approvals.map((a) => a.approval_id).join(",");
      $("approvals-tab").textContent = "Approvals (" + approvals.length + ")";
      if (!force && ids === state.approvalIds) return; // nothing changed: keep the inputs
      state.approvalIds = ids;
      AA.renderApprovals(tab, approvals, decide);
    } catch (err) {
      if (!unauthorized(err)) console.error("Agent Adapter viewer: listing approvals failed", err);
    } finally {
      busy.approvals = false;
    }
  }

  async function decide(record, approved, reason, notice) {
    notice.className = "notice";
    notice.textContent = "saving...";
    try {
      await AA.api.decide(record.approval_id, approved, reason);
      notice.className = "notice good";
      notice.textContent = (approved ? "Approved" : "Denied") + ". The app's executor.resume(approval_id) picks it up.";
      setTimeout(() => refreshApprovals(true), 4000);
    } catch (err) {
      notice.className = "notice bad";
      notice.textContent = "Not saved: " + err.message;
      unauthorized(err);
    }
  }

  async function start() {
    setupTheme();
    setupTabs();
    $("approvals-tab").hidden = !AA.approvalsOn;
    $("exported").hidden = !AA.embedded;
    if (!AA.embedded && !AA.hasToken()) {
      locked("No access token: open the full address agent-adapter ui printed (it ends in #token=...).");
      return;
    }
    if (!AA.embedded) state.search = AA.runSearch.createSearch($("run-search"), refreshRuns);
    await refreshRuns();
    if (AA.embedded || state.locked) return;
    $("live").hidden = false;
    timers.push(setInterval(refreshRuns, POLL_MS));
    if (AA.approvalsOn) {
      refreshApprovals();
      timers.push(setInterval(() => { if (state.tab === "approvals") refreshApprovals(); }, POLL_MS * 3));
    }
  }

  AA.runsView = { pick: pick, loadAll: loadAll, loadMore: loadMore, current: () => state.run };
  start();
})();
