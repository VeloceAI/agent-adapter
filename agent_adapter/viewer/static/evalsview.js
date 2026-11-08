// Evals view: the saved eval reports of the live viewer, one report's cases, and a comparison.
"use strict";

(function () {
  const AA = window.AA;
  if (!AA || !AA.views || !AA.evals || !AA.evalCards || AA.embedded) return; // live viewer only
  const E = AA.evals;
  const C = AA.evalCards;
  const el = AA.el;
  const POLL_MS = 5000;
  const state = { reports: [], key: "", report: null, baseKey: "", base: null, comparison: null, mode: "all", tag: "", seq: 0, locked: "", error: "", timer: null, busy: false };
  const ui = {};

  // Built here, not in index.html, so an export (which leaves this file out) stays unchanged.
  function build() {
    const tabs = document.getElementById("top-tabs");
    if (!tabs) return false;
    ui.button = el("button", { type: "button", role: "tab", id: "view-evals", "data-view": "evals", class: "top-tab", "aria-selected": "false", tabindex: "-1", text: "Evals" });
    tabs.appendChild(ui.button);
    ui.list = el("div", { class: "run-list", id: "evl-list" });
    ui.head = el("section", { class: "card", id: "evl-head" });
    ui.compare = el("section", { class: "card", id: "evl-compare", hidden: true });
    ui.groups = el("section", { class: "card", id: "evl-groups", hidden: true });
    ui.cases = el("section", { class: "card", id: "evl-cases", hidden: true });
    ui.refresh = el("button", { type: "button", class: "ghost small", text: "Refresh", on: { click: () => refreshList() } });
    ui.section = el("section", { id: "evals-view", class: "view", hidden: true }, el("div", { class: "evl-shell" }, [
      el("aside", { class: "card" }, [el("div", { class: "card-head" }, [el("h2", { text: "Eval reports" }), el("span", { class: "spacer" }), ui.refresh]), ui.list]),
      el("main", { class: "main" }, [ui.head, ui.compare, ui.groups, ui.cases]),
    ]));
    document.body.appendChild(ui.section);
    AA.views.register("evals", { section: ui.section, button: ui.button, onShow: onShow, onHide: stop });
    return true;
  }

  function onShow() {
    refreshList();
    if (!state.timer && !state.locked) state.timer = setInterval(refreshList, POLL_MS);
  }

  function stop() {
    if (state.timer) clearInterval(state.timer);
    state.timer = null;
  }

  function fail(err, what) {
    if (err && err.status === 401) {
      state.locked = "The viewer refused the access token (it may have restarted): open the address agent-adapter ui printed.";
      stop();
    } else {
      console.error("Agent Adapter viewer: " + what + " failed", err);
      state.error = "Could not load " + what + ": " + (err && err.message ? err.message : "unknown error");
    }
  }

  async function refreshList() {
    if (state.busy || state.locked) return;
    if (!AA.hasToken()) {
      state.locked = "No access token: open the full address agent-adapter ui printed (it ends in #token=...).";
      renderAll();
      return;
    }
    state.busy = true;
    try {
      const body = await E.api.list();
      state.reports = body.reports || [];
      state.folder = body.folder || "";
      if (state.error.indexOf("Could not load the report list") === 0) state.error = "";
    } catch (err) {
      fail(err, "the report list");
    } finally {
      state.busy = false;
    }
    if (state.locked) { renderAll(); return; }
    renderList();
    const readable = state.reports.filter((r) => !r.problem);
    const keys = readable.map((r) => r.key).join(",");
    const changed = keys !== state.keys;
    state.keys = keys;
    if (!state.key && readable.length) pick(readable[0].key);
    else if (!state.key || (changed && state.report)) renderHead(); // new reports to compare with
  }

  async function pick(key) {
    const seq = (state.seq += 1);
    state.key = key;
    state.report = state.comparison = state.base = null;
    state.baseKey = state.error = "";
    state.mode = "all";
    state.tag = "";
    renderAll();
    try {
      const report = await E.api.report(key);
      if (seq !== state.seq) return; // another report was picked meanwhile
      state.report = report;
    } catch (err) {
      if (seq === state.seq) fail(err, "that report");
    }
    if (seq === state.seq) renderAll();
  }

  async function pickBase(key) {
    const seq = (state.seq += 1);
    state.baseKey = key;
    state.base = state.comparison = null;
    state.error = "";
    renderCompare();
    if (!key) return;
    try {
      const both = await Promise.all([E.api.report(key), E.api.compare(key, state.key)]);
      if (seq !== state.seq) return;
      state.base = both[0];
      state.comparison = both[1];
    } catch (err) {
      if (seq === state.seq) fail(err, "the comparison");
    }
    if (seq !== state.seq) return;
    renderHead(); // keeps the open case cards as they are
    renderCompare();
  }

  async function openRun(runId, notice) {
    notice.className = "notice";
    notice.textContent = "finding the run...";
    try {
      const key = E.runKey(await AA.api.runs(), runId);
      if (!key) {
        notice.className = "notice bad";
        notice.textContent = "This run is not in the runs folder: record the eval (agent-adapter eval records by default) and open the same folder.";
        return;
      }
      notice.textContent = "";
      AA.views.show("runs");
      if (AA.runsView) AA.runsView.pick(key);
    } catch (err) {
      notice.className = "notice bad";
      notice.textContent = "Could not list the runs: " + err.message;
    }
  }

  function renderList() {
    AA.clear(ui.list);
    if (!state.reports.length) {
      ui.list.appendChild(el("p", { class: "muted", text: "No saved reports yet." }));
      return;
    }
    state.reports.forEach((r) => {
      const verdict = E.verdict(r);
      const meta = r.problem ? r.key + ": " + r.problem : E.fmt.score(r.mean_score) + " · " + r.n_passed + "/" + r.n_cases + " passed · " + (AA.fmt.usd(r.cost_usd) || "$0");
      ui.list.appendChild(el("button", {
        type: "button", class: "run" + (r.key === state.key ? " selected" : ""), "aria-current": r.key === state.key ? "true" : false,
        disabled: r.problem ? true : false, "data-key": r.key, on: { click: () => pick(r.key) },
      }, [
        el("span", { class: "when", text: r.problem ? "unreadable file" : (r.name || "eval") + " · " + E.fmt.when(r.started_at) }),
        C.chip(verdict),
        el("span", { class: "meta", text: meta }),
      ]));
    });
  }

  function renderHead() {
    AA.clear(ui.head);
    if (state.locked) { ui.head.appendChild(el("p", { class: "notice bad", text: state.locked })); return; }
    if (state.error) ui.head.appendChild(el("p", { class: "notice bad", text: state.error }));
    if (!state.key) {
      ui.head.appendChild(el("h2", { text: "Evals" }));
      ui.head.appendChild(el("p", { class: "muted", text: "Save a report to list it here, then pick one:" }));
      ui.head.appendChild(el("pre", null, el("code", { text: "agent-adapter eval app:agent cases.jsonl --save\n# or: run_eval(..., save_to=\".agent_adapter/evals\")" })));
      return;
    }
    if (!state.report) { if (!state.error) ui.head.appendChild(el("p", { class: "muted", text: "loading..." })); return; }
    ui.head.appendChild(C.headline(state.report));
    const others = state.reports.filter((r) => !r.problem && r.key !== state.key);
    const picker = el("select", { "aria-label": "baseline report", on: { change: () => pickBase(picker.value) } }, [
      el("option", { value: "", text: "pick a baseline" }),
      ...others.map((r) => el("option", { value: r.key, text: (r.name || "eval") + " · " + E.fmt.when(r.started_at) + " · " + E.fmt.score(r.mean_score) })),
    ]);
    picker.value = state.baseKey;
    ui.head.appendChild(el("div", { class: "ev-filters evl-pick" }, others.length
      ? [el("span", { text: "compare with" }), picker]
      : [el("span", { class: "muted", text: "Save another report to compare two." })]));
  }

  function renderCompare() {
    AA.clear(ui.compare);
    ui.compare.hidden = !state.report || !state.baseKey;
    if (ui.compare.hidden) return;
    ui.compare.appendChild(el("h2", { text: "Against the baseline" }));
    if (!state.comparison || !state.base) { ui.compare.appendChild(el("p", { class: "muted", text: state.error ? "" : "loading..." })); return; }
    ui.compare.appendChild(C.comparison(state.base, state.report, state.comparison));
  }

  function setFilter(mode, tag) {
    state.mode = mode;
    state.tag = tag;
    renderCases();
  }

  function renderGroups() {
    AA.clear(ui.groups);
    const tables = state.report ? C.groupTables(state.report, (tag) => setFilter(state.mode, tag)) : [];
    ui.groups.hidden = !tables.length;
    tables.forEach((node) => ui.groups.appendChild(node));
  }

  function renderCases() {
    AA.clear(ui.cases);
    ui.cases.hidden = !state.report;
    if (!state.report) return;
    const all = state.report.cases || [];
    const shown = E.filterCases(all, state.mode, state.tag);
    const mode = el("select", { "aria-label": "which cases", on: { change: () => setFilter(mode.value, state.tag) } }, [
      el("option", { value: "all", text: "all cases" }), el("option", { value: "failed", text: "failing cases" }),
    ]);
    mode.value = state.mode;
    ui.cases.appendChild(el("div", { class: "card-head" }, [el("h2", { text: "Cases" }), el("span", { class: "muted", text: shown.length + " of " + all.length })]));
    ui.cases.appendChild(el("div", { class: "ev-filters" }, [mode, state.tag ? el("button", { type: "button", class: "ghost small", text: "tag " + state.tag + " ×", title: "show every tag", on: { click: () => setFilter(state.mode, "") } }) : null]));
    if (!shown.length) ui.cases.appendChild(el("p", { class: "muted", text: "No case matches." }));
    shown.forEach((item) => ui.cases.appendChild(C.caseCard(item, openRun)));
  }

  function renderAll() {
    renderList();
    renderHead();
    renderCompare();
    renderGroups();
    renderCases();
  }

  if (build()) {
    renderAll();
    AA.evalsView = { refresh: refreshList, pick: pick, compareWith: pickBase, stop: stop, state: () => state };
  }
})();
