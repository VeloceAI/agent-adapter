// Evals view data: saved eval reports from the live viewer, and the view's pure logic.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});

  // The same token header and error shape as AA.api in data.js; data.js is inlined into every
  // export, so the live-only evals calls live here and exports stay unchanged.
  async function getJSON(path) {
    const headers = { Accept: "application/json", "X-AA-Token": AA.token ? AA.token() : "" };
    const response = await fetch(path, { headers: headers, cache: "no-store" });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const err = new Error(body.error || "HTTP " + response.status);
      err.status = response.status;
      throw err;
    }
    return body;
  }

  const part = encodeURIComponent;
  const api = {
    list: () => getJSON("api/evals"),
    report: (key) => getJSON("api/evals/" + part(key)),
    compare: (base, head) => getJSON("api/evals/compare/" + part(base) + "/" + part(head)),
  };

  function finite(value) {
    return typeof value === "number" && isFinite(value) ? value : null;
  }

  const fmt = {
    score(value) {
      const n = finite(value);
      return n === null ? "—" : n.toFixed(3);
    },
    percent(value) {
      const n = finite(value);
      return n === null ? "—" : Math.round(n * 100) + "%";
    },
    seconds(value) {
      const n = finite(value);
      return n === null ? "—" : AA.fmt.ms(n * 1000);
    },
    delta(value) {
      const n = finite(value);
      if (n === null) return "—";
      return (n > 0 ? "+" : n < 0 ? "−" : "±") + Math.abs(n).toFixed(3);
    },
    // An ISO time as the viewer shows run times; the raw text when it does not parse.
    when(iso) {
      const ms = Date.parse(iso || "");
      return isNaN(ms) ? String(iso || "") : AA.fmt.time(ms / 1000);
    },
  };

  // A report's headline: {text, status} with a status class the run list already styles.
  function verdict(summary) {
    if (!summary || summary.problem) return { text: "unreadable", status: "unfinished" };
    const failed = summary.n_cases - summary.n_passed;
    if (!summary.n_cases) return { text: "no cases", status: "unfinished" };
    if (summary.n_errors) return { text: summary.n_errors + " error" + (summary.n_errors > 1 ? "s" : ""), status: "error" };
    if (!failed) return { text: "all passed", status: "ok" };
    return { text: failed + " failed", status: "failed" };
  }

  // A case's chip: passed, failed, or error when it did not run to the end.
  function caseStatus(item) {
    if (item.error) return { text: "error", status: "error" };
    return item.passed ? { text: "passed", status: "ok" } : { text: "failed", status: "failed" };
  }

  // The cases to show: every case, or only failing ones, optionally with one tag.
  function filterCases(cases, mode, tag) {
    return (cases || []).filter((item) => (mode !== "failed" || !item.passed) && (!tag || (item.tags || []).indexOf(tag) >= 0));
  }

  const ORDER = { regressed: 0, improved: 1, added: 2, removed: 3, same: 4 };

  // Two reports case by case: what each side scored and how it changed, regressions first.
  function sideBySide(base, head, comparison) {
    const before = new Map(((base && base.cases) || []).map((c) => [c.case_id, c]));
    const after = new Map(((head && head.cases) || []).map((c) => [c.case_id, c]));
    const ids = (list) => new Set(((comparison && comparison[list]) || []).map((c) => c.case_id));
    const regressed = ids("regressions"), improved = ids("improvements");
    const all = Array.from(new Set([...before.keys(), ...after.keys()])).sort();
    const rows = all.map((id) => ({
      id: id,
      base: before.get(id) || null,
      head: after.get(id) || null,
      change: regressed.has(id) ? "regressed" : improved.has(id) ? "improved"
        : !before.has(id) ? "added" : !after.has(id) ? "removed" : "same",
    }));
    return rows.sort((a, b) => ORDER[a.change] - ORDER[b.change] || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  }

  // The Runs view key of a case's run, or null when that run was not recorded there.
  function runKey(runs, runId) {
    if (!runId) return null;
    const found = (runs || []).find((r) => r.run_id === runId) || (runs || []).find((r) => r.key === runId);
    return found ? found.key : null;
  }

  // Tag or grader totals as table rows, sorted by name.
  function groups(byName) {
    return Object.keys(byName || {}).sort().map((name) => {
      const g = byName[name] || {};
      return { name: name, n: g.n || 0, passed: g.n_passed || 0, score: g.mean_score };
    });
  }

  AA.evals = { api: api, fmt: fmt, verdict: verdict, caseStatus: caseStatus, filterCases: filterCases, sideBySide: sideBySide, runKey: runKey, groups: groups };
})();
