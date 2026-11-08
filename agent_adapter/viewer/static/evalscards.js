// Evals view pieces: a report's headline, tag and grader tables, case cards, a comparison.
"use strict";

(function () {
  const AA = window.AA;
  const E = AA.evals;
  const el = AA.el;

  function chip(state) {
    return el("span", { class: "status " + state.status, text: state.text });
  }

  function fact(label, value) {
    return el("div", null, [el("span", { text: label }), value]);
  }

  // A scrollable wrapper, so a wide table never widens a phone screen.
  function scroll(table) {
    return el("div", { class: "evl-scroll" }, table);
  }

  function table(head, rows) {
    return scroll(el("table", null, [
      el("thead", null, el("tr", null, head.map((h) => el("th", { text: h })))),
      el("tbody", null, rows),
    ]));
  }

  function headline(report) {
    const verdict = E.verdict(report);
    return el("div", null, [
      el("div", { class: "card-head" }, [el("h2", { text: report.name || "eval" }), chip(verdict)]),
      el("div", { class: "evl-score-line" }, [
        el("span", { class: "evl-score", text: E.fmt.score(report.mean_score) }),
        el("meter", { min: "0", max: "1", value: String(E.fmt.score(report.mean_score) === "—" ? 0 : report.mean_score), "aria-label": "mean score" }),
      ]),
      el("div", { class: "facts" }, [
        fact("passed", report.n_passed + " / " + report.n_cases + " (" + E.fmt.percent(report.pass_rate) + ")"),
        fact("errors", String(report.n_errors)),
        fact("cost", AA.fmt.usd(report.cost_usd) || "$0"),
        fact("time", E.fmt.seconds(report.wall_s)),
        fact("started", E.fmt.when(report.started_at)),
      ]),
    ]);
  }

  // Per-tag and per-grader totals; `onTag` filters the cases by a clicked tag.
  function groupTables(report, onTag) {
    const out = [];
    const tags = E.groups(report.by_tag);
    if (tags.length) {
      out.push(el("h3", { text: "By tag" }));
      out.push(table(["tag", "cases", "passed", "mean score"], tags.map((g) => el("tr", null, [
        el("td", null, el("button", { type: "button", class: "ghost small", text: g.name, title: "show only cases tagged " + g.name, on: { click: () => onTag(g.name) } })),
        el("td", { text: String(g.n) }), el("td", { text: String(g.passed) }), el("td", { text: E.fmt.score(g.score) }),
      ]))));
    }
    const graders = E.groups(report.by_grader);
    if (graders.length) {
      out.push(el("h3", { text: "By grader" }));
      out.push(table(["grader", "graded", "passed", "mean score"], graders.map((g) => el("tr", null, [
        el("td", null, el("code", { text: g.name })), el("td", { text: String(g.n) }), el("td", { text: String(g.passed) }), el("td", { text: E.fmt.score(g.score) }),
      ]))));
    }
    return out;
  }

  function block(label, text, extra) {
    if (!text) return null;
    return el("div", { class: "evl-field" }, [el("h4", { text: label }), el("pre", { class: extra || "", text: text })]);
  }

  function grades(item) {
    if (!(item.grades || []).length) return el("p", { class: "muted", text: item.error ? "Not graded: the case did not run to the end." : "No grades." });
    return table(["grader", "score", "verdict", "reason"], item.grades.map((g) => el("tr", { class: g.passed ? "" : "evl-bad" }, [
      el("td", null, el("code", { text: g.grader })),
      el("td", { text: g.score === null || g.score === undefined ? "not graded" : E.fmt.score(g.score) }),
      el("td", null, chip(g.passed ? { text: "pass", status: "ok" } : { text: "fail", status: "failed" })),
      el("td", { class: "evl-reason", text: g.notes || "" }),
    ])));
  }

  // One case, folded: the headline shows; input, expected, output and grades open below.
  function caseCard(item, onOpenRun) {
    const notice = el("span", { class: "notice", role: "status" });
    const tags = (item.tags || []).map((t) => el("span", { class: "chip", text: t }));
    const tools = (item.tool_calls || []).length ? item.tool_calls.join(", ") : "none";
    const body = el("div", { class: "evl-body" }, [
      item.error ? el("p", { class: "notice bad", text: item.error }) : null,
      block("Input", item.input),
      block("Expected", item.expected),
      block("Output", item.output),
      el("h4", { text: "Grades" }),
      grades(item),
      el("div", { class: "facts" }, [fact("tools", tools), fact("cost", AA.fmt.usd(item.cost_usd) || "$0"), fact("time", E.fmt.seconds(item.latency_s))]),
      el("div", { class: "actions" }, [
        item.run_id ? el("button", { type: "button", class: "ghost small", text: "Open run", title: "show this case's run in the Runs view", on: { click: () => onOpenRun(item.run_id, notice) } }) : null,
        item.run_id ? el("code", { class: "muted", text: item.run_id }) : null,
        notice,
      ]),
    ]);
    return el("details", { class: "evl-case", "data-case": item.case_id }, [
      el("summary", null, [chip(E.caseStatus(item)), el("strong", { text: item.case_id }), el("span", { class: "muted", text: "score " + E.fmt.score(item.score) }), ...tags]),
      body,
    ]);
  }

  function side(item) {
    if (!item) return el("td", { class: "muted", text: "—" });
    return el("td", null, [chip(E.caseStatus(item)), " ", E.fmt.score(item.score)]);
  }

  // `head` against the baseline `base`, from the server's comparison (EvalReport.compare).
  function comparison(base, head, result) {
    const delta = result.score_delta;
    const counts = [
      ["regressed", result.regressions.length], ["improved", result.improvements.length],
      ["added", result.added.length], ["removed", result.removed.length],
    ];
    const rows = E.sideBySide(base, head, result);
    return el("div", null, [
      el("p", { class: result.regressed ? "notice bad" : "notice good", text: result.regressed
        ? result.regressions.length + " case(s) regressed against the baseline."
        : "No regressions against the baseline." }),
      el("div", { class: "facts" }, [
        fact("mean score", el("span", { class: "evl-delta " + (delta > 0 ? "up" : delta < 0 ? "down" : ""), text: E.fmt.score(base.mean_score) + " → " + E.fmt.score(head.mean_score) + " (" + E.fmt.delta(delta) + ")" })),
        ...counts.map((c) => fact(c[0], String(c[1]))),
      ]),
      table(["change", "case", "baseline", "this report"], rows.map((r) => el("tr", { class: "evl-" + r.change }, [
        el("td", { text: r.change }), el("td", null, el("code", { text: r.id })), side(r.base), side(r.head),
      ]))),
    ]);
  }

  AA.evalCards = { headline: headline, groupTables: groupTables, caseCard: caseCard, comparison: comparison, chip: chip };
})();
