// The Parallel tab: how branches waited at joins, repeated calls, savings, claims and cost per agent.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const el = (...args) => AA.el(...args);

  function table(head, rows) {
    return el("table", { class: "parallel-table" }, [
      el("thead", null, [el("tr", null, head.map((h) => el("th", { text: h })))]),
      el("tbody", null, rows.map((row) => el("tr", null, row.map((cell) => el("td", { text: cell === null || cell === undefined ? "" : String(cell) }))))),
    ]);
  }

  function section(title, body) {
    return el("section", { class: "parallel-section" }, [el("h3", { text: title }), body]);
  }

  // Short sentences for the top of the tab: what overlapped and what it cost.
  AA.parallelFacts = function (report) {
    const facts = [];
    const joins = report.joins || [];
    if (joins.length) facts.push(joins.length + " parallel section(s); branches waited " + AA.fmt.ms(report.waited_ms) + " in all for the slowest one.");
    if (report.duplicate_calls) facts.push(report.duplicate_calls + " repeated call(s): the same tool with the same arguments.");
    else if (!report.arguments_recorded) facts.push("Arguments were not recorded, so repeated calls are not counted (record with capture_content=True).");
    const saved = report.saved || {};
    if (saved.inflight) facts.push(saved.inflight + " call(s) shared an identical call already running (" + AA.fmt.ms(saved.inflight_waited_ms) + " waited).");
    if (saved.cache) facts.push(saved.cache + " call(s) answered from the cache.");
    const tokens = report.tokens || {};
    const kept = (tokens.saved_by_cache || 0) + (tokens.saved_by_inflight || 0);
    if (kept) facts.push(kept + " model tokens saved: " + (tokens.saved_by_cache || 0) + " by the cache, " + (tokens.saved_by_inflight || 0) + " by sharing calls in flight.");
    const contested = (report.claims && report.claims.contested) || [];
    if (contested.length) facts.push(contested.length + " claim(s) were contested: another agent asked for work already taken.");
    return facts;
  };

  AA.renderParallel = function (container, report) {
    AA.clear(container);
    if (!report) {
      container.appendChild(el("p", { class: "muted", text: "Pick a run." }));
      return;
    }
    const facts = AA.parallelFacts(report);
    container.appendChild(facts.length
      ? el("ul", { class: "parallel-facts" }, facts.map((f) => el("li", { text: f })))
      : el("p", { class: "muted", text: "Nothing ran in parallel in this run." }));
    (report.joins || []).forEach((join) => {
      const title = (join.kind === "flow" ? "Flow " + join.where : "Agents") + " · " + AA.fmt.ms(join.wall_ms) + " · slowest: " + join.slowest;
      const rows = join.branches.map((b) => [b.name, AA.fmt.ms(b.duration_ms), b.waited_ms ? AA.fmt.ms(b.waited_ms) : "-", b.status]);
      container.appendChild(section(title, table(["branch", "took", "waited", "status"], rows)));
    });
    if ((report.duplicates || []).length) {
      const rows = report.duplicates.map((d) => [d.tool, d.count, d.agents.join(", ")]);
      container.appendChild(section("Repeated calls", table(["tool", "times", "by"], rows)));
    }
    const contested = (report.claims && report.claims.contested) || [];
    if (contested.length) {
      const rows = contested.map((c) => [c.key, c.holder, c.lost.join(", ")]);
      container.appendChild(section("Contested claims", table(["work", "held by", "also asked"], rows)));
    }
    if ((report.agents || []).length) {
      const rows = report.agents.map((a) => [a.agent, a.llm_calls, a.tool_calls, (a.input_tokens || 0) + " / " + (a.output_tokens || 0), AA.fmt.usd(a.cost_usd) || "$0", a.busy_ms ? AA.fmt.ms(a.busy_ms) : "", a.posts, a.claims_won + "/" + (a.claims_won + a.claims_lost)]);
      container.appendChild(section("Per agent", table(["agent", "model calls", "tool calls", "tokens in / out", "cost", "busy", "posts", "claims won"], rows)));
    }
    if ((report.board || []).length) {
      const rows = report.board.map((p) => [AA.fmt.time(p.ts), p.by, p.type === "board.posted" ? "posted " + p.kind : (p.won ? "claimed" : "lost claim to " + p.holder), p.key]);
      container.appendChild(section("Board history", table(["when", "who", "what", "key"], rows)));
    }
  };
})();
