// Compare two runs: status, duration, cost, tokens, tool calls, answer, and which steps differ.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});

  function number(value) {
    return typeof value === "number" && isFinite(value) ? value : 0;
  }

  // The comparable facts of one run (`answer` only when the caller knows it, e.g. a chat turn).
  function facts(run, answer) {
    const events = (run && run.events) || [];
    const summary = (run && run.summary) || {};
    const tools = new Map();
    let input = 0, output = 0, llm = 0, toolCalls = 0;
    const steps = [];
    const open = new Map();
    events.forEach((ev) => {
      const d = ev.data || {};
      // A nested instrumented call reports its own events; the outer one's say `nested`.
      const outer = d.nested === true;
      if (ev.type === "tool.started") {
        toolCalls += 1;
        tools.set(String(d.tool), (tools.get(String(d.tool)) || 0) + 1);
      } else if (((ev.type === "llm.finished" || ev.type === "llm.failed") && !outer) || ev.type === "llm.cached") llm += 1;
      if (ev.type === "llm.finished" && !outer && d.usage && typeof d.usage === "object") {
        input += number(d.usage.input_tokens) + number(d.usage.cached_input_tokens) + number(d.usage.cache_write_input_tokens);
        output += number(d.usage.output_tokens);
      } else if (ev.type === "node.started") {
        open.set(ev.call_id || d.node, steps.length);
        steps.push({ node: (d.flow ? d.flow + "." : "") + d.node, status: "running", changed: [] });
      } else if (ev.type === "node.finished" || ev.type === "node.failed" || ev.type === "node.paused") {
        const at = open.get(ev.call_id || d.node);
        if (at === undefined) return;
        steps[at].status = ev.type === "node.finished" ? "ok" : ev.type === "node.failed" ? "error" : "paused";
        steps[at].changed = Array.isArray(d.changed) ? d.changed.slice().sort() : [];
      }
    });
    return {
      status: summary.status || "", duration_ms: summary.duration_ms, cost_usd: summary.cost_usd,
      input_tokens: input, output_tokens: output, llm_calls: llm, tool_calls: toolCalls, tools: Object.fromEntries(tools),
      errors: summary.errors || 0, steps: steps, answer: answer === undefined ? null : answer,
    };
  }

  const MAX_CELLS = 4000000; // exact alignment up to this many step pairs (about 16 MB)

  function pair(x, y) {
    const same = x.status === y.status && x.changed.join() === y.changed.join();
    return { op: same ? "same" : "changed", a: x, b: y };
  }

  // Steps aligned by node name: the common start and end directly, the middle by longest common
  // subsequence; a huge middle is paired by position and the result marked `approximate`.
  function diffSteps(a, b) {
    let start = 0;
    while (start < a.length && start < b.length && a[start].node === b[start].node) start += 1;
    let endA = a.length, endB = b.length;
    while (endA > start && endB > start && a[endA - 1].node === b[endB - 1].node) { endA -= 1; endB -= 1; }
    const out = [];
    for (let i = 0; i < start; i += 1) out.push(pair(a[i], b[i]));
    const midA = a.slice(start, endA), midB = b.slice(start, endB);
    const middle = midA.length * midB.length <= MAX_CELLS ? lcs(midA, midB) : byPosition(midA, midB);
    middle.forEach((op) => out.push(op));
    for (let i = 0; i < a.length - endA; i += 1) out.push(pair(a[endA + i], b[endB + i]));
    if (middle.approximate) out.approximate = true;
    return out;
  }

  function lcs(a, b) {
    const n = a.length, m = b.length, width = m + 1;
    const table = new Uint32Array((n + 1) * width);
    for (let i = n - 1; i >= 0; i -= 1) {
      for (let j = m - 1; j >= 0; j -= 1) {
        table[i * width + j] = a[i].node === b[j].node ? table[(i + 1) * width + j + 1] + 1
          : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
      }
    }
    const out = [];
    let i = 0, j = 0;
    while (i < n || j < m) {
      if (i < n && j < m && a[i].node === b[j].node) {
        out.push(pair(a[i], b[j]));
        i += 1;
        j += 1;
      } else if (j < m && (i >= n || table[i * width + j + 1] >= table[(i + 1) * width + j])) {
        out.push({ op: "added", a: null, b: b[j] });
        j += 1;
      } else {
        out.push({ op: "removed", a: a[i], b: null });
        i += 1;
      }
    }
    return out;
  }

  function byPosition(a, b) {
    const out = [];
    for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
      if (i < a.length && i < b.length) out.push(a[i].node === b[i].node ? pair(a[i], b[i]) : { op: "changed", a: a[i], b: b[i] });
      else if (i < a.length) out.push({ op: "removed", a: a[i], b: null });
      else out.push({ op: "added", a: null, b: b[i] });
    }
    out.approximate = true;
    return out;
  }

  function toolText(tools) {
    const names = Object.keys(tools).sort();
    return names.length ? names.map((name) => name + " ×" + tools[name]).join(", ") : "none";
  }

  function compare(a, b) {
    const rows = [
      ["status", a.status, b.status],
      ["duration", AA.fmt.ms(a.duration_ms), AA.fmt.ms(b.duration_ms)],
      ["cost", AA.fmt.usd(a.cost_usd) || "—", AA.fmt.usd(b.cost_usd) || "—"],
      ["tokens in / out", a.input_tokens + " / " + a.output_tokens, b.input_tokens + " / " + b.output_tokens],
      ["model calls", String(a.llm_calls), String(b.llm_calls)],
      ["tool calls", a.tool_calls + " (" + toolText(a.tools) + ")", b.tool_calls + " (" + toolText(b.tools) + ")"],
      ["errors", String(a.errors), String(b.errors)],
      ["steps", String(a.steps.length), String(b.steps.length)],
    ];
    if (a.answer !== null || b.answer !== null) rows.unshift(["answer", a.answer || "", b.answer || ""]);
    const steps = diffSteps(a.steps, b.steps);
    return { rows: rows.map((r) => ({ label: r[0], a: r[1], b: r[2], same: r[1] === r[2] })), steps: steps, approximate: steps.approximate === true };
  }

  function stepCell(step) {
    if (!step) return AA.el("td", { class: "muted", text: "—" });
    return AA.el("td", null, [AA.el("code", { text: step.node }), " ", AA.el("span", { class: "status " + step.status, text: step.status }),
      step.changed.length ? AA.el("span", { class: "muted", text: " " + step.changed.join(", ") }) : null]);
  }

  // Renders a comparison; `labels` names the two sides.
  AA.renderComparison = function (container, result, labels) {
    AA.clear(container);
    container.appendChild(AA.el("table", { class: "cmp-table" }, [
      AA.el("thead", null, AA.el("tr", null, ["", labels[0], labels[1]].map((h) => AA.el("th", { text: h })))),
      AA.el("tbody", null, result.rows.map((r) => AA.el("tr", { class: r.same ? "" : "differs" }, [
        AA.el("th", { text: r.label }), AA.el("td", { text: r.a }), AA.el("td", { text: r.b }),
      ]))),
    ]));
    const differ = result.steps.filter((s) => s.op !== "same").length;
    container.appendChild(AA.el("h3", { text: differ ? differ + " step(s) differ" : "The steps match" }));
    if (result.approximate) container.appendChild(AA.el("p", { class: "muted", text: "Too many differing steps to align exactly: the middle is compared by position." }));
    container.appendChild(AA.el("table", { class: "cmp-steps" }, [
      AA.el("thead", null, AA.el("tr", null, ["", labels[0], labels[1]].map((h) => AA.el("th", { text: h })))),
      AA.el("tbody", null, result.steps.map((s) => AA.el("tr", { class: "op-" + s.op }, [AA.el("td", { text: s.op }), stepCell(s.a), stepCell(s.b)]))),
    ]));
  };

  // The Runs view's Compare tab: this run against one picked from the list.
  AA.createCompare = function (container, load) {
    const st = { run: null, other: null, otherKey: "", runs: [], seq: 0, error: "", memo: null, memoKey: "", refreshing: false };
    const picker = AA.el("select", { "aria-label": "run to compare with", on: { change: () => pickOther(picker.value) } });
    const out = AA.el("div", { class: "cmp-out" });
    AA.clear(container);
    container.appendChild(AA.el("div", { class: "ev-filters" }, [AA.el("span", { text: "compare this run with" }), picker]));
    container.appendChild(out);

    // `keep` reloads the run already shown (it changed) without blanking it meanwhile.
    async function pickOther(key, keep) {
      if (!keep) {
        st.otherKey = key;
        st.other = null;
        st.error = "";
      }
      const seq = (st.seq += 1);
      if (!keep) render();
      if (!key) return;
      st.refreshing = true;
      try {
        const other = await load(key);
        if (seq !== st.seq) return; // another pick won
        st.other = other;
        st.error = "";
      } catch (err) {
        console.error("Agent Adapter viewer: loading run " + key + " to compare failed", err);
        if (seq === st.seq) st.error = "Could not load that run: " + err.message;
      } finally {
        if (seq === st.seq) st.refreshing = false;
      }
      render();
    }

    // The comparison is worked out again only when either run changed.
    // Everything `facts` reads that can change: a status can turn "unfinished" with time alone.
    function inputsKey() {
      return [st.run, st.other].map((r) => {
        const s = r.summary || {};
        return [AA.runKey(r), r.events.length, s.status, s.duration_ms, s.cost_usd, s.errors].join("|");
      }).join("||");
    }

    function result() {
      const key = inputsKey();
      if (key !== st.memoKey) {
        st.memoKey = key;
        st.memo = compare(facts(st.run), facts(st.other));
      }
      return st.memo;
    }

    function refreshOther() {
      const listed = st.runs.find((r) => r.key === st.otherKey);
      const shown = st.other && st.other.summary;
      if (!listed || !shown || st.refreshing) return;
      if (listed.events !== shown.events || listed.status !== shown.status) pickOther(st.otherKey, true);
    }

    function renderPicker() {
      const current = AA.runKey(st.run);
      const wanted = st.runs.filter((r) => r.key !== current).map((r) => r.key + "|" + r.status).join(",");
      if (picker.dataset.list === wanted) return;
      picker.dataset.list = wanted;
      AA.clear(picker);
      picker.appendChild(AA.el("option", { value: "", text: "pick a run" }));
      st.runs.filter((r) => r.key !== current).forEach((r) => {
        picker.appendChild(AA.el("option", { value: r.key, text: AA.fmt.time(r.started) + " · " + r.status + " · " + r.run_id }));
      });
      picker.value = st.otherKey;
    }

    function render() {
      renderPicker();
      const shown = !st.run ? "none" : !st.otherKey ? "pick" : st.error ? "error:" + st.error : !st.other ? "loading" : "cmp:" + inputsKey();
      if (shown === st.shown) return; // nothing new: keep the DOM (and the reader's scroll)
      st.shown = shown;
      AA.clear(out);
      if (!st.run) return;
      if (!st.otherKey) { out.appendChild(AA.el("p", { class: "muted", text: "Pick another run to see what differs." })); return; }
      if (st.error) { out.appendChild(AA.el("p", { class: "notice bad", text: st.error })); return; }
      if (!st.other) { out.appendChild(AA.el("p", { class: "muted", text: "loading..." })); return; }
      AA.renderComparison(out, result(), ["this run", "other run"]);
    }

    return {
      update(run, runs) {
        if (AA.runKey(st.run) !== AA.runKey(run) && st.otherKey === AA.runKey(run)) {
          st.otherKey = "";
          st.other = null;
        }
        st.run = run;
        st.runs = runs || [];
        refreshOther();
        render();
      },
    };
  };

  AA.compareRuns = { facts: facts, diffSteps: diffSteps, compare: compare };
})();
