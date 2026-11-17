// The Document card: a template document filling in, blank by blank (compose.* events).
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const TYPES = ["compose.blank_started", "compose.blank_filled", "compose.edited"];

  function problemCount(d) {
    if (Array.isArray(d.problems)) return d.problems.length;
    return typeof d.problems === "number" ? d.problems : 0;
  }

  function statusOf(blank) {
    if (blank.state === "filling") return { cls: "running", text: "filling…" };
    if (blank.before !== null) return { cls: "edited", text: "edited" };
    if (!blank.ok) return { cls: "failed", text: blank.count ? "problems " + blank.count : "failed" };
    if (blank.cached) return { cls: "cached", text: "cached" };
    return { cls: "ok", text: "ok" };
  }

  // Documents in first-seen order, each with its blanks in first-seen order. Events apply in
  // order, so a blank reported again later (cached, after the editor) shows its latest text.
  function documents(events) {
    const docs = new Map();
    const docOf = (name) => {
      if (!docs.has(name)) docs.set(name, { name: name, blanks: new Map(), revised: 0, rejected: 0 });
      return docs.get(name);
    };
    const blankOf = (doc, name) => {
      if (!doc.blanks.has(name)) {
        doc.blanks.set(name, { name: name, kind: "", state: "filling", ok: true, cached: false, attempts: 0, cost: null, count: 0, problems: [], markdown: null, before: null, reason: "" });
      }
      return doc.blanks.get(name);
    };
    (events || []).forEach((ev) => {
      if (TYPES.indexOf(ev.type) < 0) return;
      const d = ev.data || {};
      const doc = docOf(String(d.document || ""));
      if (ev.type === "compose.edited") {
        doc.revised += Number(d.revised) || 0;
        doc.rejected += Number(d.rejected) || 0;
        (Array.isArray(d.revisions) ? d.revisions : []).forEach((r) => {
          const blank = r && r.accepted === true ? doc.blanks.get(String(r.blank || "")) : null;
          if (!blank) return; // rejected, or a blank this run never reported
          blank.before = typeof r.before === "string" ? r.before : (blank.markdown || "");
          blank.markdown = typeof r.after === "string" ? r.after : blank.markdown;
          blank.reason = String(r.reason || "");
        });
        return;
      }
      const blank = blankOf(doc, String(d.blank || ""));
      if (d.kind) blank.kind = String(d.kind);
      if (ev.type === "compose.blank_started") {
        blank.state = "filling";
        return;
      }
      blank.state = "filled";
      blank.ok = d.ok !== false;
      blank.cached = d.cached === true;
      blank.attempts = Number(d.attempts) || 0;
      blank.cost = typeof d.cost_usd === "number" ? d.cost_usd : blank.cost;
      blank.count = problemCount(d);
      blank.problems = Array.isArray(d.problems) ? d.problems.map(String) : [];
      if (typeof d.markdown === "string") blank.markdown = d.markdown;
    });
    return Array.from(docs.values()).map((doc) => Object.assign(doc, { blanks: Array.from(doc.blanks.values()) }));
  }

  // Model output: only ever drawn by markdown.js (DOM nodes, never parsed as HTML) or as text.
  function textNode(text) {
    const box = AA.el("div", { class: "md doc-md" });
    if (AA.markdown) AA.markdown.update(box, text);
    else box.appendChild(AA.el("pre", { text: text }));
    return box;
  }

  function summary(doc) {
    const done = doc.blanks.filter((b) => b.state === "filled").length;
    const cost = doc.blanks.reduce((sum, b) => sum + (b.cost || 0), 0);
    const parts = [done + " of " + doc.blanks.length + " blanks filled", AA.fmt.usd(cost)];
    if (doc.revised || doc.rejected) parts.push("editor: " + doc.revised + " changed, " + doc.rejected + " rejected");
    return parts.join(" · ");
  }

  AA.createDocument = function (container) {
    const st = { key: null, seen: -1, docs: [], captured: false, showBefore: new Set() };

    function blankItem(doc, blank) {
      const id = doc.name + "\u0000" + blank.name;
      const status = statusOf(blank);
      const row = AA.el("div", { class: "doc-row" }, [
        AA.el("strong", { text: blank.name }),
        blank.kind ? AA.el("span", { class: "muted", text: blank.kind }) : null,
        AA.el("span", { class: "status " + status.cls, text: status.text }),
        blank.attempts > 1 ? AA.el("span", { class: "muted", text: blank.attempts + " attempts" }) : null,
        AA.el("span", { class: "spacer" }),
        blank.cost !== null ? AA.el("span", { class: "doc-cost", text: AA.fmt.usd(blank.cost) }) : null,
      ]);
      const item = AA.el("li", { class: "doc-blank" }, row);
      if (blank.problems.length) item.appendChild(AA.el("ul", { class: "doc-problems" }, blank.problems.map((p) => AA.el("li", { text: p }))));
      const before = blank.before !== null && st.showBefore.has(id);
      if (blank.before !== null) {
        const toggle = AA.el("button", { type: "button", class: "ghost small", "aria-pressed": before ? "true" : "false", text: before ? "show edited" : "show before edit" });
        toggle.addEventListener("click", () => {
          if (st.showBefore.has(id)) st.showBefore.delete(id);
          else st.showBefore.add(id);
          render();
        });
        item.appendChild(AA.el("div", { class: "doc-edit" }, [toggle, blank.reason ? AA.el("span", { class: "muted", text: "why: " + blank.reason }) : null]));
      }
      const text = before ? blank.before : blank.markdown;
      if (text !== null) item.appendChild(textNode(text));
      return item;
    }

    function render() {
      AA.clear(container);
      container.appendChild(AA.el("div", { class: "card-head" }, AA.el("h2", { text: "Document" })));
      st.docs.forEach((doc) => {
        container.appendChild(AA.el("div", { class: "doc-head" }, [
          AA.el("strong", { text: doc.name || "document" }),
          AA.el("span", { class: "muted", text: summary(doc) }),
        ]));
        container.appendChild(AA.el("ol", { class: "doc-blanks" }, doc.blanks.map((blank) => blankItem(doc, blank))));
      });
      if (!st.captured) container.appendChild(AA.el("p", { class: "muted", text: "Text not recorded: record with capture_content=True to see each blank." }));
    }

    return {
      update(run) {
        const events = (run && run.events) || [];
        const key = AA.runKey(run);
        let count = 0;
        events.forEach((ev) => { if (TYPES.indexOf(ev.type) >= 0) count += 1; });
        container.hidden = count === 0;
        if (key === st.key && count === st.seen) return; // nothing new for this card
        if (key !== st.key) st.showBefore.clear();
        st.key = key;
        st.seen = count;
        st.docs = documents(events);
        st.captured = ((run && run.meta) || []).some((m) => m && m.capture_content);
        if (count) render();
      },
    };
  };

  AA.composeDoc = { documents: documents, statusOf: statusOf };
})();
