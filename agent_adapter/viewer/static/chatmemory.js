// Chat inspector Memory tab: what a turn recalled, and the user's records with Delete.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const OFF_HELP = "Memory is off. Start the viewer with --memory sqlite:memory.db --user NAME to recall across chats.";

  // A record's line: kind, when, and the text (always as text, never as HTML).
  function recordRow(rec, extra) {
    const facts = [rec.kind ? AA.el("span", { class: "chip", text: rec.kind }) : null];
    if (typeof rec.score === "number") facts.push(AA.el("span", { class: "chip", text: "score " + rec.score.toFixed(2) }));
    if (rec.created_at) facts.push(AA.el("span", { class: "muted", text: AA.fmt.time(rec.created_at) }));
    return AA.el("li", { class: "mem-row" }, [
      AA.el("div", { class: "mem-facts" }, facts.concat(extra || [])),
      AA.el("p", { class: "mem-text", text: rec.text === undefined ? "(deleted or expired)" : rec.text }),
    ]);
  }

  // What the turn recalled: the live texts, else the stored ids matched to the user's records.
  function recalledOf(turn, byId) {
    if (!turn) return [];
    if (Array.isArray(turn.live_memories)) return turn.live_memories;
    return (turn.memories || []).map((ref) => Object.assign({}, byId[ref.id] || { id: ref.id }, { score: ref.score }));
  }

  AA.createMemoryPanel = function (container) {
    const st = { info: null, records: [], byId: {}, loaded: false, busy: false, error: "", notice: "", turn: null };
    const api = () => AA.chatApi || {};

    async function load() {
      if (st.busy || typeof api().memory !== "function") return;
      st.busy = true;
      try {
        const found = await api().memory();
        st.info = found.memory || { enabled: false };
        st.records = found.records || [];
        st.byId = {};
        st.records.forEach((rec) => { st.byId[rec.id] = rec; });
        st.error = "";
      } catch (err) {
        console.error("Agent Adapter chat: loading memory failed", err);
        st.error = "Could not load the memory: " + err.message;
      } finally {
        st.busy = false;
        st.loaded = true;
      }
      render();
    }

    async function forget(rec, button) {
      button.disabled = true;
      try {
        const done = await api().forget([rec.id]);
        st.notice = done.deleted ? "Deleted." : "Already gone.";
      } catch (err) {
        button.disabled = false;
        st.notice = "Not deleted: " + err.message;
        render();
        return;
      }
      await load();
    }

    function render() {
      AA.clear(container);
      if (!st.loaded) { container.appendChild(AA.el("p", { class: "muted", text: "Loading memory..." })); return; }
      if (st.error) container.appendChild(AA.el("p", { class: "notice bad", text: st.error }));
      const info = st.info || { enabled: false };
      if (!info.enabled) { container.appendChild(AA.el("p", { class: "muted", text: OFF_HELP })); return; }
      container.appendChild(AA.el("h3", { text: "Recalled for this turn" }));
      const recalled = recalledOf(st.turn, st.byId);
      if (!st.turn) container.appendChild(AA.el("p", { class: "muted", text: "Pick a turn to see what it recalled." }));
      else if (!recalled.length) container.appendChild(AA.el("p", { class: "muted", text: "Nothing from earlier chats was recalled." }));
      else container.appendChild(AA.el("ul", { class: "mem-list" }, recalled.map((rec) => recordRow(rec))));
      const result = st.turn && st.turn.memory_result;
      const saved = result ? result.saved : st.turn && st.turn.memory_saved;
      if (result && result.error) container.appendChild(AA.el("p", { class: "notice bad", text: "Saving memories failed (" + result.error + ")." }));
      else if (saved) container.appendChild(AA.el("p", { class: "muted", text: "Saved " + saved + " record(s) after this turn." }));
      container.appendChild(AA.el("div", { class: "card-head" }, [
        AA.el("h3", { text: "Memory of " + (info.user || "this user") }),
        AA.el("span", { class: "muted", text: (info.backend || "") + " · save: " + (info.save || "none") }),
        AA.el("span", { class: "spacer" }),
        AA.el("button", { type: "button", class: "ghost small", text: "Refresh", on: { click: load } }),
      ]));
      if (st.notice) container.appendChild(AA.el("p", { class: "notice", "aria-live": "polite", text: st.notice }));
      if (!st.records.length) { container.appendChild(AA.el("p", { class: "muted", text: "No records yet." })); return; }
      container.appendChild(AA.el("ul", { class: "mem-list" }, st.records.map((rec) => {
        const button = AA.el("button", { type: "button", class: "deny small", text: "Delete", "aria-label": "delete this memory" });
        button.addEventListener("click", () => forget(rec, button));
        return recordRow(rec, [AA.el("span", { class: "spacer" }), button]);
      })));
    }

    return {
      // Called when the Memory tab is shown or the selected turn changes.
      show(turn) {
        st.turn = turn || null;
        if (!st.loaded) { render(); load(); return; }
        render();
      },
      reload: load,
      state: () => st,
    };
  };

  AA.chatMemory = { recalledOf: recalledOf };
})();
