// One chat turn: its model (folded from stream events or a stored turn) and its DOM cards.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const ICONS = { running: "▶", ok: "✓", done: "✓", failed: "✕", pending: "⏸", approved: "✓", denied: "✕", "timed-out": "⏱", stopped: "■" };
  const LABELS = { done: "done", failed: "failed", stopped: "stopped", "timed-out": "timed out", running: "running", lost: "connection lost" };

  function usageOf(turn) {
    return { input_tokens: turn.input_tokens || 0, output_tokens: turn.output_tokens || 0, cost_usd: turn.cost_usd || 0 };
  }

  // A fresh model: nothing streamed yet (a running turn replays its events from seq 0).
  function create(turn) {
    return { id: turn.id, turn: turn, status: turn.status, segments: [], tools: {}, approvals: {}, usage: usageOf(turn), error: turn.error || null, state: null };
  }

  // A stored turn: its tool calls, each approval after its call, then the answer.
  function fromStored(turn) {
    const model = create(turn);
    const placed = {};
    (turn.tools || []).forEach((tool) => {
      model.tools[tool.call_id] = tool;
      model.segments.push({ kind: "tool", id: tool.call_id });
      (turn.approvals || []).filter((a) => a.call_id === tool.call_id).forEach((a) => {
        model.approvals[a.approval_id] = a;
        placed[a.approval_id] = true;
        model.segments.push({ kind: "approval", id: a.approval_id });
      });
    });
    (turn.approvals || []).filter((a) => !placed[a.approval_id]).forEach((a) => {
      model.approvals[a.approval_id] = a;
      model.segments.push({ kind: "approval", id: a.approval_id });
    });
    if (turn.answer) model.segments.push({ kind: "text", text: turn.answer });
    return model;
  }

  // Folds one stream event into the model; returns true when something visible changed.
  function reduce(model, ev) {
    const last = model.segments[model.segments.length - 1];
    if (ev.type === "text") {
      if (!ev.text) return false;
      if (last && last.kind === "text") last.text += ev.text;
      else model.segments.push({ kind: "text", text: ev.text });
    } else if (ev.type === "tool") {
      if (!model.tools[ev.call_id]) model.segments.push({ kind: "tool", id: ev.call_id });
      model.tools[ev.call_id] = ev;
    } else if (ev.type === "approval") {
      if (!model.approvals[ev.approval_id]) model.segments.push({ kind: "approval", id: ev.approval_id });
      model.approvals[ev.approval_id] = ev;
    } else if (ev.type === "state") {
      model.state = ev.state;
    } else if (ev.type === "usage") {
      model.usage = { input_tokens: ev.input_tokens || 0, output_tokens: ev.output_tokens || 0, cost_usd: ev.cost_usd || 0 };
    } else if (ev.type === "done" || ev.type === "error") {
      model.status = ev.status;
      if (ev.turn) {
        model.turn = ev.turn;
        model.usage = usageOf(ev.turn);
      }
      model.error = ev.type === "error" ? { kind: ev.kind || "", message: ev.message || "" } : null;
    } else return false;
    return true;
  }

  // The answer text as the user saw it (all text segments).
  function answerText(model) {
    return model.segments.filter((s) => s.kind === "text").map((s) => s.text).join("");
  }

  function pretty(text) {
    if (typeof text !== "string") return AA.fmt.json(text);
    try {
      return JSON.stringify(JSON.parse(text), null, 2);
    } catch (err) {
      return text; // cut to 2 KB or plain text: shown as sent
    }
  }

  function duration(turn, now) {
    if (!turn.started) return "";
    const end = turn.ended || now / 1000;
    return AA.fmt.ms(Math.max(end - turn.started, 0) * 1000);
  }

  // ---- cards -----------------------------------------------------------------------------

  function toolCard(tool) {
    const head = AA.el("summary", null, [
      AA.el("span", { class: "tc-icon", "aria-hidden": "true" }),
      AA.el("code", { class: "tc-name" }),
      AA.el("span", { class: "status" }),
      AA.el("span", { class: "muted tc-ms" }),
    ]);
    const card = AA.el("details", { class: "tc" }, [head, AA.el("div", { class: "tc-body" })]);
    paintTool(card, tool);
    return card;
  }

  function paintTool(card, tool) {
    const status = tool.status || "running";
    card.className = "tc tc-" + status;
    const parts = card.firstChild.childNodes;
    parts[0].textContent = ICONS[status] || "•";
    parts[1].textContent = tool.name || "tool";
    parts[2].className = "status " + status;
    parts[2].textContent = status === "pending" ? "waiting for approval" : status;
    parts[3].textContent = [tool.ms !== null && tool.ms !== undefined ? AA.fmt.ms(tool.ms) : "", tool.cached ? "cached" : ""].filter(Boolean).join(" · ");
    const body = AA.clear(card.lastChild);
    if (tool.arguments !== undefined) body.appendChild(AA.el("div", { class: "tc-part" }, [AA.el("span", { class: "muted", text: "arguments" }), AA.el("pre", { text: pretty(tool.arguments) })]));
    if (tool.output !== undefined) body.appendChild(AA.el("div", { class: "tc-part" }, [AA.el("span", { class: "muted", text: "output" }), AA.el("pre", { text: pretty(tool.output) })]));
    if (tool.error_kind) body.appendChild(AA.el("p", { class: "notice bad", text: "error: " + tool.error_kind }));
    if (!body.firstChild) body.appendChild(AA.el("p", { class: "muted", text: "no arguments recorded yet" }));
  }

  // `onDecide(approval, approved, reason)` returns a Promise of the decided approval. One
  // decision per card: every control stays off unless the decision failed for another reason.
  function approvalCard(approval, onDecide) {
    const reason = AA.el("input", { type: "text", placeholder: "reason (optional)", "aria-label": "reason", maxlength: "500" });
    const notice = AA.el("p", { class: "notice", "aria-live": "polite" });
    const approve = AA.el("button", { type: "button", class: "approve", text: "Approve" });
    const deny = AA.el("button", { type: "button", class: "deny", text: "Deny" });
    const controls = [reason, approve, deny];
    const decide = (approved) => () => {
      controls.forEach((c) => { c.disabled = true; });
      notice.className = "notice";
      notice.textContent = "saving...";
      Promise.resolve(onDecide(card.approval, approved, reason.value)).then((decided) => {
        notice.textContent = "";
        if (decided) paintApproval(card, decided);
      }, (err) => {
        notice.className = "notice bad";
        notice.textContent = err.status === 409 ? "Already decided (" + err.message + ")." : "Not saved: " + err.message;
        if (err.status !== 409) controls.forEach((c) => { c.disabled = false; });
      });
    };
    approve.addEventListener("click", decide(true));
    deny.addEventListener("click", decide(false));
    const actions = AA.el("div", { class: "actions" }, controls);
    if (!supported("decide")) { // a recording decides by itself: both buttons inert, and it says so
      controls.forEach((c) => { c.disabled = true; });
      actions.appendChild(AA.el("span", { class: "muted", text: "Recorded demo: the recorded decision arrives by itself." }));
    }
    const card = AA.el("div", { class: "ac", role: "group" }, [
      AA.el("div", { class: "ac-head" }),
      AA.el("pre", { class: "ac-args" }),
      actions,
      AA.el("p", { class: "ac-result" }),
      notice,
    ]);
    paintApproval(card, approval);
    return card;
  }

  function paintApproval(card, approval) {
    card.approval = approval;
    const status = approval.status || "pending";
    card.className = "ac ac-" + status;
    card.setAttribute("aria-label", "approval for " + approval.tool);
    const head = AA.clear(card.childNodes[0]);
    head.appendChild(AA.el("span", { class: "tc-icon", "aria-hidden": "true", text: ICONS[status] || "?" }));
    head.appendChild(AA.el("strong", { text: status === "pending" ? "Approve this call?" : "Approval" }));
    head.appendChild(AA.el("code", { text: approval.tool || "" }));
    if (approval.hint) head.appendChild(AA.el("span", { class: "muted", text: approval.hint }));
    card.childNodes[1].textContent = pretty(approval.arguments || "");
    card.childNodes[2].hidden = status !== "pending";
    const result = card.childNodes[3];
    result.hidden = status === "pending";
    const who = approval.decided_by ? " by " + approval.decided_by : "";
    const why = approval.reason ? ": " + approval.reason : "";
    const text = { approved: "Approved" + who + why, denied: "Denied" + who + why, "timed-out": "Timed out: the call was denied", stopped: "The turn stopped: the call was denied" }[status];
    result.textContent = text || status;
    result.className = "ac-result " + (status === "approved" ? "good" : "bad");
  }

  // The footer, built once and updated in place (no restarted animation, no repeated alert).
  function createFoot() {
    const typing = AA.el("span", { class: "typing", "aria-label": "working" }, [AA.el("i"), AA.el("i"), AA.el("i")]);
    const status = AA.el("span", { class: "status" });
    const facts = AA.el("span", { class: "muted" });
    const error = AA.el("p", { class: "notice bad msg-error", role: "alert", hidden: true });
    const foot = AA.el("div", { class: "msg-foot" }, [typing, status, facts, error]);
    const set = (node, text) => { if (node.textContent !== text) node.textContent = text; };
    foot.paint = function (model, now) {
      const state = model.status;
      const usage = model.usage;
      const parts = [duration(model.turn, now)];
      if (usage.input_tokens || usage.output_tokens) parts.push(usage.input_tokens + " in / " + usage.output_tokens + " out tokens");
      if (usage.cost_usd) parts.push(AA.fmt.usd(usage.cost_usd));
      foot.className = "msg-foot " + state;
      typing.hidden = !(state === "running" && !model.segments.length);
      status.className = "status " + (state === "done" ? "ok" : state === "running" ? "running" : "failed");
      set(status, LABELS[state] || state);
      set(facts, parts.filter(Boolean).join(" · "));
      const message = model.error ? (model.error.kind ? model.error.kind + ": " : "") + (model.error.message || "the turn failed") : "";
      error.hidden = !message;
      set(error, message);
    };
    return foot;
  }

  // Whether the chat API offers a turn action (a recorded demo cannot replay, edit or save).
  function supported(action) {
    const api = AA.chatApi;
    return !api || typeof api.supports !== "function" || api.supports(action);
  }

  // The DOM of one turn; `h` = {onDecide, onAction(name, view, button), onEdit(view), onSelect(view), onFile(ref), copy}.
  function createView(model, h, files) {
    const userText = AA.el("div", { class: "msg-text" });
    const attach = AA.el("div", { class: "chips" });
    const edit = AA.el("button", { type: "button", class: "ghost small", text: "edit", title: "Edit and send again (a new branch)", on: { click: () => h.onEdit(view) } });
    const user = AA.el("div", { class: "msg user" }, [AA.el("div", { class: "msg-who" }, [AA.el("span", { text: "you" }), edit]), userText, attach]);
    const body = AA.el("div", { class: "msg-body" });
    const foot = createFoot();
    // A decision also updates the model, so the card stays decided if the stream is lost.
    const decide = (approval, approved, reason) => Promise.resolve(h.onDecide(approval, approved, reason)).then((decided) => {
      if (decided) model.approvals[decided.approval_id] = Object.assign({ type: "approval" }, decided);
      return decided;
    });
    const menu = AA.el("div", { class: "msg-menu", role: "toolbar", "aria-label": "turn actions" });
    const buttons = [];
    edit.hidden = !supported("edit");
    [["inspect", "Inspect"], ["replay", "Replay"], ["compare", "Compare"], ["eval", "Save as eval case"], ["copy", "Copy"]].filter((pair) => supported(pair[0])).forEach((pair) => {
      const button = AA.el("button", { type: "button", class: "ghost small", text: pair[1], on: { click: (ev) => { ev.stopPropagation(); h.onAction(pair[0], view, button); } } });
      button.action = pair[0];
      buttons.push(button);
      menu.appendChild(button);
    });
    const bot = AA.el("div", { class: "msg bot", tabindex: "-1", on: { click: (ev) => { if (!ev.target.closest || !ev.target.closest("button,a,input,summary,pre")) h.onSelect(view); } } }, [AA.el("div", { class: "msg-who" }, [AA.el("span", { text: "agent" }), menu]), body, foot]);
    const root = AA.el("article", { class: "turn", "aria-label": "turn" }, [user, bot]);
    const nodes = [];
    const view = { root: root, model: model, render: render, select: (on) => bot.classList.toggle("selected", on), focus: () => bot.focus() };

    function nodeFor(seg, i) {
      const old = nodes[i];
      if (seg.kind === "text") {
        if (old && old.kind === "text" && old.text === seg.text) return old;
        const node = old && old.kind === "text" ? old.node : AA.el("div", { class: "md" });
        AA.markdown.update(node, seg.text, { copy: h.copy });
        return { kind: "text", node: node, text: seg.text };
      }
      if (seg.kind === "tool") {
        const tool = model.tools[seg.id];
        if (old && old.kind === "tool" && old.id === seg.id) { if (old.data !== tool) paintTool(old.node, tool); return { kind: "tool", id: seg.id, node: old.node, data: tool }; }
        return { kind: "tool", id: seg.id, node: toolCard(tool), data: tool };
      }
      const approval = model.approvals[seg.id];
      if (old && old.kind === "approval" && old.id === seg.id) { if (old.data !== approval) paintApproval(old.node, approval); return { kind: "approval", id: seg.id, node: old.node, data: approval }; }
      return { kind: "approval", id: seg.id, node: approvalCard(approval, decide), data: approval };
    }

    function render(now) {
      const turn = model.turn;
      userText.textContent = turn.text || "";
      if (attach.dataset.ids !== (turn.attachments || []).join(",")) {
        attach.dataset.ids = (turn.attachments || []).join(",");
        AA.clear(attach);
        (turn.attachments || []).forEach((fid) => {
          const ref = (files && files[fid]) || { id: fid, name: fid, size: 0 };
          attach.appendChild(AA.el("button", { type: "button", class: "chip file", title: "download", text: ref.name, on: { click: () => h.onFile(ref) } }));
        });
      }
      model.segments.forEach((seg, i) => {
        const next = nodeFor(seg, i);
        if (nodes[i] && nodes[i].node !== next.node) body.replaceChild(next.node, nodes[i].node);
        else if (!nodes[i]) body.appendChild(next.node);
        nodes[i] = next;
      });
      while (nodes.length > model.segments.length) body.removeChild(nodes.pop().node);
      const running = model.status === "running";
      buttons.forEach((b) => { b.disabled = running && b.action !== "inspect"; });
      edit.disabled = running;
      root.setAttribute("aria-busy", running ? "true" : "false"); // screen readers wait for the whole answer
      foot.paint(model, now || Date.now());
    }

    render();
    return view;
  }

  AA.chatTurn = { create: create, fromStored: fromStored, reduce: reduce, answerText: answerText, pretty: pretty, createView: createView };
})();
