// Chat extras: state editor, attachments (button and drag-and-drop), compare, eval cases, copy.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const MAX_STATE = 64 * 1024;

  // The state the editor holds: {value} for a JSON object of at most 64 KB, else {error}. The
  // size is the request the server checks (compact JSON in {"state": ...}), not the pretty text.
  function parseState(text) {
    const source = String(text || "").trim() || "{}";
    let value;
    try {
      value = JSON.parse(source);
    } catch (err) {
      return { error: "not JSON: " + err.message };
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) return { error: "the state must be a JSON object, like {\"key\": \"value\"}" };
    if (new TextEncoder().encode(JSON.stringify({ state: value })).length > MAX_STATE) return { error: "the state may hold 64 KB of JSON" };
    return { value: value };
  }

  // An eval case's `expected`: JSON when it parses, else the text itself; "" means none.
  function parseExpected(text) {
    const source = String(text || "").trim();
    if (!source) return undefined;
    try {
      return JSON.parse(source);
    } catch (err) {
      return source;
    }
  }

  function parseTags(text) {
    return String(text || "").split(",").map((t) => t.trim()).filter(Boolean).slice(0, 20);
  }

  // A modal <dialog> with a title and body; closes on Esc (the browser's) or the close button.
  function dialog(title, body) {
    const box = AA.el("dialog", { class: "chat-dialog", "aria-label": title }, [
      AA.el("div", { class: "card-head" }, [AA.el("h2", { text: title }), AA.el("span", { class: "spacer" }),
        AA.el("button", { type: "button", class: "ghost small", text: "close", on: { click: () => box.close() } })]),
      body,
    ]);
    box.addEventListener("close", () => box.remove());
    document.body.appendChild(box);
    box.showModal();
    return box;
  }

  // Copies text; when the browser refuses the clipboard, shows it selected to copy by hand.
  async function copy(text, button) {
    try {
      await navigator.clipboard.writeText(text);
      if (button) {
        const before = button.textContent;
        button.textContent = "copied";
        setTimeout(() => { button.textContent = before; }, 1200);
      }
    } catch (err) {
      console.warn("Agent Adapter chat: the clipboard was refused; showing the text to copy", err);
      const area = AA.el("textarea", { class: "copy-area", readonly: true, rows: "12", "aria-label": "text to copy" });
      area.value = text;
      dialog("Copy with Ctrl+C", area);
      area.select();
    }
  }

  // The JSON state editor; `opts` = {owner(), state(), running(), save(state) -> Promise}. Text
  // being typed is never replaced by a refresh: `fill(false)` only fills a clean editor. The
  // text belongs to the chat it was filled for: Save never writes it into another chat.
  function createStateEditor(container, opts) {
    const owner = opts.owner || (() => "");
    let dirty = false;
    let filledFor = owner();
    // A draft typed before the first message belongs to the chat that message creates.
    const ours = () => filledFor === owner() || filledFor === "";
    const area = AA.el("textarea", { class: "state-area", rows: "8", spellcheck: "false", "aria-label": "session state (JSON object)" });
    const notice = AA.el("p", { class: "notice", "aria-live": "polite" });
    const save = AA.el("button", { type: "button", class: "approve", text: "Save state" });
    AA.clear(container);
    container.appendChild(AA.el("div", { class: "card-head" }, [AA.el("h2", { text: "Session state" }), AA.el("span", { class: "muted", text: "the agent reads it as session.state; secret-named values show as \"***\" and are kept" })]));
    container.appendChild(area);
    container.appendChild(AA.el("div", { class: "actions" }, [save, AA.el("button", { type: "button", class: "ghost small", text: "reset", on: { click: () => fill(true) } }), notice]));

    function check() {
      const found = parseState(area.value);
      notice.className = "notice" + (found.error ? " bad" : "");
      notice.textContent = found.error || (opts.running() ? "a turn is running: save after it ends" : "");
      save.disabled = Boolean(found.error) || opts.running();
      return found;
    }

    function fill(force) {
      if (dirty && force === false && ours()) return;
      dirty = false;
      filledFor = owner();
      area.value = JSON.stringify(opts.state() || {}, null, 2);
      check();
    }

    area.addEventListener("input", () => { dirty = true; check(); });
    save.addEventListener("click", async () => {
      const found = check();
      if (found.error) return;
      if (!ours()) { // another chat is open now: its state is shown, nothing is saved
        fill(true);
        notice.className = "notice bad";
        notice.textContent = "Another chat is open: its state is shown again; nothing was saved.";
        return;
      }
      save.disabled = true;
      notice.className = "notice";
      notice.textContent = "saving...";
      try {
        const saved = await opts.save(found.value);
        area.value = JSON.stringify(saved || {}, null, 2);
        dirty = false;
        filledFor = owner();
        notice.className = "notice good";
        notice.textContent = "Saved: the next turn starts with this state.";
      } catch (err) {
        notice.className = "notice bad";
        notice.textContent = "Not saved: " + err.message;
      } finally {
        save.disabled = Boolean(parseState(area.value).error) || opts.running();
      }
    });
    return { fill: fill, check: check, focus: () => area.focus(), dirty: () => dirty };
  }

  // Attach by button or by dropping files on `zone`; `opts` = {session() -> Promise<sid>,
  // used() -> bytes already stored, mode, onAdded(ref), onError(text), onBusy(bool)}.
  function attachUploads(zone, input, button, hint, opts) {
    async function send(files) {
      const list = Array.prototype.slice.call(files || []);
      if (!list.length) return;
      let used = opts.used();
      const ok = [];
      for (const file of list) {
        const problem = AA.chatApi.uploadProblem(file, used, opts.mode);
        if (problem) { opts.onError(problem); continue; }
        used += file.size;
        ok.push(file);
      }
      if (!ok.length) return;
      opts.onBusy(true);
      try {
        const sid = await opts.session();
        for (const file of ok) {
          try {
            opts.onAdded(await AA.chatApi.upload(sid, file));
          } catch (err) {
            console.error("Agent Adapter chat: uploading " + file.name + " failed", err);
            opts.onError(file.name + ": " + err.message);
          }
        }
      } catch (err) {
        opts.onError("Could not start a chat for the files: " + err.message);
      } finally {
        opts.onBusy(false);
      }
    }
    button.addEventListener("click", () => input.click());
    input.addEventListener("change", () => { send(input.files); input.value = ""; });
    let depth = 0;
    const hasFiles = (ev) => ev.dataTransfer && Array.prototype.indexOf.call(ev.dataTransfer.types || [], "Files") >= 0;
    zone.addEventListener("dragenter", (ev) => { if (!hasFiles(ev)) return; ev.preventDefault(); depth += 1; hint.hidden = false; });
    zone.addEventListener("dragover", (ev) => { if (hasFiles(ev)) { ev.preventDefault(); ev.dataTransfer.dropEffect = "copy"; } });
    zone.addEventListener("dragleave", () => { depth = Math.max(depth - 1, 0); if (!depth) hint.hidden = true; });
    zone.addEventListener("drop", (ev) => {
      if (!hasFiles(ev)) return;
      ev.preventDefault();
      depth = 0;
      hint.hidden = true;
      send(ev.dataTransfer.files);
    });
    return { send: send };
  }

  // Save a turn as an eval case (appended to eval_cases.jsonl in the chat folder).
  function evalCaseDialog(turn, answer) {
    const id = AA.el("input", { type: "text", maxlength: "200", required: true, "aria-label": "case id" });
    id.value = "chat-" + turn.id.slice(-8);
    const expected = AA.el("textarea", { rows: "6", "aria-label": "expected answer (JSON or text)" });
    expected.value = answer || "";
    const tags = AA.el("input", { type: "text", placeholder: "tag, another", "aria-label": "tags" });
    const notice = AA.el("p", { class: "notice", "aria-live": "polite" });
    const save = AA.el("button", { type: "button", class: "approve", text: "Save case" });
    const body = AA.el("div", { class: "form" }, [
      AA.el("label", null, ["case id", id]),
      AA.el("label", null, ["expected (JSON, or text; empty for none)", expected]),
      AA.el("label", null, ["tags (comma separated)", tags]),
      AA.el("p", { class: "muted", text: "The input is this turn's message. Load the file with optimiser.cases.load_cases." }),
      AA.el("div", { class: "actions" }, [save, notice]),
    ]);
    dialog("Save as eval case", body);
    save.addEventListener("click", async () => {
      const name = id.value.trim();
      if (!name) { notice.className = "notice bad"; notice.textContent = "give the case an id"; return; }
      const payload = { id: name, tags: parseTags(tags.value) };
      const value = parseExpected(expected.value);
      if (value !== undefined) payload.expected = value;
      save.disabled = true;
      try {
        const done = await AA.chatApi.evalCase(turn.id, payload);
        notice.className = "notice good";
        notice.textContent = "Saved " + done.case.id + " to " + done.file + ".";
      } catch (err) {
        notice.className = "notice bad";
        notice.textContent = "Not saved: " + err.message;
        save.disabled = false;
      }
    });
    id.focus();
  }

  // Compare one turn with another: `others` = [{label, turn, answer}].
  function compareDialog(base, others) {
    const picker = AA.el("select", { "aria-label": "turn to compare with" });
    const out = AA.el("div", { class: "cmp-out" });
    const api = AA.chatApi;
    const canReplay = !api || typeof api.supports !== "function" || api.supports("replay");
    const none = canReplay ? "no other turn yet: replay this one first" : "no other finished turn in this chat to compare with";
    picker.appendChild(AA.el("option", { value: "", text: others.length ? "pick a turn" : none }));
    picker.disabled = !others.length;
    others.forEach((o, i) => picker.appendChild(AA.el("option", { value: String(i), text: o.label })));
    dialog("Compare turns", AA.el("div", null, [AA.el("div", { class: "ev-filters" }, [AA.el("span", { text: base.label + " with" }), picker]), out]));
    let seq = 0;
    picker.addEventListener("change", async () => {
      const other = others[Number(picker.value)];
      const mine = (seq += 1);
      AA.clear(out);
      if (!picker.value || !other) return;
      out.appendChild(AA.el("p", { class: "muted", text: "loading..." }));
      try {
        const runs = await Promise.all([AA.runsView.loadAll(base.turn.run_id), AA.runsView.loadAll(other.turn.run_id)]);
        if (mine !== seq) return;
        const result = AA.compareRuns.compare(AA.compareRuns.facts(runs[0], base.answer), AA.compareRuns.facts(runs[1], other.answer));
        AA.renderComparison(out, result, [base.label, other.label]);
      } catch (err) {
        if (mine !== seq) return;
        console.error("Agent Adapter chat: loading runs to compare failed", err);
        AA.clear(out).appendChild(AA.el("p", { class: "notice bad", text: err.status === 404 ? "A run is not recorded yet; try again in a moment." : "Could not load the runs: " + err.message }));
      }
    });
    if (others.length === 1) { picker.value = "0"; picker.dispatchEvent(new Event("change")); }
  }

  // Saves a stored file through a temporary blob link.
  async function download(ref, onError) {
    try {
      const blob = await AA.chatApi.fileBlob(ref.id);
      const url = URL.createObjectURL(blob);
      const link = AA.el("a", { href: url, download: ref.name || "file" });
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
    } catch (err) {
      console.error("Agent Adapter chat: downloading " + ref.id + " failed", err);
      onError("Could not download " + (ref.name || ref.id) + ": " + err.message);
    }
  }

  AA.chatExtras = {
    parseState: parseState, parseExpected: parseExpected, parseTags: parseTags, copy: copy, dialog: dialog,
    createStateEditor: createStateEditor, attachUploads: attachUploads, evalCaseDialog: evalCaseDialog,
    compareDialog: compareDialog, download: download,
  };
})();
