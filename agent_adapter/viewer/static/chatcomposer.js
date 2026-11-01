// Chat composer: the message box and its keys, attachments, edit mode, Send/Stop and the state editor.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const $ = (id) => document.getElementById(id);

  function bytes(size) {
    return size < 1024 ? size + " B" : size < 1048576 ? Math.round(size / 1024) + " KB" : (size / 1048576).toFixed(1) + " MB";
  }

  // hooks: {send(body, files) -> Promise, stop(), running() -> view|null, session(title) -> Promise<sid>,
  // owner() -> the open chat's id ("" before one exists), stored(id) -> bool, storedBytes(),
  // state(), saveState(state) -> Promise<state>, setError(text)}.
  AA.createComposer = function (hooks) {
    const input = $("chat-input");
    const sendButton = $("chat-send");
    const stopButton = $("chat-stop");
    const st = { pending: [], editing: null, sending: false };

    function renderPending() {
      const box = AA.clear($("chat-attachments"));
      st.pending.forEach((ref) => {
        box.appendChild(AA.el("span", { class: "chip file" }, [ref.name + " · " + bytes(ref.size) + " ",
          AA.el("button", { type: "button", class: "chip-x", "aria-label": "remove " + ref.name, text: "×", on: { click: () => { st.pending = st.pending.filter((p) => p !== ref); renderPending(); } } })]));
      });
    }

    function renderEditing() {
      const note = AA.clear($("chat-editing"));
      note.hidden = !st.editing;
      if (!st.editing) return;
      const files = (st.editing.attachments || []).length;
      note.appendChild(AA.el("span", { text: "Editing an earlier message: sending starts a new branch from before it" + (files ? ", with its files attached again. " : ". ") }));
      note.appendChild(AA.el("button", { type: "button", class: "ghost small", text: "cancel", on: { click: cancelEdit } }));
    }

    // Cancelling an edit removes only the files the edit brought back.
    function cancelEdit() {
      st.editing = null;
      st.pending = st.pending.filter((ref) => !ref.fromEdit);
      input.value = "";
      renderEditing();
      renderPending();
    }

    // Shows Stop while a turn runs, Send otherwise.
    function renderButtons() {
      const busy = hooks.running();
      stopButton.hidden = !busy;
      sendButton.hidden = Boolean(busy);
      stopButton.disabled = Boolean(busy && busy.stopping);
      stopButton.textContent = busy && busy.stopping ? "Stopping..." : "Stop";
    }

    async function send() {
      const text = input.value.trim();
      if ((!text && !st.pending.length) || hooks.running() || st.sending) return;
      st.sending = true;
      sendButton.disabled = true;
      hooks.setError("");
      const body = { text: text };
      if (st.pending.length) body.attachment_ids = st.pending.map((p) => p.attachment_id);
      if (st.editing) body.edit_of = st.editing.id;
      try {
        await hooks.send(body, st.pending.slice());
        input.value = "";
        st.pending = [];
        st.editing = null;
        renderPending();
        renderEditing();
      } catch (err) {
        if (err.status !== 401) hooks.setError("Not sent: " + err.message); // a 401 already locked the page with its own message
      } finally {
        st.sending = false;
        sendButton.disabled = false;
        renderButtons();
      }
    }

    // Typed into a field other than the message box (a reason, a dialog): Esc belongs to it.
    function ownEsc(target) {
      return Boolean(target && target !== input && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName || ""));
    }

    $("chat-composer").addEventListener("submit", (ev) => { ev.preventDefault(); send(); });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) { ev.preventDefault(); send(); }
    });
    stopButton.addEventListener("click", () => hooks.stop());
    document.addEventListener("keydown", (ev) => { // Esc stops the running turn, or cancels an edit
      if (ev.key !== "Escape" || ev.defaultPrevented || $("chat-view").hidden || document.querySelector("dialog[open]")) return;
      if (ownEsc(ev.target) || (ev.target && ev.target.closest && ev.target.closest("#chat-inspector"))) return;
      if (hooks.running()) { ev.preventDefault(); hooks.stop(); } else if (st.editing) { ev.preventDefault(); cancelEdit(); }
    });
    AA.chatExtras.attachUploads($("chat-main"), $("chat-file"), $("chat-attach"), $("chat-drop"), {
      mode: AA.chatMode,
      session: () => hooks.session(input.value),
      // Files already stored in this chat (brought back by an edit) count once.
      used: () => hooks.storedBytes() + st.pending.filter((f) => !hooks.stored(f.attachment_id)).reduce((n, f) => n + (f.size || 0), 0),
      onAdded: (ref) => { st.pending.push(ref); renderPending(); hooks.setError(""); },
      onError: hooks.setError,
      onBusy: (on) => { $("chat-attach").disabled = on; $("chat-attach").textContent = on ? "Uploading..." : "Attach"; },
    });
    const stateEditor = AA.chatExtras.createStateEditor($("chat-state"), {
      owner: hooks.owner,
      state: hooks.state,
      running: () => Boolean(hooks.running()),
      save: hooks.saveState,
    });
    $("chat-state-toggle").addEventListener("click", () => {
      const box = $("chat-state");
      box.hidden = !box.hidden;
      $("chat-state-toggle").setAttribute("aria-expanded", box.hidden ? "false" : "true");
      if (!box.hidden) { stateEditor.fill(false); stateEditor.focus(); }
    });

    return {
      send: send,
      renderButtons: renderButtons,
      // Edit an earlier turn: its text and files go back into the composer, next to the files
      // already attached (each once).
      edit(turn, files) {
        st.editing = turn;
        const kept = st.pending.filter((ref) => !ref.fromEdit);
        const ids = new Set(kept.map((ref) => ref.attachment_id));
        st.pending = kept.concat(files.filter((ref) => !ids.has(ref.attachment_id)).map((ref) => Object.assign({ fromEdit: true }, ref)));
        input.value = turn.text || "";
        renderEditing();
        renderPending();
        input.focus();
      },
      editing: () => st.editing,
      // A chat was opened or started: drafts of another chat's files and edits are dropped.
      // The state draft belongs to its chat too: it is replaced by the new chat's state.
      reset(keepFiles) {
        st.editing = null;
        if (!keepFiles) st.pending = [];
        renderEditing();
        renderPending();
        stateEditor.fill(true);
      },
      refreshState() {
        if (!$("chat-state").hidden) stateEditor.fill(false);
        stateEditor.check();
      },
      focus: () => input.focus(),
    };
  };
})();
