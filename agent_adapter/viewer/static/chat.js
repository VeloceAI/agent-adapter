// Chat view: opens chats, streams their turns into the conversation and wires the inspector.
"use strict";

(function () {
  const AA = window.AA;
  if (!AA.chatMode || !document.getElementById("chat-log")) return; // no chat: the page stays the Runs view
  const SESSIONS_MS = 5000;
  const RECONNECTING = "Connection dropped; reconnecting...";
  const $ = (id) => document.getElementById(id);
  const st = { info: null, sessions: [], sid: "", detail: null, views: {}, order: [], selected: "", streams: {}, state: {}, locked: false, busy: false, frame: 0, dirty: {} };
  const log = $("chat-log");
  let inspector = null;
  let composer = null;
  let list = null;

  function setError(text) {
    $("chat-error").textContent = text || "";
  }

  function running() {
    return st.order.map((id) => st.views[id]).find((v) => v.model.status === "running") || null;
  }

  function filesById() {
    const out = {};
    ((st.detail && st.detail.files) || []).forEach((f) => { out[f.id] = f; });
    return out;
  }

  // ---- sessions --------------------------------------------------------------------------

  async function refreshSessions() {
    if (st.locked || st.busy) return;
    st.busy = true;
    try {
      st.sessions = await AA.chatApi.sessions();
      list.update(st.sessions, st.sid);
    } catch (err) {
      failed(err, "listing chats failed");
    } finally {
      st.busy = false;
    }
  }

  function renderHead() {
    const agent = st.info ? st.info.agent_name || st.info.mode : AA.chatMode;
    AA.renderChatHead($("chat-head"), st.detail, st.sessions, agent, openSession);
  }

  function closeStreams() {
    Object.keys(st.streams).forEach((id) => st.streams[id].close());
    st.streams = {};
  }

  function clearLog() {
    closeStreams();
    st.views = {};
    st.order = [];
    AA.clear(log);
  }

  async function openSession(sid, keepFiles) {
    if (!sid) return;
    try {
      const detail = await AA.chatApi.session(sid);
      clearLog();
      composer.reset(keepFiles && sid === st.sid); // files and edits belong to their chat
      st.sid = sid;
      st.detail = detail;
      st.state = detail.state || {};
      detail.turns.forEach((turn) => addTurn(turn, turn.status === "running"));
      if (!detail.turns.length) log.appendChild(emptyNote());
      select(st.order[st.order.length - 1] || "");
      renderHead();
      list.update(st.sessions, st.sid);
      composer.refreshState();
      composer.renderButtons();
      log.scrollTop = log.scrollHeight;
      setError("");
      composer.focus();
    } catch (err) {
      failed(err, "opening the chat failed");
      setError("Could not open the chat: " + err.message);
    }
  }

  function newChat() {
    clearLog();
    composer.reset(false);
    st.sid = "";
    st.detail = null;
    st.state = {};
    log.appendChild(emptyNote());
    select("");
    renderHead();
    list.update(st.sessions, st.sid);
    composer.renderButtons();
    composer.focus();
  }

  function emptyNote() {
    return AA.el("div", { class: "chat-empty muted" }, [
      AA.el("p", { text: "Ask the agent something. Its tool calls, approvals and cost show here as it works." }),
      $("chat-composer").hidden ? null : AA.el("p", { text: "Enter sends · Shift+Enter adds a line · Esc stops a turn" }),
    ]);
  }

  async function ensureSession(titleFrom) {
    if (st.sid) return st.sid;
    const title = String(titleFrom || "").replace(/\s+/g, " ").trim().slice(0, 60);
    const detail = await AA.chatApi.createSession(title, st.state && Object.keys(st.state).length ? st.state : null);
    st.sid = detail.session.id;
    st.detail = detail;
    st.state = detail.state || {};
    st.sessions = [detail.session].concat(st.sessions.filter((s) => s.id !== st.sid));
    list.update(st.sessions, st.sid);
    renderHead();
    return st.sid;
  }

  // ---- turns -----------------------------------------------------------------------------

  const handlers = {
    copy: AA.chatExtras.copy,
    onFile: (ref) => AA.chatExtras.download(ref, setError),
    onSelect: (view) => select(view.model.id),
    onEdit: (view) => {
      const files = filesById();
      const refs = (view.model.turn.attachments || []).filter((id) => files[id]).map((id) => ({ attachment_id: id, name: files[id].name, size: files[id].size, media_type: files[id].media_type }));
      composer.edit(view.model.turn, refs);
    },
    onDecide: async (approval, approved, reason) => {
      try {
        return (await AA.chatApi.decide(approval.approval_id, approved, reason)).approval;
      } catch (err) {
        failed(err, err.status === 409 ? null : "deciding " + approval.approval_id + " failed");
        throw err;
      }
    },
    onAction: (name, view, button) => action(name, view, button),
  };

  function addTurn(turn, live) {
    const empty = log.querySelector(".chat-empty");
    if (empty) empty.remove();
    const model = live ? AA.chatTurn.create(turn) : AA.chatTurn.fromStored(turn);
    const view = AA.chatTurn.createView(model, handlers, filesById());
    st.views[turn.id] = view;
    st.order.push(turn.id);
    log.appendChild(view.root);
    if (live) follow(turn.id, 0);
    return view;
  }

  // Streams a running turn; every event updates its model, drawn once per animation frame.
  function follow(turnId, since) {
    if (st.streams[turnId]) st.streams[turnId].close();
    const sid = st.sid;
    st.streams[turnId] = AA.chatApi.streamTurn(turnId, since, {
      onEvent: (ev) => {
        const view = st.views[turnId];
        if (!view || sid !== st.sid) return;
        if (!AA.chatTurn.reduce(view.model, ev)) return;
        if (ev.type === "state") st.state = ev.state || {};
        draw(turnId);
        if (ev.type === "done" || ev.type === "error") ended(turnId);
        else if ((ev.type === "state" || ev.type === "usage" || ev.type === "memory") && turnId === st.selected) inspect();
      },
      onStatus: (status) => {
        if (status === "retrying") setError(RECONNECTING);
        else if ($("chat-error").textContent === RECONNECTING) setError(""); // other messages stay
      },
      onFatal: (err) => {
        delete st.streams[turnId];
        failed(err, "the turn's stream stopped");
        if (err.status !== 401 && sid === st.sid) settle(turnId, err);
      },
    });
  }

  // The live stream is gone: show the turn as the server stored it, or as lost, never "running".
  async function settle(turnId, err) {
    const view = st.views[turnId];
    let stored = null;
    try {
      stored = (await AA.chatApi.session(st.sid)).turns.find((t) => t.id === turnId) || null;
    } catch (again) {
      console.error("Agent Adapter chat: reloading turn " + turnId + " failed", again);
    }
    if (!view || st.views[turnId] !== view) return;
    if (stored && stored.status !== "running") Object.assign(view.model, AA.chatTurn.fromStored(stored));
    else {
      view.model.status = "lost";
      view.model.error = { kind: "stream-lost", message: "the live view of this turn was lost (" + err.message + "); open the chat again to see its result" };
    }
    view.render();
    ended(turnId);
  }

  function ended(turnId) {
    delete st.streams[turnId];
    if (turnId === st.selected) inspect();
    composer.renderButtons();
    composer.refreshState();
    refreshSessions();
  }

  function draw(turnId) {
    st.dirty[turnId] = true;
    if (st.frame) return;
    st.frame = window.requestAnimationFrame(() => {
      st.frame = 0;
      const stick = log.scrollHeight - log.scrollTop - log.clientHeight < 80;
      Object.keys(st.dirty).forEach((id) => { if (st.views[id]) st.views[id].render(); });
      st.dirty = {};
      composer.renderButtons();
      if (stick) log.scrollTop = log.scrollHeight;
    });
  }

  function select(turnId) {
    st.selected = turnId;
    st.order.forEach((id) => st.views[id].select(id === turnId));
    inspect();
  }

  // The inspector gets the turn with its live status and running usage totals.
  function inspect() {
    const view = st.views[st.selected];
    if (!view) { inspector.show(null, null); return; }
    const model = view.model;
    const memory = { live_memories: model.memories || null, memory_result: model.memorySaved || null };
    inspector.show(Object.assign({}, model.turn, model.usage, { status: model.status }, memory), model.state);
  }

  async function action(name, view, button) {
    const model = view.model;
    try {
      if (name === "inspect") select(model.id);
      else if (name === "copy") await AA.chatExtras.copy(AA.chatTurn.answerText(model), button);
      else if (name === "eval") AA.chatExtras.evalCaseDialog(model.turn, AA.chatTurn.answerText(model));
      else if (name === "replay") {
        const started = await AA.chatApi.replay(model.id);
        await refreshSessions();
        await openSession(started.session_id);
      } else if (name === "compare") {
        const base = { label: "turn " + (st.order.indexOf(model.id) + 1), turn: model.turn, answer: AA.chatTurn.answerText(model) };
        AA.chatExtras.compareDialog(base, await compareTargets(model));
      }
    } catch (err) {
      failed(err, name + " failed");
      setError(name + " failed: " + err.message);
    }
  }

  // Turns to compare with: this chat's, its parent's and its branches' (finished ones).
  async function compareTargets(model) {
    const out = [];
    const add = (turn, name) => {
      if (turn.id !== model.id && turn.status !== "running") out.push({ label: name, turn: turn, answer: turn.answer || "" });
    };
    st.order.forEach((id, i) => add(st.views[id].model.turn, "turn " + (i + 1)));
    const session = st.detail.session;
    const others = (st.detail.branches || []).map((b) => b.id);
    if (session.parent) others.unshift(session.parent.session_id);
    const details = await Promise.all(others.map((id) => AA.chatApi.session(id).catch((err) => { console.warn("Agent Adapter chat: chat " + id + " not loaded", err); return null; })));
    details.filter(Boolean).forEach((d) => d.turns.forEach((turn, i) => add(turn, d.session.title + " · turn " + (i + 1))));
    return out;
  }

  // ---- sending ---------------------------------------------------------------------------

  async function send(body, files) {
    const sid = body.edit_of ? st.sid : await ensureSession(body.text || (files[0] && files[0].name));
    let started;
    try {
      started = await AA.chatApi.startTurn(sid, body);
    } catch (err) {
      failed(err, "sending failed");
      throw err;
    }
    if (st.detail) st.detail.files = (st.detail.files || []).concat(files.filter((f) => !filesById()[f.attachment_id]).map((p) => ({ id: p.attachment_id, name: p.name, media_type: p.media_type, size: p.size })));
    if (body.edit_of || started.session_id !== st.sid) {
      await refreshSessions();
      await openSession(started.session_id);
      return;
    }
    const now = Date.now() / 1000;
    addTurn({ id: started.turn_id, session_id: sid, run_id: started.run_id, text: body.text, attachments: body.attachment_ids || [], status: "running", answer: "", error: null, started: now, ended: null, input_tokens: 0, output_tokens: 0, cost_usd: 0, tools: [], approvals: [] }, true);
    select(started.turn_id);
    log.scrollTop = log.scrollHeight;
    refreshSessions();
  }

  async function stop() {
    const view = running();
    if (!view || view.stopping) return;
    view.stopping = true;
    composer.renderButtons();
    try {
      await AA.chatApi.stop(view.model.id);
    } catch (err) {
      view.stopping = false;
      if (err.status !== 409) { failed(err, "stopping failed"); setError("Not stopped: " + err.message); } // 409: it ended meanwhile
      composer.renderButtons();
    }
  }

  // ---- start -----------------------------------------------------------------------------

  function failed(err, what) {
    if (err && err.status === 401) {
      st.locked = true;
      closeStreams();
      setError("The viewer refused the access token (it may have restarted): open the address it printed.");
      return;
    }
    if (what) console.error("Agent Adapter chat: " + what, err);
  }

  async function start() {
    inspector = AA.createInspector($("chat-inspector"), (key) => { AA.views.show("runs"); AA.runsView.pick(key); }, () => !$("chat-view").hidden);
    list = AA.createSessionList($("chat-session-list"), (sid) => openSession(sid));
    composer = AA.createComposer({
      send: send,
      stop: stop,
      running: running,
      session: ensureSession,
      owner: () => st.sid,
      stored: (id) => Boolean(filesById()[id]),
      storedBytes: () => ((st.detail && st.detail.files) || []).reduce((n, f) => n + (f.size || 0), 0),
      state: () => st.state,
      saveState: async (state) => {
        const saved = await AA.chatApi.setState(await ensureSession(""), state);
        st.state = saved.state || {};
        return st.state;
      },
      setError: setError,
    });
    $("chat-new").addEventListener("click", newChat);
    st.locked = !AA.chatApi.recorded && !AA.hasToken(); // a recorded demo has no server to ask
    AA.views.register("chat", { onShow: () => { refreshSessions(); composer.focus(); } });
    if (!/(?:^#|&)run=/.test(window.location.hash || "")) AA.views.show("chat");
    newChat();
    if (st.locked) {
      setError("No access token: open the full address the viewer printed (it ends in #token=...).");
      return;
    }
    try {
      st.info = await AA.chatApi.info();
    } catch (err) {
      failed(err, "reading the chat mode failed");
    }
    if (st.info && !st.info.enabled) setError("Chat is off on this server.");
    renderHead();
    await refreshSessions();
    if (st.sessions.length) await openSession(st.sessions[0].id);
    setInterval(() => { if (!$("chat-view").hidden) refreshSessions(); }, SESSIONS_MS);
    setInterval(() => { const view = running(); if (view) draw(view.model.id); }, 1000); // the running turn's clock; keeps the log following
  }

  // Open a chat, list chats again, or send `text` as if typed: for the recorded demo's questions.
  function ask(text) {
    $("chat-input").value = text;
    return composer.send();
  }

  AA.chatView = { open: openSession, ask: ask, refresh: refreshSessions, sid: () => st.sid };
  AA.chatView.ready = start(); // tests wait on it
})();
