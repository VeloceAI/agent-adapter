// Recorded chat demo (docs): stands in for the chat API and replays recorded turns on timers.
"use strict";

(function () {
  const AA = window.AA;
  const demo = AA.embedded && AA.embedded.chat;
  if (!demo || !Array.isArray(demo.chats) || !AA.chatApi) return; // a plain export: no chat
  const $ = (id) => document.getElementById(id);
  const OFF = ["replay", "eval", "edit", "attach", "state", "decide"]; // what a recording cannot do
  const env = {
    setTimeout: (fn, ms) => window.setTimeout(fn, ms),
    clearTimeout: (id) => window.clearTimeout(id),
    now: () => Date.now(),
    reducedMotion: () => Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches),
  };
  const details = AA.embedded.details || {};
  const chats = {};
  const turns = {};
  const order = [];
  demo.chats.forEach((chat) => {
    chats[chat.session.id] = chat;
    chat.turns.forEach((rec, index) => {
      turns[rec.turn.id] = { chat: chat, index: index, rec: rec };
      order.push(rec.turn.id);
    });
  });
  // played: turn ids shown as finished; running: chat id -> turn id; shown: run id -> events
  // shown; reopening: the playing turn whose chat the page is opening again.
  const st = { played: {}, running: {}, shown: {}, player: null, reopening: "", buttons: {} };
  const copy = (value) => JSON.parse(JSON.stringify(value));
  const has = (table, key) => Object.prototype.hasOwnProperty.call(table, key);

  function failure(status, message) {
    return Object.assign(new Error(message), { status: status });
  }

  function notHere(what) {
    return Promise.reject(failure(409, what + " is not part of the recorded demo: run " + demo.live + " to try it"));
  }

  // ---- chats as the server would list them -----------------------------------------------

  function playedTurns(chat) {
    return chat.turns.filter((rec) => st.played[rec.turn.id]);
  }

  function summary(chat) {
    const done = playedTurns(chat);
    const last = done[done.length - 1];
    const running = st.running[chat.session.id] || null;
    return Object.assign({}, chat.session, { turns: done.length + (running ? 1 : 0), running: running, updated: last ? last.turn.ended : chat.session.created });
  }

  function detail(chat) {
    const done = playedTurns(chat);
    const views = done.map((rec) => copy(rec.turn));
    const live = st.running[chat.session.id];
    if (live) {
      const started = st.player && st.player.tid === live ? st.player.started : env.now() / 1000;
      views.push(Object.assign(copy(turns[live].rec.turn), { status: "running", answer: "", started: started, ended: null, input_tokens: 0, output_tokens: 0, cost_usd: 0, tools: [], approvals: [] }));
    }
    return { session: summary(chat), state: copy(done.length ? done[done.length - 1].state : {}), turns: views, files: [], branches: [] };
  }

  // ---- the run of a replaying turn: only the events it has reached ------------------------

  function partial(found, shown, since) {
    const events = found.events.slice(0, shown);
    const started = (found.summary && found.summary.started) || 0;
    const last = events.length ? events[events.length - 1].ts : started;
    const flows = {};
    events.forEach((ev) => { if (ev.type === "flow.started" && ev.data) flows[ev.data.flow] = true; });
    const graphs = {};
    Object.keys(found.graphs || {}).forEach((name) => { if (flows[name]) graphs[name] = found.graphs[name]; });
    const spans = (found.spans || []).filter((s) => s.events && s.events.length && s.events[0] < shown).map((s) => {
      if (s.events[s.events.length - 1] < shown) return s;
      return Object.assign({}, s, { status: "running", end: last, cost_usd: null, events: s.events.filter((i) => i < shown), marks: (s.marks || []).filter((m) => m.event < shown) });
    });
    const head = found.summary ? Object.assign({}, found.summary, { status: "running", events: shown, ended: last, duration_ms: Math.max(0, (last - started) * 1000), cost_usd: null }) : null;
    return Object.assign({}, found, { summary: head, graphs: graphs, spans: spans, spans_total: spans.length, events: events.slice(since), events_from: since, events_total: shown });
  }

  const readRun = AA.api.run;
  AA.api.run = async function (key, since) {
    if (!has(st.shown, key) || !details[key]) return readRun(key, since);
    return partial(details[key], st.shown[key], since || 0);
  };

  // The Runs view loads a run once (an export never polls): when it shows this run, load again.
  function refreshRunsView(runId) {
    env.setTimeout(() => {
      const runs = AA.runsView;
      const shown = runs && typeof runs.current === "function" ? runs.current() : null;
      if (shown && shown.summary && shown.summary.key === runId) runs.pick(runId);
    }, 0);
  }

  // ---- replay ----------------------------------------------------------------------------

  // Plays one recorded turn into `handlers` (the stream reader's): steps at their recorded pace.
  function play(rec, handlers) {
    const steps = rec.steps;
    const chatId = rec.turn.session_id;
    const runId = rec.turn.run_id;
    const player = { tid: rec.turn.id, at: 0, timer: null, closed: false, handlers: handlers, started: env.now() / 1000 };
    st.running[chatId] = player.tid;
    st.shown[runId] = 0;
    st.player = player;

    function finish(played) {
      if (player.timer !== null) env.clearTimeout(player.timer);
      player.timer = null;
      player.closed = true;
      player.handlers = null;
      delete st.shown[runId];
      if (st.running[chatId] === player.tid) delete st.running[chatId];
      if (st.player === player) st.player = null;
      if (st.reopening === player.tid) st.reopening = "";
      if (played) st.played[player.tid] = true;
      refreshRunsView(runId);
      paint();
    }

    function send(handlers, ev) {
      if (!handlers) return; // its chat is being opened again: the next stream gets every event
      try {
        handlers.onEvent(ev);
      } catch (err) {
        console.error("Agent Adapter chat demo: showing " + ev.type + " of " + player.tid + " failed", err);
      }
    }

    function deliver(step) {
      if (!step.sse) {
        st.shown[runId] = step.run;
        refreshRunsView(runId);
        return;
      }
      const ev = copy(step.sse);
      const handlers = player.handlers;
      if (ev.type === "done" || ev.type === "error") finish(true); // listed as played before the page asks
      send(handlers, ev);
    }

    function schedule() {
      if (player.closed) return;
      if (player.at >= steps.length) { finish(true); return; }
      const wait = env.reducedMotion() ? 0 : steps[player.at].at - (player.at ? steps[player.at - 1].at : 0);
      player.timer = env.setTimeout(tick, Math.max(wait, 0));
    }

    // Reduced motion shows the whole turn at once; otherwise the steps due now.
    function tick() {
      player.timer = null;
      const until = env.reducedMotion() ? Infinity : steps[player.at].at;
      while (!player.closed && player.at < steps.length && steps[player.at].at <= until) deliver(steps[player.at++]);
      schedule();
    }

    // The page closed the stream. Its chat being opened again keeps the replay going for the
    // stream that follows; anything else cuts it short, and the turn stays unplayed.
    player.close = () => {
      if (player.closed) return;
      if (st.reopening !== player.tid) { finish(false); return; }
      player.handlers = null;
      env.setTimeout(() => { if (!player.closed && !player.handlers) finish(false); }, 0); // never picked up
    };
    // A new stream of the same turn: it gets the events so far, then the rest at their pace.
    player.attach = (next) => {
      st.reopening = "";
      player.handlers = next;
      steps.slice(0, player.at).forEach((step) => { if (step.sse) send(next, copy(step.sse)); });
      paint();
    };
    schedule();
    paint();
    return player;
  }

  function streamTurn(turnId, since, handlers) {
    const found = turns[turnId];
    const quiet = { close() {}, next: () => 0 };
    if (!found) {
      env.setTimeout(() => handlers.onFatal(failure(404, "no such turn in the recording")), 0);
      return quiet;
    }
    if (st.played[turnId]) { // the server sends the end event again
      const end = found.rec.steps.filter((s) => s.sse && (s.sse.type === "done" || s.sse.type === "error")).pop();
      if (end) env.setTimeout(() => handlers.onEvent(copy(end.sse)), 0);
      return quiet;
    }
    let player = st.player;
    if (player && player.tid === turnId && !player.closed && !player.handlers) player.attach(handlers);
    else {
      st.reopening = "";
      if (player) player.close(); // one replay at a time
      player = play(found.rec, handlers);
    }
    const mine = handlers;
    return { close: () => { if (player.handlers === mine) player.close(); }, next: () => player.at };
  }

  async function startTurn(sessionId, body) {
    const chat = chats[sessionId];
    if (!chat) throw failure(404, "no such chat");
    if (st.running[sessionId]) throw failure(409, "a turn is already running in this chat");
    const next = chat.turns.find((rec) => !st.played[rec.turn.id]);
    if (!next || next.turn.text !== String((body && body.text) || "")) throw failure(400, "the recorded demo replays its suggested questions only");
    st.running[sessionId] = next.turn.id;
    paint();
    return { turn_id: next.turn.id, run_id: next.turn.run_id, session_id: sessionId };
  }

  AA.chatApi = Object.assign({}, AA.chatApi, {
    recorded: true,
    pollMs: 200,
    supports: (action) => OFF.indexOf(action) < 0,
    info: async () => copy(demo.info),
    sessions: async () => demo.chats.map(summary),
    session: async (id) => {
      if (!has(chats, id)) throw failure(404, "no such chat");
      if (st.player && st.running[id] === st.player.tid) st.reopening = st.player.tid; // the page opens it again
      return detail(chats[id]);
    },
    createSession: () => notHere("A new chat"),
    setState: () => notHere("Editing the state"),
    startTurn: startTurn,
    stop: () => Promise.reject(failure(409, "a recorded turn plays to its end")),
    replay: () => notHere("Replay"),
    decide: () => notHere("Deciding an approval"), // the recording decides; the card says so
    evalCase: () => notHere("Saving an eval case"),
    upload: () => notHere("Attaching files"),
    uploadProblem: () => "attaching files is not part of the recorded demo",
    fileBlob: () => notHere("Files"),
    streamTurn: streamTurn,
  });

  // ---- the page: chat on, composer replaced by the suggested questions ---------------------

  function paint() {
    const busy = Boolean(st.player) || Object.keys(st.running).length > 0;
    order.forEach((tid) => {
      const button = st.buttons[tid];
      if (!button) return;
      const done = Boolean(st.played[tid]);
      const playing = Boolean(st.player && st.player.tid === tid);
      button.classList.toggle("done", done);
      button.classList.toggle("playing", playing);
      button.setAttribute("aria-disabled", done || busy ? "true" : "false");
      const rec = turns[tid].rec;
      button.setAttribute("aria-label", (rec.label ? rec.label + ": " : "") + rec.turn.text + (done ? " (played)" : playing ? " (playing)" : ""));
    });
  }

  // Plays a question in its chat; earlier questions of that chat show as finished first.
  async function pick(tid) {
    const found = turns[tid];
    if (!found || st.played[tid] || st.player || Object.keys(st.running).length) return;
    const chatId = found.chat.session.id;
    const earlier = found.chat.turns.slice(0, found.index).filter((rec) => !st.played[rec.turn.id]);
    earlier.forEach((rec) => { st.played[rec.turn.id] = true; });
    try {
      if (earlier.length || AA.chatView.sid() !== chatId) {
        await AA.chatView.refresh();
        await AA.chatView.open(chatId);
      }
      await AA.chatView.ask(found.rec.turn.text);
    } catch (err) {
      console.error("Agent Adapter chat demo: playing " + tid + " failed", err);
    }
    paint();
    if (st.buttons[tid]) st.buttons[tid].focus(); // focus stays on the question, not the hidden box
  }

  async function startOver() {
    st.reopening = "";
    if (st.player) st.player.close();
    st.played = {};
    st.running = {};
    paint();
    await AA.chatView.refresh();
    await AA.chatView.open(demo.chats[0].session.id);
  }

  function buildBar(form) {
    const list = AA.el("div", { class: "demo-questions" });
    order.forEach((tid) => {
      const found = turns[tid];
      const button = AA.el("button", { type: "button", class: "demo-q", title: "plays in the chat " + found.chat.session.title, on: { click: () => pick(tid) } }, [
        AA.el("span", { class: "demo-tag", text: found.rec.label || found.chat.session.title }),
        AA.el("span", { text: found.rec.turn.text }),
      ]);
      st.buttons[tid] = button;
      list.appendChild(button);
    });
    const foot = AA.el("div", { class: "demo-foot" }, [AA.el("span", { class: "muted", text: "Pick a question to replay it in its chat." }), AA.el("span", { class: "spacer" }), AA.el("button", { type: "button", class: "ghost small", text: "Start over", on: { click: startOver } })]);
    const bar = AA.el("div", { class: "demo-bar", role: "group", "aria-label": "suggested questions" }, [
      AA.el("p", { class: "demo-note" }, [AA.el("strong", { text: "Recorded demo." }), " No agent runs here: each question replays a recorded turn. Run it live with ", AA.el("code", { text: demo.live }), "."]),
      list,
      foot,
    ]);
    const error = $("chat-error"); // it lives in the hidden composer: messages must stay visible
    if (error) bar.insertBefore(error, foot);
    form.parentNode.insertBefore(bar, form.nextSibling);
    paint();
  }

  AA.chatMode = "demo"; // the export's data.js reports no chat: this page has a recorded one
  const off = $("chat-off");
  if (off) off.hidden = true;
  const shell = $("chat-shell");
  if (shell) { shell.hidden = false; shell.classList.add("chat-demo"); }
  const label = $("exported");
  if (label) label.textContent = "recorded demo";
  const newChat = $("chat-new");
  if (newChat) newChat.hidden = true;
  const form = $("chat-composer");
  if (form) { form.hidden = true; buildBar(form); }
  if (AA.views && AA.views.has("chat")) AA.views.show("chat"); // the Runs view already put its run in the address
  AA.chatDemo = { env: env, state: st, pick: pick, startOver: startOver }; // tests drive these
})();
