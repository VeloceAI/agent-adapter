// Chat sessions list and the conversation header (title, agent, parent chat, branches).
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});

  // The list on the left; redrawn only when a session changed, and focus stays on its button.
  AA.createSessionList = function (container, onOpen) {
    let shown = null;

    function button(s, current) {
      const meta = [s.turns + " turn" + (s.turns === 1 ? "" : "s"), AA.fmt.time(s.updated)];
      const node = AA.el("button", { type: "button", class: "run" + (s.id === current ? " selected" : ""), "aria-current": s.id === current ? "true" : null, on: { click: () => onOpen(s.id) } }, [
        AA.el("span", { class: "when", text: s.title || "New chat" }),
        s.running ? AA.el("span", { class: "status running", text: "running" }) : s.parent ? AA.el("span", { class: "chip", text: "branch" }) : AA.el("span"),
        AA.el("span", { class: "meta", text: meta.join(" · ") }),
      ]);
      node.dataset.sid = s.id;
      return node;
    }

    return {
      update(sessions, current) {
        const key = JSON.stringify([current, sessions.map((s) => [s.id, s.title, s.turns, s.updated, Boolean(s.running), Boolean(s.parent)])]);
        if (key === shown) return false;
        shown = key;
        const active = document.activeElement;
        const focused = active && container.contains(active) && active.dataset ? active.dataset.sid : "";
        AA.clear(container);
        if (!sessions.length) container.appendChild(AA.el("p", { class: "muted", text: "No chats yet. Type a message to start one." }));
        sessions.forEach((s) => {
          const node = container.appendChild(button(s, current));
          if (s.id === focused) node.focus();
        });
        return true;
      },
    };
  };

  // The header above the conversation.
  AA.renderChatHead = function (container, detail, sessions, agent, onOpen) {
    AA.clear(container);
    const session = detail && detail.session;
    container.appendChild(AA.el("div", { class: "card-head" }, [
      AA.el("h2", { text: session ? session.title : "New chat" }),
      AA.el("span", { class: "chip", title: "the agent this server runs", text: agent }),
    ]));
    if (session && session.parent) {
      const parent = sessions.find((s) => s.id === session.parent.session_id);
      container.appendChild(AA.el("p", { class: "muted" }, ["Branch of ", AA.el("button", { type: "button", class: "linkish", text: parent ? parent.title : "the earlier chat", on: { click: () => onOpen(session.parent.session_id) } }), ", from before one of its turns."]));
    }
    const branches = (detail && detail.branches) || [];
    if (branches.length) {
      container.appendChild(AA.el("div", { class: "chips" }, [AA.el("span", { class: "muted", text: "branches:" })].concat(branches.map((b) => AA.el("button", { type: "button", class: "chip", text: b.title, on: { click: () => onOpen(b.id) } })))));
    }
  };
})();
