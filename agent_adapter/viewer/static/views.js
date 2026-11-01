// Top-level views (Runs, and Chat when the server runs an agent): AA.views.show(name).
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const views = {};
  let current = null;

  // A tab list: the picked tab is the one in the Tab order (roving tabindex), arrows move along.
  function selectTab(buttons, picked) {
    buttons.forEach((b) => {
      const on = b === picked;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
      b.setAttribute("tabindex", on ? "0" : "-1");
    });
  }

  function arrowKeys(list, pick) {
    list.addEventListener("keydown", (ev) => {
      const tabs = Array.from(list.querySelectorAll('[role="tab"]')).filter((b) => !b.hidden);
      const at = tabs.indexOf(document.activeElement);
      if (at < 0) return;
      const next = { ArrowRight: at + 1, ArrowLeft: at - 1, Home: 0, End: tabs.length - 1 }[ev.key];
      if (next === undefined) return;
      ev.preventDefault();
      const target = tabs[(next + tabs.length) % tabs.length];
      target.focus();
      pick(target);
    });
  }

  function linkPanel(button, panel) {
    if (!button || !panel) return;
    if (!button.id) button.id = panel.id + "-tab";
    button.setAttribute("aria-controls", panel.id);
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", button.id);
  }

  AA.tabs = { select: selectTab, arrowKeys: arrowKeys, linkPanel: linkPanel };

  function show(name) {
    if (!views[name] || !views[name].section) {
      console.warn("Agent Adapter viewer: no view named " + name);
      return false;
    }
    const before = current;
    Object.keys(views).forEach((key) => {
      if (views[key].section) views[key].section.hidden = key !== name;
    });
    const buttons = Object.keys(views).map((key) => views[key].button).filter(Boolean);
    selectTab(buttons, views[name].button);
    current = name;
    if (before && before !== name && views[before].onHide) views[before].onHide();
    if (views[name].onShow) views[name].onShow();
    return true;
  }

  // Add or extend a view: {section, button, onShow, onHide}; a button shows its view on click.
  function register(name, parts) {
    const view = (views[name] = Object.assign(views[name] || {}, parts || {}));
    if (view.button && !view.button.dataset.wired) {
      view.button.dataset.wired = "1";
      view.button.addEventListener("click", () => show(name));
      linkPanel(view.button, view.section);
    }
    return view;
  }

  AA.views = { show: show, register: register, current: () => current, has: (name) => Boolean(views[name]) };

  register("runs", { section: document.getElementById("runs-view"), button: document.getElementById("view-runs") });
  register("chat", { section: document.getElementById("chat-view"), button: document.getElementById("view-chat") });
  // The Chat tab always shows, so chat is easy to find; without an agent it says how to add one.
  const chatButton = document.getElementById("view-chat");
  if (chatButton) chatButton.hidden = false;
  const chatOff = document.getElementById("chat-off");
  if (chatOff && !AA.chatMode) {
    chatOff.hidden = false;
    const shell = document.getElementById("chat-shell");
    if (shell) shell.hidden = true;
    const why = document.getElementById("chat-off-why");
    if (why && AA.embedded) why.textContent = "This is an exported file: chat needs a live viewer.";
  }
  const topTabs = document.getElementById("top-tabs");
  if (topTabs) arrowKeys(topTabs, (button) => show(button.getAttribute("data-view")));
  if (views.runs.section) show("runs");
})();
