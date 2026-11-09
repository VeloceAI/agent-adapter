// Copy buttons, the prompt copy button, card filters and the home page code tabs.
// navigation.instant swaps pages without a reload, so everything re-binds on document$.
(function () {
  var base = new URL(".", document.currentScript.src).href;
  var promptText = "";
  fetch(new URL("../ai-prompt.txt", base)).then(function (r) { return r.text(); })
    .then(function (t) { promptText = t; }).catch(function () {});

  function toast(msg) {
    var el = document.querySelector(".aa-toast");
    if (!el) { el = document.createElement("div"); el.className = "aa-toast"; document.body.appendChild(el); }
    el.textContent = msg;
    el.classList.add("on");
    clearTimeout(el._t);
    el._t = setTimeout(function () { el.classList.remove("on"); }, 1800);
  }

  function copy(text, done) {
    var ok = function () { toast(done || "Copied"); };
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(ok, function () { fallback(text); ok(); });
    } else { fallback(text); ok(); }
  }

  function fallback(text) {
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch (e) { /* nothing else to try */ }
    ta.remove();
  }

  function promptPage() { return new URL("../../get-started/ai-prompt/", base).href; }


  document.addEventListener("click", function (e) {
    var p = e.target.closest("[data-aa-copy-prompt]");
    if (p) {
      e.preventDefault();
      if (promptText) copy(promptText, "Prompt copied: paste it into your AI");
      else window.location.href = promptPage();
      return;
    }
    var open = e.target.closest("[data-aa-open]");
    if (open) { document.getElementById(open.getAttribute("data-aa-open")).showModal(); return; }
    if (e.target.closest("[data-aa-close]") || e.target.tagName === "DIALOG") { e.target.closest("dialog").close(); return; }
    var a = e.target.closest(".aa-agent");
    if (a) { showAgent(a); return; }
    var c = e.target.closest("[data-aa-copy]");
    if (c) { e.preventDefault(); copy(c.getAttribute("data-aa-copy")); return; }
    var f = e.target.closest(".aa-filter button");
    if (f) {
      var kind = f.getAttribute("data-kind");
      f.parentNode.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b === f); });
      var grid = f.closest(".aa-filter").nextElementSibling;
      grid.querySelectorAll(".aa-gcard").forEach(function (card) {
        card.hidden = kind !== "all" && card.getAttribute("data-kind") !== kind;
      });
      return;
    }
    var tab = e.target.closest(".aa-tabs [role=tab]");
    if (tab) {
      var box = tab.closest(".aa-tabs");
      box.querySelectorAll("[role=tab]").forEach(function (t) { t.setAttribute("aria-selected", t === tab); });
      box.querySelectorAll("[role=tabpanel]").forEach(function (panel) {
        panel.hidden = panel.id !== tab.getAttribute("aria-controls");
      });
    }
  });

  // An agent chip (hero or skill section) puts that agent's install line in the skill section's box.
  function showAgent(btn) {
    var skill = document.querySelector(".aa-skill");
    if (!skill) return;
    var agent = btn.getAttribute("data-agent");
    skill.querySelectorAll(".aa-agent").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-agent") === agent); });
    var claude = agent === "claude-code";  // Claude Code has its own plugin commands
    var cmd = claude ? "/plugin marketplace add VeloceAI/agent-adapter\n/plugin install agent-adapter@agent-adapter"
      : "npx skills add VeloceAI/agent-adapter --skill agent-adapter -a " + agent;
    var tab = skill.querySelector('[aria-controls="aa-sk-any"]'), panel = skill.querySelector("#aa-sk-any");
    tab.textContent = btn.getAttribute("data-name");
    panel.querySelector(".aa-copybtn").setAttribute("data-aa-copy", cmd);
    panel.querySelector(".aa-prompt").hidden = claude;
    panel.querySelector("pre").lastChild.textContent = (claude ? "" : " ") + cmd;
    tab.click();
    if (!skill.contains(btn)) skill.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  document.addEventListener("submit", function (e) {
    var form = e.target.closest("[data-aa-contact]");
    if (!form) return;
    e.preventDefault();
    var status = form.querySelector(".aa-contact-status"), send = form.querySelector("[type=submit]");
    var to = form.getAttribute("data-aa-mailto");
    if (to) {
      var d = new FormData(form);
      var body = ["Name: " + d.get("name"), "Email: " + d.get("email"), "Company: " + d.get("company"),
        "Team size: " + d.get("team_size"), "Need: " + d.get("topic"), "", d.get("message")].join("\n");
      window.location.href = "mailto:" + to + "?subject=" + encodeURIComponent("Agent Adapter enquiry: " + d.get("topic")) +
        "&body=" + encodeURIComponent(body);
      status.textContent = "Your email app should open with this message. Press send there.";
      return;
    }
    send.disabled = true;
    status.textContent = "Sending...";
    fetch(form.action, { method: "POST", headers: { Accept: "application/json" }, body: new FormData(form) })
      .then(function (r) { return r.json(); })
      .then(function (res) {
        if (!res.success) throw new Error(res.message);
        form.reset();
        status.textContent = "Thanks, we got it. We reply within one working day.";
      })
      .catch(function () { status.textContent = "That did not send. Please try again in a minute."; send.disabled = false; });
  });

  function revealOnScroll() {
    var items = document.querySelectorAll(".aa-reveal:not(.in)");
    if (!("IntersectionObserver" in window)) { items.forEach(function (i) { i.classList.add("in"); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
    }, { threshold: 0.15 });
    items.forEach(function (i) { io.observe(i); });
  }

  function init() { revealOnScroll(); }
  if (window.document$) window.document$.subscribe(init); else document.addEventListener("DOMContentLoaded", init);
})();
