// Copy buttons, the header "Copy setup prompt" button and the home page code tabs.
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

  function headerButton() {
    var inner = document.querySelector(".md-header__inner");
    if (!inner || inner.querySelector(".aa-prompt-btn")) return;
    var b = document.createElement("button");
    b.className = "aa-prompt-btn";
    b.type = "button";
    b.setAttribute("data-aa-copy-prompt", "");
    b.title = "Copy a prompt that sets Agent Adapter up in your project, for any AI";
    b.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2a2 2 0 0 1 2 2c0 .74-.4 1.39-1 1.73V7h1a7 7 0 0 1 7 7h1a1 1 0 0 1 1 1v3a1 1 0 0 1-1 1h-1v1a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-1H2a1 1 0 0 1-1-1v-3a1 1 0 0 1 1-1h1a7 7 0 0 1 7-7h1V5.73A2 2 0 0 1 12 2M7.5 13A2.5 2.5 0 0 0 5 15.5 2.5 2.5 0 0 0 7.5 18a2.5 2.5 0 0 0 2.5-2.5A2.5 2.5 0 0 0 7.5 13m9 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 2.5 2.5 2.5 2.5 0 0 0 2.5-2.5 2.5 2.5 0 0 0-2.5-2.5"/></svg><span>Copy setup prompt</span>';
    var anchor = inner.querySelector(".md-header__source") || inner.lastElementChild;
    inner.insertBefore(b, anchor);
  }

  document.addEventListener("click", function (e) {
    var p = e.target.closest("[data-aa-copy-prompt]");
    if (p) {
      e.preventDefault();
      if (promptText) copy(promptText, "Prompt copied: paste it into your AI");
      else window.location.href = promptPage();
      return;
    }
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

  function revealOnScroll() {
    var items = document.querySelectorAll(".aa-reveal:not(.in)");
    if (!("IntersectionObserver" in window)) { items.forEach(function (i) { i.classList.add("in"); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } });
    }, { threshold: 0.15 });
    items.forEach(function (i) { io.observe(i); });
  }

  function init() { headerButton(); revealOnScroll(); }
  if (window.document$) window.document$.subscribe(init); else document.addEventListener("DOMContentLoaded", init);
})();
