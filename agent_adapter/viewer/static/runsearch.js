// Runs list search and tag filter, and the tag editor in the run header.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const el = (...args) => AA.el(...args);
  const DEBOUNCE_MS = 250;
  const MAX_TAG = 40;

  // The search box and tag filter; `onChange()` runs after the user stops typing.
  function createSearch(container, onChange) {
    const state = { q: "", tag: "", tags: [] };
    const input = el("input", { type: "search", class: "run-q", placeholder: "search runs: tool, flow, status:failed, tag:x", "aria-label": "search runs", maxlength: "200" });
    const pick = el("select", { class: "run-tag", "aria-label": "filter by tag" });
    const count = el("span", { class: "muted run-count", "aria-live": "polite" });
    let timer = null;
    input.addEventListener("input", () => {
      clearTimeout(timer);
      timer = setTimeout(() => { state.q = input.value.trim(); onChange(); }, DEBOUNCE_MS);
    });
    pick.addEventListener("change", () => { state.tag = pick.value; onChange(); });
    AA.clear(container);
    container.appendChild(el("div", { class: "run-search" }, [input, pick, count]));

    function paintTags(tags) {
      const known = (tags || []).map(String);
      if (state.tag && known.indexOf(state.tag) < 0) known.unshift(state.tag);
      if (known.join("\n") === state.tags.join("\n") && pick.firstChild) return;
      state.tags = known;
      AA.clear(pick);
      pick.appendChild(el("option", { value: "", text: "all tags" }));
      known.forEach((tag) => pick.appendChild(el("option", { value: tag, text: tag })));
      pick.value = state.tag;
      pick.hidden = !known.length;
    }

    return {
      params: () => ({ q: state.q, tag: state.tag }),
      active: () => Boolean(state.q || state.tag),
      paint(body) {
        paintTags(body.tags);
        const shown = (body.runs || []).length;
        count.textContent = state.q || state.tag ? shown + " of " + (body.total || 0) + " runs" : "";
      },
    };
  }

  // Tag chips with remove buttons and an add box; `save(change)` returns a Promise of tags.
  function renderTags(container, tags, save) {
    const box = el("div", { class: "run-tags", "aria-label": "run tags" });
    const notice = el("span", { class: "notice", "aria-live": "polite" });
    const input = el("input", { type: "text", placeholder: "add a tag", "aria-label": "new tag", maxlength: String(MAX_TAG) });
    const busy = (on) => box.querySelectorAll("button,input").forEach((node) => { node.disabled = on; });

    function apply(change) {
      busy(true);
      notice.className = "notice";
      notice.textContent = "saving...";
      return Promise.resolve(save(change)).then((found) => {
        paint(found || []);
        notice.textContent = "";
      }, (err) => {
        notice.className = "notice bad";
        notice.textContent = "Not saved: " + (err && err.message ? err.message : "error");
      }).finally(() => busy(false));
    }

    function paint(list) {
      AA.clear(box);
      list.forEach((tag) => {
        const remove = el("button", { type: "button", class: "ghost small", text: "×", title: "remove tag " + tag, "aria-label": "remove tag " + tag, on: { click: () => apply({ remove: [tag] }) } });
        box.appendChild(el("span", { class: "chip tag" }, [el("span", { text: tag }), remove]));
      });
      const form = el("form", { class: "tag-add", on: { submit: (ev) => {
        ev.preventDefault();
        const tag = input.value.trim();
        if (!tag) return;
        input.value = "";
        apply({ add: [tag] });
      } } }, [input, el("button", { type: "submit", class: "ghost small", text: "add" })]);
      box.appendChild(form);
      box.appendChild(notice);
    }

    paint((tags || []).map(String));
    container.appendChild(box);
    return box;
  }

  // The run header's tag editor, kept across re-renders of the same run so a half-typed tag
  // or a "Not saved" notice survives the live refresh; rebuilt when the tags changed elsewhere
  // and nobody is typing in it.
  let editor = null;

  function attachTags(container, key, tags, save) {
    const sig = (tags || []).join("\n");
    const same = editor && editor.key === key;
    if (same && (editor.sig === sig || editor.box.contains(document.activeElement))) {
      container.appendChild(editor.box);
      return editor.box;
    }
    const current = { key: key, sig: sig, box: null };
    current.box = renderTags(container, tags, (change) => Promise.resolve(save(change)).then((found) => {
      current.sig = (found || []).join("\n");
      return found;
    }));
    editor = current;
    return current.box;
  }

  AA.runSearch = { createSearch: createSearch, renderTags: renderTags, attachTags: attachTags };
})();
