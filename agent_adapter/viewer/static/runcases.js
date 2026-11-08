// Runs view: save the shown run as an eval case, or a failed run as a regression case.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});
  const el = (...args) => AA.el(...args);
  const MAX_ID = 200;

  // POST the case; resolves to { case, file }, rejects with the server's reason.
  async function saveCase(key, body) {
    const response = await fetch("api/runs/" + encodeURIComponent(key) + "/eval-case", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-AA-CSRF": AA.csrf, "X-AA-Token": AA.token ? AA.token() : "" },
      body: JSON.stringify(body),
    });
    const found = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(found.error || "HTTP " + response.status);
    return found;
  }

  function parseTags(text) {
    return String(text || "").split(",").map((t) => t.trim()).filter(Boolean).slice(0, 20);
  }

  // `expected`: JSON when it parses, else the text itself; undefined when left empty.
  function parseExpected(text) {
    const source = String(text || "").trim();
    if (!source) return undefined;
    try {
      return JSON.parse(source);
    } catch (err) {
      return source;
    }
  }

  function done(notice, saved) {
    notice.className = "notice good";
    notice.textContent = "Saved " + saved.case.id + " to " + saved.file + ".";
  }

  function failed(notice, err) {
    notice.className = "notice bad";
    notice.textContent = "Not saved: " + (err && err.message ? err.message : "error");
  }

  function caseDialog(key, status, save) {
    const id = el("input", { type: "text", maxlength: String(MAX_ID), required: true, "aria-label": "case id" });
    id.value = "run-" + key;
    const useOutput = el("input", { type: "checkbox", "aria-label": "use the recorded output as expected" });
    useOutput.checked = status === "ok";
    const expected = el("textarea", { rows: "4", "aria-label": "expected (JSON or text)" });
    const tags = el("input", { type: "text", placeholder: "tag, another", "aria-label": "tags" });
    const notice = el("p", { class: "notice", "aria-live": "polite" });
    const button = el("button", { type: "button", class: "approve", text: "Save case" });
    const box = el("dialog", { class: "chat-dialog", "aria-label": "Save as eval case" }, [
      el("div", { class: "card-head" }, [el("h2", { text: "Save as eval case" }), el("span", { class: "spacer" }),
        el("button", { type: "button", class: "ghost small", text: "close", on: { click: () => box.close() } })]),
      el("div", { class: "form" }, [
        el("label", null, ["case id", id]),
        el("label", null, [useOutput, " use the recorded output as expected (ok runs only)"]),
        el("label", null, ["expected (JSON, or text; replaces the recorded output; empty for none)", expected]),
        el("label", null, ["tags (comma separated)", tags]),
        el("p", { class: "muted", text: "The input is the run's first user message or tool arguments, when it was recorded with capture_content=True." }),
        el("div", { class: "actions" }, [button, notice]),
      ]),
    ]);
    box.addEventListener("close", () => box.remove());
    document.body.appendChild(box);
    box.showModal();
    button.addEventListener("click", async () => {
      const name = id.value.trim();
      if (!name) { failed(notice, new Error("give the case an id")); return; }
      const body = { id: name, tags: parseTags(tags.value), expected_from: useOutput.checked ? "output" : "none" };
      const value = parseExpected(expected.value);
      if (value !== undefined) body.expected = value;
      button.disabled = true;
      try {
        done(notice, await save(body));
      } catch (err) {
        failed(notice, err);
        button.disabled = false;
      }
    });
    id.focus();
    return box;
  }

  // The run header's case buttons, kept per run so a "Saved" notice survives the live refresh.
  let current = null;

  function attach(container, key, status, save) {
    const send = save || ((body) => saveCase(key, body));
    if (!current || current.key !== key || current.status !== status) {
      const notice = el("span", { class: "notice", "aria-live": "polite" });
      const items = [el("button", { type: "button", class: "ghost small", text: "Save as eval case", on: { click: () => caseDialog(key, status, send) } })];
      if (status === "failed") {
        const quick = el("button", { type: "button", class: "ghost small", text: "Turn into regression case", on: { click: async () => {
          quick.disabled = true;
          try {
            done(notice, await send({ regression: true }));
          } catch (err) {
            failed(notice, err);
            quick.disabled = false;
          }
        } } });
        items.push(quick);
      }
      items.push(notice);
      current = { key: key, status: status, box: el("div", { class: "run-cases", "aria-label": "eval cases" }, items) };
    }
    container.appendChild(current.box);
    return current.box;
  }

  AA.runCases = { attach: attach, caseDialog: caseDialog, saveCase: saveCase, parseExpected: parseExpected };
})();
