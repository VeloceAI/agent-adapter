// Diagrams fit the text column at any screen size (Mermaid's default). A click or tap opens a
// diagram full screen at its natural size, with zoom, so wide diagrams stay readable on a phone.
(function () {
  "use strict";
  const mermaid = window.mermaid;
  if (!mermaid || typeof mermaid.render !== "function") {
    console.warn("Agent Adapter docs: the bundled Mermaid did not load; diagrams cannot be enlarged");
    return;
  }

  const STEP = 1.25;
  const MIN_SCALE = 0.25;
  const MAX_SCALE = 4;
  const STYLE = [
    ":host { all: initial; }",
    "dialog { box-sizing: border-box; width: 100vw; height: 100dvh; max-width: none; max-height: none; margin: 0; padding: 0; border: 0;",
    "  display: flex; flex-direction: column; background: var(--md-default-bg-color, #fff); color: var(--md-default-fg-color, #222);",
    "  font-family: var(--md-text-font-family, system-ui, sans-serif); }",
    "dialog:not([open]) { display: none; }",
    "dialog::backdrop { background: rgba(0, 0, 0, .5); }",
    ".bar { display: flex; gap: 8px; align-items: center; padding: 8px 12px; border-bottom: 1px solid var(--md-default-fg-color--lightest, #ddd); }",
    ".bar span { flex: 1; font-size: 14px; }",
    "button { font: inherit; font-size: 14px; min-width: 44px; min-height: 44px; padding: 0 12px; cursor: pointer; border-radius: 6px;",
    "  border: 1px solid var(--md-default-fg-color--lighter, #bbb); background: transparent; color: inherit; }",
    "button:focus-visible { outline: 2px solid var(--md-accent-fg-color, #526cfe); outline-offset: 2px; }",
    ".view { flex: 1; overflow: auto; padding: 16px; -webkit-overflow-scrolling: touch; touch-action: pan-x pan-y pinch-zoom; }",
    ".view svg { display: block; margin: 0 auto; height: auto; max-width: none !important; }",
  ].join("\n");

  let viewer = null;

  function naturalWidth(svg) {
    const box = svg.viewBox && svg.viewBox.baseVal;
    return box && box.width ? box.width : svg.getBoundingClientRect().width || 800;
  }

  function buildViewer() {
    const host = document.createElement("div");
    host.className = "aa-diagram-viewer";
    const root = host.attachShadow({ mode: "open" });
    root.innerHTML =
      "<style>" + STYLE + "</style>" +
      '<dialog aria-label="Diagram, enlarged">' +
      '<div class="bar"><span>Diagram</span>' +
      '<button data-zoom="out" aria-label="Zoom out">−</button>' +
      '<button data-zoom="fit" aria-label="Fit to screen">Fit</button>' +
      '<button data-zoom="in" aria-label="Zoom in">+</button>' +
      '<button data-close aria-label="Close">✕</button></div>' +
      '<div class="view"></div></dialog>';
    document.body.append(host);
    const dialog = root.querySelector("dialog");
    const view = root.querySelector(".view");
    const state = { scale: 1, width: 0, opener: null };

    const apply = () => {
      const svg = view.querySelector("svg");
      if (svg) svg.style.width = Math.round(state.width * state.scale) + "px";
    };
    const fit = () => (view.clientWidth - 32) / state.width;
    const zoom = (how) => {
      const next = how === "fit" ? fit() : how === "in" ? state.scale * STEP : state.scale / STEP;
      state.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, next));
      apply();
    };

    root.addEventListener("click", (event) => {
      const button = event.target.closest("button");
      if (!button) return;
      if (button.hasAttribute("data-close")) dialog.close();
      else zoom(button.dataset.zoom);
    });
    dialog.addEventListener("keydown", (event) => {
      if (event.key === "+" || event.key === "=") zoom("in");
      else if (event.key === "-") zoom("out");
      else if (event.key === "0") zoom("fit");
    });
    dialog.addEventListener("close", () => {
      view.replaceChildren();
      if (state.opener && state.opener.isConnected) state.opener.focus();
    });

    return {
      open(svgText, opener) {
        view.innerHTML = svgText;
        const svg = view.querySelector("svg");
        if (!svg) return;
        svg.removeAttribute("width");
        state.width = naturalWidth(svg);
        state.opener = opener;
        dialog.showModal();
        // A wide diagram opens at natural size and scrolls; a small one grows to fit, at most 2x.
        state.scale = Math.min(2, Math.max(1, fit()));
        apply();
      },
    };
  }

  function enlarge(svgText, opener) {
    viewer = viewer || buildViewer();
    viewer.open(svgText, opener);
  }

  // Material renders each diagram into a closed shadow root, so keep the SVG Mermaid returned.
  function attach(host, svgText) {
    const figure = document.createElement("figure");
    figure.className = "aa-diagram";
    host.replaceWith(figure);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "aa-diagram__enlarge";
    button.setAttribute("aria-label", "Enlarge diagram");
    button.textContent = "⤢ Enlarge";
    figure.append(host, button);
    host.addEventListener("click", () => enlarge(svgText, button));
    button.addEventListener("click", () => enlarge(svgText, button));
  }

  function sourceOf(text) {
    for (const el of document.querySelectorAll(".mermaid:not([data-aa-diagram])")) {
      if (el.textContent === text) {
        el.setAttribute("data-aa-diagram", "");
        return el;
      }
    }
    return null;
  }

  const render = mermaid.render.bind(mermaid);
  mermaid.render = function (id, text, ...rest) {
    const source = sourceOf(text);
    const parent = source && source.parentNode;
    const next = source && source.nextSibling;
    return render(id, text, ...rest).then((result) => {
      // Material swaps the source for its diagram right after this resolves.
      setTimeout(() => {
        const host = next ? next.previousSibling : parent && parent.lastChild;
        if (host && host !== source && host.matches && host.matches("div.mermaid")) attach(host, result.svg);
      });
      return result;
    });
  };
})();
