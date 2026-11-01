// A small, safe Markdown subset for chat answers: built as DOM nodes, never parsed as HTML.
// Every pattern here is linear in its input: answers come from models and tools, so a
// hostile or degenerate line must never freeze the page.
"use strict";

(function () {
  const AA = (window.AA = window.AA || {});

  const MAX_RENDER = 256 * 1024; // chars of one text segment drawn as Markdown; longer is plain text
  const MAX_LABEL = 1000; // chars of a link label
  const MAX_TARGET = 2048; // chars of a link target
  const MAX_DEPTH = 8; // nested bold/italic/link levels
  const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
  const LANG = /^[\w+#.-]{1,30}/;
  const BULLET = /^([ \t]*)([-*+]|\d{1,9}[.)])[ \t]+(.*)$/;
  const HEADING = /^ {0,3}(#{1,6})(?:[ \t]+(.*))?$/;
  const QUOTE = /^ {0,3}>[ \t]?(.*)$/;
  const DELIMITER = /^\|?:?-+:?(?:\|:?-+:?)*\|?$/; // tested with all spaces removed
  const SCHEME = /^https?:\/\/\S+$/i;
  const UNSAFE_URL = /[\u0000-\u001f\u007f\p{Cf}]/u; // control and bidi/format characters
  const BARE_URL = /https?:\/\/[^\s<>]*[^\s<>.,:;"'!?)\]]/iy;
  const TICKS = /`+/y;
  const ESCAPABLE = "\\`*_{}[]()#+-.!|>~";

  // The URL to link to, or "" when it is not plain http(s) (javascript:, data:, relative...).
  function safeHref(raw) {
    const url = String(raw || "").trim();
    if (!SCHEME.test(url) || UNSAFE_URL.test(url)) return "";
    return url;
  }

  // ---- blocks ----------------------------------------------------------------------------

  function splitRow(line) {
    let text = line.trim();
    if (text.startsWith("|")) text = text.slice(1);
    if (text.endsWith("|") && !text.endsWith("\\|")) text = text.slice(0, -1);
    const cells = [];
    let cell = "";
    for (let i = 0; i < text.length; i += 1) {
      if (text[i] === "\\" && text[i + 1] === "|") { cell += "|"; i += 1; }
      else if (text[i] === "|") { cells.push(cell.trim()); cell = ""; }
      else cell += text[i];
    }
    cells.push(cell.trim());
    return cells;
  }

  function alignOf(cell) {
    const left = cell.startsWith(":"), right = cell.endsWith(":");
    return left && right ? "center" : right ? "right" : left ? "left" : "";
  }

  function isRule(line) {
    if (/^ {4}/.test(line)) return false;
    const marks = line.replace(/[ \t]/g, "");
    return marks.length >= 3 && "-*_".includes(marks[0]) && marks.split(marks[0]).length === marks.length + 1;
  }

  function isTableStart(lines, i) {
    return lines[i].includes("|") && i + 1 < lines.length && DELIMITER.test(lines[i + 1].replace(/\s+/g, ""));
  }

  // A heading's text without its optional closing `#` run (only when a space comes before it).
  function headingText(raw) {
    let text = (raw || "").trimEnd();
    let end = text.length;
    while (end > 0 && text[end - 1] === "#") end -= 1;
    if (end === 0) return "";
    if (end < text.length && (text[end - 1] === " " || text[end - 1] === "\t")) text = text.slice(0, end).trimEnd();
    return text;
  }

  function startsBlock(lines, i) {
    const line = lines[i];
    return FENCE.test(line) || HEADING.test(line) || isRule(line) || BULLET.test(line) || QUOTE.test(line) || isTableStart(lines, i);
  }

  // Blocks of the text; each keeps its source (`src`) so a re-render can keep unchanged ones.
  function parse(text) {
    const lines = String(text || "").replace(/\r\n?/g, "\n").split("\n");
    const blocks = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (!line.trim()) { i += 1; continue; }
      const start = i;
      const fence = FENCE.exec(line);
      if (fence) {
        const close = fence[1][0].repeat(fence[1].length);
        const lang = LANG.exec(fence[2].trim());
        const body = [];
        i += 1;
        while (i < lines.length && !(lines[i].trim().startsWith(close) && !lines[i].trim().replace(/[`~]/g, ""))) body.push(lines[i++]);
        const open = i >= lines.length; // still streaming, or never closed
        if (!open) i += 1;
        blocks.push({ type: "code", lang: lang ? lang[0] : "", code: body.join("\n"), open: open, src: lines.slice(start, i).join("\n") });
        continue;
      }
      const heading = HEADING.exec(line);
      if (heading) {
        blocks.push({ type: "heading", level: heading[1].length, text: headingText(heading[2]), src: line });
        i += 1;
        continue;
      }
      if (isRule(line)) { blocks.push({ type: "rule", src: line }); i += 1; continue; }
      if (isTableStart(lines, i)) {
        const head = splitRow(line);
        const align = splitRow(lines[i + 1]).map(alignOf);
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].trim() && lines[i].includes("|")) rows.push(splitRow(lines[i++]));
        blocks.push({ type: "table", head: head, align: align, rows: rows, src: lines.slice(start, i).join("\n") });
        continue;
      }
      if (BULLET.test(line)) {
        const items = [];
        const ordered = /\d/.test(BULLET.exec(line)[2]);
        while (i < lines.length) {
          const item = BULLET.exec(lines[i]);
          if (item) {
            items.push({ depth: Math.min(Math.floor(item[1].replace(/\t/g, "  ").length / 2), 3), text: item[3], start: /\d/.test(item[2]) ? parseInt(item[2], 10) : null });
            i += 1;
          } else if (lines[i].trim() && /^[ \t]/.test(lines[i]) && items.length) {
            items[items.length - 1].text += "\n" + lines[i].trim(); // a wrapped item line
            i += 1;
          } else break;
        }
        blocks.push({ type: "list", ordered: ordered, items: items, src: lines.slice(start, i).join("\n") });
        continue;
      }
      if (QUOTE.test(line)) {
        const body = [];
        while (i < lines.length && QUOTE.test(lines[i])) body.push(QUOTE.exec(lines[i++])[1]);
        blocks.push({ type: "quote", text: body.join("\n"), src: lines.slice(start, i).join("\n") });
        continue;
      }
      const body = [line];
      i += 1;
      while (i < lines.length && lines[i].trim() && !startsBlock(lines, i)) body.push(lines[i++]);
      blocks.push({ type: "para", text: body.join("\n"), src: body.join("\n") });
    }
    return blocks;
  }

  // ---- inline ----------------------------------------------------------------------------

  // `next(marker, from)`: the first unescaped `marker` at or after `from`, or -1. Results are
  // reused while scanning forward, so a run of unclosed openers costs one pass, not one each.
  function finder(text) {
    const seen = {};
    return function next(marker, from) {
      const hit = seen[marker];
      if (hit && hit.from <= from && (hit.at < 0 || hit.at >= from)) return hit.at;
      let at = text.indexOf(marker, from);
      while (marker !== "\n" && at > 0 && text[at - 1] === "\\") at = text.indexOf(marker, at + 1);
      seen[marker] = { from: from, at: at };
      return at;
    };
  }

  function wordChar(ch) {
    return Boolean(ch) && /[\p{L}\p{N}]/u.test(ch);
  }

  // `[label](target)` starting at `i`: {label, href, end} or null.
  function linkAt(text, i, next) {
    const mid = next("](", i + 1);
    if (mid < 0 || mid - i > MAX_LABEL) return null;
    const newline = next("\n", i + 1);
    if (newline >= 0 && newline < mid) return null;
    let end = next(")", mid + 2);
    if (end < 0 || end - mid > MAX_TARGET) return null;
    const open = next("(", mid + 2);
    if (open >= 0 && open < end) { // one level of parentheses inside the target, e.g. a wiki URL
      end = next(")", end + 1);
      if (end < 0 || end - mid > MAX_TARGET) return null;
    }
    const target = text.slice(mid + 2, end);
    if (target.includes("\n")) return null;
    return { label: text.slice(i + 1, mid), href: safeHref(target.trim().split(/\s+/)[0]), end: end };
  }

  // Inline pieces: {t: "text"|"code"|"strong"|"em"|"link"|"br", v?, c?, href?}.
  function inline(text, depth) {
    const level = depth || 0;
    if (level > MAX_DEPTH) return [{ t: "text", v: text }];
    const next = finder(text);
    const out = [];
    let plain = "";
    const flush = () => { if (plain) out.push({ t: "text", v: plain }); plain = ""; };
    let i = 0;
    while (i < text.length) {
      const ch = text[i];
      if (ch === "\\" && ESCAPABLE.includes(text[i + 1] || "\u0000")) { plain += text[i + 1]; i += 2; continue; }
      if (ch === "\n") { flush(); out.push({ t: "br" }); i += 1; continue; }
      if (ch === "`") {
        TICKS.lastIndex = i;
        const ticks = TICKS.exec(text)[0];
        const end = next(ticks, i + ticks.length);
        if (end > 0 && text[end + ticks.length] !== "`") {
          flush();
          out.push({ t: "code", v: text.slice(i + ticks.length, end).replace(/^ (.+) $/, "$1") });
          i = end + ticks.length;
          continue;
        }
        plain += ticks;
        i += ticks.length;
        continue;
      }
      if ((ch === "*" || ch === "_") && text[i + 1] === ch) {
        const end = next(ch + ch, i + 2);
        if (end > i + 2 && (ch === "*" || (!wordChar(text[i - 1]) && !wordChar(text[end + 2])))) {
          flush();
          out.push({ t: "strong", c: inline(text.slice(i + 2, end), level + 1) });
          i = end + 2;
          continue;
        }
      } else if ((ch === "*" || ch === "_") && text[i + 1] && text[i + 1] !== " ") {
        const end = next(ch, i + 1);
        if (end > i + 1 && text[end - 1] !== " " && (ch === "*" || (!wordChar(text[i - 1]) && !wordChar(text[end + 1])))) {
          flush();
          out.push({ t: "em", c: inline(text.slice(i + 1, end), level + 1) });
          i = end + 1;
          continue;
        }
      }
      if (ch === "[") {
        const link = linkAt(text, i, next);
        if (link) {
          flush();
          const label = inline(link.label, level + 1);
          if (link.href) out.push({ t: "link", href: link.href, c: label.length ? label : [{ t: "text", v: link.href }] });
          else Array.prototype.push.apply(out, label); // unsafe or relative target: the label only
          i = link.end + 1;
          continue;
        }
      }
      if ((ch === "h" || ch === "H") && !wordChar(text[i - 1])) {
        BARE_URL.lastIndex = i;
        const bare = BARE_URL.exec(text);
        if (bare) {
          // A refused URL (control or format characters) is plain text, skipped whole, so the
          // next `h` inside it does not scan the same run again.
          if (safeHref(bare[0])) {
            flush();
            out.push({ t: "link", href: bare[0], c: [{ t: "text", v: bare[0] }] });
          } else plain += bare[0];
          i += bare[0].length;
          continue;
        }
      }
      plain += ch;
      i += 1;
    }
    flush();
    return out;
  }

  // ---- DOM -------------------------------------------------------------------------------

  function textOf(pieces) {
    return pieces.map((p) => (p.t === "text" || p.t === "code" ? p.v : p.t === "br" ? "\n" : textOf(p.c || []))).join("");
  }

  function inlineNodes(parent, pieces) {
    pieces.forEach((piece) => {
      if (piece.t === "text") parent.appendChild(document.createTextNode(piece.v));
      else if (piece.t === "br") parent.appendChild(document.createElement("br"));
      else if (piece.t === "code") parent.appendChild(AA.el("code", { text: piece.v }));
      else if (piece.t === "link") {
        // The title shows where a link goes when its label says something else.
        const title = textOf(piece.c) !== piece.href ? piece.href : null;
        const link = AA.el("a", { href: piece.href, target: "_blank", rel: "noopener noreferrer", title: title });
        inlineNodes(link, piece.c);
        parent.appendChild(link);
      } else {
        const node = document.createElement(piece.t);
        inlineNodes(node, piece.c);
        parent.appendChild(node);
      }
    });
    return parent;
  }

  function codeNode(block, options) {
    const head = AA.el("div", { class: "md-code-head" }, [AA.el("span", { class: "muted", text: block.lang || "code" })]);
    if (!block.open && options.copy) {
      const button = AA.el("button", { type: "button", class: "ghost small", text: "copy", "aria-label": "copy code" });
      button.addEventListener("click", () => options.copy(block.code, button));
      head.appendChild(button);
    }
    return AA.el("div", { class: "md-code" }, [head, AA.el("pre", null, AA.el("code", { text: block.code }))]);
  }

  function listNode(block) {
    const root = AA.el(block.ordered ? "ol" : "ul");
    const first = block.items[0];
    if (block.ordered && first && first.start !== null && first.start !== 1) root.setAttribute("start", String(first.start));
    block.items.forEach((item) => root.appendChild(inlineNodes(AA.el("li", { class: item.depth ? "md-d" + item.depth : null }), inline(item.text))));
    return root;
  }

  function tableNode(block) {
    const cell = (tag, text, i) => inlineNodes(AA.el(tag, { class: block.align[i] ? "md-" + block.align[i] : null }), inline(text));
    const head = AA.el("tr", null, block.head.map((text, i) => cell("th", text, i)));
    const rows = block.rows.map((row) => AA.el("tr", null, block.head.map((_, i) => cell("td", row[i] || "", i))));
    return AA.el("div", { class: "md-table" }, AA.el("table", null, [AA.el("thead", null, head), AA.el("tbody", null, rows)]));
  }

  function blockNode(block, options) {
    if (block.type === "code") return codeNode(block, options);
    if (block.type === "list") return listNode(block);
    if (block.type === "table") return tableNode(block);
    if (block.type === "rule") return AA.el("hr");
    if (block.type === "heading") return inlineNodes(AA.el("p", { class: "md-h md-h" + Math.min(block.level, 3) }), inline(block.text));
    if (block.type === "quote") return inlineNodes(AA.el("blockquote"), inline(block.text));
    return inlineNodes(AA.el("p"), inline(block.text));
  }

  // A text too long to lay out as Markdown each frame: one plain block, updated in place.
  function updateRaw(container, text) {
    let box = container.firstChild;
    if (!box || box.mdKey !== "raw" || container.childNodes.length !== 1) {
      while (container.firstChild) container.removeChild(container.firstChild);
      box = AA.el("div", { class: "md-raw" }, [AA.el("p", { class: "muted" }), AA.el("pre")]);
      box.mdKey = "raw";
      container.appendChild(box);
    }
    box.firstChild.textContent = "A long answer (" + Math.round(text.length / 1024) + " KB): shown as plain text.";
    box.lastChild.textContent = text;
    return container;
  }

  // Renders `text` into `container`, keeping the nodes of blocks that did not change (streaming).
  function update(container, text, options) {
    const opts = options || {};
    const source = String(text || "");
    if (source.length > MAX_RENDER || opts.raw) return updateRaw(container, source);
    const blocks = parse(source);
    const kept = Array.prototype.slice.call(container.childNodes);
    blocks.forEach((block, i) => {
      const key = block.type + "\u0000" + (block.open ? "open\u0000" : "") + block.src;
      const old = kept[i];
      if (old && old.mdKey === key) return;
      const node = blockNode(block, opts);
      node.mdKey = key;
      if (old) container.replaceChild(node, old);
      else container.appendChild(node);
    });
    for (let i = kept.length - 1; i >= blocks.length; i -= 1) container.removeChild(kept[i]);
    return container;
  }

  AA.markdown = { parse: parse, inline: inline, safeHref: safeHref, update: update, MAX_RENDER: MAX_RENDER };
})();
