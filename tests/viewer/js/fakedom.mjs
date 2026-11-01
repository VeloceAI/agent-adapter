// A tiny DOM, enough to drive the viewer's scripts in Node: elements, text nodes, events,
// simple selectors, and a small parser for the page's own well-formed markup.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import vm from "node:vm";

import { STATIC } from "./load.mjs";

class ClassList {
  constructor(el) { this.el = el; }
  get set() { return new Set((this.el.getAttribute("class") || "").split(/\s+/).filter(Boolean)); }
  write(set) { this.el.setAttribute("class", Array.from(set).join(" ")); }
  contains(name) { return this.set.has(name); }
  add(...names) { const s = this.set; names.forEach((n) => s.add(n)); this.write(s); }
  remove(...names) { const s = this.set; names.forEach((n) => s.delete(n)); this.write(s); }
  toggle(name, on) { const s = this.set; const want = on === undefined ? !s.has(name) : Boolean(on); if (want) s.add(name); else s.delete(name); this.write(s); return want; }
}

class Node {
  constructor(doc) { this.ownerDocument = doc; this.childNodes = []; this.parentNode = null; }
  get firstChild() { return this.childNodes[0] || null; }
  get lastChild() { return this.childNodes[this.childNodes.length - 1] || null; }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
}

export class Text extends Node {
  constructor(data, doc) { super(doc); this.nodeType = 3; this.data = String(data); }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
}

export class Element extends Node {
  constructor(tag, doc) {
    super(doc);
    this.nodeType = 1;
    this.tagName = tag.toUpperCase();
    this.attrs = new Map();
    this.listeners = {};
    this.style = {};
    this.dataset = {};
    this.classList = new ClassList(this);
    this.value = "";
    this.disabled = false;
    this.clientWidth = 800;
    this.clientHeight = 400;
    this.scrollTop = 0;
    this.scrollHeight = 0;
  }
  get children() { return this.childNodes.filter((c) => c instanceof Element); }
  get options() { return this.children.filter((c) => c.tagName === "OPTION"); }
  get id() { return this.getAttribute("id") || ""; }
  set id(v) { this.setAttribute("id", v); }
  get className() { return this.getAttribute("class") || ""; }
  set className(v) { this.setAttribute("class", v); }
  get hidden() { return this.attrs.has("hidden"); }
  set hidden(v) { if (v) this.attrs.set("hidden", ""); else this.attrs.delete("hidden"); }
  get offsetParent() { return this.hidden ? null : this.ownerDocument.body; }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(""); }
  set textContent(v) {
    this.childNodes.forEach((c) => { c.parentNode = null; });
    this.childNodes = [];
    if (String(v) !== "") this.appendChild(new Text(v, this.ownerDocument));
  }
  setAttribute(name, value) { this.attrs.set(name, String(value)); if (name === "value") this.value = String(value); if (name === "disabled") this.disabled = true; }
  getAttribute(name) { return this.attrs.has(name) ? this.attrs.get(name) : null; }
  removeAttribute(name) { this.attrs.delete(name); }
  hasAttribute(name) { return this.attrs.has(name); }
  appendChild(child) {
    if (child.isFragment) { child.childNodes.slice().forEach((c) => this.appendChild(c)); return child; }
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }
  insertBefore(child, ref) {
    if (!ref) return this.appendChild(child);
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.splice(this.childNodes.indexOf(ref), 0, child);
    return child;
  }
  removeChild(child) {
    const at = this.childNodes.indexOf(child);
    if (at < 0) throw new Error("not a child");
    this.childNodes.splice(at, 1);
    child.parentNode = null;
    return child;
  }
  replaceChild(next, old) {
    const at = this.childNodes.indexOf(old);
    if (at < 0) throw new Error("not a child");
    if (next.parentNode) next.parentNode.removeChild(next);
    this.childNodes[this.childNodes.indexOf(old)] = next;
    next.parentNode = this;
    old.parentNode = null;
    return old;
  }
  contains(other) { for (let at = other; at; at = at.parentNode) if (at === this) return true; return false; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn); }
  dispatchEvent(ev) {
    ev.target = ev.target || this;
    for (let at = this; at && !ev.stopped; at = at.parentNode) {
      ev.currentTarget = at;
      (at.listeners[ev.type] || []).slice().forEach((fn) => fn(ev));
    }
    if (!ev.stopped && this.ownerDocument && this !== this.ownerDocument.root) (this.ownerDocument.listeners[ev.type] || []).forEach((fn) => fn(ev));
    return !ev.defaultPrevented;
  }
  dispatch(type, props) { const ev = event(type, props); this.dispatchEvent(ev); return ev; }
  focus() { this.ownerDocument.activeElement = this; }
  blur() { this.ownerDocument.activeElement = this.ownerDocument.body; }
  select() {}
  click() { if (!this.disabled) this.dispatchEvent(event("click")); }
  showModal() { this.setAttribute("open", ""); }
  close() { this.removeAttribute("open"); this.dispatchEvent(event("close")); }
  setPointerCapture() {}
  getBoundingClientRect() { return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight }; }
  matches(selector) { return selector.split(",").some((part) => matchChain(this, part.trim().match(/(?:[^\s[]+|\[[^\]]*\])+/g))); }
  closest(selector) { for (let at = this; at instanceof Element; at = at.parentNode) if (at.matches(selector)) return at; return null; }
  querySelectorAll(selector) {
    const found = [];
    const walk = (el) => el.children.forEach((c) => { if (c.matches(selector)) found.push(c); walk(c); });
    walk(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  // Every element below this one whose tag is `tag` ("*" for all), depth first.
  all(tag) { return this.querySelectorAll(tag === "*" ? "*" : tag); }
}

// A descendant chain such as `#runs-view .tab`: the last part matches `el`, the others ancestors.
function matchChain(el, parts) {
  if (!matchOne(el, parts[parts.length - 1])) return false;
  let at = el.parentNode;
  for (let i = parts.length - 2; i >= 0; i -= 1) {
    while (at instanceof Element && !matchOne(at, parts[i])) at = at.parentNode;
    if (!(at instanceof Element)) return false;
    at = at.parentNode;
  }
  return true;
}

// tag, .class, #id, [attr] or [attr="v"], `*`, combined without spaces (e.g. tr.selected).
function matchOne(el, part) {
  if (part === "*") return true;
  const re = /([.#]?[\w-]+)|\[([\w-]+)(?:="([^"]*)")?\]/g;
  let m;
  while ((m = re.exec(part))) {
    if (m[2]) {
      if (!el.hasAttribute(m[2]) || (m[3] !== undefined && el.getAttribute(m[2]) !== m[3])) return false;
    } else if (m[1][0] === ".") {
      if (!el.classList.contains(m[1].slice(1))) return false;
    } else if (m[1][0] === "#") {
      if (el.id !== m[1].slice(1)) return false;
    } else if (el.tagName !== m[1].toUpperCase()) return false;
  }
  return true;
}

export function event(type, props) {
  return Object.assign({
    type, defaultPrevented: false, stopped: false,
    preventDefault() { this.defaultPrevented = true; }, stopPropagation() { this.stopped = true; },
  }, props || {});
}

const VOID = new Set(["input", "meta", "link", "br", "hr", "img"]);

// Builds nodes from simple, well-formed markup (the page's own HTML; no comments or scripts).
export function parseHtml(doc, markup) {
  const root = new Element("#fragment", doc);
  root.isFragment = true;
  const stack = [root];
  const token = /<\/([\w-]+)\s*>|<([\w-]+)((?:\s+[\w:-]+(?:="[^"]*")?)*)\s*\/?>|([^<]+)/g;
  let m;
  while ((m = token.exec(markup))) {
    if (m[1]) { stack.pop(); continue; }
    if (m[4]) { if (m[4].trim()) stack[stack.length - 1].appendChild(new Text(m[4].replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">"), doc)); continue; }
    const el = new Element(m[2], doc);
    for (const a of m[3].matchAll(/([\w:-]+)(?:="([^"]*)")?/g)) el.setAttribute(a[1], a[2] === undefined ? "" : a[2]);
    stack[stack.length - 1].appendChild(el);
    if (!VOID.has(m[2].toLowerCase())) stack.push(el);
  }
  return root;
}

// Installs the fake document (`hash` is the address fragment; `html` goes into the body).
export function installDom(options) {
  const opts = options || {};
  const doc = {
    listeners: {},
    createElement: (tag) => new Element(tag, doc),
    createElementNS: (ns, tag) => new Element(tag, doc),
    createTextNode: (text) => new Text(text, doc),
    createDocumentFragment: () => Object.assign(new Element("#fragment", doc), { isFragment: true }),
    getElementById: (id) => doc.body.querySelector("#" + id),
    querySelector: (sel) => doc.body.querySelector(sel),
    querySelectorAll: (sel) => doc.body.querySelectorAll(sel),
    addEventListener: (type, fn) => { (doc.listeners[type] = doc.listeners[type] || []).push(fn); },
    removeEventListener: (type, fn) => { doc.listeners[type] = (doc.listeners[type] || []).filter((f) => f !== fn); },
  };
  doc.documentElement = new Element("html", doc);
  doc.body = doc.documentElement.appendChild(new Element("body", doc));
  doc.activeElement = doc.body;
  if (opts.html) doc.body.appendChild(parseHtml(doc, opts.html));
  globalThis.window = globalThis;
  globalThis.document = doc;
  globalThis.location = { hash: opts.hash || "", pathname: "/" };
  globalThis.history = { replaceState() {} };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => "" });
  return doc;
}

export function runScripts(...names) {
  for (const name of names) vm.runInThisContext(readFileSync(join(STATIC, name), "utf8"), { filename: name });
  return globalThis.AA;
}

// Installs the fake document, then loads the named static scripts into this process.
export function loadDom(...names) {
  const options = typeof names[0] === "object" ? names.shift() : {};
  const doc = installDom(options);
  return { AA: runScripts(...names), document: doc };
}

// Markup-like text of a node, to assert on structure: <a href="x">text</a>.
export function html(node) {
  if (node.nodeType === 3) return node.data.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const attrs = Array.from(node.attrs.keys()).sort().map((k) => " " + k + '="' + node.attrs.get(k) + '"').join("");
  const inner = node.childNodes.map(html).join("");
  const tag = node.tagName.toLowerCase();
  return tag === "#fragment" ? inner : "<" + tag + attrs + ">" + inner + "</" + tag + ">";
}
