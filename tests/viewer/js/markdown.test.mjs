// markdown.js: the safe subset, built as DOM nodes; hostile input stays text.
import assert from "node:assert/strict";
import { test } from "node:test";

import { html, loadDom } from "./fakedom.mjs";

const { AA, document: doc } = loadDom("data.js", "markdown.js");
const render = (text, options) => AA.markdown.update(doc.createElement("div"), text, options);
const tags = (node) => node.all("*").map((n) => n.tagName.toLowerCase());

test("paragraphs, bold, italic, inline code and line breaks", () => {
  const out = html(render("Hello **bold** and *it* or _it_ with `x < y`\nnext line\n\nsecond"));
  assert.equal(out, "<div><p>Hello <strong>bold</strong> and <em>it</em> or <em>it</em> with <code>x &lt; y</code><br></br>next line</p><p>second</p></div>");
});

test("snake_case and lone stars stay plain text", () => {
  assert.equal(html(render("call my_tool_name with 2 * 3 * 4")), "<div><p>call my_tool_name with 2 * 3 * 4</p></div>");
  assert.equal(html(render("\\*not em\\*")), "<div><p>*not em*</p></div>");
});

test("lists: bullets, numbers with a start, nesting and wrapped lines", () => {
  const out = html(render("- one\n- two\n  - inner\n  more\n\n3. c\n4. d"));
  assert.equal(out, '<div><ul><li>one</li><li>two</li><li class="md-d1">inner<br></br>more</li></ul><ol start="3"><li>c</li><li>d</li></ol></div>');
});

test("tables with alignment and escaped pipes", () => {
  const out = html(render("| a | b |\n|:--|--:|\n| 1 | x \\| y |\n| 2 |"));
  assert.equal(out, '<div><div class="md-table"><table><thead><tr><th class="md-left">a</th><th class="md-right">b</th></tr></thead>'
    + '<tbody><tr><td class="md-left">1</td><td class="md-right">x | y</td></tr><tr><td class="md-left">2</td><td class="md-right"></td></tr></tbody></table></div></div>');
});

test("code blocks keep HTML as text and get a copy button once closed", () => {
  const copied = [];
  const node = render("```html\n<script>alert(1)</script>\n<img src=x onerror=alert(1)>\n```", { copy: (text) => copied.push(text) });
  assert.ok(!tags(node).includes("script") && !tags(node).includes("img"));
  const code = node.all("code")[0];
  assert.equal(code.textContent, "<script>alert(1)</script>\n<img src=x onerror=alert(1)>");
  node.all("button")[0].click();
  assert.deepEqual(copied, [code.textContent]);
  const open = render("```py\nprint(1)", { copy: () => {} }); // still streaming: no copy yet
  assert.equal(open.all("button").length, 0);
  assert.equal(open.all("code")[0].textContent, "print(1)");
});

test("raw HTML anywhere is text, never an element", () => {
  const node = render('<script>alert(1)</script> <img src=x onerror="alert(1)"> **<b onclick=x>hi</b>**\n\n| <i>x</i> |\n|--|\n| <svg onload=1> |');
  assert.deepEqual(tags(node).filter((t) => !["p", "strong", "div", "table", "thead", "tbody", "tr", "th", "td"].includes(t)), []);
  assert.ok(node.textContent.includes("<script>alert(1)</script>"));
  assert.ok(node.textContent.includes('onerror="alert(1)"'));
  node.all("*").forEach((el) => assert.deepEqual(Array.from(el.attrs.keys()).filter((a) => a.startsWith("on")), []));
});

test("links: only http(s), opened safely; others keep their label as text", () => {
  const ok = render("see [docs](HTTPS://example.org/a?b=1) and https://example.org/x.");
  const links = ok.all("a");
  assert.equal(links.length, 2);
  assert.deepEqual(links.map((a) => [a.getAttribute("href"), a.getAttribute("rel"), a.getAttribute("target")]), [
    ["HTTPS://example.org/a?b=1", "noopener noreferrer", "_blank"],
    ["https://example.org/x", "noopener noreferrer", "_blank"],
  ]);
  assert.ok(ok.textContent.endsWith("x."), "the trailing dot is not part of the link");
  for (const bad of ["javascript:alert(1)", "JaVaScRiPt:alert(1)", " javascript:alert(1)", "data:text/html,<script>1</script>", "vbscript:x", "/relative", "//example.org", "http:\tx"]) {
    const node = render("[click](" + bad + ")");
    assert.equal(node.all("a").length, 0, bad);
    assert.equal(node.textContent, "click", bad);
  }
  assert.equal(AA.markdown.safeHref("https://a.b/\u0000x"), "");
  assert.equal(AA.markdown.safeHref("http://a.b"), "http://a.b");
});

test("headings, quotes and rules are plain styled blocks", () => {
  assert.equal(html(render("## Title\n> quoted\n\n---")), '<div><p class="md-h md-h2">Title</p><blockquote>quoted</blockquote><hr></hr></div>');
});

test("update keeps unchanged block nodes while text streams in", () => {
  const box = doc.createElement("div");
  AA.markdown.update(box, "first paragraph\n\nsecond");
  const first = box.childNodes[0];
  AA.markdown.update(box, "first paragraph\n\nsecond and more\n\n- a");
  assert.equal(box.childNodes[0], first, "the finished block was not rebuilt");
  assert.equal(box.childNodes.length, 3);
  assert.equal(box.childNodes[1].textContent, "second and more");
  AA.markdown.update(box, "first paragraph");
  assert.equal(box.childNodes.length, 1);
  assert.equal(box.childNodes[0], first);
});

test("empty and odd input never throws", () => {
  for (const text of ["", null, undefined, "\n\n\n", "**", "`", "[x](", "| a |", "```", "1. ", "~~~\n~~~"]) render(text);
  assert.equal(render("").childNodes.length, 0);
});

test("links whose label differs show their target as a title; bidi characters are refused", () => {
  const node = render("[docs](https://example.org/a) and [https://example.org/b](https://example.org/b)");
  const [named, plainLink] = node.all("a");
  assert.equal(named.getAttribute("title"), "https://example.org/a");
  assert.equal(plainLink.getAttribute("title"), null);
  for (const mark of ["‮", "​", "⁦"]) {
    assert.equal(AA.markdown.safeHref("https://example.org/" + mark + "gpj.exe"), "", JSON.stringify(mark));
  }
});

test("headings drop a closing # run only after a space", () => {
  const texts = ["# Title ##", "## C# ", "### a #b", "#", "# x\t#"].map((t) => AA.markdown.parse(t)[0]);
  assert.deepEqual(texts.map((b) => [b.type, b.text]), [["heading", "Title"], ["heading", "C#"], ["heading", "a #b"], ["heading", ""], ["heading", "x"]]);
  assert.equal(AA.markdown.parse("#hashtag")[0].type, "para");
});

// Each case is a shape that used to backtrack or rescan; all must stay linear.
const SLOW = {
  "heading with 10,000 inner spaces": "# a" + " ".repeat(10000) + "b",
  "8 KB heading of spaces and hashes": "# " + "a #".repeat(2730),
  "80 KB run of [a](": "[a](".repeat(20000),
  "80 KB run of [\](": "[\](".repeat(20000),
  "80 KB run of unclosed ``a`": "``a`".repeat(20000),
  "table delimiter with 20,000 spaces": "| a |\n|-" + " ".repeat(20000) + "x",
  "rule-like line with 20,000 spaces": "-" + " ".repeat(20000) + "x",
  "80 KB of ** and _ openers": "**a _b ".repeat(11000),
  "a URL of 40,000 dots": "see https://a" + ".".repeat(40000) + " now",
  "180 KB of refused URLs (zero-width spaces)": "(http://​".repeat(20000),
  "refused URLs with control characters": "(http://\u0001".repeat(20000),
  "refused URLs with bidi marks": "(http://‮".repeat(20000),
};

test("a refused bare URL stays plain text, whole", () => {
  const node = render("see https://exa​mple.org/x and https://ok.example.org");
  assert.deepEqual(node.all("a").map((a) => a.getAttribute("href")), ["https://ok.example.org"]);
  assert.ok(node.textContent.includes("https://exa​mple.org/x and"));
});

for (const [name, text] of Object.entries(SLOW)) {
  test("worst case stays fast: " + name, () => {
    const started = performance.now();
    render(text);
    const ms = performance.now() - started;
    assert.ok(ms < 100, name + " took " + ms.toFixed(1) + " ms");
  });
}

test("a text over the render cap is one plain block, updated in place", () => {
  const box = doc.createElement("div");
  const big = "**x** ".repeat(AA.markdown.MAX_RENDER / 6 + 10);
  AA.markdown.update(box, big);
  assert.equal(box.childNodes.length, 1);
  const raw = box.firstChild;
  assert.equal(raw.querySelector("pre").textContent, big);
  assert.equal(raw.all("strong").length, 0);
  AA.markdown.update(box, big + "more");
  assert.equal(box.firstChild, raw, "the same block, new text");
  assert.ok(raw.querySelector("pre").textContent.endsWith("more"));
  AA.markdown.update(box, "short **again**");
  assert.equal(box.all("strong").length, 1, "back to Markdown under the cap");
});
