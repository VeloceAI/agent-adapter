// Reel v4. Every visual lands on the word that names it; word times come from the voice transcript.
const END = 29;
const tl = gsap.timeline({ paused: true });
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const pop = (sel, at, from = { y: 50 }) => tl.fromTo(sel, { opacity: 0, ...from }, { opacity: 1, x: 0, y: 0, scale: 1, rotation: 0, duration: 0.3, ease: "power3.out" }, at);
const drop = (sel, at) => tl.fromTo(sel, { opacity: 0, y: -140, scale: 1.08 }, { opacity: 1, y: 0, scale: 1, duration: 0.42, ease: "bounce.out" }, at);
const punch = (sel, at, s = 1.08) => tl.fromTo(sel, { scale: 1 }, { scale: s, duration: 0.07, yoyo: true, repeat: 1, ease: "power2.out" }, at);
const show = (sel, from, to) => { tl.set(sel, { opacity: 0 }, 0); tl.set(sel, { opacity: 1 }, from); if (to) tl.set(sel, { opacity: 0 }, to); };
// Whip: the old scene flies out blurred, the new one flies in.
const CUTS = [["#s1", "#s2", 3.2], ["#s2", "#s3", 4.8], ["#s3", "#s4", 8.16], ["#s4", "#s5", 13.84], ["#s5", "#s6", 18.88], ["#s6", "#s7", 21.28], ["#s7", "#s8", 23.68]];
CUTS.forEach(([a, b, at], i) => {
  const dir = i % 2 ? -1 : 1;
  tl.to(`${a} .wrap`, { x: -280 * dir, filter: "blur(16px)", opacity: 0, duration: 0.14, ease: "power2.in" }, at - 0.14);
  tl.fromTo(`${b} .wrap`, { x: 280 * dir, filter: "blur(16px)", opacity: 0 }, { x: 0, filter: "blur(0px)", opacity: 1, duration: 0.22, ease: "power3.out" }, at);
});
[3.2, 23.68].forEach((at) => { tl.set("#flash", { opacity: 0.7 }, at); tl.to("#flash", { opacity: 0, duration: 0.18, ease: "power2.out" }, at); });
// The glow changes colour with each section.
[[3.2, "rgba(99,102,241,.34)"], [4.8, "rgba(45,212,191,.26)"], [8.16, "rgba(99,102,241,.34)"], [13.84, "rgba(168,85,247,.32)"],
  [18.88, "rgba(45,212,191,.28)"], [21.28, "rgba(245,158,11,.26)"], [23.68, "rgba(99,102,241,.36)"]]
  .forEach(([at, c]) => tl.to("#glow", { backgroundColor: c, duration: 0.3 }, at));

// 1. Hook: the same tool, rewritten three times. Frame 0 already reads.
tl.fromTo("#f1", { x: -40, rotation: -2 }, { x: 0, rotation: -2, duration: 0.3, ease: "power3.out" }, 0);
tl.fromTo("#f2", { opacity: 0, x: 120, rotation: 3 }, { opacity: 1, x: 0, rotation: 2, duration: 0.25, ease: "power3.out" }, 0.48);
tl.fromTo("#f3", { opacity: 0, x: -120, rotation: -3 }, { opacity: 1, x: 0, rotation: -1, duration: 0.25, ease: "power3.out" }, 1.28);
punch("#hookline", 1.84, 1.05);
tl.fromTo("#strike-p", { strokeDasharray: 920, strokeDashoffset: 920 }, { strokeDashoffset: 0, duration: 0.2, ease: "power3.in" }, 2.5);
tl.to(["#f1", "#f2", "#f3"], { filter: "grayscale(1) brightness(.55)", duration: 0.2 }, 2.6);
tl.fromTo("#s1 .wrap", { x: 0 }, { x: 16, duration: 0.04, yoyo: true, repeat: 5, ease: "none" }, 2.7);

// 2. One function.
pop("#fn", 3.25, { y: 90 });
punch("#fn", 3.52, 1.06);
tl.fromTo("#deco", { color: "#79c0ff" }, { color: "#2dd4bf", duration: 0.2 }, 3.76);

// 3. Agent Adapter turns it into a tool for every framework.
tl.fromTo("#fn2", { scale: 1 }, { scale: 0.62, duration: 0.4, ease: "power3.inOut" }, 4.85);
tl.fromTo("#beam", { xPercent: -100 }, { xPercent: 340, duration: 0.55, ease: "power1.inOut" }, 5.9);
const TILES = [["OpenAI", 80, 640], ["Anthropic", 365, 640], ["Gemini", 650, 640], ["LangChain", 80, 800], ["CrewAI", 365, 800], ["MCP", 650, 800]];
$("#tiles").innerHTML = TILES.map(([n, x, y], i) => `<div class="tile" id="t${i}" style="left:${x}px;top:${y}px"><b>${n}</b><span>&check; tool</span></div>`).join("");
$("#links").innerHTML = TILES.map(([, x, y], i) =>
  `<path id="ln${i}" d="M490 525 C 490 590, ${x + 125} ${y - 70}, ${x + 125} ${y}" stroke="#2dd4bf" stroke-width="4" fill="none" stroke-opacity=".8"/>`).join("");
TILES.forEach((_, i) => {
  tl.fromTo(`#ln${i}`, { strokeDasharray: 600, strokeDashoffset: 600 }, { strokeDashoffset: 0, duration: 0.3, ease: "power2.out" }, 6.55 + i * 0.06);
  tl.fromTo(`#t${i}`, { opacity: 0, scale: 0.5 }, { opacity: 1, scale: 1, duration: 0.3, ease: "back.out(2.4)" }, 6.7 + i * 0.08);
});
pop("#count", 7.1, { y: 30 });
const cnt = { v: 6 };
tl.fromTo(cnt, { v: 6 }, { v: 23, duration: 0.8, ease: "power2.out", onUpdate: () => ($("#count-n").textContent = String(Math.round(cnt.v))) }, 7.2);
punch("#count", 8.0, 1.1);

// 4. One line plugs it in; the name follows the voice.
pop("#att", 8.2, { y: 60 });
const FW = [["langchain", "LangChain", 9.28], ["crewai", "CrewAI", 10.24], ["openai_agents", "OpenAI Agents", 11.28], ["mcp", "MCP", 12.96]];
FW.forEach(([code, name, at], i) => {
  if (i) tl.set("#fw", { textContent: code }, at);
  tl.set("#fwbig", { textContent: name }, at);
  tl.fromTo("#fwbig", { opacity: 0, y: 70, filter: "blur(10px)", scale: 1.15 }, { opacity: 1, y: 0, filter: "blur(0px)", scale: 1, duration: 0.25, ease: "power3.out" }, at);
  tl.fromTo("#fw", { color: "#2dd4bf" }, { color: "#a5d6ff", duration: 0.4 }, at);
});
tl.set("#fwbig", { opacity: 0 }, 0);
pop("#native", 9.5, { y: 20 });

// 5. Backends drop in as they are named.
[["#bt1", 15.68], ["#bt2", 16.72], ["#bt3", 17.36], ["#bt4", 18.2]].forEach(([s, at]) => drop(s, at));
pop("#brow", 18.35, { y: 20 });

// 6. Every call flows through one safety layer.
const SRC = [["LangChain", 230, 420], ["CrewAI", 490, 380], ["MCP", 760, 420], ["FastAPI", 230, 560], ["Celery", 490, 600], ["Lambda", 760, 560]];
const GX = 490, GY = 820;
$("#rails").innerHTML = SRC.map(([, x, y], i) => `<path id="r${i}" d="M${x} ${y + 26} C ${x} ${y + 180}, ${GX} ${GY - 160}, ${GX} ${GY} L ${GX} 1170" stroke="#818cf8" stroke-opacity=".4" stroke-width="3" fill="none"/>`).join("")
  + Array.from({ length: 24 }, (_, i) => `<circle id="d${i}" r="9" cx="-50" cy="-50" fill="#2dd4bf" style="filter: drop-shadow(0 0 8px #2dd4bf)"/>`).join("");
$("#chips").innerHTML = SRC.map(([n, x, y], i) => `<div class="chip" id="ch${i}" style="left:${x}px;top:${y}px">${n}</div>`).join("");
SRC.forEach((_, i) => tl.fromTo(`#ch${i}`, { opacity: 0, scale: 0.5 }, { opacity: 1, scale: 1, duration: 0.25, ease: "back.out(2)" }, 19.0 + i * 0.05));
pop("#gate", 19.05, { y: 40, scale: 0.9 });
const paths = SRC.map((_, i) => $(`#r${i}`));
const LEN = paths.map((p) => p.getTotalLength());
let seed = 7;
const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const DOTS = Array.from({ length: 24 }, (_, i) => ({ el: $(`#d${i}`), k: i % 6, phase: rand(), speed: 0.6 + rand() * 0.4 }));
const flow = { t: 0 };
tl.fromTo(flow, { t: 0 }, { t: 2.1, duration: 2.1, ease: "none", onUpdate: () => DOTS.forEach((d) => {
  const p = flow.t < 0.05 ? -1 : (d.phase + flow.t * d.speed) % 1;
  const pt = p < 0 ? { x: -50, y: -50 } : paths[d.k].getPointAtLength(LEN[d.k] * p);
  d.el.setAttribute("cx", pt.x); d.el.setAttribute("cy", pt.y);
}) }, 19.15);
tl.fromTo("#gate", { boxShadow: "0 0 60px rgba(45,212,191,.45)" }, { boxShadow: "0 0 140px rgba(45,212,191,.95)", duration: 0.15, yoyo: true, repeat: 1 }, 20.32);
punch("#gate", 20.32, 1.06);

// 7. Four fast cuts, one per word.
show("#ft1", 21.28, 21.92); show("#ft2", 21.92, 22.4); show("#ft3", 22.4, 22.88); show("#ft4", 22.88);
tl.fromTo("#v-appr", { scale: 0.92 }, { scale: 1.04, duration: 0.64, ease: "power2.out" }, 21.28);
tl.fromTo("#drain", { scaleX: 1 }, { scaleX: 0.03, duration: 0.3, ease: "power2.in" }, 21.95);
show("#stop", 22.22);
punch("#stop", 22.22, 1.2);
tl.fromTo("#im-cost img", { scale: 1 }, { scale: 1.14, duration: 0.48, ease: "none" }, 22.4);
tl.fromTo("#im-trace img", { scale: 1.12 }, { scale: 1, duration: 0.8, ease: "power2.out" }, 22.88);
["#ft1", "#ft2", "#ft3", "#ft4"].forEach((s, i) => punch(`${s} .big`, [21.28, 21.92, 22.4, 22.88][i], 1.1));

// 8. End card, held: the loop line echoes the hook.
punch("#once", 23.75, 1.06);
tl.fromTo("#emark", { opacity: 0, scale: 0.4, rotation: -25 }, { opacity: 1, scale: 1, rotation: 0, duration: 0.45, ease: "back.out(2)" }, 23.9);
pop("#term", 24.35, { y: 30 });
tl.fromTo("#pip", { width: 0 }, { width: () => $("#pip").scrollWidth, duration: 1.3, ease: "steps(25)" }, 24.56);
pop("#gh", 26.0, { y: 20 });
pop("#docs", 26.3, { y: 20 });
pop("#eos", 26.6, { y: 20 });

// Karaoke captions: groups of up to three words, broken after punctuation; hidden on the end card.
const WORDS = %%WORDS%%.filter((w) => w.t < 23.6);
const GROUPS = [];
WORDS.forEach((w) => {
  const g = GROUPS[GROUPS.length - 1];
  if (!g || g.words.length >= 3 || /[.,?]$/.test(g.words[g.words.length - 1].w)) GROUPS.push({ words: [w] });
  else g.words.push(w);
});
GROUPS.forEach((g, i) => { g.start = i ? g.words[0].t : 0; g.end = GROUPS[i + 1] ? GROUPS[i + 1].words[0].t : 23.6; });
const cap = $("#cap");
let shown = -2;
const clock = { t: 0 };
tl.fromTo(clock, { t: 0 }, { t: END, duration: END, ease: "none", onUpdate: () => {
  const t = clock.t;
  const gi = GROUPS.findIndex((g) => g.start <= t && t < g.end);
  if (gi !== shown) {
    cap.innerHTML = gi < 0 ? "" : GROUPS[gi].words.map((w) => `<span>${w.w}</span>`).join(" ");
    shown = gi;
  }
  if (gi < 0) return;
  const spans = cap.children;
  GROUPS[gi].words.forEach((w, k) => {
    const next = GROUPS[gi].words[k + 1];
    spans[k].className = t >= w.t ? (next && t >= next.t ? "on" : "now") : "";
  });
  const k = Math.min(1, (t - GROUPS[gi].start) / 0.1);
  cap.style.transform = `scale(${0.88 + 0.12 * (1 - (1 - k) ** 3)})`;
} }, 0);

window.__timelines = window.__timelines || {};
window.__timelines["main"] = tl;
tl.seek(0);
