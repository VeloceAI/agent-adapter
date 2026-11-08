// Shared by the 9:16 and 16:9 cuts. Times are seconds; 120 bpm, so a beat is 0.5 s.
// Every readable line holds at least 0.5 s + 0.3 s a word (the HyperFrames check enforces it).
const END = 30;
const tl = gsap.timeline({ paused: true });
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const enter = (sel, at, from = { y: 50 }) => tl.fromTo(sel, { opacity: 0, ...from }, { opacity: 1, x: 0, y: 0, scale: 1, duration: 0.4, ease: "power3.out" }, at);
const leave = (sel, at) => tl.to(sel, { opacity: 0, y: -40, duration: 0.2, ease: "power2.in" }, at);
const pop = (sel, at) => tl.fromTo(sel, { opacity: 0, scale: 0.6 }, { opacity: 1, scale: 1, duration: 0.45, ease: "back.out(2)" }, at);
// Reveals a line as one text node (so it reads as words, not letters), stepping like typing.
function typed(sel, at, step) {
  const el = $(sel);
  const n = el.textContent.length;
  tl.fromTo(el, { width: 0 }, { width: el.scrollWidth, duration: n * step, ease: `steps(${n})` }, at);
}
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1;

tl.fromTo("#grid", { y: 0 }, { y: 112, duration: END, ease: "none" }, 0);
tl.fromTo("#glow-a", { x: 0 }, { x: 260, duration: END, ease: "sine.inOut" }, 0);
tl.set("#flash", { opacity: 0 }, 0);
[7.5, 23].forEach((at) => {
  tl.to("#flash", { opacity: 0.9, duration: 0.06 }, at);
  tl.to("#flash", { opacity: 0, duration: 0.9, ease: "power2.out" }, at + 0.06);
});

// 1. Hook: on screen from frame 0 (the thumbnail); the call types, then is held.
typed("#call", 0.4, 0.026);
tl.fromTo("#stamp", { opacity: 0, scale: 2.4, rotation: -16 }, { opacity: 1, scale: 1, rotation: -6, duration: 0.18, ease: "power4.in" }, 1.62); // lands 1.8
tl.fromTo("#s1-in", { scale: 1 }, { scale: 1.035, duration: 0.08, yoyo: true, repeat: 1, ease: "power2.out" }, 1.8);
tl.fromTo("#term1", { x: 0 }, { x: 10, duration: 0.05, yoyo: true, repeat: 5, ease: "none" }, 1.8);
leave("#s1-in", 3.3);

// 2. Problem: every framework has its own rules.
$$("#p1 .w").forEach((el, i) => enter(el, 3.55 + i * 0.08, { y: 30 }));
$$("#chips .chip").forEach((el, i) => pop(el, 3.95 + i * 0.06));
enter("#p2", 5.0);
$$("#chips .chip").forEach((el, i) => tl.to(el, { x: rand() * 160, y: rand() * 120, rotation: rand() * 25, opacity: 0.25, duration: 0.6, ease: "power2.out" }, 5.8 + i * 0.01));
leave("#s2-in", 7.3);

// 3. Reveal: what it is, then the one decorator. beat-locked: 7.5 s (score impact)
pop("#brand", 7.5);
enter("#what", 7.8);
enter("#codecard", 9.6, { y: 80 });
$$("#codecard .ln").forEach((el, i) => enter(el, 9.7 + i * 0.1, { x: 40 }));
tl.to("#confirm", { backgroundColor: "rgba(45,212,191,.22)", color: "#2dd4bf", duration: 0.25 }, 11.0);
enter("#note", 11.1, { y: 20 });
leave("#s3-in", 13.3);

// 4. Proof: one tab per card, each held about 2.3 s. beat-grid: 13.8, 16.1, 18.4, 20.7
enter("#every", 13.55);
enter("#tabs", 13.65);
enter("#deck", 13.75, { y: 70 });
const CARDS = [["#c1", "#tab1", 13.8], ["#c2", "#tab2", 16.1], ["#c3", "#tab3", 18.4], ["#c4", "#tab4", 20.7]];
tl.set(CARDS.map((c) => c[0]), { opacity: 0 }, 0);
CARDS.forEach(([card, tab, at], i) => {
  tl.fromTo(card, { opacity: 0, x: 60 }, { opacity: 1, x: 0, duration: 0.35, ease: "power3.out" }, at);
  tl.to(tab, { backgroundColor: "#818cf8", borderColor: "#818cf8", color: "#0b1020", duration: 0.2 }, at);
  if (i < CARDS.length - 1) {
    const next = CARDS[i + 1][2];
    tl.to(card, { opacity: 0, x: -60, duration: 0.25, ease: "power2.in" }, next - 0.25);
    tl.to(tab, { backgroundColor: "rgba(0,0,0,0)", borderColor: "#26304d", color: "#94a3b8", duration: 0.2 }, next);
  }
});
// Approval: the cursor taps Approve at 14.4.
const btn = $("#approve");
const cur = $("#cursor");
cur.style.left = `${btn.offsetLeft + btn.offsetWidth / 2 - 23}px`;
cur.style.top = `${btn.offsetTop + btn.offsetHeight / 2 - 23}px`;
tl.set(["#approved", "#pend-ok"], { opacity: 0 }, 0);
tl.fromTo(cur, { opacity: 0, x: 300, y: 180 }, { opacity: 1, x: 0, y: 0, duration: 0.45, ease: "power2.inOut" }, 13.9);
tl.to(cur, { scale: 0.75, duration: 0.08, yoyo: true, repeat: 1 }, 14.4);
tl.to(btn, { scale: 0.94, duration: 0.08, yoyo: true, repeat: 1 }, 14.4);
tl.to("#pend", { opacity: 0, duration: 0.15 }, 14.45);
tl.to("#pend-ok", { opacity: 1, duration: 0.15 }, 14.45);
enter("#approved", 14.45, { y: 20 });
tl.to(cur, { opacity: 0, duration: 0.3 }, 14.75);
// Budgets: the team meter fills and the call stops at 16.65.
const spent = { v: 0 };
tl.fromTo("#fill", { scaleX: 0 }, { scaleX: 1, duration: 0.45, ease: "power1.in" }, 16.2);
tl.fromTo(spent, { v: 0 }, { v: 50, duration: 0.45, ease: "power1.in", onUpdate: () => ($("#spent").textContent = `$${Math.round(spent.v)} / $50`) }, 16.2);
enter("#stopped", 16.65, { y: 20 });
tl.fromTo("#deck", { x: 0 }, { x: 8, duration: 0.05, yoyo: true, repeat: 3, ease: "none" }, 16.65);
leave("#s4-in", 22.8);

// 5. Tagline. beat-locked: 23 s (score impact)
enter("#tg1", 23.0, { y: 60 });
enter("#tg2", 23.5, { y: 60 });
tl.fromTo("#s5-in", { scale: 1 }, { scale: 1.03, duration: 2.8, ease: "none" }, 23.0);
leave("#s5-in", 25.8);

// 6. Call to action, held to the end.
pop("#brand2", 26.05);
enter("#term2", 26.2);
typed("#pip", 26.3, 0.028);
enter("#oss", 27.0, { y: 30 });
enter("#url", 27.4, { y: 30 });

window.__timelines = window.__timelines || {};
window.__timelines["main"] = tl;
tl.seek(0);
