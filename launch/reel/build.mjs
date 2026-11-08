// Writes composition/ (9:16) from src/, assets/ and the word times in assets/voice/transcript.json. Run: node build.mjs
import fs from "node:fs";

const at = (f) => new URL(f, import.meta.url);
const read = (f) => fs.readFileSync(at(f), "utf8");
const mark = read("../../docs/assets/mark-dark.svg").replace(/<title>.*?<\/title>/, "").replace("<svg ", '<svg width="100%" height="100%" ');
// Word times from the transcript, spelled as the script spells them.
const FIX = { "Langchain,": "LangChain,", "Django": "Django,", "at": "it" };
const words = [];
for (const { text, start } of JSON.parse(read("assets/voice/transcript.json"))) {
  const prev = words[words.length - 1];
  if (prev && prev.w === "Crew" && text === "AI,") prev.w = "CrewAI,";
  else words.push({ w: FIX[text] || text, t: Math.round(start * 1000) / 1000 });
}
// Effects: a whoosh on every cut, a tick on every named landing, an impact on the big beats.
const SFX = [
  ...[3.2, 4.8, 8.16, 13.84, 18.88, 21.28, 23.68].map((t) => ["synth_whoosh", t - 0.12, 0.45, 0.35]),
  ...[0.48, 1.28, 6.7, 6.78, 6.86, 6.94, 7.02, 7.1, 9.28, 10.24, 11.28, 12.96, 15.68, 16.72, 17.36, 18.2, 21.92, 22.4, 22.88].map((t) => ["synth_tick", t, 0.07, 0.5]),
  ["synth_impact_small", 2.5, 1.2, 0.45], ["synth_impact_small", 20.32, 1.2, 0.35], ["synth_impact", 23.68, 2.0, 0.4], ["glass_ping", 26.0, 0.12, 0.45],
];
const sfx = SFX.map(([f, t, d, v], i) => `      <audio id="fx${i}" src="assets/sfx/${f}.wav" data-start="${t.toFixed(2)}" data-duration="${d}" data-track-index="${12 + (i % 6)}" data-volume="${v}"></audio>`).join("\n");
const HF = "0.8.202";
const END = 29;
const dir = "composition";

fs.rmSync(at(`${dir}/assets`), { recursive: true, force: true });
fs.mkdirSync(at(dir), { recursive: true });
fs.cpSync(at("assets"), at(`${dir}/assets`), { recursive: true });
fs.writeFileSync(at(`${dir}/hyperframes.json`), JSON.stringify({
  $schema: "https://hyperframes.heygen.com/schema/hyperframes.json",
  paths: { assets: "assets" }, media: { autoProxy: true },
}, null, 2) + "\n");
fs.writeFileSync(at(`${dir}/package.json`), JSON.stringify({
  name: dir, private: true, type: "module",
  scripts: { check: `npx --yes hyperframes@${HF} check`, render: `npx --yes hyperframes@${HF} render` },
}, null, 2) + "\n");
fs.writeFileSync(at(`${dir}/index.html`), `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1080, height=1920" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
${read("src/style.css")}    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="${END}" data-width="1080" data-height="1920">
${read("src/body.html").replaceAll("%%MARK%%", mark).replace("%%SFX%%", sfx)}    </div>
    <script>
${read("src/scenes.js").replace("%%WORDS%%", JSON.stringify(words))}    </script>
  </body>
</html>
`);
console.log("wrote", dir);
