// Writes composition/ (9:16) and composition-wide/ (16:9) from src/ and assets/. Run: node build.mjs
import fs from "node:fs";

const at = (f) => new URL(f, import.meta.url);
const read = (f) => fs.readFileSync(at(f), "utf8");
const mark = read("../../docs/assets/mark-dark.svg").replace(/<title>.*?<\/title>/, "");
const body = read("src/body.html").replaceAll("%%MARK%%", mark);
const base = read("src/base.css");
const scenes = read("src/scenes.js");
const HF = "0.8.202";
const FORMATS = {
  composition: { w: 1080, h: 1920, css: read("src/tall.css") },
  "composition-wide": { w: 1920, h: 1080, css: read("src/wide.css") },
};

for (const [dir, f] of Object.entries(FORMATS)) {
  const out = at(`${dir}/`);
  fs.mkdirSync(out, { recursive: true });
  fs.cpSync(at("assets"), at(`${dir}/assets`), { recursive: true });
  fs.writeFileSync(at(`${dir}/hyperframes.json`), JSON.stringify({
    $schema: "https://hyperframes.heygen.com/schema/hyperframes.json",
    registry: "https://raw.githubusercontent.com/heygen-com/hyperframes/main/registry",
    paths: { blocks: "compositions", components: "compositions/components", assets: "assets" },
    media: { autoProxy: true },
  }, null, 2) + "\n");
  fs.writeFileSync(at(`${dir}/package.json`), JSON.stringify({
    name: dir, private: true, type: "module",
    scripts: { check: `npx --yes hyperframes@${HF} check`, render: `npx --yes hyperframes@${HF} render` },
  }, null, 2) + "\n");
  fs.writeFileSync(at(`${dir}/index.html`), `<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=${f.w}, height=${f.h}" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
${base}${f.css}    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="30" data-width="${f.w}" data-height="${f.h}">
${body}    </div>
    <script>
${scenes}    </script>
  </body>
</html>
`);
  console.log("wrote", dir);
}
