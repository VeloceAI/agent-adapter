# Launch reel: sources and how to rebuild

A 30 s reel in two cuts, 9:16 (Instagram, Shorts, LinkedIn, X) and 16:9 (docs, README), built with
HyperFrames from `src/` by `node build.mjs`. The plan and storyboard are in `brag-plan.md`.

| Item | Source | Licence |
|---|---|---|
| Score | synthesised from `cues.json` by the VeloceAI video kit's `audio/score.py` | Veloce AI, own work |
| Sounds in `assets/sfx/` | Kenney Interface Sounds and Impact Sounds (kenney.nl) | CC0 1.0 |
| Inter Tight, Geist Mono | Google Fonts / Vercel, copied from the video kit (not tracked here) | SIL OFL 1.1 |
| Mark | `docs/assets/mark-dark.svg` | Apache-2.0 (this repo) |
| GSAP 3.14.2 | loaded from jsdelivr at render time | GSAP Standard "No Charge" licence |
| Trace rows | the docs' sample run | this repo |

Numbers on the Cost card are illustrative and the card says "sample data". All code on screen runs on 1.1.0.

Rebuild: copy the two fonts into `assets/fonts/`, run `score.py cues.json assets/music/score.wav`, then
`node build.mjs`, `npx hyperframes@0.8.202 check` and `render` in `composition/` and `composition-wide/`.
