# Launch reel: sources and how to rebuild

A 29 s vertical reel (1080x1920, Instagram, Shorts, LinkedIn, X) with a voiceover and word-timed captions,
built with HyperFrames from `src/` by `node build.mjs`. The docs home page and the README show it from
`docs/assets/video/`.

| Item | Source | Licence |
|---|---|---|
| Voiceover | Kokoro TTS (voice `af_heart`), script in `audio/narration.json` | Apache-2.0 model, own script |
| Word times | transcribed from the voiceover by `npx hyperframes transcribe` | generated |
| Score and `synth_*` sounds | synthesised by the VeloceAI video kit's `audio/score.py` from `cues.json` | Veloce AI, own work |
| `glass_ping.wav` | Kenney Interface Sounds (kenney.nl) | CC0 1.0 |
| Product footage | the docs' recorded chat demo (`docs/assets/demo/chat-demo.html`), captured by `capture.py` | this repo |
| Inter Tight, Geist Mono | Google Fonts / Vercel, copied from the video kit (not tracked here) | SIL OFL 1.1 |
| Mark | `docs/assets/mark-dark.svg` | Apache-2.0 (this repo) |
| GSAP 3.14.2 | loaded from jsdelivr at render time | GSAP Standard "No Charge" licence |

All code on screen uses the 1.1.0 API. Editing rules (safe zone y 230-1440 and right 230 px, hook text at frame 1,
every visual on its spoken word, karaoke captions, the end line echoing the hook) follow the MIT-licensed
`instagram-agent-skill` and Artem Novitckii's editing and animation skills; no code was copied.

Rebuild:

1. Copy the two fonts into `assets/fonts/` and run `python capture.py` for `assets/clips/`.
2. With the video kit: `audio/voice.py <this folder>`, then `audio/captions.py <this folder> --transcribe`
   (writes `assets/voice/`), then `audio/score.py cues.json assets/music/score.wav`.
3. `node build.mjs`, then in `composition/`: `npx hyperframes@0.8.202 check` and `render --output ../renders/tall_raw.mp4`.
4. Master to -14 LUFS with the kit's `tools/finish.py`, and copy the result to `docs/assets/video/agent-adapter-reel.mp4`.
