# Brag Plan: Agent Adapter

## What is this app?
An open-source Python SDK: write a tool once as a plain function, and every call to it, from any agent
framework or none, is approved, budgeted, priced and traced by one executor.

## The angle
Open on the moment every team fears: an agent about to move money. Agent Adapter holds it for a person.
Then show why that is hard today (every framework has its own rules) and how small the fix is (one decorator,
one executor). Calm, confident, dark product film in the docs' indigo and teal.

## Hook (first 3 seconds)
"Your agent just tried to refund $5,000." over a terminal line that types `refund(order="o-7", amount=5000)`,
then a red HELD FOR APPROVAL stamp slams onto it.

## Key moments
- The real decorator: `@aa.tool(confirm=True)`, with `confirm=True` lit as "asks a person first".
- Four cards behind one tab bar (like the run viewer): Approval (a tap on Approve), Budgets (a team meter hits
  its $50/day cap and stops), Cost (every call priced, sample data), Traces (the viewer's real event rows).

## Outro / punchline
"Build with any framework. Ship with one safety layer." Then `pip install agent-adapter`, "Open source ·
Apache-2.0" and the docs address.

## User flow worth showing
Agent calls a risky tool → the call is held → a person approves → the run is traced and priced.

## Tone
- Preset: app-store, leaning polished
- Creative direction: dark developer product film, one accent per scene, nothing cute
- Interpretation: fast entrances, holds sized for reading, one stamp and two big hits, no jokes.

## Format: vertical 1080x1920 (Instagram, Shorts, LinkedIn, X) plus wide 1920x1080 (docs, README)
## Duration: 25.0 s

## Visual identity (from the docs theme)
- Background: #0b1020 (stage), cards #121a30, lines #26304d
- Accent: indigo #818cf8 / #a5b4fc; teal #2dd4bf; ok #4ade80; blocked #f87171; amber #fbbf24
- Text: #e2e8f0, muted #94a3b8
- Display font: Inter Tight 800 (kit, OFL). Code: Geist Mono (kit, OFL)
- Strongest visual element: the mark (indigo ring + teal bars) and the run viewer's event table

## Platform rules (from the research)
- Safe zones on 9:16: nothing important in the top 250 px, the bottom 350 px or the right 120 px.
- Text 60 px and up for reading lines; code 40 px and up, 8 lines at most, one line highlighted.
- Sound-off first: every beat carries its words on screen. No voice.
- Original audio only: the kit's synthesised score and Kenney CC0 sounds. The brag pop tracks have no logged
  licence, and business accounts cannot use most trending audio anyway.
- At most 5 hashtags.

## Storyboard (120 bpm, a beat is 0.5 s)

| # | Time | On screen | Sound |
|---|---|---|---|
| 1 Hook | 0–3.0 | "Your agent just tried to refund $5,000." Terminal types the call; red HELD FOR APPROVAL stamp at 2.0 | sparse pulse; key ticks; stamp hit at 2.0 |
| 2 Problem | 3.0–6.5 | "Every framework has its own rules." Framework names cascade as mono chips (LangChain, CrewAI, ADK, OpenAI Agents SDK, LlamaIndex, Pydantic AI, LangGraph, Haystack, DSPy, AutoGen, smolagents, Agno, Strands, +8). At 4.8: "So safety gets rebuilt. Every time." chips scatter | drive; whoosh at 3.0 |
| 3 Reveal | 6.5–11.5 | Mark + "Agent Adapter" (6.5); code card (8.0) with the 4-line refund tool; "One plain Python function."; `confirm=True` lights at 9.6, label "asks a person first" | riser into impact at 6.5; lift |
| 4 Proof | 11.5–18.5 | Header "Every call. Any framework." Tab bar Approval · Budgets · Cost · Traces lights per card at 11.5, 13.25, 15.0, 16.75. Approval: tap Approve → "Approved by ana". Budgets: `BudgetHook(50, scope="team", per="day")`, meter fills, "Stopped: team cap $50/day". Cost: 3 priced rows + total, "sample data". Traces: 4 real event rows | tick per card; click + confirm on Approve |
| 5 Tagline | 18.5–21.5 | "Build with any framework." then "Ship with one safety layer." (teal) | impact at 18.5; resolve |
| 6 CTA | 21.5–25.0 | `$ pip install agent-adapter` types; mark; "Open source · Apache-2.0"; agent-adapter.veloceai.in | whoosh at 21.5; bed fades out |

Reading holds: every line is settled for at least 0.3 s a word (hook line 2.1 s, tagline 2.7 s).

## Honesty
- All code on screen runs (checked against 1.1.0). "21 frameworks and engines" is the README's count; the reel
  says "every framework" and shows names as plain text, never logos.
- The $5,000 refund and the cost rows are illustrative and the cost card says "sample data". The trace rows come
  from the docs' sample run.

## Share copy (draft)
Your agent just tried to refund $5,000. Agent Adapter held it for a person.
One safety layer for every agent framework: approvals, budgets, cost and traces.
pip install agent-adapter · open source, Apache-2.0
#python #aiagents #opensource #llm #devtools
