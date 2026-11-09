---
description: Add the Agent Adapter skill to Claude Code, Codex, Cursor, Copilot or Gemini CLI with one command.
---

# Agent skill

The skill teaches your coding agent the Agent Adapter API, its safety rules and which extras
your stack needs. Add it once; then ask "add Agent Adapter to this project" and the agent
reads your code, shows a plan and waits for your yes.

<div class="aa-logolist" markdown>

- :brand-claude-code: Claude Code
- :brand-codex: Codex
- :brand-cursor: Cursor
- :brand-github-copilot: GitHub Copilot
- :brand-gemini-cli: Gemini CLI
- :brand-windsurf: Windsurf
- :brand-antigravity: Antigravity
- :brand-kiro: Kiro
- :brand-cline: Cline
- :brand-roo-code: Roo Code
- :brand-opencode: OpenCode
- :brand-junie: Junie
- :brand-zed: Zed
- :brand-amp: Amp
- :brand-goose: Goose
- :brand-devin: Devin
- :brand-trae: Trae
- :brand-kilo-code: Kilo Code
- :brand-replit: Replit
- :brand-qwen: Qwen Code
- :brand-mistral: Mistral Vibe
- :brand-hermes: Hermes Agent
- and 60+ more agents

</div>

## Add it

=== "Any agent"

    ```bash
    npx skills add VeloceAI/agent-adapter --skill agent-adapter
    ```

    Works with Claude Code, Codex, Cursor, GitHub Copilot, Gemini CLI, Windsurf and more.
    Add `-g` to install it for every project, or `-a codex` to pick one agent.

=== "Claude Code"

    Paste these in Claude Code, one at a time:

    ```text
    /plugin marketplace add VeloceAI/agent-adapter
    /plugin install agent-adapter@agent-adapter
    ```

    The skill then shows as `/agent-adapter:agent-adapter`, and Claude also uses it on its own
    when you talk about agent tools.

=== "By hand"

    Copy [`skills/agent-adapter/SKILL.md`](https://github.com/VeloceAI/agent-adapter/blob/main/skills/agent-adapter/SKILL.md)
    into your project:

    | Agent | Folder |
    |---|---|
    | Claude Code | `.claude/skills/agent-adapter/` |
    | Codex, Cursor, Copilot, Gemini CLI | `.agents/skills/agent-adapter/` |

=== "Any chat"

    No skill support? Paste the [prompt for any AI](get-started/ai-prompt.md) instead.

## What it teaches

| Part | The agent learns |
|---|---|
| Before you write code | read the API guide, list the stack, show a plan and wait for a yes |
| The plan | the install line and extras, `@registry.tool`, `confirm=True`, one executor, how to connect |
| Connect | `attach(...)` for a framework, `await executor.run` or `executor.run_sync` in a backend, MCP or HTTP for other languages |
| Rules | no direct tool calls, no approval in `attach`, no secrets in logs, auth on every server |
| Moving existing code | `agent-adapter migrate --diff` and `registry.wrap()` |
| Check your work | tests, a denied `confirm=True` call, `agent-adapter doctor` |

The skill is short on purpose. For signatures it sends the agent to the full guide, so it
stays right when the API grows. To keep the whole guide in your repository, see
[For AI coding agents](get-started/for-ai-agents.md).

## Files

| File | Does |
|---|---|
| `skills/agent-adapter/SKILL.md` | the skill (open Agent Skills format) |
| `.claude-plugin/marketplace.json` | lets Claude Code add this repository as a plugin marketplace |
| `.claude-plugin/plugin.json` | the plugin's name, version and licence |
