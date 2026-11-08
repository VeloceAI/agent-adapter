# Install

Agent Adapter is on [PyPI](https://pypi.org/project/agent-adapter/) under the Apache-2.0
licence. Install it with pip, or from a checkout of the
[GitHub repository](https://github.com/VeloceAI/agent-adapter).

!!! tip "Using an AI assistant?"
    Paste the [prompt for any AI](ai-prompt.md) into Claude, ChatGPT, Gemini or Cursor.
    It reads your project and tells you which extras to install and how to connect them.

## Requirements

| Need | Version |
|---|---|
| Python | 3.11 or later (developed on 3.12) |
| pydantic | `>=2.12,<3` (installed for you) |
| google-genai | `>=2.30`, only for `GeminiModel` and `gemini.genai_tool()` |

The OpenAI and Anthropic renderers return plain dicts. They need no SDK.

## Install from PyPI

```bash
pip install agent-adapter                 # core (pydantic only)
pip install "agent-adapter[openai]"       # + one extra; see Extras below
```

Use a virtual environment per project. A company mirror of PyPI works the same way: set it
as pip's only `index-url`. Avoid `--extra-index-url`: pip then takes the highest version
from any index, so a package name on two indexes can resolve to the wrong one (dependency
confusion). `agent-adapter doctor` warns when one is set.

## Install from a checkout

1. Clone the repository and open its root folder:
   `git clone https://github.com/VeloceAI/agent-adapter.git && cd agent-adapter`
2. Create a virtual environment.

    === "Windows"

        ```powershell
        python -m venv .venv
        .venv\Scripts\Activate.ps1
        ```

    === "macOS / Linux"

        ```bash
        python -m venv .venv
        source .venv/bin/activate
        ```

3. Install the package, with the extras you need.

    ```bash
    pip install -e .                 # core only
    pip install -e ".[gemini]"       # + google-genai, for GeminiModel and genai_tool()
    pip install -e ".[dev,docs]"     # + tests, linters and the docs site
    ```

4. Check it works.

    ```bash
    python -c "import agent_adapter; print(agent_adapter.__version__)"
    ```

    You should see `1.1.0`.

## Extras

The core needs only pydantic. Each extra adds the packages for one integration or feature.

| Extra | Adds | Status |
|---|---|---|
| `openai`, `gemini`, `anthropic` | the provider SDK, for the model ports (`GeminiModel`, `genai_tool()`, ...) | :material-check-circle:{ .status-ok } Available ([Model clients](../integrations/index.md)) |
| one per framework: `langchain`, `langgraph`, `mcp`, `openai-agents`, `pydantic-ai`, `llamaindex`, `agent-framework`, `agno`, `haystack`, `claude-agent`, `smolagents`, `autogen`, `ag2` | that framework | :material-check-circle:{ .status-ok } Available ([Integrations](../integrations/index.md)) |
| own environment: `crewai`, `strands`, `semantic-kernel`, `adk`, `camel`, `dspy` | that framework; each conflicts with another extra, so install it in its own venv | :material-check-circle:{ .status-ok } Available ([why](../integrations/index.md#install-several)) |
| `a2a`, `ag-ui` | that protocol (A2A server and client; AG-UI event stream) | :material-check-circle:{ .status-ok } Available ([A2A](../integrations/a2a.md), [AG-UI](../integrations/ag-ui.md)) |
| `openapi` | `httpx`, `pyyaml`, for `import_openapi` | :material-check-circle:{ .status-ok } Available ([OpenAPI](../integrations/openapi.md)) |
| `server`, `server-wsgi` | Starlette, Uvicorn, the MCP SDK; a2wsgi | :material-check-circle:{ .status-ok } Available ([HTTP server](../cross-language/http-server.md)) |
| `redis`, `otel` | Redis cache and stores; OpenTelemetry API | :material-check-circle:{ .status-ok } Available |
| `sqlalchemy`, `dynamodb` | database approval stores (SQLAlchemy async; aioboto3) | :material-check-circle:{ .status-ok } Available ([Database stores](../concepts/approval-stores-db.md)) |
| `all` | every extra that co-installs from wheels; not `crewai`, `strands`, `semantic-kernel`, `adk`, `camel`, `dspy` ([why](../integrations/index.md#install-several)) | :material-check-circle:{ .status-ok } Available |
| `dev` | pytest, pytest-asyncio, pytest-cov, ruff, mypy, nox, hatchling, tzdata, fakeredis, lupa, sqlalchemy, aiosqlite, moto | :material-check-circle:{ .status-ok } Available |
| `docs` | mkdocs-material, mkdocstrings | :material-check-circle:{ .status-ok } Available |

Pinned ranges and test sessions per extra: [Support matrix](../support-matrix.md).

## Next

Go to the [Quickstart](quickstart.md).
