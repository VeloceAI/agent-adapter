"""Fake framework packages, just enough for each generated template to run offline.

Each fake supports the names its integration module reads when it exports tools, and an
agent whose run answers `"<model> answered with <tool names>"`.
"""

from __future__ import annotations

from pathlib import Path

_NAMESPACE = "from types import SimpleNamespace\n"

PYDANTIC_AI = {
    "pydantic_ai/__init__.py": _NAMESPACE
    + """
class Tool:
    def __init__(self, function, *, name, description):
        self.function, self.name, self.description = function, name, description

    @classmethod
    def from_schema(cls, function, *, name, description, json_schema, takes_ctx):
        return cls(function, name=name, description=description)


class DeferredToolRequests:
    pass


class Agent:
    def __init__(self, model, *, tools, output_type):
        self.model, self.tools = model, tools

    async def run(self, prompt=None, **options):
        names = ",".join(tool.name for tool in self.tools)
        return SimpleNamespace(output=f"{self.model} answered with {names}", run_id="r1")
""",
}

AGNO = {
    "agno/__init__.py": '__version__ = "3.1.0"\n',
    "agno/tools/__init__.py": "",
    "agno/tools/function.py": """
class Function:
    __private_attributes__ = {"_run_context": None}

    def __init__(self, **fields):
        self.__dict__.update(fields)


class FunctionCall:
    model_fields = {"call_id": None}
""",
    "agno/agent/__init__.py": _NAMESPACE
    + """
class Agent:
    def __init__(self, *, model, tools, telemetry):
        self.model, self.tools = model, tools

    async def arun(self, prompt):
        names = ",".join(tool.name for tool in self.tools)
        return SimpleNamespace(is_paused=False, content=f"{self.model} answered with {names}")
""",
}

STRANDS = {
    "strands/__init__.py": """
class _Hooks:
    def __init__(self):
        self.callbacks = []

    def add_callback(self, event, callback):
        self.callbacks.append((event, callback))


class Agent:
    def __init__(self, *, model, tools, callback_handler):
        self.model, self.tools, self.hooks = model, tools, _Hooks()

    def __call__(self, prompt):
        names = ",".join(tool.tool_name for tool in self.tools)
        return f"{self.model} answered with {names} ({len(self.hooks.callbacks)} hook)"
""",
    "strands/hooks/__init__.py": "class BeforeToolCallEvent:\n    pass\n",
    "strands/types/__init__.py": "",
    "strands/types/tools.py": """
ToolSpec = dict


class AgentTool:
    def __init__(self):
        pass
""",
}

CREWAI = {
    "crewai/__init__.py": """
class Agent:
    def __init__(self, *, role, goal, backstory, tools, llm):
        self.tools, self.llm = tools, llm


class Task:
    def __init__(self, *, description, expected_output, agent):
        self.agent = agent


class Crew:
    def __init__(self, *, agents, tasks):
        self.agents = agents

    def kickoff(self):
        agent = self.agents[0]
        names = ",".join(tool.name for tool in agent.tools)
        return f"{agent.llm} answered with {names}"
""",
    "crewai/tools/__init__.py": """
from typing import Any

from pydantic import BaseModel, ConfigDict


class BaseTool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    description: str
    args_schema: Any = None
    cache_function: Any = None
""",
}

ADK = {
    "google/adk/__init__.py": "",
    "google/adk/version.py": '__version__ = "2.13.0"\n',
    "google/adk/agents/__init__.py": """
class LlmAgent:
    def __init__(self, *, name, model, tools):
        self.model, self.tools = model, tools
""",
    "google/adk/agents/context.py": """
class Context:
    def run_node(self):
        pass

    resume_inputs = None
""",
    "google/genai/__init__.py": "",
    "google/genai/types.py": """
class Part:
    def __init__(self, *, text):
        self.text = text


class Content:
    def __init__(self, *, role, parts):
        self.role, self.parts = role, parts
""",
    "google/adk/tools/__init__.py": "",
    "google/adk/tools/base_tool.py": """
class BaseTool:
    def __init__(self, *, name, description):
        self.name, self.description = name, description

    async def check_require_confirmation(self, args, tool_context):
        return False
""",
    "google/adk/sessions/__init__.py": _NAMESPACE
    + """
class InMemorySessionService:
    async def create_session(self, *, app_name, user_id):
        return SimpleNamespace(id="s1")
""",
    "google/adk/runners/__init__.py": _NAMESPACE
    + """
class _Event:
    def __init__(self, text):
        self.content = SimpleNamespace(parts=[SimpleNamespace(text=text)])

    def is_final_response(self):
        return True

    def get_function_calls(self):
        return []

    def get_function_responses(self):
        return []


class Runner:
    def __init__(self, *, app_name, agent, session_service):
        self.agent = agent

    async def run_async(self, *, user_id, session_id, new_message):
        names = ",".join(tool.name for tool in self.agent.tools)
        yield _Event(f"{self.agent.model} answered with {names}")
""",
}

FAKES: dict[str, dict[str, str]] = {
    "pydantic-ai": PYDANTIC_AI,
    "agno": AGNO,
    "strands": STRANDS,
    "crewai": CREWAI,
    "adk": ADK,
}


def write_fake(template: str, folder: Path) -> Path:
    """Write the fake framework for `template` under `folder`; return `folder`."""
    for relative, source in FAKES[template].items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return folder
