"""Template documents: Markdown with `{{ name }}` blanks, each filled by a model or an agent."""

from agent_adapter.compose.blanks import Blank, Kind
from agent_adapter.compose.checks import check_value
from agent_adapter.compose.editor import Editor
from agent_adapter.compose.results import BlankResult, ComposeResult, Revision
from agent_adapter.compose.run import compose
from agent_adapter.compose.template import Template

__all__ = [
    "Blank",
    "BlankResult",
    "ComposeResult",
    "Editor",
    "Kind",
    "Revision",
    "Template",
    "check_value",
    "compose",
]
