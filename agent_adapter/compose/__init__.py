"""Template documents: Markdown, HTML, XML or text with blanks, each filled by a model or agent."""

from agent_adapter.compose.blanks import Blank, Kind
from agent_adapter.compose.checks import check_value
from agent_adapter.compose.citations import Citations
from agent_adapter.compose.editor import Editor
from agent_adapter.compose.results import BlankResult, Citation, ComposeResult, Revision, StageUsage
from agent_adapter.compose.run import compose, compose_sync
from agent_adapter.compose.template import Template
from agent_adapter.compose.verifier import Verifier

__all__ = [
    "Blank",
    "BlankResult",
    "Citation",
    "Citations",
    "ComposeResult",
    "Editor",
    "Kind",
    "Revision",
    "StageUsage",
    "Template",
    "Verifier",
    "check_value",
    "compose",
    "compose_sync",
]
