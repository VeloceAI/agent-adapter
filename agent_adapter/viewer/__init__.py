"""Run viewer: record runs to JSONL files, browse them and chat locally (`agent-adapter ui`)."""

from agent_adapter.viewer._files import DEFAULT_FOLDER
from agent_adapter.viewer.approvals import ApprovalDesk
from agent_adapter.viewer.chat import Attachment, ChatAgent, ChatMessage, ChatSession
from agent_adapter.viewer.export import export
from agent_adapter.viewer.lanes import assign_lanes, build_spans
from agent_adapter.viewer.layout import layout
from agent_adapter.viewer.recorder import Recorder, record
from agent_adapter.viewer.runs import RunFolder
from agent_adapter.viewer.server import LOOPBACK, ViewerServer, check_host, make_server
from agent_adapter.viewer.serving import serve

__all__ = [
    "DEFAULT_FOLDER",
    "LOOPBACK",
    "ApprovalDesk",
    "Attachment",
    "ChatAgent",
    "ChatMessage",
    "ChatSession",
    "Recorder",
    "RunFolder",
    "ViewerServer",
    "assign_lanes",
    "build_spans",
    "check_host",
    "export",
    "layout",
    "make_server",
    "record",
    "serve",
]
