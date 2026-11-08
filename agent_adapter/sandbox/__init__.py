"""Sandboxed code execution: `SandboxPort`, Docker, callable and dev-only local backends."""

from agent_adapter.sandbox._base import GuardedSandbox
from agent_adapter.sandbox._checks import Limits, SandboxRequest
from agent_adapter.sandbox.callable import CallableSandbox, RunFn
from agent_adapter.sandbox.docker import DockerSandbox, DockerStatus, docker_status
from agent_adapter.sandbox.local import LocalSubprocessSandbox
from agent_adapter.sandbox.port import (
    CODE_FILES,
    InputFiles,
    SandboxFile,
    SandboxPort,
    SandboxResult,
)
from agent_adapter.sandbox.tool import CODE_KIND, OnResult, code_tool

__all__ = [
    "CODE_FILES",
    "CODE_KIND",
    "CallableSandbox",
    "DockerSandbox",
    "DockerStatus",
    "GuardedSandbox",
    "InputFiles",
    "Limits",
    "LocalSubprocessSandbox",
    "OnResult",
    "RunFn",
    "SandboxFile",
    "SandboxPort",
    "SandboxRequest",
    "SandboxResult",
    "code_tool",
    "docker_status",
]
