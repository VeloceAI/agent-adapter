"""The sandbox port: code in, stdout, stderr, an exit code and the files it produced out."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from agent_adapter.core.context import ToolContext

# The file each language's code is written to inside the work directory.
CODE_FILES: Mapping[str, str] = {"python": "main.py", "javascript": "main.js", "bash": "main.sh"}
# Commands for languages other than Python (which runs on the sandbox's own interpreter).
COMMANDS: Mapping[str, tuple[str, ...]] = {
    "javascript": ("node", "main.js"),
    "bash": ("bash", "main.sh"),
}

InputFiles = Mapping[str, bytes | str]


class SandboxFile(BaseModel):
    """A file the code left in its work directory; `data` may be cut (see `size`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    data: bytes
    size: Annotated[int, Field(ge=0)]  # the file's full size in bytes

    @property
    def truncated(self) -> bool:
        """True when `data` holds less than the whole file."""
        return len(self.data) < self.size

    def text(self) -> str:
        """`data` as UTF-8, with undecodable bytes replaced."""
        return self.data.decode("utf-8", errors="replace")


class SandboxResult(BaseModel):
    """What one run produced; output and files are bounded by the run's limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exit_code: int  # -1 when the run timed out or was killed before it exited
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    duration_ms: Annotated[float, Field(ge=0)] = 0.0
    files: tuple[SandboxFile, ...] = ()
    files_truncated: bool = False  # some files or bytes were left out
    backend: str = ""

    @property
    def ok(self) -> bool:
        """True when the code exited 0 within its time limit."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def truncated(self) -> bool:
        """True when any output or file was cut."""
        return self.stdout_truncated or self.stderr_truncated or self.files_truncated


@runtime_checkable
class SandboxPort(Protocol):
    """Runs untrusted code in isolation, within a time limit and an output limit."""

    async def run(
        self,
        code: str,
        *,
        language: str = "python",
        files: InputFiles | None = None,
        timeout_s: float = 30.0,
        max_output_bytes: int = 64_000,
        network: bool = False,
        env: Mapping[str, str] | None = None,
        ctx: ToolContext | None = None,
    ) -> SandboxResult: ...
