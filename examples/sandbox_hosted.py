"""Hosted sandboxes: E2B and Modal as a `CallableSandbox`, each a short recipe.

Pass the real SDK class (`from e2b_code_interpreter import Sandbox`, `import modal`); this
example passes small offline stand-ins so it runs without an account.
"""

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar

from agent_adapter.errors import ToolUsageError
from agent_adapter.sandbox import CallableSandbox, SandboxRequest, SandboxResult


# --8<-- [start:e2b]
def e2b_runner(Sandbox: Any) -> Any:
    """`fn(request)` for `CallableSandbox`, over E2B's code-interpreter `Sandbox`."""

    def run(request: SandboxRequest) -> SandboxResult:
        box = Sandbox.create(
            timeout=int(request.timeout_s) + 30,
            envs=dict(request.env),
            allow_internet_access=request.network,
        )
        try:
            for name, data in request.files.items():
                box.files.write(name, data)
            done = box.run_code(request.code, timeout=request.timeout_s)
            error = done.error
            stderr = "".join(done.logs.stderr) + (error.traceback if error else "")
            return SandboxResult(
                exit_code=1 if error else 0, stdout="".join(done.logs.stdout), stderr=stderr
            )
        finally:
            box.kill()

    return run


# --8<-- [end:e2b]


# --8<-- [start:modal]
def modal_runner(modal: Any, app: Any, image: Any) -> Any:
    """`fn(request)` for `CallableSandbox`, over a Modal `Sandbox`; code only, no input files."""

    def run(request: SandboxRequest) -> SandboxResult:
        if request.files:
            raise ToolUsageError("this Modal recipe takes no input files")
        box = modal.Sandbox.create(
            "python",
            "-c",
            request.code,
            app=app,
            image=image,
            timeout=int(request.timeout_s),
            env=dict(request.env),
            block_network=not request.network,
        )
        box.wait(raise_on_termination=False)
        code = box.returncode
        return SandboxResult(
            exit_code=-1 if code is None else code,
            timed_out=code is None,
            stdout=box.stdout.read(),
            stderr=box.stderr.read(),
        )

    return run


# --8<-- [end:modal]


class FakeE2B:
    """Offline stand-in for E2B's `Sandbox`: echoes a fixed result."""

    seen: ClassVar[list[str]] = []

    @classmethod
    def create(cls, **kw: Any) -> "FakeE2B":
        cls.seen.append(f"create internet={kw['allow_internet_access']}")
        return cls()

    def __init__(self) -> None:
        self.files = SimpleNamespace(write=lambda name, data: self.seen.append(f"write {name}"))

    def run_code(self, code: str, timeout: float) -> Any:
        logs = SimpleNamespace(stdout=["6\n"], stderr=[])
        return SimpleNamespace(logs=logs, error=None)

    def kill(self) -> None:
        self.seen.append("kill")


def fake_modal() -> Any:
    """Offline stand-in for the `modal` module."""
    stream = SimpleNamespace(read=lambda: "7\n")
    box = SimpleNamespace(wait=lambda **kw: None, returncode=0, stdout=stream, stderr=stream)
    return SimpleNamespace(Sandbox=SimpleNamespace(create=lambda *a, **kw: box))


async def main() -> None:
    # --8<-- [start:use]
    e2b = CallableSandbox(e2b_runner(FakeE2B), name="e2b")
    result = await e2b.run("print(2 * 3)", files={"data.csv": "a,b"})
    print("e2b", result.stdout.strip(), result.exit_code)
    # --8<-- [end:use]
    print(FakeE2B.seen)
    modal = CallableSandbox(modal_runner(fake_modal(), app=None, image=None), name="modal")
    print("modal", (await modal.run("print(7)")).stdout.strip())


if __name__ == "__main__":
    asyncio.run(main())
