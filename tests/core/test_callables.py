"""`core.callables.is_async` is the one check for async user callables."""

from pathlib import Path

from agent_adapter.core.callables import is_async

PACKAGE = Path(__file__).resolve().parents[2] / "agent_adapter"


async def coroutine() -> None:
    """An async function."""


def plain() -> None:
    """A sync function."""


class AsyncCall:
    async def __call__(self) -> None:
        """An object with an async `__call__`."""


def test_is_async_sees_coroutine_functions_and_async_call_objects():
    assert is_async(coroutine) and is_async(AsyncCall())
    assert not is_async(plain) and not is_async(len)


def test_no_module_keeps_its_own_copy_of_the_check():
    copies = [
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "iscoroutinefunction(type(" in path.read_text(encoding="utf-8")
    ]
    assert copies == ["core/callables.py"]
