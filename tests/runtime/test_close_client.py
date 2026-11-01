from agent_adapter.runtime.loop_local import LoopLocal, close_client


class _Client:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


async def test_close_client_awaits_aclose_when_the_client_has_one() -> None:
    client = _Client()
    await close_client(client)
    assert client.closed == 1


async def test_close_client_ignores_a_client_without_aclose() -> None:
    await close_client(object())


async def test_close_client_closes_the_loop_local_value() -> None:
    local: LoopLocal[_Client] = LoopLocal(_Client, close=close_client)
    client = local.get()
    await local.aclose()
    assert client.closed == 1
