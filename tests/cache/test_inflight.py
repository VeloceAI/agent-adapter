import asyncio

from agent_adapter.cache._inflight import _FLIGHTS, NOTHING, share

PORT = object()


def _keep(payload, waited_ms):
    return ("shared", payload)


async def test_an_empty_key_never_shares():
    runs = []

    async def lead():
        runs.append(1)
        await asyncio.sleep(0)
        return "mine"

    one = [share(PORT, "", lead, handout=lambda r: r, follow=_keep) for _ in range(3)]
    assert await asyncio.gather(*one) == ["mine"] * 3 and len(runs) == 3


async def test_a_nested_call_in_the_leaders_task_runs_instead_of_waiting():
    async def inner():
        return "inner"

    async def outer():
        nested = await share(PORT, "k", inner, handout=lambda r: r, follow=_keep)
        return f"outer+{nested}"

    assert await share(PORT, "k", outer, handout=lambda r: r, follow=_keep) == "outer+inner"
    assert not _FLIGHTS.get()


async def test_nothing_to_hand_out_lets_each_waiter_run():
    runs = []

    async def lead():
        runs.append(1)
        await asyncio.sleep(0)
        return "mine"

    calls = [share(PORT, "k", lead, handout=lambda r: NOTHING, follow=_keep) for _ in range(3)]
    assert await asyncio.gather(*calls) == ["mine"] * 3 and len(runs) == 3


async def test_other_ports_do_not_share():
    runs = []

    async def lead():
        runs.append(1)
        await asyncio.sleep(0)
        return "mine"

    calls = [share(object(), "k", lead, handout=lambda r: r, follow=_keep) for _ in range(2)]
    assert await asyncio.gather(*calls) == ["mine", "mine"] and len(runs) == 2
