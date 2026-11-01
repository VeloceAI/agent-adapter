"""Notebooks and apps: Jupyter awaits, Streamlit reruns call run_sync, Gradio handlers await."""

from typing import Any

from backend_common import Backend, build_backend, call

from agent_adapter import run_sync
from agent_adapter.runtime import new_id

TIMEOUT_S = 30


async def shipping_rate(backend: Backend, region: str) -> float:
    """One tool call as its own run."""
    ids = {"run_id": new_id("run_"), "call_id": new_id("call_")}
    rate: float = await call(backend, "shipping_rate", {"region": region}, **ids)
    return rate


# --8<-- [start:jupyter]
async def notebook_cell(backend: Backend) -> float:
    """In a Jupyter cell, write the body at top level: `rate = await shipping_rate(...)`."""
    return await shipping_rate(backend, "us")


# --8<-- [end:jupyter]


# --8<-- [start:streamlit]
def streamlit_page(st: Any) -> None:
    """A Streamlit script: the backend is cached across reruns, each call is one run_sync."""
    backend = st.cache_resource(build_backend)()  # once per server process
    region = st.selectbox("Region", ["eu", "us", "other"])
    rate = run_sync(shipping_rate(backend, region), timeout=TIMEOUT_S)  # never asyncio.run here
    st.metric("Shipping %", rate)


# --8<-- [end:streamlit]


# --8<-- [start:gradio]
def gradio_handler(backend: Backend) -> Any:
    """A Gradio `async def` event handler: Gradio awaits it on its own loop."""

    async def rate(region: str) -> str:
        return f"{await shipping_rate(backend, region)}%"

    return rate


# --8<-- [end:gradio]
