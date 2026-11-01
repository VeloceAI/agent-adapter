"""ToolContext in a tool: ids, custom events, child contexts; errors as ErrorInfo."""

import asyncio
import json

from agent_adapter import Event, Registry, ToolContext, ToolError
from agent_adapter.errors import ErrorInfo

registry = Registry()


class PrintSink:
    """An EventSink: sync, quick, never raises."""

    def emit(self, event: Event) -> None:
        print(event.type, dict(event.data))


# --8<-- [start:tool]
@registry.tool
async def import_rows(source: str, ctx: ToolContext) -> int:
    """Import rows from a source."""
    ctx.emit("progress", pct=50)  # sent as "custom.progress"; a no-op without a sink
    if source == "broken":
        raise ToolError("source is offline", details={"source": source})
    return 42


# --8<-- [end:tool]


async def main() -> None:
    # --8<-- [start:call]
    ctx = ToolContext(run_id="run_1", call_id="call_1", thread_id="t1", sink=PrintSink())
    tool = registry.get("import_rows")  # the function alone, as in a test; apps use the executor
    print(await tool.invoke({"source": "s3"}, ctx))
    nested = ctx.child(call_id="call_2")  # same run, sink and ledger; extras copied
    print(nested.run_id, nested.call_id, nested.approval)
    # --8<-- [end:call]
    # --8<-- [start:error]
    try:
        await tool.invoke({"source": "broken"}, ctx)
    except ToolError as exc:
        print(json.dumps(ErrorInfo.from_exception(exc).model_dump()))
    print(ErrorInfo.from_exception(KeyError("secret")).message)  # foreign errors: class only
    # --8<-- [end:error]


if __name__ == "__main__":
    asyncio.run(main())
