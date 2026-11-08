from agent_adapter import Registry, ToolCall
from agent_adapter.exec import Executor, agent_tool
from agent_adapter.viewer.chat import ChatMessage, ChatSession, chat_agent_task
from tests.fakes.sinks import RecordingSink


async def test_a_chat_agent_runs_as_a_sub_agent_in_the_calling_run():
    sink = RecordingSink()
    registry = Registry()

    @registry.tool
    def opening_hours(day: str) -> str:
        """Museum opening hours."""
        return "10-18"

    executor = Executor(registry, sink=sink)
    seen = {}

    async def guide(message: ChatMessage, session: ChatSession):
        seen["session"] = session
        hours = await executor.run("opening_hours", {"day": "mon"}, session.context())

        async def chunks():
            yield "Open "
            yield hours

        return chunks()

    registry.add(agent_tool(chat_agent_task(guide, executor), name="guide", description="Guide."))
    ctx = executor.context(state={"visitor": "v1"})
    result = await executor.run_call(
        ToolCall(name="guide", arguments={"task": "hours?"}, id="call_guide"), ctx
    )
    assert result.value == "Open 10-18"
    session = seen["session"]
    assert (session.run_id, session.state, session.history) == (ctx.run_id, {"visitor": "v1"}, ())
    nested = [e for e in sink.of_type("tool.started") if e.data["tool"] == "opening_hours"]
    assert nested[0].parent_id == "call_guide"


async def test_a_sync_chat_agent_returning_text():
    executor = Executor(Registry())
    spec = agent_tool(
        chat_agent_task(lambda message, session: f"echo {message.text}", executor),
        name="echo",
        description="Echo.",
    )
    assert await executor.run_spec(spec, {"task": "hi"}) == "echo hi"
