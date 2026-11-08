"""Sandboxed code: a shop assistant model writes Python, a person approves it, a sandbox runs it."""

import asyncio
import json

from agent_adapter import (
    ApprovalRequest,
    CallableModel,
    CallbackApproval,
    Executor,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
    run_tool_loop,
)
from agent_adapter.sandbox import CallableSandbox, SandboxRequest, SandboxResult, code_tool
from agent_adapter.tracing import EventLog

CODE = "prices = [12.5, 8.0, 4.5]\nprint(sum(prices))"


# --8<-- [start:sandbox]
def hosted_service(request: SandboxRequest) -> SandboxResult:
    """Stands in for a hosted sandbox (E2B, Modal, Docker); it never runs code on this machine."""
    answers = {CODE: "25.0\n"}
    return SandboxResult(exit_code=0, stdout=answers.get(request.code, ""))


sandbox = CallableSandbox(hosted_service, name="hosted")
# --8<-- [end:sandbox]

# --8<-- [start:tool]
registry = Registry()
registry.add(code_tool(sandbox, timeout_s=20, max_output_bytes=10_000))  # always confirm=True
# --8<-- [end:tool]


async def scripted_llm(request: LLMRequest) -> LLMResponse:
    """Stands in for a real model: writes code once, then answers from its output."""
    last = request.messages[-1]
    if last.role == "tool":
        total = json.loads(last.content)["stdout"].strip()
        text = f"The basket costs {total} in total."
        return LLMResponse(
            provider="host",
            model="shop-v1",
            text=text,
            message=Message(role="assistant", content=text),
        )
    call = ToolCall(name="run_code", arguments={"code": CODE}, id="call_code")
    return LLMResponse(
        provider="host",
        model="shop-v1",
        tool_calls=(call,),
        finish_reason="tool_calls",
        message=Message(role="assistant", tool_calls=(call,)),
    )


# --8<-- [start:approval]
def reviewer(request: ApprovalRequest) -> bool:
    """EXAMPLE ONLY: approves everything. In production a person reads the code and decides."""
    print("approve?", request.tool, repr(request.arguments["code"]))
    return True


# --8<-- [end:approval]


async def main() -> None:
    # --8<-- [start:loop]
    sink = EventLog()
    executor = Executor(registry, approval=CallbackApproval(reviewer), sink=sink)
    model = CallableModel(scripted_llm, provider="host", model="shop-v1")
    question = (Message(role="user", content="What does my basket cost in total?"),)
    result = await run_tool_loop(model, registry, question, executor=executor)
    print(result.final.text)
    [finished] = sink.of_type("sandbox.finished")
    print(finished.data["exit_code"], finished.data["stdout_bytes"], "code" in finished.data)
    # --8<-- [end:loop]


if __name__ == "__main__":
    asyncio.run(main())
