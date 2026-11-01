"""Chat wrappers: LangGraph, ADK, CrewAI or your own HTTP backend as a chat agent.

Each factory returns a function to pass to `serve(agent=...)` or `agent-adapter ui --agent`.
"""

import json
import urllib.request
from collections.abc import AsyncIterator
from typing import Any

from agent_adapter.viewer import ChatAgent, ChatMessage, ChatSession


# --8<-- [start:langgraph]
def langgraph_agent(app: Any) -> ChatAgent:
    """A compiled LangGraph app (messages state) as a chat agent."""

    async def reply(message: ChatMessage, session: ChatSession) -> str:
        history = [(m.role, m.content) for m in session.history]
        out = await app.ainvoke({"messages": [*history, ("user", message.text)]})
        return str(out["messages"][-1].content)

    return reply


# --8<-- [end:langgraph]


# --8<-- [start:adk]
def adk_agent(runner: Any, user_id: str = "local") -> ChatAgent:
    """An ADK `Runner` as a chat agent; each chat is one ADK session."""

    async def reply(message: ChatMessage, session: ChatSession) -> AsyncIterator[str]:
        from google.genai import types  # installed with google-adk

        sessions, app = runner.session_service, runner.app_name
        if await sessions.get_session(app_name=app, user_id=user_id, session_id=session.id) is None:
            await sessions.create_session(app_name=app, user_id=user_id, session_id=session.id)
        content = types.Content(role="user", parts=[types.Part(text=message.text)])
        async for event in runner.run_async(
            user_id=user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response() and event.content and event.content.parts:
                yield "".join(part.text or "" for part in event.content.parts)

    return reply


# --8<-- [end:adk]


# --8<-- [start:crewai]
def crewai_agent(crew: Any) -> ChatAgent:
    """A CrewAI `Crew` as a chat agent; its task reads `{question}`."""

    async def reply(message: ChatMessage, session: ChatSession) -> str:
        result = await crew.kickoff_async(inputs={"question": message.text})
        return str(result.raw)

    return reply


# --8<-- [end:crewai]


# --8<-- [start:http]
def http_agent(url: str, timeout: float = 60.0) -> ChatAgent:
    """Your own backend (FastAPI or any) that answers POST {message, session_id} with {answer}."""

    def reply(message: ChatMessage, session: ChatSession) -> str:  # sync: runs in a thread
        body = json.dumps({"message": message.text, "session_id": session.id}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        request = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return str(json.loads(response.read())["answer"])

    return reply


# --8<-- [end:http]
