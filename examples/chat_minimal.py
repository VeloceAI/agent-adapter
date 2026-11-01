"""The smallest chat agent: one async function served in the local chat page.

Run it and open the printed link; Ctrl+C stops it. Replace the function body with your agent.
"""

# --8<-- [start:minimal]
from agent_adapter.viewer import ChatMessage, ChatSession, serve


async def chat(message: ChatMessage, session: ChatSession) -> str:
    """Answers each message; `session.history` holds the earlier turns."""
    return f"You said: {message.text}"


if __name__ == "__main__":
    serve(agent=chat)  # blocks; prints the link and opens the page on 127.0.0.1:8780
# --8<-- [end:minimal]
