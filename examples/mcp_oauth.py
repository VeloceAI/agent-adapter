"""Import tools from an MCP server that wants OAuth: code flow, PKCE and a CIMD client id."""

import asyncio
import json

from mcp.shared.auth import AuthorizationCodeResult

from agent_adapter.integrations.mcp import client_metadata_document, mcp_client, oauth_client

CLIENT_ID = "https://agent.example.com/oauth/client.json"  # where you publish the document
REDIRECT = "http://127.0.0.1:33418/callback"

# --8<-- [start:document]
document = client_metadata_document(CLIENT_ID, redirect_uris=[REDIRECT], scope="tools")
# serve json.dumps(document) at CLIENT_ID (https, a path, its own client_id)
# --8<-- [end:document]


async def open_browser(url: str) -> None:
    """Send the person to the authorization page (a real app opens a browser)."""
    print("visit the authorization page")


async def wait_for_redirect() -> AuthorizationCodeResult:
    """Return the code and state the redirect to REDIRECT carried."""
    raise RuntimeError("a real app listens on REDIRECT here")


# --8<-- [start:client]
auth = oauth_client(
    "https://mcp.example.com/mcp",
    client_metadata_url=CLIENT_ID,
    redirect_uri=REDIRECT,
    redirect_handler=open_browser,
    callback_handler=wait_for_redirect,
    scope="tools",
)
client = mcp_client("https://mcp.example.com/mcp", auth=auth)
# then: async with client: await import_mcp(registry, client)
# --8<-- [end:client]


async def main() -> None:
    print(json.dumps(document["client_id"]))
    print(type(auth).__name__, auth.context.client_metadata_url == CLIENT_ID)


if __name__ == "__main__":
    asyncio.run(main())
