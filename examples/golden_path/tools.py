"""Golden path: the support desk's tools: orders, a confirmed refund, help, memory, tickets."""

import hashlib

from agent_adapter import Registry, ToolContext
from agent_adapter.embeddings import CallableEmbedder
from agent_adapter.errors import ToolUsageError
from agent_adapter.memory import SQLiteMemory, memory_tools
from agent_adapter.retrieval import Document, EmbeddingRetriever, Source, retriever_tool

# Your order database: order id -> (owner, status, total in EUR).
ORDERS = {
    "1042": ("ann", "shipped", 30.0),
    "1043": ("ann", "packed", 18.5),
    "2001": ("ben", "shipped", 12.0),
    "3001": ("lena", "packed", 40.0),
}
WORDS = ("return", "refund", "delivery", "days", "address", "track", "parcel", "exchange")
# The tools the model may call; the flow tools below are run only by the escalation flow.
CHAT_TOOLS = ("order_status", "refund_order", "search_help", "remember", "recall", "forget")


def embed(texts: list[str]) -> list[list[float]]:
    """Stands in for a real embedding model: one count per known word."""
    return [[float(text.lower().count(word)) for word in WORDS] for text in texts]


def help_page(slug: str, text: str) -> Document:
    """One page of the help centre, with the link the agent cites."""
    source = Source(uri=f"https://help.example.com/{slug}", title=slug.replace("-", " "))
    return Document(id=slug, text=text, source=source)


HELP = [
    help_page("returns", "A return is free within 30 days; the refund takes 5 days."),
    help_page("delivery", "Delivery takes 2 to 4 days; track a parcel from the order page."),
    help_page("change-address", "Change the delivery address until the parcel is packed."),
]


# --8<-- [start:registry]
def build_registry(memory: SQLiteMemory) -> Registry:
    """Every tool of the desk; the model sees `registry.subset(names=CHAT_TOOLS)`."""
    registry = Registry()

    @registry.tool
    def order_status(order_id: str, ctx: ToolContext) -> str:
        """Where an order is: packed, shipped or unknown."""
        order = own_order(order_id, ctx)
        return order[1] if order else "unknown"

    @registry.tool(confirm=True)  # runs only after a reviewer's one-use approval
    def refund_order(order_id: str, ctx: ToolContext) -> str:
        """Refund an order in full."""
        order = own_order(order_id, ctx)
        if order is None:
            raise ToolUsageError(f"no order {order_id}")
        return f"refunded {order[2]:.2f} EUR on order {order_id}"

    retriever = EmbeddingRetriever(
        CallableEmbedder(embed, provider="local", model="words-v1"), HELP
    )
    registry.add(retriever_tool(retriever, name="search_help", k=2))
    registry.add_many(memory_tools(memory))  # remember, recall, forget: scoped to ctx.user_id

    @registry.tool(idempotent=True)  # the same order and summary give the same ticket
    def open_ticket(order_id: str, summary: str, ctx: ToolContext) -> str:
        """Open a support ticket for a person to follow up."""
        if own_order(order_id, ctx) is None:
            raise ToolUsageError(f"no order {order_id}")
        digest = hashlib.sha256(f"{order_id}:{summary}".encode()).hexdigest()[:8]
        return f"T-{digest}"

    @registry.tool
    def notify_team(ticket_id: str) -> str:
        """Page the support team about a ticket (must not run twice)."""
        return page_team(ticket_id)

    return registry


def own_order(order_id: str, ctx: ToolContext) -> tuple[str, str, float] | None:
    """The order if the signed-in user owns it; another user's order looks like no order."""
    order = ORDERS.get(order_id)
    return order if order is not None and ctx.user_id and order[0] == ctx.user_id else None


def page_team(ticket_id: str) -> str:
    """Stands in for your pager or chat-ops client."""
    return f"team notified about {ticket_id}"


# --8<-- [end:registry]
