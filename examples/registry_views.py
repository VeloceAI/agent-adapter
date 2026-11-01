"""Filter a registry per agent and render a prompt catalogue from it."""

from agent_adapter import Executor, Registry, ToolContext

registry = Registry()


# --8<-- [start:tools]
@registry.tool(kind="retrieval", tags=["search", "public"], prompt_hint="First step for any query.")
def search(query: str, top_k: int = 8) -> list[str]:
    """Search articles.

    Args:
        query: Search text.
        top_k: How many hits.
    """
    return [query] * top_k


@registry.tool(kind="action", tags="admin", confirm=True)
def delete_note(note_id: str) -> bool:
    """Delete a saved note.

    Args:
        note_id: The note to delete.
    """
    return True


# --8<-- [end:tools]


def main() -> None:
    # --8<-- [start:views]
    public = registry.subset(tags=["public"])  # a new registry; the original is unchanged
    print(public.names())  # ['search']
    print(registry.subset(kinds=["action"], tags=["public"]).names())  # [] (filters AND)
    print(registry.catalogue())  # Markdown for a planner prompt
    removed = registry.remove("delete_note")
    print(removed.name, "delete_note" in registry)  # delete_note False
    # --8<-- [end:views]


async def call_search(ctx: ToolContext) -> list[list[str]]:
    """Run `search` through the executor, then alone as a unit test would."""
    # --8<-- [start:call]
    executor = Executor(registry)  # in an application: every policy option applies
    hits = await executor.run("search", {"query": "returns"}, ctx)

    spec = registry.get("search")  # in a unit test: the function alone, no policies
    alone = await spec.invoke({"query": "returns"}, ctx)
    validated = spec.validate_args({"query": "returns", "top_k": 2})  # validate now...
    later = await spec.call(validated, ctx)  # ...run later
    # --8<-- [end:call]
    return [hits, alone, later]


if __name__ == "__main__":
    main()
