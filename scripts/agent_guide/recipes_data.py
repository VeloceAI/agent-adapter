"""Recipes for memory, retrieval (RAG), chat memory and approval memory."""

from __future__ import annotations

from scripts.agent_guide.recipe_model import Recipe

# Test-only input: a stand-in embedding model.
EMBED = (
    "def my_embed(texts):  # your embedding model\n"
    "    return [[float(len(t)), 1.0] for t in texts]\n"
)

DATA: tuple[Recipe, ...] = (
    Recipe(
        "remember facts per user across chats (recall, forget, export, erase)",
        "`SQLiteMemory` + `memory_tools` (`agent_adapter.memory`); the scope comes from "
        "`ctx.user_id`",
        (
            "from agent_adapter.memory import MemoryScope, SQLiteMemory, memory_tools",
            "",
            'memory = SQLiteMemory("memory.db", retention_days=90)  # embedder= adds vector search',
            "registry.add_many(memory_tools(memory))  # remember, recall, forget: scope from the "
            "ctx",
            'ctx = executor.context(thread_id="chat-1", user_id="ann")  # the signed-in user',
            'await executor.run("remember", {"text": "Ann prefers aisle seats"}, ctx)',
            'print(await memory.export(MemoryScope(user_id="ann")))  # a data request',
            'await memory.delete(scope=MemoryScope(user_id="ann"))  # erase the user',
        ),
        full_only=True,
    ),
    Recipe(
        "let the model search your documents and cite sources (RAG)",
        "`retriever_tool` over any `RetrieverPort` (`agent_adapter.retrieval`)",
        (
            "from agent_adapter.embeddings import CallableEmbedder",
            "from agent_adapter.retrieval import Document, EmbeddingRetriever, retriever_tool",
            "",
            'pages = [Document(id="hours", text="Open 9 to 17.", metadata={"shop": "north"})]',
            'retriever = EmbeddingRetriever(CallableEmbedder(my_embed), pages, scope_key="shop")',
            "# the host sets the scope (here from ctx.state), never the model",
            'registry.add(retriever_tool(retriever, filters=lambda ctx: {"shop": '
            'ctx.state["shop"]}))',
            "# a vector DB or search API: CallableRetriever(fn); or LlamaIndexRetriever, "
            "LangChainRetriever",
        ),
        setup=EMBED,
        full_only=True,
    ),
    Recipe(
        "remember a chat user across chats (recall each turn, save facts or a summary)",
        "`serve(memory=SQLiteMemory(...), user_id=..., memory_save=...)` or "
        "`ui --memory sqlite:PATH --user NAME`",
        (
            "from agent_adapter.memory import SQLiteMemory",
            "from agent_adapter.viewer import serve",
            "",
            'if __name__ == "__main__":',
            '    serve(model=model, memory=SQLiteMemory("memory.db"), user_id="ann", '
            'memory_save="facts")',
            "# session.memories holds the recalled records; tools see ctx.user_id",
            "# CLI: agent-adapter ui --model app:model --memory sqlite:memory.db --user ann",
        ),
        full_only=True,
    ),
    Recipe(
        "show reviewers similar past approvals; propose confirm rules a person accepts",
        "`approval_context` / `propose_rules` / `accept` from `agent_adapter.approval`; "
        "advice only",
        (
            "from agent_adapter.approval import accept, propose_rules",
            "",
            "store = aa.MemoryApprovalStore()  # the store your DeferredApproval uses",
            "for rule in propose_rules(await store.list(), min_count=5):  # data: changes nothing",
            "    print(rule.describe())",
            '# accepted = accept(rule, accepted_by="ann"); commit accepted.code (confirm= rule)',
            "# approval_context(record.request, records): counts and deciders; the call still "
            "waits",
        ),
        full_only=True,
    ),
    Recipe(
        "let parallel or sequential agents see what the others are doing and found",
        "`board_of(ctx)`, `board_tools()` and `LoopAgent(board=\"auto\")` "
        "(`agent_adapter.board`)",
        (
            "from agent_adapter.board import board_of, board_tools",
            "from agent_adapter.exec import LoopAgent",
            "",
            "registry.add_many(board_tools())  # board_read, board_post, board_claim",
            'scout = LoopAgent(model, registry, board="auto")  # sees only what is new',
            "ctx = executor.context()  # one board per run, shared by every agent in it",
            'await board_of(ctx).post("finding", "Oslo museums open 10-17", by="planner")',
            "print(await board_of(ctx).summary())",
        ),
    ),
    Recipe(
        "fill a document template (Markdown, HTML, XML, text) from many model calls, "
        "with citations",
        "`compose()` / `compose_sync()` (`agent_adapter.compose`)",
        (
            "from agent_adapter.compose import compose",
            "",
            "page = '# Weather\\n\\n{{ summary | \"one sentence on Oslo\" }}'",
            'docs = [{"id": "met", "text": "Oslo: sunny, 21 C", "source": {"title": "Met"}}]',
            "result = await compose(page, model=model, sources=docs)  # editor and checks on",
            "print(result.markdown, result.usage)  # citations: numeric, linked, footnote...",
        ),
        full_only=True,
    ),
)
