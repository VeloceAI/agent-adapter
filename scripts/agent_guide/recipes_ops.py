"""Recipes for durable runs, engines, sandboxes, team budgets, streaming and the CLI tools."""

from __future__ import annotations

from scripts.agent_guide.recipe_model import Recipe

# Test-only inputs: a one-node flow and its key; a stand-in OTLP export.
DURABLE = (
    "from pydantic import BaseModel\n"
    "class Order(BaseModel):\n"
    "    id: str\n"
    "    weather: str = ''\n"
    "flow = aa.Flow('order', state=Order).node('look', lambda o: {'weather': 'sun'}).end('look')\n"
    "state = {'id': 'o-7'}\n"
    "KEY = b'k' * 32\n"
)

OPS: tuple[Recipe, ...] = (
    Recipe(
        "make a flow survive a crash (checkpoints, recover)",
        "`FlowRunner(checkpoints=SQLiteCheckpointStore(path), checkpoint_key=KEY)`, `recover_all`",
        (
            "from agent_adapter.exec.checkpoints import SQLiteCheckpointStore",
            "",
            'store = SQLiteCheckpointStore("runs.db")  # Redis or SQLAlchemy for many hosts',
            "runner = aa.FlowRunner(executor, checkpoints=store, checkpoint_key=KEY)",
            'result = await runner.run(flow, state, executor.context(run_id="order-7"))',
            "# on a new worker after a crash: finished tool calls replay, never run twice",
            "results = await runner.recover_all(flow)",
        ),
        setup=DURABLE,
        index="durable runs",
    ),
    Recipe(
        "run a flow on Temporal or DBOS (durable engine)",
        "`DurableFlows(runner, [flow])`, `flow_worker`, `start_flow`, `notify`; DBOS: `DbosFlows`",
        (
            "from agent_adapter.integrations.temporal import "
            "DurableFlows, flow_worker, notify, start_flow",
            "",
            "runner = aa.FlowRunner(executor, checkpoints=store, checkpoint_key=KEY)",
            "flows = DurableFlows(runner, [flow])",
            'async with flow_worker(client, flows, task_queue="refunds"):',
            '    handle = await start_flow(client, "refunds", state, run_id="t-17", '
            'task_queue="refunds")',
            "    # a reviewer decides in the approval store; the notice only wakes the run",
            "    await approvals.decide(pending_id, aa.ApprovalDecision(approved=True))",
            "    await notify(handle, pending_id)",
        ),
        requires=("temporalio",),
        index="Temporal and DBOS",
    ),
    Recipe(
        "let the model write and run code safely (sandbox, always approved)",
        "`code_tool` over a `SandboxPort` (`agent_adapter.sandbox`)",
        (
            "from agent_adapter.sandbox import DockerSandbox, code_tool",
            "",
            'sandbox = DockerSandbox("python:3.12-slim")  # no network, read-only, non-root',
            "registry.add(code_tool(sandbox, timeout_s=20))  # confirm=True, always",
            "# hosted (E2B, Modal): CallableSandbox(fn); dev only: "
            "LocalSubprocessSandbox(allow_unsafe_local=True)",
        ),
        index="sandboxed code",
    ),
    Recipe(
        "cap spend per user or team, per day or month, across workers",
        "`BudgetHook(scope=, per=, store=)` (`agent_adapter.hooks.builtin.budgets`)",
        (
            "from agent_adapter.hooks.builtin.budgets import SQLiteBudgetStore",
            "",
            'store = SQLiteBudgetStore("budgets.db")  # RedisBudgetStore, SQLAlchemyBudgetStore',
            'per_user = aa.BudgetHook(0.50, scope="user", per="day", store=store)  # ctx.user_id',
            "",
            "",
            "def team_of(ctx: aa.ToolContext) -> str:  # your user -> team lookup",
            '    return "support"',
            "",
            "",
            'per_team = aa.BudgetHook(0.80, scope="team", per="day", scope_key=team_of, '
            "store=store)",
            "executor = aa.Executor(registry, hooks=[per_user, per_team])  # reserve, settle, "
            "release",
        ),
        index="team budgets",
    ),
    Recipe(
        "show tool arguments and partial structured answers while the model streams",
        "`run_tool_loop(on_event=)`; `ToolCallDelta`, `PartialOutput` (`agent_adapter.models`)",
        (
            "from agent_adapter.models import PartialOutput, ToolCallDelta",
            "",
            "",
            "def show(event):  # every stream event but text; validate only the final reply",
            "    if isinstance(event, ToolCallDelta | PartialOutput):",
            "        print(event)",
            "",
            "",
            'messages = [aa.Message(role="user", content="Weather in Oslo?")]',
            "result = await aa.run_tool_loop(model, registry, messages, on_event=show)",
        ),
        index="streaming deltas",
    ),
    Recipe(
        "browse OTLP traces from another app in the run viewer; search and tag runs",
        "`agent-adapter ui --import traces.json`, or `import_otlp` (`agent_adapter.viewer`)",
        (
            "from pathlib import Path",
            "from agent_adapter.viewer import import_otlp",
            "",
            'Path("traces.json").write_text(\'{"resourceSpans": []}\')  # your OTLP JSON export',
            'import_otlp("traces.json", ".agent_adapter/runs")  # parsed only, never run',
            "# then `agent-adapter ui`: search box (tool names, status:failed, tag:vip), tags",
        ),
        index="viewer search, tags and OTLP import",
    ),
    Recipe(
        "start a framework project, or move existing framework tools to the registry",
        "`agent-adapter init <framework>`, `agent-adapter migrate PATH --diff`",
        (
            "# agent-adapter init pydantic-ai my-agent  (also adk, agno, strands, crewai)",
            "# agent-adapter migrate src/ --diff > migrate.patch  (read-only; review, then git "
            "apply)",
        ),
        index="framework templates and migrate",
    ),
)
