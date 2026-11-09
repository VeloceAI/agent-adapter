---
description: Six ways teams put Agent Adapter to work, each with code you can run.
---

# Use cases

Each use case names the problem, the features that solve it and an example that runs in this
repository with no API keys. These are example scenarios, not customer stories.

<div class="grid cards" markdown>

-   :material-headset:{ .lg .middle } **Customer support that cannot overspend**

    ---

    An agent answers orders, refunds and cancellations. Refunds wait for a person; each reply
    has a cost and a trace you can open later.

    **Uses:** `confirm=True` approvals, `BudgetHook`, the run viewer.
    **Run:** `examples/approval.py` · [Human in the loop](how-to/human-in-the-loop.md)

-   :material-bank:{ .lg .middle } **Finance operations with an audit trail**

    ---

    Payment and invoice tools that spend real money. Every call is capped per user, written to
    an audit log and kept free of secrets in logs and errors.

    **Uses:** per-user budgets, the audit log, redaction guards.
    **Run:** `examples/cookbook_user_budget.py`, `examples/audit_log.py` · [Audit](concepts/audit.md)

-   :material-database-arrow-right:{ .lg .middle } **Legacy systems to agent tools**

    ---

    An old service has an OpenAPI spec and classes no one wants to touch. Import the spec,
    wrap the classes, and keep the same approval and budget rules.

    **Uses:** OpenAPI import, `registry.wrap()`, `agent-adapter migrate`.
    **Run:** `examples/openapi_import.py`, `examples/howto_wrap_legacy.py` · [Migration](migration/index.md)

-   :material-shield-account:{ .lg .middle } **Compliance and whistleblowing intake**

    ---

    Reports arrive by chat. Personal data is redacted before it reaches the model, a case is
    filed only after a reviewer approves, and every step is in the audit log.

    **Uses:** guardrails, approvals, the audit log, a policy file.
    **Run:** `examples/cookbook_redact.py`, `examples/policy_shop.py` · [Guardrails](concepts/guardrails.md)

-   :material-graph:{ .lg .middle } **Research with graph flows**

    ---

    Several agents search, read and write in parallel waves, then merge. The same `Flow` runs
    on its own or compiles to LangGraph and Google ADK.

    **Uses:** `Flow`, `run_plan`, multi-agent handoffs, durable runs.
    **Run:** `examples/flow_research.py`, `examples/multi_agent.py` · [Flows](concepts/flows.md)

-   :material-server:{ .lg .middle } **Tools inside your backend**

    ---

    A FastAPI, Django or Celery service gets agent tools without an agent framework, with
    the same safety rules as the agents.

    **Uses:** `executor.run`, `run_sync`, the HTTP tool server, MCP.
    **Run:** `examples/backend_fastapi.py`, `examples/backend_celery.py` · [Sync code and backends](how-to/sync-backends.md)

</div>

## Run one

```bash
git clone https://github.com/VeloceAI/agent-adapter.git && cd agent-adapter
pip install -e . && python examples/approval.py
```

Some examples need an extra (FastAPI, OpenAPI, Celery): the [examples gallery](examples.md)
names it in each row.

Want an AI to map your own project to these? Paste the [prompt for any AI](get-started/ai-prompt.md),
or add the [skill](skill.md) to your coding agent.
