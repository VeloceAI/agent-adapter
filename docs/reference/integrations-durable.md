# Integrations: durable engines

Flows on Temporal and DBOS: [Run flows on Temporal or DBOS](../how-to/durable-engines.md). Each module imports its SDK lazily: install the extra first.

## `agent_adapter.integrations.temporal`

::: agent_adapter.integrations.temporal
    options:
      members: [ADVANCE, FlowWorkflow, NOTIFY, PENDING, RESUME, WORKFLOW, flow_activities, flow_worker, flow_workflow, notify, pending, result, start_flow]

Also exported here, documented under the canonical path: [`ApprovalNotice`](#agent_adapter.integrations.dbos.ApprovalNotice), [`DurableFlows`](#agent_adapter.integrations.dbos.DurableFlows), [`FlowStepReport`](#agent_adapter.integrations.dbos.FlowStepReport), [`PendingApproval`](#agent_adapter.integrations.dbos.PendingApproval), [`RunIdentity`](#agent_adapter.integrations.dbos.RunIdentity).

## `agent_adapter.integrations.dbos`

::: agent_adapter.integrations.dbos
    options:
      members: [ADVANCE, ApprovalNotice, DbosFlows, DurableFlows, FlowStepReport, NOTIFY_TOPIC, PENDING_EVENT, PendingApproval, RESUME, RunIdentity, StepRetry, WORKFLOW]
