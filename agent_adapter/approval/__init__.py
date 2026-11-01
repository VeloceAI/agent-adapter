"""Approval (HITL): confirm rules, approval ports and the approval store."""

from agent_adapter.approval.batch import PreparedApproval, ask_each
from agent_adapter.approval.builtin import (
    ApprovalCallback,
    AutoApprove,
    AutoDeny,
    CallbackApproval,
    ConfirmRule,
    evaluate_confirm,
)
from agent_adapter.approval.deferred import DeferredApproval
from agent_adapter.approval.dynamodb_store import DynamoDBApprovalStore
from agent_adapter.approval.redis_store import RedisApprovalStore
from agent_adapter.approval.sqlalchemy_store import SQLAlchemyApprovalStore
from agent_adapter.approval.sqlite_store import SQLiteApprovalStore
from agent_adapter.approval.store import (
    ApprovalRecord,
    ApprovalStatus,
    ApprovalStore,
    MemoryApprovalStore,
    request_digest,
)
from agent_adapter.core.errors import ApprovalStoreCorrupt
from agent_adapter.core.ports import ApprovalPort
from agent_adapter.core.types import ApprovalDecision, ApprovalRequest, Decision

__all__ = [
    "ApprovalCallback",
    "ApprovalDecision",
    "ApprovalPort",
    "ApprovalRecord",
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalStore",
    "ApprovalStoreCorrupt",
    "AutoApprove",
    "AutoDeny",
    "CallbackApproval",
    "ConfirmRule",
    "Decision",
    "DeferredApproval",
    "DynamoDBApprovalStore",
    "MemoryApprovalStore",
    "PreparedApproval",
    "RedisApprovalStore",
    "SQLAlchemyApprovalStore",
    "SQLiteApprovalStore",
    "ask_each",
    "evaluate_confirm",
    "request_digest",
]
