"""Approval (HITL): confirm rules, approval ports, the approval store and approval memory."""

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
from agent_adapter.approval.memory import ApprovalContext, SimilarDecision, approval_context
from agent_adapter.approval.redis_store import RedisApprovalStore
from agent_adapter.approval.rules import (
    AcceptedRule,
    Condition,
    ProposedRule,
    RuleConfirm,
    RuleEvidence,
    accept,
    confirm_from_config,
    propose_rules,
)
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
    "AcceptedRule",
    "ApprovalCallback",
    "ApprovalContext",
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
    "Condition",
    "ConfirmRule",
    "Decision",
    "DeferredApproval",
    "DynamoDBApprovalStore",
    "MemoryApprovalStore",
    "PreparedApproval",
    "ProposedRule",
    "RedisApprovalStore",
    "RuleConfirm",
    "RuleEvidence",
    "SQLAlchemyApprovalStore",
    "SQLiteApprovalStore",
    "SimilarDecision",
    "accept",
    "approval_context",
    "ask_each",
    "confirm_from_config",
    "evaluate_confirm",
    "propose_rules",
    "request_digest",
]
