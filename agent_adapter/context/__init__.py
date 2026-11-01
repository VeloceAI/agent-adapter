"""Context compression: shrink the messages sent to a model, under fixed safety invariants."""

from agent_adapter.context.compress import CompressionReport, SkipReason, StepFailure, compress
from agent_adapter.context.conversation import (
    Protection,
    Unit,
    carries_approval_state,
    has_approval_marker,
    split_units,
)
from agent_adapter.context.formats import compress_for, compress_for_sync
from agent_adapter.context.guard import enforce_invariants
from agent_adapter.context.model import CompressingModel
from agent_adapter.context.policy import CompressionLimits, CompressionPolicy
from agent_adapter.context.recall import recall_tool
from agent_adapter.context.relevant import KeepMostRelevant
from agent_adapter.context.rerank import CallableReranker, RerankerPort, RerankResult
from agent_adapter.context.scorer import EmbeddingScorer, MyScorer, RerankerScorer, Scorer
from agent_adapter.context.shrink import ShrinkToolResults
from agent_adapter.context.step import BaseCompressionStep, CompressionStep, StepContext
from agent_adapter.context.steps import (
    CutLongMessages,
    MyOwnStep,
    RemoveDuplicates,
    RemoveOldMessages,
)
from agent_adapter.context.summarize import Summarize, SummarizeLongMessage
from agent_adapter.context.summary_prompt import ContextSummary
from agent_adapter.context.tokens import TokenCounter, TokenMeter, estimate_tokens

__all__ = [
    "BaseCompressionStep",
    "CallableReranker",
    "CompressingModel",
    "CompressionLimits",
    "CompressionPolicy",
    "CompressionReport",
    "CompressionStep",
    "ContextSummary",
    "CutLongMessages",
    "EmbeddingScorer",
    "KeepMostRelevant",
    "MyOwnStep",
    "MyScorer",
    "Protection",
    "RemoveDuplicates",
    "RemoveOldMessages",
    "RerankResult",
    "RerankerPort",
    "RerankerScorer",
    "Scorer",
    "ShrinkToolResults",
    "SkipReason",
    "StepContext",
    "StepFailure",
    "Summarize",
    "SummarizeLongMessage",
    "TokenCounter",
    "TokenMeter",
    "Unit",
    "carries_approval_state",
    "compress",
    "compress_for",
    "compress_for_sync",
    "enforce_invariants",
    "estimate_tokens",
    "has_approval_marker",
    "recall_tool",
    "split_units",
]
