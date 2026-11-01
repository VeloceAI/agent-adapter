# Context compression

Compression policies, steps, scorers, `CompressingModel` and the message-format converters. Concepts: [Context compression](../concepts/context-compression.md).

## `agent_adapter.context`

::: agent_adapter.context
    options:
      members: [BaseCompressionStep, CallableReranker, CompressingModel, CompressionLimits, CompressionPolicy, CompressionReport, CompressionStep, ContextSummary, CutLongMessages, EmbeddingScorer, KeepMostRelevant, MyOwnStep, MyScorer, Protection, RemoveDuplicates, RemoveOldMessages, RerankResult, RerankerPort, RerankerScorer, Scorer, ShrinkToolResults, SkipReason, StepContext, StepFailure, Summarize, SummarizeLongMessage, TokenCounter, TokenMeter, Unit, carries_approval_state, compress, compress_for, compress_for_sync, enforce_invariants, estimate_tokens, has_approval_marker, recall_tool, split_units]

## `agent_adapter.context.formats`

::: agent_adapter.context.formats
    options:
      members: [Conversion, FORMATS, FormatName, Pinned, convert, detect_format, register_format, registered_names]

Also exported here, documented under the canonical path: [`compress_for`](#agent_adapter.context.compress_for), [`compress_for_sync`](#agent_adapter.context.compress_for_sync).
