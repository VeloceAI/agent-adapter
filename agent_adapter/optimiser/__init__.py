"""Offline harness optimiser: Propose -> Probe -> Compose over a host's eval cases.

It writes a reviewable patch and report; nothing is applied to the host.
"""

from agent_adapter.optimiser import graders
from agent_adapter.optimiser.apply import layout_from_tree, load_harness
from agent_adapter.optimiser.callable_host import CallableHost, NullModel, RunCase
from agent_adapter.optimiser.cases import load_cases
from agent_adapter.optimiser.hosts import FlowHost, Host, Judge, RegistryHost, apply_tree
from agent_adapter.optimiser.lessons import LessonsStore
from agent_adapter.optimiser.loop import optimise
from agent_adapter.optimiser.model import (
    BudgetUsage,
    Candidate,
    CaseRun,
    Edit,
    EvalCase,
    HarnessTree,
    Lesson,
    OptimiserBudget,
    OptimiserConfig,
    OptimiseResult,
    Score,
    SurfaceCaps,
)

__all__ = [
    "BudgetUsage",
    "CallableHost",
    "Candidate",
    "CaseRun",
    "Edit",
    "EvalCase",
    "FlowHost",
    "HarnessTree",
    "Host",
    "Judge",
    "Lesson",
    "LessonsStore",
    "NullModel",
    "OptimiseResult",
    "OptimiserBudget",
    "OptimiserConfig",
    "RegistryHost",
    "RunCase",
    "Score",
    "SurfaceCaps",
    "apply_tree",
    "graders",
    "layout_from_tree",
    "load_cases",
    "load_harness",
    "optimise",
]
