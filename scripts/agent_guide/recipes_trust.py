"""Recipes for the policy file, the audit log, evals from production and the cost advisor."""

from __future__ import annotations

from scripts.agent_guide.recipe_model import Recipe
from scripts.agent_guide.recipes_hosts import EVAL_SETUP

# Test-only inputs: a cheaper model and prices for both; a recorded runs folder.
ADVISOR_SETUP = (
    EVAL_SETUP + '\nsmall = aa.CallableModel(my_llm, provider="local", model="my-llm-small")\n'
    'prices = {"version": "mine", "models": {"my-llm": {"provider": "local", "input": 3.0, '
    '"output": 15.0}, "my-llm-small": {"provider": "local", "input": 0.3, "output": 1.5}}}'
)
RUNS_SETUP = "from pathlib import Path\nPath('.agent_adapter/runs').mkdir(parents=True)"

TRUST: tuple[Recipe, ...] = (
    Recipe(
        "set access, approvals, guards and budgets per tool, team and env in one file",
        "`load_policy(path).apply(executor)` (`agent_adapter.policy`)",
        (
            "from agent_adapter.policy import load_policy",
            "",
            'rules = [{"tools": "get_*", "allow": True, "confirm": "city == \'Oslo\'"}]',
            'policy = load_policy({"default": "deny", "rules": rules})  # or a .yaml/.toml/.json '
            "path",
            "executor = policy.apply(executor)  # the tool loop, frameworks, MCP and HTTP obey it",
            'print(policy.explain("cancel_order"))  # denied; set AA_ENV for per-env rules',
        ),
        index="policy file",
    ),
    Recipe(
        "keep a signed, tamper-evident audit log of approvals, blocks and erasures",
        "`audit(executor, folder=, key=)`, `verify`, `export` (`agent_adapter.audit`)",
        (
            "from agent_adapter.audit import audit, export, verify",
            "",
            'key = "from-your-secret-manager-16+"  # or set AA_AUDIT_KEY',
            'log = audit(executor, folder=".agent_adapter/audit", key=key)  # never args or '
            "content",
            "try:",
            '    executor.run_sync("cancel_order", {"order_id": "o-1"})  # no approval port here',
            "except aa.ApprovalDenied as denied:  # the denial is in the log",
            "    print(denied.kind)",
            "log.stop()  # seals the last segment",
            'print(verify(".agent_adapter/audit", key).summary())  # edits, gaps, reordering',
            'csv_text = export(".agent_adapter/audit", format="csv", key=key)',
        ),
        index="audit log",
    ),
    Recipe(
        "turn recorded production runs into eval cases",
        "`cases_from_runs` + `write_cases` (`agent_adapter.evals`)",
        (
            "from agent_adapter.evals import cases_from_runs, write_cases",
            "",
            "# record with viewer.record(executor, capture_content=True); runs without content "
            "skip",
            'found = cases_from_runs(".agent_adapter/runs", status="failed")',
            'write_cases("eval_cases.jsonl", found.cases)  # fill `expected` for failed runs',
            "print(found.skipped)  # each skipped run with the reason",
            "# CI: agent-adapter eval app:agent eval_cases.jsonl",
        ),
        setup=RUNS_SETUP,
        index="evals from production",
    ),
    Recipe(
        "pick a cheaper model per flow node or stage that keeps the eval score",
        "`agent_adapter.evals.advisor.advise_sync`, then `apply_mapping` yourself",
        (
            "from agent_adapter.evals import ToolLoopTarget, load_cases",
            "from agent_adapter.evals.advisor import advise_sync, apply_mapping",
            "from agent_adapter.evals.graders import contains",
            "",
            'models = {"big": model, "small": small}',
            'target, cases = (lambda m: ToolLoopTarget(m, registry)), load_cases("cases.jsonl")',
            'report = advise_sync(target, cases, [contains()], candidates=models, baseline="big", '
            "prices=prices)",
            "print(report.to_markdown())  # per step (flow node or stage): advice, score, cost, "
            "latency",
            "cheaper = apply_mapping(model, report.mapping, models)  # you apply it: "
            "FlowRunner(model=)",
        ),
        setup=ADVISOR_SETUP,
        index="cost advisor",
    ),
)
