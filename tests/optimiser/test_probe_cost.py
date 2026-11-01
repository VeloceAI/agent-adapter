"""A host's reported spend for a run counts toward the budget beyond what the ledger saw."""

import math

import pytest

from agent_adapter.cost.ledger import Ledger
from agent_adapter.models.instrument import InstrumentedModel
from agent_adapter.optimiser import OptimiserBudget, optimise
from agent_adapter.optimiser.model import CaseRun, EvalCase, HarnessTree
from agent_adapter.optimiser.probe import probe
from tests.optimiser.fakes import CALL_USD, FakeHost, ScriptedProposer, ToyAgent, proposal
from tests.optimiser.loop_support import CITE, config


class BillingHost(FakeHost):
    """An external agent: it never calls the passed model and reports its own spend."""

    def __init__(self, cost: float, *, call_model: bool = False) -> None:
        super().__init__()
        self.cost = cost
        self.call_model = call_model

    async def run(self, tree: HarnessTree, case: EvalCase, *, run_id: str, model) -> CaseRun:
        if self.call_model:
            run = await super().run(tree, case, run_id=run_id, model=model)
        else:
            run = CaseRun(
                case_id=case.id, label="", ok=True, delivered=True, output="80C", transcript=""
            )
        return run.model_copy(update={"metrics": {**run.metrics, "cost_usd": self.cost}})


async def _probe(host, ledger):
    model = InstrumentedModel(ToyAgent().port, ledger=ledger, namespace="t")
    return await probe(
        host, host.surface(), "base", model=model, run_prefix="o", ledger=ledger, concurrency=2
    )


async def test_host_reported_cost_counts_when_the_ledger_saw_nothing():
    ledger = Ledger()
    report = await _probe(BillingHost(0.25), ledger)
    assert [r.metrics["cost_usd"] for r in report.runs] == [0.25, 0.25, 0.25]
    assert report.cost_usd == pytest.approx(0.75)
    assert ledger.total_usd() == pytest.approx(0.75)
    [entry] = ledger.entries("o/base/q1")
    assert (entry.kind, entry.provider, entry.cost_usd) == ("api", "host", 0.25)


async def test_the_ledger_wins_when_it_saw_at_least_the_reported_spend():
    ledger = Ledger()
    report = await _probe(BillingHost(CALL_USD / 2, call_model=True), ledger)
    assert report.runs[0].metrics["cost_usd"] == pytest.approx(CALL_USD)
    assert ledger.total_usd() == pytest.approx(3 * CALL_USD)


async def test_mixed_spend_adds_what_the_ledger_did_not_see():
    ledger = Ledger()
    report = await _probe(BillingHost(9.0, call_model=True), ledger)
    assert report.runs[0].metrics["cost_usd"] == pytest.approx(9.0)
    assert ledger.total_usd() == pytest.approx(27.0)
    host = [e for e in ledger.entries("o/base/q1") if e.provider == "host"]
    assert [e.cost_usd for e in host] == [pytest.approx(9.0 - CALL_USD)]


@pytest.mark.parametrize("cost", [-1.0, math.inf, math.nan, 0.0])
async def test_a_bad_or_zero_reported_cost_is_ignored(cost, caplog):
    ledger = Ledger()
    report = await _probe(BillingHost(cost), ledger)
    assert report.cost_usd == 0.0
    assert ledger.entries() == []
    if cost != 0.0:
        assert "reported cost_usd" in caplog.text


async def test_max_usd_stops_on_host_reported_spend(tmp_path):
    proposer = ScriptedProposer({"know1": proposal(CITE)})
    result = await optimise(
        BillingHost(0.25),
        config(tmp_path, lessons_path="", budget=OptimiserBudget(max_usd=0.5)),
        proposer=proposer.port,
        model=ToyAgent().port,
    )
    assert result.budget.stopped == "max_usd"
    assert result.budget.usd == pytest.approx(0.75)
    assert result.budget.case_runs == 3
    assert proposer.requests == []
