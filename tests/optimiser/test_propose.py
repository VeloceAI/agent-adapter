import pytest

from agent_adapter.core.errors import (
    BudgetExceeded,
    ModelAuthError,
    ModelBadRequest,
    ModelRateLimited,
    OptimiserError,
)
from agent_adapter.optimiser.model import Candidate
from agent_adapter.optimiser.propose import propose
from tests.optimiser.fakes import FakeHost, ScriptedProposer, keep_base, proposal

HOST = FakeHost()
BASE = Candidate(label="base", dim="base", edits=(), tree=HOST.surface(), parent_label="")


async def run(script, *, dim="knowledge", host=HOST, label="know1"):
    proposer = ScriptedProposer({label: script})
    cand = await propose(
        proposer.port,
        label=label,
        dim=dim,
        parent=BASE,
        caps=host.caps(),
        host=host,
        system="SYSTEM",
        evidence="EVIDENCE",
        seed=7,
    )
    return cand, proposer


async def test_valid_proposal_becomes_a_candidate():
    cand, proposer = await run(proposal(("prompt/system", "Cite the section."), notes="why"))
    assert cand.label == "know1" and cand.dim == "knowledge" and cand.parent_label == "base"
    assert cand.tree.files["prompt/system"] == "Cite the section."
    assert [e.path for e in cand.edits] == ["prompt/system"]
    assert cand.edits[0].reason == "because prompt/system"
    assert (cand.attempts, cand.fell_back, cand.notes) == (1, False, "why")
    request = proposer.requests[0]
    assert request.stage == "propose:know1" and request.seed == 7
    assert request.output_schema.name == "harness_proposal"
    assert [m.role for m in request.messages] == ["system", "user"]
    assert request.messages[1].content == "EVIDENCE"


async def test_keep_base_and_no_op_edits_keep_the_parent():
    cand, _ = await run(keep_base("fine as is"))
    assert cand.edits == () and cand.tree == BASE.tree and not cand.fell_back
    assert cand.notes == "fine as is"
    same, _ = await run(proposal(("prompt/system", BASE.tree.files["prompt/system"])))
    assert same.edits == () and same.tree == BASE.tree


async def test_deleting_a_component():
    cand, _ = await run(proposal(("tool/search/description", None)), dim="procedure", label="p")
    assert "tool/search/description" not in cand.tree.files
    assert cand.edits[0].content is None


async def test_invalid_proposal_is_retried_once_with_the_problems():
    cand, proposer = await run(
        [
            proposal(("tool/search/description", "x")),  # off-dimension for knowledge
            proposal(("prompt/system", "Cite it.")),
        ]
    )
    assert cand.attempts == 2 and not cand.fell_back
    retry = proposer.requests[1].messages
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert "the knowledge editor may only edit prompt/*, flow/* paths" in retry[-1].content


async def test_second_failure_falls_back_to_the_parent():
    too_long = "x" * 20_000
    cand, proposer = await run([proposal(("prompt/system", too_long))] * 2)
    assert cand.fell_back and cand.tree == BASE.tree and cand.edits == ()
    assert cand.attempts == 2 and len(proposer.requests) == 2
    assert "20000 chars, cap is 12000" in cand.error


async def test_host_validation_and_bad_shapes_are_rejected():
    host = FakeHost(
        problems=lambda tree: (
            ["must mention tax"] if "tax" not in tree.files["prompt/system"] else []
        )
    )
    cand, _ = await run([proposal(("prompt/system", "Cite it."))] * 2, host=host)
    assert cand.fell_back and cand.error == "must mention tax"
    bad, _ = await run([{"edits": [{"path": 3}]}] * 2)
    assert bad.fell_back and "edits.0.path" in bad.error
    dup, _ = await run([proposal(("prompt/system", "a"), ("prompt/system", "b"))] * 2)
    assert dup.fell_back and "prompt/system: edited more than once" in dup.error
    reasonless = proposal(("prompt/system", "a"))
    reasonless["edits"][0]["reason"] = " "
    blank, _ = await run([reasonless] * 2)
    assert "prompt/system: an edit needs a reason" in blank.error


async def test_invalid_json_is_retried():
    cand, _ = await run(["not json", proposal(("prompt/system", "Cite it."))])
    assert cand.attempts == 2 and not cand.fell_back


async def test_model_errors_fall_back_without_retry_and_budget_stops_propagate():
    cand, proposer = await run(ModelRateLimited("slow down", provider="openai"))
    assert cand.fell_back and cand.attempts == 1 and len(proposer.requests) == 1
    assert cand.error == "model-rate-limited: slow down"
    with pytest.raises(BudgetExceeded):
        await run(BudgetExceeded(1.0, 1.5))


@pytest.mark.parametrize(
    "error",
    [ModelAuthError("bad key", provider="openai"), ModelBadRequest("bad", provider="openai")],
)
async def test_fatal_model_errors_raise(error):
    with pytest.raises(OptimiserError, match=type(error).kind):
        await run(error)
