import pytest

import agent_adapter as aa
from agent_adapter.compose import Blank, Template, compose
from agent_adapter.retrieval import Document
from tests.compose.helpers import Script, executor, parsed

TOPICS = ["revenue", "costs", "hiring", "churn", "pricing", "europe", "support", "risk"]
SOURCES = [
    Document(
        id=f"s{i}", text=" ".join(f"Fact {j} about {TOPICS[i % 8]} in Q3." for j in range(100))
    )
    for i in range(10)
]


def chars(request: aa.LLMRequest) -> int:
    return sum(len(message.content) for message in request.messages)


def six() -> Template:
    text = [Blank(t, f"Write about {t}.") for t in TOPICS[:4]]
    table = Blank("pricing", "Pricing by plan", "table", columns=["Plan", "Price"])
    items = Blank("risks", "Main risks", "list")
    body = "\n\n".join(f"{{{{ {b.name} }}}}" for b in [*text, table, items])
    return Template(body, [*text, table, items], name="six")


def answers() -> dict:
    said = {
        t: f"In Q3 {t} [moved a lot](cite:s{i}). " + " ".join(["More detail here."] * 15)
        for i, t in enumerate(TOPICS[:4])
    }
    rows = {"rows": [{"Plan": "Pro", "Price": "10"}]}
    return {**said, "pricing": parsed(rows), "risks": parsed({"items": ["Churn"]})}


async def test_prompts_stay_small_for_six_blanks_and_ten_sources():
    script = Script(answers())
    result = await compose(six(), model=script.model, sources=SOURCES, executor=executor())
    every_source = sum(len(doc.text) for doc in SOURCES)
    blank_prompts = [chars(r) for r in script.requests if r.stage.split(".")[-1] in TOPICS[:4]]
    (editor,) = script.calls("aa_editor")
    (verifier,) = script.calls("aa_verify")
    assert every_source > 25_000
    assert max(blank_prompts) < 3_200  # 4 passages of 600 characters, the hint and the rules
    assert chars(editor) < 2_000  # the 4 text blanks only, never the sources or the table
    assert chars(verifier) < 2_200  # 4 cited phrases with 300 characters of source each
    assert result.ok, result.issues


async def test_usage_per_stage_and_cost():
    prices = {"scripted": {"provider": "local", "input": 1_000_000.0, "output": 0.0}}
    ledger = aa.Ledger(prices=aa.Prices.load({"version": "t", "models": prices}))
    usage = {"input_tokens": 1, "output_tokens": 2}
    script = Script(
        {
            "a": {"text": "A " * 100, "usage": usage},
            "b": {"text": "B " * 100, "usage": usage},
            "aa_editor": {**parsed({"revisions": []}), "usage": usage},
        }
    )
    template = Template('{{ a | "A" }} {{ b | "B" }}')
    result = await compose(template, model=script.model, executor=executor(ledger=ledger))
    assert result.usage["blanks"] == aa.compose.StageUsage(2, 2, 4, pytest.approx(2.0))
    assert result.usage["editor"].calls == 1 and result.usage["verifier"].calls == 0
    assert result.cost == pytest.approx(3.0)


async def test_max_usd_stops_the_run():
    prices = {"scripted": {"provider": "local", "input": 1_000_000.0, "output": 0.0}}
    ledger = aa.Ledger(prices=aa.Prices.load({"version": "t", "models": prices}))
    usage = {"input_tokens": 1, "output_tokens": 0}
    script = Script({"a": {"text": "A", "usage": usage}, "b": {"text": "B", "usage": usage}})
    template = Template('{{ a | "A" }}\n{{ b | "B" | uses(a) }}')
    with pytest.raises(aa.AdapterError) as caught:
        await compose(template, model=script.model, executor=executor(ledger=ledger), max_usd=0.5)
    assert "budget" in str(caught.value).lower()
    assert script.order() == ["a"]
