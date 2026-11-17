import json

from agent_adapter.compose import Blank, Template, Verifier, compose
from agent_adapter.retrieval import Document
from tests.compose.helpers import Script, executor, parsed, prompt_of

LONG = Document(
    id="long", text=("Filler words here. " * 60) + "Revenue grew 12% in Q3. " + "x " * 300
)
ANSWERS = {"x": "Revenue [grew 12%](cite:long).", "y": "Costs held."}


def two() -> Template:
    return Template("{{ x }}\n\n{{ y }}", [Blank("x", "Revenue"), Blank("y", "Costs")])


def issues(*items: tuple[str, str]) -> dict:
    return parsed({"issues": [{"blank": b, "issue": i} for b, i in items]})


async def test_the_verifier_sees_only_cited_phrases_in_short_windows():
    script = Script(ANSWERS)
    result = await compose(two(), model=script.model, sources=[LONG], executor=executor())
    (sent,) = script.calls("aa_verify")
    data = json.loads(prompt_of(sent))
    (claim,) = data["claims"]
    assert claim["blank"] == "x" and claim["phrase"] == "grew 12%"
    assert "Revenue grew 12% in Q3." in claim["source"] and len(claim["source"]) <= 300
    assert "rules" not in data and result.ok


async def test_a_flagged_blank_is_redone_once_alone_and_rechecked():
    flagged = issues(("x", "12% is not in the source"))
    script = Script({**ANSWERS, "x": [ANSWERS["x"], "Revenue [grew](cite:long)."]})
    script.answers["aa_verify"] = [flagged, issues()]
    result = await compose(two(), model=script.model, sources=[LONG], executor=executor())
    assert [len(script.calls(p)) for p in ("x", "y", "aa_verify")] == [2, 1, 2]
    redo = prompt_of(script.calls("x")[1])
    assert "An earlier answer was:\nRevenue [grew 12%](cite:long)." in redo
    assert "- 12% is not in the source" in redo
    assert result.ok and result.text.startswith("Revenue grew [1].")
    assert result.blanks["x"].edited


async def test_an_issue_left_after_the_redo_is_reported_not_retried():
    still = issues(("x", "unsupported"))
    script = Script(ANSWERS)
    script.answers["aa_verify"] = still
    result = await compose(two(), model=script.model, sources=[LONG], executor=executor())
    assert len(script.calls("x")) == 2 and len(script.calls("aa_verify")) == 2
    assert result.issues == {"x": ("unsupported",)} and not result.ok


async def test_rules_and_a_small_model():
    small = Script({"aa_verify": issues(("y", "says nothing"), ("ghost", "ignored"))})
    big = Script({**ANSWERS, "x": "Revenue grew 12%."})
    verifier = Verifier(model=small.model, rules=["Each part states a number."])
    result = await compose(two(), model=big.model, verifier=verifier, executor=executor())
    data = json.loads(prompt_of(small.calls("aa_verify")[0]))
    assert data["rules"] == ["Each part states a number."]
    assert data["parts"] == {"x": "Revenue grew 12%.", "y": "Costs held."}
    assert big.calls("aa_verify") == [] and len(big.calls("y")) == 2
    assert result.issues == {"y": ("says nothing",)}


async def test_code_checks_run_without_a_model_call():
    broken = Template('<div><p>{{ x | "X" }}</p>', format="html")
    script = Script({"x": ["", ""]})
    result = await compose(broken, model=script.model, verifier=False, executor=executor())
    assert script.calls("aa_verify") == []
    assert result.issues["x"] == ("the answer is empty",)
    assert result.issues["document"] == ("<div> is never closed",)
    no_sources = Script(ANSWERS)
    await compose(two(), model=no_sources.model, executor=executor())
    assert no_sources.calls("aa_verify") == []
