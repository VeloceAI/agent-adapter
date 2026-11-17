import pytest

import agent_adapter as aa
from agent_adapter.compose import Blank, Template, compose
from agent_adapter.errors import ToolUsageError
from agent_adapter.exec import LoopAgent
from agent_adapter.exec.checkpoints import MemoryCheckpointStore
from agent_adapter.flow import FlowRunner
from agent_adapter.tracing import EventLog
from tests.compose.helpers import Script, executor, parsed, prompt_of

ROWS = {"rows": [{"Region": "EU", "Sales": "10"}, {"Region": "US", "Sales": "12"}]}


def report(summary: str = "Summarise {{ title }}") -> Template:
    return Template(
        "# {{ title }}\n\n{{ summary }}\n\n{{ sales }}\n\n{{ outlook }}\n",
        [
            Blank("summary", summary, max_words=20),
            Blank("sales", "Sales by region", "table", columns=["Region", "Sales"]),
            Blank("outlook", "Outlook given {{ summary }} and {{ sales }}"),
        ],
        name="report",
    )


async def test_a_blank_waits_for_the_blanks_its_prompt_uses():
    script = Script({"summary": "Sales grew.", "sales": parsed(ROWS), "outlook": "Good."})
    result = await compose(
        report(), model=script.model, inputs={"title": "Q3"}, executor=executor()
    )
    assert script.order()[-1] == "outlook"
    sent = prompt_of(script.calls("outlook")[0])
    assert "Sales grew." in sent and "| US | 12 |" in sent
    assert prompt_of(script.calls("summary")[0]) == "Summarise Q3"
    assert result.ok and result.problems == {}
    assert result.markdown.startswith("# Q3\n\nSales grew.\n\n| Region | Sales |")
    assert result.blanks["sales"].value == ROWS["rows"]
    assert {b.attempts for b in result.blanks.values()} == {1}


async def test_independent_blanks_fill_in_parallel_under_the_limit():
    blanks = [Blank(f"part{i}", f"Part {i}") for i in range(4)]
    template = Template(" ".join(f"{{{{ part{i} }}}}" for i in range(4)), blanks)
    script = Script({f"part{i}": f"p{i}" for i in range(4)}, delay=0.05)
    result = await compose(template, model=script.model, executor=executor())
    assert script.most == 4
    assert result.markdown == "p0 p1 p2 p3"
    capped = Script({f"part{i}": f"p{i}" for i in range(4)}, delay=0.05)
    await compose(template, model=capped.model, executor=executor(), concurrency=2)
    assert capped.most == 2


async def test_a_failing_blank_is_retried_alone_with_its_problems():
    too_long = " ".join(["word"] * 30)
    script = Script({"summary": [too_long, "Sales grew."], "sales": parsed(ROWS), "outlook": "Ok."})
    result = await compose(
        report(), model=script.model, inputs={"title": "Q3"}, executor=executor()
    )
    assert [len(script.calls(p)) for p in ("summary", "sales", "outlook")] == [2, 1, 1]
    retry = script.calls("summary")[1].messages
    assert retry[-2].content == too_long
    assert "30 words; the limit is 20" in retry[-1].content
    assert result.blanks["summary"].attempts == 2 and result.ok


async def test_a_blank_that_fails_twice_is_kept_with_its_problems():
    bad_rows = parsed({"rows": [{"Region": "EU"}]})
    script = Script({"summary": "S.", "sales": [bad_rows, "not json"], "outlook": "O."})
    result = await compose(
        report(), model=script.model, inputs={"title": "Q3"}, executor=executor()
    )
    assert len(script.calls("sales")) == 2
    assert result.problems == {"sales": ("the answer is not valid JSON",)}
    assert not result.ok and not result.blanks["sales"].ok
    assert "row 1 lacks columns ['Sales']" in script.calls("sales")[1].messages[-1].content


async def test_number_list_and_diagram_blanks():
    template = Template(
        "{{ total }}\n{{ steps }}\n{{ flow }}",
        [
            Blank("total", "Total", "number"),
            Blank("steps", "Steps", "list"),
            Blank("flow", "Flow", "diagram"),
        ],
    )
    nodes = [{"id": "a", "label": "Start"}, {"id": "b", "label": "End"}]
    script = Script(
        {
            "total": parsed({"value": 42}),
            "steps": parsed({"items": ["one", "two"]}),
            "flow": parsed(
                {"nodes": nodes, "edges": [{"source": "a", "target": "b", "label": ""}]}
            ),
        }
    )
    result = await compose(template, model=script.model, executor=executor())
    assert result.markdown.splitlines()[:3] == ["42", "- one", "- two"]
    assert '  n1["Start"]' in result.markdown and "  n1 --> n2" in result.markdown
    assert script.calls("total")[0].output_schema.json_schema["required"] == ["value"]


async def test_a_blank_can_use_a_loop_agent_with_tools():
    registry = aa.Registry()

    @registry.tool
    def lookup(region: str) -> str:
        """Sales for a region."""
        return f"{region}: 10"

    def agent_reply(request: aa.LLMRequest):
        if request.messages[-1].role == "tool":
            return f"From the tool: {request.messages[-1].content}"
        return {"tool_calls": [{"id": "c1", "name": "lookup", "arguments": {"region": "EU"}}]}

    agent = LoopAgent(
        aa.CallableModel(agent_reply, model="agent"), registry, instructions="Be exact."
    )
    template = Template("{{ eu }}", [Blank("eu", "EU sales?", agent=agent)])
    result = await compose(template, executor=aa.Executor(registry))
    assert result.markdown == "From the tool: EU: 10"


async def test_cache_reruns_only_the_changed_blank_and_its_users():
    shared = executor(cache=aa.MemoryCache())
    answers = {"summary": "S.", "sales": parsed(ROWS), "outlook": "O."}
    first = Script(answers)
    options = {"inputs": {"title": "Q3"}, "executor": shared, "thread_id": "doc-1"}
    await compose(report(), model=first.model, **options)
    assert len(first.requests) == 3
    again = Script(answers)
    result = await compose(report(), model=again.model, **options)
    assert again.requests == [] and all(b.cached for b in result.blanks.values())
    changed = report("Summarise {{ title }} briefly")
    third = Script({**answers, "summary": "Short."})
    result = await compose(changed, model=third.model, **options)
    assert sorted(third.order()) == ["outlook", "summary"]
    assert result.blanks["sales"].cached and not result.blanks["outlook"].cached
    assert "Short." in result.markdown


async def test_events_and_on_blank_follow_every_blank():
    events, seen = EventLog(), []
    script = Script({"summary": "S.", "sales": parsed(ROWS), "outlook": "O."})
    await compose(
        report(),
        model=script.model,
        inputs={"title": "Q3"},
        executor=executor(sink=events),
        on_blank=seen.append,
    )
    started = [e.data["blank"] for e in events.events if e.type == "compose.blank_started"]
    filled = [e for e in events.events if e.type == "compose.blank_filled"]
    assert sorted(started) == ["outlook", "sales", "summary"]
    assert {e.data["blank"]: e.data["ok"] for e in filled} == {
        "summary": True,
        "sales": True,
        "outlook": True,
    }
    assert [b.name for b in seen][-1] == "outlook" and seen[0].markdown


async def test_cost_is_counted_per_blank():
    prices = {"scripted": {"provider": "local", "input": 1_000_000.0, "output": 0.0}}
    ledger = aa.Ledger(prices=aa.Prices.load({"version": "t", "models": prices}))
    usage = {"input_tokens": 1, "output_tokens": 0}
    script = Script(
        {
            "summary": {"text": "S.", "usage": usage},
            "sales": parsed(ROWS, usage),
            "outlook": {"text": "O.", "usage": usage},
        }
    )
    result = await compose(
        report(), model=script.model, inputs={"title": "Q3"}, executor=executor(ledger=ledger)
    )
    assert result.blanks["summary"].cost_usd == pytest.approx(1.0)
    assert result.cost == pytest.approx(3.0)


async def test_missing_inputs_and_models_are_refused():
    with pytest.raises(ToolUsageError, match="needs inputs"):
        await compose(report(), model=Script({}).model, executor=executor())
    template = Template("{{ a }}", [Blank("a", "x")])
    with pytest.raises(aa.AdapterError, match="no agent"):
        await compose(template, executor=executor())


async def test_a_runner_with_checkpoints_saves_each_layer():
    store = MemoryCheckpointStore()
    shared = executor()
    script = Script({"summary": "S.", "sales": parsed(ROWS), "outlook": "O."})
    runner = FlowRunner(shared, checkpoints=store)
    result = await compose(report(), model=script.model, inputs={"title": "Q3"}, runner=runner)
    assert result.ok and "O." in result.markdown
    saved = await store.latest(result.run_id)
    assert saved is not None and saved.step == 4  # begin, two layers, the editor
