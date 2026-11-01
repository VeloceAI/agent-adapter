"""Shared inputs and the run helper of the optimiser loop tests."""

from agent_adapter.optimiser import OptimiserConfig, optimise
from tests.optimiser.fakes import (
    FakeHost,
    ScriptedProposer,
    ToyAgent,
)

HINT = ("tool/search/prompt_hint", "search first")
CITE = ("prompt/system", "You answer tax questions. Cite the section.")
LESSON = {
    "rows": [
        {
            "path": "prompt/system",
            "candidates": "know1",
            "change": "asked for citations",
            "observed": "3/3 cite 80C",
            "verdict": "helped",
        }
    ],
    "no_entry": False,
    "cue": "answers lack the governing section",
    "lesson": "in prompt/system, ask the agent to cite the section",
    "evidence": "know1: improved 3/3",
    "confidence": "medium",
    "component_paths": ["prompt/system"],
}


def config(tmp_path, **kwargs):
    options = {"out_dir": str(tmp_path / "out"), "lessons_path": str(tmp_path / "lessons.json")}
    options.update(kwargs)
    return OptimiserConfig(**options)


async def run(tmp_path, script, *, host=None, agent=None, **kwargs):
    host = host or FakeHost()
    agent = agent or ToyAgent()
    proposer = ScriptedProposer(script)
    extra = {k: kwargs.pop(k) for k in ("cache", "ledger", "composer") if k in kwargs}
    result = await optimise(
        host, config(tmp_path, **kwargs), proposer=proposer.port, model=agent.port, **extra
    )
    return result, proposer, agent, host
