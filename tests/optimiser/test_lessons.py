import json
import os

import pytest

from agent_adapter.core.errors import (
    BudgetExceeded,
    ModelAuthError,
    ModelUnavailable,
    OptimiserError,
)
from agent_adapter.optimiser._files import write_atomic
from agent_adapter.optimiser.lessons import LessonsStore, reflect
from agent_adapter.optimiser.model import HarnessTree, Lesson
from tests.optimiser.fakes import ScriptedProposer


def lesson(
    id_, *, domain="tax", paths=("tool/search/prompt_hint",), at="2026-10-01T00:00:00+00:00"
):
    return Lesson(
        id=id_,
        domain=domain,
        created_at=at,
        cue="cue",
        lesson="do x",
        evidence="e",
        confidence="low",
        component_paths=list(paths),
        run="runs/x",
        adopted=False,
    )


def test_store_round_trips_and_appends_atomically(tmp_path):
    store = LessonsStore(tmp_path / "dir" / "lessons.json")
    assert store.load() == []
    store.add([lesson("a")])
    store.add([lesson("b"), lesson("a")])  # same id is not duplicated
    data = json.loads((tmp_path / "dir" / "lessons.json").read_text("utf-8"))
    assert data["version"] == 1 and [x["id"] for x in data["lessons"]] == ["a", "b"]
    assert [x.id for x in store.load()] == ["a", "b"]
    assert not [p for p in os.listdir(tmp_path / "dir") if p.endswith(".tmp")]


def test_relevant_filters_by_domain_paths_and_limit(tmp_path):
    store = LessonsStore(tmp_path / "l.json")
    store.add(
        [
            lesson("old", at="2026-01-01T00:00:00+00:00"),
            lesson("new", at="2026-09-01T00:00:00+00:00"),
            lesson("other", domain="hr"),
            lesson("absent", paths=("tool/gone/description",)),
            lesson("general", paths=(), at="2026-05-01T00:00:00+00:00"),
        ]
    )
    tree = HarnessTree(files={"tool/search/prompt_hint": "x", "prompt/system": "s"})
    assert [x.id for x in store.relevant(domain="tax", tree=tree, limit=8)] == [
        "new",
        "general",
        "old",
    ]
    assert [x.id for x in store.relevant(domain="tax", tree=tree, limit=1)] == ["new"]
    # a prompt_hint path counts as present when its tool's description is present
    tree2 = HarnessTree(files={"tool/search/description": "d"})
    assert "new" in [x.id for x in store.relevant(domain="tax", tree=tree2, limit=8)]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"version": 9, "lessons": []}),
        json.dumps({"version": 1, "lessons": [{"id": 1}]}),
    ],
)
def test_corrupt_store_raises(tmp_path, content):
    path = tmp_path / "l.json"
    path.write_text(content, "utf-8")
    with pytest.raises(OptimiserError, match="lessons file"):
        LessonsStore(path).load()


def test_write_atomic_cleans_up_on_failure(tmp_path, monkeypatch):
    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        write_atomic(tmp_path / "f.json", "x")
    assert os.listdir(tmp_path) == []


def reflection(**overrides):
    reply = {
        "rows": [
            {
                "path": "tool/search/prompt_hint",
                "candidates": "proc1",
                "change": "added a hint",
                "observed": "searched on 3/3",
                "verdict": "helped",
            },
        ],
        "no_entry": False,
        "cue": "when answers lack a citation",
        "lesson": "in tool/search/prompt_hint, search before answering",
        "evidence": "proc1: improved 3/3",
        "confidence": "medium",
        "component_paths": ["tool/search/prompt_hint", "not a path"],
    }
    reply.update(overrides)
    return reply


async def run_reflect(reply, *, chosen_paths=("tool/search/prompt_hint",)):
    composer = ScriptedProposer({"reflect": reply})
    result = await reflect(
        composer.port,
        domain="tax",
        run="runs/opt",
        evidence="EVIDENCE",
        adopted_paths=set(chosen_paths),
        created_at="2026-10-30T12:00:00+00:00",
    )
    return result, composer


async def test_reflect_builds_a_lesson():
    (found, why), composer = await run_reflect(reflection())
    assert why == ""
    assert found.domain == "tax" and found.run == "runs/opt" and found.adopted
    assert found.component_paths == ["tool/search/prompt_hint"]
    assert found.id.startswith("ls_2026-10-30_") and len(found.id) == len("ls_2026-10-30_") + 8
    request = composer.requests[0]
    assert request.stage == "reflect" and request.messages[1].content == "EVIDENCE"
    (again, _), _ = await run_reflect(reflection())
    assert again.id == found.id  # deterministic
    (not_adopted, _), _ = await run_reflect(reflection(), chosen_paths=())
    assert not not_adopted.adopted


@pytest.mark.parametrize(
    ("reply", "why"),
    [
        (reflection(no_entry=True), "NO_ENTRY"),
        (reflection(cue=" "), "incomplete lesson"),
        (
            reflection(
                rows=[
                    {
                        "path": "tool/search/prompt_hint",
                        "candidates": "proc1",
                        "change": "c",
                        "observed": "o",
                        "verdict": "confounded",
                    },
                    {
                        "path": "prompt/system",
                        "candidates": "know1",
                        "change": "c",
                        "observed": "o",
                        "verdict": "helped",
                    },
                ]
            ),
            "rests only on confounded rows",
        ),
        (reflection(rows=[]), "no evidence rows for the lesson's component paths"),
        ({"rows": "nope"}, "invalid reply"),
        (ModelUnavailable("down", provider="openai"), "model-unavailable"),
    ],
)
async def test_reflect_drops_no_entry_confounded_and_failures(reply, why):
    (found, reason), _ = await run_reflect(reply)
    assert found is None and reason == why


async def test_reflect_budget_stop_propagates():
    with pytest.raises(BudgetExceeded):
        await run_reflect(BudgetExceeded(1.0, 1.0))


async def test_reflect_reason_when_no_row_is_about_the_lessons_components():
    (found, why), _ = await run_reflect(reflection(component_paths=["prompt/system"]))
    assert found is None and why == "no evidence rows for the lesson's component paths"


async def test_reflect_fatal_errors_raise():
    with pytest.raises(OptimiserError, match="model-auth-failed"):
        await run_reflect(ModelAuthError("bad key", provider="openai"))
