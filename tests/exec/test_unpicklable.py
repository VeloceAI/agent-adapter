"""Executor, Registry and Ledger refuse pickling with a recipe, not a cryptic lock error."""

import copy
import pickle

import pytest

from agent_adapter import Executor, Ledger, Registry
from agent_adapter.errors import ToolUsageError


@pytest.mark.parametrize("make", [Executor, Registry, Ledger])
@pytest.mark.parametrize("clone", [pickle.dumps, copy.deepcopy, copy.copy])
def test_pickling_names_the_class_and_the_recipe(make, clone):
    value = make()
    with pytest.raises(ToolUsageError, match="cannot be pickled") as found:
        clone(value)
    assert make.__name__ in str(found.value)
    assert "docs/backends/pipelines.md" in str(found.value)
    assert isinstance(found.value, TypeError)  # code catching TypeError still works


def test_a_subclass_is_refused_too():
    class Spy(Executor):
        pass

    with pytest.raises(ToolUsageError, match="Spy cannot be pickled"):
        pickle.dumps(Spy())
