import sys
import types
from dataclasses import dataclass

import pytest

from agent_adapter.core.errors import ConfigError, GuardrailBlocked, MissingDependency
from agent_adapter.hooks.builtin.guards import PresidioGuard
from tests.hooks.builtin.helpers import ok, tool_inv


@dataclass
class Found:
    entity_type: str
    start: int
    end: int
    score: float = 0.9


class FakeAnalyzer:
    def __init__(self):
        self.calls = []

    def analyze(self, *, text, language, entities, score_threshold):
        self.calls.append((language, entities, score_threshold))
        at = text.find("Ana")
        return [Found("PERSON", at, at + 3)] if at >= 0 else []


class FakeAnonymizer:
    def anonymize(self, *, text, analyzer_results):
        for found in sorted(analyzer_results, key=lambda f: -f.start):
            text = text[: found.start] + f"<{found.entity_type}>" + text[found.end :]
        return types.SimpleNamespace(text=text)


async def test_blocks_with_entity_types_only():
    analyzer = FakeAnalyzer()
    guard = PresidioGuard(analyzer=analyzer, entities=["PERSON"], language="de", threshold=0.7)
    with pytest.raises(GuardrailBlocked) as info:
        await guard.before_tool(tool_inv(arguments={"user": "Ana Lee", "password": "x"}))
    assert info.value.reason == "found PERSON"
    assert analyzer.calls[0] == ("de", ["PERSON"], 0.7)


async def test_clean_text_passes():
    guard = PresidioGuard(analyzer=FakeAnalyzer())
    assert await guard.before_tool(tool_inv(arguments={"user": "Bo"})) is None


async def test_redacts_with_the_anonymizer():
    guard = PresidioGuard(
        analyzer=FakeAnalyzer(), anonymizer=FakeAnonymizer(), on="result", action="redact"
    )
    assert (await guard.after_tool(tool_inv(), ok("Hi Ana"))).value == "Hi <PERSON>"


async def test_without_an_anonymizer_redact_blocks():
    guard = PresidioGuard(analyzer=FakeAnalyzer(), on="result")
    with pytest.raises(GuardrailBlocked):
        await guard.after_tool(tool_inv(), ok("Hi Ana"))


def test_default_engines_come_from_presidio(monkeypatch):
    analyzer_mod = types.ModuleType("presidio_analyzer")
    analyzer_mod.AnalyzerEngine = FakeAnalyzer
    anonymizer_mod = types.ModuleType("presidio_anonymizer")
    anonymizer_mod.AnonymizerEngine = FakeAnonymizer
    monkeypatch.setitem(sys.modules, "presidio_analyzer", analyzer_mod)
    monkeypatch.setitem(sys.modules, "presidio_anonymizer", anonymizer_mod)
    guard = PresidioGuard(on="result", action="redact")
    assert isinstance(guard.analyzer, FakeAnalyzer)
    assert isinstance(guard.anonymizer, FakeAnonymizer)
    assert PresidioGuard().anonymizer is None  # only redaction needs it


@pytest.mark.parametrize("missing", ["presidio_analyzer", "presidio_anonymizer"])
def test_missing_presidio_names_the_extra(monkeypatch, missing):
    analyzer_mod = types.ModuleType("presidio_analyzer")
    analyzer_mod.AnalyzerEngine = FakeAnalyzer
    monkeypatch.setitem(sys.modules, "presidio_analyzer", analyzer_mod)
    monkeypatch.setitem(sys.modules, missing, None)
    with pytest.raises(MissingDependency, match=r"agent-adapter\[presidio\]"):
        PresidioGuard(on="result", action="redact")


def test_threshold_is_checked():
    with pytest.raises(ConfigError, match="threshold"):
        PresidioGuard(analyzer=FakeAnalyzer(), threshold=2)


@pytest.mark.contract
async def test_real_presidio_finds_an_email():
    pytest.importorskip("presidio_analyzer")
    pytest.importorskip("presidio_anonymizer")
    guard = PresidioGuard(entities=["EMAIL_ADDRESS"], on="result", action="redact")
    redacted = await guard.after_tool(tool_inv(), ok("write to ana@example.com"))
    assert "ana@example.com" not in redacted.value


def test_install_hint_works_in_every_shell():
    from agent_adapter.hooks.builtin.guards.presidio import INSTALL

    assert INSTALL == 'pip install "agent-adapter[presidio]"'
