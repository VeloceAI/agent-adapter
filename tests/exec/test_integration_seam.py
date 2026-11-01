"""The executor seam re-exports the internals integrations use, unchanged."""

from agent_adapter.exec import _approve, _journal, _resume, integration, parallel


def test_every_seam_name_resolves():
    assert all(hasattr(integration, name) for name in integration.__all__)


def test_seam_aliases_are_the_internal_objects():
    assert integration.settle_own is _approve._settle_own
    assert integration.make_answer is _resume._make_answer
    assert integration.visit_scope is _resume._visit_scope
    assert integration.make_journal is _journal._make_journal
    assert integration.gather_unwrapped is parallel._gather_unwrapped


def test_the_seam_is_not_a_package_export():
    import agent_adapter.exec as package

    assert not set(integration.__all__) & set(package.__all__)
