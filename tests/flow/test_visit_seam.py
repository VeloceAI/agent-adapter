"""The flow seam re-exports the internals flow compilers use, unchanged."""

from agent_adapter.flow import _gates, _pause, visit


def test_every_seam_name_resolves():
    assert all(hasattr(visit, name) for name in visit.__all__)


def test_seam_aliases_are_the_internal_objects():
    assert visit.node_gate_request is _gates._request
    assert visit.ask_node_gate is _gates._ask
    assert visit.superseded is _pause._superseded


def test_the_seam_is_not_a_package_export():
    import agent_adapter.flow as package

    assert not set(visit.__all__) & set(package.__all__)
