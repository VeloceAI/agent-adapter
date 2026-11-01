import agent_adapter.flow as flow


def test_every_contract_name_is_exported():
    assert all(hasattr(flow, name) for name in flow.__all__)
    assert {
        "Flow",
        "Call",
        "Map",
        "Join",
        "Route",
        "Merge",
        "MergeRule",
        "keyed",
        "merge_rules",
        "apply_updates",
        "FlowGraph",
        "NodeSpec",
        "RouteSpec",
        "FlowRunner",
        "FlowResult",
        "FlowCheckpoint",
        "node_prompt",
        "node_model",
    } <= set(flow.__all__)
    assert flow.keyed is flow.Merge.keyed
