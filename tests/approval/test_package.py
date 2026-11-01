import agent_adapter.approval as approval


def test_every_public_name_is_exported():
    assert all(hasattr(approval, name) for name in approval.__all__)
    assert {
        "AutoApprove",
        "AutoDeny",
        "CallbackApproval",
        "DeferredApproval",
        "ApprovalStore",
        "ApprovalRecord",
        "MemoryApprovalStore",
        "evaluate_confirm",
    } <= set(approval.__all__)
