"""GET /approvals/<id>: approvers see similar past decisions; the call still waits."""

from typing import Any

from tests.server.helpers import call, invoke, make_registry, make_service

EMAIL = {"to": "a@example.com", "body": "hi"}


async def parked(service: Any) -> str:
    reply = await invoke(service, "send_email", EMAIL)
    assert reply.status == 202, reply.body
    return str(reply.body["approval_id"])


async def test_an_approver_sees_the_context_and_the_request_still_waits() -> None:
    log: list[str] = []
    service = make_service(make_registry(log))
    for _ in range(3):
        first = await parked(service)
        done = await call(
            service,
            "POST",
            f"/approvals/{first}/decide",
            body={"approved": True},
            token="t-reviewer",
        )
        assert done.status == 200
    second = await parked(service)
    shown = await call(service, "GET", f"/approvals/{second}", token="t-reviewer")
    context = shown.body["context"]
    assert (context["similar"], context["approved"], context["denied"]) == (3, 3, 0)
    assert context["deciders"] == ["reviewer"]
    assert context["examples"][0]["matched"] == ["body", "to"]
    assert "a@example.com" not in repr(context)
    assert shown.body["status"] == "pending"
    assert log == ["send a@example.com"] * 3  # history never ran the fourth call


async def test_the_invoker_gets_no_context() -> None:
    service = make_service(make_registry([]))
    approval_id = await parked(service)
    shown = await call(service, "GET", f"/approvals/{approval_id}")
    assert shown.status == 200 and "context" not in shown.body


async def test_the_openapi_document_describes_the_context() -> None:
    doc = await call(make_service(), "GET", "/openapi.json")
    schemas = doc.body["components"]["schemas"]
    assert schemas["Approval"]["properties"]["context"] == {
        "$ref": "#/components/schemas/ApprovalContext"
    }
    assert "summary" in schemas["ApprovalContext"]["properties"]
