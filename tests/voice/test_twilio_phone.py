import asyncio
import base64
from urllib.parse import parse_qs

import pytest

httpx = pytest.importorskip("httpx")

from agent_adapter.errors import ConfigError, ModelAuthError  # noqa: E402
from agent_adapter.voice.socket import text_socket  # noqa: E402
from agent_adapter.voice.testing import settle  # noqa: E402
from agent_adapter.voice.twilio import TwilioStream  # noqa: E402
from agent_adapter.voice.twilio_calls import TwilioCalls, dial_twiml, stream_twiml  # noqa: E402
from tests.voice.test_transports import Far, collect, twilio  # noqa: E402

SID = "CA" + "0123456789abcdef" * 2
ACCOUNT = "AC" + "f" * 32
TOKEN = "token-not-real"


class FakeTwilio:
    """Twilio's REST API on a fake transport: records each form and answers with a Call."""

    def __init__(self, status: int = 200) -> None:
        self.status = status
        self.requests: list[httpx.Request] = []

    def calls(self, **options) -> TwilioCalls:
        def handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if self.status >= 400:
                return httpx.Response(self.status, json={"code": 20003, "message": "denied"})
            return httpx.Response(201, json={"sid": SID, "status": "queued"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        return TwilioCalls(ACCOUNT, TOKEN, client=client, **options)

    def form(self, n: int = -1) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(self.requests[n].content.decode()).items()}


def phone(**options) -> tuple[Far, TwilioStream]:
    far = Far()
    return far, TwilioStream(text_socket(far._receive, far._messages()), **options)


async def test_keypad_digits_reach_the_callbacks():
    far, stream = phone()
    pressed: list[str] = []

    def broken(digit: str) -> None:
        raise RuntimeError("callback bug")

    stream.on_digit += [pressed.append, broken]
    for item in (
        twilio("start", start={"callSid": SID}),
        twilio("dtmf", sequenceNumber="5", dtmf={"track": "inbound", "digit": "1"}),
        twilio("dtmf", dtmf={"track": "inbound", "digit": "#"}),
        twilio("dtmf", dtmf={"track": "inbound", "digit": "12"}),
        twilio("dtmf", dtmf="7"),
        twilio("stop"),
    ):
        far.inbox.put_nowait(item)
    await collect(stream.frames())
    assert pressed == ["1", "#"]


async def test_hang_up_ends_the_call_through_rest_and_stops_the_frames():
    fake = FakeTwilio()
    far, stream = phone(calls=fake.calls())
    far.inbox.put_nowait(twilio("start", start={"callSid": SID}))
    payload = base64.b64encode(b"\xff" * 160).decode()
    far.inbox.put_nowait(twilio("media", media={"payload": payload}))
    reading = asyncio.ensure_future(collect(stream.frames()))
    await settle()
    assert not reading.done()  # waiting on Twilio for the next message
    await stream.hang_up()
    assert len(await asyncio.wait_for(reading, 1)) == 1
    request = fake.requests[0]
    assert request.url.path == f"/2010-04-01/Accounts/{ACCOUNT}/Calls/{SID}.json"
    assert fake.form() == {"Status": "completed"}
    assert request.headers["authorization"].startswith("Basic ")
    assert await collect(stream.frames()) == []


async def test_hang_up_without_rest_only_ends_our_side():
    _far, stream = phone()
    reading = asyncio.ensure_future(collect(stream.frames()))
    await settle()
    await stream.hang_up()
    assert await asyncio.wait_for(reading, 1) == []


async def test_transfer_dials_a_person_from_the_live_call():
    fake = FakeTwilio()
    far, stream = phone(calls=fake.calls())
    far.inbox.put_nowait(twilio("start", start={"callSid": SID}))
    far.inbox.put_nowait(twilio("stop"))
    await collect(stream.frames())
    await stream.transfer("+15558675310", caller_id="+15551112222")
    assert fake.form()["Twiml"] == (
        '<Response><Dial callerId="+15551112222"><Number>+15558675310</Number></Dial></Response>'
    )
    with pytest.raises(ConfigError, match="calls="):
        await phone()[1].transfer("+1555")


async def test_place_an_outbound_call_into_a_stream():
    fake = FakeTwilio()
    calls = fake.calls()
    twiml = stream_twiml("wss://example.com/twilio", {"lead": 'a"b'})
    answer = await calls.call(
        "+15550001111",
        "+15552223333",
        twiml=twiml,
        timeout_s=30,
        status_callback="https://example.com/status",
    )
    assert answer == {"sid": SID, "status": "queued"}
    assert fake.requests[0].url.path == f"/2010-04-01/Accounts/{ACCOUNT}/Calls.json"
    form = fake.form()
    assert form["To"] == "+15550001111" and form["From"] == "+15552223333"
    assert form["Timeout"] == "30" and form["StatusCallback"] == "https://example.com/status"
    assert form["Twiml"] == (
        '<Response><Connect><Stream url="wss://example.com/twilio">'
        '<Parameter name="lead" value=\'a"b\'/></Stream></Connect></Response>'
    )
    await calls.call("+1", "+2", url="https://example.com/answer.xml")
    assert fake.form()["Url"] == "https://example.com/answer.xml"
    for bad in ({}, {"twiml": "<Response/>", "url": "https://x"}):
        with pytest.raises(ConfigError, match="exactly one"):
            await calls.call("+1", "+2", **bad)
    await calls.aclose()


def test_dial_twiml_escapes_and_dials_sip():
    assert dial_twiml("sip:desk@example.com") == (
        "<Response><Dial><Sip>sip:desk@example.com</Sip></Dial></Response>"
    )
    assert "<Number>&lt;Hangup/&gt;</Number>" in dial_twiml("<Hangup/>")


async def test_credentials_errors_and_call_sids(monkeypatch):
    for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_API_KEY", "TWILIO_API_SECRET"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ConfigError, match="TWILIO_ACCOUNT_SID"):
        await TwilioCalls(client=httpx.AsyncClient()).hang_up(SID)
    with pytest.raises(ConfigError, match="TWILIO_AUTH_TOKEN"):
        await TwilioCalls(ACCOUNT, client=httpx.AsyncClient()).hang_up(SID)
    with pytest.raises(ConfigError, match="call SID"):
        await TwilioCalls(ACCOUNT, TOKEN).hang_up("CA../../Accounts")

    fake = FakeTwilio()
    monkeypatch.setenv("TWILIO_API_KEY", "SK1")
    monkeypatch.setenv("TWILIO_API_SECRET", "secret")
    await fake.calls().hang_up(SID)
    expected = "Basic " + base64.b64encode(b"SK1:secret").decode()
    assert fake.requests[0].headers["authorization"] == expected

    denied = FakeTwilio(status=401)
    with pytest.raises(ModelAuthError) as caught:
        await denied.calls().hang_up(SID)
    assert caught.value.provider == "twilio" and TOKEN not in str(caught.value)
