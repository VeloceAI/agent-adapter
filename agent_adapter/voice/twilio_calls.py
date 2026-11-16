"""`TwilioCalls`: place, transfer and end phone calls through Twilio's Calls REST API.

Credentials come from the constructor or the environment: `TWILIO_ACCOUNT_SID` with
`TWILIO_AUTH_TOKEN`, or an API key (`TWILIO_API_KEY` and `TWILIO_API_SECRET`). They are
sent only as HTTP Basic auth and never logged. Each call here has a side effect: when an
agent may trigger one, wrap it in a tool marked `confirm=True`.
"""

from __future__ import annotations

import os
import re
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from agent_adapter.core.errors import ConfigError
from agent_adapter.voice._http import HttpClient, post

_PROVIDER = "twilio"
_CALL_SID = re.compile(r"CA[0-9a-fA-F]{32}")


def stream_twiml(url: str, parameters: dict[str, str] | None = None) -> str:
    """TwiML that connects a call to your `TwilioStream` WebSocket at `url` (wss://...)."""
    params = "".join(
        f"<Parameter name={quoteattr(k)} value={quoteattr(v)}/>"
        for k, v in (parameters or {}).items()
    )
    return f"<Response><Connect><Stream url={quoteattr(url)}>{params}</Stream></Connect></Response>"


def dial_twiml(to: str, *, caller_id: str | None = None) -> str:
    """TwiML that dials `to` (a phone number, or a `sip:` address) from the live call."""
    caller = f" callerId={quoteattr(caller_id)}" if caller_id else ""
    noun = f"<Sip>{escape(to)}</Sip>" if to.startswith("sip:") else f"<Number>{escape(to)}</Number>"
    return f"<Response><Dial{caller}>{noun}</Dial></Response>"


class TwilioCalls:
    """A small async client for the Calls resource; `client` takes your `httpx.AsyncClient`."""

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://api.twilio.com",
        client: Any = None,
        timeout: float = 15,
    ) -> None:
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID", "")
        self._token = auth_token
        self._api_key = api_key
        self._api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._http = HttpClient(client, timeout)

    async def call(
        self,
        to: str,
        from_: str,
        *,
        twiml: str | None = None,
        url: str | None = None,
        status_callback: str | None = None,
        timeout_s: int | None = None,
    ) -> dict[str, Any]:
        """Place an outbound call that runs `twiml` (or fetches it from `url`) on answer."""
        if (twiml is None) == (url is None):
            raise ConfigError("pass exactly one of twiml= or url=")
        form = {"To": to, "From": from_, **({"Twiml": twiml} if twiml else {"Url": str(url)})}
        if status_callback:
            form["StatusCallback"] = status_callback
        if timeout_s is not None:
            form["Timeout"] = str(timeout_s)
        return await self._post("Calls.json", form)

    async def transfer(
        self, call_sid: str, to: str, *, caller_id: str | None = None
    ) -> dict[str, Any]:
        """Hand a live call to a person at `to`; the media stream ends as Twilio dials."""
        return await self.update(call_sid, Twiml=dial_twiml(to, caller_id=caller_id))

    async def hang_up(self, call_sid: str) -> dict[str, Any]:
        """End a live call."""
        return await self.update(call_sid, Status="completed")

    async def update(self, call_sid: str, **form: str) -> dict[str, Any]:
        """Change a live call (`Twiml`, `Url` or `Status`, as Twilio names them)."""
        if not _CALL_SID.fullmatch(call_sid):
            raise ConfigError("call_sid must be a Twilio call SID (CA and 32 hex digits)")
        return await self._post(f"Calls/{call_sid}.json", form)

    async def aclose(self) -> None:
        """Close the HTTP client this object made (one you passed in stays open)."""
        await self._http.aclose()

    async def _post(self, path: str, form: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}/2010-04-01/Accounts/{self._account()}/{path}"
        response = await post(self._http, _PROVIDER, "", url, data=form, auth=self._auth())
        body = response.json()
        return body if isinstance(body, dict) else {}

    def _account(self) -> str:
        if not self.account_sid:
            raise ConfigError("Twilio needs an account SID: pass it or set TWILIO_ACCOUNT_SID")
        return self.account_sid

    def _auth(self) -> tuple[str, str]:
        key = self._api_key or os.environ.get("TWILIO_API_KEY", "")
        secret = self._api_secret or os.environ.get("TWILIO_API_SECRET", "")
        if key and secret:
            return key, secret
        token = self._token or os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not token:
            raise ConfigError(
                "Twilio needs TWILIO_AUTH_TOKEN (or TWILIO_API_KEY and TWILIO_API_SECRET)"
            )
        return self._account(), token


__all__ = ["TwilioCalls", "dial_twiml", "stream_twiml"]
