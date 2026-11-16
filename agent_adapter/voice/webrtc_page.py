"""A one-page browser client for `WebRTCAudio`: serve it, open it, talk to your agent."""

from __future__ import annotations

import json

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice agent</title>
<style>
  body { font: 16px system-ui, sans-serif; max-width: 32rem; margin: 3rem auto; padding: 0 1rem; }
  button { font: inherit; padding: .6rem 1.2rem; margin-right: .5rem; }
  #state { margin-top: 1rem; color: #555; }
</style>
</head>
<body>
<h1>Voice agent</h1>
<p>Talk to the agent from this page. Your browser cancels its echo, so speakers work.</p>
<button id="call">Call</button><button id="hang" disabled>Hang up</button>
<p id="state">Not connected.</p>
<audio id="agent" autoplay></audio>
<script>
const OFFER_URL = __OFFER_URL__;
let pc = null;
const state = (text) => { document.getElementById("state").textContent = text; };

async function call() {
  document.getElementById("call").disabled = true;
  const mic = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  pc = new RTCPeerConnection();
  mic.getTracks().forEach((track) => pc.addTrack(track, mic));
  pc.ontrack = (event) => { document.getElementById("agent").srcObject = event.streams[0]
    || new MediaStream([event.track]); };
  pc.onconnectionstatechange = () => state(pc.connectionState);
  await pc.setLocalDescription(await pc.createOffer());
  await new Promise((done) => {
    if (pc.iceGatheringState === "complete") return done();
    pc.onicegatheringstatechange = () => pc.iceGatheringState === "complete" && done();
  });
  const reply = await fetch(OFFER_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
  });
  await pc.setRemoteDescription(await reply.json());
  document.getElementById("hang").disabled = false;
}

function hang() {
  if (pc) { pc.getSenders().forEach((s) => s.track && s.track.stop()); pc.close(); pc = null; }
  document.getElementById("hang").disabled = true;
  document.getElementById("call").disabled = false;
  state("Hung up.");
}

document.getElementById("call").onclick = () => call().catch((e) => { state(String(e)); hang(); });
document.getElementById("hang").onclick = hang;
</script>
</body>
</html>
"""


def webrtc_page(offer_url: str = "/offer") -> str:
    """The HTML page; it POSTs its SDP offer as JSON to `offer_url` and plays the answer."""
    safe = json.dumps(offer_url).replace("<", "\\u003c")
    return _PAGE.replace("__OFFER_URL__", safe)


__all__ = ["webrtc_page"]
