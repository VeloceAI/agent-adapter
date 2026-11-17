"""A one-page browser client for `WebRTCAudio`: talk to your agent, show it a camera or screen."""

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
  button[aria-pressed="true"] { background: #1f6feb; color: #fff; }
  #preview { width: 100%; margin-top: 1rem; border-radius: .5rem; background: #000; }
</style>
</head>
<body>
<h1>Voice agent</h1>
<p>Talk to the agent from this page. Your browser cancels its echo, so speakers work.
During a call, show it your camera or your screen.</p>
<button id="call">Call</button><button id="hang" disabled>Hang up</button>
<p>
<button id="camera" aria-pressed="false" disabled>Camera</button><button id="screen"
  aria-pressed="false" disabled>Share screen</button>
</p>
<p id="state">Not connected.</p>
<audio id="agent" autoplay></audio>
<video id="preview" autoplay muted playsinline hidden></video>
<script>
const OFFER_URL = __OFFER_URL__;
let pc = null;
let sender = null;
let shown = null;
let stream = null;
const $ = (id) => document.getElementById(id);
const state = (text) => { $("state").textContent = text; };

function showVideo(kind) {
  shown = kind;
  for (const id of ["camera", "screen"]) $(id).setAttribute("aria-pressed", String(id === kind));
  $("preview").srcObject = stream;
  $("preview").hidden = !stream;
}

function stopVideo() {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  if (sender) sender.replaceTrack(null);
  showVideo(null);
}

async function toggleVideo(kind) {
  const was = shown;
  stopVideo();
  if (was === kind || !sender) return;
  stream = kind === "camera"
    ? await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
    : await navigator.mediaDevices.getDisplayMedia({ video: true });
  const track = stream.getVideoTracks()[0];
  track.onended = () => { if (shown === kind) stopVideo(); };  // "Stop sharing" in the browser
  await sender.replaceTrack(track);
  showVideo(kind);
}

async function call() {
  $("call").disabled = true;
  const mic = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  pc = new RTCPeerConnection();
  mic.getTracks().forEach((track) => pc.addTrack(track, mic));
  sender = pc.addTransceiver("video", { direction: "sendonly" }).sender;
  pc.ontrack = (event) => { $("agent").srcObject = event.streams[0]
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
  for (const id of ["hang", "camera", "screen"]) $(id).disabled = false;
}

function hang() {
  stopVideo();
  sender = null;
  if (pc) { pc.getSenders().forEach((s) => s.track && s.track.stop()); pc.close(); pc = null; }
  for (const id of ["hang", "camera", "screen"]) $(id).disabled = true;
  $("call").disabled = false;
  state("Hung up.");
}

$("call").onclick = () => call().catch((e) => { state(String(e)); hang(); });
$("hang").onclick = hang;
for (const kind of ["camera", "screen"]) {
  $(kind).onclick = () => toggleVideo(kind).catch((e) => { state(String(e)); stopVideo(); });
}
</script>
</body>
</html>
"""


def webrtc_page(offer_url: str = "/offer") -> str:
    """The HTML page; it POSTs its SDP offer as JSON to `offer_url` and plays the answer."""
    safe = json.dumps(offer_url).replace("<", "\\u003c")
    return _PAGE.replace("__OFFER_URL__", safe)


__all__ = ["webrtc_page"]
