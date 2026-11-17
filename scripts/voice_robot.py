"""Talk to a robot arm in simulation, then shout "Stop!": how fast does it halt?

The whole loop runs: a Kokoro voice speaks as the caller, our `VoiceSession` with Moonshine
and Smart Turn hears it, a scripted model calls the gated robot tools, and a `SimRobot`
moves in real time with MuJoCo as the gate's shadow simulation (kinematic if MuJoCo is
missing). The caller asks for a slow move, then says "Stop!" a second into it. We time
from the first sample of "Stop!" to the arm holding still, and check the e-stop latched.

Needs the `voice-local` and `vad` extras (and `mujoco`). Simulation only: no hardware.
Usage: python scripts/voice_robot.py [--runs 5] [--silero PATH --smart-turn PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from voice_live import ScriptedLine
from voice_live_common import Measure, model_paths, to_pcm16

from agent_adapter import (
    CallableModel,
    Executor,
    LLMRequest,
    LLMResponse,
    Message,
    Registry,
    ToolCall,
)
from agent_adapter.robotics import ActionSpec, Box, RobotGate, SimRobot
from agent_adapter.voice import VoiceAgent, VoiceSession
from agent_adapter.voice.moonshine import MoonshineSTT, MoonshineTTS
from agent_adapter.voice.silero import SileroTurns
from agent_adapter.voice.smart_turn import RATE, SmartTurn

SPEC = ActionSpec(
    workspace=Box(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.6), name="cell"),
    max_speed=0.25,
    max_force=20,
    hard_workspace=Box(lo=(-0.8, -0.8, 0.0), hi=(0.8, 0.8, 0.9)),
    hard_max_speed=1.0,
    hard_max_force=60,
)
HOME = (0.0, -0.3, 0.2)
CUP = {"x": 0.3, "y": 0.3, "z": 0.2, "speed": 0.08}  # about 9 s away
LINES = ["Please move the arm over to the cup.", "Stop! Stop now!"]


def reply(text: str = "", *calls: ToolCall) -> LLMResponse:
    message = Message(role="assistant", content=text, tool_calls=calls)
    return LLMResponse(
        provider="script",
        model="script",
        text=text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        message=message,
    )


def robot_model() -> CallableModel:
    async def answer(request: LLMRequest) -> LLMResponse:
        last = request.messages[-1]
        if last.role == "tool":
            return reply("Done." if "stopped" not in last.content else "Stopped.")
        said = str(last.content).lower()
        if "stop" in said:
            return reply("", ToolCall(name="estop", arguments={"reason": "caller said stop"}))
        if "move" in said or "cup" in said:
            return reply("Moving to the cup now.", ToolCall(name="move_to", arguments=CUP))
        return reply("Sorry, could you say that again?")

    return CallableModel(answer, provider="script", model="script")


def shadow() -> object | None:
    try:
        from agent_adapter.robotics.mujoco_sim import MujocoSim

        return MujocoSim()
    except Exception:
        return None


async def still(robot: SimRobot, since: float) -> float:
    """The clock time the arm stopped moving for good (still for 200 ms)."""
    last, quiet_from = robot.position(), None
    while True:
        await asyncio.sleep(0.005)
        now, at = time.perf_counter(), robot.position()
        if at != last:
            last, quiet_from = at, None
        elif quiet_from is None:
            quiet_from = now
        elif now - quiet_from >= 0.2:
            return max(quiet_from, since)
        if now - since > 10:
            return float("nan")


async def one_run(args: argparse.Namespace, lines: list[bytes]) -> dict[str, float | bool]:
    silero, smart = model_paths(args.silero, args.smart_turn)
    line = ScriptedLine(Measure(), silent=True)
    session = VoiceSession(
        line, stt=MoonshineSTT(), tts=MoonshineTTS(), turns=SmartTurn(smart, SileroTurns(silero))
    )
    robot = SimRobot(HOME)
    registry = Registry()
    sim = shadow()
    gate = RobotGate(SPEC, robot=robot, sim=sim) if sim else RobotGate(SPEC, robot=robot)  # type: ignore[arg-type]
    gate.add_tools(registry)
    agent = VoiceAgent(robot_model(), Executor(registry), session.speaker)
    call = asyncio.create_task(session.run(agent))

    async def say(pcm: bytes) -> float:
        first = time.perf_counter()
        for i in range(0, len(pcm), RATE * 2 // 50):
            await line.send(pcm[i : i + RATE * 2 // 50])
            await asyncio.sleep(0.02)
        return first

    await say(lines[0] + bytes(RATE))
    for _ in range(400):  # wait for the arm to start
        if robot.moving:
            break
        await asyncio.sleep(0.025)
    moving = robot.moving
    await asyncio.sleep(1.0)
    shout = asyncio.create_task(say(lines[1] + bytes(RATE * 2)))
    stop_at = time.perf_counter()
    halted = await still(robot, stop_at)
    await shout
    await asyncio.sleep(1.5)
    line.frames_in.put_nowait(None)
    await asyncio.wait_for(call, 30)
    return {
        "moved": moving,
        "halt_ms": (halted - stop_at) * 1000,
        "latched": gate._latched is not None,
        "reached_cup": robot.position() == (CUP["x"], CUP["y"], CUP["z"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--silero", type=Path)
    parser.add_argument("--smart-turn", type=Path)
    args = parser.parse_args()
    from moonshine_voice import TextToSpeech

    tts = TextToSpeech().language("en_us").voice("kokoro_am_adam").load()
    lines = [to_pcm16(*tts.synthesize(text), out_rate=RATE) for text in LINES]
    results = [asyncio.run(one_run(args, lines)) for _ in range(args.runs)]
    for r in results:
        print(r)
    halts = [float(r["halt_ms"]) for r in results if r["moved"]]
    ok = all(r["moved"] and r["latched"] and not r["reached_cup"] for r in results)
    print(
        f"arm halted after 'Stop!' began: median {statistics.median(halts):.0f} ms, "
        f"range {min(halts):.0f} to {max(halts):.0f} ms; every run stopped and latched: {ok}"
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
