"""See, then act: a robot camera, a local vision model and the robot gate, all offline.

A MuJoCo camera looks down at a table with one block, red, green or blue. The agent calls
`look` (`VideoFeed` + `look_tool`), SmolVLM-256M names the colour, and the agent moves the
arm to that colour's bin with `move_to`, through the gate (a `SimRobot` in real time, MuJoCo
as the shadow). Each colour twice; the script prints what the model saw and where the arm
stopped. Simulation only. In a separate venv (never add torch to the package):
    pip install -e . pillow transformers num2words mujoco
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python scripts/robot_vision.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import mujoco
from vision_check import SmolVLM

import agent_adapter as aa
from agent_adapter.robotics import ActionSpec, Box, RobotGate, SimRobot
from agent_adapter.robotics.mujoco_sim import MujocoSim
from agent_adapter.voice.video import VideoFeed, look_tool

COLOURS = {"red": "1 0 0 1", "green": "0 0.8 0 1", "blue": "0 0 1 1"}
BINS = {"red": (-0.3, 0.3, 0.2), "green": (0.0, 0.3, 0.2), "blue": (0.3, 0.3, 0.2)}
HOME = (0.0, -0.3, 0.2)
SPEC = ActionSpec(
    workspace=Box(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.6), name="cell"),
    max_speed=0.5,
    max_force=20,
)
QUESTION = "What colour is the block on the table? Answer with one word."


def scene(colour: str) -> str:
    return f"""
<mujoco>
  <visual><global offwidth="384" offheight="288"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.01" rgba="0.85 0.85 0.85 1"/>
    <geom type="box" pos="0 0 0.06" size="0.12 0.12 0.06" rgba="{COLOURS[colour]}"/>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0"/>
  </worldbody>
</mujoco>"""


def camera(colour: str) -> Any:
    model = mujoco.MjModel.from_xml_string(scene(colour))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 288, 384)

    def render() -> Any:
        renderer.update_scene(data, camera="top")
        return renderer.render()  # uint8 RGB, as VideoFeed takes it

    return render


class Sorter:
    """The agent's model: look first, ask SmolVLM for the colour, then move to that bin."""

    def __init__(self, vlm: SmolVLM) -> None:
        self.vlm = vlm
        self.seen = ""

    def __call__(self, request: aa.LLMRequest) -> dict[str, Any]:
        last = request.messages[-1]
        if last.role == "user":
            return {"tool_calls": [{"name": "look", "arguments": {}}]}
        if last.role == "tool" and last.name == "look" and last.media:
            from io import BytesIO

            from PIL import Image

            image = Image.open(BytesIO(last.media[0].data)).convert("RGB")
            self.seen = self.vlm.answer(image, QUESTION)
            colour = next((c for c in BINS if c in self.seen.lower()), None)
            if colour is None:
                return {"text": f"I am not sure what I see: {self.seen!r}. I will not move."}
            x, y, z = BINS[colour]
            move = {"x": x, "y": y, "z": z, "speed": 0.4}
            return {"tool_calls": [{"name": "move_to", "arguments": move}]}
        return {"text": f"Done: {last.content}"}


async def one(vlm: SmolVLM, colour: str) -> tuple[str, str, bool]:
    feed = VideoFeed(fps=10, max_side=384)
    robot = SimRobot(HOME)
    registry = aa.Registry()
    registry.add(look_tool(feed))
    RobotGate(SPEC, robot=robot, sim=MujocoSim()).add_tools(registry)
    executor = aa.Executor(registry)
    sorter = Sorter(vlm)
    poller = feed.poll(camera(colour))
    await asyncio.sleep(0.3)
    ask = [aa.Message(role="user", content="Put the arm over the bin for the block's colour.")]
    model = aa.CallableModel(sorter, provider="local", model="smolvlm-256m")
    await aa.run_tool_loop(model, registry, ask, executor=executor, max_steps=4)
    poller.cancel()
    await feed.close()
    return sorter.seen, str(robot.position()), robot.position() == BINS[colour]


async def main() -> int:
    vlm = SmolVLM()
    right = 0
    for colour in [*COLOURS, *COLOURS]:
        seen, at, ok = await one(vlm, colour)
        right += ok
        print(f"{'PASS' if ok else 'FAIL'} {colour:5} block: model saw {seen!r}; arm at {at}")
    print(f"{right} of {2 * len(COLOURS)} blocks sorted to the right bin")
    return 0 if right >= 2 * len(COLOURS) - 1 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
