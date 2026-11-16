"""MuJoCo shadow simulator (extra `mujoco`): the path comes from physics, not a straight line."""

from __future__ import annotations

import asyncio
import math
import threading
from typing import Any

from agent_adapter.core.errors import MissingDependency
from agent_adapter.robotics.kinematic import straight_line
from agent_adapter.robotics.spec import ActionSpec, Command, Point, Waypoint

__all__ = ["GANTRY_XML", "MujocoSim"]

# A gantry: three damped slide joints under position servos, no gravity (a counterbalanced z).
GANTRY_XML = """
<mujoco model="agent_adapter_gantry">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="ee">
      <joint name="x" type="slide" axis="1 0 0" damping="30"/>
      <joint name="y" type="slide" axis="0 1 0" damping="30"/>
      <joint name="z" type="slide" axis="0 0 1" damping="30"/>
      <geom type="sphere" size="0.02" mass="0.5"/>
      <site name="ee"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="x" kp="400"/>
    <position joint="y" kp="400"/>
    <position joint="z" kp="400"/>
  </actuator>
</mujoco>
"""


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        raise MissingDependency(
            'MujocoSim needs mujoco: pip install "agent-adapter[mujoco]"'
        ) from exc
    return mujoco


class MujocoSim:
    """`SimPort` on MuJoCo: servos chase a setpoint that moves at the commanded speed.

    The default model is a gantry. A model of your own needs the same layout: the first
    three position actuators drive x, y and z slide joints at qpos 0-2, and a site named
    `site` marks the end effector. Overshoot and lag show up in the path the gate checks.
    """

    def __init__(
        self, xml: str = GANTRY_XML, *, site: str = "ee", dt: float = 0.02, settle: float = 0.3
    ) -> None:
        mujoco = _mujoco()
        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)
        self._site = self.model.site(site).id
        self.dt, self.settle = dt, settle
        self._lock = threading.Lock()

    async def simulate(self, start: Point, command: Command, spec: ActionSpec) -> list[Waypoint]:
        if command.kind != "move" or command.target is None:
            return []
        return await asyncio.to_thread(self._run, start, command.target, command.speed)

    def _run(self, start: Point, target: Point, speed: float) -> list[Waypoint]:
        mj = self._mj
        data = mj.MjData(self.model)
        data.qpos[:3] = start
        data.ctrl[:3] = start
        mj.mj_forward(self.model, data)
        setpoints = straight_line(start, target, speed, self.model.opt.timestep)
        duration = setpoints[-1].t + self.settle
        steps = math.ceil(duration / self.model.opt.timestep)
        every = max(1, round(self.dt / self.model.opt.timestep))
        path: list[Waypoint] = []
        with self._lock:  # MjModel is shared; each run has its own MjData
            for i in range(steps):
                data.ctrl[:3] = setpoints[min(i, len(setpoints) - 1)].at
                mj.mj_step(self.model, data)
                if (i + 1) % every == 0 or i == steps - 1:
                    at = tuple(float(v) for v in data.site_xpos[self._site])
                    path.append(Waypoint(t=float(data.time), at=at))
        return path
