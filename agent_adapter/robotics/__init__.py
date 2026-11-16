"""Safe robot actions: envelopes, a shadow simulation, approvals and an e-stop (simulation first).

`RobotGate` turns a `RobotPort` into executor tools. `MujocoSim` (extra `mujoco`) is
imported from `agent_adapter.robotics.mujoco_sim`.
"""

from agent_adapter.robotics.gate import RobotGate
from agent_adapter.robotics.kinematic import KinematicSim, SimRobot
from agent_adapter.robotics.ports import CallbackRobot, RobotPort, SimPort
from agent_adapter.robotics.spec import (
    ActionSpec,
    Box,
    Command,
    Verdict,
    Waypoint,
    judge,
    judge_path,
)
from agent_adapter.robotics.watchdog import Deadman

__all__ = [
    "ActionSpec",
    "Box",
    "CallbackRobot",
    "Command",
    "Deadman",
    "KinematicSim",
    "RobotGate",
    "RobotPort",
    "SimPort",
    "SimRobot",
    "Verdict",
    "Waypoint",
    "judge",
    "judge_path",
]
