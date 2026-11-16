# Robot safety

Action envelopes, a shadow simulation, approvals through the executor, an e-stop and a deadman watchdog, for robots in simulation. Guide: [Gate a robot](../how-to/robot-safety.md).

## `agent_adapter.robotics`

::: agent_adapter.robotics
    options:
      members: [ActionSpec, Box, CallbackRobot, Command, Deadman, KinematicSim, RobotGate, RobotPort, SimPort, SimRobot, Verdict, Waypoint, judge, judge_path]

## `agent_adapter.robotics.gate`

`agent_adapter.robotics.gate` re-exports, documented under the canonical path: [`RobotGate`](#agent_adapter.robotics.RobotGate).

## `agent_adapter.robotics.spec`

::: agent_adapter.robotics.spec
    options:
      members: [Action, Point]

Also exported here, documented under the canonical path: [`ActionSpec`](#agent_adapter.robotics.ActionSpec), [`Box`](#agent_adapter.robotics.Box), [`Command`](#agent_adapter.robotics.Command), [`Verdict`](#agent_adapter.robotics.Verdict), [`Waypoint`](#agent_adapter.robotics.Waypoint), [`judge`](#agent_adapter.robotics.judge), [`judge_path`](#agent_adapter.robotics.judge_path).

## `agent_adapter.robotics.ports`

`agent_adapter.robotics.ports` re-exports, documented under the canonical path: [`CallbackRobot`](#agent_adapter.robotics.CallbackRobot), [`RobotPort`](#agent_adapter.robotics.RobotPort), [`SimPort`](#agent_adapter.robotics.SimPort).

## `agent_adapter.robotics.kinematic`

::: agent_adapter.robotics.kinematic
    options:
      members: [straight_line]

Also exported here, documented under the canonical path: [`KinematicSim`](#agent_adapter.robotics.KinematicSim), [`SimRobot`](#agent_adapter.robotics.SimRobot).

## `agent_adapter.robotics.watchdog`

`agent_adapter.robotics.watchdog` re-exports, documented under the canonical path: [`Deadman`](#agent_adapter.robotics.Deadman).

## `agent_adapter.robotics.mujoco_sim`

::: agent_adapter.robotics.mujoco_sim
    options:
      members: [GANTRY_XML, MujocoSim]
