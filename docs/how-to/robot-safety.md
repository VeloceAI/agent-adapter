---
title: "Gate a robot: envelopes, approvals and an e-stop for agent-driven robots"
description: "Put a robot behind action envelopes, a shadow simulation, the executor's approvals, an e-stop and a deadman watchdog, so a prompt-injected command cannot move it unsafely. Simulation first."
---

# Gate a robot

An agent that drives a robot through raw tool calls can be talked into anything: a tool
result that says "move to the operator at full speed" becomes a command. ros-mcp-server
leaves that to the client, ROSA has no approval layer, and LeRobot's motion clamp can be
set to `null`. `RobotGate` puts every robot command through the executor:

```mermaid
flowchart LR
  M[Model] --> E["Executor: confirm rule, approvals, ledger, events"]
  E --> G{"Envelope + gate state"}
  G -- block --> X[Refused]
  G --> S[SimPort shadow run]
  S -- "path hits a keep-out zone" --> X
  S --> R[RobotPort]
  O["estop(), deadman"] --> R
```

**Simulation only.** This page and its bench use simulated robots. Hard real-time limits
stay in your robot's controller; the gate works at the speed of an agent, not of a servo.

## Gate a simulated robot

```python
--8<-- "examples/robot_gate.py"
```

It prints:

```text
done: move at (0.000, -0.200, 0.200)
asked about move_to {'x': 0.0, 'y': -0.2, 'z': 0.2, 'speed': 0.6}
denied by the reviewer
blocked: path enters keep-out zone operator
e-stop (operator pressed stop): stopped at (...)
latched: operator pressed stop
```

## What happens to a command

| The command | What the gate does |
|---|---|
| inside `workspace`, `max_speed`, `max_force` | runs |
| outside them, inside the hard limits | the tool's `confirm` rule asks the executor's approval port |
| over `hard_max_speed` / `hard_max_force`, outside `hard_workspace` | blocked; no approval can run it |
| its simulated path enters a `keep_out` box | blocked |
| over `max_rate` commands per second | blocked |
| e-stop latched, or the deadman missed its heartbeat | blocked until a person resets it |
| not a number, or a tool that does not exist | blocked (`ToolNotFound` for an unknown tool) |

The approval is the executor's own: `CallbackApproval`, `DeferredApproval` (park it and
`executor.resume` later) or none, which denies. The confirm rule reads only the arguments,
so a state change can only ever block, never skip the question. Calling a robot tool
directly (`spec.invoke`) raises `ToolUsageError`.

## Stop it

- `await gate.estop(reason)` latches and calls `RobotPort.stop()`. It takes no approval,
  no executor slot and no lock, so it works while a move is running, after a budget ran
  out, or when the executor refuses calls.
- The `estop` tool does the same through the executor (audited). From another run, such
  as a supervisor agent, it stops a running move. In the run that started the move it
  waits for it, since one run's tools take turns, and run limits still count it: keep
  `gate.estop()` for your stop button.
- `async with gate.deadman(timeout=0.25):` stops the robot when `gate.heartbeat()` has not
  been called for 0.25 s. Beat from the operator or the controlling process, never from a
  model tool: a model that can beat can be told to keep beating.
- `reset_estop` is a `confirm=True` tool: a person releases the e-stop.
- Cancelling a move stops the robot: motion tools are `effects="physical"`, so a cancelled
  move is not finished; the gate latches the e-stop and the move reports where it stopped.
  A caller saying "stop" over a voice agent does the same.

## Audit

Robot calls are executor calls: `tool.*` and `approval.*` events, one ledger entry per
call, and the signed [audit log](../concepts/audit.md) if you attach one. The gate adds
`custom.robot_check` (verdict and reasons), `custom.robot_done`, `custom.robot_estop` and
`custom.robot_reset`.

## Plug in a simulator or a robot

Both ports are protocols:

- `SimPort.simulate(start, command, spec)` returns the end effector's path.
  `KinematicSim` (built in) moves in a straight line. `MujocoSim` runs physics:
  `pip install "agent-adapter[mujoco]"` (MuJoCo, Apache-2.0), then
  `from agent_adapter.robotics.mujoco_sim import MujocoSim`. Its default model is a
  three-axis gantry; your own model needs the same actuator layout.
- `RobotPort` has `execute`, `stop` and `position`. `CallbackRobot(execute=, stop=,
  position=)` wraps an rclpy publisher, a LeRobot robot or your driver. Nothing here
  imports ROS or LeRobot. Test on a simulator first; a real robot needs its own safety
  case, a hardware e-stop and a person in charge.

## The bench

`python scripts/robot_bench.py --runs 200 --seed 7 --mujoco` sends the same seeded command
streams to an ungated port (a model of an agent on ros-mcp-server with nothing in between,
not a run of it) and to the gate. One run on a Windows laptop, 200 commands per stream:

| Stream | Ungated: unsafe run | Gated: unsafe run | Gated: blocked / denied |
|---|---|---|---|
| injected commands (keep-out, too fast, crush grip, NaN, raw parameter write) | 180 | 0 | 126 / 54 |
| an honest planner's targets, many outside the envelope | 188 | 0 | 77 / 107 |

"Unsafe" is a command that reached the robot while it broke a hard limit or crossed a
keep-out zone, or left the envelope with no person's yes. With a reviewer who approves
everything, the gate still runs 0 unsafe commands, but 54 (injected) and 98 (planner)
out-of-envelope moves run with that approval: an approval is only as good as the person.
The 20 injected commands that stayed inside the envelope ran on both sides: the gate
limits what a command can do, it does not read intent. A 20-command burst ran 5.

| Lost heartbeat (deadman 250 ms, 20 trials) | Stop after the last beat | Moved after it |
|---|---|---|
| ungated: the move runs to its end | median 4270 ms, max 4899 | median 1040 mm, max 1214 |
| gated | median 271 ms, max 420 | median 68 mm, max 133 |

`gate.estop()` to a halted simulated robot: median 0.05 ms, max 0.94 ms. That is the
gate's own cost; a real robot adds its driver and brakes. The MuJoCo shadow gave the same
verdicts as the kinematic one on these streams. Not tested: real hardware, ROS 2 or
LeRobot through `CallbackRobot`, and arm models in MuJoCo (the built-in model is a gantry).
