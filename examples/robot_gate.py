"""Gate a simulated robot: an envelope, approvals through the executor, an e-stop."""

import asyncio

import agent_adapter as aa
from agent_adapter import ToolError
from agent_adapter.errors import GuardrailBlocked
from agent_adapter.robotics import ActionSpec, Box, KinematicSim, RobotGate, SimRobot

spec = ActionSpec(
    workspace=Box(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.6)),  # metres
    max_speed=0.25,  # m/s without a person
    max_force=20,  # N
    max_rate=5,  # motion commands per second
    keep_out=(Box(lo=(0.25, -0.15, 0.0), hi=(0.5, 0.15, 0.4), name="operator"),),
    hard_max_speed=1.0,  # above this, no approval can run it
)


def reviewer(request: aa.ApprovalRequest) -> bool:  # ask in your UI; here: say no
    print(f"asked about {request.tool} {request.arguments}")
    return False


registry = aa.Registry()
executor = aa.Executor(registry, approval=aa.CallbackApproval(reviewer))
robot = SimRobot((0.0, -0.3, 0.2))  # simulation only: swap in your RobotPort later
gate = RobotGate(spec, robot=robot, sim=KinematicSim(), executor=executor)
gate.add_tools(registry)  # move_to, grip, estop, reset_estop


async def main() -> None:
    print(await executor.run("move_to", {"x": 0.0, "y": -0.2, "z": 0.2, "speed": 0.2}))
    try:  # outside the envelope: the executor's approval port decides
        await executor.run("move_to", {"x": 0.0, "y": -0.2, "z": 0.2, "speed": 0.6})
    except aa.ApprovalDenied:
        print("denied by the reviewer")
    try:  # the shadow simulation's path crosses the keep-out zone: never runs
        await executor.run("move_to", {"x": 0.45, "y": 0.0, "z": 0.2, "speed": 0.2})
    except GuardrailBlocked as exc:
        print(f"blocked: {exc.reason}")

    async with gate.deadman(timeout=0.25):  # operator heartbeats; none for 0.25 s stops it
        move = asyncio.create_task(
            executor.run("move_to", {"x": -0.4, "y": 0.4, "z": 0.2, "speed": 0.25})
        )
        await asyncio.sleep(0.1)
        gate.heartbeat()
        await asyncio.sleep(0.1)
        await gate.estop("operator pressed stop")  # no approval, no executor needed
        try:
            await move
        except ToolError as exc:
            print(exc)
    print(f"latched: {gate.latched}")


asyncio.run(main())
