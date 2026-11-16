import pytest
from pydantic import ValidationError

from agent_adapter.robotics import ActionSpec, Box, Command, Waypoint, judge, judge_path

CELL = Box(lo=(-0.5, -0.5, 0.0), hi=(0.5, 0.5, 0.6), name="cell")
SPEC = ActionSpec(
    workspace=CELL,
    max_speed=0.25,
    max_force=20,
    keep_out=(Box(lo=(0.2, -0.1, 0.0), hi=(0.3, 0.1, 0.3), name="operator"),),
    hard_workspace=Box(lo=(-1, -1, 0), hi=(1, 1, 1)),
    hard_max_speed=1.0,
    hard_max_force=60,
)


def move(x: float, y: float, z: float, speed: float = 0.1) -> Command:
    return Command(kind="move", target=(x, y, z), speed=speed)


def test_box_contains_and_crosses() -> None:
    zone = SPEC.keep_out[0]
    assert zone.contains((0.25, 0, 0.1))
    assert not zone.contains((0, 0, 0.1))
    assert zone.crosses((0, 0, 0.1), (0.4, 0, 0.1))
    assert not zone.crosses((0, 0, 0.4), (0.4, 0, 0.4))
    assert not zone.crosses((0, 0.5, 0.1), (0.4, 0.5, 0.1))
    assert zone.crosses((0.25, 0, 0.1), (0.25, 0, 0.1))  # a still point inside
    assert not zone.crosses((0, 0, 0.1), (0.1, 0, 0.1))  # stops short


def test_box_rejects_reversed_corners() -> None:
    with pytest.raises(ValidationError):
        Box(lo=(1, 0, 0), hi=(0, 1, 1))


@pytest.mark.parametrize(
    ("command", "action"),
    [
        (move(0.1, 0.1, 0.3), "run"),
        (move(0.1, 0.1, 0.3, speed=0.5), "approve"),
        (move(0.8, 0.1, 0.3), "approve"),
        (move(0.1, 0.1, 0.3, speed=2.0), "block"),
        (move(1.5, 0.1, 0.3), "block"),
        (Command(kind="grip", force=10), "run"),
        (Command(kind="grip", force=40), "approve"),
        (Command(kind="grip", force=100), "block"),
    ],
)
def test_judge(command: Command, action: str) -> None:
    assert judge(SPEC, command).action == action


def test_judge_without_hard_limits_only_asks() -> None:
    spec = ActionSpec(workspace=CELL, max_speed=0.25, max_force=20)
    assert judge(spec, move(9, 9, 9, speed=9)).action == "approve"
    assert judge(spec, Command(kind="grip", force=999)).action == "approve"


def test_verdict_summary() -> None:
    assert judge(SPEC, move(0, 0, 0.1)).summary == "inside the envelope"
    assert "outside the workspace" in judge(SPEC, move(0.8, 0, 0.1)).summary


def test_commands_reject_nan_and_bad_shapes() -> None:
    with pytest.raises(ValidationError):
        Command(kind="move", target=(float("nan"), 0, 0), speed=0.1)
    with pytest.raises(ValidationError):
        Command(kind="move", target=(0, 0, 0), speed=0)
    with pytest.raises(ValidationError):
        Command(kind="grip", force=-1)


def test_spec_hard_limits_must_be_wider() -> None:
    with pytest.raises(ValidationError):
        ActionSpec(workspace=CELL, max_speed=1, max_force=1, hard_max_speed=0.5)
    with pytest.raises(ValidationError):
        ActionSpec(workspace=CELL, max_speed=1, max_force=10, hard_max_force=5)


def test_judge_path_keep_out_hard_workspace_and_speed() -> None:
    start = (0.0, 0.0, 0.1)
    through = [Waypoint(t=1, at=(0.4, 0, 0.1))]
    assert "operator" in judge_path(SPEC, start, through).summary
    outside = [Waypoint(t=1, at=(0, 0, 0.1)), Waypoint(t=2, at=(0, 0, 1.5))]
    assert "hard workspace" in judge_path(SPEC, start, outside).summary
    fast = [Waypoint(t=0.1, at=(0, 0.4, 0.1))]
    assert "over the hard limit" in judge_path(SPEC, start, fast).summary
    fine = [Waypoint(t=0, at=start), Waypoint(t=2, at=(0, 0.4, 0.1))]
    assert judge_path(SPEC, start, fine).action == "run"
