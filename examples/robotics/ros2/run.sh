#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash  # not safe under `set -u`
ros2 run turtlesim turtlesim_node > /tmp/turtlesim.log 2>&1 &
sim=$!
trap 'kill $sim 2>/dev/null || true' EXIT
/venv/bin/python /app/drive_turtlesim.py
