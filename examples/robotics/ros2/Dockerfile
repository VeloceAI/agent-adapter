# turtlesim behind the robot gate, headless. Build from the repository root:
#   docker build -f examples/robotics/ros2/Dockerfile -t agent-adapter-ros2 .
#   docker run --rm agent-adapter-ros2
FROM ros:jazzy-ros-base
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-turtlesim python3-venv \
    && rm -rf /var/lib/apt/lists/*
# --system-site-packages keeps the apt Python packages; rclpy comes from /opt/ros on PYTHONPATH
RUN python3 -m venv --system-site-packages /venv
COPY pyproject.toml README.md PYPI.md CHANGELOG.md LICENSE NOTICE /src/
COPY agent_adapter /src/agent_adapter
RUN /venv/bin/pip install --no-cache-dir /src
COPY examples/robotics/ros2/drive_turtlesim.py examples/robotics/ros2/run.sh /app/
ENV QT_QPA_PLATFORM=offscreen
CMD ["bash", "/app/run.sh"]
