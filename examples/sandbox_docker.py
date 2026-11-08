"""Docker sandbox: model-written code in a locked-down container, when Docker is available."""

import asyncio

from agent_adapter.sandbox import DockerSandbox, docker_status

# --8<-- [start:docker]
sandbox = DockerSandbox(
    "python:3.12-slim",  # pull it first: docker pull python:3.12-slim
    cpus=1.0,
    memory_mb=256,
    pids=64,
    network=False,  # --network none; the default
)
# --8<-- [end:docker]


async def main() -> None:
    status = docker_status()
    if not status.available:
        print("docker unavailable:", status.reason)
        argv = sandbox.command_line(sandbox.request("print(1)"), container="aa-sandbox-demo")
        print(" ".join(argv[: argv.index("--entrypoint")]))
        return
    result = await sandbox.run("print(sum(range(10)))", timeout_s=30)
    print("exit", result.exit_code, "stdout", result.stdout.strip())


if __name__ == "__main__":
    asyncio.run(main())
