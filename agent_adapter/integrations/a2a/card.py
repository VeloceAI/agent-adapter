"""The AgentCard of a registry: one `AgentSkill` per tool exported to `a2a`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_adapter._version import __version__
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.registry import Registry, check_target
from agent_adapter.core.tool import ToolSpec
from agent_adapter.integrations.a2a._wire import JSON_MEDIA, TARGET, TOOLS_EXTENSION, require_a2a

if TYPE_CHECKING:
    from a2a.types import AgentCard, AgentSkill

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "1.0"
MAX_SKILLS = 512  # a card is fetched whole by every client


def agent_skill(spec: ToolSpec) -> AgentSkill:
    """The skill for one tool: id and name are the tool name.

    Raises:
        ToolUsageError: `spec` is not exported to `a2a`.
        MissingDependency: `a2a-sdk` is not installed.
    """
    require_a2a()
    check_target(spec, TARGET)
    from a2a.types import AgentSkill

    return AgentSkill(
        id=spec.name,
        name=spec.name,
        description=spec.description,
        tags=sorted(spec.tags) or ["tool"],
        input_modes=[JSON_MEDIA],
        output_modes=[JSON_MEDIA],
    )


def agent_card(
    registry: Registry,
    *,
    url: str,
    name: str = "agent-adapter",
    description: str = "Tools served by agent-adapter.",
    version: str = __version__,
) -> AgentCard:
    """An AgentCard for `registry.for_target("a2a")`, served at `url` over JSON-RPC.

    The `TOOLS_EXTENSION` extension carries each skill's JSON input schema, so
    `import_a2a` can rebuild typed tools. Streaming and push notifications are off.

    Raises:
        ConfigError: `url` is not http(s), or more than `MAX_SKILLS` tools.
        MissingDependency: `a2a-sdk` is not installed.
    """
    require_a2a()
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        raise ConfigError(f"the agent card url must be http(s), got {url!r}")
    specs = list(registry.for_target(TARGET))
    if len(specs) > MAX_SKILLS:
        raise ConfigError(f"{len(specs)} tools exceed the {MAX_SKILLS}-skill card limit")
    from a2a.types import AgentCapabilities, AgentCard, AgentExtension, AgentInterface
    from a2a.utils.constants import TransportProtocol
    from google.protobuf import json_format, struct_pb2

    schemas: dict[str, Any] = {spec.name: spec.json_schema() for spec in specs}
    params = json_format.ParseDict({"version": 1, "tools": schemas}, struct_pb2.Struct())
    extension = AgentExtension(
        uri=TOOLS_EXTENSION,
        description="Each skill is one tool; send a data part {tool, arguments}.",
        required=False,
        params=params,
    )
    card = AgentCard(
        name=name,
        description=description,
        version=version,
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC.value,
                protocol_version=PROTOCOL_VERSION,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=False, push_notifications=False, extensions=[extension]
        ),
        default_input_modes=[JSON_MEDIA],
        default_output_modes=[JSON_MEDIA],
        skills=[agent_skill(spec) for spec in specs],
    )
    logger.debug("agent card %r: %d skills", name, len(specs))
    return card
