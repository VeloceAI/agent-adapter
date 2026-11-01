"""Find integrations through entry points and export a registry with one."""

from agent_adapter import Registry

registry = Registry()


@registry.tool
def shipping_rate(region: str) -> float:
    """Shipping rate for a region, in percent of the order value."""
    return 18.0


def main() -> None:
    # --8<-- [start:discover]
    from agent_adapter.integrations.discovery import available_integrations, get_integration

    # IntegrationError if the name is unknown; MissingDependency if its extra is missing.
    openai = get_integration("openai")
    # Protocol-only API: export the tools whose `targets` include this integration.
    tools = [openai.export_tool(spec) for spec in registry.for_target(openai.name)]
    # {name: integration}; cached, refresh=True reloads.
    names = available_integrations()
    # --8<-- [end:discover]
    print(tools[0]["function"]["name"])
    print(sorted({"openai", "gemini", "anthropic"} & set(names)))


if __name__ == "__main__":
    main()
