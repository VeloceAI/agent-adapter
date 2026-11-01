"""The gallery areas and which example belongs to each."""

from __future__ import annotations

from dataclasses import dataclass


class GalleryError(ValueError):
    """An example cannot be placed in the gallery (no area, no docstring)."""


@dataclass(frozen=True)
class Area:
    slug: str
    title: str
    blurb: str


AREAS = (
    Area("basics", "Basics", "Tools, the registry, schemas, the executor, models and guides."),
    Area("backends", "No-framework backends", "Web apps, workers, scripts, serverless, deploy."),
    Area("frameworks", "Agent frameworks", "Registry tools inside each agent framework."),
    Area("compression", "Context compression", "Shrink what each model call sends."),
    Area("approval", "Approval", "Human review, durable stores, resume once."),
    Area("cost-cache", "Cost and cache", "Ledgers, prices, budgets, caches, embeddings."),
    Area("parallel", "Parallel fan-out and fan-in", "Several calls at once, then merge."),
    Area("flows", "Flows", "Graphs of nodes with joins, loops and pauses."),
    Area("serving", "Serving and protocols", "MCP, A2A, AG-UI, HTTP and OpenAPI."),
    Area("optimiser", "Optimiser", "Improve an agent's harness against eval cases."),
    Area("cookbook", "Cookbook", "Short recipes for common production needs."),
)

# Exact names win over prefixes. A new example must match one of these rules.
AREA_OF = {
    "adk_flow": "flows",
    "approval": "approval",
    "approval_stores": "approval",
    "approval_stores_sql": "approval",
    "cache": "cost-cache",
    "connect_server": "serving",
    "cost": "cost-cache",
    "cost_own_models": "cost-cache",
    "deploy/app": "backends",
    "embeddings": "cost-cache",
    "first_flow": "flows",
    "flow_research": "flows",
    "gemini_afc": "frameworks",
    "guide_migrate_approval": "approval",
    "hello_approval": "approval",
    "howto_budgets": "cost-cache",
    "howto_caching": "cost-cache",
    "howto_hitl": "approval",
    "howto_parallel": "parallel",
    "langgraph_flow": "frameworks",
    "openapi_gateway": "serving",
    "openapi_import": "serving",
    "optimiser_demo": "optimiser",
    "serve_tools": "serving",
    "template_celery": "backends",
    "template_fastapi": "backends",
    "template_flask": "backends",
    "template_langgraph": "frameworks",
    "template_mcp_server": "serving",
    "template_openai_agents": "frameworks",
    "tutorial_langgraph": "frameworks",
    "tutorial_research": "parallel",
    "viewer_fan_out": "parallel",
}
PREFIX_AREAS = (
    ("a2a_", "serving"),
    ("ag_ui_", "serving"),
    ("backend_", "backends"),
    ("compression_", "compression"),
    ("chat_", "basics"),
    ("cookbook_", "cookbook"),
    ("guide_", "basics"),
    ("harness_", "optimiser"),
    ("howto_", "basics"),
    ("mcp_", "serving"),
    ("template_", "basics"),
    ("viewer_", "basics"),
)
BASICS = {
    "anthropic_results", "attach_any", "context", "discover_integrations", "errors", "executor",
    "hello", "hooks", "imported_tools", "models", "providers", "quickstart", "registry_views",
    "runtime", "strict_mode", "tool_loop", "tracing", "wrap_existing",
}  # fmt: skip
FRAMEWORK_SUFFIX = "_tools"  # crewai_tools, strands_tools, ...


def area_of(rel: str) -> str:
    """The area slug for an example path under examples/ (no `.py`)."""
    if rel in AREA_OF:
        return AREA_OF[rel]
    for prefix, slug in PREFIX_AREAS:
        if rel.startswith(prefix):
            return slug
    if rel in BASICS:
        return "basics"
    if rel.endswith(FRAMEWORK_SUFFIX):
        return "frameworks"
    raise GalleryError(f"examples/{rel}.py has no gallery area: add it to AREA_OF")
