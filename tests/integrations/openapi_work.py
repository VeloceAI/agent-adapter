"""OpenAPI imports and builds run in a child interpreter with a line-count cap (`tests.work`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.work import run_child

CHILD = """
import json, logging, tracemalloc
import httpx, yaml  # imported lazily by the importer: load them before counting
from tests.work import counting
from agent_adapter.core.errors import ConfigError
from agent_adapter.core.registry import Registry
from agent_adapter.integrations.openapi import import_openapi
from agent_adapter.integrations.openapi._document import load_document
from agent_adapter.integrations.openapi._schema import build_operations
warnings = []
class Keep(logging.Handler):
    def emit(self, record):
        warnings.append(record.getMessage())
logging.getLogger("agent_adapter.integrations.openapi").addHandler(Keep(logging.WARNING))
source = {source!r}
names, outputs, schemas, outcome = [], [], [], "ok"
if {memory!r}:
    tracemalloc.start()
with counting({cap!r}) as work:
    try:
        if {stage!r} == "build":
            found = build_operations(load_document(source), prefix="", only=None)
            names = [op.name for op in found]
        else:
            specs = import_openapi(Registry(), source, client_factory=lambda **k: None)
            names = [spec.name for spec in specs]
            outputs = [spec.output_schema is not None for spec in specs]
            schemas = [spec.json_schema() for spec in specs] if {schemas!r} else []
    except ConfigError as exc:
        outcome = f"ConfigError: {{exc}}"
print(json.dumps({{"outcome": outcome, "names": names, "outputs": outputs, "schemas": schemas,
                  "warnings": warnings, "lines": work.lines,
                  "peak_mb": tracemalloc.get_traced_memory()[1] / 2**20}}))
"""


def run_counted(
    document: dict[str, Any] | str,
    tmp_path: Path,
    *,
    cap: int,
    stage: str = "import",
    memory: bool = False,
    schemas: bool = False,
) -> dict[str, Any]:
    """`build` (operations only) or `import` (tools too) `document` in a child, at most `cap`
    lines of agent_adapter; returns outcome, names, warnings, lines and peak MB.

    A `str` document is spec text (YAML or JSON), passed as is; a mapping is written as JSON.
    """
    if isinstance(document, str):
        source: str = document
    else:
        path = tmp_path / "spec.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        source = str(path)
    script = CHILD.format(source=source, cap=cap, stage=stage, memory=memory, schemas=schemas)
    return run_child(script)
