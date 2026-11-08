"""Turn an OTLP JSON trace export into viewer runs: GenAI spans become recorded events.

The file is only parsed, never executed. Each trace (or `gen_ai.conversation.id`) becomes one
run file in the runs folder; values are masked and clipped like a recorder's.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_adapter._version import __version__
from agent_adapter.core.errors import ConfigError
from agent_adapter.tracing import semconv as sc
from agent_adapter.viewer._files import FORMAT_VERSION, dumps, run_key, segment_name
from agent_adapter.viewer._otlp_spans import MAX_ID, span_events
from agent_adapter.viewer._scrub import scrub_data

logger = logging.getLogger(__name__)

MAX_IMPORT_BYTES = 32 * 1024 * 1024
MAX_SPANS = 100_000
MAX_RUNS = 1_000
IMPORT_PID = 0  # segment pid of imported runs: never a live writer
SOURCE = "otlp-import"


@dataclass
class _Run:
    run_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    content: bool = False


def import_otlp(source: str | Path, folder: str | Path) -> list[str]:
    """Write one run file per trace of the OTLP JSON file `source` into `folder`; run keys back.

    Importing the same file again rewrites the same run files.

    Raises:
        ConfigError: `source` is too big, not JSON, not an OTLP trace export, or holds too many
            spans or traces.
        OSError: `source` cannot be read or `folder` cannot be written.
    """
    path = Path(source)
    if path.stat().st_size > MAX_IMPORT_BYTES:
        raise ConfigError(f"{path.name} is over {MAX_IMPORT_BYTES // (1024 * 1024)} MB")
    try:
        document = json.loads(path.read_bytes())
    except (ValueError, RecursionError) as exc:
        raise ConfigError(f"{path.name} is not JSON") from exc
    runs = _runs(document)
    target = Path(folder)
    target.mkdir(parents=True, exist_ok=True)
    keys = [_write(target, run) for run in runs.values() if run.events]
    logger.info("imported %d run(s) from an OTLP file into %s", len(keys), target)
    return keys


def _runs(document: Any) -> dict[str, _Run]:
    if not isinstance(document, Mapping) or not isinstance(document.get("resourceSpans"), list):
        raise ConfigError("not an OTLP JSON trace export (no resourceSpans)")
    runs: dict[str, _Run] = {}
    skipped = count = 0
    for span in _spans(document["resourceSpans"]):
        count += 1
        if count > MAX_SPANS:
            raise ConfigError(f"the file holds over {MAX_SPANS} spans")
        attrs = attributes(span.get("attributes"))
        trace = str(span.get("traceId") or "")
        run_id = "otlp-" + str(attrs.get(sc.GEN_AI_CONVERSATION_ID) or trace or "trace")[:MAX_ID]
        events = span_events(span, attrs, run_id)
        if not events:
            skipped += 1
            continue
        run = runs.get(run_id)
        if run is None:
            if len(runs) >= MAX_RUNS:
                raise ConfigError(f"the file holds over {MAX_RUNS} traces")
            run = runs[run_id] = _Run(run_id)
        run.content = run.content or any(_has_content(e["data"]) for e in events)
        run.events.extend(events)
    if skipped:
        logger.info("OTLP import skipped %d span(s) that are not GenAI or flow spans", skipped)
    return runs


def _spans(resources: list[Any]) -> Iterator[Mapping[str, Any]]:
    for resource in resources:
        if not isinstance(resource, Mapping):
            continue
        scopes = resource.get("scopeSpans") or resource.get("instrumentationLibrarySpans") or []
        for scope in scopes if isinstance(scopes, list) else []:
            spans = scope.get("spans") if isinstance(scope, Mapping) else None
            for span in spans if isinstance(spans, list) else []:
                if isinstance(span, Mapping):
                    yield span


def attributes(items: Any) -> dict[str, Any]:
    """OTLP `[{key, value: {stringValue | intValue | ...}}]` as a plain mapping."""
    out: dict[str, Any] = {}
    for item in items if isinstance(items, list) else []:
        if isinstance(item, Mapping) and isinstance(item.get("key"), str):
            out[item["key"]] = _value(item.get("value"))
    return out


def _value(value: Any, depth: int = 0) -> Any:
    if not isinstance(value, Mapping) or depth > 8:
        return None
    if "stringValue" in value:
        return str(value["stringValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        try:
            return int(value["intValue"])  # OTLP JSON sends 64-bit ints as strings
        except (TypeError, ValueError):
            return None
    if "doubleValue" in value:
        try:
            return float(value["doubleValue"])
        except (TypeError, ValueError):
            return None
    array = value.get("arrayValue")
    if isinstance(array, Mapping) and isinstance(array.get("values"), list):
        return [_value(item, depth + 1) for item in array["values"]]
    return None


def _has_content(data: Mapping[str, Any]) -> bool:
    return any(key in data for key in ("arguments", "result", "input_messages", "output_messages"))


def _write(folder: Path, run: _Run) -> str:
    run.events.sort(key=lambda event: event["ts"])
    started = run.events[0]["ts"]
    key = run_key(run.run_id)
    meta = {
        "kind": "meta",
        "v": FORMAT_VERSION,
        "run_id": run.run_id,
        "pid": IMPORT_PID,
        "started": started,
        "source": SOURCE,
        "capture_content": run.content,
        "agent_adapter": __version__,
        "python": sys.version.split()[0],
    }
    lines = [dumps(meta)]
    for event in run.events:
        clean, _ = scrub_data(event["data"], capture_content=True)
        lines.append(dumps({"kind": "event", **event, "data": clean}))
    path = folder / segment_name(int(started * 1000), key, IMPORT_PID)
    path.write_bytes(b"".join(lines))
    return key
