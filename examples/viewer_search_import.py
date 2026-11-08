"""Viewer: import an OTLP JSON trace file from another app as a run you can browse and search."""

import json
import tempfile
from pathlib import Path

from agent_adapter.viewer import RunFolder, import_otlp

NS = 1_000_000_000  # nanoseconds per second


def attr(key: str, value: object) -> dict[str, object]:
    """One OTLP JSON attribute."""
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    return {"key": key, "value": {"stringValue": str(value)}}


def span(
    span_id: str, name: str, start: float, end: float, *attrs: dict[str, object]
) -> dict[str, object]:
    """One OTLP JSON span of trace `t1`, under the chat span `s1` (except `s1` itself)."""
    return {
        "traceId": "t1",
        "spanId": span_id,
        "parentSpanId": "" if span_id == "s1" else "s1",
        "name": name,
        "startTimeUnixNano": str(int(start * NS)),
        "endTimeUnixNano": str(int(end * NS)),
        "attributes": list(attrs),
    }


# What another app exported with its OpenTelemetry GenAI instrumentation.
TRACES = {
    "resourceSpans": [
        {
            "scopeSpans": [
                {
                    "spans": [
                        span(
                            "s1",
                            "chat shop-model",
                            1_800_000_000.0,
                            1_800_000_001.5,
                            attr("gen_ai.operation.name", "chat"),
                            attr("gen_ai.provider.name", "openai"),
                            attr("gen_ai.request.model", "shop-model"),
                            attr("gen_ai.usage.input_tokens", 120),
                            attr("gen_ai.usage.output_tokens", 8),
                        ),
                        span(
                            "s2",
                            "execute_tool stock",
                            1_800_000_000.2,
                            1_800_000_000.4,
                            attr("gen_ai.operation.name", "execute_tool"),
                            attr("gen_ai.tool.name", "stock"),
                        ),
                    ]
                }
            ]
        }
    ]
}


# --8<-- [start:import]
def main(folder: Path) -> list[str]:
    """Import the traces into a runs folder, then list what the viewer shows."""
    source = folder / "traces.json"
    source.write_text(json.dumps(TRACES), encoding="utf-8")
    runs = folder / "runs"
    import_otlp(source, runs)  # parsed only, never run; same as `agent-adapter ui --import`
    found = RunFolder(runs)
    found.refresh()
    return [
        f"{run['run_id']} {run['status']} {run['events']} events {run['framework']}"
        for run in found.summaries()
    ]


# --8<-- [end:import]


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        for line in main(Path(tmp)):
            print(line)
