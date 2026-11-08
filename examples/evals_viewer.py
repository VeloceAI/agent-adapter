"""Save two eval reports of a ticket desk agent and open them in the viewer's Evals view.

Runs offline. Each case run is recorded, so "Open run" in the Evals view shows its graph,
timeline and events. Then run the printed `agent-adapter ui` command.
"""

import sys
import tempfile
from pathlib import Path
from typing import Any

from agent_adapter import Executor, Registry
from agent_adapter.evals import EvalCase, EvalReport, run_eval_sync
from agent_adapter.evals.graders import contains, tools_called
from agent_adapter.viewer import record

registry = Registry()
SEATS = {"concert": "12 seats left", "museum": "sold out"}


@registry.tool
def seats_left(event: str) -> str:
    """How many tickets an event still has."""
    return SEATS.get(event, "no such event")


CASES = [
    EvalCase(id="concert", input="concert", expected="12 seats", tags=("tickets",)),
    EvalCase(id="museum", input="museum", expected="sold out", tags=("tickets",)),
    EvalCase(id="hello", input="hi there", expected="help", tags=("chat",)),
]


executor = Executor(registry)


async def careful_agent(question: str, ctx: Any) -> str:
    """Looks the event up with the tool (in the case's run, via `ctx`) before it answers."""
    if question in SEATS:
        return f"{question}: {await executor.run('seats_left', {'event': question}, ctx)}"
    return "Ask me about an event and I will help."


async def hasty_agent(question: str, ctx: Any) -> str:
    """A worse version: it guesses instead of calling the tool."""
    return "Tickets are available." if question in SEATS else "Ask me about an event, I can help."


# --8<-- [start:save]
def evaluate(agent: Any, runs: Path, evals: Path) -> EvalReport:
    """Grade every case, record each case run, and save the report for the Evals view."""
    recorder = record(executor, folder=runs)
    try:
        return run_eval_sync(
            agent,
            CASES,
            {"answer": contains(), "looked_up": tools_called("seats_left")},
            executor=executor,
            name="ticket-desk",
            save_to=evals,  # the folder `agent-adapter ui --evals` lists
        )
    finally:
        recorder.stop()


# --8<-- [end:save]


def main(folder: str | Path | None = None) -> dict[str, Any]:
    """Save a good and a worse report into `folder` (a temporary one by default)."""
    base = Path(folder or tempfile.mkdtemp(prefix="aa-evals-"))
    runs, evals = base / "runs", base / "evals"
    good = evaluate(careful_agent, runs, evals)
    worse = evaluate(hasty_agent, runs, evals)
    regressed = [change.case_id for change in worse.compare(good).regressions]
    print(f"saved {len(list(evals.glob('*.json')))} reports; regressions: {regressed}")
    print(f'see them: agent-adapter ui "{runs}" --evals "{evals}"  (then the Evals tab)')
    return {"scores": [good.mean_score, worse.mean_score], "regressed": regressed, "evals": evals}


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
