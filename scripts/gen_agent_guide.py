"""Write AGENTS.md, docs/llms-full.txt and agent_adapter/_guide/AGENTS.md from the code.

Run from the repo root: `python scripts/gen_agent_guide.py` (add `--check` to only compare).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `scripts.` imports resolve

from scripts.agent_guide.render import main  # needs the path above

if __name__ == "__main__":
    raise SystemExit(main())
