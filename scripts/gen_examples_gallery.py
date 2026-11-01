"""Write the Examples gallery (docs/examples.md and docs/examples/<area>.md) from examples/.

Run from the repo root: `python scripts/gen_examples_gallery.py` (add `--check` to only compare).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `scripts.` imports resolve

from scripts.examples_gallery.render import main  # needs the path above

if __name__ == "__main__":
    raise SystemExit(main())
