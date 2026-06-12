"""Make `import agentic_ip_reuse` resolve to the real planner package.

The planner lives at agentic_ip_reuse/agentic_ip_reuse/ (with __init__.py). If
the repo root alone is on sys.path, the OUTER directory is importable as a
namespace package and shadows the real one for every test collected afterwards.
Putting the planner root first makes resolution order-independent.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = REPO_ROOT / "agentic_ip_reuse"
for path in (str(PLANNER_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
