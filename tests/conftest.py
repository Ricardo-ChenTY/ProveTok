from __future__ import annotations

import sys
from pathlib import Path


# Ensure repo root is on sys.path so `import provetok` works under all pytest import modes.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

