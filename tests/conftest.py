from __future__ import annotations

import os
import sys

# Ensure the repository root is on sys.path for test imports
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


