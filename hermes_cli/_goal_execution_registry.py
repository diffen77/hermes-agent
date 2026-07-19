"""Process-owned registry state for reloadable goal recovery wiring."""

from __future__ import annotations

import threading


SUPERVISORS: dict[str, object] = {}
SUPERVISORS_LOCK = threading.Lock()
