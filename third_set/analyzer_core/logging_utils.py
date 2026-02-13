"""Logging helpers for analyzer internals."""

from __future__ import annotations

import math
import os
from typing import List


def _dbg(msg: str) -> None:
    if os.getenv("THIRDSET_DEBUG") in ("1", "true", "yes"):
        print(f"[debug] {msg}", flush=True)


def _log_step(msg: str) -> None:
    """
    Verbose progress logging for long-running analysis.
    Enabled when THIRDSET_PROGRESS or THIRDSET_TG_LOG is set.
    """
    if os.getenv("THIRDSET_PROGRESS") in ("1", "true", "yes") or os.getenv("THIRDSET_TG_LOG") in ("1", "true", "yes"):
        print(f"[progress] {msg}", flush=True)


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return float(min(values))
    if q >= 1.0:
        return float(max(values))
    s = sorted(float(v) for v in values)
    pos = (len(s) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(s[lo])
    w = pos - lo
    return float(s[lo] * (1.0 - w) + s[hi] * w)
