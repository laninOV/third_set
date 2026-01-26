from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def sd(self) -> Optional[float]:
        if self.n < 2:
            return None
        return math.sqrt(self.m2 / (self.n - 1))


@dataclass(frozen=True)
class MetricSummary:
    n: int
    mean: Optional[float]
    sd: Optional[float]


def summarize(values: Iterable[Optional[float]]) -> MetricSummary:
    rs = RunningStats()
    for v in values:
        if v is None:
            continue
        rs.add(float(v))
    return MetricSummary(n=rs.n, mean=(rs.mean if rs.n else None), sd=rs.sd)


def normalize_surface(surface: Optional[str]) -> str:
    s = (surface or "").lower()
    if "clay" in s:
        return "clay"
    if "grass" in s:
        return "grass"
    if "hard" in s or "hardcourt" in s:
        return "hard"
    if "carpet" in s:
        return "carpet"
    return "unknown"


def deviation(current: Optional[float], summary: MetricSummary) -> Optional[float]:
    if current is None or summary.mean is None:
        return None
    return float(current) - float(summary.mean)

