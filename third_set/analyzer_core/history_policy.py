"""Policy/state objects for history scan tuning."""

from dataclasses import dataclass
from typing import Set

from .orchestrator import _tab_unreachable_failfast_hits, _timeout_failfast_hits


@dataclass(frozen=True)
class HistoryScanPolicy:
    tab_unreachable_failfast_hits: int
    timeout_failfast_hits: int
    reset_codes: Set[str]


def default_history_scan_policy(reset_codes: Set[str]) -> HistoryScanPolicy:
    return HistoryScanPolicy(
        tab_unreachable_failfast_hits=_tab_unreachable_failfast_hits(),
        timeout_failfast_hits=_timeout_failfast_hits(),
        reset_codes=set(reset_codes or set()),
    )
