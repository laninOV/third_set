"""History source discovery exports and source state."""

from dataclasses import dataclass

from .orchestrator import _history_source_unavailable  # noqa: F401


@dataclass(frozen=True)
class HistorySourceState:
    links_count: int
    api_ok: bool
    candidates: int
    recovery_pass_used: bool
    source_health: str


def history_source_unavailable(
    *,
    links_count: int,
    api_history_fetch_failed: int,
    candidates: int,
    valid: int,
) -> bool:
    return _history_source_unavailable(
        links_count=links_count,
        api_history_fetch_failed=api_history_fetch_failed,
        candidates=candidates,
        valid=valid,
    )
