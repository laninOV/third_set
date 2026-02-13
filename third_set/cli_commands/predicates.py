"""CLI predicates for event state checks."""

from __future__ import annotations


def _is_bo3(event_payload: dict) -> bool:
    e = event_payload.get("event") or {}
    try:
        return int(e.get("defaultPeriodCount") or 0) == 3
    except Exception:
        return False


def _is_live(event_payload: dict) -> bool:
    e = event_payload.get("event") or {}
    status = e.get("status") or {}
    return (status.get("type") or "").lower() == "inprogress"


def _is_set_score_1_1_after_two_sets(event_payload: dict) -> bool:
    e = event_payload.get("event") or {}
    status = e.get("status") or {}
    if (status.get("type") or "").lower() != "inprogress":
        return False
    home = e.get("homeScore") or {}
    away = e.get("awayScore") or {}
    if home.get("current") != 1 or away.get("current") != 1:
        return False
    return (
        home.get("period1") is not None
        and away.get("period1") is not None
        and home.get("period2") is not None
        and away.get("period2") is not None
    )
