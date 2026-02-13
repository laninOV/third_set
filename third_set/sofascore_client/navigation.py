"""Navigation/network helpers for Sofascore scraping."""

from .client_impl import (  # noqa: F401
    _safe_goto,
    _collect_json_via_navigation,
    fetch_json_via_page,
    get_event_via_navigation,
    get_team_last_events_via_navigation,
)
