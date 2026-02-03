from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Ratio:
    won: int
    total: int

    @property
    def rate(self) -> Optional[float]:
        if self.total <= 0:
            return None
        return self.won / self.total


_RATIO_RE = re.compile(r"(?P<won>\d+)\s*/\s*(?P<total>\d+)")


def parse_ratio(text: Any) -> Optional[Ratio]:
    if text is None:
        return None
    s = str(text)
    m = _RATIO_RE.search(s)
    if not m:
        return None
    won = int(m.group("won"))
    total = int(m.group("total"))
    if won < 0 or total < 0 or won > total:
        return None
    return Ratio(won=won, total=total)


def _iter_period_groups(stats_json: Dict[str, Any], *, periods: Tuple[str, ...]):
    wanted = {p.upper() for p in periods}
    for st in stats_json.get("statistics", []) or []:
        if (st.get("period") or "").upper() not in wanted:
            continue
        for group in st.get("groups", []) or []:
            yield st.get("period"), group


def sum_event_value(
    stats_json: Dict[str, Any], *, periods: Tuple[str, ...], group_name: str, item_name: str
) -> Tuple[int, int]:
    home = 0
    away = 0
    for _period, group in _iter_period_groups(stats_json, periods=periods):
        if group.get("groupName") != group_name:
            continue
        for item in group.get("statisticsItems", []) or []:
            if item.get("name") != item_name:
                continue
            home += int(item.get("homeValue") or 0)
            away += int(item.get("awayValue") or 0)
    return home, away


def sum_ratio_stat(
    stats_json: Dict[str, Any], *, periods: Tuple[str, ...], group_name: str, item_name: str
) -> Tuple[Optional[Ratio], Optional[Ratio]]:
    home_won = 0
    home_total = 0
    away_won = 0
    away_total = 0
    seen = False
    for _period, group in _iter_period_groups(stats_json, periods=periods):
        if group.get("groupName") != group_name:
            continue
        for item in group.get("statisticsItems", []) or []:
            if item.get("name") != item_name:
                continue
            rh = parse_ratio(item.get("home"))
            ra = parse_ratio(item.get("away"))
            if rh is None or ra is None:
                continue
            home_won += rh.won
            home_total += rh.total
            away_won += ra.won
            away_total += ra.total
            seen = True
    if not seen:
        return None, None
    # Keep Ratio even when total==0 so callers can distinguish "present but 0/0"
    # from "missing entirely". Rate will still be None via Ratio.rate.
    home_ratio = Ratio(home_won, home_total)
    away_ratio = Ratio(away_won, away_total)
    return home_ratio, away_ratio


def get_periods_present(stats_json: Dict[str, Any]) -> set:
    return {(s.get("period") or "").upper() for s in (stats_json.get("statistics") or [])}


def get_item_raw(
    stats_json: Dict[str, Any], *, period: str, group_name: str, item_name: str
) -> Optional[Dict[str, Any]]:
    p = (period or "").upper()
    for st in stats_json.get("statistics", []) or []:
        if (st.get("period") or "").upper() != p:
            continue
        for group in st.get("groups", []) or []:
            if group.get("groupName") != group_name:
                continue
            for item in group.get("statisticsItems", []) or []:
                if item.get("name") != item_name:
                    continue
                return item
    return None
