from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from third_set.dominance import DominanceLivePoints
from third_set.stats_parser import Ratio, get_item_raw, get_periods_present, parse_ratio, sum_event_value, sum_ratio_stat


@dataclass(frozen=True)
class MatchSnapshot:
    event_id: int
    stats: Dict[str, Any]

    def periods(self) -> set:
        return get_periods_present(self.stats)

    def points_by_periods(self, periods: Tuple[str, ...]) -> Optional[DominanceLivePoints]:
        spw_home, spw_away = sum_event_value(self.stats, periods=periods, group_name="Points", item_name="Service points won")
        rpw_home, rpw_away = sum_event_value(self.stats, periods=periods, group_name="Points", item_name="Receiver points won")
        if (spw_home + spw_away + rpw_home + rpw_away) == 0:
            return None
        return DominanceLivePoints(spw_home=spw_home, rpw_home=rpw_home, spw_away=spw_away, rpw_away=rpw_away)

    def ratio(self, *, periods: Tuple[str, ...], group: str, item: str) -> Tuple[Optional[Ratio], Optional[Ratio]]:
        return sum_ratio_stat(self.stats, periods=periods, group_name=group, item_name=item)

    def value(self, *, periods: Tuple[str, ...], group: str, item: str) -> Tuple[int, int]:
        return sum_event_value(self.stats, periods=periods, group_name=group, item_name=item)

    def raw(self, *, period: str, group: str, item: str) -> Optional[Dict[str, Any]]:
        return get_item_raw(self.stats, period=period, group_name=group, item_name=item)
