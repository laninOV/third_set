from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DominanceLivePoints:
    spw_home: int
    rpw_home: int
    spw_away: int
    rpw_away: int

    @property
    def tp_home(self) -> int:
        return self.spw_home + self.rpw_home

    @property
    def tp_away(self) -> int:
        return self.spw_away + self.rpw_away

    @property
    def n_points(self) -> int:
        return self.tp_home + self.tp_away


@dataclass(frozen=True)
class PlayerProfile:
    n_matches: int
    close_win_rate: Optional[float]
    close_n: int
    decider_win_rate: Optional[float]
    decider_n: int


@dataclass(frozen=True)
class DominanceVerdict:
    side: str  # "home" | "away" | "neutral"
    strength: int  # 0..3
    tpw_home: float
    n_points: int
    rd_home: Optional[float]
    explain: List[str]
    profile_home: Optional[PlayerProfile] = None
    profile_away: Optional[PlayerProfile] = None


def _safe_div(n: float, d: float) -> Optional[float]:
    if d == 0:
        return None
    return n / d


def compute_live_metrics(points: DominanceLivePoints) -> Tuple[Optional[float], int, Optional[float]]:
    n_points = points.n_points
    tpw_home = _safe_div(float(points.tp_home), float(n_points))
    rd_home = _safe_div(float(points.rpw_home), float(points.rpw_home + points.rpw_away))
    return tpw_home, n_points, rd_home


def base_side(tpw_home: float) -> str:
    if tpw_home > 0.5:
        return "home"
    if tpw_home < 0.5:
        return "away"
    return "neutral"


def strength_from_tpw_and_n(*, tpw_home: float, n_points: int) -> int:
    delta = abs(tpw_home - 0.5)
    if n_points >= 80 and delta >= 0.03:
        return 3
    if n_points >= 60 and delta >= 0.02:
        return 2
    if n_points >= 50 and delta >= 0.015:
        return 1
    return 0


def apply_return_dominance_adjustment(
    *, strength: int, side: str, rd_home: Optional[float]
) -> Tuple[int, List[str]]:
    if strength <= 0 or side == "neutral" or rd_home is None:
        return strength, []

    reasons: List[str] = []
    adjusted = strength
    if rd_home >= 0.56:
        if side == "home":
            adjusted += 1
            reasons.append(f"RD_home={rd_home:.3f} -> +1")
        else:
            adjusted -= 1
            reasons.append(f"RD_home={rd_home:.3f} -> -1")
    elif rd_home <= 0.44:
        if side == "away":
            adjusted += 1
            reasons.append(f"RD_home={rd_home:.3f} -> +1")
        else:
            adjusted -= 1
            reasons.append(f"RD_home={rd_home:.3f} -> -1")

    return max(0, min(3, adjusted)), reasons


def compute_player_profile(rows: List[Dict[str, Any]]) -> PlayerProfile:
    # rows: {won: bool, tpw: float, n_points: int, is_bo3: bool, played_3rd: bool}
    close_den = 0
    close_num = 0
    dec_den = 0
    dec_num = 0

    for r in rows:
        won = bool(r["won"])
        tpw = float(r["tpw"])
        if 0.49 <= tpw <= 0.51:
            close_den += 1
            if won:
                close_num += 1

        if bool(r.get("is_bo3")) and bool(r.get("played_3rd")):
            dec_den += 1
            if won:
                dec_num += 1

    close_rate = (close_num / close_den) if close_den >= 5 else None
    dec_rate = (dec_num / dec_den) if dec_den >= 8 else None
    return PlayerProfile(
        n_matches=len(rows),
        close_win_rate=close_rate,
        close_n=close_den,
        decider_win_rate=dec_rate,
        decider_n=dec_den,
    )


def apply_profile_adjustment(
    *,
    side: str,
    strength: int,
    base_strength: int,
    profile_home: Optional[PlayerProfile],
    profile_away: Optional[PlayerProfile],
) -> Tuple[int, str, List[str]]:
    reasons: List[str] = []
    adjusted_strength = strength
    adjusted_side = side

    # Rescue neutral with “close-match ability” only when the live signal is neutral/weak.
    if base_strength == 0:
        if profile_home and profile_away:
            if (
                profile_home.close_win_rate is not None
                and profile_away.close_win_rate is not None
                and profile_home.close_win_rate >= 0.60
                and profile_away.close_win_rate <= 0.40
            ):
                adjusted_side = "home"
                adjusted_strength = max(adjusted_strength, 1)
                reasons.append(
                    f"CloseWinRate home={profile_home.close_win_rate:.2f} (n={profile_home.close_n}) vs away={profile_away.close_win_rate:.2f} (n={profile_away.close_n}) -> home"
                )
            elif (
                profile_home.close_win_rate is not None
                and profile_away.close_win_rate is not None
                and profile_away.close_win_rate >= 0.60
                and profile_home.close_win_rate <= 0.40
            ):
                adjusted_side = "away"
                adjusted_strength = max(adjusted_strength, 1)
                reasons.append(
                    f"CloseWinRate away={profile_away.close_win_rate:.2f} (n={profile_away.close_n}) vs home={profile_home.close_win_rate:.2f} (n={profile_home.close_n}) -> away"
                )

    # Strengthen weak/medium with decider profile difference.
    if adjusted_side in ("home", "away") and adjusted_strength in (1, 2):
        if profile_home and profile_away and profile_home.decider_win_rate is not None and profile_away.decider_win_rate is not None:
            diff = profile_home.decider_win_rate - profile_away.decider_win_rate
            if adjusted_side == "away":
                diff = -diff
            if diff >= 0.20:
                adjusted_strength = min(3, adjusted_strength + 1)
                reasons.append(
                    f"DeciderWinRate edge={diff:+.2f} (home={profile_home.decider_win_rate:.2f}, away={profile_away.decider_win_rate:.2f}) -> +1"
                )

    return max(0, min(3, adjusted_strength)), adjusted_side if adjusted_strength > 0 else "neutral", reasons


def make_verdict(
    *,
    tpw_home: float,
    n_points: int,
    rd_home: Optional[float],
    side: str,
    strength: int,
    explain: List[str],
    profile_home: Optional[PlayerProfile],
    profile_away: Optional[PlayerProfile],
) -> DominanceVerdict:
    return DominanceVerdict(
        side=side,
        strength=strength,
        tpw_home=tpw_home,
        n_points=n_points,
        rd_home=rd_home,
        explain=explain,
        profile_home=profile_home,
        profile_away=profile_away,
    )

