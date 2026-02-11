from __future__ import annotations

import math
import asyncio
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from playwright.async_api import Page

from third_set.calibration import MetricSummary, deviation, normalize_surface, summarize
from third_set.dominance import DominanceLivePoints
from third_set.snapshot import MatchSnapshot
from third_set.modules import (
    ModuleResult,
    module1_dominance,
    module1_history_tpw12,
    module2_second_serve_fragility,
    module2_history_serve,
    module3_return_pressure,
    module3_history_return,
    module4_clutch,
    module4_history_clutch,
    module5_form_profile,
)
from third_set.sofascore import (
    SofascoreError,
    SOFASCORE_TENNIS_URL,
    _sofascore_base_from_url,
    get_event_votes,
    get_last_finished_singles_events,
    get_event_from_match_url_auto,
    get_event_from_match_url_via_navigation,
    get_event_via_navigation,
    parse_event_id_from_match_link,
    is_singles_event,
    summarize_event_for_team,
    warm_player_page,
    discover_player_match_links,
    discover_player_match_links_from_profile_url,
    discover_player_profile_urls_from_match,
)
from third_set.stats_parser import sum_event_value
from third_set.dom_stats import DomStatsError, extract_statistics_dom


@dataclass(frozen=True)
class MatchContext:
    event_id: int
    url: str
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    surface: str
    set2_winner: str  # "home"|"away"|"neutral"


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


def _metric_summary_to_dict(ms: Optional[MetricSummary]) -> Optional[Dict[str, Any]]:
    if ms is None:
        return None
    return {"n": ms.n, "mean": ms.mean, "sd": ms.sd}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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


def _tab_unreachable_failfast_hits() -> int:
    try:
        v = int(os.getenv("THIRDSET_HISTORY_TAB_UNREACHABLE_FAILFAST_HITS") or "2")
    except Exception:
        v = 2
    return max(1, min(6, v))


def _timeout_failfast_hits() -> int:
    try:
        v = int(os.getenv("THIRDSET_HISTORY_TIMEOUT_FAILFAST_HITS") or "4")
    except Exception:
        v = 4
    return max(1, min(6, v))


def _score_from_history(
    current: Optional[float],
    *,
    hist: Optional[MetricSummary],
    fallback_mean: float,
    fallback_sd: float,
    higher_is_better: bool = True,
    max_abs_z: float = 2.0,
) -> Optional[float]:
    """
    Maps a metric into a [0..1] score using a clipped z-score.
    0.5 = baseline, >0.5 better, <0.5 worse.
    """
    if current is None:
        return None
    mean = fallback_mean
    sd = fallback_sd
    if hist and hist.mean is not None and hist.sd is not None and hist.n >= 3 and hist.sd > 1e-9:
        mean = float(hist.mean)
        sd = float(hist.sd)
    z = (float(current) - mean) / sd
    z = _clamp(z, -max_abs_z, max_abs_z)
    if not higher_is_better:
        z = -z
    # linear mapping with clip
    return _clamp(0.5 + 0.25 * z, 0.0, 1.0)


def _weighted_score(parts: List[Tuple[Optional[float], float]]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for v, w in parts:
        if v is None or w <= 0:
            continue
        num += float(v) * float(w)
        den += float(w)
    if den <= 0:
        return None
    return num / den


def _index_pack(*, home: Optional[float], away: Optional[float]) -> Dict[str, Any]:
    diff = None
    if home is not None and away is not None:
        diff = float(home) - float(away)
    return {"home": home, "away": away, "diff": diff}


def _stability_from_sd(sd: Optional[float], *, scale: float) -> Optional[float]:
    if sd is None:
        return None
    s = float(sd)
    if s < 0:
        return None
    # exp(-sd/scale) in (0..1], higher=more stable
    return float(math.exp(-s / float(scale)))


def _signals_from_features(
    *,
    features: Dict[str, Any],
    max_history: int,
) -> Dict[str, Any]:
    """
    Returns three per-player signals (0..1):
      - strength: HISTORICAL strength proxy from last-N (always available)
      - form: recent change/momentum from last-N (always available)
      - stability: how consistent player is in last-N history (low variance)

    Uses only available metrics; gracefully degrades.
    """
    idx = (features.get("indices") or {}) if isinstance(features, dict) else {}
    cur = (features.get("current") or {}) if isinstance(features, dict) else {}
    hist = (features.get("history") or {}) if isinstance(features, dict) else {}
    cal = (hist.get("calibration") or {}) if isinstance(hist, dict) else {}

    # 1) Form: recent results + recent per-match stat quality (if available)
    def _form_window(side: str, *, window_n: int) -> Optional[float]:
        h = hist.get(side) if isinstance(hist, dict) else None
        if not isinstance(h, dict):
            return None
        n = int(window_n or 0)
        rows = ((hist.get("rows") or {}).get(side)) if isinstance(hist.get("rows"), dict) else None

        surface = None
        try:
            surface = (features.get("context") or {}).get("surface")
        except Exception:
            surface = None

        def _w(i: int, row_surface: Optional[str]) -> float:
            # Recency weights (half-life ~3 matches for small N).
            half_life = 2.5 if max_history <= 5 else 4.0
            lam = math.log(2.0) / max(0.5, half_life)
            w = math.exp(-lam * float(i))
            if surface and row_surface and str(row_surface) != str(surface):
                w *= 0.65
            return float(w)

        # 2.1 Results EWMA (wins + decider wins)
        wr_base = h.get("win_rate")
        dec_base = h.get("dec_rate")
        result_score = None
        if isinstance(rows, list) and rows:
            rows = rows[:window_n]
            wsum = 0.0
            wwin = 0.0
            wsum_dec = 0.0
            wwin_dec = 0.0
            for i, r in enumerate(rows):
                if not isinstance(r, dict):
                    continue
                w = _w(i, r.get("surface"))
                wsum += w
                wwin += w * (1.0 if r.get("won") else 0.0)
                if r.get("is_decider"):
                    wsum_dec += w
                    wwin_dec += w * (1.0 if r.get("won") else 0.0)
            wr = (wwin / wsum) if wsum > 0 else None
            dr = (wwin_dec / wsum_dec) if wsum_dec >= 1.5 else None
            wr_score = _score_from_history(wr, hist=None, fallback_mean=0.50, fallback_sd=0.25) if isinstance(wr, (int, float)) else None
            dr_score = _score_from_history(dr, hist=None, fallback_mean=0.50, fallback_sd=0.30) if isinstance(dr, (int, float)) else None
            result_score = _weighted_score([(wr_score, 0.75), (dr_score, 0.25 if dr_score is not None else 0.0)])
        else:
            wr_score = _score_from_history(wr_base, hist=None, fallback_mean=0.50, fallback_sd=0.25) if isinstance(wr_base, (int, float)) else None
            dr_score = _score_from_history(dec_base, hist=None, fallback_mean=0.50, fallback_sd=0.30) if isinstance(dec_base, (int, float)) else None
            result_score = _weighted_score([(wr_score, 0.75), (dr_score, 0.25 if dr_score is not None else 0.0)])

        # 2.2 Stat-quality EWMA (if available in history rows)
        def _norm(v: Optional[float], *, mean: float, sd: float) -> Optional[float]:
            if not isinstance(v, (int, float)):
                return None
            z = (float(v) - mean) / max(1e-6, sd)
            # softer than pure z: keep in reasonable range
            return _sigmoid(z / 1.6)

        stat_score = None
        if isinstance(rows, list) and rows:
            rows = rows[:window_n]
            parts: List[Tuple[Optional[float], float]] = []
            # Each metric becomes 0..1 via sigmoid, then aggregated with weights.
            # (baselines are generic, not player-specific; player-specific calibration is handled elsewhere)
            def _ewma_metric(key: str, mean: float, sd: float) -> Optional[float]:
                wsum = 0.0
                wacc = 0.0
                for i, r in enumerate(rows):
                    if not isinstance(r, dict):
                        continue
                    v = r.get(key)
                    s = _norm(v, mean=mean, sd=sd)
                    if s is None:
                        continue
                    w = _w(i, r.get("surface"))
                    wsum += w
                    wacc += w * float(s)
                return (wacc / wsum) if wsum > 0 else None

            parts.append((_ewma_metric("tpw", 0.50, 0.03), 0.28))
            parts.append((_ewma_metric("rpr_12", 0.38, 0.06), 0.26))
            parts.append((_ewma_metric("ssw_12", 0.50, 0.10), 0.20))
            parts.append((_ewma_metric("bpsr_12", 0.60, 0.22), 0.14))
            parts.append((_ewma_metric("bpconv_12", 0.35, 0.20), 0.12))
            stat_score = _weighted_score(parts)

        base = _weighted_score(
            [
                (result_score, 0.35),
                (stat_score, 0.65 if stat_score is not None else 0.0),
            ]
        )
        if base is None:
            return None

        # Reliability shrink towards 0.5 for small N / missing stats.
        rel = _clamp(n / max(3.0, float(max_history)), 0.0, 1.0)
        if stat_score is None:
            rel *= 0.75
        return (0.5 * (1.0 - rel)) + (base * rel)

    # History form signal: use TPW12 Rating (0..1) computed from last5/last3 (always numeric).
    tpw12_scores = (hist.get("tpw12_scores") or {}) if isinstance(hist, dict) else {}

    def _rating_norm(side: str, key: str) -> Optional[float]:
        sc = (tpw12_scores.get(side) or {}).get(key)
        if isinstance(sc, dict):
            r = sc.get("rating")
            if isinstance(r, (int, float)):
                return _clamp(float(r) / 100.0, 0.0, 1.0)
        return None

    def _score_from_hist_mean(side: str, key: str, *, mean: float, sd: float, higher: bool = True) -> Optional[float]:
        c = cal.get(side) if isinstance(cal, dict) else None
        if not isinstance(c, dict):
            return None
        ms = c.get(key)
        if isinstance(ms, MetricSummary):
            v = ms.mean
        elif isinstance(ms, dict):
            v = ms.get("mean")
        else:
            v = None
        if not isinstance(v, (int, float)):
            return None
        return _score_from_history(float(v), hist=None, fallback_mean=mean, fallback_sd=sd, higher_is_better=higher)

    strength3_h = _rating_norm("home", "last3")
    strength3_a = _rating_norm("away", "last3")
    strength5_h = _rating_norm("home", "last5")
    strength5_a = _rating_norm("away", "last5")

    # History strength: combine TPW rating with historical serve/return/clutch quality.
    serve_h = _weighted_score(
        [
            (_score_from_hist_mean("home", "ssw_12", mean=0.50, sd=0.10), 0.60),
            (_score_from_hist_mean("home", "bpsr_12", mean=0.60, sd=0.20), 0.40),
        ]
    )
    serve_a = _weighted_score(
        [
            (_score_from_hist_mean("away", "ssw_12", mean=0.50, sd=0.10), 0.60),
            (_score_from_hist_mean("away", "bpsr_12", mean=0.60, sd=0.20), 0.40),
        ]
    )
    ret_h = _weighted_score(
        [
            (_score_from_hist_mean("home", "rpr_12", mean=0.38, sd=0.06), 0.65),
            (_score_from_hist_mean("home", "bpconv_12", mean=0.35, sd=0.20), 0.35),
        ]
    )
    ret_a = _weighted_score(
        [
            (_score_from_hist_mean("away", "rpr_12", mean=0.38, sd=0.06), 0.65),
            (_score_from_hist_mean("away", "bpconv_12", mean=0.35, sd=0.20), 0.35),
        ]
    )
    clutch_h = _weighted_score(
        [
            (_score_from_hist_mean("home", "bpsr_12", mean=0.60, sd=0.25), 0.60),
            (_score_from_hist_mean("home", "bpconv_12", mean=0.35, sd=0.25), 0.40),
        ]
    )
    clutch_a = _weighted_score(
        [
            (_score_from_hist_mean("away", "bpsr_12", mean=0.60, sd=0.25), 0.60),
            (_score_from_hist_mean("away", "bpconv_12", mean=0.35, sd=0.25), 0.40),
        ]
    )

    strength_h = _weighted_score(
        [
            (strength5_h, 0.55),
            (serve_h, 0.15 if serve_h is not None else 0.0),
            (ret_h, 0.20 if ret_h is not None else 0.0),
            (clutch_h, 0.10 if clutch_h is not None else 0.0),
        ]
    )
    strength_a = _weighted_score(
        [
            (strength5_a, 0.55),
            (serve_a, 0.15 if serve_a is not None else 0.0),
            (ret_a, 0.20 if ret_a is not None else 0.0),
            (clutch_a, 0.10 if clutch_a is not None else 0.0),
        ]
    )

    # Form: momentum from last3 vs last5 + optional windowed form.
    # Trend only makes sense when both windows exist; do not invent baselines.
    trend_h = _clamp(0.5 + (strength3_h - strength5_h) * 1.2, 0.0, 1.0) if (strength3_h is not None and strength5_h is not None) else None
    trend_a = _clamp(0.5 + (strength3_a - strength5_a) * 1.2, 0.0, 1.0) if (strength3_a is not None and strength5_a is not None) else None
    form_win_h = _form_window("home", window_n=min(5, max_history))
    form_win_a = _form_window("away", window_n=min(5, max_history))
    form_h = _weighted_score([(trend_h, 0.55), (form_win_h, 0.45 if form_win_h is not None else 0.0)])
    form_a = _weighted_score([(trend_a, 0.55), (form_win_a, 0.45 if form_win_a is not None else 0.0)])

    # Also expose ratings as separate signals for printing.
    form3_h = strength3_h
    form3_a = strength3_a
    form5_h = strength5_h
    form5_a = strength5_a

    # 3) Stability: exp(-sd/scale) over key history metrics, weighted by availability.
    def _stability(side: str) -> Optional[float]:
        c = cal.get(side) if isinstance(cal, dict) else None
        if not isinstance(c, dict):
            return None
        def _ms_sd(key: str) -> Optional[float]:
            ms = c.get(key)
            if isinstance(ms, MetricSummary):
                return ms.sd
            if isinstance(ms, dict):
                sd = ms.get("sd")
                return float(sd) if isinstance(sd, (int, float)) else None
            return None

        def _ms_n(key: str) -> int:
            ms = c.get(key)
            if isinstance(ms, MetricSummary):
                return int(ms.n)
            if isinstance(ms, dict):
                try:
                    return int(ms.get("n") or 0)
                except Exception:
                    return 0
            return 0
        parts: List[Tuple[Optional[float], float]] = []
        parts.append((_stability_from_sd(_ms_sd("ssw_12"), scale=0.12), 0.30))
        parts.append((_stability_from_sd(_ms_sd("rpr_12"), scale=0.06), 0.30))
        parts.append((_stability_from_sd(_ms_sd("bpsr_12"), scale=0.25), 0.20))
        parts.append((_stability_from_sd(_ms_sd("bpconv_12"), scale=0.30), 0.20))
        base = _weighted_score(parts)
        if base is None:
            return None
        # If history sample is tiny, shrink towards 0.5 (unknown stability)
        n_min = 0
        try:
            n_min = min(
                _ms_n("ssw_12"),
                _ms_n("rpr_12"),
            )
        except Exception:
            n_min = 0
        rel = _clamp(n_min / 8.0, 0.0, 1.0)
        return (0.5 * (1.0 - rel)) + (base * rel)

    # Stability: prefer TPW volatility (0..100) from last5 if present, else SD-based.
    def _vol_to_stab(side: str) -> Optional[float]:
        sc = (tpw12_scores.get(side) or {}).get("last5")
        if isinstance(sc, dict):
            v = sc.get("volatility")
            if isinstance(v, (int, float)):
                return _clamp(1.0 - float(v) / 100.0, 0.0, 1.0)
        return None

    stability_h = _vol_to_stab("home")
    if stability_h is None:
        stability_h = _stability("home")
    stability_a = _vol_to_stab("away")
    if stability_a is None:
        stability_a = _stability("away")

    return {
        "strength": _index_pack(home=strength_h, away=strength_a),
        "form": _index_pack(home=form_h, away=form_a),
        "form3": _index_pack(home=form3_h, away=form3_a),
        "form5": _index_pack(home=form5_h, away=form5_a),
        "stability": _index_pack(home=stability_h, away=stability_a),
    }


def _probability_from_signals(
    *,
    signals: Dict[str, Any],
    mods: List[ModuleResult],
) -> Dict[str, Any]:
    """
    Forecast probability from strength+form adjusted by stability and module evidence.
    Output:
      p_home, p_away, confidence (0..1), diff, components.
    """
    def g(name: str) -> Optional[float]:
        it = signals.get(name) or {}
        if not isinstance(it, dict):
            return None
        d = it.get("diff")
        return float(d) if isinstance(d, (int, float)) else None

    d_strength = g("strength")
    d_form = g("form")
    d_stab = g("stability")  # optional

    # Module evidence: normalized into [-1..1]
    mod_score = 0.0
    mod_home = 0.0
    mod_away = 0.0
    active = 0
    for m in mods:
        if m.side == "neutral" or m.strength <= 0:
            continue
        active += 1
        s = float(m.strength)
        if m.side == "home":
            mod_home += s
            mod_score += s
        else:
            mod_away += s
            mod_score -= s
    mod_total = mod_home + mod_away
    if mod_total > 0:
        # Balance directional vote by total evidence volume.
        # If modules conflict heavily, the normalized signal weakens.
        mod_dir = mod_score / mod_total  # [-1..1]
        mod_vol = min(1.0, mod_total / 6.0)  # saturates around medium evidence
        mod_norm = _clamp(mod_dir * mod_vol, -1.0, 1.0)
        mod_consistency = abs(mod_dir)
    else:
        mod_norm = 0.0
        mod_consistency = 0.0

    # Base linear logit from signals
    # Strength is primary (historical rating), form is secondary (momentum).
    no_signal = (d_strength is None and d_form is None and active == 0)
    ds = float(d_strength) if isinstance(d_strength, (int, float)) else 0.0
    df = float(d_form) if isinstance(d_form, (int, float)) else 0.0
    x = 3.0 * ds + 1.2 * df + 1.0 * mod_norm
    # Stability gates confidence more than direction: if both unstable, flatten probability.
    gate = 1.0
    if isinstance(d_stab, (int, float)):
        # d_stab is (home-away) of stability, but we want absolute stability amount; approximate via home/away.
        sh = (signals.get("stability") or {}).get("home")
        sa = (signals.get("stability") or {}).get("away")
        if isinstance(sh, (int, float)) and isinstance(sa, (int, float)):
            gate = 0.70 + 0.30 * float(min(sh, sa))
    # Additional confidence gate from module agreement.
    # Contradicting modules should reduce certainty even if one side barely leads.
    if mod_total > 0:
        gate *= 0.85 + 0.15 * mod_consistency
    x *= gate

    p_home = _sigmoid(x) if not no_signal else 0.5
    p_away = 1.0 - p_home
    conf = (abs(p_home - 0.5) * 2.0) if not no_signal else 0.0
    pick = "home" if p_home > 0.5 else "away"
    if conf < 0.08:
        pick = "neutral"

    return {
        "p_home": p_home,
        "p_away": p_away,
        "confidence": conf,
        "pick": pick,
        "logit": x,
        "gate": gate,
        "components": {
            "d_strength": d_strength,
            "d_form": d_form,
            "mod_norm": mod_norm,
            "mod_score": mod_score,
            "mod_total": mod_total,
            "mod_consistency": mod_consistency,
            "active_mods": active,
            "no_signal": no_signal,
        },
    }

def _build_feature_dump(
    *,
    ctx: MatchContext,
    snapshot: MatchSnapshot,
    points: Optional[DominanceLivePoints],
    cal_home: Optional[Dict[str, MetricSummary]],
    cal_away: Optional[Dict[str, MetricSummary]],
    rows_home: List[Dict[str, Any]],
    rows_away: List[Dict[str, Any]],
) -> Dict[str, Any]:
    periods_12 = ("1ST", "2ND")

    # Current match (1ST+2ND)
    cur: Dict[str, Any] = {}
    if points is not None:
        spw_h, spw_a = points.spw_home, points.spw_away
        rpw_h, rpw_a = points.rpw_home, points.rpw_away
        tp_h, tp_a = spw_h + rpw_h, spw_a + rpw_a
        n_pts = tp_h + tp_a
        tpw_h = (tp_h / n_pts) if n_pts else None
        ret_n_h = rpw_h + spw_a
        ret_n_a = rpw_a + spw_h
        rr_h = (rpw_h / ret_n_h) if ret_n_h else None
        rr_a = (rpw_a / ret_n_a) if ret_n_a else None
        dr = (rr_h / rr_a) if (rr_h is not None and rr_a not in (None, 0)) else None
        cur["points"] = {
            "spw": {"home": spw_h, "away": spw_a},
            "rpw": {"home": rpw_h, "away": rpw_a},
            "tp": {"home": tp_h, "away": tp_a},
            "n": n_pts,
            "tpw_home": tpw_h,
            "rr": {"home": rr_h, "away": rr_a},
            "retN": {"home": ret_n_h, "away": ret_n_a},
            "dr": dr,
        }

    ss_h, ss_a = snapshot.ratio(periods=periods_12, group="Service", item="Second serve points")
    fspts_h, fspts_a = snapshot.ratio(periods=periods_12, group="Service", item="First serve points")
    fsin_h, fsin_a = snapshot.ratio(periods=periods_12, group="Service", item="First serve")
    aces_h, aces_a = snapshot.value(periods=periods_12, group="Service", item="Aces")
    df_h, df_a = snapshot.value(periods=periods_12, group="Service", item="Double faults")
    bps_h, bps_a = snapshot.ratio(periods=periods_12, group="Service", item="Break points saved")
    tb_h, tb_a = snapshot.value(periods=periods_12, group="Miscellaneous", item="Tiebreaks")
    rg_h, rg_a = snapshot.value(periods=periods_12, group="Return", item="Return games played")
    sgw_h, sgw_a = snapshot.value(periods=periods_12, group="Games", item="Service games won")
    r1_h, r1_a = snapshot.ratio(periods=periods_12, group="Return", item="First serve return points")
    r2_h, r2_a = snapshot.ratio(periods=periods_12, group="Return", item="Second serve return points")
    bpc_h, bpc_a = snapshot.value(periods=periods_12, group="Return", item="Break points converted")

    spn_h = (fspts_h.total if fspts_h else 0) + (ss_h.total if ss_h else 0)
    spn_a = (fspts_a.total if fspts_a else 0) + (ss_a.total if ss_a else 0)
    dfr_h = (df_h / spn_h) if spn_h else None
    dfr_a = (df_a / spn_a) if spn_a else None
    ace_rate_h = (aces_h / spn_h) if spn_h else None
    ace_rate_a = (aces_a / spn_a) if spn_a else None
    cur["serve"] = {
        "ssw": {"home": ss_h.rate if ss_h else None, "away": ss_a.rate if ss_a else None, "n2_min": min(ss_h.total if ss_h else 0, ss_a.total if ss_a else 0)},
        "fsw": {"home": fspts_h.rate if fspts_h else None, "away": fspts_a.rate if fspts_a else None, "n1_min": min(fspts_h.total if fspts_h else 0, fspts_a.total if fspts_a else 0)},
        "fsin": {"home": fsin_h.rate if fsin_h else None, "away": fsin_a.rate if fsin_a else None, "nfs_min": min(fsin_h.total if fsin_h else 0, fsin_a.total if fsin_a else 0)},
        "aces": {"home": aces_h, "away": aces_a, "ace_rate": {"home": ace_rate_h, "away": ace_rate_a}},
        "df": {"home": df_h, "away": df_a, "dfr": {"home": dfr_h, "away": dfr_a}},
        "bps": {"home": {"won": bps_h.won, "total": bps_h.total, "rate": bps_h.rate} if bps_h else None, "away": {"won": bps_a.won, "total": bps_a.total, "rate": bps_a.rate} if bps_a else None},
        "tb": {"home": tb_h, "away": tb_a},
        "serve_points_n": {"home": spn_h, "away": spn_a},
    }

    # RBR is only reliable if "Service games won" exists in stats (otherwise it may default to 0).
    sgw_present = bool(snapshot.raw(period="1ST", group="Games", item="Service games won") or snapshot.raw(period="2ND", group="Games", item="Service games won"))
    rb_h = max(0, rg_h - sgw_a) if sgw_present else None
    rb_a = max(0, rg_a - sgw_h) if sgw_present else None
    rbr_h = (rb_h / rg_h) if (rb_h is not None and rg_h) else None
    rbr_a = (rb_a / rg_a) if (rb_a is not None and rg_a) else None
    r1r_h = r1_h.rate if r1_h else None
    r1r_a = r1_a.rate if r1_a else None
    r2r_h = r2_h.rate if r2_h else None
    r2r_a = r2_a.rate if r2_a else None
    rpr_h = None
    rpr_a = None
    if r1r_h is not None and r1r_a is not None:
        if r2_h and r2_a and min(r2_h.total, r2_a.total) >= 20 and r2r_h is not None and r2r_a is not None:
            rpr_h = 0.4 * r1r_h + 0.6 * r2r_h
            rpr_a = 0.4 * r1r_a + 0.6 * r2r_a
        else:
            rpr_h = r1r_h
            rpr_a = r1r_a

    bp_tot_h = bps_a.total if bps_a else 0
    bp_tot_a = bps_h.total if bps_h else 0
    bpconv_h = (bpc_h / bp_tot_h) if bp_tot_h and 0 <= bpc_h <= bp_tot_h else None
    bpconv_a = (bpc_a / bp_tot_a) if bp_tot_a and 0 <= bpc_a <= bp_tot_a else None

    cur["return"] = {
        "rg": {"home": rg_h, "away": rg_a},
        "sgw": {"home": sgw_h, "away": sgw_a},
        "sgw_present": sgw_present,
        "rb": {"home": rb_h, "away": rb_a},
        "rbr": {"home": rbr_h, "away": rbr_a},
        "r1": {"home": {"won": r1_h.won, "total": r1_h.total, "rate": r1_h.rate} if r1_h else None, "away": {"won": r1_a.won, "total": r1_a.total, "rate": r1_a.rate} if r1_a else None},
        "r2": {"home": {"won": r2_h.won, "total": r2_h.total, "rate": r2_h.rate} if r2_h else None, "away": {"won": r2_a.won, "total": r2_a.total, "rate": r2_a.rate} if r2_a else None},
        "rpr": {"home": rpr_h, "away": rpr_a},
        "bpc": {"home": bpc_h, "away": bpc_a},
        "bp_tot": {"home": bp_tot_h, "away": bp_tot_a},
        "bpconv": {"home": bpconv_h, "away": bpconv_a},
    }

    # History summaries (last N)
    def _hist_basic(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        r = rows
        wins = sum(1 for x in r if x.get("won") is True)
        dec = [x for x in r if x.get("is_bo3_decider")]
        dec_w = sum(1 for x in dec if x.get("won") is True)
        s1 = [x for x in r if x.get("won_set1_known")]
        s1w = sum(1 for x in s1 if x.get("won_set1") is True)
        s2 = [x for x in r if x.get("won_set2_known")]
        s2w = sum(1 for x in s2 if x.get("won_set2") is True)
        return {
            "n": len(r),
            "wins": wins,
            "win_rate": (wins / len(r)) if r else None,
            "deciders": len(dec),
            "dec_wins": dec_w,
            "dec_rate": (dec_w / len(dec)) if len(dec) else None,
            "set1": {"n": len(s1), "wins": s1w, "rate": (s1w / len(s1)) if len(s1) else None},
            "set2": {"n": len(s2), "wins": s2w, "rate": (s2w / len(s2)) if len(s2) else None},
        }

    def _row_12_mean(row: Dict[str, Any], k1: str, k2: str) -> Optional[float]:
        a = row.get(k1)
        b = row.get(k2)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return float((a + b) / 2.0)
        if isinstance(a, (int, float)):
            return float(a)
        if isinstance(b, (int, float)):
            return float(b)
        return None

    def _hist_rows_compact(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "eventId": r.get("eventId"),
                    "surface": r.get("surface"),
                    "won": bool(r.get("won")),
                    "is_decider": bool(r.get("is_bo3_decider")),
                    "won_set1": bool(r.get("won_set1")),
                    "won_set1_known": bool(r.get("won_set1_known")),
                    "won_set2": bool(r.get("won_set2")),
                    "won_set2_known": bool(r.get("won_set2_known")),
                    "won_set3": bool(r.get("won_set3")),
                    "won_set3_known": bool(r.get("won_set3_known")),
                    "stats_ok": bool(r.get("stats_ok")),
                    "missing_all_metrics": bool(r.get("missing_all_metrics")),
                    "tpw": r.get("tpw"),
                    "tpw12": r.get("tpw12"),
                    "ssw_12": _row_12_mean(r, "ssw_1", "ssw_2"),
                    "rpr_12": _row_12_mean(r, "rpr_1", "rpr_2"),
                    "bpsr_12": _row_12_mean(r, "bpsr_1", "bpsr_2"),
                    "bpconv_12": _row_12_mean(r, "bpconv_1", "bpconv_2"),
                }
            )
        return out

    hist = {
        "home": _hist_basic(rows_home),
        "away": _hist_basic(rows_away),
        "rows": {
            "home": _hist_rows_compact(rows_home),
            "away": _hist_rows_compact(rows_away),
        },
        "tpw12_scores": {
            "home": {
                "last5": _tpw12_history_scores(rows_home, max_n=5),
                "last3": _tpw12_history_scores(rows_home, max_n=3),
            },
            "away": {
                "last5": _tpw12_history_scores(rows_away, max_n=5),
                "last3": _tpw12_history_scores(rows_away, max_n=3),
            },
        },
        "calibration": {
            "home": {k: _metric_summary_to_dict(v) for k, v in (cal_home or {}).items()},
            "away": {k: _metric_summary_to_dict(v) for k, v in (cal_away or {}).items()},
        },
    }

    # Computed indices (0..1) built from available metrics.
    idx: Dict[str, Any] = {}
    calh = cal_home or {}
    cala = cal_away or {}

    # Dominance index (points + return) - uses generic baselines.
    tpw = cur.get("points", {}).get("tpw_home")
    dr = cur.get("points", {}).get("dr")
    tpw_score_h = _score_from_history(tpw, hist=None, fallback_mean=0.50, fallback_sd=0.02, higher_is_better=True)
    tpw_score_a = _score_from_history((1 - tpw) if isinstance(tpw, (int, float)) else None, hist=None, fallback_mean=0.50, fallback_sd=0.02, higher_is_better=True)
    dr_score_h = _score_from_history(dr, hist=None, fallback_mean=1.00, fallback_sd=0.15, higher_is_better=True)
    dr_score_a = _score_from_history((1 / dr) if isinstance(dr, (int, float)) and dr > 0 else None, hist=None, fallback_mean=1.00, fallback_sd=0.15, higher_is_better=True)
    idx["dominance_index"] = _index_pack(
        home=_weighted_score([(tpw_score_h, 0.65), (dr_score_h, 0.35)]),
        away=_weighted_score([(tpw_score_a, 0.65), (dr_score_a, 0.35)]),
    )

    # Serve index: SSW, FSW, FSIN, DFR (lower better), Aces, BPSR.
    s = cur.get("serve") or {}
    ssw_h = (s.get("ssw") or {}).get("home")
    ssw_a = (s.get("ssw") or {}).get("away")
    fsw_h = (s.get("fsw") or {}).get("home")
    fsw_a = (s.get("fsw") or {}).get("away")
    fsinr_h = (s.get("fsin") or {}).get("home")
    fsinr_a = (s.get("fsin") or {}).get("away")
    dfr_h = (s.get("df") or {}).get("dfr", {}).get("home")
    dfr_a = (s.get("df") or {}).get("dfr", {}).get("away")
    ace_h = (s.get("aces") or {}).get("ace_rate", {}).get("home")
    ace_a = (s.get("aces") or {}).get("ace_rate", {}).get("away")
    bpsr_h = (s.get("bps") or {}).get("home", {}).get("rate") if isinstance((s.get("bps") or {}).get("home"), dict) else None
    bpsr_a = (s.get("bps") or {}).get("away", {}).get("rate") if isinstance((s.get("bps") or {}).get("away"), dict) else None

    idx["serve_index"] = _index_pack(
        home=_weighted_score(
            [
                (_score_from_history(ssw_h, hist=calh.get("ssw_12"), fallback_mean=0.50, fallback_sd=0.10), 0.22),
                (_score_from_history(fsw_h, hist=None, fallback_mean=0.75, fallback_sd=0.08), 0.22),
                (_score_from_history(fsinr_h, hist=None, fallback_mean=0.60, fallback_sd=0.10), 0.12),
                (_score_from_history(dfr_h, hist=None, fallback_mean=0.05, fallback_sd=0.02, higher_is_better=False), 0.16),
                (_score_from_history(ace_h, hist=None, fallback_mean=0.06, fallback_sd=0.03), 0.10),
                (_score_from_history(bpsr_h, hist=calh.get("bpsr_12"), fallback_mean=0.60, fallback_sd=0.20), 0.18),
            ]
        ),
        away=_weighted_score(
            [
                (_score_from_history(ssw_a, hist=cala.get("ssw_12"), fallback_mean=0.50, fallback_sd=0.10), 0.22),
                (_score_from_history(fsw_a, hist=None, fallback_mean=0.75, fallback_sd=0.08), 0.22),
                (_score_from_history(fsinr_a, hist=None, fallback_mean=0.60, fallback_sd=0.10), 0.12),
                (_score_from_history(dfr_a, hist=None, fallback_mean=0.05, fallback_sd=0.02, higher_is_better=False), 0.16),
                (_score_from_history(ace_a, hist=None, fallback_mean=0.06, fallback_sd=0.03), 0.10),
                (_score_from_history(bpsr_a, hist=cala.get("bpsr_12"), fallback_mean=0.60, fallback_sd=0.20), 0.18),
            ]
        ),
    )

    # Return pressure index: RPR (+BPconv), optionally RBR if reliable.
    r = cur.get("return") or {}
    rpr_h = (r.get("rpr") or {}).get("home")
    rpr_a = (r.get("rpr") or {}).get("away")
    bpconv_h = (r.get("bpconv") or {}).get("home")
    bpconv_a = (r.get("bpconv") or {}).get("away")
    rbr_h = (r.get("rbr") or {}).get("home")
    rbr_a = (r.get("rbr") or {}).get("away")
    include_rbr = bool(r.get("sgw_present")) and rbr_h is not None and rbr_a is not None
    idx["return_index"] = _index_pack(
        home=_weighted_score(
            [
                (_score_from_history(rpr_h, hist=calh.get("rpr_12"), fallback_mean=0.38, fallback_sd=0.06), 0.55),
                (_score_from_history(bpconv_h, hist=calh.get("bpconv_12"), fallback_mean=0.35, fallback_sd=0.20), 0.25),
                (_score_from_history(rbr_h, hist=None, fallback_mean=0.25, fallback_sd=0.15), 0.20 if include_rbr else 0.0),
            ]
        ),
        away=_weighted_score(
            [
                (_score_from_history(rpr_a, hist=cala.get("rpr_12"), fallback_mean=0.38, fallback_sd=0.06), 0.55),
                (_score_from_history(bpconv_a, hist=cala.get("bpconv_12"), fallback_mean=0.35, fallback_sd=0.20), 0.25),
                (_score_from_history(rbr_a, hist=None, fallback_mean=0.25, fallback_sd=0.15), 0.20 if include_rbr else 0.0),
            ]
        ),
    )

    # Clutch index (defense+offense clutch) on current match.
    idx["clutch_index"] = _index_pack(
        home=_weighted_score(
            [
                (_score_from_history(bpsr_h, hist=calh.get("bpsr_12"), fallback_mean=0.60, fallback_sd=0.25), 0.60),
                (_score_from_history(bpconv_h, hist=calh.get("bpconv_12"), fallback_mean=0.35, fallback_sd=0.25), 0.40),
            ]
        ),
        away=_weighted_score(
            [
                (_score_from_history(bpsr_a, hist=cala.get("bpsr_12"), fallback_mean=0.60, fallback_sd=0.25), 0.60),
                (_score_from_history(bpconv_a, hist=cala.get("bpconv_12"), fallback_mean=0.35, fallback_sd=0.25), 0.40),
            ]
        ),
    )

    # History rating index based on TPW in sets 1+2 (0..1, baseline 0.5). No NA.
    def _hist_rating_norm(side: str) -> Optional[float]:
        try:
            sc = ((hist.get("tpw12_scores") or {}).get(side) or {}).get("last5")
        except Exception:
            sc = None
        if isinstance(sc, dict):
            r = sc.get("rating")
            if isinstance(r, (int, float)):
                return _clamp(float(r) / 100.0, 0.0, 1.0)
        return None

    idx["history_form_index"] = _index_pack(home=_hist_rating_norm("home"), away=_hist_rating_norm("away"))

    return {
        "context": {"eventId": ctx.event_id, "url": ctx.url, "surface": ctx.surface},
        "current": cur,
        "history": hist,
        "indices": idx,
        "formula_notes": {
            "index_scale": "0..1 (0.5=baseline)",
            "serve_index": "SSW/FSW/FSIN/(1-DFR)/Aces/BPSR",
            "return_index": "RPR/BPconv (+RBR if Service games won present)",
            "clutch_index": "BPSR+BPconv",
            "dominance_index": "TPW+DR (RR ratio)",
            "history_form_index": "History rating from TPW(sets1+2): Power/Form/Volatility -> Rating",
        },
    }


def _sigmoid(x: float) -> float:
    # numerically safe-ish for our small x
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _tpw12_history_scores(rows: List[Dict[str, Any]], *, max_n: int = 5) -> Dict[str, Any]:
    """
    Исторические числовые показатели по TPW в 1+2 сетах (tpw12), newest-first:
    - power (0..100): сила по среднему TPW
    - form (0..100): импульс (среднее last3 vs lastN)
    - volatility (0..100): качели (sd; больше = хуже)
    - rating (0..100): итоговая оценка
    - reliability (0..100): надёжность (по N)

    Если данных нет (n=0) — НЕ придумывает нейтральные 50/50, а возвращает None-поля.
    """
    vals: List[float] = []
    for r in (rows or [])[: max(0, int(max_n))]:
        if not isinstance(r, dict):
            continue
        v = r.get("tpw12")
        if not isinstance(v, (int, float)):
            v = r.get("tpw")
        if isinstance(v, (int, float)):
            vv = float(v)
            if 0.0 <= vv <= 1.0:
                vals.append(vv)

    def clamp100(x: float) -> float:
        return float(max(0.0, min(100.0, x)))

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs))

    def sd(xs: List[float]) -> float:
        m = mean(xs)
        v = sum((x - m) ** 2 for x in xs) / len(xs)
        return float(math.sqrt(v))

    n = len(vals)
    if n == 0:
        return {
            "n": 0,
            "mu_pp": None,
            "delta_pp": None,
            "sigma_pp": None,
            "power": None,
            "form": None,
            "volatility": None,
            "rating": None,
            "reliability": 0.0,
            "values": [],
        }

    mu = mean(vals)
    sig = sd(vals)
    k = min(3, n)
    mu_recent = mean(vals[:k])
    delta = mu_recent - mu

    mu_pp = (mu - 0.5) * 100.0
    delta_pp = delta * 100.0
    sigma_pp = sig * 100.0

    # Scale constants (v1)
    S = 8.0  # power scale, pp
    F = 3.0  # form scale, pp
    V = 6.0  # volatility scale, pp

    power = clamp100(50.0 + (mu_pp / S) * 50.0)
    form = clamp100(50.0 + (delta_pp / F) * 50.0)
    volatility = clamp100((sigma_pp / V) * 100.0)
    rating = clamp100(0.60 * power + 0.25 * form + 0.15 * (100.0 - volatility))
    reliability = clamp100((n / 20.0) * 100.0)

    return {
        "n": n,
        "mu_pp": mu_pp,
        "delta_pp": delta_pp,
        "sigma_pp": sigma_pp,
        "power": power,
        "form": form,
        "volatility": volatility,
        "rating": rating,
        "reliability": reliability,
        "values": vals,
    }

def _compact_summary_from(
    *,
    mods: List[ModuleResult],
    indices: Dict[str, Any],
    signals: Dict[str, Any],
    forecast: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns compact summary:
    - p_home/p_away: heuristic win probabilities from [-] indices + module votes
    - conf: 0..1 (distance from 50/50)
    - top: top 2 contributors among indices + modules
    """
    comp = (forecast.get("components") or {}) if isinstance(forecast, dict) else {}
    mod_score = float(comp.get("mod_score") or 0.0)
    mod_norm = float(comp.get("mod_norm") or 0.0)
    active_mods = int(comp.get("active_mods") or 0)

    def idiff(key: str) -> Optional[float]:
        v = (indices.get(key) or {}).get("diff")
        return float(v) if isinstance(v, (int, float)) else None

    diffs = {
        "dom": idiff("dominance_index"),
        "serve": idiff("serve_index"),
        "return": idiff("return_index"),
        "clutch": idiff("clutch_index"),
        "form": idiff("history_form_index"),
    }
    # Weights: serve slightly higher, clutch lower. Return often noisy (missing RBR), keep moderate.
    w = {"dom": 1.00, "serve": 1.15, "return": 0.95, "clutch": 0.70, "form": 0.60}
    idx_score = 0.0
    idx_wsum = 0.0
    for k, d in diffs.items():
        if d is None:
            continue
        idx_score += w[k] * float(d)
        idx_wsum += w[k]
    idx_norm = (idx_score / idx_wsum) if idx_wsum > 0 else 0.0
    idx_norm = _clamp(idx_norm * 2.0, -1.0, 1.0)  # diffs are small; stretch

    p_home = float(forecast.get("p_home") or 0.5) if isinstance(forecast, dict) else 0.5
    p_away = float(forecast.get("p_away") or (1.0 - p_home)) if isinstance(forecast, dict) else (1.0 - p_home)
    conf = float(forecast.get("confidence") or 0.0) if isinstance(forecast, dict) else 0.0
    pick = str(forecast.get("pick") or "neutral") if isinstance(forecast, dict) else "neutral"

    contrib: List[Tuple[str, float]] = []
    for k, d in diffs.items():
        if d is None:
            continue
        contrib.append((f"idx.{k}", float(d)))
    contrib.append(("mods", mod_norm))
    for sig_k in ("strength", "form", "stability"):
        d = (signals.get(sig_k) or {}).get("diff") if isinstance(signals.get(sig_k), dict) else None
        if isinstance(d, (int, float)):
            contrib.append((f"sig.{sig_k}", float(d)))
    contrib.sort(key=lambda x: abs(x[1]), reverse=True)
    top = [{"key": k, "value": v} for k, v in contrib[:2]]

    return {
        "p_home": p_home,
        "p_away": p_away,
        "confidence": conf,
        "pick": pick,
        "mod_score": mod_score,
        "mod_norm": mod_norm,
        "idx_norm": idx_norm,
        "idx_diffs": diffs,
        "signals": signals,
        "forecast": forecast,
        "top": top,
        "active_mods": active_mods,
    }


@dataclass
class HistoryAudit:
    team_id: int
    max_history: int
    surface_filter: Optional[str]
    candidates: int
    scanned: int
    valid: int
    used: int
    over_max_history: int
    dropped_to_match_opponent: int
    excluded_by_reason: Dict[str, int]
    used_events: List[Dict[str, Any]]
    excluded_events: List[Dict[str, Any]]
    coverage: Dict[str, Dict[str, int]]
    failfast_trigger: Optional[str]
    dom_timeouts: int
    spent_s: float
    budget_hit: bool
    context_resets: int
    candidate_discovery_ms: float = 0.0
    candidate_prefilter_ms: float = 0.0
    dom_extract_ms_total: float = 0.0
    dom_extract_ms_per_event_p50: float = 0.0
    dom_extract_ms_per_event_p95: float = 0.0
    wasted_dom_attempts: int = 0
    budget_extended: bool = False
    budget_extended_by_s: float = 0.0
    transient_start_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "max_history": self.max_history,
            "surface_filter": self.surface_filter,
            "candidates": self.candidates,
            "scanned": self.scanned,
            "valid": self.valid,
            "used": self.used,
            "over_max_history": self.over_max_history,
            "dropped_to_match_opponent": self.dropped_to_match_opponent,
            "excluded_by_reason": dict(self.excluded_by_reason),
            "used_events": list(self.used_events),
            "excluded_events": list(self.excluded_events),
            "coverage": dict(self.coverage),
            "failfast_trigger": self.failfast_trigger,
            "dom_timeouts": self.dom_timeouts,
            "spent_s": self.spent_s,
            "budget_hit": self.budget_hit,
            "context_resets": self.context_resets,
            "candidate_discovery_ms": self.candidate_discovery_ms,
            "candidate_prefilter_ms": self.candidate_prefilter_ms,
            "dom_extract_ms_total": self.dom_extract_ms_total,
            "dom_extract_ms_per_event_p50": self.dom_extract_ms_per_event_p50,
            "dom_extract_ms_per_event_p95": self.dom_extract_ms_per_event_p95,
            "wasted_dom_attempts": self.wasted_dom_attempts,
            "budget_extended": self.budget_extended,
            "budget_extended_by_s": self.budget_extended_by_s,
            "transient_start_failures": self.transient_start_failures,
        }


def _set2_winner_from_event_payload(event_payload: Dict[str, Any]) -> str:
    e = event_payload.get("event") or {}
    hs = e.get("homeScore") or {}
    aw = e.get("awayScore") or {}
    s2h = hs.get("period2")
    s2a = aw.get("period2")
    if s2h is None or s2a is None:
        return "neutral"
    if s2h > s2a:
        return "home"
    if s2a > s2h:
        return "away"
    return "neutral"


def _points_1st2nd_from_stats(stats_json: Dict[str, Any]) -> Optional[DominanceLivePoints]:
    # Prefer explicit Points group (counts).
    spw_home, spw_away = sum_event_value(
        stats_json, periods=("1ST", "2ND"), group_name="Points", item_name="Service points won"
    )
    rpw_home, rpw_away = sum_event_value(
        stats_json, periods=("1ST", "2ND"), group_name="Points", item_name="Receiver points won"
    )
    if (spw_home + spw_away + rpw_home + rpw_away) > 0:
        return DominanceLivePoints(spw_home=spw_home, rpw_home=rpw_home, spw_away=spw_away, rpw_away=rpw_away)

    # Fallback: derive counts from 1st/2nd serve points ratios (won/total) which are often present even when
    # Points group is missing. This still respects the “no ALL” principle if the caller provided a pre-decider
    # snapshot (ALL == 1ST+2ND).
    snap = MatchSnapshot(event_id=0, stats=stats_json)

    fs_h, fs_a = snap.ratio(periods=("1ST", "2ND"), group="Service", item="First serve points")
    ss_h, ss_a = snap.ratio(periods=("1ST", "2ND"), group="Service", item="Second serve points")
    r1_h, r1_a = snap.ratio(periods=("1ST", "2ND"), group="Return", item="First serve return points")
    r2_h, r2_a = snap.ratio(periods=("1ST", "2ND"), group="Return", item="Second serve return points")

    spw_home = (fs_h.won if fs_h else 0) + (ss_h.won if ss_h else 0)
    spw_away = (fs_a.won if fs_a else 0) + (ss_a.won if ss_a else 0)
    rpw_home = (r1_h.won if r1_h else 0) + (r2_h.won if r2_h else 0)
    rpw_away = (r1_a.won if r1_a else 0) + (r2_a.won if r2_a else 0)

    if (spw_home + spw_away + rpw_home + rpw_away) == 0:
        return None
    return DominanceLivePoints(spw_home=spw_home, rpw_home=rpw_home, spw_away=spw_away, rpw_away=rpw_away)


async def _history_rows_for_player_audit(
    page: Page,
    *,
    match_url: Optional[str] = None,
    match_event_id: Optional[int] = None,
    match_side: str = "",
    team_id: int,
    team_slug: Optional[str] = None,
    max_history: int,
    surface_filter: Optional[str] = None,
    slow_mode: bool = False,
    progress_cb: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    progress_label: str = "",
) -> Tuple[List[Dict[str, Any]], HistoryAudit]:
    side_started_at = asyncio.get_running_loop().time()
    # Keep bounded defaults for stable unattended runs.
    # Unlimited mode is opt-in via THIRDSET_NO_LIMITS=1.
    no_limits_enabled = os.getenv("THIRDSET_NO_LIMITS", "0").strip().lower() not in ("0", "false", "no")
    failfast_enabled = (not no_limits_enabled) and (
        os.getenv("THIRDSET_HISTORY_FAILFAST", "1").strip().lower() not in ("0", "false", "no")
    )
    reset_codes_raw = os.getenv(
        "THIRDSET_HISTORY_CONTEXT_RESET_ON_CODES",
        "varnish_503,scrape_eval_timeout,statistics_tab_unreachable,period_select_failed",
    )
    reset_codes = {x.strip().lower() for x in str(reset_codes_raw).split(",") if x.strip()}
    try:
        reset_threshold = int(os.getenv("THIRDSET_DOM_CONTEXT_RESET_THRESHOLD") or "3")
    except Exception:
        reset_threshold = 3
    reset_threshold = max(1, min(12, reset_threshold))
    tab_unreachable_failfast_hits = _tab_unreachable_failfast_hits()
    timeout_failfast_hits = _timeout_failfast_hits()
    # Hard limits to keep analysis fast on servers.
    # If we can't collect enough rows within budget, we return what we have (no fake filling).
    raw_per_event = os.getenv("THIRDSET_HISTORY_EVENT_TIMEOUT_S")
    raw_per_event_hist = os.getenv("THIRDSET_HISTORY_EVENT_TIMEOUT_S_HISTORY_ONLY")
    default_per_event = "22" if (slow_mode and failfast_enabled) else "18"
    selected_per_event = raw_per_event
    if slow_mode and raw_per_event_hist not in (None, ""):
        selected_per_event = raw_per_event_hist
    elif slow_mode and raw_per_event in (None, ""):
        selected_per_event = default_per_event
    try:
        per_event_timeout_s = float(selected_per_event or default_per_event)
    except Exception:
        per_event_timeout_s = float(default_per_event)
    if no_limits_enabled and slow_mode:
        per_event_timeout_s = max(30.0, per_event_timeout_s)
    else:
        per_event_timeout_s = max(8.0, min(45.0, per_event_timeout_s))
    if slow_mode and failfast_enabled and raw_per_event in (None, ""):
        per_event_timeout_s = min(per_event_timeout_s, 22.0)
    elif slow_mode and raw_per_event in (None, ""):
        # In history-only mode we process many candidate events; too-large per-event timeout
        # causes global analyze timeout (TG default 240s). Keep default tighter unless user overrides.
        per_event_timeout_s = min(per_event_timeout_s, 18.0)
    if os.getenv("THIRDSET_SLOW_LOAD") in ("1", "true", "yes") and not (slow_mode and failfast_enabled):
        per_event_timeout_s = max(per_event_timeout_s, 22.0)

    def _nav_timeout_ms(total_s: float) -> int:
        # Keep navigation under the per-event budget to avoid hard timeouts.
        ms = int(float(total_s) * 1000.0 * 0.6)
        return max(8_000, min(20_000, ms))

    try:
        wait_stats_base_ms = int(
            os.getenv("THIRDSET_HISTORY_WAIT_STATS_MS") or ("1800" if slow_mode else "0")
        )
    except Exception:
        wait_stats_base_ms = 1800 if slow_mode else 0
    try:
        wait_stats_fallback_ms = int(os.getenv("THIRDSET_HISTORY_WAIT_STATS_MS_FALLBACK") or "3500")
    except Exception:
        wait_stats_fallback_ms = 3500
    wait_stats_base_ms = max(0, min(20_000, wait_stats_base_ms))
    wait_stats_fallback_ms = max(wait_stats_base_ms, min(25_000, wait_stats_fallback_ms))
    wait_stats_ms: Optional[int] = wait_stats_base_ms if slow_mode else None

    raw_player_budget = os.getenv("THIRDSET_HISTORY_PLAYER_BUDGET_S")
    raw_player_budget_hist = os.getenv("THIRDSET_HISTORY_PLAYER_BUDGET_S_HISTORY_ONLY")
    default_player_budget = "90" if (slow_mode and failfast_enabled) else "75"
    selected_player_budget = raw_player_budget
    if slow_mode and raw_player_budget_hist not in (None, ""):
        selected_player_budget = raw_player_budget_hist
    elif slow_mode and raw_player_budget in (None, ""):
        selected_player_budget = default_player_budget
    try:
        player_budget_s = float(selected_player_budget or default_player_budget)
    except Exception:
        player_budget_s = float(default_player_budget)
    if no_limits_enabled and slow_mode:
        # Keep "no limits" practically unbounded, but finite for logging/int conversions.
        player_budget_s = 10_000_000.0
    else:
        player_budget_s = max(20.0, min(180.0, player_budget_s))
    if (not no_limits_enabled) and slow_mode and raw_player_budget in (None, ""):
        # Keep a generous default budget for strict 5/5 history collection.
        player_budget_s = min(player_budget_s, 180.0)
    loop = asyncio.get_running_loop()
    side_started_at = loop.time()
    side_deadline = side_started_at + player_budget_s
    # Candidate discovery must not consume the whole side budget.
    if no_limits_enabled and slow_mode:
        candidate_build_budget_s = player_budget_s
    else:
        candidate_build_budget_s = min(12.0, max(6.0, player_budget_s * 0.30)) if (slow_mode and failfast_enabled) else min(30.0, max(8.0, player_budget_s * 0.35))
    candidate_deadline = min(side_deadline, side_started_at + candidate_build_budget_s)
    try:
        candidate_mult = int(os.getenv("THIRDSET_HISTORY_CANDIDATE_MULT") or "12")
    except Exception:
        candidate_mult = 12
    candidate_mult = max(6, min(40, candidate_mult))
    try:
        scan_initial_cfg = int(os.getenv("THIRDSET_HISTORY_SCAN_INITIAL") or "12")
    except Exception:
        scan_initial_cfg = 12
    try:
        scan_chunk_cfg = int(os.getenv("THIRDSET_HISTORY_SCAN_CHUNK") or "6")
    except Exception:
        scan_chunk_cfg = 6
    try:
        scan_cap_balanced_cfg = int(os.getenv("THIRDSET_HISTORY_SCAN_CAP_BALANCED") or "36")
    except Exception:
        scan_cap_balanced_cfg = 36
    scan_initial_cfg = max(6, min(40, scan_initial_cfg))
    scan_chunk_cfg = max(2, min(20, scan_chunk_cfg))
    scan_cap_balanced_cfg = max(scan_initial_cfg, min(120, scan_cap_balanced_cfg))
    require_period12 = os.getenv("THIRDSET_HISTORY_PREFILTER_REQUIRE_PERIOD12", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )
    if slow_mode:
        # Keep history-only traversal deterministic and simple.
        candidate_target = min(24, max(12, max_history * 4 + 4))
    elif no_limits_enabled:
        candidate_target = max(max_history * candidate_mult, 1_000_000)
    else:
        candidate_target = max(max_history * 4 + 4, scan_initial_cfg)
        candidate_target = min(candidate_target, scan_cap_balanced_cfg)
        candidate_target = max(12, candidate_target)

    def _candidate_timeout_s(default_s: float = 4.0) -> float:
        # Keep all candidate-discovery operations under candidate budget.
        if no_limits_enabled and slow_mode:
            return max(30.0, float(default_s))
        left = max(1.5, candidate_deadline - loop.time())
        return max(1.5, min(float(default_s), left))

    # We only need `max_history` valid rows. For fullness we always parse history via DOM,
    # so we fetch a larger candidate list and stop early once we collected enough rows.
    # HISTORY EVENTS SOURCE (DOM-first):
    # We deliberately avoid /api/v1/team/<id>/events/last/* because it's often blocked (403) on servers.
    # Instead, we open player's page and take match links from DOM, then resolve each match to its event payload.
    candidate_discovery_started = loop.time()
    candidate_prefilter_ms = 0.0
    candidate_discovery_ms = 0.0
    events: List[Dict[str, Any]] = []
    seen_event_ids: set[int] = set()
    seen_candidate_keys: set[str] = set()
    links: List[str] = []
    excluded_by_reason: Dict[str, int] = {}
    if no_limits_enabled and slow_mode:
        link_limit = max(max_history * 80, 400)
    else:
        link_limit = max(120, min(2000, candidate_target * 8))

    def _candidate_prefilter_reason(ev: Dict[str, Any]) -> Optional[str]:
        nonlocal candidate_prefilter_ms
        t0 = loop.time()
        try:
            status = str(((ev.get("status") or {}).get("type") or "")).strip().lower()
            if status != "finished":
                return "not_finished"
            if not is_singles_event(ev):
                return "not_singles"
            if not require_period12:
                return None
            hs = ev.get("homeScore") or {}
            aw = ev.get("awayScore") or {}
            p12 = (hs.get("period1"), aw.get("period1"), hs.get("period2"), aw.get("period2"))
            status_desc = str(((ev.get("status") or {}).get("description") or "")).strip().lower()
            if any(x in status_desc for x in ("walkover", "w/o", " wo ", "retired", "ret.", "abandoned")):
                return "stats_absent_likely"
            if any(v is None for v in p12):
                return "stats_absent_likely"
            return None
        finally:
            candidate_prefilter_ms += max(0.0, (loop.time() - t0) * 1000.0)

    def _accept_candidate_event(ev: Dict[str, Any]) -> bool:
        if not isinstance(ev, dict):
            return False
        reason = _candidate_prefilter_reason(ev)
        if reason is not None:
            if reason == "stats_absent_likely":
                excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
            else:
                return False
        ev_id = ev.get("id")
        if not isinstance(ev_id, int):
            return False
        if ev_id in seen_event_ids:
            return False
        seen_event_ids.add(ev_id)
        events.append(ev)
        return True

    def _accept_candidate_link(link_s: str, *, source: str) -> bool:
        if not isinstance(link_s, str) or not link_s:
            return False
        eid = parse_event_id_from_match_link(link_s)
        key = f"id:{eid}" if isinstance(eid, int) else f"link:{link_s.split('#', 1)[0]}"
        if key in seen_candidate_keys:
            return False
        seen_candidate_keys.add(key)
        if isinstance(eid, int):
            seen_event_ids.add(eid)
        events.append(
            {
                "id": int(eid) if isinstance(eid, int) else None,
                "_candidate_link": link_s,
                "_candidate_key": key,
                "_candidate_source": source,
            }
        )
        return True

    async def _resolve_event_from_link(link_s: str, *, resolve_page: Optional[Page] = None) -> Optional[Dict[str, Any]]:
        if not isinstance(link_s, str) or not link_s:
            return None
        p = resolve_page or page
        eid = parse_event_id_from_match_link(link_s)
        nav_payload_t = 120.0 if (no_limits_enabled and slow_mode) else (6.0 if (slow_mode and failfast_enabled) else 12.0)
        if isinstance(eid, int):
            if no_limits_enabled and slow_mode:
                payload = await get_event_via_navigation(p, int(eid), timeout_ms=120_000)
            else:
                payload = await asyncio.wait_for(
                    get_event_via_navigation(p, int(eid), timeout_ms=int(nav_payload_t * 1000.0)),
                    timeout=nav_payload_t,
                )
        else:
            if no_limits_enabled and slow_mode:
                payload = await get_event_from_match_url_auto(p, link_s, timeout_ms=120_000)
            else:
                payload = await asyncio.wait_for(get_event_from_match_url_auto(p, link_s), timeout=nav_payload_t)
        ev = payload.get("event") if isinstance(payload, dict) else None
        return ev if isinstance(ev, dict) else None

    def _event_needs_resolution(ev: Dict[str, Any]) -> bool:
        """
        Link candidates in slow mode are intentionally lightweight.
        Resolve them lazily only when the event is actually scanned.
        """
        if not isinstance(ev, dict):
            return True
        if ev.get("_candidate_link"):
            return True
        if not isinstance(ev.get("id"), int):
            return True
        status = ev.get("status")
        if not isinstance(status, dict) or not status:
            return True
        if not isinstance(ev.get("homeTeam"), dict) or not isinstance(ev.get("awayTeam"), dict):
            return True
        if not isinstance(ev.get("homeScore"), dict) or not isinstance(ev.get("awayScore"), dict):
            return True
        return False

    resolved_event_by_key: Dict[str, Optional[Dict[str, Any]]] = {}
    # 1) Try the player profile URL derived from the match page DOM (most reliable; avoids 404).
    if match_url and match_event_id and match_side in ("home", "away"):
        try:
            if no_limits_enabled and slow_mode:
                prof = await discover_player_profile_urls_from_match(
                    page, match_url=str(match_url), event_id=int(match_event_id)
                )
            else:
                prof = await asyncio.wait_for(
                    discover_player_profile_urls_from_match(page, match_url=str(match_url), event_id=int(match_event_id)),
                    timeout=_candidate_timeout_s(3.0),
                )
        except Exception:
            prof = {}
        profile_url = prof.get(match_side) if isinstance(prof, dict) else None
        _log_step(
            f"History[{progress_label or match_side}]: profile_url={'yes' if isinstance(profile_url, str) and profile_url else 'no'}"
        )
        if isinstance(profile_url, str) and profile_url:
            try:
                # Player pages may contain many upcoming/live matches first; we need enough depth to reach finished ones.
                if no_limits_enabled and slow_mode:
                    links = await discover_player_match_links_from_profile_url(
                        page, profile_url=profile_url, limit=link_limit
                    )
                else:
                    links = await asyncio.wait_for(
                        discover_player_match_links_from_profile_url(
                            page, profile_url=profile_url, limit=link_limit
                        ),
                        timeout=_candidate_timeout_s(4.0),
                    )
            except Exception:
                links = []
    _log_step(f"History[{progress_label or match_side}]: links={len(links)}")
    if not links:
        excluded_by_reason["profile_links_empty"] = 1
    # 2) Fallback within DOM-only approach: try constructing the profile URL from slug/id (can 404).
    if not links and isinstance(team_slug, str) and team_slug.strip():
        try:
            if no_limits_enabled and slow_mode:
                links = await discover_player_match_links(
                    page, team_id=int(team_id), team_slug=str(team_slug), limit=link_limit
                )
            else:
                links = await asyncio.wait_for(
                    discover_player_match_links(
                        page, team_id=int(team_id), team_slug=str(team_slug), limit=link_limit
                    ),
                    timeout=_candidate_timeout_s(4.0),
                )
        except Exception:
            links = []
        _log_step(f"History[{progress_label or match_side}]: links_fallback={len(links)}")
    if links:
        for link in links:
            if asyncio.get_running_loop().time() >= candidate_deadline:
                break
            if len(events) >= candidate_target:
                break
            try:
                if slow_mode:
                    _accept_candidate_link(str(link), source="profile_links")
                else:
                    ev = await _resolve_event_from_link(str(link))
                    if ev is None:
                        continue
                    _accept_candidate_event(ev)
            except Exception:
                continue
    # 3) Augment with API team-last history if DOM links are too shallow.
    # This does not replace DOM stats extraction; it only broadens candidate event ids.
    if len(events) < max(max_history * 2, scan_initial_cfg):
        api_fetch_ok = True
        try:
            api_needed = max(candidate_target - len(events), max_history * 6)
            if no_limits_enabled and slow_mode:
                api_timeout = 120.0
            elif slow_mode and failfast_enabled:
                # Preserve enough time budget for at least one DOM stats attempt.
                remain = max(1.5, side_deadline - loop.time() - per_event_timeout_s)
                api_timeout = min(7.0, max(2.5, remain))
            else:
                api_timeout = min(18.0, max(8.0, candidate_build_budget_s))
            if no_limits_enabled and slow_mode:
                api_events = await get_last_finished_singles_events(page, int(team_id), limit=min(200, int(api_needed)))
            else:
                api_events = await asyncio.wait_for(
                    get_last_finished_singles_events(page, int(team_id), limit=min(200, int(api_needed))),
                    timeout=api_timeout,
                )
        except Exception:
            api_fetch_ok = False
            api_events = []
        if not api_fetch_ok:
            excluded_by_reason["api_history_fetch_failed"] = excluded_by_reason.get("api_history_fetch_failed", 0) + 1
        for ev in api_events or []:
            if not isinstance(ev, dict):
                continue
            ev_id = ev.get("id")
            if not isinstance(ev_id, int):
                continue
            if ev_id in seen_event_ids:
                continue
            if not _accept_candidate_event(ev):
                continue
            if len(events) >= candidate_target:
                break
    candidate_discovery_ms = max(0.0, (loop.time() - candidate_discovery_started) * 1000.0)
    _log_step(f"History[{progress_label or match_side}]: candidate_events={len(events)} target={candidate_target}")
    # Start DOM-scrape budget only after we built candidate events.
    deadline = side_deadline
    max_total_budget_s = min(120.0, max(20.0, float(player_budget_s)) + 40.0)
    max_deadline = side_started_at + max_total_budget_s
    budget_extended = False
    budget_extended_by_s = 0.0
    transient_start_failures = 0

    # No API fallbacks for history candidates: source of truth is player page DOM.
    if surface_filter:
        sf = normalize_surface(surface_filter)
    else:
        sf = None
    # DOM parsing opens match pages. Extra parallel tabs increase CAPTCHA/CF challenges,
    # so default to sequential scraping unless explicitly overridden.
    try:
        workers = int(os.getenv("THIRDSET_HISTORY_WORKERS", "1"))
    except Exception:
        workers = 1
    workers = max(1, min(4, workers))
    hist_pages: List[Page] = []
    try:
        if workers > 1:
            for _ in range(workers):
                hist_pages.append(await page.context.new_page())
    except Exception:
        hist_pages = []
    if not hist_pages:
        workers = 1
    single_hist_page: Page = page
    context_resets = 0
    consecutive_step_failures = 0
    spawned_pages: List[Page] = []

    def _extract_dom_code(dom_err: str) -> Optional[str]:
        try:
            m = re.search(r"\bcode=([a-z0-9_]+)\b", str(dom_err or "").lower())
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    async def _reset_single_page(reason_code: str) -> None:
        nonlocal single_hist_page, context_resets
        if workers != 1:
            return
        try:
            new_page = await page.context.new_page()
        except Exception:
            return
        old = single_hist_page
        single_hist_page = new_page
        spawned_pages.append(new_page)
        context_resets += 1
        _log_step(f"History[{progress_label or match_side}]: context_reset code={reason_code} n={context_resets}")
        if old is not page:
            try:
                await old.close()
            except Exception:
                pass

    async def one(idx: int, ev: Dict[str, Any], *, hist_page: Optional[Page]) -> Tuple[int, Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        # If we didn't create extra pages (workers=1), use the main page.
        if hist_page is None:
            hist_page = page
        cur_ev = ev
        if _event_needs_resolution(cur_ev):
            link_s = str(cur_ev.get("_candidate_link") or "").strip()
            raw_eid = cur_ev.get("id")
            cache_key = str(cur_ev.get("_candidate_key") or (f"id:{raw_eid}" if isinstance(raw_eid, int) else f"link:{link_s}"))
            cached = resolved_event_by_key.get(cache_key)
            if cached is None and cache_key not in resolved_event_by_key:
                try:
                    if link_s:
                        cached = await _resolve_event_from_link(link_s, resolve_page=hist_page)
                    elif isinstance(raw_eid, int):
                        base = _sofascore_base_from_url(hist_page.url or SOFASCORE_TENNIS_URL)
                        fallback_link = f"{base}/event/{int(raw_eid)}#id:{int(raw_eid)}"
                        cached = await _resolve_event_from_link(fallback_link, resolve_page=hist_page)
                except Exception:
                    cached = None
                resolved_event_by_key[cache_key] = cached
            elif cache_key in resolved_event_by_key:
                cached = resolved_event_by_key.get(cache_key)
            if isinstance(cached, dict):
                merged = dict(cached)
                for k in ("_candidate_link", "_candidate_key", "_candidate_source"):
                    if k in cur_ev and k not in merged:
                        merged[k] = cur_ev.get(k)
                cur_ev = merged

        eid = cur_ev.get("id")
        if not eid:
            return idx, None, "missing_event_id", {}
        summary = summarize_event_for_team(cur_ev, team_id=team_id)
        if summary.get("won") is None:
            return idx, None, "winner_unknown", summary
        # Determine if match is BO3 decider via score fields.
        hs = cur_ev.get("homeScore") or {}
        aw = cur_ev.get("awayScore") or {}
        has_set3 = hs.get("period3") is not None and aw.get("period3") is not None
        if slow_mode or os.getenv("THIRDSET_HISTORY_SKIP_SET3") in ("1", "true", "yes"):
            has_set3 = False
        s1h = hs.get("period1")
        s1a = aw.get("period1")
        s2h = hs.get("period2")
        s2a = aw.get("period2")
        s3h = hs.get("period3")
        s3a = aw.get("period3")

        is_home = bool(summary.get("isHome"))
        won_set1 = None
        if s1h is not None and s1a is not None:
            won_set1 = (s1h > s1a) if is_home else (s1a > s1h)
        # Determine if player won set2 (only meaningful if set2 exists).
        won_set2 = None
        if s2h is not None and s2a is not None:
            won_set2 = (s2h > s2a) if is_home else (s2a > s2h)
        won_set3 = None
        if s3h is not None and s3a is not None:
            won_set3 = (s3h > s3a) if is_home else (s3a > s3h)

        surface = normalize_surface(cur_ev.get("groundType"))
        if sf and surface != sf:
            return idx, None, "surface_mismatch", summary

        # History statistics: STRICT DOM-only.
        # User requirement: maximum completeness and CF stability; no /api/v1/... fetches for history stats.
        stats = None
        dom_used = False
        dom_attempted = False
        dom_extract_started = loop.time()
        dom_extract_ms = 0.0
        dom_err: Optional[str] = None
        if hist_page is not None:
            dom_attempted = True
            try:
                slug = cur_ev.get("slug")
                custom = cur_ev.get("customId")
                base = _sofascore_base_from_url(hist_page.url or SOFASCORE_TENNIS_URL)
                if slug and custom:
                    url = f"{base}/tennis/match/{slug}/{custom}"
                else:
                    url = f"{base}/event/{int(eid)}"
                periods = ("1ST", "2ND", "3RD") if has_set3 else ("1ST", "2ND")
                try:
                    _log_step(f"History[{progress_label or match_side}]: stats eid={eid} url={url}")
                    if slow_mode:
                        stats = await extract_statistics_dom(
                            hist_page,
                            match_url=url,
                            event_id=int(eid),
                            periods=periods,
                            nav_timeout_ms=_nav_timeout_ms(max(45.0, per_event_timeout_s)),
                            wait_stats_ms=wait_stats_ms,
                        )
                    else:
                        stats = await asyncio.wait_for(
                            extract_statistics_dom(
                                hist_page,
                                match_url=url,
                                event_id=int(eid),
                                periods=periods,
                                nav_timeout_ms=_nav_timeout_ms(per_event_timeout_s),
                                wait_stats_ms=wait_stats_ms,
                            ),
                            timeout=per_event_timeout_s,
                        )
                except asyncio.TimeoutError:
                    timeout_msg = (
                        f"step=extract_statistics code=dom_extract_timeout "
                        f"TimeoutError: exceeded {per_event_timeout_s:.1f}s"
                    )
                    # Long retries on one event can stall the whole side pass.
                    # Keep retries short in history-only mode to continue scanning.
                    if slow_mode and failfast_enabled:
                        raise DomStatsError(
                            timeout_msg,
                            step="extract_statistics",
                            code="dom_extract_timeout",
                            diag={"timeout_s": float(per_event_timeout_s)},
                        )
                    if slow_mode:
                        retry_t = min(18.0, max(10.0, per_event_timeout_s * 0.8))
                    else:
                        retry_t = min(30.0, max(per_event_timeout_s * 1.5, per_event_timeout_s + 6.0))
                    _log_step(f"History[{progress_label or match_side}]: retry stats eid={eid} t={retry_t}")
                    try:
                        stats = await asyncio.wait_for(
                            extract_statistics_dom(
                                hist_page,
                                match_url=url,
                                event_id=int(eid),
                                periods=periods,
                                nav_timeout_ms=_nav_timeout_ms(retry_t),
                                wait_stats_ms=wait_stats_ms,
                            ),
                            timeout=retry_t,
                        )
                    except asyncio.TimeoutError as rex:
                        raise DomStatsError(
                            f"step=extract_statistics code=dom_extract_timeout TimeoutError: exceeded {retry_t:.1f}s",
                            step="extract_statistics",
                            code="dom_extract_timeout",
                            diag={"timeout_s": float(retry_t)},
                        ) from rex
                except Exception as ex:
                    # Sofascore can trigger a locale redirect / internal navigation right after goto(),
                    # which occasionally destroys the JS execution context during DOM parsing.
                    # Retry once; this improves history coverage on large tournaments.
                    msg = str(ex)
                    if "Execution context was destroyed" in msg or "most likely because of a navigation" in msg:
                        await asyncio.sleep(0.35)
                        _log_step(f"History[{progress_label or match_side}]: retry nav ctx eid={eid}")
                        retry_nav_t = min(45.0, max(per_event_timeout_s * 1.5, per_event_timeout_s + 5.0))
                        try:
                            stats = await asyncio.wait_for(
                                extract_statistics_dom(
                                    hist_page,
                                    match_url=url,
                                    event_id=int(eid),
                                    periods=periods,
                                    nav_timeout_ms=_nav_timeout_ms(per_event_timeout_s + 5.0),
                                    wait_stats_ms=wait_stats_ms,
                                ),
                                timeout=retry_nav_t,
                            )
                        except asyncio.TimeoutError as rex:
                            raise DomStatsError(
                                f"step=extract_statistics code=dom_extract_timeout TimeoutError: exceeded {retry_nav_t:.1f}s",
                                step="extract_statistics",
                                code="dom_extract_timeout",
                                diag={"timeout_s": float(retry_nav_t)},
                            ) from rex
                    else:
                        raise
                dom_used = True
                dom_err = None
            except Exception as ex:
                dom_err = f"{type(ex).__name__}: {ex}"
                stats = None
            finally:
                dom_extract_ms = max(0.0, (loop.time() - dom_extract_started) * 1000.0)

        if stats is None:
            # Strict history: if we can't read per-set stats, skip and scan more candidates.
            if dom_err:
                summary = dict(summary)
                summary["dom_error"] = dom_err
                _log_step(f"History[{progress_label or match_side}]: dom_stats_error eid={eid} err={dom_err}")
            if isinstance(summary, dict):
                summary["_dom_attempted"] = bool(dom_attempted)
                summary["_dom_extract_ms"] = float(dom_extract_ms)
                summary["_dom_wasted"] = bool(dom_attempted)
            return idx, None, "dom_stats_error", summary

        snap = MatchSnapshot(event_id=int(eid), stats=stats) if stats is not None else None

        # Whole-match TPW for CloseWinRate
        # Prefer sum across sets (1ST/2ND/3RD) to avoid relying on ALL.
        pts_all = None
        if snap is not None:
            pts_all = snap.points_by_periods(("1ST", "2ND", "3RD"))

        tpw_player = None
        if pts_all is not None:
            tp_home = pts_all.tp_home
            tp_away = pts_all.tp_away
            if tp_home + tp_away > 0:
                tpw_home = tp_home / (tp_home + tp_away)
                tpw_player = tpw_home if summary.get("isHome") else (1 - tpw_home)

        # TPW only for sets 1+2 (tpw12) — main history base for Power/Form/Volatility.
        tpw12_player = None
        if snap is not None:
            pts_12 = snap.points_by_periods(("1ST", "2ND"))
            if pts_12 is not None:
                tp_home = pts_12.tp_home
                tp_away = pts_12.tp_away
                if tp_home + tp_away > 0:
                    tpw_home = tp_home / (tp_home + tp_away)
                    tpw12_player = tpw_home if summary.get("isHome") else (1 - tpw_home)

        # Per-set metrics for calibration (SSW/RPR/BPSR/BPconv)
        def ssw(period: str) -> Optional[float]:
            if snap is None:
                return None
            h, a = snap.ratio(periods=(period,), group="Service", item="Second serve points")
            if h is None or a is None:
                return None
            r = h.rate if summary.get("isHome") else a.rate
            return float(r) if r is not None else None

        def rpr(period: str) -> Optional[float]:
            if snap is None:
                return None
            r1h, r1a = snap.ratio(periods=(period,), group="Return", item="First serve return points")
            if r1h is None or r1a is None or r1h.rate is None or r1a.rate is None:
                return None
            r2h, r2a = snap.ratio(periods=(period,), group="Return", item="Second serve return points")
            r1_rate = r1h.rate if summary.get("isHome") else r1a.rate
            if r2h is None or r2a is None or r2h.rate is None or r2a.rate is None:
                return float(r1_rate)
            r2_rate = r2h.rate if summary.get("isHome") else r2a.rate
            return float(0.4 * r1_rate + 0.6 * r2_rate)

        def bpsr(period: str) -> Optional[float]:
            if snap is None:
                return None
            h, a = snap.ratio(periods=(period,), group="Service", item="Break points saved")
            if h is None or a is None:
                return None
            r0 = h if summary.get("isHome") else a
            # If no break points were faced (0/0), treat as 0.0 so the metric is not "missing".
            # This matches what Sofascore prints ("0/0 (0%)") and avoids n=0 in history.
            if r0.total <= 0:
                return 0.0
            return float(r0.won / r0.total)

        def bpconv(period: str) -> Optional[float]:
            if snap is None:
                return None
            # Converted count is in Return; denominator is opponent's BPS_total.
            bpc_h, bpc_a = snap.value(periods=(period,), group="Return", item="Break points converted")
            bps_h, bps_a = snap.ratio(periods=(period,), group="Service", item="Break points saved")
            if bps_h is None or bps_a is None:
                return None
            if summary.get("isHome"):
                denom = bps_a.total
                num = bpc_h
            else:
                denom = bps_h.total
                num = bpc_a
            if denom <= 0:
                # If there were no break points in the set, treat as 0 conversion (non-informative but present).
                return 0.0 if num == 0 else None
            if num < 0 or num > denom:
                return None
            return num / denom

        set_metrics = {
            "ssw_1": ssw("1ST"),
            "ssw_2": ssw("2ND"),
            "ssw_3": ssw("3RD") if has_set3 else None,
            "rpr_1": rpr("1ST"),
            "rpr_2": rpr("2ND"),
            "rpr_3": rpr("3RD") if has_set3 else None,
            "bpsr_1": bpsr("1ST"),
            "bpsr_2": bpsr("2ND"),
            "bpsr_3": bpsr("3RD") if has_set3 else None,
            "bpconv_1": bpconv("1ST"),
            "bpconv_2": bpconv("2ND"),
            "bpconv_3": bpconv("3RD") if has_set3 else None,
        }

        missing_all_metrics = tpw_player is None and all(v is None for v in set_metrics.values())
        if missing_all_metrics:
            # Strict history: do not count rows without any usable stats.
            if isinstance(summary, dict):
                summary["_dom_attempted"] = bool(dom_attempted)
                summary["_dom_extract_ms"] = float(dom_extract_ms)
                summary["_dom_wasted"] = bool(dom_attempted)
            return idx, None, "missing_all_metrics", summary

        row = {
            "won": bool(summary.get("won")),
            "tpw": tpw_player,
            "tpw12": tpw12_player,
            "is_bo3_decider": bool(has_set3),
            "won_set1": bool(won_set1) if won_set1 is not None else False,
            "won_set1_known": bool(won_set1 is not None),
            "won_set2": bool(won_set2) if won_set2 is not None else False,
            "won_set2_known": bool(won_set2 is not None),
            "won_set3": bool(won_set3) if won_set3 is not None else False,
            "won_set3_known": bool(won_set3 is not None),
            "surface": surface,
            "eventId": int(eid),
            "opponentName": summary.get("opponentName"),
            "tournament": summary.get("tournament"),
            "homeScore": summary.get("homeScore"),
            "awayScore": summary.get("awayScore"),
            "startTimestamp": summary.get("startTimestamp"),
            **set_metrics,
        }
        # Keep rows even if stats were missing (they still help W/L form and decider rates).
        row["stats_ok"] = bool(stats is not None)
        row["stats_dom"] = bool(dom_used)
        row["missing_all_metrics"] = bool(missing_all_metrics)
        if isinstance(summary, dict):
            summary["_dom_attempted"] = bool(dom_attempted)
            summary["_dom_extract_ms"] = float(dom_extract_ms)
            summary["_dom_wasted"] = False
        return idx, row, None, summary

    excluded_events: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    scanned = 0
    consecutive_dom_timeouts = 0
    total_dom_timeouts = 0
    statistics_tab_unreachable_hits = 0
    consecutive_statistics_tab_unreachable_hits = 0
    rows_not_ready_hits_total = 0
    rows_not_ready_hits_consecutive = 0
    stats_not_provided_hits = 0
    ui_unavailable_hits = 0
    consecutive_ui_unavailable_hits = 0
    last_ui_unavailable_code: Optional[str] = None
    dom_extract_durations_ms: List[float] = []
    wasted_dom_attempts = 0
    failfast_trigger: Optional[str] = None
    budget_hit = False
    if slow_mode:
        scan_cap_max = min(len(events), 24)
    else:
        scan_cap_max = len(events) if (no_limits_enabled and slow_mode) else min(len(events), scan_cap_balanced_cfg)
    if scan_cap_max <= 0:
        scan_cap_max = len(events)
    if slow_mode:
        scan_cap = scan_cap_max
    else:
        scan_cap = min(scan_cap_max, scan_initial_cfg)
    if scan_cap <= 0 and scan_cap_max > 0:
        scan_cap = scan_cap_max
    if scan_cap_max > 0:
        scan_cap = max(1, scan_cap)
    _log_step(
        f"History[{progress_label or match_side}]: scan_cap={scan_cap}/{scan_cap_max} "
        f"chunk={scan_chunk_cfg} budget_s={int(player_budget_s)}"
    )

    # Parallel scheduler: keep up to `workers` tasks in flight.
    tasks: set = set()
    ev_iter = iter(list(enumerate(events)))
    enforce_side_budget = (not no_limits_enabled) and (not slow_mode)

    def _next_page(i: int) -> Optional[Page]:
        if not hist_pages:
            return single_hist_page
        return hist_pages[i % len(hist_pages)]

    async def _launch_next() -> bool:
        nonlocal tasks
        # Limit total scanned events; this cap is independent from max_history.
        if (not no_limits_enabled) and (scanned + len(tasks)) >= scan_cap:
            return False
        # Stop scheduling new work if we exceeded the budget.
        if enforce_side_budget and asyncio.get_running_loop().time() >= deadline:
            return False
        try:
            i, ev = next(ev_iter)
        except StopIteration:
            return False
        t = asyncio.create_task(one(i, ev, hist_page=_next_page(i)))
        tasks.add(t)
        return True

    transient_signal_codes = {
        "dom_extract_timeout",
        "scrape_eval_timeout",
        "statistics_tab_unreachable",
        "consent_overlay_blocked",
        "navigation_failed",
        "period_select_failed",
    }
    # Prime initial tasks
    for _ in range(workers):
        if not await _launch_next():
            break

    while tasks:
        # Budget stop: cancel all in-flight tasks and return whatever we collected so far.
        if enforce_side_budget and asyncio.get_running_loop().time() >= deadline:
            for t in tasks:
                t.cancel()
            tasks.clear()
            excluded_by_reason["budget_exceeded"] = excluded_by_reason.get("budget_exceeded", 0) + 1
            budget_hit = True
            failfast_trigger = failfast_trigger or "budget_exceeded"
            break
        if scanned >= scan_cap:
            for t in tasks:
                t.cancel()
            tasks.clear()
            excluded_by_reason["scan_cap_reached"] = excluded_by_reason.get("scan_cap_reached", 0) + 1
            failfast_trigger = failfast_trigger or "scan_cap_reached"
            break
        remaining = max(0.0, deadline - asyncio.get_running_loop().time()) if enforce_side_budget else None
        done, tasks = await asyncio.wait(
            tasks,
            timeout=(max(0.1, remaining) if isinstance(remaining, float) else None),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            for t in tasks:
                t.cancel()
            tasks.clear()
            if enforce_side_budget:
                excluded_by_reason["budget_exceeded"] = excluded_by_reason.get("budget_exceeded", 0) + 1
                budget_hit = True
                failfast_trigger = failfast_trigger or "budget_exceeded_wait"
            else:
                excluded_by_reason["wait_no_result"] = excluded_by_reason.get("wait_no_result", 0) + 1
            break
        for t in done:
            stop_side_now = False
            idx, row, reason, summary = t.result()
            scanned += 1
            if isinstance(summary, dict) and summary.get("_dom_attempted"):
                try:
                    dom_ms = float(summary.get("_dom_extract_ms") or 0.0)
                except Exception:
                    dom_ms = 0.0
                if dom_ms > 0:
                    dom_extract_durations_ms.append(dom_ms)
                if bool(summary.get("_dom_wasted")):
                    wasted_dom_attempts += 1
            if reason is not None:
                if reason == "dom_stats_error":
                    derr = str((summary or {}).get("dom_error") or "") if isinstance(summary, dict) else ""
                    code = _extract_dom_code(derr)
                    if code:
                        excluded_by_reason[f"code_{code}"] = excluded_by_reason.get(f"code_{code}", 0) + 1
                    if len(valid_rows) == 0 and code in transient_signal_codes:
                        transient_start_failures += 1
                    if (
                        slow_mode
                        and isinstance(wait_stats_ms, int)
                        and wait_stats_ms < wait_stats_fallback_ms
                        and len(valid_rows) == 0
                        and scanned <= 3
                        and code in ("rows_not_ready", "period_select_failed")
                    ):
                        prev_wait = wait_stats_ms
                        wait_stats_ms = wait_stats_fallback_ms
                        _log_step(
                            f"History[{progress_label or match_side}]: wait_stats_raise "
                            f"reason={code} old={prev_wait} new={wait_stats_ms}"
                        )
                    if code == "rows_not_ready":
                        rows_not_ready_hits_total += 1
                        rows_not_ready_hits_consecutive += 1
                    else:
                        rows_not_ready_hits_consecutive = 0
                    if code in ("statistics_tab_unreachable", "consent_overlay_blocked"):
                        ui_unavailable_hits += 1
                        consecutive_ui_unavailable_hits += 1
                        statistics_tab_unreachable_hits += 1
                        consecutive_statistics_tab_unreachable_hits += 1
                        last_ui_unavailable_code = str(code)
                    else:
                        consecutive_ui_unavailable_hits = 0
                        consecutive_statistics_tab_unreachable_hits = 0
                        last_ui_unavailable_code = None
                    if code == "stats_not_provided":
                        stats_not_provided_hits += 1
                    timeout_like_codes = {"dom_extract_timeout", "scrape_eval_timeout"}
                    is_timeout_err = bool(code in timeout_like_codes) or (code is None and "TimeoutError" in derr)
                    is_varnish_503 = "503 backend read error" in derr
                    if is_timeout_err:
                        consecutive_dom_timeouts += 1
                        total_dom_timeouts += 1
                    elif not is_varnish_503:
                        consecutive_dom_timeouts = 0
                    if is_varnish_503:
                        excluded_by_reason["varnish_503"] = excluded_by_reason.get("varnish_503", 0) + 1
                    consecutive_step_failures += 1
                    if workers == 1 and (
                        (code is not None and code in reset_codes) or consecutive_step_failures >= reset_threshold
                    ):
                        await _reset_single_page(code or "consecutive_failures")
                        consecutive_step_failures = 0
                else:
                    consecutive_dom_timeouts = 0
                    consecutive_step_failures = 0
                    consecutive_ui_unavailable_hits = 0
                    consecutive_statistics_tab_unreachable_hits = 0
                    rows_not_ready_hits_consecutive = 0
                    last_ui_unavailable_code = None
                excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
                if len(excluded_events) < 12:
                    excluded_events.append(
                        {
                            "eventId": summary.get("eventId") if isinstance(summary, dict) else None,
                            "opponentName": (summary or {}).get("opponentName") if isinstance(summary, dict) else None,
                            "tournament": (summary or {}).get("tournament") if isinstance(summary, dict) else None,
                            "reason": reason,
                            "dom_error": (summary or {}).get("dom_error") if isinstance(summary, dict) else None,
                        }
                    )
                # Strict history rule requested by user:
                # if one of required 5 matches has no stats, stop this side immediately.
                if slow_mode and len(valid_rows) < max_history and reason == "dom_stats_error":
                    excluded_by_reason["missing_stats_in_required_five"] = (
                        excluded_by_reason.get("missing_stats_in_required_five", 0) + 1
                    )
                    failfast_trigger = failfast_trigger or "missing_stats_in_required_five"
                    _log_step(
                        f"History[{progress_label or match_side}]: failfast missing stats "
                        f"valid={len(valid_rows)}/{max_history} scanned={scanned}"
                    )
                    for tx in tasks:
                        tx.cancel()
                    tasks.clear()
                    stop_side_now = True
            elif row is None:
                consecutive_dom_timeouts = 0
                consecutive_step_failures = 0
                consecutive_ui_unavailable_hits = 0
                consecutive_statistics_tab_unreachable_hits = 0
                rows_not_ready_hits_consecutive = 0
                last_ui_unavailable_code = None
                excluded_by_reason["unknown_drop"] = excluded_by_reason.get("unknown_drop", 0) + 1
            else:
                consecutive_dom_timeouts = 0
                consecutive_step_failures = 0
                consecutive_ui_unavailable_hits = 0
                consecutive_statistics_tab_unreachable_hits = 0
                rows_not_ready_hits_consecutive = 0
                last_ui_unavailable_code = None
                valid_rows.append((idx, row))
                if progress_cb is not None:
                    try:
                        await progress_cb(
                            {
                                "label": progress_label,
                                "team_id": team_id,
                                "target": int(max_history),
                                "have": int(len(valid_rows)),
                                "scanned": int(scanned),
                                "eventId": row.get("eventId"),
                                "opponent": row.get("opponentName"),
                            }
                        )
                    except Exception:
                        pass
            if stop_side_now:
                break
            # Hard fail-fast on repeated UI-unavailable states (statistics tab unreachable / consent blocked).
            if ui_unavailable_hits >= tab_unreachable_failfast_hits:
                excluded_by_reason["ui_unavailable_storm"] = excluded_by_reason.get("ui_unavailable_storm", 0) + 1
                excluded_by_reason["statistics_tab_unreachable_storm"] = excluded_by_reason.get("statistics_tab_unreachable_storm", 0) + 1
                failfast_trigger = failfast_trigger or "statistics_tab_unreachable_repeated"
                _log_step(
                    f"History[{progress_label or match_side}]: failfast ui_unavailable "
                    f"hits={ui_unavailable_hits} threshold={tab_unreachable_failfast_hits} "
                    f"code={last_ui_unavailable_code or '-'}"
                )
                for tx in tasks:
                    tx.cancel()
                tasks.clear()
                break
            if consecutive_ui_unavailable_hits >= tab_unreachable_failfast_hits:
                excluded_by_reason["ui_unavailable_storm"] = excluded_by_reason.get("ui_unavailable_storm", 0) + 1
                excluded_by_reason["statistics_tab_unreachable_storm"] = excluded_by_reason.get("statistics_tab_unreachable_storm", 0) + 1
                failfast_trigger = failfast_trigger or "statistics_tab_unreachable_repeated"
                _log_step(
                    f"History[{progress_label or match_side}]: failfast consecutive ui_unavailable "
                    f"hits={consecutive_ui_unavailable_hits} threshold={tab_unreachable_failfast_hits} "
                    f"code={last_ui_unavailable_code or '-'}"
                )
                for tx in tasks:
                    tx.cancel()
                tasks.clear()
                break
            if len(valid_rows) == 0 and (
                rows_not_ready_hits_total >= 2 or rows_not_ready_hits_consecutive >= 2
            ):
                excluded_by_reason["rows_not_ready_repeated"] = excluded_by_reason.get("rows_not_ready_repeated", 0) + 1
                failfast_trigger = failfast_trigger or "rows_not_ready_repeated"
                _log_step(
                    f"History[{progress_label or match_side}]: failfast rows_not_ready "
                    f"total={rows_not_ready_hits_total} consecutive={rows_not_ready_hits_consecutive} threshold=2"
                )
                for tx in tasks:
                    tx.cancel()
                tasks.clear()
                break
            # Soft timeout fail-fast: only after representative scan depth.
            if (not no_limits_enabled) and len(valid_rows) == 0 and scanned >= 8 and consecutive_dom_timeouts >= 6:
                excluded_by_reason["dom_timeout_storm"] = excluded_by_reason.get("dom_timeout_storm", 0) + 1
                failfast_trigger = failfast_trigger or "timeout_storm_consecutive"
                _log_step(
                    f"History[{progress_label or match_side}]: failfast timeout storm "
                    f"consecutive={consecutive_dom_timeouts} threshold=6"
                )
                for tx in tasks:
                    tx.cancel()
                tasks.clear()
                break
            if stats_not_provided_hits >= tab_unreachable_failfast_hits:
                excluded_by_reason["stats_not_provided_storm"] = excluded_by_reason.get(
                    "stats_not_provided_storm", 0
                ) + 1
                failfast_trigger = failfast_trigger or "stats_not_provided_repeated"
                _log_step(
                    f"History[{progress_label or match_side}]: failfast stats_not_provided "
                    f"hits={stats_not_provided_hits} threshold={tab_unreachable_failfast_hits}"
                )
                for tx in tasks:
                    tx.cancel()
                tasks.clear()
                break

            # Launch next task only after fail-fast checks above.
            if len(valid_rows) < max_history:
                await _launch_next()
        if str(failfast_trigger or "").strip().lower() == "missing_stats_in_required_five":
            break

        if len(valid_rows) >= max_history:
            # Cancel any remaining tasks; we have enough.
            for t in tasks:
                t.cancel()
            tasks.clear()
            break

    # Second pass for DOM timeouts: retry problematic events in a fresh page context.
    if (
        0 < len(valid_rows) < max_history
        and total_dom_timeouts <= 0
        and str(failfast_trigger or "").strip().lower() != "missing_stats_in_required_five"
    ):
        retry_candidates: List[int] = []
        if isinstance(excluded_events, list):
            for ex in excluded_events:
                if not isinstance(ex, dict):
                    continue
                if ex.get("reason") != "dom_stats_error":
                    continue
                rid = ex.get("eventId")
                if isinstance(rid, int):
                    retry_candidates.append(rid)
        if retry_candidates:
            idx_by_eid: Dict[int, int] = {}
            for i, ev in enumerate(events):
                rid = ev.get("id")
                if isinstance(rid, int) and rid not in idx_by_eid:
                    idx_by_eid[rid] = i
            have_idx = {i for i, _r in valid_rows}
            try:
                retry_page = await page.context.new_page()
            except Exception:
                retry_page = None
            if retry_page is not None:
                try:
                    for rid in retry_candidates[:24]:
                        if len(valid_rows) >= max_history:
                            break
                        idx = idx_by_eid.get(int(rid))
                        if not isinstance(idx, int) or idx in have_idx:
                            continue
                        ev = events[idx]
                        try:
                            i2, row2, reason2, summary2 = await one(idx, ev, hist_page=retry_page)
                            scanned += 1
                            if isinstance(summary2, dict) and summary2.get("_dom_attempted"):
                                try:
                                    dom_ms = float(summary2.get("_dom_extract_ms") or 0.0)
                                except Exception:
                                    dom_ms = 0.0
                                if dom_ms > 0:
                                    dom_extract_durations_ms.append(dom_ms)
                                if bool(summary2.get("_dom_wasted")):
                                    wasted_dom_attempts += 1
                            if reason2 is None and row2 is not None:
                                valid_rows.append((i2, row2))
                                have_idx.add(i2)
                                if progress_cb is not None:
                                    try:
                                        await progress_cb(
                                            {
                                                "label": progress_label,
                                                "team_id": team_id,
                                                "target": int(max_history),
                                                "have": int(len(valid_rows)),
                                                "scanned": int(scanned),
                                                "eventId": row2.get("eventId"),
                                                "opponent": row2.get("opponentName"),
                                            }
                                        )
                                    except Exception:
                                        pass
                            else:
                                excluded_by_reason[reason2 or "unknown_drop"] = excluded_by_reason.get(reason2 or "unknown_drop", 0) + 1
                                if len(excluded_events) < 20:
                                    excluded_events.append(
                                        {
                                            "eventId": (summary2 or {}).get("eventId") if isinstance(summary2, dict) else rid,
                                            "opponentName": (summary2 or {}).get("opponentName") if isinstance(summary2, dict) else None,
                                            "tournament": (summary2 or {}).get("tournament") if isinstance(summary2, dict) else None,
                                            "reason": reason2 or "unknown_drop",
                                            "dom_error": (summary2 or {}).get("dom_error") if isinstance(summary2, dict) else None,
                                        }
                                    )
                        except Exception:
                            excluded_by_reason["retry_exception"] = excluded_by_reason.get("retry_exception", 0) + 1
                            continue
                finally:
                    try:
                        await retry_page.close()
                    except Exception:
                        pass

    # Sort by original index (newest first) and trim.
    valid_rows = [r for _i, r in sorted(valid_rows, key=lambda x: x[0])]
    if len(events) == 0:
        excluded_by_reason["no_finished_singles"] = excluded_by_reason.get("no_finished_singles", 0) + 1

    used_rows = valid_rows[:max_history]
    over_max_history = max(0, len(valid_rows) - len(used_rows))
    if over_max_history > 0:
        excluded_by_reason["over_max_history"] = excluded_by_reason.get("over_max_history", 0) + over_max_history

    keys = [
        "tpw",
        "ssw_1",
        "ssw_2",
        "ssw_3",
        "rpr_1",
        "rpr_2",
        "rpr_3",
        "bpsr_1",
        "bpsr_2",
        "bpsr_3",
        "bpconv_1",
        "bpconv_2",
        "bpconv_3",
    ]
    coverage: Dict[str, Dict[str, int]] = {}
    denom = len(used_rows)
    for k in keys:
        num = 0
        for r in used_rows:
            if r.get(k) is not None:
                num += 1
        coverage[k] = {"have": num, "total": denom}

    used_events: List[Dict[str, Any]] = []
    for r in used_rows:
        hs = (r.get("homeScore") or {}).get("display")
        aw = (r.get("awayScore") or {}).get("display")
        used_events.append(
            {
                "eventId": r.get("eventId"),
                "opponentName": r.get("opponentName"),
                "tournament": r.get("tournament"),
                "surface": r.get("surface"),
                "score": f"{hs}:{aw}" if hs is not None and aw is not None else None,
                "tpw": r.get("tpw"),
                "has_set3": bool(r.get("is_bo3_decider")),
                "stats_ok": bool(r.get("stats_ok")),
                "stats_dom": bool(r.get("stats_dom")),
                "missing_all_metrics": bool(r.get("missing_all_metrics")),
            }
        )

    spent_s = max(0.0, asyncio.get_running_loop().time() - side_started_at)
    dom_extract_ms_total = float(sum(dom_extract_durations_ms)) if dom_extract_durations_ms else 0.0
    dom_extract_ms_p50 = _percentile(dom_extract_durations_ms, 0.50) if dom_extract_durations_ms else 0.0
    dom_extract_ms_p95 = _percentile(dom_extract_durations_ms, 0.95) if dom_extract_durations_ms else 0.0
    audit = HistoryAudit(
        team_id=team_id,
        max_history=max_history,
        surface_filter=sf if sf else None,
        candidates=len(events),
        scanned=scanned,
        valid=len(valid_rows),
        used=len(used_rows),
        over_max_history=over_max_history,
        dropped_to_match_opponent=0,
        excluded_by_reason=excluded_by_reason,
        used_events=used_events,
        excluded_events=excluded_events,
        coverage=coverage,
        failfast_trigger=failfast_trigger,
        dom_timeouts=total_dom_timeouts,
        spent_s=spent_s,
        budget_hit=budget_hit,
        context_resets=context_resets,
        candidate_discovery_ms=float(candidate_discovery_ms),
        candidate_prefilter_ms=float(candidate_prefilter_ms),
        dom_extract_ms_total=dom_extract_ms_total,
        dom_extract_ms_per_event_p50=dom_extract_ms_p50,
        dom_extract_ms_per_event_p95=dom_extract_ms_p95,
        wasted_dom_attempts=int(wasted_dom_attempts),
        budget_extended=bool(budget_extended),
        budget_extended_by_s=float(budget_extended_by_s),
        transient_start_failures=int(transient_start_failures),
    )
    try:
        for p in hist_pages:
            await p.close()
        for p in spawned_pages:
            if p not in hist_pages:
                await p.close()
    except Exception:
        pass
    return used_rows, audit


def build_surface_calibration(
    rows: List[Dict[str, Any]], *, surface: str, min_rows: int = 8
) -> Dict[str, MetricSummary]:
    # Prefer same-surface if enough rows, else all.
    surface_norm = normalize_surface(surface)
    same = [r for r in rows if r.get("surface") == surface_norm]
    use = same if len(same) >= min_rows else rows

    return {
        "ssw_12": summarize([v for r in use for v in (r.get("ssw_1"), r.get("ssw_2"))]),
        "rpr_12": summarize([v for r in use for v in (r.get("rpr_1"), r.get("rpr_2"))]),
        "bpsr_12": summarize([v for r in use for v in (r.get("bpsr_1"), r.get("bpsr_2"))]),
        "bpconv_12": summarize([v for r in use for v in (r.get("bpconv_1"), r.get("bpconv_2"))]),
        "ssw_3": summarize([r.get("ssw_3") for r in use]),
        "rpr_3": summarize([r.get("rpr_3") for r in use]),
        "bpsr_3": summarize([r.get("bpsr_3") for r in use]),
        "bpconv_3": summarize([r.get("bpconv_3") for r in use]),
    }


def _build_m1_history_profiles(rows_home: List[Dict[str, Any]], rows_away: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    def profile(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        # CloseWinRate: among matches with 0.49 <= tpw <= 0.51
        close_w = 0
        close_n = 0
        dec_w = 0
        dec_n = 0
        for r in rows:
            tpw = r.get("tpw")
            if isinstance(tpw, (int, float)) and 0.49 <= float(tpw) <= 0.51:
                close_n += 1
                if r.get("won") is True:
                    close_w += 1
            if r.get("is_bo3_decider") is True:
                dec_n += 1
                if r.get("won") is True:
                    dec_w += 1
        return {
            "close_rate": (close_w / close_n) if close_n >= 5 else None,
            "close_n": close_n,
            "dec_rate": (dec_w / dec_n) if dec_n >= 5 else None,
            "dec_n": dec_n,
        }

    return profile(rows_home), profile(rows_away)


def ensemble(mods: List[ModuleResult]) -> Tuple[str, int, Dict[str, Any]]:
    score = 0
    votes_home = 0
    votes_away = 0
    strong_home = 0
    strong_away = 0
    active = 0
    for m in mods:
        if m.strength <= 0 or m.side == "neutral":
            continue
        active += 1
        if m.side == "home":
            score += m.strength
            votes_home += 1
            if m.strength >= 2:
                strong_home += 1
        elif m.side == "away":
            score -= m.strength
            votes_away += 1
            if m.strength >= 2:
                strong_away += 1

    final_side = "neutral"
    # Require both margin and agreement. Conservative by design.
    if score >= 3 and (strong_home >= 2 or votes_home >= 3) and active >= 2:
        final_side = "home"
    elif score <= -3 and (strong_away >= 2 or votes_away >= 3) and active >= 2:
        final_side = "away"

    return final_side, score, {
        "votes_home": votes_home,
        "votes_away": votes_away,
        "strong_home": strong_home,
        "strong_away": strong_away,
        "active": active,
    }


def bet_decision(mods: List[ModuleResult], *, mode: str = "normal") -> Tuple[str, str]:
    """
    Decision layer:
    - conservative: >=2 active agree and none oppose; or single strength=3
    - normal: conservative + single strength=2 allowed if it's M1 and no oppose
    - aggressive: any single strength>=2 allowed if no oppose
    Returns (decision, side).
    """
    mode = (mode or "normal").lower()
    active = [m for m in mods if m.side != "neutral" and m.strength >= 1]
    home = [m for m in active if m.side == "home"]
    away = [m for m in active if m.side == "away"]
    if home and not away:
        if len(home) >= 2:
            return "BET", "home"
        if len(home) == 1 and home[0].strength >= 3:
            return "BET", "home"
        if mode == "aggressive" and len(home) == 1 and home[0].strength >= 2:
            return "BET", "home"
        if mode == "normal" and len(home) == 1 and home[0].strength >= 2 and home[0].name == "M1_dominance":
            return "BET", "home"
    if away and not home:
        if len(away) >= 2:
            return "BET", "away"
        if len(away) == 1 and away[0].strength >= 3:
            return "BET", "away"
        if mode == "aggressive" and len(away) == 1 and away[0].strength >= 2:
            return "BET", "away"
        if mode == "normal" and len(away) == 1 and away[0].strength >= 2 and away[0].name == "M1_dominance":
            return "BET", "away"
    return "SKIP", "neutral"


async def analyze_once(
    page: Page,
    *,
    event_payload: Dict[str, Any],
    match_url: str,
    event_id: int,
    max_history: int,
    history_only: bool = False,
    progress_cb: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    audit_history: bool = False,
    audit_features: bool = False,
) -> Tuple[MatchContext, List[ModuleResult], str, int, Dict[str, Any]]:
    _log_step(f"analyze_once start eventId={event_id} history_only={history_only}")
    e = event_payload.get("event") or {}
    home = e.get("homeTeam") or {}
    away = e.get("awayTeam") or {}
    ctx = MatchContext(
        event_id=event_id,
        url=match_url,
        home_id=int(home.get("id")),
        away_id=int(away.get("id")),
        home_name=str(home.get("name") or ""),
        away_name=str(away.get("name") or ""),
        surface=normalize_surface((e.get("groundType") or "")),
        set2_winner=_set2_winner_from_event_payload(event_payload),
    )
    surface = ctx.surface

    stats: Optional[Dict[str, Any]] = None
    points: Optional[DominanceLivePoints] = None
    if history_only:
        # User-requested mode: ignore current match statistics entirely.
        # Keep a placeholder snapshot so downstream code remains consistent.
        stats = {"statistics": []}
    # DOM stats only (matches what a human sees).
    # The UI can lag right after a set ends; retry briefly.
    last_dom_err: Optional[Exception] = None
    if not history_only:
        for _ in range(5):
            try:
                _dbg(f"DOM stats: eventId={event_id} (1ST+2ND)")
                _log_step(f"DOM stats: eventId={event_id} attempt")
                stats = await asyncio.wait_for(
                    extract_statistics_dom(page, match_url=match_url, event_id=event_id, periods=("1ST", "2ND")),
                    timeout=60.0,
                )
                break
            except Exception as ex:
                last_dom_err = ex
                # If DOM doesn't expose per-set switching (or tab can't be opened),
                # retrying likely won't help.
                if isinstance(ex, DomStatsError):
                    break
            await page.wait_for_timeout(600)

    dom_diag: Optional[Dict[str, Any]] = None
    dom_seen: Optional[Dict[str, Any]] = None
    dom_unmapped: Optional[Dict[str, Any]] = None
    if stats is not None:
        # Strict: no fallbacks. We expect real 1ST/2ND to be present.
        try:
            meta_dom = stats.get("_meta") if isinstance(stats, dict) else None
            if isinstance(meta_dom, dict) and isinstance(meta_dom.get("diag"), dict):
                dom_diag = meta_dom.get("diag")
            if isinstance(meta_dom, dict) and isinstance(meta_dom.get("seen"), dict):
                dom_seen = meta_dom.get("seen")
            if isinstance(meta_dom, dict) and isinstance(meta_dom.get("unmapped"), dict):
                dom_unmapped = meta_dom.get("unmapped")
            stats.pop("_meta", None)
        except Exception:
            pass
        if not history_only:
            points = _points_1st2nd_from_stats(stats)

    # Sofascore community votes ("Кто победит?") if available.
    votes_payload: Optional[Dict[str, Any]] = None
    try:
        _dbg(f"Votes: eventId={event_id}")
        _log_step(f"Votes: eventId={event_id}")
        votes_payload = await get_event_votes(page, event_id)
    except Exception:
        votes_payload = None
    if stats is None:
        # Strict: no fallbacks. If current stats are required, error out.
        if last_dom_err is not None:
            raise DomStatsError(str(last_dom_err))
        raise DomStatsError("DOM stats missing (no 1ST/2ND)")
    snap = MatchSnapshot(event_id=event_id, stats=stats)

    # History:
    # User wants strict recency. We collect exactly `max_history` per player and compute both last3 and last5
    # windows from that same pool (so dynamics are visible without “размывания”).
    #
    # IMPORTANT: running multiple tabs in parallel increases CAPTCHA/CF challenges.
    # We scrape sequentially in the same tab by default.
    history_pool = int(max_history)
    _dbg(f"History: pool={history_pool} (sequential, single tab)")
    _log_step(f"History: pool={history_pool}")

    _dbg(f"History: home teamId={ctx.home_id} max={history_pool}")
    _log_step(f"History: home teamId={ctx.home_id}")
    rows_home_pool, audit_home = await _history_rows_for_player_audit(
        page,
        match_url=match_url,
        match_event_id=event_id,
        match_side="home",
        team_id=ctx.home_id,
        team_slug=str(home.get("slug") or ""),
        max_history=history_pool,
        slow_mode=history_only,
        progress_cb=progress_cb,
        progress_label="home",
    )
    def _home_unavailable_for_fair_compare(rows: List[Dict[str, Any]], audit: HistoryAudit) -> bool:
        # If one side has no valid rows and hit a strong failure signal,
        # skip the opposite side scan: comparison would be biased anyway.
        ff = str(getattr(audit, "failfast_trigger", "") or "").strip().lower()
        if ff == "missing_stats_in_required_five":
            return True
        if len(rows) > 0:
            return False
        exr = dict(getattr(audit, "excluded_by_reason", {}) or {})
        if ff in (
            "statistics_tab_unreachable_repeated",
            "rows_not_ready_repeated",
            "timeout_storm_consecutive",
        ):
            return True
        if ff in ("budget_exceeded", "budget_exceeded_wait"):
            if history_only:
                return False
            scanned_n = int(getattr(audit, "scanned", 0) or 0)
            valid_n = int(getattr(audit, "valid", 0) or 0)
            hard = False
            hard = hard or int(exr.get("rows_not_ready_repeated", 0) or 0) > 0
            hard = hard or int(exr.get("statistics_tab_unreachable_storm", 0) or 0) > 0
            hard = hard or int(exr.get("stats_not_provided_storm", 0) or 0) > 0
            hard = hard or int(exr.get("no_finished_singles", 0) or 0) > 0
            hard = hard or (int(exr.get("profile_links_empty", 0) or 0) > 0 and int(getattr(audit, "candidates", 0) or 0) <= 0)
            if hard:
                return True
            return scanned_n >= 8 and valid_n == 0
        if int(exr.get("code_statistics_tab_unreachable", 0) or 0) >= _tab_unreachable_failfast_hits():
            return True
        if int(exr.get("code_stats_not_provided", 0) or 0) >= _tab_unreachable_failfast_hits():
            return True
        for k in (
            "statistics_tab_unreachable_storm",
            "rows_not_ready_repeated",
            "stats_not_provided_storm",
            "dom_timeout_storm",
            "no_finished_singles",
            "api_history_fetch_failed",
        ):
            if int(exr.get(k, 0) or 0) > 0:
                return True
        if int(exr.get("profile_links_empty", 0) or 0) > 0 and int(getattr(audit, "candidates", 0) or 0) <= 0:
            return True
        if (not history_only) and int(getattr(audit, "scanned", 0) or 0) >= 8 and int(getattr(audit, "valid", 0) or 0) == 0:
            return True
        return False

    if _home_unavailable_for_fair_compare(rows_home_pool, audit_home):
        _log_step("History: away skipped (home unavailable for fair comparison)")
        rows_away_pool = []
        audit_away = HistoryAudit(
            team_id=ctx.away_id,
            max_history=history_pool,
            surface_filter=surface if surface else None,
            candidates=0,
            scanned=0,
            valid=0,
            used=0,
            over_max_history=0,
            dropped_to_match_opponent=0,
            excluded_by_reason={"skipped_due_home_unavailable": 1},
            used_events=[],
            excluded_events=[],
            coverage={},
            failfast_trigger="skipped_due_home_unavailable",
            dom_timeouts=0,
            spent_s=0.0,
            budget_hit=False,
            context_resets=0,
        )
    else:
        _dbg(f"History: away teamId={ctx.away_id} max={history_pool}")
        _log_step(f"History: away teamId={ctx.away_id}")
        rows_away_pool, audit_away = await _history_rows_for_player_audit(
            page,
            match_url=match_url,
            match_event_id=event_id,
            match_side="away",
            team_id=ctx.away_id,
            team_slug=str(away.get("slug") or ""),
            max_history=history_pool,
            slow_mode=history_only,
            progress_cb=progress_cb,
            progress_label="away",
        )
    pool_n = min(len(rows_home_pool), len(rows_away_pool), history_pool)
    _log_step(
        f"History: collected home={len(rows_home_pool)}/{history_pool} away={len(rows_away_pool)}/{history_pool} pool_n={pool_n}"
    )
    meta_diag = {
        "home": {
            "timeouts": int(getattr(audit_home, "dom_timeouts", 0) or 0),
            "scanned": int(getattr(audit_home, "scanned", 0) or 0),
            "valid": int(getattr(audit_home, "valid", 0) or 0),
            "spent_s": float(getattr(audit_home, "spent_s", 0.0) or 0.0),
            "budget_hit": bool(getattr(audit_home, "budget_hit", False)),
            "failfast_trigger": getattr(audit_home, "failfast_trigger", None),
            "context_resets": int(getattr(audit_home, "context_resets", 0) or 0),
            "candidate_discovery_ms": float(getattr(audit_home, "candidate_discovery_ms", 0.0) or 0.0),
            "candidate_prefilter_ms": float(getattr(audit_home, "candidate_prefilter_ms", 0.0) or 0.0),
            "dom_extract_ms_total": float(getattr(audit_home, "dom_extract_ms_total", 0.0) or 0.0),
            "dom_extract_ms_p50": float(getattr(audit_home, "dom_extract_ms_per_event_p50", 0.0) or 0.0),
            "dom_extract_ms_p95": float(getattr(audit_home, "dom_extract_ms_per_event_p95", 0.0) or 0.0),
            "wasted_dom_attempts": int(getattr(audit_home, "wasted_dom_attempts", 0) or 0),
            "budget_extended": bool(getattr(audit_home, "budget_extended", False)),
            "budget_extended_by_s": float(getattr(audit_home, "budget_extended_by_s", 0.0) or 0.0),
            "transient_start_failures": int(getattr(audit_home, "transient_start_failures", 0) or 0),
        },
        "away": {
            "timeouts": int(getattr(audit_away, "dom_timeouts", 0) or 0),
            "scanned": int(getattr(audit_away, "scanned", 0) or 0),
            "valid": int(getattr(audit_away, "valid", 0) or 0),
            "spent_s": float(getattr(audit_away, "spent_s", 0.0) or 0.0),
            "budget_hit": bool(getattr(audit_away, "budget_hit", False)),
            "failfast_trigger": getattr(audit_away, "failfast_trigger", None),
            "context_resets": int(getattr(audit_away, "context_resets", 0) or 0),
            "candidate_discovery_ms": float(getattr(audit_away, "candidate_discovery_ms", 0.0) or 0.0),
            "candidate_prefilter_ms": float(getattr(audit_away, "candidate_prefilter_ms", 0.0) or 0.0),
            "dom_extract_ms_total": float(getattr(audit_away, "dom_extract_ms_total", 0.0) or 0.0),
            "dom_extract_ms_p50": float(getattr(audit_away, "dom_extract_ms_per_event_p50", 0.0) or 0.0),
            "dom_extract_ms_p95": float(getattr(audit_away, "dom_extract_ms_per_event_p95", 0.0) or 0.0),
            "wasted_dom_attempts": int(getattr(audit_away, "wasted_dom_attempts", 0) or 0),
            "budget_extended": bool(getattr(audit_away, "budget_extended", False)),
            "budget_extended_by_s": float(getattr(audit_away, "budget_extended_by_s", 0.0) or 0.0),
            "transient_start_failures": int(getattr(audit_away, "transient_start_failures", 0) or 0),
        },
    }
    for side_key, audit_obj in (("home", audit_home), ("away", audit_away)):
        try:
            exr = dict(getattr(audit_obj, "excluded_by_reason", {}) or {})
            code_pairs = [(k.replace("code_", "", 1), int(v)) for k, v in exr.items() if str(k).startswith("code_")]
            if code_pairs:
                code_pairs.sort(key=lambda kv: kv[1], reverse=True)
                meta_diag[side_key]["code"] = code_pairs[0][0]
        except Exception:
            pass
    strict_full_history = os.getenv("THIRDSET_REQUIRE_FULL_HISTORY", "1").strip().lower() not in ("0", "false", "no")
    if strict_full_history and (
        int(len(rows_home_pool)) < int(history_pool) or int(len(rows_away_pool)) < int(history_pool)
    ):
        def _derive_strict_code(a_home: HistoryAudit, a_away: HistoryAudit) -> str:
            try:
                ff_vals = {
                    str(getattr(a_home, "failfast_trigger", "") or "").strip().lower(),
                    str(getattr(a_away, "failfast_trigger", "") or "").strip().lower(),
                }
                if (not history_only) and any(x in ff_vals for x in ("budget_exceeded_wait", "budget_exceeded")):
                    trans_hits = 0
                    absent_hits = 0
                    for a in (a_home, a_away):
                        exr = dict(getattr(a, "excluded_by_reason", {}) or {})
                        cand_n = int(getattr(a, "candidates", 0) or 0)
                        trans_hits += int(exr.get("dom_timeout_storm", 0) or 0)
                        trans_hits += int(exr.get("code_dom_extract_timeout", 0) or 0)
                        trans_hits += int(exr.get("code_scrape_eval_timeout", 0) or 0)
                        trans_hits += int(exr.get("code_statistics_tab_unreachable", 0) or 0)
                        trans_hits += int(exr.get("code_consent_overlay_blocked", 0) or 0)
                        absent_hits += int(exr.get("rows_not_ready_repeated", 0) or 0)
                        absent_hits += int(exr.get("code_rows_not_ready", 0) or 0)
                        absent_hits += int(exr.get("stats_not_provided_storm", 0) or 0)
                        absent_hits += int(exr.get("code_stats_not_provided", 0) or 0)
                        absent_hits += int(exr.get("no_finished_singles", 0) or 0)
                        if int(exr.get("profile_links_empty", 0) or 0) > 0 and cand_n <= 0:
                            absent_hits += 1
                    if absent_hits > 0 and absent_hits >= trans_hits:
                        return "stats_absent"
                    if trans_hits > 0:
                        return "stats_not_loaded"
                if any(x in ff_vals for x in ("timeout_storm_consecutive", "timeout_storm_rate")):
                    return "stats_not_loaded"
                if "rows_not_ready_repeated" in ff_vals:
                    return "stats_absent"
                for a in (a_home, a_away):
                    exr = dict(getattr(a, "excluded_by_reason", {}) or {})
                    if int(exr.get("dom_timeout_storm", 0) or 0) > 0:
                        return "stats_not_loaded"
                    if int(exr.get("rows_not_ready_repeated", 0) or 0) > 0:
                        return "stats_absent"
                    if int(exr.get("code_statistics_tab_unreachable", 0) or 0) > 0:
                        return "stats_not_loaded"
                    if int(exr.get("code_rows_not_ready", 0) or 0) > 0 and int(getattr(a, "valid", 0) or 0) == 0:
                        return "stats_absent"
                    if int(exr.get("code_consent_overlay_blocked", 0) or 0) > 0:
                        return "stats_not_loaded"
                    if int(exr.get("code_dom_extract_timeout", 0) or 0) > 0:
                        return "stats_not_loaded"
                    if int(exr.get("code_scrape_eval_timeout", 0) or 0) > 0:
                        return "stats_not_loaded"
                    if int(exr.get("statistics_tab_unreachable_storm", 0) or 0) > 0:
                        return "stats_not_loaded"
                    if int(exr.get("ui_unavailable_storm", 0) or 0) > 0:
                        return "stats_not_loaded"
                    if int(exr.get("profile_links_empty", 0) or 0) > 0 and int(getattr(a, "candidates", 0) or 0) <= 0:
                        return "stats_absent"
                    if int(exr.get("no_finished_singles", 0) or 0) > 0:
                        return "stats_absent"
                return "insufficient_history"
            except Exception:
                return "insufficient_history"

        def _fmt_audit_short(side: str, audit: HistoryAudit) -> str:
            parts: List[str] = [f"{side}: cand={audit.candidates} scanned={audit.scanned} valid={audit.valid}"]
            if getattr(audit, "failfast_trigger", None):
                parts.append(f"ff={getattr(audit, 'failfast_trigger')}")
            if bool(getattr(audit, "budget_extended", False)):
                parts.append(f"ext=+{int(round(float(getattr(audit, 'budget_extended_by_s', 0.0) or 0.0)))}s")
            if int(getattr(audit, "transient_start_failures", 0) or 0) > 0:
                parts.append(f"tstart={int(getattr(audit, 'transient_start_failures', 0) or 0)}")
            exr = dict(getattr(audit, "excluded_by_reason", {}) or {})
            if exr:
                cand_n = int(getattr(audit, "candidates", 0) or 0)
                top = sorted(
                    (
                        (k, int(v))
                        for k, v in exr.items()
                        if int(v) > 0
                        and not (str(k) == "profile_links_empty" and cand_n > 0)
                        and not (history_only and str(k) in ("budget_exceeded", "budget_exceeded_wait"))
                    ),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[:3]
                if top:
                    parts.append("reasons=" + ",".join(f"{k}:{v}" for k, v in top))
            return " ".join(parts)

        strict_code = _derive_strict_code(audit_home, audit_away)
        raise SofascoreError(
            f"code={strict_code} "
            f"История не собрана: требуется home={history_pool}/{history_pool}, away={history_pool}/{history_pool}; "
            f"получено home={len(rows_home_pool)}/{history_pool}, away={len(rows_away_pool)}/{history_pool}. "
            f"Диагностика: {_fmt_audit_short('home', audit_home)} | {_fmt_audit_short('away', audit_away)}"
        )
    if pool_n <= 0:
        def _derive_failure_code(a_home: HistoryAudit, a_away: HistoryAudit) -> Optional[str]:
            try:
                counts: Dict[str, int] = {}
                transient_codes = {
                    "statistics_tab_unreachable",
                    "consent_overlay_blocked",
                    "dom_extract_timeout",
                    "scrape_eval_timeout",
                    "rows_not_ready",
                    "period_select_failed",
                    "navigation_failed",
                    "api_history_fetch_failed",
                    "dom_timeout_storm",
                    "ui_unavailable_storm",
                    "statistics_tab_unreachable_storm",
                }
                absent_codes = {
                    "profile_links_empty",
                    "no_finished_singles",
                    "stats_absent_likely",
                    "stats_not_provided",
                    "rows_not_ready_repeated",
                }
                for a in (a_home, a_away):
                    exr = dict(getattr(a, "excluded_by_reason", {}) or {})
                    cand_n = int(getattr(a, "candidates", 0) or 0)
                    for k, v in exr.items():
                        if not isinstance(v, int) or v <= 0:
                            continue
                        if str(k).startswith("code_"):
                            code = str(k).replace("code_", "", 1).strip().lower()
                        elif str(k) in (
                            "profile_links_empty",
                            "no_finished_singles",
                            "api_history_fetch_failed",
                            "stats_absent_likely",
                            "stats_not_provided",
                            "rows_not_ready_repeated",
                            "dom_timeout_storm",
                            "ui_unavailable_storm",
                            "statistics_tab_unreachable_storm",
                        ):
                            code = str(k).strip().lower()
                        else:
                            continue
                        if code:
                            grouped = code
                            if code in transient_codes:
                                grouped = "stats_not_loaded"
                            elif code in absent_codes:
                                if code == "profile_links_empty" and cand_n > 0:
                                    continue
                                grouped = "stats_absent"
                            counts[grouped] = counts.get(grouped, 0) + int(v)
                if not counts:
                    return None
                top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[0][0]
                return str(top)
            except Exception:
                return None

        def _fmt_audit(side: str, audit: HistoryAudit) -> str:
            try:
                parts: List[str] = []
                parts.append(
                    f"{side}: cand={audit.candidates} scanned={audit.scanned} valid={audit.valid}"
                )
                if getattr(audit, "dom_timeouts", 0):
                    parts.append(f"timeouts={int(getattr(audit, 'dom_timeouts', 0) or 0)}")
                if getattr(audit, "budget_hit", False):
                    if not history_only:
                        parts.append("budget_hit=yes")
                if getattr(audit, "failfast_trigger", None):
                    parts.append(f"ff={getattr(audit, 'failfast_trigger')}")
                if bool(getattr(audit, "budget_extended", False)):
                    parts.append(f"ext=+{int(round(float(getattr(audit, 'budget_extended_by_s', 0.0) or 0.0)))}s")
                if int(getattr(audit, "transient_start_failures", 0) or 0) > 0:
                    parts.append(f"tstart={int(getattr(audit, 'transient_start_failures', 0) or 0)}")
                if getattr(audit, "context_resets", 0):
                    parts.append(f"resets={int(getattr(audit, 'context_resets', 0) or 0)}")
                if float(getattr(audit, "dom_extract_ms_per_event_p95", 0.0) or 0.0) > 0:
                    parts.append(f"dom_p95={int(float(getattr(audit, 'dom_extract_ms_per_event_p95', 0.0) or 0.0))}ms")
                if int(getattr(audit, "wasted_dom_attempts", 0) or 0) > 0:
                    parts.append(f"wasted={int(getattr(audit, 'wasted_dom_attempts', 0) or 0)}")
                if isinstance(audit.excluded_by_reason, dict) and audit.excluded_by_reason:
                    cand_n = int(getattr(audit, "candidates", 0) or 0)
                    top = sorted(
                        (
                            (k, int(v))
                            for k, v in audit.excluded_by_reason.items()
                            if not (str(k) == "profile_links_empty" and cand_n > 0)
                            and not (history_only and str(k) in ("budget_exceeded", "budget_exceeded_wait"))
                        ),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )
                    top = top[:4]
                    parts.append("reasons=" + ",".join(f"{k}:{v}" for k, v in top))
                dom_err = None
                if isinstance(audit.excluded_events, list):
                    for evx in audit.excluded_events:
                        if isinstance(evx, dict) and evx.get("dom_error"):
                            dom_err = str(evx.get("dom_error") or "")
                            break
                if dom_err:
                    # Keep it short for TG.
                    dom_err = dom_err.strip().replace("\n", " ")
                    if len(dom_err) > 140:
                        dom_err = dom_err[:140] + "…"
                    parts.append(f"dom_err={dom_err}")
                # Add a couple of concrete examples for fast debugging.
                examples: List[str] = []
                if isinstance(audit.excluded_events, list):
                    for evx in audit.excluded_events:
                        if not isinstance(evx, dict):
                            continue
                        rid = evx.get("eventId")
                        reason = evx.get("reason")
                        derr = evx.get("dom_error")
                        if rid is None or reason is None:
                            continue
                        s = f"eid={rid}:{reason}"
                        if derr:
                            s += f":{str(derr).splitlines()[0][:80]}"
                        examples.append(s)
                        if len(examples) >= 2:
                            break
                if examples:
                    parts.append("ex=" + ";".join(examples))
                return " ".join(parts)
            except Exception:
                return f"{side}: audit_error"

        storm_like = bool(getattr(audit_home, "dom_timeouts", 0) and getattr(audit_away, "dom_timeouts", 0))
        storm_like = storm_like and (audit_home.valid == 0 and audit_away.valid == 0)
        if storm_like:
            raise SofascoreError(
                "code=timeout_storm История не собрана: timeout storm "
                f"(home: {audit_home.dom_timeouts}/{audit_home.scanned}, away: {audit_away.dom_timeouts}/{audit_away.scanned}, "
                f"budget={int(getattr(audit_home, 'spent_s', 0.0) + getattr(audit_away, 'spent_s', 0.0))}s)."
            )
        diag = " | ".join(
            p for p in (_fmt_audit("home", audit_home), _fmt_audit("away", audit_away)) if p
        )
        code = _derive_failure_code(audit_home, audit_away)
        code_prefix = f"code={code} " if code else ""
        raise SofascoreError(
            f"{code_prefix}История не собрана (home={len(rows_home_pool)}/{history_pool}, away={len(rows_away_pool)}/{history_pool}). "
            f"Диагностика: {diag}"
        )
    rows_home_pool = rows_home_pool[:history_pool]
    rows_away_pool = rows_away_pool[:history_pool]

    recent_n_home = min(len(rows_home_pool), int(max_history))
    recent_n_away = min(len(rows_away_pool), int(max_history))
    rows_home_recent = rows_home_pool[:recent_n_home]
    rows_away_recent = rows_away_pool[:recent_n_away]
    # For summary/form signals we always use the freshest last-N (no substitution from older matches).
    rows_home_form = rows_home_recent
    rows_away_form = rows_away_recent

    m1_hist_home, m1_hist_away = _build_m1_history_profiles(rows_home_recent, rows_away_recent)
    # Calibration prefers more samples.
    cal_home = build_surface_calibration(rows_home_pool, surface=surface)
    cal_away = build_surface_calibration(rows_away_pool, surface=surface)

    mods: List[ModuleResult] = []
    if history_only:
        # History-only: do not use current match stats at all.
        h5 = _tpw12_history_scores(rows_home_form, max_n=5)
        a5 = _tpw12_history_scores(rows_away_form, max_n=5)
        r5h = h5.get("rating") if isinstance(h5, dict) else None
        r5a = a5.get("rating") if isinstance(a5, dict) else None
        mods.append(module1_history_tpw12(rating5_home=r5h, rating5_away=r5a))
        mods.append(module2_history_serve(cal_home=cal_home, cal_away=cal_away))
        mods.append(module3_history_return(cal_home=cal_home, cal_away=cal_away))
        mods.append(module4_history_clutch(cal_home=cal_home, cal_away=cal_away))
    else:
        # Live 1:1 after two sets: current stats + history calibration.
        if points is None:
            mods.append(
                ModuleResult(
                    "M1_dominance", "neutral", 0, ["missing points stats for 1ST+2ND"], ["missing_points_stats"]
                )
            )
        else:
            mods.append(module1_dominance(points_1st2nd=points, history_home=m1_hist_home, history_away=m1_hist_away))
        mods.append(module2_second_serve_fragility(snapshot=snap, cal_home=cal_home, cal_away=cal_away))
        mods.append(module3_return_pressure(snapshot=snap, cal_home=cal_home, cal_away=cal_away))
        mods.append(module4_clutch(snapshot=snap, cal_home=cal_home, cal_away=cal_away))

    mods.append(
        module5_form_profile(
            history_rows_home=rows_home_pool,
            history_rows_away=rows_away_pool,
            current_set2_winner=ctx.set2_winner,
            cal_home=cal_home,
            cal_away=cal_away,
        )
    )

    # No "stat filling": if we're in history-only mode and even TPW(1+2) history is missing,
    # abort this match early (user wants a clear "insufficient stats" instead of 50/50 noise).
    if history_only:
        try:
            m1 = mods[0]
        except Exception:
            m1 = None
        if isinstance(m1, ModuleResult) and ("rating5_missing" in (m1.flags or [])):
            raise SofascoreError("Недостаточно статистики по истории (TPW 1+2).")

    final_side, score, meta = ensemble(mods)
    meta["history_n"] = min(recent_n_home, recent_n_away)
    meta["history_n_home"] = recent_n_home
    meta["history_n_away"] = recent_n_away
    meta["history_pool_n"] = pool_n
    meta["history_diag"] = meta_diag
    meta["surface"] = surface
    # Compact module results for user-facing output (console/TG).
    meta["mods_compact"] = [
        {"name": m.name, "side": m.side, "strength": m.strength, "flags": list(m.flags or [])} for m in mods
    ]
    if isinstance(dom_diag, dict):
        meta["dom_diag"] = dom_diag
    if isinstance(dom_seen, dict):
        meta["dom_seen"] = dom_seen
    if isinstance(dom_unmapped, dict):
        meta["dom_unmapped"] = dom_unmapped
    if history_only:
        meta["stats_mode"] = "history_only"
    else:
        meta["stats_mode"] = "per_set"
    # Always compute a compact summary from indices + modules; full features are optional.
    feat_for_summary = _build_feature_dump(
        ctx=ctx,
        snapshot=snap,
        points=points,
        cal_home=cal_home,
        cal_away=cal_away,
        rows_home=rows_home_form,
        rows_away=rows_away_form,
    )
    signals = _signals_from_features(features=feat_for_summary, max_history=max_history)
    feat_for_summary["signals"] = signals
    forecast = _probability_from_signals(signals=signals, mods=mods)
    feat_for_summary["forecast"] = forecast
    meta["summary"] = _compact_summary_from(
        mods=mods, indices=(feat_for_summary.get("indices") or {}), signals=signals, forecast=forecast
    )
    if audit_history:
        meta["history_audit"] = {"home": audit_home.to_dict(), "away": audit_away.to_dict()}
    if audit_features:
        meta["features"] = feat_for_summary
    if votes_payload and isinstance(votes_payload, dict):
        meta["sofascore_votes"] = votes_payload
    return ctx, mods, final_side, score, meta
