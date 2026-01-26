from __future__ import annotations

import math
import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from playwright.async_api import Page

from third_set.calibration import MetricSummary, deviation, normalize_surface, summarize
from third_set.dominance import DominanceLivePoints
from third_set.snapshot import MatchSnapshot
from third_set.modules import (
    ModuleResult,
    module1_dominance,
    module2_second_serve_fragility,
    module3_return_pressure,
    module4_clutch,
    module5_form_profile,
)
from third_set.sofascore import (
    SofascoreError,
    get_event_votes,
    get_last_finished_singles_events,
    is_singles_event,
    summarize_event_for_team,
)
from third_set.stats_parser import get_periods_present, sum_event_value
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


def _metric_summary_to_dict(ms: Optional[MetricSummary]) -> Optional[Dict[str, Any]]:
    if ms is None:
        return None
    return {"n": ms.n, "mean": ms.mean, "sd": ms.sd}


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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


def _ewma(values: List[float], *, alpha: float) -> Optional[float]:
    if not values:
        return None
    a = float(alpha)
    if a <= 0.0:
        return float(values[0])
    if a >= 1.0:
        return float(values[-1])
    m = float(values[0])
    for v in values[1:]:
        m = a * float(v) + (1.0 - a) * m
    return m


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

    def _rating_norm(side: str, key: str) -> float:
        sc = (tpw12_scores.get(side) or {}).get(key)
        if isinstance(sc, dict):
            r = sc.get("rating")
            if isinstance(r, (int, float)):
                return _clamp(float(r) / 100.0, 0.0, 1.0)
        return 0.5

    strength3_h = _rating_norm("home", "last3")
    strength3_a = _rating_norm("away", "last3")
    strength5_h = _rating_norm("home", "last5")
    strength5_a = _rating_norm("away", "last5")
    # `strength` in signals == last5 rating (our default history pool size)
    strength_h = strength5_h
    strength_a = strength5_a

    # Form is explicit momentum: EWMA results+stats window vs baseline window.
    # Fall back to the TPW rating delta if windowed form isn't available.
    form_h = _form_window("home", window_n=min(5, max_history)) or strength5_h
    form_a = _form_window("away", window_n=min(5, max_history)) or strength5_a

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

    stability_h = _stability("home")
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

    d_strength = g("strength") or 0.0
    d_form = g("form") or 0.0
    d_stab = g("stability")  # optional

    # Module evidence: normalized into [-1..1]
    mod_score = 0.0
    active = 0
    for m in mods:
        if m.side == "neutral" or m.strength <= 0:
            continue
        active += 1
        mod_score += float(m.strength) * (1.0 if m.side == "home" else -1.0)
    mod_norm = _clamp(mod_score / 6.0, -1.0, 1.0)

    # Base linear logit from signals
    # Strength is primary (historical rating), form is secondary (momentum).
    x = 2.6 * d_strength + 1.4 * d_form + 1.2 * mod_norm
    # Stability gates confidence more than direction: if both unstable, flatten probability.
    gate = 1.0
    if isinstance(d_stab, (int, float)):
        # d_stab is (home-away) of stability, but we want absolute stability amount; approximate via home/away.
        sh = (signals.get("stability") or {}).get("home")
        sa = (signals.get("stability") or {}).get("away")
        if isinstance(sh, (int, float)) and isinstance(sa, (int, float)):
            gate = 0.70 + 0.30 * float(min(sh, sa))
    x *= gate

    p_home = _sigmoid(x)
    p_away = 1.0 - p_home
    conf = abs(p_home - 0.5) * 2.0
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
        "components": {"d_strength": d_strength, "d_form": d_form, "mod_norm": mod_norm, "mod_score": mod_score, "active_mods": active},
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
    def _hist_rating_norm(side: str) -> float:
        try:
            sc = ((hist.get("tpw12_scores") or {}).get(side) or {}).get("last5")
        except Exception:
            sc = None
        if isinstance(sc, dict):
            r = sc.get("rating")
            if isinstance(r, (int, float)):
                return _clamp(float(r) / 100.0, 0.0, 1.0)
        return 0.5

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

    Всегда возвращает числа (без NA).
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
            "mu_pp": 0.0,
            "delta_pp": 0.0,
            "sigma_pp": 0.0,
            "power": 50.0,
            "form": 50.0,
            "volatility": 50.0,
            "rating": 50.0,
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


async def _history_rows_for_player(
    page: Page,
    *,
    team_id: int,
    max_history: int,
    surface_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    rows, _audit = await _history_rows_for_player_audit(
        page, team_id=team_id, max_history=max_history, surface_filter=surface_filter
    )
    return rows


async def _history_rows_for_player_audit(
    page: Page,
    *,
    team_id: int,
    max_history: int,
    surface_filter: Optional[str] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    progress_label: str = "",
) -> Tuple[List[Dict[str, Any]], HistoryAudit]:
    # We only need `max_history` valid rows. For fullness we always parse history via DOM,
    # so we fetch a larger candidate list and stop early once we collected enough rows.
    events = await get_last_finished_singles_events(page, team_id, limit=max(max_history * 10, 60))
    if surface_filter:
        sf = normalize_surface(surface_filter)
    else:
        sf = None
    # DOM parsing opens match pages; keep it sequential for stability.
    sem = asyncio.Semaphore(1)
    hist_page: Optional[Page] = None
    try:
        hist_page = await page.context.new_page()
    except Exception:
        hist_page = None

    async def one(idx: int, ev: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
        eid = ev.get("id")
        if not eid:
            return idx, None, "missing_event_id", {}
        summary = summarize_event_for_team(ev, team_id=team_id)
        if summary.get("won") is None:
            return idx, None, "winner_unknown", summary
        # Determine if match is BO3 decider via score fields.
        hs = ev.get("homeScore") or {}
        aw = ev.get("awayScore") or {}
        has_set3 = hs.get("period3") is not None and aw.get("period3") is not None
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

        surface = normalize_surface(ev.get("groundType"))
        if sf and surface != sf:
            return idx, None, "surface_mismatch", summary

        # Always use DOM for history (fullness > speed; avoids API inconsistencies).
        stats = None
        dom_used = False
        dom_err: Optional[str] = None
        async with sem:
            try:
                slug = ev.get("slug")
                custom = ev.get("customId")
                if slug and custom and hist_page is not None:
                    url = f"https://www.sofascore.com/ru/tennis/match/{slug}/{custom}"
                    # If the match ended 2:0, don't waste time trying to select 3RD.
                    periods = ("1ST", "2ND", "3RD") if has_set3 else ("1ST", "2ND")
                    stats = await asyncio.wait_for(
                        extract_statistics_dom(hist_page, match_url=url, event_id=int(eid), periods=periods),
                        timeout=45.0,
                    )
                    dom_used = True
            except Exception as ex:
                dom_err = f"{type(ex).__name__}: {ex}"
                stats = None

        if stats is None:
            # Strict history: if we can't read per-set stats, skip and scan more candidates.
            if dom_err:
                summary = dict(summary)
                summary["dom_error"] = dom_err
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
            r = h.rate if summary.get("isHome") else a.rate
            return float(r) if r is not None else None

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
                return None
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
        return idx, row, None, summary

    excluded_by_reason: Dict[str, int] = {}
    excluded_events: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    scanned = 0
    for i, ev in enumerate(events):
        idx, row, reason, summary = await one(i, ev)
        scanned += 1
        if reason is not None:
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
        elif row is None:
            excluded_by_reason["unknown_drop"] = excluded_by_reason.get("unknown_drop", 0) + 1
        else:
            valid_rows.append(row)
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
        if len(valid_rows) >= max_history:
            break

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
    )
    try:
        if hist_page is not None:
            await hist_page.close()
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

    def collect(key: str):
        return [r.get(key) for r in use]

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
    n = min(len(rows_home), len(rows_away))
    rows_home = rows_home[:n]
    rows_away = rows_away[:n]

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
    history_pool = int(max_history)
    _dbg(f"History: home teamId={ctx.home_id} max={history_pool}")
    rows_home_pool, audit_home = await _history_rows_for_player_audit(
        page,
        team_id=ctx.home_id,
        max_history=history_pool,
        progress_cb=progress_cb,
        progress_label="home",
    )
    _dbg(f"History: away teamId={ctx.away_id} max={history_pool}")
    rows_away_pool, audit_away = await _history_rows_for_player_audit(
        page,
        team_id=ctx.away_id,
        max_history=history_pool,
        progress_cb=progress_cb,
        progress_label="away",
    )
    pool_n = min(len(rows_home_pool), len(rows_away_pool), history_pool)
    if len(rows_home_pool) > pool_n:
        audit_home.dropped_to_match_opponent = len(rows_home_pool) - pool_n
    if len(rows_away_pool) > pool_n:
        audit_away.dropped_to_match_opponent = len(rows_away_pool) - pool_n
    rows_home_pool = rows_home_pool[:pool_n]
    rows_away_pool = rows_away_pool[:pool_n]

    recent_n = min(pool_n, int(max_history))
    rows_home_recent = rows_home_pool[:recent_n]
    rows_away_recent = rows_away_pool[:recent_n]

    def _select_recent_with_stats(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        # Keep recency order, but prefer rows where at least some stats are available
        # (missing_all_metrics=False). Fill with the rest if needed.
        if n <= 0:
            return []
        good: List[Dict[str, Any]] = []
        bad: List[Dict[str, Any]] = []
        for r in rows:
            if bool(r.get("missing_all_metrics")) is False:
                good.append(r)
            else:
                bad.append(r)
        out = (good + bad)[:n]
        return out

    # For summary/form signals we want last-N, but if within last-N Sofascore
    # doesn't provide usable stats, fill from slightly older matches (still within pool).
    rows_home_form = _select_recent_with_stats(rows_home_pool, recent_n)
    rows_away_form = _select_recent_with_stats(rows_away_pool, recent_n)

    m1_hist_home, m1_hist_away = _build_m1_history_profiles(rows_home_recent, rows_away_recent)
    # Calibration prefers more samples.
    cal_home = build_surface_calibration(rows_home_pool, surface=surface)
    cal_away = build_surface_calibration(rows_away_pool, surface=surface)

    mods: List[ModuleResult] = []
    if points is None:
        mods.append(
            ModuleResult("M1_dominance", "neutral", 0, ["missing points stats for 1ST+2ND"], ["missing_points_stats"])
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

    final_side, score, meta = ensemble(mods)
    meta["history_n"] = recent_n
    meta["history_pool_n"] = pool_n
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
