from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from third_set.dominance import DominanceLivePoints
from third_set.snapshot import MatchSnapshot
from third_set.calibration import MetricSummary, deviation
from third_set.stats_parser import Ratio, get_periods_present, parse_ratio, sum_event_value, sum_ratio_stat


@dataclass(frozen=True)
class ModuleResult:
    name: str
    side: str  # "home"|"away"|"neutral"
    strength: int  # 0..3
    explain: List[str]
    flags: List[str]


def _cap_strength(strength: int) -> int:
    return max(0, min(3, int(strength)))

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _weighted_mean(parts: List[Tuple[Optional[float], float]]) -> Optional[float]:
    num = 0.0
    den = 0.0
    for v, w in parts:
        if v is None:
            continue
        num += float(v) * float(w)
        den += float(w)
    if den <= 0:
        return None
    return num / den


def _reliability(n: int, *, full: int) -> float:
    if full <= 0:
        return 0.0
    if n <= 0:
        return 0.0
    return min(1.0, n / full)

def _side_from_value(value: float, *, eps: float) -> str:
    if value >= 0.5 + eps:
        return "home"
    if value <= 0.5 - eps:
        return "away"
    return "neutral"


def module1_dominance(
    *,
    points_1st2nd: DominanceLivePoints,
    history_home: Optional[Dict[str, Any]],
    history_away: Optional[Dict[str, Any]],
) -> ModuleResult:
    flags: List[str] = []
    explain: List[str] = []

    spw_home = points_1st2nd.spw_home
    rpw_home = points_1st2nd.rpw_home
    spw_away = points_1st2nd.spw_away
    rpw_away = points_1st2nd.rpw_away

    tp_home = spw_home + rpw_home
    tp_away = spw_away + rpw_away
    n = tp_home + tp_away
    if n <= 0:
        return ModuleResult("M1_dominance", "neutral", 0, ["no points"], ["no_points"])

    explain.append(f"PTS: SPW={spw_home}/{spw_away} RPW={rpw_home}/{rpw_away} TP={tp_home}/{tp_away} n={n}")

    tpw_home = tp_home / n
    delta_pp = 100.0 * (tpw_home - 0.5)

    # Return rate vs opponent serve points: RPW / (RPW + opponent SPW)
    ret_n_home = rpw_home + spw_away
    ret_n_away = rpw_away + spw_home
    rr_home = (rpw_home / ret_n_home) if ret_n_home > 0 else None
    rr_away = (rpw_away / ret_n_away) if ret_n_away > 0 else None

    dr = None
    logdr = None
    if rr_home is not None and rr_away is not None and rr_away > 0:
        dr = rr_home / rr_away
        if dr > 0:
            logdr = math.log(dr)
    else:
        flags.append("rd_missing")

    eps_tpw = 0.005
    eps_logdr = 0.05
    tpw_side = _side_from_value(tpw_home, eps=eps_tpw)
    dr_side = "neutral"
    if logdr is not None:
        if logdr >= eps_logdr:
            dr_side = "home"
        elif logdr <= -eps_logdr:
            dr_side = "away"

    delta = abs(tpw_home - 0.5)
    if n >= 90:
        strength_tpw = 3 if delta >= 0.03 else 2 if delta >= 0.02 else 1 if delta >= 0.015 else 0
    elif n >= 60:
        strength_tpw = 3 if delta >= 0.035 else 2 if delta >= 0.025 else 1 if delta >= 0.018 else 0
    else:
        strength_tpw = 2 if delta >= 0.03 else 1 if delta >= 0.02 else 0

    strength_dr = 0
    if logdr is not None:
        abs_logdr = abs(logdr)
        ret_n_min = min(ret_n_home, ret_n_away)
        if ret_n_min >= 60:
            strength_dr = 3 if abs_logdr >= 0.25 else 2 if abs_logdr >= 0.16 else 1 if abs_logdr >= 0.10 else 0
        elif ret_n_min >= 35:
            strength_dr = 2 if abs_logdr >= 0.25 else 1 if abs_logdr >= 0.16 else 0
        else:
            strength_dr = 1 if abs_logdr >= 0.25 else 0

    explain.append(f"TPW={tpw_home:.3f} (Δ={delta_pp:+.1f}pp) n={n} -> strength_TPW={strength_tpw}")
    if rr_home is not None and rr_away is not None:
        if logdr is not None:
            explain.append(
                f"RR_home={rr_home:.3f} RR_away={rr_away:.3f} retN={min(ret_n_home, ret_n_away)} DR={dr:.2f} logDR={logdr:+.3f} -> strength_DR={strength_dr}"
            )
        else:
            explain.append(f"RR_home={rr_home:.3f} RR_away={rr_away:.3f} retN={min(ret_n_home, ret_n_away)} DR=missing")
    else:
        explain.append("RR_home/RR_away=missing")

    # Composite dominance score (signed, home-positive), with reliability scaling by samples.
    rel_tpw = _reliability(n, full=120)
    rel_ret = _reliability(min(ret_n_home, ret_n_away), full=80)
    # Use TPW as base + a smaller logDR tilt (both are already "return/points" dominance).
    dom_score = rel_tpw * (tpw_home - 0.5) + (0.30 * rel_ret * ((logdr or 0.0) / 2.0))
    explain.append(f"CompositeDom={dom_score:+.3f}")

    # Side decision + conflict logic
    side = "neutral"
    strength = 0
    if logdr is None:
        # Allowed: side by TPW only, but cap strength at 2.
        side = tpw_side
        strength = min(2, strength_tpw)
        flags.append("rd_missing_side_by_tpw")
    else:
        if tpw_side == dr_side and tpw_side != "neutral":
            side = tpw_side
            strength = max(strength_tpw, strength_dr)
        elif tpw_side != "neutral" and dr_side == "neutral":
            side = tpw_side
            strength = strength_tpw
            flags.append("dr_neutral_side_by_tpw")
        elif tpw_side == "neutral" and dr_side != "neutral":
            side = dr_side
            # TPW is not confirming, so stay conservative.
            strength = min(2, strength_dr)
            flags.append("tpw_neutral_side_by_dr")
        else:
            # Conflict handling: 3 vs <=1 becomes side of strong with strength=2
            if tpw_side != "neutral" and dr_side != "neutral" and tpw_side != dr_side:
                hi = max(strength_tpw, strength_dr)
                lo = min(strength_tpw, strength_dr)
                if hi == 3 and lo <= 1:
                    strength = 2
                    side = tpw_side if strength_tpw == 3 else dr_side
                    flags.append("conflict_resolved_3v1")
                else:
                    flags.append("conflict_tpw_rd")
            else:
                flags.append("neutral_tpw_or_rd")

    # Guardrails
    if n < 45:
        strength = min(strength, 1)
        flags.append("too_few_points")

    # last-N calibration (Close/Decider) if provided
    if side != "neutral" and history_home and history_away:
        # history dict includes: close_rate, close_n, dec_rate, dec_n
        if strength >= 1:
            if side == "home":
                dec_diff = (history_home.get("dec_rate") or 0) - (history_away.get("dec_rate") or 0)
                dec_ok = history_home.get("dec_n", 0) >= 5 and history_away.get("dec_n", 0) >= 5
            else:
                dec_diff = (history_away.get("dec_rate") or 0) - (history_home.get("dec_rate") or 0)
                dec_ok = history_home.get("dec_n", 0) >= 5 and history_away.get("dec_n", 0) >= 5
            if dec_ok and abs(dec_diff) >= 0.20:
                strength += 1 if dec_diff > 0 else -1
                explain.append(f"LastN deciders diff={dec_diff:+.2f} -> {'+1' if dec_diff>0 else '-1'}")

        strength = _cap_strength(strength)

    # Neutral rescue via CloseWinRate only (when base ended neutral)
    if side == "neutral" and history_home and history_away:
        ch = history_home.get("close_rate")
        ca = history_away.get("close_rate")
        if (
            ch is not None
            and ca is not None
            and history_home.get("close_n", 0) >= 5
            and history_away.get("close_n", 0) >= 5
        ):
            if ch >= 0.60 and ca <= 0.40:
                side = "home"
                strength = 1
                explain.append("LastN close rescue -> home")
            elif ca >= 0.60 and ch <= 0.40:
                side = "away"
                strength = 1
                explain.append("LastN close rescue -> away")

    strength = _cap_strength(strength)
    explain.append(f"FINAL: {side} strength={strength}")
    return ModuleResult("M1_dominance", side, strength, explain[:6], flags)


def module2_second_serve_fragility(*, snapshot: MatchSnapshot, cal_home: Optional[Dict[str, MetricSummary]] = None, cal_away: Optional[Dict[str, MetricSummary]] = None) -> ModuleResult:
    flags: List[str] = []
    explain: List[str] = []
    periods = ("1ST", "2ND")

    # Second serve points (ratio)
    ss_home, ss_away = snapshot.ratio(periods=periods, group="Service", item="Second serve points")
    fspts_home, fspts_away = snapshot.ratio(periods=periods, group="Service", item="First serve points")
    fsin_home, fsin_away = snapshot.ratio(periods=periods, group="Service", item="First serve")
    aces_home, aces_away = snapshot.value(periods=periods, group="Service", item="Aces")
    df_home, df_away = snapshot.value(periods=periods, group="Service", item="Double faults")
    bps_home, bps_away = snapshot.ratio(periods=periods, group="Service", item="Break points saved")
    fsp_home, fsp_away = fspts_home, fspts_away

    if ss_home is None or ss_away is None:
        return ModuleResult("M2_second_serve", "neutral", 0, ["missing Second serve points"], ["missing_fields"])

    ssw_home = ss_home.rate
    ssw_away = ss_away.rate
    ssn = min(ss_home.total, ss_away.total)
    if ssw_home is None or ssw_away is None:
        return ModuleResult("M2_second_serve", "neutral", 0, ["bad second serve rates"], ["missing_fields"])

    fsw_home = fspts_home.rate if fspts_home else None
    fsw_away = fspts_away.rate if fspts_away else None
    fsn = min(fspts_home.total if fspts_home else 0, fspts_away.total if fspts_away else 0)

    fsinr_home = fsin_home.rate if fsin_home else None
    fsinr_away = fsin_away.rate if fsin_away else None
    fsin_n = min(fsin_home.total if fsin_home else 0, fsin_away.total if fsin_away else 0)

    # DFR: DF / total serve points (need totals from 1st+2nd serve points)
    dfr_home = None
    dfr_away = None
    serve_pts_n = 0
    if fsp_home is not None and fsp_away is not None and ss_home.total > 0 and ss_away.total > 0:
        spn_home = fsp_home.total + ss_home.total
        spn_away = fsp_away.total + ss_away.total
        if spn_home > 0 and spn_away > 0:
            dfr_home = df_home / spn_home
            dfr_away = df_away / spn_away
            serve_pts_n = min(spn_home, spn_away)
    if dfr_home is None or dfr_away is None:
        flags.append("dfr_missing")

    ace_rate_home = None
    ace_rate_away = None
    aces_present = (
        snapshot.raw(period="1ST", group="Service", item="Aces")
        or snapshot.raw(period="2ND", group="Service", item="Aces")
    )
    if aces_present and fsp_home is not None and fsp_away is not None:
        spn_home = fsp_home.total + ss_home.total
        spn_away = fsp_away.total + ss_away.total
        if spn_home > 0 and spn_away > 0:
            ace_rate_home = aces_home / spn_home
            ace_rate_away = aces_away / spn_away
    elif not aces_present:
        flags.append("aces_missing_field")

    # BP saved rate
    bpsr_home = bps_home.rate if bps_home else None
    bpsr_away = bps_away.rate if bps_away else None
    bpn = min(bps_home.total if bps_home else 0, bps_away.total if bps_away else 0)

    # Quality caps
    if ssn < 12:
        flags.append("too_few_2nd_points")
    if bpn < 3:
        flags.append("bp_sample_too_small")

    # Votes (adaptive quorum) with eps
    votes_home = 0
    votes_away = 0
    votes_total = 0
    eps_ssw = 0.03
    eps_fsw = 0.03
    eps_fsin = 0.03
    eps_ace = 0.015
    eps_dfr = 0.005
    eps_bpsr = 0.10

    if ssw_home is not None and ssw_away is not None:
        if ssw_home > ssw_away + eps_ssw:
            votes_home += 1
        elif ssw_away > ssw_home + eps_ssw:
            votes_away += 1
        votes_total += 1
        explain.append(f"SSW: home={ssw_home:.3f} away={ssw_away:.3f} (n2={ssn})")

    if fsw_home is not None and fsw_away is not None and fsn >= 20:
        if fsw_home > fsw_away + eps_fsw:
            votes_home += 1
        elif fsw_away > fsw_home + eps_fsw:
            votes_away += 1
        votes_total += 1
        explain.append(f"FSW: home={fsw_home:.3f} away={fsw_away:.3f} (n1={fsn})")
    else:
        flags.append("fsw_missing_or_small")

    if fsinr_home is not None and fsinr_away is not None and fsin_n >= 20:
        if fsinr_home > fsinr_away + eps_fsin:
            votes_home += 1
        elif fsinr_away > fsinr_home + eps_fsin:
            votes_away += 1
        votes_total += 1
        explain.append(f"FSIN: home={fsinr_home:.3f} away={fsinr_away:.3f} (nFS={fsin_n})")
    else:
        flags.append("fsin_missing_or_small")

    if dfr_home is not None and dfr_away is not None and serve_pts_n > 0:
        if dfr_home < dfr_away - eps_dfr:
            votes_home += 1
        elif dfr_away < dfr_home - eps_dfr:
            votes_away += 1
        votes_total += 1
        explain.append(f"DFR: home={dfr_home:.3f} away={dfr_away:.3f} (nSP={serve_pts_n})")
    else:
        flags.append("dfr_missing")

    if ace_rate_home is not None and ace_rate_away is not None and serve_pts_n >= 50:
        if ace_rate_home > ace_rate_away + eps_ace:
            votes_home += 1
        elif ace_rate_away > ace_rate_home + eps_ace:
            votes_away += 1
        votes_total += 1
        explain.append(f"ACE: home={aces_home} away={aces_away} (rate {ace_rate_home:.3f}/{ace_rate_away:.3f})")
    else:
        flags.append("ace_missing_or_small")

    if bpsr_home is not None and bpsr_away is not None and bpn >= 3:
        if bpsr_home > bpsr_away + eps_bpsr:
            votes_home += 1
        elif bpsr_away > bpsr_home + eps_bpsr:
            votes_away += 1
        votes_total += 1
        explain.append(f"BPSR: home={bpsr_home:.3f} away={bpsr_away:.3f} (nBP={bpn})")
    else:
        flags.append("bpsr_missing_or_small")

    side = "neutral"
    quorum = 2
    if votes_total >= 4:
        quorum = 3
    if votes_home >= quorum and votes_home > votes_away:
        side = "home"
    elif votes_away >= quorum and votes_away > votes_home:
        side = "away"

    # Strength components
    strength = 0
    side_before_strength = side
    if side != "neutral":
        # SSW strength
        diff_ssw = abs(ssw_home - ssw_away)
        strength_ssw = 0
        if ssn >= 30:
            strength_ssw = 3 if diff_ssw >= 0.08 else 2 if diff_ssw >= 0.05 else 1 if diff_ssw >= 0.03 else 0
        elif ssn >= 12:
            strength_ssw = 2 if diff_ssw >= 0.07 else 1 if diff_ssw >= 0.04 else 0

        strength_fsw = 0
        if fsw_home is not None and fsw_away is not None and fsn >= 20:
            diff_fsw = abs(fsw_home - fsw_away)
            if fsn >= 60:
                strength_fsw = 3 if diff_fsw >= 0.07 else 2 if diff_fsw >= 0.05 else 1 if diff_fsw >= 0.03 else 0
            else:
                strength_fsw = 2 if diff_fsw >= 0.07 else 1 if diff_fsw >= 0.04 else 0

        strength_df = 0
        if dfr_home is not None and dfr_away is not None and serve_pts_n > 0:
            diff_df = abs(dfr_home - dfr_away)
            if serve_pts_n >= 60:
                strength_df = 3 if diff_df >= 0.015 else 2 if diff_df >= 0.010 else 1 if diff_df >= 0.006 else 0
            else:
                strength_df = 2 if diff_df >= 0.020 else 1 if diff_df >= 0.012 else 0

        strength_ace = 0
        if ace_rate_home is not None and ace_rate_away is not None and serve_pts_n >= 50:
            diff_ace = abs(ace_rate_home - ace_rate_away)
            strength_ace = 2 if diff_ace >= 0.030 else 1 if diff_ace >= 0.018 else 0

        strength_bp = 0
        if bpsr_home is not None and bpsr_away is not None and bpn >= 3:
            diff_bp = abs(bpsr_home - bpsr_away)
            strength_bp = 2 if diff_bp >= 0.25 else 1 if diff_bp >= 0.15 else 0

        strengths: List[int] = []
        if side == "home":
            if ssw_home > ssw_away + eps_ssw:
                strengths.append(strength_ssw)
            if fsw_home is not None and fsw_away is not None and fsw_home > fsw_away + eps_fsw:
                strengths.append(strength_fsw)
            if dfr_home is not None and dfr_away is not None and dfr_home < dfr_away - eps_dfr:
                strengths.append(strength_df)
            if ace_rate_home is not None and ace_rate_away is not None and ace_rate_home > ace_rate_away + eps_ace:
                strengths.append(strength_ace)
            if bpsr_home is not None and bpsr_away is not None and bpsr_home > bpsr_away + eps_bpsr:
                strengths.append(strength_bp)
        else:
            if ssw_away > ssw_home + eps_ssw:
                strengths.append(strength_ssw)
            if fsw_home is not None and fsw_away is not None and fsw_away > fsw_home + eps_fsw:
                strengths.append(strength_fsw)
            if dfr_home is not None and dfr_away is not None and dfr_away < dfr_home - eps_dfr:
                strengths.append(strength_df)
            if ace_rate_home is not None and ace_rate_away is not None and ace_rate_away > ace_rate_home + eps_ace:
                strengths.append(strength_ace)
            if bpsr_home is not None and bpsr_away is not None and bpsr_away > bpsr_home + eps_bpsr:
                strengths.append(strength_bp)

        strength = max(strengths) if strengths else 0
        if ssn < 12:
            strength = min(strength, 1)

    if strength == 0 and side_before_strength != "neutral":
        side = "neutral"
        flags.append("side_reset_strength0")

    # Composite stability score (home-positive): higher is better.
    # - SSW and BPSR are in [0,1], higher better.
    # - DFR is small; scale to 0..1 where higher = worse, then subtract.
    if ssw_home is not None and ssw_away is not None:
        dfr_scaled_home = _clamp01((dfr_home or 0.0) / 0.08) if dfr_home is not None else None
        dfr_scaled_away = _clamp01((dfr_away or 0.0) / 0.08) if dfr_away is not None else None
        w_ssw = 0.38 * _reliability(ssn, full=30)
        w_fsw = 0.32 * _reliability(int(fsn), full=60) if fsw_home is not None and fsw_away is not None else 0.0
        w_dfr = 0.15 * _reliability(int(serve_pts_n), full=80) if serve_pts_n else 0.0
        w_ace = 0.05 * _reliability(int(serve_pts_n), full=80) if ace_rate_home is not None and ace_rate_away is not None else 0.0
        w_bps = 0.10 * _reliability(int(bpn), full=10) if bpn else 0.0
        comp_home = _weighted_mean(
            [
                (ssw_home, w_ssw),
                (fsw_home, w_fsw),
                ((1.0 - dfr_scaled_home) if dfr_scaled_home is not None else None, w_dfr),
                (ace_rate_home, w_ace),
                (bpsr_home, w_bps),
            ]
        )
        comp_away = _weighted_mean(
            [
                (ssw_away, w_ssw),
                (fsw_away, w_fsw),
                ((1.0 - dfr_scaled_away) if dfr_scaled_away is not None else None, w_dfr),
                (ace_rate_away, w_ace),
                (bpsr_away, w_bps),
            ]
        )
        if comp_home is not None and comp_away is not None:
            comp_diff = comp_home - comp_away
            explain.append(f"CompositeServe={comp_diff:+.3f}")
            # Composite adjustments cannot exceed data-driven caps.
            max_cap = 3
            if ssn < 30:
                max_cap = min(max_cap, 2)
            if "bp_sample_too_small" in flags:
                max_cap = min(max_cap, 2)
            if side != "neutral" and strength in (1, 2) and abs(comp_diff) >= 0.08 and (w_ssw + w_fsw + w_dfr + w_ace + w_bps) >= 0.35:
                if (side == "home" and comp_diff > 0) or (side == "away" and comp_diff < 0):
                    strength = min(max_cap, strength + 1)
                    flags.append("composite_boost")
                else:
                    strength = max(0, strength - 1)
                    flags.append("composite_penalty")

    # Per-set trend (2ND serve stability): if side is set, compare set2 vs set1 SSW.
    if side != "neutral":
        ss1h, ss1a = snapshot.ratio(periods=("1ST",), group="Service", item="Second serve points")
        ss2h, ss2a = snapshot.ratio(periods=("2ND",), group="Service", item="Second serve points")
        if ss1h and ss1a and ss2h and ss2a and min(ss1h.total, ss1a.total, ss2h.total, ss2a.total) >= 10:
            d1 = (ss1h.rate or 0.0) - (ss1a.rate or 0.0)
            d2 = (ss2h.rate or 0.0) - (ss2a.rate or 0.0)
            trend = d2 - d1
            if abs(trend) >= 0.08 and strength in (1, 2):
                if (side == "home" and trend > 0) or (side == "away" and trend < 0):
                    strength = min(3, strength + 1)
                    flags.append("trend_boost")
                    explain.append(f"TrendSSW={trend:+.3f} -> +1")
                else:
                    strength = max(0, strength - 1)
                    flags.append("trend_penalty")
                    explain.append(f"TrendSSW={trend:+.3f} -> -1")

    # last-N calibration: compare current SSW deviation vs player's norm (set1+2 combined)
    if side != "neutral" and strength in (1, 2) and cal_home and cal_away:
        cur_home = ssw_home
        cur_away = ssw_away
        dh = deviation(cur_home, cal_home.get("ssw_12", MetricSummary(0, None, None)))
        da = deviation(cur_away, cal_away.get("ssw_12", MetricSummary(0, None, None)))
        if dh is not None and da is not None and cal_home.get("ssw_12").n >= 10 and cal_away.get("ssw_12").n >= 10:
            diff = dh - da
            if abs(diff) >= 0.05:
                if (side == "home" and diff > 0) or (side == "away" and diff < 0):
                    strength = min(3, strength + 1)
                    flags.append("cal_boost")
                    explain.append(f"CalSSW diff={diff:+.3f} -> +1")
                else:
                    strength = max(0, strength - 1)
                    flags.append("cal_penalty")
                    explain.append(f"CalSSW diff={diff:+.3f} -> -1")

    explain.append(f"votes: home={votes_home} away={votes_away} -> {side}")
    explain.append(f"FINAL: {side} strength={strength}")
    return ModuleResult("M2_second_serve", side, _cap_strength(strength), explain[:10], flags)


def module3_return_pressure(*, snapshot: MatchSnapshot, cal_home: Optional[Dict[str, MetricSummary]] = None, cal_away: Optional[Dict[str, MetricSummary]] = None) -> ModuleResult:
    flags: List[str] = []
    explain: List[str] = []
    periods = ("1ST", "2ND")

    # Return games played
    rg_home, rg_away = snapshot.value(periods=periods, group="Return", item="Return games played")
    # Service games won (to derive breaks)
    sgw_home, sgw_away = snapshot.value(periods=periods, group="Games", item="Service games won")
    rb_home = max(0, rg_home - sgw_away)
    rb_away = max(0, rg_away - sgw_home)

    # Return points vs 1st/2nd serve
    r1_home, r1_away = snapshot.ratio(periods=periods, group="Return", item="First serve return points")
    r2_home, r2_away = snapshot.ratio(periods=periods, group="Return", item="Second serve return points")

    # BP converted (count) + optional normalized conversion rate via opponent BP total
    bpc_home, bpc_away = snapshot.value(periods=periods, group="Return", item="Break points converted")
    bps_home, bps_away = snapshot.ratio(periods=periods, group="Service", item="Break points saved")
    bpconv_home = None
    bpconv_away = None
    bp_tot_home = 0
    bp_tot_away = 0
    if bps_home is not None and bps_away is not None:
        bp_tot_home = int(bps_away.total)
        bp_tot_away = int(bps_home.total)
        if bp_tot_home > 0 and 0 <= bpc_home <= bp_tot_home:
            bpconv_home = bpc_home / bp_tot_home
        if bp_tot_away > 0 and 0 <= bpc_away <= bp_tot_away:
            bpconv_away = bpc_away / bp_tot_away
        if (bp_tot_home > 0 and bpconv_home is None) or (bp_tot_away > 0 and bpconv_away is None):
            flags.append("bpc_inconsistent")
    else:
        flags.append("bp_total_missing_for_conv")

    min_rg = min(rg_home, rg_away)
    if min_rg < 8:
        flags.append("too_few_return_games")

    rbr_home = (rb_home / rg_home) if rg_home > 0 else None
    rbr_away = (rb_away / rg_away) if rg_away > 0 else None
    explain.append(f"RBR: home={rb_home}/{rg_home} away={rb_away}/{rg_away}")

    # RPR
    if r1_home is None or r1_away is None or r2_home is None or r2_away is None:
        return ModuleResult("M3_return_pressure", "neutral", 0, ["missing Return points (R1/R2)"], ["return_points_missing"])
    else:
        r1r_home = r1_home.rate or 0.0
        r1r_away = r1_away.rate or 0.0
        r2r_home = r2_home.rate or 0.0
        r2r_away = r2_away.rate or 0.0
        r2_tot_min = min(r2_home.total, r2_away.total)
        if r2_tot_min < 20:
            flags.append("return2_sample_small")
        rpr_home = 0.4 * r1r_home + 0.6 * r2r_home
        rpr_away = 0.4 * r1r_away + 0.6 * r2r_away
        rtot = min((r1_home.total + r2_home.total), (r1_away.total + r2_away.total))
        explain.append(
            f"RPR: home={rpr_home:.3f} (R1={r1r_home:.3f}, R2={r2r_home:.3f}) "
            f"away={rpr_away:.3f} (R1={r1r_away:.3f}, R2={r2r_away:.3f})"
        )

    eps_rbr = 0.05
    eps_rpr = 0.03
    eps_bpconv = 0.18

    votes_home = 0
    votes_away = 0
    votes_total = 0
    if rbr_home is not None and rbr_away is not None:
        if rbr_home > rbr_away + eps_rbr:
            votes_home += 1
        elif rbr_away > rbr_home + eps_rbr:
            votes_away += 1
        votes_total += 1

    if rpr_home is not None and rpr_away is not None:
        if rpr_home > rpr_away + eps_rpr:
            votes_home += 1
        elif rpr_away > rpr_home + eps_rpr:
            votes_away += 1
        votes_total += 1

    if bpconv_home is not None and bpconv_away is not None and min(bp_tot_home, bp_tot_away) >= 4:
        explain.append(f"BPconv: home={bpc_home}/{bp_tot_home}={bpconv_home:.2f} away={bpc_away}/{bp_tot_away}={bpconv_away:.2f}")
        if bpconv_home > bpconv_away + eps_bpconv:
            votes_home += 1
        elif bpconv_away > bpconv_home + eps_bpconv:
            votes_away += 1
        votes_total += 1
        flags.append("bpconv_support_used")
    else:
        flags.append("bpconv_missing_or_small")

    side = "neutral"
    quorum = 2
    if votes_total >= 4:
        quorum = 3
    if votes_home >= quorum and votes_home > votes_away:
        side = "home"
    elif votes_away >= quorum and votes_away > votes_home:
        side = "away"

    strength = 0
    side_before_strength = side
    if side != "neutral":
        # Strength RBR
        strength_rbr = 0
        if min_rg >= 16 and rbr_home is not None and rbr_away is not None:
            d = abs(rbr_home - rbr_away)
            strength_rbr = 3 if d >= 0.20 else 2 if d >= 0.12 else 1 if d >= 0.07 else 0
        elif min_rg >= 8 and rbr_home is not None and rbr_away is not None:
            d = abs(rbr_home - rbr_away)
            strength_rbr = 2 if d >= 0.18 else 1 if d >= 0.10 else 0

        # Strength RPR
        strength_rpr = 0
        if rpr_home is not None and rpr_away is not None and rtot >= 80:
            d = abs(rpr_home - rpr_away)
            strength_rpr = 3 if d >= 0.06 else 2 if d >= 0.04 else 1 if d >= 0.03 else 0
        elif rpr_home is not None and rpr_away is not None and rtot >= 40:
            d = abs(rpr_home - rpr_away)
            strength_rpr = 2 if d >= 0.06 else 1 if d >= 0.04 else 0

        strength_bpconv = 0
        if bpconv_home is not None and bpconv_away is not None and min(bp_tot_home, bp_tot_away) >= 5:
            d = abs(bpconv_home - bpconv_away)
            strength_bpconv = 2 if d >= 0.35 else 1 if d >= 0.20 else 0

        strengths: List[int] = []
        if side == "home":
            if rbr_home is not None and rbr_away is not None and rbr_home > rbr_away + eps_rbr:
                strengths.append(strength_rbr)
            if rpr_home is not None and rpr_away is not None and rpr_home > rpr_away + eps_rpr:
                strengths.append(strength_rpr)
            if bpconv_home is not None and bpconv_away is not None and bpconv_home > bpconv_away + eps_bpconv:
                strengths.append(strength_bpconv)
        else:
            if rbr_home is not None and rbr_away is not None and rbr_away > rbr_home + eps_rbr:
                strengths.append(strength_rbr)
            if rpr_home is not None and rpr_away is not None and rpr_away > rpr_home + eps_rpr:
                strengths.append(strength_rpr)
            if bpconv_home is not None and bpconv_away is not None and bpconv_away > bpconv_home + eps_bpconv:
                strengths.append(strength_bpconv)

        strength = max(strengths) if strengths else 0

        # High break environment cap
        if rbr_home is not None and rbr_away is not None:
            if (rbr_home + rbr_away) / 2 >= 0.40:
                flags.append("high_break_environment")
                strength = min(strength, 2)

        if min_rg < 8:
            strength = min(strength, 1)

    if strength == 0 and side_before_strength != "neutral":
        side = "neutral"
        flags.append("side_reset_strength0")

    # Composite return pressure (home-positive): combine RPR/RBR (+ BPconv if available).
    if rpr_home is not None and rpr_away is not None and rbr_home is not None and rbr_away is not None:
        w_rpr = 0.55 * _reliability(int(rtot), full=80) if rtot else 0.0
        w_rbr = 0.30 * _reliability(int(min_rg), full=16) if min_rg else 0.0
        w_bpc = 0.15 * _reliability(int(min(bp_tot_home, bp_tot_away)), full=12) if bpconv_home is not None and bpconv_away is not None else 0.0
        comp_home = _weighted_mean([(rpr_home, w_rpr), (rbr_home, w_rbr), (bpconv_home, w_bpc)])
        comp_away = _weighted_mean([(rpr_away, w_rpr), (rbr_away, w_rbr), (bpconv_away, w_bpc)])
        if comp_home is not None and comp_away is not None:
            comp_diff = comp_home - comp_away
            explain.append(f"CompositeReturn={comp_diff:+.3f}")
            if side != "neutral" and strength in (1, 2) and abs(comp_diff) >= 0.07 and (w_rpr + w_rbr + w_bpc) >= 0.35:
                if (side == "home" and comp_diff > 0) or (side == "away" and comp_diff < 0):
                    strength = min(3, strength + 1)
                    flags.append("composite_boost")
                else:
                    strength = max(0, strength - 1)
                    flags.append("composite_penalty")

    # Per-set trend (1ST -> 2ND) if available: only affects strength when side is set.
    if side != "neutral":
        # Compute RPR per set if possible
        def rpr_for(period: str) -> Optional[Tuple[float, int]]:
            r1h, r1a = snapshot.ratio(periods=(period,), group="Return", item="First serve return points")
            r2h, r2a = snapshot.ratio(periods=(period,), group="Return", item="Second serve return points")
            if r1h is None or r1a is None or r2h is None or r2a is None:
                return None
            r1rh = r1h.rate
            r1ra = r1a.rate
            if r1rh is None or r1ra is None:
                return None
            if min(r2h.total, r2a.total) < 10:
                return None
            r2rh = r2h.rate
            r2ra = r2a.rate
            if r2rh is None or r2ra is None:
                return None
            rprh = 0.4 * r1rh + 0.6 * r2rh
            rpra = 0.4 * r1ra + 0.6 * r2ra
            return rprh - rpra, (r1h.total + r2h.total + r1a.total + r2a.total)

        t1 = rpr_for("1ST")
        t2 = rpr_for("2ND")
        if t1 and t2 and t1[1] >= 40 and t2[1] >= 40:
            # Trend in home advantage on return points
            trend = t2[0] - t1[0]
            if abs(trend) >= 0.06:
                # If trend moves towards the chosen side, boost, else penalize.
                if (side == "home" and trend > 0) or (side == "away" and trend < 0):
                    strength = min(3, strength + 1)
                    flags.append("trend_boost")
                    explain.append(f"TrendRPR={trend:+.3f} -> +1")
                else:
                    strength = max(0, strength - 1)
                    flags.append("trend_penalty")
                    explain.append(f"TrendRPR={trend:+.3f} -> -1")

    # last-N calibration: deviation on RPR (set1+2 combined)
    if side != "neutral" and strength in (1, 2) and cal_home and cal_away and rpr_home is not None and rpr_away is not None:
        dh = deviation(rpr_home, cal_home.get("rpr_12", MetricSummary(0, None, None)))
        da = deviation(rpr_away, cal_away.get("rpr_12", MetricSummary(0, None, None)))
        if dh is not None and da is not None and cal_home.get("rpr_12").n >= 10 and cal_away.get("rpr_12").n >= 10:
            diff = dh - da
            if abs(diff) >= 0.04:
                if (side == "home" and diff > 0) or (side == "away" and diff < 0):
                    strength = min(3, strength + 1)
                    flags.append("cal_boost")
                    explain.append(f"CalRPR diff={diff:+.3f} -> +1")
                else:
                    strength = max(0, strength - 1)
                    flags.append("cal_penalty")
                    explain.append(f"CalRPR diff={diff:+.3f} -> -1")

    explain.append(f"votes: home={votes_home} away={votes_away} -> {side}")
    explain.append(f"FINAL: {side} strength={strength}")
    return ModuleResult("M3_return_pressure", side, _cap_strength(strength), explain[:10], flags)


def module4_clutch(*, snapshot: MatchSnapshot, cal_home: Optional[Dict[str, MetricSummary]] = None, cal_away: Optional[Dict[str, MetricSummary]] = None) -> ModuleResult:
    flags: List[str] = []
    explain: List[str] = []
    periods = ("1ST", "2ND")

    bps_home, bps_away = snapshot.ratio(periods=periods, group="Service", item="Break points saved")
    bpc_home, bpc_away = snapshot.value(periods=periods, group="Return", item="Break points converted")
    tb_home, tb_away = snapshot.value(periods=periods, group="Miscellaneous", item="Tiebreaks")
    tiebreak_context = (tb_home + tb_away) >= 1
    if tiebreak_context:
        flags.append("tiebreak_context")

    if bps_home is None or bps_away is None:
        return ModuleResult("M4_clutch", "neutral", 0, ["BPS missing"], ["bps_missing"])

    bpsr_home = bps_home.rate
    bpsr_away = bps_away.rate
    bpn_home = bps_home.total
    bpn_away = bps_away.total
    bpn = min(bpn_home, bpn_away)
    if bpn < 3:
        flags.append("bp_sample_too_small")

    # Reconstruct BP totals and conversion rates
    bptot_home = bpn_away
    bptot_away = bpn_home
    bpconv_home = (bpc_home / bptot_home) if bptot_home > 0 else None
    bpconv_away = (bpc_away / bptot_away) if bptot_away > 0 else None
    if bpconv_home is None or bpconv_away is None:
        flags.append("bpconv_missing_or_small")
    else:
        if bpc_home > bptot_home or bpc_away > bptot_away:
            flags.append("bpc_inconsistent")
            bpconv_home = None
            bpconv_away = None
            flags.append("bpconv_missing_or_small")

    explain.append(
        f"DEF(BPS): home={bps_home.won}/{bps_home.total}={bpsr_home:.2f} away={bps_away.won}/{bps_away.total}={bpsr_away:.2f} (nBP={bpn})"
    )
    if bpconv_home is not None and bpconv_away is not None:
        explain.append(
            f"OFF(BPconv): home={bpc_home}/{bptot_home}={bpconv_home:.2f} away={bpc_away}/{bptot_away}={bpconv_away:.2f} (nBPtot={min(bptot_home, bptot_away)})"
        )

    # Adaptive eps: clutch is noisy at low BP counts.
    eps_def = 0.20 if bpn < 5 else 0.12
    eps_off = 0.25 if bpn < 5 else 0.15
    vote_def = "neutral"
    if bpsr_home is not None and bpsr_away is not None and abs(bpsr_home - bpsr_away) >= eps_def:
        vote_def = "home" if bpsr_home > bpsr_away else "away"

    vote_off = "neutral"
    if bpconv_home is not None and bpconv_away is not None and abs(bpconv_home - bpconv_away) >= eps_off:
        vote_off = "home" if bpconv_home > bpconv_away else "away"

    side = "neutral"
    if vote_def != "neutral" and vote_off != "neutral":
        if vote_def == vote_off:
            side = vote_def
        else:
            flags.append("clutch_conflict")
    elif vote_def != "neutral":
        side = vote_def
        flags.append("clutch_side_by_def")
    elif vote_off != "neutral":
        side = vote_off
        flags.append("clutch_side_by_off")

    strength_def = 0
    g_def = abs((bpsr_home or 0.0) - (bpsr_away or 0.0))
    if bpn >= 10:
        strength_def = 3 if g_def >= 0.30 else 2 if g_def >= 0.20 else 1 if g_def >= 0.12 else 0
    elif bpn >= 5:
        strength_def = 2 if g_def >= 0.30 else 1 if g_def >= 0.18 else 0
    elif bpn >= 3:
        strength_def = 1 if g_def >= 0.30 else 0

    strength_off = 0
    if bpconv_home is not None and bpconv_away is not None:
        g_off = abs(bpconv_home - bpconv_away)
        if bpn >= 10:
            strength_off = 3 if g_off >= 0.35 else 2 if g_off >= 0.25 else 1 if g_off >= 0.15 else 0
        elif bpn >= 5:
            strength_off = 2 if g_off >= 0.35 else 1 if g_off >= 0.20 else 0
        elif bpn >= 3:
            strength_off = 1 if g_off >= 0.35 else 0

    strength = 0
    side_before_strength = side
    if side != "neutral":
        strengths: List[int] = []
        if side == "home":
            if vote_def == "home":
                strengths.append(strength_def)
            if vote_off == "home":
                strengths.append(strength_off)
        else:
            if vote_def == "away":
                strengths.append(strength_def)
            if vote_off == "away":
                strengths.append(strength_off)
        strength = max(strengths) if strengths else 0
        if "bp_sample_too_small" in flags:
            strength = min(strength, 1)

    if strength == 0 and side_before_strength != "neutral":
        side = "neutral"
        flags.append("side_reset_strength0")

    # Composite clutch (home-positive): average defense/offense if available.
    if bpsr_home is not None and bpsr_away is not None:
        w = _reliability(int(bpn), full=10) if bpn else 0.0
        w_def = 0.65 * w
        w_off = 0.35 * w if bpconv_home is not None and bpconv_away is not None else 0.0
        comp_home = _weighted_mean([(bpsr_home, w_def), (bpconv_home, w_off)])
        comp_away = _weighted_mean([(bpsr_away, w_def), (bpconv_away, w_off)])
        if comp_home is not None and comp_away is not None:
            comp_diff = comp_home - comp_away
            explain.append(f"CompositeClutch={comp_diff:+.3f}")
            if side != "neutral" and strength in (1, 2) and abs(comp_diff) >= 0.14 and w >= 0.5:
                if (side == "home" and comp_diff > 0) or (side == "away" and comp_diff < 0):
                    strength = min(3, strength + 1)
                    flags.append("composite_boost")
                else:
                    strength = max(0, strength - 1)
                    flags.append("composite_penalty")

    # Per-set trend: if clutch improved/worsened from set1 to set2 in the chosen direction, small adjustment.
    if side != "neutral" and strength in (1, 2):
        bps1h, bps1a = snapshot.ratio(periods=("1ST",), group="Service", item="Break points saved")
        bps2h, bps2a = snapshot.ratio(periods=("2ND",), group="Service", item="Break points saved")
        # Only if both sets have at least 2 BPs each side (avoid noise).
        if bps1h and bps1a and bps2h and bps2a and min(bps1h.total, bps1a.total, bps2h.total, bps2a.total) >= 4:
            d1 = (bps1h.rate or 0.0) - (bps1a.rate or 0.0)
            d2 = (bps2h.rate or 0.0) - (bps2a.rate or 0.0)
            trend = d2 - d1
            if abs(trend) >= 0.35:
                if (side == "home" and trend > 0) or (side == "away" and trend < 0):
                    strength = min(3, strength + 1)
                    flags.append("trend_boost")
                    explain.append(f"TrendBPSR={trend:+.2f} -> +1")
                else:
                    strength = max(0, strength - 1)
                    flags.append("trend_penalty")
                    explain.append(f"TrendBPSR={trend:+.2f} -> -1")

    # last-N calibration: BPSR and BPconv deviations (set1+2 combined)
    if side != "neutral" and strength in (1, 2) and cal_home and cal_away:
        def _get_rate(bps: Optional[Ratio]) -> Optional[float]:
            return bps.rate if bps is not None else None

        dh_def = deviation(_get_rate(bps_home), cal_home.get("bpsr_12", MetricSummary(0, None, None)))
        da_def = deviation(_get_rate(bps_away), cal_away.get("bpsr_12", MetricSummary(0, None, None)))
        # BPconv current
        dh_off = deviation(bpconv_home, cal_home.get("bpconv_12", MetricSummary(0, None, None))) if bpconv_home is not None else None
        da_off = deviation(bpconv_away, cal_away.get("bpconv_12", MetricSummary(0, None, None))) if bpconv_away is not None else None

        boost = 0
        if dh_def is not None and da_def is not None and cal_home.get("bpsr_12").n >= 10 and cal_away.get("bpsr_12").n >= 10:
            diff = dh_def - da_def
            if abs(diff) >= 0.15:
                boost += 1 if ((side == "home" and diff > 0) or (side == "away" and diff < 0)) else -1
                explain.append(f"CalBPSR diff={diff:+.2f}")
        if dh_off is not None and da_off is not None and cal_home.get("bpconv_12").n >= 10 and cal_away.get("bpconv_12").n >= 10:
            diff = dh_off - da_off
            if abs(diff) >= 0.20:
                boost += 1 if ((side == "home" and diff > 0) or (side == "away" and diff < 0)) else -1
                explain.append(f"CalBPconv diff={diff:+.2f}")

        if boost >= 1:
            strength = min(3, strength + 1)
            flags.append("cal_boost")
        elif boost <= -1:
            strength = max(0, strength - 1)
            flags.append("cal_penalty")

    explain.append(f"votes: def={vote_def} off={vote_off} -> {side}")
    explain.append(f"FINAL: {side} strength={strength}")
    return ModuleResult("M4_clutch", side, _cap_strength(strength), explain[:10], flags)


def module5_form_profile(
    *,
    history_rows_home: List[Dict[str, Any]],
    history_rows_away: List[Dict[str, Any]],
    current_set2_winner: str,
    cal_home: Optional[Dict[str, MetricSummary]] = None,
    cal_away: Optional[Dict[str, MetricSummary]] = None,
) -> ModuleResult:
    flags: List[str] = []
    explain: List[str] = []

    n = min(len(history_rows_home), len(history_rows_away))
    # User default history is small (3–5); this module must still produce a weak
    # signal when possible, but stay cautious about small denominators.
    if n < 3:
        flags.append("history_insufficient")
        return ModuleResult("M5_form_profile", "neutral", 0, ["history_insufficient"], flags)

    def _count(rows: List[Dict[str, Any]]):
        r = rows[:n]
        wins = sum(1 for x in r if x.get("won") is True)
        return wins, len(r)

    def _count_deciders(rows: List[Dict[str, Any]]):
        r = [x for x in rows[:n] if x.get("is_bo3_decider")]
        wins = sum(1 for x in r if x.get("won") is True)
        return wins, len(r)

    def _count_deciders_by_set2(rows: List[Dict[str, Any]], won_set2: bool):
        r = [
            x
            for x in rows[:n]
            if x.get("is_bo3_decider") and bool(x.get("won_set2_known")) and bool(x.get("won_set2")) == bool(won_set2)
        ]
        wins = sum(1 for x in r if x.get("won") is True)
        return wins, len(r)

    def _rate(w: int, t: int, *, min_n: int) -> Optional[float]:
        if t < min_n or t <= 0:
            return None
        return w / t

    wh, th = _count(history_rows_home)
    wa, ta = _count(history_rows_away)
    win_rate_h = wh / th if th else 0.0
    win_rate_a = wa / ta if ta else 0.0

    dwh, dth = _count_deciders(history_rows_home)
    dwa, dta = _count_deciders(history_rows_away)
    dec_rate_h = _rate(dwh, dth, min_n=3)
    dec_rate_a = _rate(dwa, dta, min_n=3)

    d2wh, d2th = _count_deciders_by_set2(history_rows_home, True)
    d2wa, d2ta = _count_deciders_by_set2(history_rows_away, True)
    dec_after_set2_h = _rate(d2wh, d2th, min_n=3)
    dec_after_set2_a = _rate(d2wa, d2ta, min_n=3)

    explain.append(f"History last{n}: wins first={wh}/{th} second={wa}/{ta}")
    if dec_rate_h is not None or dec_rate_a is not None:
        dhs = f"{dec_rate_h:.2f}" if dec_rate_h is not None else "NA"
        das = f"{dec_rate_a:.2f}" if dec_rate_a is not None else "NA"
        explain.append(f"Deciders: first={dwh}/{dth} ({dhs}) second={dwa}/{dta} ({das})")
    else:
        flags.append("deciders_insufficient")

    set3_vote = "neutral"
    set3_strength = 0
    set3_index_h = None
    set3_index_a = None
    if cal_home and cal_away:
        ssw3h = cal_home.get("ssw_3")
        ssw3a = cal_away.get("ssw_3")
        rpr3h = cal_home.get("rpr_3")
        rpr3a = cal_away.get("rpr_3")
        bps3h = cal_home.get("bpsr_3")
        bps3a = cal_away.get("bpsr_3")
        bpc3h = cal_home.get("bpconv_3")
        bpc3a = cal_away.get("bpconv_3")

        set3_index_h = _weighted_mean(
            [
                (rpr3h.mean if rpr3h else None, 0.35),
                (ssw3h.mean if ssw3h else None, 0.25),
                (bps3h.mean if bps3h else None, 0.25),
                (bpc3h.mean if bpc3h else None, 0.15),
            ]
        )
        set3_index_a = _weighted_mean(
            [
                (rpr3a.mean if rpr3a else None, 0.35),
                (ssw3a.mean if ssw3a else None, 0.25),
                (bps3a.mean if bps3a else None, 0.25),
                (bpc3a.mean if bpc3a else None, 0.15),
            ]
        )
        n3 = min(
            (ssw3h.n if ssw3h else 0),
            (ssw3a.n if ssw3a else 0),
            (rpr3h.n if rpr3h else 0),
            (rpr3a.n if rpr3a else 0),
        )
        if set3_index_h is not None and set3_index_a is not None and n3 >= 3:
            diff = set3_index_h - set3_index_a
            if abs(diff) >= 0.05:
                set3_vote = "home" if diff > 0 else "away"
                set3_strength = 2 if abs(diff) >= 0.08 else 1
        else:
            flags.append("set3_metrics_insufficient")
        s3h = f"{set3_index_h:.3f}" if set3_index_h is not None else "NA"
        s3a = f"{set3_index_a:.3f}" if set3_index_a is not None else "NA"
        explain.append(f"Set3Index: first={s3h} second={s3a} (n3≈{n3})")
    else:
        flags.append("history_metrics_missing")

    votes_home = 0
    votes_away = 0
    votes_total = 0
    strength = 0

    diff_form = win_rate_h - win_rate_a
    form_vote = "neutral"
    form_strength = 0
    # With small N, one match swings the rate a lot; require ~1/3 gap to speak.
    if abs(diff_form) >= (0.33 if n <= 4 else 0.40):
        form_vote = "home" if diff_form > 0 else "away"
        form_strength = 2 if abs(diff_form) >= 0.60 and n >= 5 else 1
    if form_vote == "home":
        votes_home += 1
        strength = max(strength, form_strength)
    elif form_vote == "away":
        votes_away += 1
        strength = max(strength, form_strength)
    votes_total += 1

    dec_vote = "neutral"
    dec_strength = 0
    if dec_rate_h is not None and dec_rate_a is not None:
        diff = dec_rate_h - dec_rate_a
        if abs(diff) >= 0.34:
            dec_vote = "home" if diff > 0 else "away"
            dec_strength = 2 if abs(diff) >= 0.50 and min(dth, dta) >= 5 else 1
        if dec_vote == "home":
            votes_home += 1
            strength = max(strength, dec_strength)
        elif dec_vote == "away":
            votes_away += 1
            strength = max(strength, dec_strength)
        votes_total += 1

    if set3_vote == "home":
        votes_home += 1
        strength = max(strength, set3_strength)
        votes_total += 1
    elif set3_vote == "away":
        votes_away += 1
        strength = max(strength, set3_strength)
        votes_total += 1

    if current_set2_winner in ("home", "away") and dec_after_set2_h is not None and dec_after_set2_a is not None:
        diff = dec_after_set2_h - dec_after_set2_a
        scen_vote = "home" if diff > 0 else "away"
        if abs(diff) >= 0.34 and scen_vote == current_set2_winner:
            if scen_vote == "home":
                votes_home += 1
            else:
                votes_away += 1
            strength = max(strength, 1)
            votes_total += 1
        explain.append(
            f"Set2->Decider: first={d2wh}/{d2th} ({(f'{dec_after_set2_h:.2f}' if dec_after_set2_h is not None else 'NA')}) "
            f"second={d2wa}/{d2ta} ({(f'{dec_after_set2_a:.2f}' if dec_after_set2_a is not None else 'NA')}) winner_set2={current_set2_winner}"
        )
    else:
        flags.append("set2_profile_insufficient")

    side = "neutral"
    quorum = 2 if votes_total >= 2 else 1
    if votes_home >= quorum and votes_home > votes_away:
        side = "home"
    elif votes_away >= quorum and votes_away > votes_home:
        side = "away"
    else:
        strength = 0

    explain.append(f"votes: form={form_vote} dec={dec_vote} set3={set3_vote} -> {side}")
    explain.append(f"FINAL: {side} strength={strength}")
    return ModuleResult("M5_form_profile", side, _cap_strength(strength), explain[:10], flags)
