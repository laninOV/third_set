from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Optional
from html import escape as _html_escape

from playwright.async_api import async_playwright

from third_set.analyzer import analyze_once, bet_decision
from third_set.sofascore import (
    SOFASCORE_TENNIS_URL,
    LiveEvent,
    discover_match_links,
    get_live_match_links,
    get_event,
    get_event_from_match_url_via_navigation,
    get_event_statistics,
    get_last_finished_singles_events,
    is_singles_event,
    parse_event_id_from_match_link,
    summarize_event_for_team,
)
from third_set.snapshot import MatchSnapshot
from third_set.tg_runtime import TelegramClient, get_telegram_config
from third_set.dom_stats import extract_statistics_dom


def _print_audit(snapshot: MatchSnapshot) -> None:
    items = [
        ("Points", "Service points won"),
        ("Points", "Receiver points won"),
        ("Service", "First serve points"),
        ("Service", "Second serve points"),
        ("Service", "Double faults"),
        ("Service", "Break points saved"),
        ("Return", "First serve return points"),
        ("Return", "Second serve return points"),
        ("Return", "Return games played"),
        ("Return", "Break points converted"),
        ("Games", "Service games won"),
        ("Miscellaneous", "Tiebreaks"),
    ]
    for per in ("1ST", "2ND"):
        print(f"  [{per}]")
        for group, name in items:
            raw = snapshot.raw(period=per, group=group, item=name)
            if not raw:
                continue
            home = raw.get("home")
            away = raw.get("away")
            print(f"    {group}/{name}: home={home} away={away}")


def _print_missing_reason(*, meta: dict) -> None:
    """
    Explain why some current-match fields ended up missing (—/NA).
    Uses DOM diagnostics captured by extract_statistics_dom (_meta.seen/_meta.unmapped).
    """
    if not isinstance(meta, dict):
        return
    seen = meta.get("dom_seen")
    unmapped = meta.get("dom_unmapped")
    if not (isinstance(seen, dict) or isinstance(unmapped, dict)):
        return
    # Show a short hint per period and group.
    def _fmt_group(g: str) -> str:
        ru = {
            "Service": "Подача",
            "Points": "Очки",
            "Return": "Возврат",
            "Games": "Игр",
            "Miscellaneous": "Разное",
        }
        return ru.get(g, g)

    # Keep it short: one line per period with the top unmapped labels.
    print("  Почему есть '—' (диагностика DOM, кратко):")
    for per in ("1ST", "2ND"):
        u_per = unmapped.get(per) if isinstance(unmapped, dict) else None
        if not isinstance(u_per, dict) or not u_per:
            continue
        parts: List[str] = []
        for g in sorted(u_per.keys()):
            labs = u_per.get(g)
            if not isinstance(labs, list) or not labs:
                continue
            ex = ", ".join(str(x) for x in labs[:6])
            more = f"+{len(labs)-6}" if len(labs) > 6 else ""
            parts.append(f"{_fmt_group(g)}: {ex}{(' ' + more) if more else ''}")
        if parts:
            print(f"  - {per}: " + " | ".join(parts[:3]))


def _used_stats_subset(stats: dict, *, event_id: int) -> dict:
    snap = MatchSnapshot(event_id=event_id, stats=stats)
    items = [
        ("Points", "Service points won"),
        ("Points", "Receiver points won"),
        ("Service", "First serve"),
        ("Service", "First serve points"),
        ("Service", "Second serve points"),
        ("Service", "Aces"),
        ("Service", "Double faults"),
        ("Service", "Break points saved"),
        ("Return", "First serve return points"),
        ("Return", "Second serve return points"),
        ("Return", "Return games played"),
        ("Return", "Break points converted"),
        ("Games", "Service games won"),
        ("Miscellaneous", "Tiebreaks"),
    ]
    out: dict = {"eventId": event_id, "periods": {}}
    for per in ("ALL", "1ST", "2ND", "3RD"):
        per_items: dict = {}
        for group, name in items:
            raw = snap.raw(period=per, group=group, item=name)
            if not raw:
                continue
            per_items[f"{group}/{name}"] = {"home": raw.get("home"), "away": raw.get("away"), "homeValue": raw.get("homeValue"), "awayValue": raw.get("awayValue")}
        if per_items:
            out["periods"][per] = per_items
    return out


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _dump_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _format_tg_message(
    *,
    event_id: int,
    match_url: str,
    home_name: str,
    away_name: str,
    set2_winner: str,
    decision: str,
    decision_side: str,
    score: int,
    meta: dict,
    mods: list,
) -> str:
    # Small build stamp so we can confirm the bot runs the latest code.
    try:
        build_ts = int(Path(__file__).stat().st_mtime)
        build = time.strftime("%Y-%m-%d %H:%M", time.localtime(build_ts))
    except Exception:
        build = ""
    title = f"<b>{_html_escape(home_name)} vs {_html_escape(away_name)}</b>"
    link = f"<a href=\"{_html_escape(match_url)}\">Sofascore</a>"
    header = (
        f"{title}\n{link}\n<code>eventId={event_id}</code>"
        + (f"\n<code>build={_html_escape(build)}</code>" if build else "")
    )

    stats_mode = str(meta.get("stats_mode") or "")
    summ = meta.get("summary") if isinstance(meta, dict) else None
    feat0 = meta.get("features") if isinstance(meta, dict) else None
    hist0 = (feat0.get("history") if isinstance(feat0, dict) else None) if feat0 else None

    def _pct01(x: Any) -> Optional[int]:
        return int(round(float(x) * 100.0)) if isinstance(x, (int, float)) else None

    def _s100(v: Any) -> str:
        return str(int(round(float(v) * 100.0))) if isinstance(v, (int, float)) else "—"

    def _line_kv(k: str, v: str) -> str:
        return f"<b>{_html_escape(k)}</b>: {_html_escape(v)}"

    def _mode_ru(m: str) -> str:
        if m == "per_set":
            return "матч (1+2) + история"
        if m == "history_only":
            return "только история"
        return m or "—"

    lines: List[str] = [header]
    lines.append(_line_kv("Режим", _mode_ru(stats_mode)))
    if set2_winner in ("home", "away", "neutral"):
        lines.append(_line_kv("Победил 2-й сет", {"home": "1", "away": "2", "neutral": "—"}.get(set2_winner, "—")))

    def _pp(x: Any) -> Optional[float]:
        return float(x) * 100.0 if isinstance(x, (int, float)) else None

    ru_diff = {
        "dom": "Доминирование (TPW/DR)",
        "serve": "Подача (2-я/1-я/DF/эйсы/BP)",
        "return": "Приём (RPR/BPconv)",
        "clutch": "Клатч (BPSR/BPconv)",
        "form": "История TPW(1+2) (rating5)",
    }

    if isinstance(summ, dict):
        p1 = _pct01(summ.get("p_home"))
        p2 = _pct01(summ.get("p_away"))
        conf = _pct01(summ.get("confidence"))
        leader = "нейтр"
        if isinstance(p1, int) and isinstance(p2, int):
            leader = "1" if p1 > p2 else "2" if p2 > p1 else "нейтр"
        if isinstance(p1, int) and isinstance(p2, int) and isinstance(conf, int):
            lines.append(f"<b>Прогноз</b>: 1={p1}%  2={p2}%  | уверенность={conf}%  | лидер={leader}")

        diffs = summ.get("idx_diffs") if isinstance(summ.get("idx_diffs"), dict) else {}
        contrib: List[tuple] = []
        for k, v in diffs.items():
            if k not in ru_diff:
                continue
            pp = _pp(v)
            if pp is None:
                continue
            contrib.append((k, pp))
        contrib.sort(key=lambda kv: abs(kv[1]), reverse=True)
        if contrib:
            parts: List[str] = []
            for k, pp in contrib[:2]:
                if k == "form":
                    # Prefer explicit Δrating5 for readability.
                    try:
                        tpw = (hist0.get("tpw12_scores") or {}) if isinstance(hist0, dict) else {}
                        r5h = (((tpw.get("home") or {}).get("last5") or {}).get("rating") if isinstance(tpw.get("home"), dict) else None)
                        r5a = (((tpw.get("away") or {}).get("last5") or {}).get("rating") if isinstance(tpw.get("away"), dict) else None)
                        if isinstance(r5h, (int, float)) and isinstance(r5a, (int, float)):
                            d = int(round(float(r5h) - float(r5a)))
                            win = "1" if d > 0 else "2" if d < 0 else "нейтр"
                            parts.append(f"Δrating5={d:+d} (в пользу {win})")
                            continue
                    except Exception:
                        pass
                parts.append(f"{ru_diff.get(k, k)}={pp:+.1f}пп")
            if parts:
                lines.append("<b>Причины (топ)</b>: " + "; ".join(parts))

    vh = meta.get("votes_home") if isinstance(meta, dict) else None
    va = meta.get("votes_away") if isinstance(meta, dict) else None
    active = meta.get("active") if isinstance(meta, dict) else None
    strong_h = meta.get("strong_home") if isinstance(meta, dict) else None
    strong_a = meta.get("strong_away") if isinstance(meta, dict) else None
    if isinstance(vh, int) and isinstance(va, int):
        vote_winner = "1" if vh > va else "2" if va > vh else "нейтр"
        line = f"<b>Голоса моделей</b>: 1={vh}  2={va}  → победа={vote_winner}"
        if isinstance(active, int):
            line += f" (активных={active})"
        if isinstance(strong_h, int) and isinstance(strong_a, int):
            line += f" | сильные: 1={strong_h} 2={strong_a}"
        lines.append(line)

    try:
        sv = meta.get("sofascore_votes") if isinstance(meta, dict) else None
        vote = (sv.get("vote") or {}) if isinstance(sv, dict) else None
        if isinstance(vote, dict):
            v1 = vote.get("vote1")
            v2 = vote.get("vote2")
            vx = vote.get("voteX")
            tot = sum(int(x) for x in (v1, v2, vx) if isinstance(x, int))
            if tot > 0 and isinstance(v1, int) and isinstance(v2, int):
                p1 = (v1 / tot) * 100.0
                p2 = (v2 / tot) * 100.0
                lines.append(f"<b>Sofascore голосование</b>: 1={p1:.0f}% 2={p2:.0f}% (всего {tot})")
    except Exception:
        pass

    feat = meta.get("features") if isinstance(meta, dict) else None
    if isinstance(feat, dict) and isinstance(summ, dict):
        sigs = summ.get("signals") if isinstance(summ.get("signals"), dict) else {}
        idx = feat.get("indices") if isinstance(feat.get("indices"), dict) else {}
        hist = feat.get("history") if isinstance(feat.get("history"), dict) else {}
        cal = (hist.get("calibration") or {}) if isinstance(hist, dict) else {}
        tpw_scores = (hist.get("tpw12_scores") or {}) if isinstance(hist, dict) else {}
        cur = feat.get("current") if isinstance(feat.get("current"), dict) else {}

        def _sig(side: str, key: str) -> Any:
            it = sigs.get(key)
            return it.get(side) if isinstance(it, dict) else None

        def _i100(key: str, side: str) -> str:
            it = idx.get(key)
            if not isinstance(it, dict):
                return "—"
            v = it.get(side)
            return str(int(round(float(v) * 100.0))) if isinstance(v, (int, float)) else "—"

        def _tpw_rating(side: str, window: str) -> str:
            s = (tpw_scores.get(side) or {}) if isinstance(tpw_scores.get(side), dict) else {}
            key = "last3" if window == "3" else "last5"
            it = s.get(key)
            if not isinstance(it, dict):
                return "50"
            r = it.get("rating")
            return str(int(round(float(r)))) if isinstance(r, (int, float)) else "50"

        def _tpw_trend(side: str) -> str:
            try:
                r3 = int(_tpw_rating(side, "3"))
                r5 = int(_tpw_rating(side, "5"))
                return f"{r3 - r5:+d}"
            except Exception:
                return "0"

        def _mods_line(name: str) -> str:
            # Per-module compact comparison (no walls of numbers).
            # Uses current metrics if available; otherwise history means.
            by_name = {m.name: m for m in (mods or []) if hasattr(m, "name")}
            m = by_name.get(name)
            if not m:
                return ""
            side_ru = {"home": "1", "away": "2", "neutral": "нейтр"}.get(m.side, "нейтр")
            strength = int(m.strength or 0)
            # Pull a couple of headline numbers.
            if name == "M1_dominance":
                pts = (cur.get("points") or {}) if isinstance(cur.get("points"), dict) else {}
                tpw = pts.get("tpw_home")
                dr = pts.get("dr")
                if isinstance(tpw, (int, float)) and isinstance(dr, (int, float)):
                    return f"M1 Доминирование: {side_ru} (сила {strength}) | доля очков={tpw*100:.1f}%  DR={dr:.3f}"
                # history fallback in analyzer prints rating5; reflect that too
                r5h = _tpw_rating("home", "5")
                r5a = _tpw_rating("away", "5")
                return f"M1 Доминирование (ист): {side_ru} (сила {strength}) | rating5={r5h}/{r5a}"
            if name == "M2_second_serve":
                s = (cur.get("serve") or {}) if isinstance(cur.get("serve"), dict) else {}
                ssw = (s.get("ssw") or {}) if isinstance(s.get("ssw"), dict) else {}
                dfr = ((s.get("df") or {}).get("dfr") or {}) if isinstance((s.get("df") or {}).get("dfr"), dict) else {}
                if isinstance(ssw.get("home"), (int, float)) and isinstance(ssw.get("away"), (int, float)):
                    return f"M2 Подача: {side_ru} (сила {strength}) | 2-я подача={ssw.get('home')*100:.0f}%/{ssw.get('away')*100:.0f}%  двойные={dfr.get('home',0)*100:.1f}%/{dfr.get('away',0)*100:.1f}%"
                return (
                    f"M2 Подача (ист): {side_ru} (сила {strength}) | "
                    f"SSW12={_hist_metric('home','ssw_12')}/{_hist_metric('away','ssw_12')} "
                    f"BPSR12={_hist_metric('home','bpsr_12')}/{_hist_metric('away','bpsr_12')}"
                )
            if name == "M3_return_pressure":
                r = (cur.get("return") or {}) if isinstance(cur.get("return"), dict) else {}
                rpr = (r.get("rpr") or {}) if isinstance(r.get("rpr"), dict) else {}
                bpconv = (r.get("bpconv") or {}) if isinstance(r.get("bpconv"), dict) else {}
                if isinstance(rpr.get("home"), (int, float)) and isinstance(rpr.get("away"), (int, float)):
                    return f"M3 Давление на приёме: {side_ru} (сила {strength}) | очки на приёме={rpr.get('home')*100:.0f}%/{rpr.get('away')*100:.0f}%  конверсия БП={bpconv.get('home',0)*100:.0f}%/{bpconv.get('away',0)*100:.0f}%"
                return (
                    f"M3 Давление на приёме (ист): {side_ru} (сила {strength}) | "
                    f"RPR12={_hist_metric('home','rpr_12')}/{_hist_metric('away','rpr_12')} "
                    f"BPconv12={_hist_metric('home','bpconv_12')}/{_hist_metric('away','bpconv_12')}"
                )
            if name == "M4_clutch":
                s = (cur.get("serve") or {}) if isinstance(cur.get("serve"), dict) else {}
                bps = (s.get("bps") or {}) if isinstance(s.get("bps"), dict) else {}
                bpsr_h = (bps.get("home") or {}).get("rate") if isinstance(bps.get("home"), dict) else None
                bpsr_a = (bps.get("away") or {}).get("rate") if isinstance(bps.get("away"), dict) else None
                r = (cur.get("return") or {}) if isinstance(cur.get("return"), dict) else {}
                bpconv = (r.get("bpconv") or {}) if isinstance(r.get("bpconv"), dict) else {}
                if isinstance(bpsr_h, (int, float)) and isinstance(bpsr_a, (int, float)):
                    return (
                        f"M4 Клатч: {side_ru} (сила {strength}) | "
                        f"спасение БП={bpsr_h*100:.0f}%/{bpsr_a*100:.0f}%  "
                        f"конверсия БП={bpconv.get('home',0)*100:.0f}%/{bpconv.get('away',0)*100:.0f}%"
                    )
                return (
                    f"M4 Клатч (ист): {side_ru} (сила {strength}) | "
                    f"BPSR12={_hist_metric('home','bpsr_12')}/{_hist_metric('away','bpsr_12')} "
                    f"BPconv12={_hist_metric('home','bpconv_12')}/{_hist_metric('away','bpconv_12')}"
                )
            if name == "M5_form_profile":
                return f"M5 Форма/сеты: {side_ru} (сила {strength}) | {_hist_sets('home')} vs {_hist_sets('away')}"
            return f"{name}: {side_ru} (сила {strength})"

        def _hist_metric(side: str, key: str) -> str:
            src = cal.get(side) if isinstance(cal, dict) else None
            if not isinstance(src, dict):
                return "—[n=0]"
            ms = src.get(key)
            if not isinstance(ms, dict):
                return "—[n=0]"
            m = ms.get("mean")
            n = ms.get("n")
            if not isinstance(m, (int, float)) or not isinstance(n, int) or n <= 0:
                return "—[n=0]"
            return f"{float(m)*100:.0f}%[n={n}]"

        def _hist_sets(side: str) -> str:
            it = hist.get(side)
            if not isinstance(it, dict):
                return "—"
            s1 = it.get("set1") if isinstance(it.get("set1"), dict) else {}
            s2 = it.get("set2") if isinstance(it.get("set2"), dict) else {}
            d = it.get("dec_rate")
            dn = it.get("deciders")

            def _r(x: Any, n: Any) -> str:
                if not isinstance(x, (int, float)) or not isinstance(n, int) or n <= 0:
                    return "—"
                return f"{int(round(float(x) * 100.0))}%({n})"

            return f"1сет={_r(s1.get('rate'), s1.get('n'))} 2сет={_r(s2.get('rate'), s2.get('n'))} решающие={_r(d, dn)}"

        lines.append("")
        lines.append("<b>Таблица (история)</b>")

        def _clip(s: str, w: int) -> str:
            s = (s or "").strip()
            if len(s) <= w:
                return s
            if w <= 1:
                return s[:w]
            return s[: max(0, w - 1)] + "…"

        def _col(s: str, w: int) -> str:
            s = s or ""
            if len(s) >= w:
                return s[:w]
            return s + (" " * (w - len(s)))

        n1 = _clip(home_name, 16)
        n2 = _clip(away_name, 16)
        col1 = max(10, len("1) " + n1))
        col2 = max(10, len("2) " + n2))
        colm = 18

        def row(metric: str, v1: str, v2: str) -> str:
            return f"{_col(metric, colm)} {_col(v1, col1)} {_col(v2, col2)}"

        t: List[str] = []
        t.append(row("метрика", "1) " + n1, "2) " + n2))
        t.append(row("сила last5", _tpw_rating("home", "5"), _tpw_rating("away", "5")))
        t.append(row("сила last3", _tpw_rating("home", "3"), _tpw_rating("away", "3")))
        t.append(row("динамика 3-5", _tpw_trend("home"), _tpw_trend("away")))
        t.append(row("стабильность", _s100(_sig("home", "stability")), _s100(_sig("away", "stability"))))
        t.append(row("подача SSW12", _hist_metric("home", "ssw_12"), _hist_metric("away", "ssw_12")))
        t.append(row("приём RPR12", _hist_metric("home", "rpr_12"), _hist_metric("away", "rpr_12")))
        t.append(row("клатч BPSR12", _hist_metric("home", "bpsr_12"), _hist_metric("away", "bpsr_12")))
        t.append(row("клатч BPc12", _hist_metric("home", "bpconv_12"), _hist_metric("away", "bpconv_12")))

        def _set_rate(side: str, key: str) -> str:
            it = hist.get(side)
            if not isinstance(it, dict):
                return "—"
            s = it.get(key) if isinstance(it.get(key), dict) else {}
            r = s.get("rate")
            n = s.get("n")
            if not isinstance(r, (int, float)) or not isinstance(n, int) or n <= 0:
                return "—"
            return f"{int(round(float(r)*100))}%({n})"

        def _dec_rate(side: str) -> str:
            it = hist.get(side)
            if not isinstance(it, dict):
                return "—"
            r = it.get("dec_rate")
            n = it.get("deciders")
            if not isinstance(r, (int, float)) or not isinstance(n, int) or n <= 0:
                return "—"
            return f"{int(round(float(r)*100))}%({n})"

        t.append(row("сет1 WR", _set_rate("home", "set1"), _set_rate("away", "set1")))
        t.append(row("сет2 WR", _set_rate("home", "set2"), _set_rate("away", "set2")))
        t.append(row("решающие WR", _dec_rate("home"), _dec_rate("away")))
        lines.append("<code>" + "\n".join(_html_escape(x) for x in t) + "</code>")

        lines.append("")
        lines.append("<b>Модели</b>")

        def _side_ru(s: str) -> str:
            return {"home": "1", "away": "2", "neutral": "нейтр"}.get(s, "нейтр")

        mod_tbl: List[str] = []
        mod_tbl.append(row("модель", "победа/сила", "ключевые"))
        for name, ru in [
            ("M1_dominance", "M1 доминирование"),
            ("M2_second_serve", "M2 подача"),
            ("M3_return_pressure", "M3 приём"),
            ("M4_clutch", "M4 клатч"),
            ("M5_form_profile", "M5 форма"),
        ]:
            by_name = {m.name: m for m in (mods or []) if hasattr(m, "name")}
            m = by_name.get(name)
            if not m:
                continue
            left = f"{_side_ru(m.side)}/{int(m.strength or 0)}"
            key = ""
            if name == "M1_dominance":
                key = f"rating5={_tpw_rating('home','5')}/{_tpw_rating('away','5')}"
            elif name == "M2_second_serve":
                key = f"SSW12={_hist_metric('home','ssw_12')}/{_hist_metric('away','ssw_12')}"
            elif name == "M3_return_pressure":
                key = f"RPR12={_hist_metric('home','rpr_12')}/{_hist_metric('away','rpr_12')}"
            elif name == "M4_clutch":
                key = f"BPSR12={_hist_metric('home','bpsr_12')}/{_hist_metric('away','bpsr_12')}"
            elif name == "M5_form_profile":
                key = f"сеты: {_hist_sets('home')} vs {_hist_sets('away')}"
            mod_tbl.append(row(_clip(ru, colm), left, _clip(key, col1 + col2 + 1)))
        lines.append("<code>" + "\n".join(_html_escape(x) for x in mod_tbl) + "</code>")

    return "\n".join(lines)[:3800]


def _print_summary(meta: dict) -> None:
    s = meta.get("summary") if isinstance(meta, dict) else None
    if not isinstance(s, dict):
        return
    ph = s.get("p_home")
    pa = s.get("p_away")
    conf = s.get("confidence")
    if not isinstance(ph, (int, float)) or not isinstance(pa, (int, float)) or not isinstance(conf, (int, float)):
        return
    top = s.get("top") or []
    top_str = ""
    if isinstance(top, list) and top:
        parts = []
        for it in top[:2]:
            if not isinstance(it, dict):
                continue
            k = it.get("key")
            v = it.get("value")
            if isinstance(k, str) and isinstance(v, (int, float)):
                parts.append(f"{k}={v:+.3f}")
        if parts:
            top_str = " | top: " + ", ".join(parts)
    print(f"SUMMARY: P(first)={ph:.2f} P(second)={pa:.2f} conf={conf:.2f}{top_str}")

def _print_numbers(
    ctx,
    meta: dict,
    *,
    show_current: bool = True,
    show_history: bool = True,
    focus: bool = False,
) -> None:
    """
    Compact numeric view:
    - probabilities + index diffs
    - key current metrics (if features are present)
    """
    s = meta.get("summary") if isinstance(meta, dict) else None
    try:
        print(f"PLAYERS: 1={ctx.home_name} | 2={ctx.away_name}")
    except Exception:
        pass
    if isinstance(s, dict) and isinstance(s.get("p_home"), (int, float)) and isinstance(s.get("p_away"), (int, float)):
        diffs = s.get("idx_diffs") if isinstance(s.get("idx_diffs"), dict) else {}
        sigs = s.get("signals") if isinstance(s.get("signals"), dict) else {}
        def _fmt(x, *, nd: int = 3, signed: bool = True):
            if x is None:
                return "NA"
            try:
                v = float(x)
            except Exception:
                return "NA"
            return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
        def _sig(name: str):
            it = sigs.get(name) if isinstance(sigs, dict) else None
            if not isinstance(it, dict):
                return ("NA", "NA", "NA")
            return (it.get("home"), it.get("away"), it.get("diff"))
        sh, sa, sd = _sig("strength")
        fh, fa, fd = _sig("form")
        sth, sta, std = _sig("stability")

        def _to100(x) -> str:
            if x is None:
                return "NA"
            try:
                return f"{round(float(x) * 100):.0f}"
            except Exception:
                return "NA"

        def _pp(x) -> str:
            if x is None:
                return "NA"
            try:
                return f"{float(x) * 100:+.1f}pp"
            except Exception:
                return "NA"

        print(
            "NORM (0..100, 50=равно): "
            f"P={_to100(s['p_home'])}/{_to100(s['p_away'])} conf={_to100(float(s.get('confidence', 0)))} | "
            f"сила(история)={_to100(sh)}/{_to100(sa)} (Δ{_pp(sd)}) | "
            f"форма(lastN)={_to100(fh)}/{_to100(fa)} (Δ{_pp(fd)}) | "
            f"стаб(история)={_to100(sth)}/{_to100(sta)} (Δ{_pp(std)})"
        )

        def _idx100(key: str) -> str:
            v = diffs.get(key)
            if not isinstance(v, (int, float)):
                return "NA"
            return f"{float(v) * 100:+.1f}pp"

        print(
            "IDX (вклад, pp): "
            f"доминирование={_idx100('dom')} подача={_idx100('serve')} приём={_idx100('return')} "
            f"клатч={_idx100('clutch')} история={_idx100('form')}"
        )
        # Intentionally skip TOP contributors in default numeric view (too noisy).

    feat = meta.get("features") if isinstance(meta, dict) else None
    if not isinstance(feat, dict):
        return
    def f(x, nd=3):
        if x is None:
            return "NA"
        if isinstance(x, (int, float)):
            return f"{float(x):.{nd}f}"
        return "NA"

    # Quality summary (sample sizes / availability) - always useful.
    try:
        feat = meta.get("features") if isinstance(meta, dict) else None
        if isinstance(feat, dict):
            cur = feat.get("current") or {}
            pts = cur.get("points") or {}
            srv = cur.get("serve") or {}
            ret = cur.get("return") or {}
            hist = feat.get("history") or {}
            hh = hist.get("home") if isinstance(hist, dict) else None
            n_hist = (hh.get("n") if isinstance(hh, dict) else None)
            nPts = pts.get("n")
            n2min = (srv.get("ssw") or {}).get("n2_min")
            bp_tot_h = (ret.get("bp_tot") or {}).get("home")
            bp_tot_a = (ret.get("bp_tot") or {}).get("away")
            bpmin = min(int(bp_tot_h or 0), int(bp_tot_a or 0)) if (bp_tot_h is not None and bp_tot_a is not None) else None
            cal = (hist.get("calibration") or {}) if isinstance(hist, dict) else {}
            calh = cal.get("home") if isinstance(cal, dict) else None
            cala = cal.get("away") if isinstance(cal, dict) else None
            s3h = ((calh.get("ssw_3") or {}).get("n") if isinstance(calh, dict) and isinstance(calh.get("ssw_3"), dict) else None)
            s3a = ((cala.get("ssw_3") or {}).get("n") if isinstance(cala, dict) and isinstance(cala.get("ssw_3"), dict) else None)
            r3h = ((calh.get("rpr_3") or {}).get("n") if isinstance(calh, dict) and isinstance(calh.get("rpr_3"), dict) else None)
            r3a = ((cala.get("rpr_3") or {}).get("n") if isinstance(cala, dict) and isinstance(cala.get("rpr_3"), dict) else None)

            # History row metric coverage (how many recent matches had usable stats)
            rows = (hist.get("rows") or {}) if isinstance(hist, dict) else {}
            rh = rows.get("home") if isinstance(rows, dict) else None
            ra = rows.get("away") if isinstance(rows, dict) else None
            def _cov(rows_list, key: str) -> int:
                if not isinstance(rows_list, list):
                    return 0
                c = 0
                for r in rows_list:
                    if isinstance(r, dict) and r.get(key) is not None:
                        c += 1
                return c
            if isinstance(rh, list) and isinstance(ra, list):
                cov = (
                    f"tpw={_cov(rh,'tpw')}/{_cov(ra,'tpw')} "
                    f"ssw={_cov(rh,'ssw_12')}/{_cov(ra,'ssw_12')} "
                    f"rpr={_cov(rh,'rpr_12')}/{_cov(ra,'rpr_12')} "
                    f"bps={_cov(rh,'bpsr_12')}/{_cov(ra,'bpsr_12')} "
                    f"bpconv={_cov(rh,'bpconv_12')}/{_cov(ra,'bpconv_12')}"
                )
                # stats_ok can be true even if Sofascore returns minimal stats (missing_all_metrics=true)
                ok_h = sum(1 for r in rh if isinstance(r, dict) and r.get("stats_ok") is True)
                ok_a = sum(1 for r in ra if isinstance(r, dict) and r.get("stats_ok") is True)
                full_h = sum(1 for r in rh if isinstance(r, dict) and r.get("missing_all_metrics") is False)
                full_a = sum(1 for r in ra if isinstance(r, dict) and r.get("missing_all_metrics") is False)
                stats_line = f"stats_ok={ok_h}/{ok_a} stats_full={full_h}/{full_a}"
            else:
                cov = None
                stats_line = None
            print(
                "QUALITY: "
                f"nPts={nPts} n2_min={n2min} BPmin={bpmin} histN={n_hist} "
                f"hist_set3(ssw)={s3h}/{s3a} hist_set3(rpr)={r3h}/{r3a}"
                + (f" hist_cov[{cov}]" if cov else "")
                + (f" {stats_line}" if stats_line else "")
            )
    except Exception:
        pass

    if focus:
        # Focus mode ends here; remaining lines are intentionally suppressed.
        return

    if show_current:
        cur = feat.get("current") or {}
        pts = cur.get("points") or {}
        srv = cur.get("serve") or {}
        ret = cur.get("return") or {}

        tpw = pts.get("tpw_home")
        dr = pts.get("dr")
        npts = pts.get("n")

        ssw_h = (srv.get("ssw") or {}).get("home")
        ssw_a = (srv.get("ssw") or {}).get("away")
        fsw_h = (srv.get("fsw") or {}).get("home")
        fsw_a = (srv.get("fsw") or {}).get("away")
        fsin_h = (srv.get("fsin") or {}).get("home")
        fsin_a = (srv.get("fsin") or {}).get("away")
        dfr_h = (srv.get("df") or {}).get("dfr", {}).get("home")
        dfr_a = (srv.get("df") or {}).get("dfr", {}).get("away")
        ace_h = (srv.get("aces") or {}).get("ace_rate", {}).get("home")
        ace_a = (srv.get("aces") or {}).get("ace_rate", {}).get("away")
        bpsr_h = (srv.get("bps") or {}).get("home", {}).get("rate") if isinstance((srv.get("bps") or {}).get("home"), dict) else None
        bpsr_a = (srv.get("bps") or {}).get("away", {}).get("rate") if isinstance((srv.get("bps") or {}).get("away"), dict) else None

        rpr_h = (ret.get("rpr") or {}).get("home")
        rpr_a = (ret.get("rpr") or {}).get("away")
        bpconv_h = (ret.get("bpconv") or {}).get("home")
        bpconv_a = (ret.get("bpconv") or {}).get("away")
        rg_h = (ret.get("rg") or {}).get("home")
        rg_a = (ret.get("rg") or {}).get("away")
        bp_tot_h = (ret.get("bp_tot") or {}).get("home")
        bp_tot_a = (ret.get("bp_tot") or {}).get("away")

        n2min = (srv.get("ssw") or {}).get("n2_min")
        n1min = (srv.get("fsw") or {}).get("n1_min")
        nfsmin = (srv.get("fsin") or {}).get("nfs_min")
        bpmin = min(int(bp_tot_h or 0), int(bp_tot_a or 0)) if (bp_tot_h is not None and bp_tot_a is not None) else None

        def pct(x) -> str:
            if x is None:
                return "NA"
            try:
                return f"{float(x) * 100:.1f}%"
            except Exception:
                return "NA"

        print(
            "CUR: "
            f"TPW={pct(tpw)} DR={f(dr,3)} nPts={npts} | "
            f"SSW={pct(ssw_h)}/{pct(ssw_a)}(n2>={n2min}) "
            f"FSW={pct(fsw_h)}/{pct(fsw_a)}(n1>={n1min}) "
            f"FSIN={pct(fsin_h)}/{pct(fsin_a)}(nFS>={nfsmin}) "
            f"DFR={pct(dfr_h)}/{pct(dfr_a)} ACEr={pct(ace_h)}/{pct(ace_a)} "
            f"BPSR={pct(bpsr_h)}/{pct(bpsr_a)}(BPmin={bpmin}) | "
            f"RPR={pct(rpr_h)}/{pct(rpr_a)} BPconv={pct(bpconv_h)}/{pct(bpconv_a)} "
            f"(RG={rg_h}/{rg_a} BPtot={bp_tot_h}/{bp_tot_a})"
        )

        # Missing-data hints (why some metrics show NA)
        notes: List[str] = []
        try:
            if isinstance(ret, dict) and ret.get("sgw_present") is False:
                notes.append("RBR=NA (Sofascore не даёт 'Games→Service games won')")
        except Exception:
            pass
        try:
            cal = ((feat.get("history") or {}).get("calibration") or {}) if isinstance(feat.get("history"), dict) else {}
            calh = cal.get("home") if isinstance(cal, dict) else None
            cala = cal.get("away") if isinstance(cal, dict) else None
            if isinstance(calh, dict) and isinstance(cala, dict):
                n_h = int(((calh.get("ssw_3") or {}).get("n") or 0) if isinstance(calh.get("ssw_3"), dict) else 0)
                n_a = int(((cala.get("ssw_3") or {}).get("n") or 0) if isinstance(cala.get("ssw_3"), dict) else 0)
                if n_h == 0:
                    notes.append("history set3=NA для игрока 1 (нет per-set статов 3-го сета в lastN)")
                if n_a == 0:
                    notes.append("history set3=NA для игрока 2 (нет per-set статов 3-го сета в lastN)")
        except Exception:
            pass
        if notes:
            print("MISS: " + " | ".join(notes))

    if show_history:
        hist = feat.get("history") or {}
        hh = hist.get("home") if isinstance(hist, dict) else None
        ha = hist.get("away") if isinstance(hist, dict) else None
        if isinstance(hh, dict) and isinstance(ha, dict):
            n = hh.get("n")
            def pct2(x) -> str:
                if x is None:
                    return "NA"
                try:
                    return f"{float(x) * 100:.0f}%"
                except Exception:
                    return "NA"

            print(
                "HIST: "
                f"last{n} WR={pct2(hh.get('win_rate'))}/{pct2(ha.get('win_rate'))} "
                f"DecWR={pct2(hh.get('dec_rate'))}/{pct2(ha.get('dec_rate'))} "
                f"(W={hh.get('wins')}/{hh.get('n')} vs {ha.get('wins')}/{ha.get('n')}, "
                f"Dec={hh.get('dec_wins')}/{hh.get('deciders')} vs {ha.get('dec_wins')}/{ha.get('deciders')})"
            )


def _print_numbers_legend() -> None:
    print("LEGEND: Δ>0 = преимущество игрока 1. conf=уверенность 0..100. CUR/HIST — сырые проценты/счётчики.")

def _print_brief(ctx, meta: dict) -> None:
    s = meta.get("summary") if isinstance(meta, dict) else None
    if not isinstance(s, dict):
        return
    ph = s.get("p_home")
    pa = s.get("p_away")
    conf = s.get("confidence")
    if not isinstance(ph, (int, float)) or not isinstance(pa, (int, float)) or not isinstance(conf, (int, float)):
        return

    def pct(x: float) -> str:
        return f"{round(float(x) * 100):.0f}%"

    diffs = s.get("idx_diffs") if isinstance(s.get("idx_diffs"), dict) else {}
    ru = {
        "dom": "доминирование (очки/DR)",
        "serve": "подача (SSW/FSW/FSIN/DF/Ace/BP)",
        "return": "приём (RPR/BPconv)",
        "clutch": "клатч (BPSR/BPconv)",
        "form": "история TPW(1+2) (rating5)",
    }
    items = []
    for k, v in diffs.items():
        if k not in ru or not isinstance(v, (int, float)):
            continue
        items.append((k, float(v)))
    items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    reasons = []
    # If we have history rating5, show it as "ΔRating5=..." (human-readable).
    r5_home = None
    r5_away = None
    try:
        feat = meta.get("features") if isinstance(meta, dict) else None
        hist = (feat.get("history") or {}) if isinstance(feat, dict) else {}
        tpw = (hist.get("tpw12_scores") or {}) if isinstance(hist, dict) else {}
        r5_home = (((tpw.get("home") or {}).get("last5") or {}).get("rating") if isinstance(tpw.get("home"), dict) else None)
        r5_away = (((tpw.get("away") or {}).get("last5") or {}).get("rating") if isinstance(tpw.get("away"), dict) else None)
    except Exception:
        r5_home = r5_away = None
    for k, v in items[:2]:
        if k == "form" and isinstance(r5_home, (int, float)) and isinstance(r5_away, (int, float)):
            d = int(round(float(r5_home) - float(r5_away)))
            win = "1" if d > 0 else "2" if d < 0 else "нейтр"
            reasons.append(f"Δrating5={d:+d} (в пользу {win})")
        else:
            reasons.append(f"{ru[k]}={v*100:+.1f}пп")
    reason_str = (" | " + ", ".join(reasons)) if reasons else ""

    # Basis: show core contributors in a human way (no deep details).
    basis_parts = []
    try:
        sigs = s.get("signals") if isinstance(s.get("signals"), dict) else {}
        d_strength = (sigs.get("strength") or {}).get("diff") if isinstance(sigs.get("strength"), dict) else None
        d_form = (sigs.get("form") or {}).get("diff") if isinstance(sigs.get("form"), dict) else None
        d_stab = (sigs.get("stability") or {}).get("diff") if isinstance(sigs.get("stability"), dict) else None
        # Always show the three basis components; if missing, say why.
        def _pp_or_na(v, label: str) -> str:
            if isinstance(v, (int, float)):
                return f"{label}Δ={float(v)*100:+.1f}pp"
            return f"{label}Δ=NA"

        basis_parts.append(_pp_or_na(d_strength, "сила(история)"))
        basis_parts.append(_pp_or_na(d_form, "история TPW(1+2)"))
        basis_parts.append(_pp_or_na(d_stab, "стаб(история)"))

        mod_norm = s.get("mod_norm")
        active_mods = s.get("active_mods")
        if isinstance(mod_norm, (int, float)):
            if isinstance(active_mods, int):
                basis_parts.append(f"модули={float(mod_norm):+.2f} (активных={active_mods})")
            else:
                basis_parts.append(f"модули={float(mod_norm):+.2f}")
    except Exception:
        pass
    basis_str = (" | основание: " + ", ".join(basis_parts)) if basis_parts else ""

    # Quality: show what history coverage we actually had.
    q_parts = []
    set_parts = []
    tpw12_parts = []
    try:
        feat = meta.get("features") if isinstance(meta, dict) else None
        if isinstance(feat, dict):
            mode_stats = meta.get("stats_mode") if isinstance(meta, dict) else None
            cur = feat.get("current") or {}
            npts = ((cur.get("points") or {}).get("n"))
            n2min = (((cur.get("serve") or {}).get("ssw") or {}).get("n2_min"))
            bp_tot = (cur.get("return") or {}).get("bp_tot") if isinstance((cur.get("return") or {}), dict) else None
            bpmin = None
            if isinstance(bp_tot, dict):
                h = bp_tot.get("home")
                a = bp_tot.get("away")
                if h is not None and a is not None:
                    bpmin = min(int(h or 0), int(a or 0))
            hist = feat.get("history") or {}
            hh = hist.get("home") if isinstance(hist, dict) else None
            n_hist = hh.get("n") if isinstance(hh, dict) else None
            rows = (hist.get("rows") or {}) if isinstance(hist, dict) else {}
            rh = rows.get("home") if isinstance(rows, dict) else None
            ra = rows.get("away") if isinstance(rows, dict) else None
            if isinstance(rh, list) and isinstance(ra, list):
                full_h = sum(1 for r in rh if isinstance(r, dict) and r.get("missing_all_metrics") is False)
                full_a = sum(1 for r in ra if isinstance(r, dict) and r.get("missing_all_metrics") is False)
            else:
                full_h = full_a = None
            has_current = isinstance(npts, int) and npts > 0
            if mode_stats != "history_only":
                if has_current:
                    q_parts.append(f"сыгранных очков={npts}")
                # Treat 0/None as missing (Sofascore often omits these stats).
                if isinstance(n2min, int) and n2min > 0:
                    q_parts.append(f"объём 2-й подачи (min)={n2min}")
                if isinstance(bpmin, int) and bpmin > 0:
                    q_parts.append(f"брейк-пойнты (min)={bpmin}")
            if n_hist is not None:
                q_parts.append(f"история (матчей)={n_hist}")
            if full_h is not None and full_a is not None and n_hist is not None:
                q_parts.append(f"история со статой: 1={full_h}/{n_hist} 2={full_a}/{n_hist}")
            if mode_stats != "history_only":
                if not has_current:
                    # This is a DOM-parsing pipeline; missing current stats usually means our parser
                    # didn't find the required rows in the Statistics tab yet (UI lag / different layout).
                    err = meta.get("stats_error") if isinstance(meta, dict) else None
                    if isinstance(err, str) and err:
                        q_parts.append(f"текущая статистика=нет (ошибка парсера DOM: {err})")
                    else:
                        q_parts.append("текущая статистика=нет (парсер DOM не нашёл нужные поля)")

            # Historical set win-rates (lastN)
            hist = feat.get("history") or {}
            hh = hist.get("home") if isinstance(hist, dict) else None
            ha = hist.get("away") if isinstance(hist, dict) else None
            def _set_rate(hside: dict, key: str) -> str:
                if not isinstance(hside, dict):
                    return "NA"
                s = hside.get(key)
                if not isinstance(s, dict):
                    return "NA"
                r = s.get("rate")
                n = s.get("n")
                if not isinstance(r, (int, float)) or not isinstance(n, int) or n <= 0:
                    return "NA"
                return f"{r*100:.0f}%({n})"
            if isinstance(hh, dict) and isinstance(ha, dict):
                s1 = f"{_set_rate(hh,'set1')}/{_set_rate(ha,'set1')}"
                s2 = f"{_set_rate(hh,'set2')}/{_set_rate(ha,'set2')}"
                # set3 = decider match win-rate
                def _dec(hside: dict) -> str:
                    d = hside.get('dec_rate'); n = hside.get('deciders')
                    if not isinstance(d,(int,float)) or not isinstance(n,int) or n<=0: return 'NA'
                    return f"{d*100:.0f}%({n})"
                s3 = f"{_dec(hh)}/{_dec(ha)}"
                set_parts.append(f"сеты lastN: 1сет={s1} 2сет={s2} решающий={s3}")

            # TPW(1+2) history scores (0..100), always numeric.
            tpw12 = (hist.get("tpw12_scores") or {}) if isinstance(hist, dict) else {}
            if isinstance(tpw12, dict):
                h5 = (tpw12.get("home") or {}).get("last5")
                a5 = (tpw12.get("away") or {}).get("last5")
                h3 = (tpw12.get("home") or {}).get("last3")
                a3 = (tpw12.get("away") or {}).get("last3")

                def _g(sc: dict, k: str, d: float) -> float:
                    if not isinstance(sc, dict):
                        return float(d)
                    v = sc.get(k)
                    return float(v) if isinstance(v, (int, float)) else float(d)

                if isinstance(h5, dict) and isinstance(a5, dict):
                    tpw12_parts.append(
                        "история TPW(1+2): "
                        f"rating5={_g(h5,'rating',50):.0f}/{_g(a5,'rating',50):.0f} "
                        f"power5={_g(h5,'power',50):.0f}/{_g(a5,'power',50):.0f} "
                        f"form5={_g(h5,'form',50):.0f}/{_g(a5,'form',50):.0f} "
                        f"vol5={_g(h5,'volatility',50):.0f}/{_g(a5,'volatility',50):.0f}"
                    )
                if isinstance(h3, dict) and isinstance(a3, dict):
                    tpw12_parts.append(
                        "история TPW(1+2): "
                        f"rating3={_g(h3,'rating',50):.0f}/{_g(a3,'rating',50):.0f} "
                        f"power3={_g(h3,'power',50):.0f}/{_g(a3,'power',50):.0f} "
                        f"vol3={_g(h3,'volatility',50):.0f}/{_g(a3,'volatility',50):.0f}"
                    )
    except Exception:
        pass
    q_str = (" | q: " + " ".join(q_parts)) if q_parts else ""
    sets_str = (" | " + " ".join(set_parts)) if set_parts else ""

    leader = "1" if ph > pa else "2" if pa > ph else "нейтр"

    # Voting summary from modules (who "wins" by votes)
    vh = meta.get("votes_home") if isinstance(meta, dict) else None
    va = meta.get("votes_away") if isinstance(meta, dict) else None
    active = meta.get("active") if isinstance(meta, dict) else None
    strong_h = meta.get("strong_home") if isinstance(meta, dict) else None
    strong_a = meta.get("strong_away") if isinstance(meta, dict) else None
    vote_winner = "нейтр"
    if isinstance(vh, int) and isinstance(va, int):
        vote_winner = "1" if vh > va else "2" if va > vh else "нейтр"

    # Multi-line, easy-to-scan brief output (comparison-first).
    print(f"ПРОГНОЗ: матч={ctx.event_id}")
    print(f"  1={ctx.home_name}")
    print(f"  2={ctx.away_name}")
    print(f"  Вероятность: 1={pct(ph)} 2={pct(pa)} | уверенность={pct(conf)} | лидер={leader}")
    # Sofascore community votes (Кто победит?) — show right after probability (user-friendly).
    try:
        sv = meta.get("sofascore_votes") if isinstance(meta, dict) else None
        vote = (sv.get("vote") or {}) if isinstance(sv, dict) else None
        if isinstance(vote, dict):
            v1 = vote.get("vote1")
            v2 = vote.get("vote2")
            vx = vote.get("voteX")
            tot = sum(int(x) for x in (v1, v2, vx) if isinstance(x, int))
            if tot > 0 and isinstance(v1, int) and isinstance(v2, int):
                p1 = (v1 / tot) * 100.0
                p2 = (v2 / tot) * 100.0
                print(f"  Голосование Sofascore: 1={p1:.0f}% 2={p2:.0f}% (всего {tot})")
    except Exception:
        pass
    if reasons:
        print("  Причины (топ): " + "; ".join(reasons))
    if isinstance(vh, int) and isinstance(va, int):
        line = f"  Голосование моделей: 1={vh} 2={va} → победа={vote_winner}"
        if isinstance(active, int):
            line += f" (активных={active})"
        if isinstance(strong_h, int) and isinstance(strong_a, int):
            line += f" | сильные: 1={strong_h} 2={strong_a}"
        print(line)
    if q_parts:
        try:
            mode_stats = meta.get("stats_mode") if isinstance(meta, dict) else None
            if mode_stats == "per_set":
                q_parts.append("текущая статистика=по сетам (1-й+2-й)")
            elif mode_stats == "history_only":
                q_parts.append("текущая статистика=НЕ используется (только история)")
        except Exception:
            pass
        print("  Качество данных: " + "; ".join(q_parts))
    if set_parts:
        for sp in set_parts:
            print("  " + sp)
    if tpw12_parts:
        for sp in tpw12_parts:
            print("  " + sp)
    if basis_parts:
        print("  Основание: " + "; ".join(basis_parts))
    if getattr(ctx, "url", None):
        print(f"  Sofascore: {ctx.url}")

    # Module-by-module compact comparison (no walls of numbers).
    try:
        mods = meta.get("mods_compact") if isinstance(meta, dict) else None
        if isinstance(mods, list) and mods:
            feat = meta.get("features") if isinstance(meta, dict) else None
            cur = (feat.get("current") or {}) if isinstance(feat, dict) else {}
            pts = (cur.get("points") or {}) if isinstance(cur, dict) else {}
            srv = (cur.get("serve") or {}) if isinstance(cur, dict) else {}
            ret = (cur.get("return") or {}) if isinstance(cur, dict) else {}
            hist = (feat.get("history") or {}) if isinstance(feat, dict) else {}
            cal = (hist.get("calibration") or {}) if isinstance(hist, dict) else {}
            calh = (cal.get("home") or {}) if isinstance(cal, dict) else {}
            cala = (cal.get("away") or {}) if isinstance(cal, dict) else {}

            def _hist_mean(side: str, key: str) -> Optional[float]:
                src = calh if side == "home" else cala
                if not isinstance(src, dict):
                    return None
                ms = src.get(key)
                if not isinstance(ms, dict):
                    return None
                m = ms.get("mean")
                n = ms.get("n")
                if not isinstance(m, (int, float)) or not isinstance(n, int) or n <= 0:
                    return None
                return float(m)

            def _hist_pct(side: str, key: str) -> str:
                src = calh if side == "home" else cala
                if not isinstance(src, dict):
                    return "—"
                ms = src.get(key)
                if not isinstance(ms, dict):
                    return "—"
                m = ms.get("mean")
                n = ms.get("n")
                if not isinstance(m, (int, float)) or not isinstance(n, int) or n <= 0:
                    return "—"
                return f"{float(m)*100:.1f}%[n={n}]"

            def _pct(x):
                return "—" if not isinstance(x, (int, float)) else f"{float(x)*100:.1f}%"

            def _pct_pp(x):
                return "—" if not isinstance(x, (int, float)) else f"{float(x):+.1f}пп"

            def _line(name: str, side: str, strength: int, details: str) -> str:
                winner = "нейтр" if side == "neutral" or strength <= 0 else ("1" if side == "home" else "2")
                return f"  {name}: победа={winner} сила={strength} | {details}"

            print("  Модули (кто выше и почему):")
            for m in mods:
                if not isinstance(m, dict):
                    continue
                mname = str(m.get("name") or "")
                side = str(m.get("side") or "neutral")
                strength = int(m.get("strength") or 0)
                ru_name = _ru_module(mname)

                if mname == "M1_dominance":
                    tpw = pts.get("tpw_home")
                    dr = pts.get("dr")
                    if isinstance(tpw, (int, float)):
                        tpw1 = float(tpw)
                        tpw2 = 1.0 - tpw1
                        tpw_s = f"TPW={tpw1*100:.1f}%/{tpw2*100:.1f}% (Δ{(tpw1-tpw2)*100:+.1f}пп)"
                    else:
                        tpw_s = "TPW=NA"
                    dr_s = f"DR={float(dr):.3f}" if isinstance(dr, (int, float)) else "DR=NA"
                    print(_line(ru_name, side, strength, f"{tpw_s} {dr_s}"))
                    continue

                if mname == "M2_second_serve":
                    ssw = (srv.get("ssw") or {}) if isinstance(srv.get("ssw"), dict) else {}
                    dfr = (srv.get("df") or {}).get("dfr") if isinstance(srv.get("df"), dict) else None
                    bps = (srv.get("bps") or {}) if isinstance(srv.get("bps"), dict) else {}
                    if isinstance(ssw.get("home"), (int, float)) or isinstance(ssw.get("away"), (int, float)):
                        ssw_s = f"SSW={_pct(ssw.get('home'))}/{_pct(ssw.get('away'))}"
                    else:
                        ssw_s = f"SSW(hist)={_hist_pct('home','ssw_12')}/{_hist_pct('away','ssw_12')}"
                    dfr_s = "DFR=NA/NA"
                    if isinstance(dfr, dict):
                        dfr_s = f"DFR={_pct(dfr.get('home'))}/{_pct(dfr.get('away'))}"
                    if isinstance((bps.get("home") or {}).get("rate"), (int, float)) or isinstance((bps.get("away") or {}).get("rate"), (int, float)):
                        bps_s = f"BPSR={_pct((bps.get('home') or {}).get('rate'))}/{_pct((bps.get('away') or {}).get('rate'))}"
                    else:
                        bps_s = f"BPSR(hist)={_hist_pct('home','bpsr_12')}/{_hist_pct('away','bpsr_12')}"
                    print(_line(ru_name, side, strength, f"{ssw_s} {dfr_s} {bps_s}"))
                    continue

                if mname == "M3_return_pressure":
                    rpr = (ret.get("rpr") or {}) if isinstance(ret.get("rpr"), dict) else {}
                    bpconv = (ret.get("bpconv") or {}) if isinstance(ret.get("bpconv"), dict) else {}
                    rbr = (ret.get("rbr") or {}) if isinstance(ret.get("rbr"), dict) else {}
                    if isinstance(rpr.get("home"), (int, float)) or isinstance(rpr.get("away"), (int, float)):
                        rpr_s = f"RPR={_pct(rpr.get('home'))}/{_pct(rpr.get('away'))}"
                    else:
                        rpr_s = f"RPR(hist)={_hist_pct('home','rpr_12')}/{_hist_pct('away','rpr_12')}"
                    if isinstance(bpconv.get("home"), (int, float)) or isinstance(bpconv.get("away"), (int, float)):
                        bp_s = f"BPconv={_pct(bpconv.get('home'))}/{_pct(bpconv.get('away'))}"
                    else:
                        bp_s = f"BPconv(hist)={_hist_pct('home','bpconv_12')}/{_hist_pct('away','bpconv_12')}"
                    rbr_s = (
                        f"RBR={_pct(rbr.get('home'))}/{_pct(rbr.get('away'))}"
                        if (isinstance(rbr.get("home"), (int, float)) or isinstance(rbr.get("away"), (int, float)))
                        else "RBR=—"
                    )
                    # If everything is missing, don't print a wall of dashes.
                    if "—" in (rpr_s + bp_s + rbr_s) and ("[n=" not in (rpr_s + bp_s)) and "RPR=" not in rpr_s and "BPconv=" not in bp_s:
                        print(_line(ru_name, side, strength, "нет данных (ни в матче, ни в истории)"))
                    else:
                        print(_line(ru_name, side, strength, f"{rpr_s} {bp_s} {rbr_s}"))
                    continue

                if mname == "M4_clutch":
                    bps = (srv.get("bps") or {}) if isinstance(srv.get("bps"), dict) else {}
                    bpconv = (ret.get("bpconv") or {}) if isinstance(ret.get("bpconv"), dict) else {}
                    if isinstance((bps.get("home") or {}).get("rate"), (int, float)) or isinstance((bps.get("away") or {}).get("rate"), (int, float)):
                        bps_s = f"BPSR={_pct((bps.get('home') or {}).get('rate'))}/{_pct((bps.get('away') or {}).get('rate'))}"
                    else:
                        bps_s = f"BPSR(hist)={_hist_pct('home','bpsr_12')}/{_hist_pct('away','bpsr_12')}"
                    if isinstance(bpconv.get("home"), (int, float)) or isinstance(bpconv.get("away"), (int, float)):
                        bp_s = f"BPconv={_pct(bpconv.get('home'))}/{_pct(bpconv.get('away'))}"
                    else:
                        bp_s = f"BPconv(hist)={_hist_pct('home','bpconv_12')}/{_hist_pct('away','bpconv_12')}"
                    if "—" in (bps_s + bp_s) and ("[n=" not in (bps_s + bp_s)) and "BPSR=" not in bps_s and "BPconv=" not in bp_s:
                        print(_line(ru_name, side, strength, "нет данных (ни в матче, ни в истории)"))
                    else:
                        print(_line(ru_name, side, strength, f"{bps_s} {bp_s}"))
                    continue

                if mname == "M5_form_profile":
                    h = hist.get("home") if isinstance(hist, dict) else None
                    a = hist.get("away") if isinstance(hist, dict) else None
                    wr_s = f"WR={_pct((h or {}).get('win_rate'))}/{_pct((a or {}).get('win_rate'))}"
                    # Prefer explicit counts for deciders: it's clearer than NA.
                    dhn = (h or {}).get("deciders")
                    dhw = (h or {}).get("dec_wins")
                    dan = (a or {}).get("deciders")
                    daw = (a or {}).get("dec_wins")
                    if isinstance(dhn, int) and isinstance(dhw, int) and isinstance(dan, int) and isinstance(daw, int):
                        d_s = f"решающие={dhw}/{dhn} vs {daw}/{dan}"
                    else:
                        d_s = f"решающие={_pct((h or {}).get('dec_rate'))}/{_pct((a or {}).get('dec_rate'))}"
                    s1 = (h or {}).get("set1") if isinstance((h or {}).get("set1"), dict) else {}
                    s2 = (h or {}).get("set2") if isinstance((h or {}).get("set2"), dict) else {}
                    s1a = (a or {}).get("set1") if isinstance((a or {}).get("set1"), dict) else {}
                    s2a = (a or {}).get("set2") if isinstance((a or {}).get("set2"), dict) else {}
                    sets_s = f"сет1={_pct(s1.get('rate'))}/{_pct(s1a.get('rate'))} сет2={_pct(s2.get('rate'))}/{_pct(s2a.get('rate'))}"
                    print(_line(ru_name, side, strength, f"{wr_s} {d_s} {sets_s}"))
                    continue

                print(_line(ru_name, side, strength, "NA"))
    except Exception:
        pass

    # If current stats were attempted but some fields are missing, print a small DOM hint.
    try:
        mode_stats = meta.get("stats_mode") if isinstance(meta, dict) else None
        if (
            os.getenv("THIRDSET_EXPLAIN_DASH") in ("1", "true", "yes")
            and mode_stats == "per_set"
            and isinstance(meta.get("dom_unmapped"), dict)
        ):
            _print_missing_reason(meta=meta)
    except Exception:
        pass

    # Per-player compact calculations (only if we have features/signals available)
    feat = meta.get("features") if isinstance(meta, dict) else None
    if not isinstance(feat, dict):
        return

    # Comparison helper block: show which numbers are higher.
    try:
        hist = feat.get("history") if isinstance(feat.get("history"), dict) else {}
        tpw_scores = hist.get("tpw12_scores") if isinstance(hist, dict) else None
        tpw_scores = tpw_scores if isinstance(tpw_scores, dict) else {}
        last3 = tpw_scores.get("last3") if isinstance(tpw_scores.get("last3"), dict) else {}
        last5 = tpw_scores.get("last5") if isinstance(tpw_scores.get("last5"), dict) else {}

        def _num(v):
            return float(v) if isinstance(v, (int, float)) else None

        def _cmp(label: str, a, b, *, suffix: str = "") -> None:
            av = _num(a)
            bv = _num(b)
            if av is None and bv is None:
                return
            if av is None or bv is None:
                side = "1" if bv is None else "2" if av is None else "нейтр"
                print(f"  {label}: 1={av if av is not None else 'NA'} 2={bv if bv is not None else 'NA'} → {side}{suffix}")
                return
            if abs(av - bv) < 1e-9:
                win = "нейтр"
            else:
                win = "1" if av > bv else "2"
            delta = av - bv
            print(f"  {label}: 1={av:.0f} 2={bv:.0f} (Δ{delta:+.0f}) → {win}{suffix}")

        # Rating (history) and stability (signals) comparisons are easiest to interpret.
        try:
            st = (s.get("signals") or {}) if isinstance(s.get("signals"), dict) else {}
            stab = st.get("stability") if isinstance(st.get("stability"), dict) else {}
            stab_h = stab.get("home")
            stab_a = stab.get("away")
            if isinstance(stab_h, (int, float)) or isinstance(stab_a, (int, float)):
                _cmp("Стабильность (история, 0..100)", (stab_h or 0) * 100 if isinstance(stab_h, (int, float)) else None, (stab_a or 0) * 100 if isinstance(stab_a, (int, float)) else None)
        except Exception:
            pass
        _cmp("Форма (rating5, история TPW(1+2))", (last5.get("home") or {}).get("rating"), (last5.get("away") or {}).get("rating"))
        _cmp("Динамика (rating3, история TPW(1+2))", (last3.get("home") or {}).get("rating"), (last3.get("away") or {}).get("rating"))
    except Exception:
        pass

    def _p100(v) -> str:
        if not isinstance(v, (int, float)):
            return "NA"
        return f"{round(float(v) * 100):.0f}"

    sigs = s.get("signals") if isinstance(s.get("signals"), dict) else {}
    def _sig(side: str, key: str):
        it = sigs.get(key) if isinstance(sigs, dict) else None
        if not isinstance(it, dict):
            return None
        return it.get(side)

    # Indices (0..1, 0.5=baseline)
    idx = feat.get("indices") if isinstance(feat.get("indices"), dict) else {}
    def _idx(side: str, key: str):
        it = idx.get(key) if isinstance(idx, dict) else None
        if not isinstance(it, dict):
            return None
        return it.get(side)

    # History (set success rates)
    hist = feat.get("history") if isinstance(feat.get("history"), dict) else {}
    hh = hist.get("home") if isinstance(hist, dict) else None
    ha = hist.get("away") if isinstance(hist, dict) else None
    def _rate(side_dict, key: str):
        if not isinstance(side_dict, dict):
            return None, None
        x = side_dict.get(key)
        if not isinstance(x, dict):
            return None, None
        return x.get("rate"), x.get("n")

    s1h, n1h = _rate(hh, "set1")
    s1a, n1a = _rate(ha, "set1")
    s2h, n2h = _rate(hh, "set2")
    s2a, n2a = _rate(ha, "set2")
    dch = hh.get("dec_rate") if isinstance(hh, dict) else None
    dnh = hh.get("deciders") if isinstance(hh, dict) else None
    dca = ha.get("dec_rate") if isinstance(ha, dict) else None
    dna = ha.get("deciders") if isinstance(ha, dict) else None

    def _sr(r, n):
        if not isinstance(r, (int, float)) or not isinstance(n, int) or n <= 0:
            return "NA"
        return f"{r*100:.0f}%({n})"

    def _indices_line(side: str) -> str:
        dom = _idx(side, "dominance_index")
        srv = _idx(side, "serve_index")
        ret = _idx(side, "return_index")
        cl = _idx(side, "clutch_index")
        hf = _idx(side, "history_form_index")
        if all(v is None for v in (dom, srv, ret, cl, hf)):
            return "индексы: нет (нет текущей статистики по сетам)"
        return (
            f"индексы: дом={_p100(dom)} подача={_p100(srv)} приём={_p100(ret)} "
            f"клатч={_p100(cl)} история={_p100(hf)}"
        )

    # Print 2 short lines: one per player
    tpw12 = (hist.get("tpw12_scores") or {}) if isinstance(hist, dict) else {}
    h5 = (tpw12.get("home") or {}).get("last5") if isinstance(tpw12, dict) else None
    a5 = (tpw12.get("away") or {}).get("last5") if isinstance(tpw12, dict) else None
    h3 = (tpw12.get("home") or {}).get("last3") if isinstance(tpw12, dict) else None
    a3 = (tpw12.get("away") or {}).get("last3") if isinstance(tpw12, dict) else None

    def _tpw_line(side: str) -> str:
        if side == "home":
            sc5, sc3 = h5, h3
        else:
            sc5, sc3 = a5, a3
        if not isinstance(sc5, dict):
            return "TPW(1+2) история: NA"
        def g(sc: dict, k: str, d: float) -> float:
            v = sc.get(k)
            return float(v) if isinstance(v, (int, float)) else float(d)
        r5 = g(sc5, "rating", 50.0)
        p5 = g(sc5, "power", 50.0)
        f5 = g(sc5, "form", 50.0)
        v5 = g(sc5, "volatility", 50.0)
        r3 = g(sc3, "rating", 50.0) if isinstance(sc3, dict) else 50.0
        return f"TPW(1+2): rating3={r3:.0f} rating5={r5:.0f} power5={p5:.0f} form5={f5:.0f} vol5={v5:.0f}"
    print(
        "  ИГРОК1: "
        f"сила={_p100(_sig('home','strength'))} rating3={_p100(_sig('home','form3'))} rating5={_p100(_sig('home','form5'))} "
        f"стаб={_p100(_sig('home','stability'))} | "
        f"{_indices_line('home')} | "
        f"сеты(lastN): 1сет={_sr(s1h,n1h)} 2сет={_sr(s2h,n2h)} решающий={_sr(dch,dnh)} | {_tpw_line('home')}"
    )
    print(
        "  ИГРОК2: "
        f"сила={_p100(_sig('away','strength'))} rating3={_p100(_sig('away','form3'))} rating5={_p100(_sig('away','form5'))} "
        f"стаб={_p100(_sig('away','stability'))} | "
        f"{_indices_line('away')} | "
        f"сеты(lastN): 1сет={_sr(s1a,n1a)} 2сет={_sr(s2a,n2a)} решающий={_sr(dca,dna)} | {_tpw_line('away')}"
    )


def _print_models_numeric(meta: dict) -> None:
    feat = meta.get("features") if isinstance(meta, dict) else None
    if not isinstance(feat, dict):
        return
    cur = feat.get("current") or {}
    pts = cur.get("points") or {}
    srv = cur.get("serve") or {}
    ret = cur.get("return") or {}
    hist = feat.get("history") or {}
    hh = hist.get("home") if isinstance(hist, dict) else None
    ha = hist.get("away") if isinstance(hist, dict) else None
    cal = (hist.get("calibration") or {}) if isinstance(hist, dict) else {}
    calh = cal.get("home") if isinstance(cal, dict) else None
    cala = cal.get("away") if isinstance(cal, dict) else None

    def f(x, nd=3):
        if x is None:
            return "NA"
        if isinstance(x, (int, float)):
            return f"{float(x):.{nd}f}"
        return "NA"

    def pct(x, nd=1) -> str:
        if x is None:
            return "NA"
        if isinstance(x, (int, float)):
            return f"{float(x) * 100:.{nd}f}%"
        return "NA"

    # M1: dominance
    print(
        "M1: "
        f"TPW={pct(pts.get('tpw_home'))}/{pct((1-pts.get('tpw_home')) if isinstance(pts.get('tpw_home'), (int, float)) else None)} "
        f"DR={f(pts.get('dr'))}"
    )

    # M2: serve robustness (2nd/df/aces/bps)
    ssw = srv.get("ssw") or {}
    df = (srv.get("df") or {}).get("dfr") or {}
    aces = (srv.get("aces") or {}).get("ace_rate") or {}
    bps = srv.get("bps") or {}
    bps_h = (bps.get("home") or {}).get("rate") if isinstance(bps.get("home"), dict) else None
    bps_a = (bps.get("away") or {}).get("rate") if isinstance(bps.get("away"), dict) else None
    print(
        "M2: "
        f"SSW={pct(ssw.get('home'))}/{pct(ssw.get('away'))} "
        f"DFR={pct(df.get('home'))}/{pct(df.get('away'))} "
        f"ACEr={pct(aces.get('home'))}/{pct(aces.get('away'))} "
        f"BPSR={pct(bps_h)}/{pct(bps_a)}"
    )

    # M3: return pressure (rpr + bpconv + rbr if available)
    rpr = ret.get("rpr") or {}
    bpconv = ret.get("bpconv") or {}
    rbr = ret.get("rbr") or {}
    rbr_h = rbr.get("home") if isinstance(rbr, dict) else None
    rbr_a = rbr.get("away") if isinstance(rbr, dict) else None
    rbr_s = f"{f(rbr_h)}/{f(rbr_a)}" if (rbr_h is not None or rbr_a is not None) else "NA"
    print(
        "M3: "
        f"RPR={pct(rpr.get('home'))}/{pct(rpr.get('away'))} "
        f"BPconv={pct(bpconv.get('home'))}/{pct(bpconv.get('away'))} "
        f"RBR={rbr_s}"
    )

    # M4: clutch (bpsr + bpconv)
    tb = (srv.get("tb") or {})
    print(
        "M4: "
        f"BPSR={pct(bps_h)}/{pct(bps_a)} "
        f"BPconv={pct(bpconv.get('home'))}/{pct(bpconv.get('away'))} "
        f"TB={tb.get('home')}/{tb.get('away')}"
    )

    # M5: history/profile (last-N win/decider) + optional set3 calibration means
    if isinstance(hh, dict) and isinstance(ha, dict):
        print(
            "M5: "
            f"N={hh.get('n')} WR={f(hh.get('win_rate'),2)}/{f(ha.get('win_rate'),2)} "
            f"DecWR={f(hh.get('dec_rate'),2)}/{f(ha.get('dec_rate'),2)}"
        )
    else:
        print("M5: N=NA WR=NA DecWR=NA")
    if isinstance(calh, dict) and isinstance(cala, dict):
        ssw3h = (calh.get("ssw_3") or {}).get("mean") if isinstance(calh.get("ssw_3"), dict) else getattr(calh.get("ssw_3", None), "mean", None)
        ssw3a = (cala.get("ssw_3") or {}).get("mean") if isinstance(cala.get("ssw_3"), dict) else getattr(cala.get("ssw_3", None), "mean", None)
        rpr3h = (calh.get("rpr_3") or {}).get("mean") if isinstance(calh.get("rpr_3"), dict) else getattr(calh.get("rpr_3", None), "mean", None)
        rpr3a = (cala.get("rpr_3") or {}).get("mean") if isinstance(cala.get("rpr_3"), dict) else getattr(cala.get("rpr_3", None), "mean", None)
        if ssw3h is not None or rpr3h is not None:
            print(f"    set3(mean): SSW={f(ssw3h)}/{f(ssw3a)} RPR={f(rpr3h)}/{f(rpr3a)}")


def _compact_meta(meta: dict) -> dict:
    if not isinstance(meta, dict):
        return {}
    keys = ("votes_home", "votes_away", "strong_home", "strong_away", "active", "history_n", "history_pool_n", "surface", "mode")
    return {k: meta.get(k) for k in keys if k in meta}


_RU_MODULES = {
    "M1_dominance": "Доминирование (очки/приём)",
    "M2_second_serve": "Подача: 2-я + двойные + защита БП",
    "M3_return_pressure": "Давление на приёме",
    "M4_clutch": "Клатч (ключевые очки)",
    "M5_form_profile": "Профиль/форма по истории",
}

_RU_SIDES = {"home": "первый", "away": "второй", "neutral": "нейтрал"}

_RU_FLAGS = {
    "history_insufficient": "истории недостаточно (нужно минимум 3 матча на игрока)",
    "missing_points_stats": "нет стат по очкам",
    "missing_fields": "нет нужных полей",
    "too_few_points": "мало очков",
    "too_few_2nd_points": "мало очков на 2-й подаче",
    "bp_sample_too_small": "мало брейк-пойнтов",
    "fsw_missing_or_small": "нет/мало стат по 1-й подаче (очки)",
    "fsin_missing_or_small": "нет/мало стат по % первой подачи",
    "ace_missing_or_small": "нет/мало эйсов/объёма",
    "aces_missing_field": "нет поля 'Aces' в Sofascore",
    "return_points_missing": "нет стат по очкам на приёме",
    "too_few_return_games": "мало приёмных геймов",
    "return2_sample_small": "мало очков на приёме vs 2-й подачи",
    "bp_total_missing_for_conv": "нет знаменателя для BPconv (нужен Break points saved)",
    "bpconv_missing_or_small": "нет/мало BPconv (нормализовано)",
    "bpconv_support_used": "учли BPconv (нормализовано)",
    "bpc_support_used": "учли BPc (сырое число)",
    "high_break_environment": "очень брейковый матч (высокая дисперсия)",
    "bps_missing": "нет stat 'brейк-пойнты защищены'",
    "bpconv_missing": "нет конверсии брейк-пойнтов",
    "bpconv_missing_or_small": "нет/мало BPconv (нормализовано)",
    "clutch_side_by_def": "сторона по защите БП",
    "clutch_side_by_off": "сторона по реализации БП",
    "clutch_conflict": "клатч-конфликт (защита vs реализация)",
    "deciders_insufficient": "мало решающих матчей в истории",
    "set3_metrics_insufficient": "мало стат по 3-му сету в истории",
    "history_metrics_missing": "нет стат истории для метрик",
    "set2_profile_insufficient": "мало истории сценария (выиграл сет2 → решающий)",
    "rd_missing": "нет RD (очки на приёме)",
    "conflict_tpw_rd": "конфликт TPW vs RD",
    "conflict_resolved_3v1": "конфликт разрешён (3 против ≤1)",
    "neutral_tpw_or_rd": "нет перевеса по TPW/RD",
    "tpw_neutral_side_by_dr": "TPW нейтрал, берём DR",
    "dr_neutral_side_by_tpw": "DR нейтрал, берём TPW",
}


def _ru_module(name: str) -> str:
    return _RU_MODULES.get(name, name)


def _ru_side(side: str) -> str:
    return _RU_SIDES.get(side, side)


def _ru_flags(flags: list) -> list:
    return [(_RU_FLAGS.get(f, f)) for f in (flags or [])]


def _ru_line(line: str) -> str:
    s = line.strip()
    if s in _RU_FLAGS:
        return _RU_FLAGS[s]

    repl = [
        ("strength_TPW=", "сила_TPW="),
        ("strength_RD=", "сила_RD="),
        ("votes: home=", "голоса: первый="),
        ("votes: def=", "голоса: защита="),
        (" off=", " атака="),
        ("CompositeDom=", "Композит доминирования="),
        ("PTS:", "Очки:"),
        ("SPW=", "очки на подаче="),
        ("RPW=", "очки на приёме="),
        ("TP=", "очки всего="),
        ("RR_home=", "RR первого="),
        ("RR_away=", "RR второго="),
        ("retN=", "N_приёма="),
        ("DR=", "DR="),
        ("logDR=", "ln(DR)="),
        ("SSW:", "2-я подача, очки (SSW):"),
        ("FSW:", "1-я подача, очки (FSW):"),
        ("FSIN:", "% первой подачи (FSIN):"),
        ("ACE:", "Эйсы (ACE):"),
        ("DFR:", "Двойные ошибки (DFR):"),
        ("BPSR:", "Защита БП (BPSR):"),
        ("CompositeServe=", "Композит подачи="),
        ("RBR:", "Брейки на приёме (RBR):"),
        ("RPR:", "Очки на приёме (RPR):"),
        ("BPconv:", "Конверсия БП на приёме (BPconv):"),
        ("CompositeReturn=", "Композит приёма="),
        ("TrendSSW=", "Тренд 2-й подачи (TrendSSW)="),
        ("TrendRPR=", "Тренд приёма (TrendRPR)="),
        ("CalSSW diff=", "Отклонение от нормы SSW (CalSSW) diff="),
        ("CalRPR diff=", "Отклонение от нормы RPR (CalRPR) diff="),
        ("DEF(BPS):", "Клатч-защита (BPS):"),
        ("OFF(BPconv):", "Клатч-атака (BPconv):"),
        ("CompositeClutch=", "Композит клатча="),
        ("TrendBPSR=", "Тренд защиты БП (TrendBPSR)="),
        ("CalBPSR diff=", "Отклонение от нормы BPSR (CalBPSR) diff="),
        ("CalBPconv diff=", "Отклонение от нормы BPconv (CalBPconv) diff="),
        ("History last", "История last"),
        ("Deciders:", "Решающие матчи:"),
        ("Set3Index:", "Индекс 3-го сета:"),
        ("Set2->Decider:", "Сет2→решающий:"),
        ("winner_set2=", "победитель_сета2="),
        ("FINAL:", "ИТОГ:"),
        (" -> ", " → "),
        ("home=", "первый="),
        ("away=", "второй="),
        ("=home", "=первый"),
        ("=away", "=второй"),
        ("neutral", "нейтрал"),
        ("strength=", "сила="),
    ]
    for a, b in repl:
        s = s.replace(a, b)
    s = re.sub(r"(?<!_)TPW=", "Доля очков (TPW)=", s)
    s = re.sub(r"(?<!_)RD=", "Доминирование на приёме (RD)=", s)
    s = re.sub(r"\bhome\b", "первый", s)
    s = re.sub(r"\baway\b", "второй", s)
    return s


def _print_history_audit(label: str, audit: dict) -> None:
    print(f"AUDIT HISTORY [{label}] teamId={audit.get('team_id')}")
    print(
        f"  candidates={audit.get('candidates')} scanned={audit.get('scanned')} valid={audit.get('valid')} used={audit.get('used')} "
        f"over_max_history={audit.get('over_max_history')} dropped_to_equalize={audit.get('dropped_to_match_opponent')}"
    )
    excluded = audit.get("excluded_by_reason") or {}
    if excluded:
        parts = [f"{k}={excluded[k]}" for k in sorted(excluded.keys())]
        print(f"  excluded_by_reason: {', '.join(parts)}")
    coverage = audit.get("coverage") or {}
    if coverage:
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
        print("  coverage (have/total):")
        for k in keys:
            c = coverage.get(k)
            if not c:
                continue
            print(f"    {k}: {c.get('have')}/{c.get('total')}")
    used_events = audit.get("used_events") or []
    if used_events:
        print("  used_events:")
        for ev in used_events:
            print(
                f"    - eventId={ev.get('eventId')} surface={ev.get('surface')} "
                f"tpw={ev.get('tpw')} set3={ev.get('has_set3')} stats_ok={ev.get('stats_ok')} stats_dom={ev.get('stats_dom')} "
                f"missing_all_metrics={ev.get('missing_all_metrics')} opponent={ev.get('opponentName')} score={ev.get('score')}"
            )
    excluded_events = audit.get("excluded_events") or []
    if excluded_events:
        print("  excluded_examples:")
        for ev in excluded_events:
            print(
                f"    - eventId={ev.get('eventId')} opponent={ev.get('opponentName')} "
                f"tournament={ev.get('tournament')} reason={ev.get('reason')}"
            )


def _print_live_event(e: LiveEvent) -> None:
    print(f"- eventId={e.id} url={e.match_url}")
    print(f"  {e.home_team_name} vs {e.away_team_name}")


async def _with_browser(headless: bool, fn):
    async with async_playwright() as p:
        # Sofascore is sensitive to automation fingerprints; use a more "real" context.
        browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-default-browser-check",
                "--disable-dev-shm-usage",
            ],
        )
        context = await browser.new_context(
            locale="ru-RU",
            timezone_id="Europe/Moscow",
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
        )
        # Hide webdriver and add a few common navigator fields.
        try:
            await context.add_init_script(
                """
                // Playwright/Chromium automation hardening (minimal).
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['ru-RU','ru','en-US','en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
                """
            )
        except Exception:
            pass
        context.set_default_timeout(15_000)
        context.set_default_navigation_timeout(25_000)
        page = await context.new_page()
        try:
            return await fn(page)
        finally:
            await browser.close()


async def _stop_on_enter(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, sys.stdin.readline)
    except Exception:
        return
    stop_event.set()


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


def _is_pre_decider_break(event_payload: dict) -> bool:
    """
    True when match is BO3 singles and just reached 1:1 after two sets,
    but 3rd set hasn't started yet (period3 games are 0/None).

    In this window, ALL-stats == 1ST+2ND, so we can safely snapshot ALL via DOM.
    """
    e = event_payload.get("event") or {}
    status = e.get("status") or {}
    if (status.get("type") or "").lower() != "inprogress":
        return False
    home = e.get("homeScore") or {}
    away = e.get("awayScore") or {}
    if home.get("current") != 1 or away.get("current") != 1:
        return False
    # If point scoring already started in set3, ALL may include 3rd-set points (mixed).
    # Only consider it a safe between-sets window when current point score is missing/zero-ish.
    hpt = home.get("point")
    apt = away.get("point")
    if isinstance(hpt, str) and hpt.strip() not in ("", "0", "0:0"):
        return False
    if isinstance(apt, str) and apt.strip() not in ("", "0", "0:0"):
        return False
    s3h = home.get("period3")
    s3a = away.get("period3")
    if (s3h is None or s3h == 0) and (s3a is None or s3a == 0):
        return True
    return False


async def _print_team_histories(page, *, event_payload: dict, history: int) -> None:
    e = event_payload.get("event") or {}
    home = e.get("homeTeam") or {}
    away = e.get("awayTeam") or {}
    home_id = int(home.get("id"))
    away_id = int(away.get("id"))

    for team_id, label in ((home_id, "home"), (away_id, "away")):
        events = await get_last_finished_singles_events(page, team_id, limit=history)
        print(f"  {label} last{len(events)}:")
        for ev in events:
            summary = summarize_event_for_team(ev, team_id=team_id)
            hs = summary["homeScore"]["display"]
            as_ = summary["awayScore"]["display"]
            print(
                f"    - eventId={summary['eventId']} won={summary['won']} opponent={summary['opponentName']} score={hs}:{as_}"
            )


async def cmd_live(limit: int, history: int, headless: bool) -> int:
    async def run(page):
        match_links = await get_live_match_links(page, limit=limit)
        if not match_links:
            print("No match links found.")
            return 0

        shown = 0
        for link in match_links:
            event_id = parse_event_id_from_match_link(link)
            if not event_id:
                continue
            payload = await get_event_from_match_url_via_navigation(page, match_url=link, event_id=event_id)
            if not _is_live(payload):
                continue
            e = payload.get("event") or {}
            le = LiveEvent(
                id=int(e.get("id")),
                slug=str(e.get("slug") or "").strip(),
                custom_id=str(e.get("customId") or "").strip(),
                home_team_id=int((e.get("homeTeam") or {}).get("id")),
                home_team_name=str((e.get("homeTeam") or {}).get("name") or ""),
                away_team_id=int((e.get("awayTeam") or {}).get("id")),
                away_team_name=str((e.get("awayTeam") or {}).get("name") or ""),
            )
            _print_live_event(le)
            await _print_team_histories(page, event_payload=payload, history=history)
            shown += 1
            if shown >= limit:
                break
        return 0

    return await _with_browser(headless, run)


async def cmd_analyze(
    match_url: str,
    max_history: int,
    *,
    details: bool,
    audit: bool,
    audit_history: bool,
    dump_dir: Optional[str],
    features: bool,
    dump_features: Optional[str],
    tg: bool,
    tg_token: str,
    tg_chat: str,
    tg_send: str,
    numbers: bool,
    no_action: bool,
    focus: bool,
    brief: bool,
    mode: str,
    headless: bool,
    history_only: bool,
) -> int:
    event_id = parse_event_id_from_match_link(match_url)
    if not event_id:
        raise SystemExit("match-url must contain #id:<eventId>")

    async def run(page):
        send_policy = tg_send if tg_send in ("all", "bet") else "bet"
        payload = await get_event_from_match_url_via_navigation(page, match_url=match_url, event_id=event_id)
        if audit:
            stats = await extract_statistics_dom(page, match_url=match_url.split("#id:")[0], event_id=event_id, periods=("1ST", "2ND"))
            print("AUDIT raw Sofascore stats (1ST/2ND):")
            _print_audit(MatchSnapshot(event_id=event_id, stats=stats))
        ctx, mods, final_side, score, meta = await analyze_once(
            page,
            event_payload=payload,
            match_url=match_url.split("#id:")[0],
            event_id=event_id,
            max_history=max_history,
            history_only=history_only,
            audit_history=audit_history,
            audit_features=(features or bool(dump_features) or numbers or brief),
        )
        meta["mode"] = mode
        decision, decision_side = ("SKIP", "neutral")
        if not no_action:
            decision, decision_side = bet_decision(mods, mode=mode)
        lean = "neutral"
        if meta.get("active", 0) > 0:
            lean = "home" if score > 0 else "away" if score < 0 else "neutral"
        if brief:
            _print_brief(ctx, meta)
            return 0

        print(f"{ctx.home_name} vs {ctx.away_name} | eventId={ctx.event_id} | set2_winner={ctx.set2_winner}")
        if numbers:
            _print_numbers_legend()
        if no_action:
            print(f"META: score={score} meta={_compact_meta(meta)} mode={mode}")
        else:
            print(
                f"LEAN: {lean}({_ru_side(lean)}) score={score} meta={_compact_meta(meta)} | "
                f"ACTION: {decision} {decision_side}({_ru_side(decision_side)}) mode={mode}"
            )
        if numbers:
            _print_numbers(ctx, meta, show_current=True, show_history=True, focus=focus)
        else:
            _print_summary(meta)
        if focus:
            return 0
        if numbers and not details and not focus:
            _print_models_numeric(meta)
        else:
            for m in mods:
                if numbers and not details and (m.side == "neutral" or m.strength <= 0):
                    continue
                print(
                    f"- {m.name} ({_ru_module(m.name)}): {m.side}({_ru_side(m.side)}) "
                    f"strength={m.strength} flags={_ru_flags(m.flags)}"
                )
                if details:
                    for line in m.explain:
                        print(f"  {_ru_line(line)}")
        if audit_history and isinstance(meta.get("history_audit"), dict):
            _print_history_audit("home", meta["history_audit"].get("home") or {})
            _print_history_audit("away", meta["history_audit"].get("away") or {})

        if features and isinstance(meta.get("features"), dict):
            print("FEATURES (computed):")
            print(json.dumps(meta["features"], ensure_ascii=False, indent=2))

        if dump_features and isinstance(meta.get("features"), dict):
            ts = int(time.time())
            p = Path(dump_features)
            if p.suffix.lower() != ".json":
                p = p / f"features_{event_id}_{ts}.json"
            _dump_json(p, {"ts": ts, "eventId": event_id, "features": meta["features"]})
            print(f"Dumped computed features to: {p}")

        if dump_dir:
            ts = int(time.time())
            out_dir = Path(dump_dir)
            # Current match dump
            stats = await get_event_statistics(page, event_id)
            _dump_json(
                out_dir / f"current_{event_id}_{ts}.json",
                {
                    "ts": ts,
                    "match_url": match_url.split("#id:")[0],
                    "event_payload": payload,
                    "stats_used_subset": _used_stats_subset(stats, event_id=event_id),
                },
            )
            # History dumps (used events only)
            if isinstance(meta.get("history_audit"), dict):
                for side_label in ("home", "away"):
                    ha = (meta["history_audit"].get(side_label) or {})
                    team_id = ha.get("team_id")
                    used = ha.get("used_events") or []
                    if not team_id or not used:
                        continue
                    jl = out_dir / f"history_{side_label}_{team_id}_{event_id}_{ts}.jsonl"
                    for ev in used:
                        hid = ev.get("eventId")
                        if not hid:
                            continue
                        try:
                            hstats = await get_event_statistics(page, int(hid))
                        except Exception:
                            continue
                        _dump_jsonl(
                            jl,
                            {
                                "ts": ts,
                                "team_id": team_id,
                                "eventId": int(hid),
                                "meta": ev,
                                "stats_used_subset": _used_stats_subset(hstats, event_id=int(hid)),
                            },
                        )
            print(f"Dumped raw used stats to: {out_dir}")

        if tg:
            if send_policy == "bet" and decision != "BET":
                return 0
            cfg = get_telegram_config(token=tg_token, chat_id=tg_chat)
            if not cfg:
                print("TG: missing token/chat_id (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID or pass --tg-token/--tg-chat)")
            else:
                msg = _format_tg_message(
                    event_id=ctx.event_id,
                    match_url=match_url.split("#id:")[0],
                    home_name=ctx.home_name,
                    away_name=ctx.away_name,
                    set2_winner=ctx.set2_winner,
                    decision=decision,
                    decision_side=decision_side,
                    score=score,
                    meta=meta,
                    mods=mods,
                )
                client = TelegramClient(token=cfg.token, chat_id=cfg.chat_id)
                res = await asyncio.to_thread(client.send_text_result, msg)
                if isinstance(res, dict) and res.get("ok") and isinstance(res.get("result"), dict):
                    print(f"TG: sent mid={res['result'].get('message_id')}")
                else:
                    desc = (res or {}).get("description") if isinstance(res, dict) else None
                    print(f"TG: send failed: {desc or res}")
        return 0

    return await _with_browser(headless, run)


async def cmd_watch(
    poll_s: float,
    max_history: int,
    *,
    details: bool,
    audit: bool,
    audit_history: bool,
    dump_dir: Optional[str],
    features: bool,
    dump_features: Optional[str],
    tg: bool,
    tg_token: str,
    tg_chat: str,
    tg_send: str,
    tg_progress: bool,
    numbers: bool,
    no_action: bool,
    focus: bool,
    brief: bool,
    mode: str,
    headless: bool,
    history_only: bool,
    only_1_1: bool,
) -> int:
    async def run(page):
        send_policy = tg_send if tg_send in ("all", "bet") else "bet"
        if no_action:
            # User decides; still send notifications.
            send_policy = "all"
        if numbers and not brief:
            _print_numbers_legend()
        stop_event = asyncio.Event()
        stop_task = asyncio.create_task(_stop_on_enter(stop_event))
        print("Нажми Enter, чтобы остановить.")
        # Per-event done states. We can analyze a match twice:
        # - history-only early (before 1:1)
        # - current+history once it becomes 1:1 after 2 sets.
        done_state: dict = {}  # event_id -> set({"history","live"})
        failures: dict = {}  # (event_id, mode) -> count
        # Per-event progress tracking (history scraping can take time).
        progress_seen: dict = {}
        tg_client: Optional[TelegramClient] = None
        if tg:
            cfg = get_telegram_config(token=tg_token, chat_id=tg_chat)
            if cfg:
                tg_client = TelegramClient(token=cfg.token, chat_id=cfg.chat_id)
            else:
                print("TG: missing token/chat_id (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID or pass --tg-token/--tg-chat)")
        while not stop_event.is_set():
            # Heartbeat counters: show progress even when no 1:1 matches exist.
            hb_links = 0
            hb_live = 0
            hb_bo3_singles = 0
            hb_set_1_1 = 0
            hb_triggerable = 0
            hb_errors = 0
            match_links = await get_live_match_links(page)
            hb_links = len(match_links)
            for link in match_links:
                if stop_event.is_set():
                    break
                event_id = parse_event_id_from_match_link(link)
                if not event_id:
                    continue
                payload = None
                try:
                    payload = await get_event_from_match_url_via_navigation(page, match_url=link, event_id=event_id)
                    if not _is_live(payload):
                        continue
                    hb_live += 1
                    if not _is_bo3(payload):
                        continue
                    e = payload.get("event") or {}
                    if not is_singles_event(e):
                        continue
                    hb_bo3_singles += 1
                    is_1_1 = _is_set_score_1_1_after_two_sets(payload)
                    if is_1_1:
                        hb_set_1_1 += 1

                    if not isinstance(done_state, dict):
                        done_state = {}
                    state = done_state.setdefault(event_id, set())
                    desired_mode = None  # "history" | "live"
                    if history_only:
                        desired_mode = "history"
                    elif is_1_1:
                        desired_mode = "live"
                    elif only_1_1:
                        desired_mode = None
                    else:
                        desired_mode = "history"

                    if desired_mode is None:
                        continue
                    if desired_mode in state:
                        continue
                    hb_triggerable += 1

                    # Strict mode: no prefetch/mixed fallbacks.

                    # Progress marker: history-only analysis can take time (history scraping).
                    try:
                        hn = str(((payload.get("event") or {}).get("homeTeam") or {}).get("name") or "")
                        an = str(((payload.get("event") or {}).get("awayTeam") or {}).get("name") or "")
                    except Exception:
                        hn, an = "", ""
                    mode_label = "history-only" if desired_mode == "history" else "1:1"
                    print(f"\n=== eventId={event_id} ===", flush=True)
                    if hn or an:
                        print(f"ANALYZE ({mode_label}): {hn} vs {an}", flush=True)

                    async def progress_cb(ev: dict) -> None:
                        if not isinstance(ev, dict):
                            return
                        if stop_event.is_set():
                            raise asyncio.CancelledError()
                        lab = str(ev.get("label") or "")
                        target = int(ev.get("target") or max_history)
                        have = int(ev.get("have") or 0)
                        scanned = int(ev.get("scanned") or 0)
                        opp = str(ev.get("opponent") or "")
                        # Console: always show.
                        print(
                            f"[history eventId={event_id}] {lab} {have}/{target} scanned={scanned}"
                            + (f" last_opponent={opp}" if opp else ""),
                            flush=True,
                        )
                        # Telegram: send only milestones to avoid spam.
                        if not tg_progress:
                            return
                        if tg_client is None:
                            return
                        if send_policy != "all":
                            return
                        st = progress_seen.get(event_id)
                        if not isinstance(st, dict):
                            st = {"home": set(), "away": set()}
                            progress_seen[event_id] = st
                        milestones = {1, 3, target}
                        if have not in milestones:
                            return
                        key = st.get(lab)
                        if not isinstance(key, set):
                            key = set()
                            st[lab] = key
                        if have in key:
                            return
                        key.add(have)
                        try:
                            msg = (
                                f"<b>{_html_escape(hn)} vs {_html_escape(an)}</b>\n"
                                f"<a href=\"{_html_escape(link.split('#id:')[0])}\">Sofascore</a>\n"
                                f"<code>eventId={event_id}</code>\n"
                                f"Сбор истории: <b>{lab}</b> {have}/{target}"
                            )
                            res = await asyncio.to_thread(tg_client.send_text_result, msg)
                            if not (isinstance(res, dict) and res.get("ok")):
                                pass
                        except Exception:
                            pass

                    analyze_task = asyncio.create_task(
                        analyze_once(
                            page,
                            event_payload=payload,
                            match_url=link.split("#id:")[0],
                            event_id=event_id,
                            max_history=max_history,
                            history_only=(desired_mode == "history"),
                            progress_cb=progress_cb,
                            audit_history=audit_history,
                            audit_features=(features or bool(dump_features) or numbers or brief or (tg_client is not None)),
                        )
                    )
                    # Allow stop-on-enter to interrupt long history scraping.
                    while True:
                        done_tasks, _pending = await asyncio.wait({analyze_task}, timeout=0.25)
                        if done_tasks:
                            break
                        if stop_event.is_set():
                            analyze_task.cancel()
                            break
                    if stop_event.is_set():
                        try:
                            await analyze_task
                        except Exception:
                            pass
                        break
                    ctx, mods, final_side, score, meta = analyze_task.result()
                    # Mark done. (Strict: no fallbacks, so "live" only succeeds when 1ST/2ND is available.)
                    state.add(desired_mode)
                    failures.pop((event_id, desired_mode), None)
                    # no prefetch/fallback state
                except Exception as ex:
                    print(f"\n=== eventId={event_id} ===", flush=True)
                    print(f"ERROR: {ex}", flush=True)
                    hb_errors += 1
                    # Soft-fail: keep retrying a few polls in a row (UI lag / temporary DOM issues).
                    # Track failures by event + mode (adaptive watcher can do two passes per match).
                    # Best-effort infer which mode we were attempting:
                    inferred = "history"
                    try:
                        if not history_only and isinstance(payload, dict) and _is_set_score_1_1_after_two_sets(payload):
                            inferred = "live"
                    except Exception:
                        inferred = "history"
                    mode_key = (event_id, inferred)
                    failures[mode_key] = int(failures.get(mode_key, 0)) + 1
                    # Soft-fail: keep retrying the same match a few polls in a row.
                    # Sofascore stats UI can lag right when set2 ends.
                    if failures[mode_key] >= 6:
                        print("ERROR: too many failures, skipping this match.")
                        if isinstance(done_state, dict):
                            done_state.setdefault(event_id, set()).add(mode_key[1])
                    continue

                decision, decision_side = ("SKIP", "neutral")
                if not no_action:
                    decision, decision_side = bet_decision(mods, mode=mode)
                meta["mode"] = mode
                lean = "neutral"
                if meta.get("active", 0) > 0:
                    lean = "home" if score > 0 else "away" if score < 0 else "neutral"

                # Telegram: send forecast for this match (even in --brief mode).
                if tg_client is not None:
                    try:
                        if send_policy == "all" or (send_policy == "bet" and decision == "BET"):
                            msg = _format_tg_message(
                                event_id=ctx.event_id,
                                match_url=ctx.url,
                                home_name=ctx.home_name,
                                away_name=ctx.away_name,
                                set2_winner=ctx.set2_winner,
                                decision=decision,
                                decision_side=decision_side,
                                score=score,
                                meta=meta,
                                mods=mods,
                            )
                            res = await asyncio.to_thread(tg_client.send_text_result, msg)
                            if isinstance(res, dict) and res.get("ok") and isinstance(res.get("result"), dict):
                                print(f"TG: sent mid={res['result'].get('message_id')}")
                            else:
                                desc = (res or {}).get("description") if isinstance(res, dict) else None
                                print(f"TG: send failed: {desc or res}")
                    except Exception as e:
                        print(f"TG: send error: {e}")
                print(f"\n=== eventId={ctx.event_id} ===")
                if brief:
                    _print_brief(ctx, meta)
                    continue

                print(f"{ctx.home_name} vs {ctx.away_name} | set2_winner={ctx.set2_winner}")
                if no_action:
                    print(f"META: score={score} meta={_compact_meta(meta)} mode={mode}")
                else:
                    print(
                        f"LEAN: {lean}({_ru_side(lean)}) score={score} meta={_compact_meta(meta)} | "
                        f"ACTION: {decision} {decision_side}({_ru_side(decision_side)}) mode={mode}"
                    )
                if numbers:
                    _print_numbers(ctx, meta, show_current=True, show_history=True, focus=focus)
                else:
                    _print_summary(meta)
                if focus:
                    continue
                if audit:
                    try:
                        stats = await extract_statistics_dom(page, match_url=ctx.url, event_id=ctx.event_id, periods=("1ST", "2ND"))
                        print("AUDIT raw Sofascore stats (1ST/2ND):")
                        _print_audit(MatchSnapshot(event_id=ctx.event_id, stats=stats))
                    except Exception as ex:
                        print(f"AUDIT: unavailable: {ex}")
                if numbers and not details and not focus:
                    _print_models_numeric(meta)
                else:
                    for m in mods:
                        if numbers and not details and (m.side == "neutral" or m.strength <= 0):
                            continue
                        print(
                            f"- {m.name} ({_ru_module(m.name)}): {m.side}({_ru_side(m.side)}) "
                            f"strength={m.strength} flags={_ru_flags(m.flags)}"
                        )
                        if details:
                            for line in m.explain:
                                print(f"  {_ru_line(line)}")
                if audit_history and isinstance(meta.get("history_audit"), dict):
                    _print_history_audit("home", meta["history_audit"].get("home") or {})
                    _print_history_audit("away", meta["history_audit"].get("away") or {})
                if features and isinstance(meta.get("features"), dict):
                    print("FEATURES (computed):")
                    print(json.dumps(meta["features"], ensure_ascii=False, indent=2))
                if dump_features and isinstance(meta.get("features"), dict):
                    ts = int(time.time())
                    p = Path(dump_features)
                    if p.suffix.lower() != ".json":
                        p = p / f"features_{ctx.event_id}_{ts}.json"
                    _dump_json(p, {"ts": ts, "eventId": ctx.event_id, "features": meta["features"]})
                    print(f"Dumped computed features to: {p}")
                if dump_dir:
                    ts = int(time.time())
                    out_dir = Path(dump_dir)
                    try:
                        stats = await get_event_statistics(page, ctx.event_id)
                        _dump_json(
                            out_dir / f"current_{ctx.event_id}_{ts}.json",
                            {
                                "ts": ts,
                                "match_url": link.split("#id:")[0],
                                "event_payload": payload,
                                "stats_used_subset": _used_stats_subset(stats, event_id=ctx.event_id),
                            },
                        )
                        if isinstance(meta.get("history_audit"), dict):
                            for side_label in ("home", "away"):
                                ha = (meta["history_audit"].get(side_label) or {})
                                team_id = ha.get("team_id")
                                used = ha.get("used_events") or []
                                if not team_id or not used:
                                    continue
                                jl = out_dir / f"history_{side_label}_{team_id}_{ctx.event_id}_{ts}.jsonl"
                                for ev in used:
                                    hid = ev.get("eventId")
                                    if not hid:
                                        continue
                                    try:
                                        hstats = await get_event_statistics(page, int(hid))
                                    except Exception:
                                        continue
                                    _dump_jsonl(
                                        jl,
                                        {
                                            "ts": ts,
                                            "team_id": team_id,
                                            "eventId": int(hid),
                                            "meta": ev,
                                            "stats_used_subset": _used_stats_subset(hstats, event_id=int(hid)),
                                        },
                                    )
                        print(f"Dumped raw used stats to: {out_dir}")
                    except Exception:
                        pass
            # One-line heartbeat each poll (so you see it's scanning).
            now = time.strftime("%H:%M:%S")
            label = "history-only" if history_only else ("only-1:1" if only_1_1 else "adaptive")
            triggered_runs = sum(len(v) for v in done_state.values()) if isinstance(done_state, dict) else 0
            print(
                f"[watch {label} {now}] live_links={hb_links} live={hb_live} bo3_singles={hb_bo3_singles} "
                f"set1-1={hb_set_1_1} triggerable={hb_triggerable} triggered={triggered_runs} "
                f"errors={hb_errors} next={int(poll_s)}s",
                flush=True,
            )
            # Allow Enter-stop to interrupt the poll sleep immediately.
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=float(poll_s))
            except asyncio.TimeoutError:
                pass

        stop_task.cancel()
        return 0

    return await _with_browser(headless, run)


async def cmd_tg_test(*, tg_token: str, tg_chat: str) -> int:
    cfg = get_telegram_config(token=tg_token, chat_id=tg_chat)
    if not cfg:
        raise SystemExit("TG test: missing token/chat_id (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID or pass --tg-token/--tg-chat)")
    client = TelegramClient(token=cfg.token, chat_id=cfg.chat_id)
    msg = "<b>third_set</b>: TG test ping"
    res = await asyncio.to_thread(client.send_text_result, msg)
    if isinstance(res, dict) and res.get("ok") and isinstance(res.get("result"), dict):
        print(f"TG test: sent mid={res['result'].get('message_id')}")
        return 0
    desc = (res or {}).get("description") if isinstance(res, dict) else None
    print(f"TG test: failed: {desc or res}")
    return 2


async def cmd_tg_updates(*, tg_token: str, tg_chat: str, limit: int) -> int:
    """
    Print recent chats seen by the bot via getUpdates.

    Usage:
      1) Open @your_bot and press Start (or send any message).
      2) Run: python3 -m third_set.cli tg-updates --tg-token ... --limit 50
      3) Copy the нужный chat.id into TELEGRAM_CHAT_ID / THIRDSET_TG_CHAT_ID.
    """
    token = (tg_token or "").strip() or (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("THIRDSET_TG_TOKEN") or "").strip()
    if not token:
        raise SystemExit("TG updates: missing token (set TELEGRAM_BOT_TOKEN or pass --tg-token)")
    client = TelegramClient(token=token, chat_id=(tg_chat or "0"))
    res = await asyncio.to_thread(client.get_updates, limit=limit)
    if not isinstance(res, dict) or not res.get("ok"):
        desc = res.get("description") if isinstance(res, dict) else None
        print(f"TG updates: failed: {desc or res}")
        return 2
    items = res.get("result") or []
    chats = {}
    for upd in items if isinstance(items, list) else []:
        if not isinstance(upd, dict):
            continue
        msg = upd.get("message") or upd.get("edited_message") or upd.get("channel_post") or upd.get("edited_channel_post")
        if not isinstance(msg, dict):
            continue
        chat = msg.get("chat")
        if not isinstance(chat, dict):
            continue
        cid = chat.get("id")
        if cid is None:
            continue
        key = str(cid)
        title = chat.get("title") or ""
        username = chat.get("username") or ""
        ctype = chat.get("type") or ""
        chats[key] = {"id": cid, "type": ctype, "title": title, "username": username}
    if not chats:
        print("TG updates: no chats found. Send /start to your bot (or write in the target group) and retry.")
        return 0
    print("TG chats seen in updates:")
    for k in sorted(chats.keys(), key=lambda s: int(s) if s.lstrip('-').isdigit() else 0):
        c = chats[k]
        line = f"- chat_id={c['id']} type={c['type']}"
        if c.get("title"):
            line += f" title={c['title']}"
        if c.get("username"):
            line += f" username=@{c['username']}"
        print(line)
    print("\nSet TELEGRAM_CHAT_ID (or THIRDSET_TG_CHAT_ID) to the desired chat_id.")
    return 0


async def cmd_probe_stats(*, limit: int, headless: bool) -> int:
    """
    Probe /api/v1/event/<id>/statistics across live tennis matches.

    This helps validate which periods/items are actually present for different matches.
    """

    async def run(page):
        links = await get_live_match_links(page, limit=limit)
        if not links:
            print("No live tennis matches found.")
            return 0

        required = [
            ("Points", "Service points won"),
            ("Points", "Receiver points won"),
            ("Service", "Second serve points"),
            ("Return", "First serve return points"),
            ("Return", "Second serve return points"),
            ("Service", "Break points saved"),
            ("Return", "Break points converted"),
            ("Return", "Return games played"),
            ("Games", "Service games won"),
        ]

        def has_item(stats: dict, *, period: str, group: str, item: str) -> bool:
            periods = stats.get("statistics")
            if not isinstance(periods, list):
                return False
            per = next((p for p in periods if p.get("period") == period), None)
            if not isinstance(per, dict):
                return False
            groups = per.get("groups") or []
            g = next((x for x in groups if x.get("groupName") == group), None)
            if not isinstance(g, dict):
                return False
            items = g.get("statisticsItems") or []
            return any(isinstance(it, dict) and it.get("name") == item for it in items)

        missing_counts = {f"{g}/{i}": 0 for g, i in required}
        missing_periods_12 = 0
        checked = 0

        for link in links:
            event_id = parse_event_id_from_match_link(link)
            if not event_id:
                continue
            checked += 1

            try:
                ev = await get_event(page, int(event_id))
            except Exception:
                ev = {}
            e = (ev.get("event") or {}) if isinstance(ev, dict) else {}
            home = (e.get("homeTeam") or {}).get("name") or "?"
            away = (e.get("awayTeam") or {}).get("name") or "?"

            try:
                stats = await get_event_statistics(page, int(event_id))
            except Exception as ex:
                print(f"- eventId={event_id} {home} vs {away} | statistics ERROR: {ex}")
                for g, i in required:
                    missing_counts[f"{g}/{i}"] += 1
                missing_periods_12 += 1
                continue

            periods = stats.get("statistics")
            period_names = [p.get("period") for p in periods] if isinstance(periods, list) else []
            has_12 = ("1ST" in period_names) and ("2ND" in period_names)
            if not has_12:
                missing_periods_12 += 1

            flags = []
            for g, i in required:
                ok = has_item(stats, period="1ST", group=g, item=i) or has_item(stats, period="2ND", group=g, item=i)
                if not ok:
                    missing_counts[f"{g}/{i}"] += 1
                    flags.append(f"-{g}/{i}")

            pstr = ",".join([p for p in period_names if isinstance(p, str)])
            print(f"- eventId={event_id} {home} vs {away} | periods=[{pstr}] | missing={len(flags)}")
            if flags:
                print("  " + " ".join(flags[:12]) + (" ..." if len(flags) > 12 else ""))

        print("\nSUMMARY:")
        print(f"- checked={checked}")
        print(f"- missing_1ST_or_2ND_period={missing_periods_12}")
        top = sorted(missing_counts.items(), key=lambda kv: kv[1], reverse=True)
        for k, v in top[:10]:
            print(f"- missing {k}: {v}/{checked}")
        return 0

    return await _with_browser(headless, run)


async def cmd_probe_dom(*, limit: int, headless: bool) -> int:
    """
    Probe DOM label coverage for Statistics tab across live tennis matches.
    Prints which row labels were not mapped (root cause of '—'/NA).
    """

    async def run(page):
        links = await get_live_match_links(page, limit=limit)
        if not links:
            print("No live tennis matches found.")
            return 0

        agg: dict = {}  # group -> label -> count
        for link in links:
            event_id = parse_event_id_from_match_link(link)
            if not event_id:
                continue
            try:
                # Navigate and read DOM stats for 1ST/2ND.
                await get_event_from_match_url_via_navigation(page, match_url=link, event_id=event_id)
                stats = await extract_statistics_dom(
                    page, match_url=link.split("#id:")[0], event_id=event_id, periods=("1ST", "2ND")
                )
                meta = stats.get("_meta") if isinstance(stats, dict) else None
                unm = (meta or {}).get("unmapped") if isinstance(meta, dict) else None
                if not isinstance(unm, dict):
                    continue
                for per, groups in unm.items():
                    if not isinstance(groups, dict):
                        continue
                    for g, labels in groups.items():
                        if not isinstance(labels, list):
                            continue
                        for lab in labels:
                            key = f"{per}:{g}"
                            agg.setdefault(key, {})
                            agg[key][str(lab)] = int(agg[key].get(str(lab), 0)) + 1
            except Exception:
                continue

        if not agg:
            print("DOM probe: no unmapped labels found (or stats not available).")
            return 0

        print("DOM probe: unmapped labels (add these to _ITEM_MAP):")
        for key in sorted(agg.keys()):
            print(f"- {key}")
            items = sorted(agg[key].items(), key=lambda kv: kv[1], reverse=True)
            for lab, cnt in items[:25]:
                print(f"  * {lab} (x{cnt})")
        return 0

    return await _with_browser(headless, run)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="third_set")
    parser.add_argument("--headed", action="store_true", help="Run with visible browser window")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_live = sub.add_parser("live", help="List matches on Sofascore tennis page")
    p_live.add_argument("--limit", type=int, default=5)
    p_live.add_argument("--history", type=int, default=5)

    p_analyze = sub.add_parser("analyze", help="Run 5-module analysis for one match URL")
    p_analyze.add_argument("--match-url", type=str, required=True, help="Full match URL with #id:<eventId>")
    p_analyze.add_argument("--max-history", type=int, default=5)
    p_analyze.add_argument("--details", action="store_true", help="Print module calculations")
    p_analyze.add_argument(
        "--audit",
        action="store_true",
        help="Print raw Sofascore stats used (1ST/2ND) for manual verification",
    )
    p_analyze.add_argument(
        "--audit-history",
        action="store_true",
        help="Print how last-N history was collected/filtered and which metrics were available",
    )
    p_analyze.add_argument("--mode", choices=["conservative", "normal", "aggressive"], default="normal")
    p_analyze.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Write raw used Sofascore stats (current + used history matches) into this directory",
    )
    p_analyze.add_argument(
        "--features",
        action="store_true",
        help="Print structured computed features (current + history aggregates) as JSON",
    )
    p_analyze.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="Write structured computed features to this file or directory (features_*.json)",
    )
    p_analyze.add_argument("--tg", action="store_true", help="Send result to Telegram")
    p_analyze.add_argument("--tg-token", default=os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("THIRDSET_TG_TOKEN") or "")
    p_analyze.add_argument("--tg-chat", default=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("THIRDSET_TG_CHAT_ID") or "")
    p_analyze.add_argument("--tg-send", choices=["bet", "all"], default="bet", help="Send only BET (default) or all triggers")
    p_analyze.add_argument("--numbers", action="store_true", help="Print compact numeric output (probability + key metrics)")
    p_analyze.add_argument("--focus", action="store_true", help="With --numbers: print only NORM/IDX/QUALITY/MISS (hide raw blocks)")
    p_analyze.add_argument("--brief", action="store_true", help="Print only one-line prediction (no details)")
    p_analyze.add_argument("--no-action", action="store_true", help="Do not compute/print ACTION (you decide yourself)")
    p_analyze.add_argument(
        "--history-only",
        action="store_true",
        help="Analyze using ONLY historical data (ignore current match statistics; runs even when sets are 0:0)",
    )

    p_watch = sub.add_parser(
        "watch",
        help="Watch live BO3 singles: if set score is 1:1 use current (set1+2) + history; otherwise history-only",
    )
    p_watch.add_argument("--poll", type=float, default=15.0)
    p_watch.add_argument("--max-history", type=int, default=5)
    p_watch.add_argument("--details", action="store_true", help="Print module calculations")
    p_watch.add_argument(
        "--audit",
        action="store_true",
        help="Print raw Sofascore stats used (1ST/2ND) for manual verification",
    )
    p_watch.add_argument(
        "--audit-history",
        action="store_true",
        help="Print how last-N history was collected/filtered and which metrics were available",
    )
    p_watch.add_argument("--mode", choices=["conservative", "normal", "aggressive"], default="normal")
    p_watch.add_argument(
        "--dump-dir",
        type=str,
        default=None,
        help="Write raw used Sofascore stats (current + used history matches) into this directory",
    )
    p_watch.add_argument(
        "--features",
        action="store_true",
        help="Print structured computed features (current + history aggregates) as JSON",
    )
    p_watch.add_argument(
        "--dump-features",
        type=str,
        default=None,
        help="Write structured computed features to this file or directory (features_*.json)",
    )
    p_watch.add_argument("--tg", action="store_true", help="Send result to Telegram")
    p_watch.add_argument("--tg-token", default=os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("THIRDSET_TG_TOKEN") or "")
    p_watch.add_argument("--tg-chat", default=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("THIRDSET_TG_CHAT_ID") or "")
    p_watch.add_argument("--tg-send", choices=["bet", "all"], default="bet", help="Send only BET (default) or all triggers")
    p_watch.add_argument("--tg-progress", action="store_true", help="Also send history collection progress to Telegram")
    p_watch.add_argument("--numbers", action="store_true", help="Print compact numeric output (probability + key metrics)")
    p_watch.add_argument("--focus", action="store_true", help="With --numbers: print only NORM/IDX/QUALITY/MISS (hide raw blocks)")
    p_watch.add_argument("--brief", action="store_true", help="Print only one-line prediction per match (no details)")
    p_watch.add_argument("--no-action", action="store_true", help="Do not compute/print ACTION (you decide yourself)")
    p_watch.add_argument(
        "--history-only",
        action="store_true",
        help="Analyze using ONLY historical data (ignore current match statistics; analyzes every live BO3 singles match)",
    )
    p_watch.add_argument(
        "--only-1-1",
        action="store_true",
        help="Legacy mode: analyze ONLY matches with set score 1:1 after 2 sets (skip early 0:0/1:0/0:1)",
    )

    p_tg = sub.add_parser("tg-test", help="Send a test message to Telegram")
    p_tg.add_argument("--tg-token", default=os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("THIRDSET_TG_TOKEN") or "")
    p_tg.add_argument("--tg-chat", default=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("THIRDSET_TG_CHAT_ID") or "")

    p_tg_u = sub.add_parser("tg-updates", help="List recent chats seen by the bot (getUpdates)")
    p_tg_u.add_argument("--tg-token", default=os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("THIRDSET_TG_TOKEN") or "")
    p_tg_u.add_argument("--tg-chat", default=os.getenv("TELEGRAM_CHAT_ID") or os.getenv("THIRDSET_TG_CHAT_ID") or "")
    p_tg_u.add_argument("--limit", type=int, default=50)

    p_probe = sub.add_parser("probe-stats", help="Probe statistics availability across live tennis matches")
    p_probe.add_argument("--limit", type=int, default=15)

    p_probe_dom = sub.add_parser("probe-dom", help="Probe DOM stats labels (find unmapped rows causing NA/—)")
    p_probe_dom.add_argument("--limit", type=int, default=8)

    args = parser.parse_args(argv)
    headless = not args.headed

    if args.cmd == "live":
        return asyncio.run(cmd_live(args.limit, args.history, headless=headless))
    if args.cmd == "analyze":
        return asyncio.run(
            cmd_analyze(
                args.match_url,
                args.max_history,
                details=args.details,
                audit=args.audit,
                audit_history=args.audit_history,
                dump_dir=args.dump_dir,
                features=args.features,
                dump_features=args.dump_features,
                tg=args.tg,
                tg_token=args.tg_token,
                tg_chat=args.tg_chat,
                tg_send=args.tg_send,
                numbers=args.numbers,
                no_action=args.no_action,
                focus=args.focus,
                brief=args.brief,
                mode=args.mode,
                headless=headless,
                history_only=args.history_only,
            )
        )
    if args.cmd == "watch":
        return asyncio.run(
            cmd_watch(
                args.poll,
                args.max_history,
                details=args.details,
                audit=args.audit,
                audit_history=args.audit_history,
                dump_dir=args.dump_dir,
                features=args.features,
                dump_features=args.dump_features,
                tg=args.tg,
                tg_token=args.tg_token,
                tg_chat=args.tg_chat,
                tg_send=args.tg_send,
                tg_progress=args.tg_progress,
                numbers=args.numbers,
                no_action=args.no_action,
                focus=args.focus,
                brief=args.brief,
                mode=args.mode,
                headless=headless,
                history_only=args.history_only,
                only_1_1=args.only_1_1,
            )
        )
    if args.cmd == "tg-test":
        return asyncio.run(cmd_tg_test(tg_token=args.tg_token, tg_chat=args.tg_chat))
    if args.cmd == "tg-updates":
        return asyncio.run(cmd_tg_updates(tg_token=args.tg_token, tg_chat=args.tg_chat, limit=args.limit))
    if args.cmd == "probe-stats":
        return asyncio.run(cmd_probe_stats(limit=args.limit, headless=headless))
    if args.cmd == "probe-dom":
        return asyncio.run(cmd_probe_dom(limit=args.limit, headless=headless))
    raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
