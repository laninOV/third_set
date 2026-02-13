from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from third_set.sofascore import SofascoreError


def _sofascore_error_code(ex: SofascoreError) -> Tuple[Optional[str], str]:
    ex_text = str(ex or "").strip()
    lo = ex_text.lower()
    m = re.search(r"\bcode=([a-z0-9_]+)\b", lo)
    raw_code = m.group(1) if m else None
    if "timeout storm" in lo:
        raw_code = "timeout_storm"
    elif "503 backend read error" in lo and not raw_code:
        raw_code = "varnish_503"
    elif "profile_links_empty" in lo and not raw_code:
        raw_code = "profile_links_empty"
    elif "no_finished_singles" in lo and not raw_code:
        raw_code = "no_finished_singles"
    elif "history_source_unavailable" in lo and not raw_code:
        raw_code = "history_source_unavailable"

    source_blocked_codes = {"varnish_503", "cloudflare_block", "source_blocked"}
    stats_absent_codes = {
        "stats_absent",
        "profile_links_empty",
        "no_finished_singles",
        "stats_absent_likely",
        "stats_panel_absent",
        "stats_rows_absent",
        "stats_not_provided",
    }
    stats_not_loaded_codes = {
        "stats_not_loaded",
        "statistics_tab_unreachable",
        "consent_overlay_blocked",
        "dom_extract_timeout",
        "period_select_failed",
        "scrape_eval_timeout",
        "rows_not_ready",
        "navigation_failed",
        "api_history_fetch_failed",
    }
    code = raw_code
    if raw_code in source_blocked_codes or "cloudflare" in lo or "503 backend read error" in lo:
        code = "source_blocked"
    elif raw_code == "timeout_storm":
        code = "timeout_storm"
    elif raw_code in stats_absent_codes:
        code = "stats_absent"
    elif raw_code in stats_not_loaded_codes:
        code = "stats_not_loaded"
    if raw_code == "history_source_unavailable":
        code = "history_source_unavailable"
    if code:
        ex_text = re.sub(r"\bcode=[a-z0-9_]+\b\s*", "", ex_text, flags=re.I).strip()
        ex_text = re.sub(r"\s{2,}", " ", ex_text).strip()
    return code, ex_text


def _skip_reason_parts(
    skip_reasons: Dict[str, int],
    *,
    base_keys: Tuple[str, ...],
    include_codes: bool = True,
    code_limit: int = 4,
) -> List[str]:
    parts: List[str] = []
    for k in base_keys:
        v = int(skip_reasons.get(k, 0) or 0)
        if v > 0:
            parts.append(f"{k}={v}")
    hsua = int(skip_reasons.get("analyze_failed_history_source_unavailable", 0) or 0)
    if hsua > 0 and "history_source_unavailable" not in base_keys:
        parts.append(f"history_source_unavailable={hsua}")
    if include_codes:
        code_parts = sorted(
            ((k, int(v)) for k, v in skip_reasons.items() if str(k).startswith("analyze_failed_") and int(v) > 0),
            key=lambda kv: int(kv[1]),
            reverse=True,
        )
        if code_parts:
            top = ", ".join(f"{k.replace('analyze_failed_', '')}={v}" for k, v in code_parts[: max(1, int(code_limit))])
            parts.append(f"codes[{top}]")
    return parts
