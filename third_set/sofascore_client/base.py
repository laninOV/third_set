"""Base Sofascore URL/config helpers."""

from __future__ import annotations

import os
import re
from typing import Any

from playwright.async_api import Page

SOFASCORE_TENNIS_URL = "https://www.sofascore.com/en-us/tennis"
SOFASCORE_API_BASE = "https://www.sofascore.com/api/v1"
SOFASCORE_TENNIS_LIVE_API = f"{SOFASCORE_API_BASE}/sport/tennis/events/live"


_SOFASCORE_LOCALE_SEG_RE = re.compile(r"^/[a-z]{2}(?:-[a-z]{2})?(?=/|$)", re.I)


class SofascoreError(RuntimeError):
    pass


def _forced_sofascore_base() -> str:
    raw = (os.getenv("THIRDSET_SOFASCORE_LOCALE") or "en-us").strip().lower()
    if re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", raw):
        return f"https://www.sofascore.com/{raw}"
    return "https://www.sofascore.com/en-us"


def _normalize_sofascore_url(url: str) -> str:
    raw = (url or "").strip()
    if not raw:
        return _forced_sofascore_base()
    base = _forced_sofascore_base()
    m_abs = re.match(r"^https?://www\.sofascore\.com(?P<path>/[^?#]*)?(?P<tail>[?#].*)?$", raw, re.I)
    if m_abs:
        path = m_abs.group("path") or ""
        tail = m_abs.group("tail") or ""
        path = _SOFASCORE_LOCALE_SEG_RE.sub("", path, count=1)
        return f"{base}{path}{tail}"
    if raw.startswith("/"):
        path = _SOFASCORE_LOCALE_SEG_RE.sub("", raw, count=1)
        return f"{base}{path}"
    return raw


def _sofascore_base_from_url(url: str) -> str:
    return _forced_sofascore_base()


def _tennis_url_from_page(page: Page) -> str:
    try:
        cur = page.url or ""
    except Exception:
        cur = ""
    base = _sofascore_base_from_url(cur) or _forced_sofascore_base()
    return f"{base}/tennis"
