from __future__ import annotations

import asyncio
import re
import os
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Page

from third_set.browser_utils import disable_network_cache


class DomStatsError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        step: Optional[str] = None,
        code: Optional[str] = None,
        attempt: Optional[int] = None,
        diag: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.step = step
        self.code = code
        self.attempt = attempt
        self.diag = diag or {}
        super().__init__(message)


def _dom_err(
    *,
    step: str,
    code: str,
    message: str,
    attempt: Optional[int] = None,
    diag: Optional[Dict[str, Any]] = None,
) -> DomStatsError:
    parts = [f"step={step}", f"code={code}"]
    if isinstance(attempt, int):
        parts.append(f"attempt={attempt}")
    if message:
        parts.append(message)
    return DomStatsError(" ".join(parts), step=step, code=code, attempt=attempt, diag=(diag or {}))


try:
    _NAV_TIMEOUT_MS = int(os.getenv("THIRDSET_NAV_TIMEOUT_MS", "45000"))
except Exception:
    _NAV_TIMEOUT_MS = 45_000
try:
    _UI_TIMEOUT_MS = int(os.getenv("THIRDSET_UI_TIMEOUT_MS", "10000"))
except Exception:
    _UI_TIMEOUT_MS = 10_000
try:
    _SLOW_LOAD_MS = int(os.getenv("THIRDSET_SLOW_LOAD_MS", "0"))
except Exception:
    _SLOW_LOAD_MS = 0
try:
    _WAIT_STATS_MS = int(os.getenv("THIRDSET_WAIT_STATS_MS", "0"))
except Exception:
    _WAIT_STATS_MS = 0
try:
    _HUMAN_PAUSE_MS = int(os.getenv("THIRDSET_HUMAN_PAUSE_MS", "0"))
except Exception:
    _HUMAN_PAUSE_MS = 0
try:
    _DOM_READY_POLL_MS = int(os.getenv("THIRDSET_DOM_READY_POLL_MS", "180"))
except Exception:
    _DOM_READY_POLL_MS = 180
_DOM_READY_POLL_MS = max(80, min(600, _DOM_READY_POLL_MS))
_ROWCOUNT_STATS_PRESENT_THRESHOLD = 8


def _dbg(msg: str) -> None:
    if os.getenv("THIRDSET_DEBUG") in ("1", "true", "yes"):
        print(f"[dom] {msg}", flush=True)


_NO_LIMITS_MODE = os.getenv("THIRDSET_NO_LIMITS", "0").strip().lower() not in ("0", "false", "no")


async def _safe_goto(page: Page, url: str, *, timeout_ms: int) -> None:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        return
    except Exception:
        pass
    try:
        await page.goto(url, wait_until="commit", timeout=timeout_ms)
    except Exception:
        pass

async def _is_varnish_503(page: Page) -> bool:
    """
    Sofascore sometimes returns a Varnish error page:
      "Error 503 backend read error"
    This is transient and should be retried instead of treated as "no stats".
    """
    try:
        t = (await page.title()) or ""
        if "503" in t and "backend" in t.lower():
            return True
    except Exception:
        pass
    try:
        # Fast text checks; avoid heavy DOM extraction.
        loc = page.locator("text=/Error\\s*503/i")
        if await loc.count():
            return True
        loc2 = page.locator("text=/backend read error/i")
        if await loc2.count():
            return True
    except Exception:
        pass
    return False


_GROUP_MAP = {
    # RU
    "Подача": "Service",
    "Возврат": "Return",
    "Очки": "Points",
    "Игр": "Games",
    "Разное": "Miscellaneous",
    # EN
    "Service": "Service",
    "Return": "Return",
    "Points": "Points",
    "Games": "Games",
    "Miscellaneous": "Miscellaneous",
}


_ITEM_MAP: Dict[Tuple[str, str], Tuple[str, str]] = {
    # Service
    ("Service", "Эйсы"): ("Service", "Aces"),
    ("Service", "Aces"): ("Service", "Aces"),
    ("Service", "Двойные ошибки"): ("Service", "Double faults"),
    ("Service", "Double faults"): ("Service", "Double faults"),
    ("Service", "Первая подача"): ("Service", "First serve"),
    ("Service", "1-я подача"): ("Service", "First serve"),
    ("Service", "First serve"): ("Service", "First serve"),
    ("Service", "Вторая подача"): ("Service", "Second serve"),
    ("Service", "2-я подача"): ("Service", "Second serve"),
    ("Service", "Поинты первой подачи"): ("Service", "First serve points"),
    ("Service", "First serve points"): ("Service", "First serve points"),
    ("Service", "Поинты второй подачи"): ("Service", "Second serve points"),
    ("Service", "Second serve points"): ("Service", "Second serve points"),
    ("Service", "Сыгранные подачи"): ("Games", "Service games played"),
    ("Service", "Брейк-поинты защищены"): ("Service", "Break points saved"),
    ("Service", "Брейк-пойнты защищены"): ("Service", "Break points saved"),
    ("Service", "Брейк-пойнты отыграны"): ("Service", "Break points saved"),
    ("Service", "Брейк-поинты отыграны"): ("Service", "Break points saved"),
    ("Service", "Break points saved"): ("Service", "Break points saved"),
    # Return
    ("Return", "Поинты первой подачи"): ("Return", "First serve return points"),
    ("Return", "First serve return points"): ("Return", "First serve return points"),
    ("Return", "Поинты второй подачи"): ("Return", "Second serve return points"),
    ("Return", "Second serve return points"): ("Return", "Second serve return points"),
    ("Return", "Сыграно ответных"): ("Return", "Return games played"),
    ("Return", "Сыграно ответных матчей"): ("Return", "Return games played"),
    ("Return", "Сыгранные ответные геймы"): ("Return", "Return games played"),
    ("Return", "Сыгранные геймы на приёме"): ("Return", "Return games played"),
    ("Return", "Сыгранные геймы на приеме"): ("Return", "Return games played"),
    ("Return", "Return games played"): ("Return", "Return games played"),
    ("Return", "Брейк-поинты использованы"): ("Return", "Break points converted"),
    ("Return", "Брейк-пойнты использованы"): ("Return", "Break points converted"),
    ("Return", "Брейк-пойнты реализованы"): ("Return", "Break points converted"),
    ("Return", "Брейк-поинты реализованы"): ("Return", "Break points converted"),
    ("Return", "Break points converted"): ("Return", "Break points converted"),
    # Points
    ("Points", "Очки на подаче"): ("Points", "Service points won"),
    ("Points", "Service points won"): ("Points", "Service points won"),
    ("Points", "Очки на приёме"): ("Points", "Receiver points won"),
    ("Points", "Очки на приеме"): ("Points", "Receiver points won"),
    ("Points", "Очки на своей подаче"): ("Points", "Service points won"),
    ("Points", "Очки на подаче соперника"): ("Points", "Receiver points won"),
    ("Points", "Очки, выигранные на подаче соперника"): ("Points", "Receiver points won"),
    ("Points", "Receiver points won"): ("Points", "Receiver points won"),
    ("Points", "Очки, выигранные на своей подаче"): ("Points", "Service points won"),
    ("Points", "Очки, выигранные на приёме"): ("Points", "Receiver points won"),
    ("Points", "Очки, выигранные на приеме"): ("Points", "Receiver points won"),
    ("Points", "Максимум очков подряд"): ("Miscellaneous", "Max points in a row"),
    # Games
    ("Games", "Выигранные подачи"): ("Games", "Service games won"),
    ("Games", "Service games won"): ("Games", "Service games won"),
    ("Games", "Выигранные подачи соперника"): ("Games", "Return games won"),
    ("Games", "Максимум геймов подряд"): ("Miscellaneous", "Max games in a row"),
    # Misc
    ("Miscellaneous", "Тай-брейки"): ("Miscellaneous", "Tiebreaks"),
    ("Miscellaneous", "Tiebreaks"): ("Miscellaneous", "Tiebreaks"),
}


async def _is_cloudflare_block(page: Page) -> bool:
    try:
        title = (await page.title()) or ""
    except Exception:
        title = ""
    t = title.lower()
    if "attention required" in t or "cloudflare" in t:
        return True
    try:
        # Common CF / bot-check strings
        for txt in (
            "Attention Required",
            "Checking your browser",
            "DDoS protection by Cloudflare",
            "Please enable cookies",
        ):
            loc = page.get_by_text(txt, exact=False)
            if await loc.count() and await loc.first.is_visible():
                return True
    except Exception:
        pass
    return False


async def _consent_overlay_block_state(page: Page) -> Dict[str, Any]:
    """
    Detect whether consent/captcha overlays are still intercepting interactions.
    """
    try:
        state = await page.evaluate(
            """
            () => {
              function vis(el) {
                if (!el) return false;
                const st = window.getComputedStyle(el);
                if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                const r = el.getBoundingClientRect();
                return r.width > 1 && r.height > 1;
              }
              const sels = ['.fc-consent-root', '#fc-consent-root', '#onetrust-banner-sdk', '.fc-dialog-overlay'];
              let root = null;
              let selector = null;
              for (const s of sels) {
                const el = document.querySelector(s);
                if (vis(el)) {
                  root = el;
                  selector = s;
                  break;
                }
              }
              const recaptchaVisible = Array.from(document.querySelectorAll('iframe[src*=\"recaptcha\"]')).some((fr) => {
                if (!fr) return false;
                const st = window.getComputedStyle(fr);
                if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                const r = fr.getBoundingClientRect();
                return r.width > 8 && r.height > 8;
              });
              if (!root) {
                return { blocked: false, visible: false, recaptcha: recaptchaVisible };
              }
              const st = window.getComputedStyle(root);
              const r = root.getBoundingClientRect();
              const blocks = (st.pointerEvents || '').toLowerCase() !== 'none' && r.width > 120 && r.height > 120;
              const text = (root.innerText || root.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 220);
              return {
                blocked: Boolean(blocks || (recaptchaVisible && r.width > 120 && r.height > 120)),
                visible: true,
                selector: selector || '',
                pointerEvents: st.pointerEvents || '',
                width: Math.round(r.width || 0),
                height: Math.round(r.height || 0),
                recaptcha: Boolean(recaptchaVisible),
                text,
              };
            }
            """
        )
        return state if isinstance(state, dict) else {"blocked": False, "visible": False}
    except Exception:
        return {"blocked": False, "visible": False}


async def _wait_for_stats_rows(page: Page, *, timeout_ms: int) -> bool:
    """
    Wait until stats rows (or headers) are present in DOM.
    This avoids scraping too early when the tab is visible but data is still loading.
    """
    try:
        await page.wait_for_function(
            """
            () => {
              const root = document.querySelector('#tabpanel-statistics') || document;
              const rows = root.querySelectorAll(
                'div.d_flex.ai_center.jc_space-between, div.d_flex.jc_space-between, div.jc_space-between, div[class*="statisticsRow"], div[class*="statRow"]'
              ).length;
              const headers = root.querySelectorAll('span[class*="textStyle_display"]').length;
              return rows >= 2 || headers >= 2;
            }
            """,
            timeout=timeout_ms,
            polling=_DOM_READY_POLL_MS,
        )
        return True
    except Exception:
        return False


async def _probe_stats_availability(page: Page) -> Dict[str, Any]:
    try:
        return await page.evaluate(
            """
            () => {
              function norm(s){return (s||'').replace(/\\s+/g,' ').trim();}
              function low(s){return norm(s).toLowerCase();}
              function isVisible(el) {
                if (!el) return false;
                const st = window.getComputedStyle(el);
                if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                const r = el.getClientRects();
                return !!(r && r.length && r[0].width > 1 && r[0].height > 1);
              }
              const root = document.querySelector('#tabpanel-statistics') || document;
              const texts = Array.from(root.querySelectorAll('button,a,[role="tab"],[role="button"],[role="link"],div,span,p'))
                .filter(isVisible)
                .map(el => norm(el.innerText || el.textContent || ''))
                .filter(Boolean)
                .slice(0, 1000);
              const txtAll = low(root.innerText || root.textContent || '');
              const hasEmptyMarker =
                /нет статистик|no stat|statistics not available|no data available|данные отсутств|no match statistics/.test(txtAll);
              const hasAll = texts.some(t => /^(все|all)$/i.test(t));
              const rx1 = /(1\\s*[-\\u2010-\\u2014]?\\s*й(\\s*сет)?|1st(\\s*set)?|set\\s*1|1\\s*set|^\\s*1\\s*$)/i;
              const rx2 = /(2\\s*[-\\u2010-\\u2014]?\\s*й(\\s*сет)?|2nd(\\s*set)?|set\\s*2|2\\s*set|^\\s*2\\s*$)/i;
              const has1st = texts.some(t => rx1.test(t));
              const has2nd = texts.some(t => rx2.test(t));
              const periodTokens = texts.filter(t => rx1.test(t) || rx2.test(t)).slice(0, 12);
              const groups = Array.from(root.querySelectorAll('h1,h2,h3,h4,span,div,p'))
                .filter(isVisible)
                .map(el => norm(el.textContent || ''))
                .filter(t => /^(Подача|Очки|Возврат|Игр|Разное|Service|Points|Return|Games|Miscellaneous)$/i.test(t))
                .slice(0, 20);
              const rowCount = root.querySelectorAll(
                'div.d_flex.ai_center.jc_space-between, div.d_flex.jc_space-between, div.jc_space-between, div[class*="statisticsRow"], div[class*="statRow"]'
              ).length;
              const rowsSuggestPresent = rowCount >= 8;
              const absentLikely = !!(
                hasEmptyMarker ||
                ((!has1st || !has2nd) && groups.length === 0 && hasAll && !rowsSuggestPresent)
              );
              return {
                absent_likely: absentLikely,
                has_empty_marker: hasEmptyMarker,
                has_all: hasAll,
                has_1st: has1st,
                has_2nd: has2nd,
                groups: groups,
                groups_detected: groups.length,
                period_tokens_detected: periodTokens,
                rowCount: rowCount,
              };
            }
            """
        )
    except Exception:
        return {}


async def _wait_page_ready(page: Page, *, timeout_ms: int) -> None:
    """
    Wait for page hydration before interacting with tabs/stats.
    Sofascore often finishes DOMContentLoaded early while tab controls are still mounting.
    """
    t = max(1200, int(timeout_ms))
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=min(t, 3_500))
    except Exception:
        pass
    try:
        await page.wait_for_function("() => document.readyState === 'complete'", timeout=min(t, 2_500))
    except Exception:
        pass
    # Sofascore often keeps background requests open; keep this short best-effort.
    try:
        await page.wait_for_load_state("networkidle", timeout=min(t, 1_200))
    except Exception:
        pass
    # Small settle time to let async widgets mount.
    try:
        await page.wait_for_timeout(max(120, _DOM_READY_POLL_MS))
    except Exception:
        pass


def _period_text_markers(period_code: str) -> List[str]:
    code = (period_code or "").upper()
    if code == "ALL":
        return ["all", "все", "вcе"]
    if code == "1ST":
        return ["1st", "1-й", "1"]
    if code == "2ND":
        return ["2nd", "2-й", "2"]
    if code == "3RD":
        return ["3rd", "3-й", "3"]
    return []


async def _wait_period_selected(page: Page, *, period_code: str, timeout_ms: int) -> bool:
    """
    Best-effort check that the requested period control is selected/active.
    Sofascore changes markup often, so we treat this as advisory and keep stats rows
    as the primary readiness signal.
    """
    markers = _period_text_markers(period_code)
    if not markers:
        return False
    try:
        await page.wait_for_function(
            """
            (arg) => {
              const marks = (arg?.markers || []).map((x) => String(x || '').toLowerCase());
              if (!marks.length) return false;
              const root = document.querySelector('#tabpanel-statistics') || document;
              const controls = Array.from(
                root.querySelectorAll(
                  'button, a, [role="tab"], [role="button"], [aria-selected], [aria-pressed], [class*="button"]'
                )
              );
              const norm = (s) => String(s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              for (const el of controls) {
                const txt = norm(el.innerText || el.textContent || '');
                if (!txt) continue;
                const hit = marks.some((m) => txt === m || txt.startsWith(m) || txt.includes(m));
                if (!hit) continue;
                const ariaSel = String(el.getAttribute('aria-selected') || '').toLowerCase();
                const ariaPressed = String(el.getAttribute('aria-pressed') || '').toLowerCase();
                const cls = norm(el.className || '');
                if (ariaSel === 'true' || ariaPressed === 'true') return true;
                if (cls.includes('variant_filled') || cls.includes('active') || cls.includes('selected')) return true;
              }
              return false;
            }
            """,
            {"markers": markers},
            timeout=timeout_ms,
        )
        return True
    except Exception:
        return False


async def _dismiss_overlays(page: Page) -> None:
    _dbg("dismiss_overlays")
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass

    # Age verification modal (Sofascore gambling ads). Choose "25 years or older" and confirm.
    try:
        age_dialog = page.locator(".modalRecipe__contentWrapper").filter(
            has_text=re.compile(r"Age Verification|Проверка возраста|Подтверждение возраста", re.I)
        )
        if await age_dialog.count() and await age_dialog.first.is_visible():
            dlg = age_dialog.first
            # Click the 25+ radio input (by id) or by label text.
            try:
                inp = dlg.locator("input[id*='25_or_older']").first
                if await inp.count():
                    await inp.click(timeout=1500, force=True)
                    await page.wait_for_timeout(250)
            except Exception:
                pass
            try:
                label = dlg.locator("label:has-text(\"25 years or older\")").first
                if await label.count():
                    await label.click(timeout=1500, force=True)
                    await page.wait_for_timeout(250)
            except Exception:
                pass
            # Click confirm button (becomes enabled after selecting).
            try:
                btn = dlg.locator("button:has-text(\"Подтвердить\")").first
                if await btn.count():
                    await btn.click(timeout=1500, force=True)
                    await page.wait_for_timeout(350)
            except Exception:
                pass
    except Exception:
        pass

    # Common consent / modal roots: try clicking "Accept" style buttons inside.
    for root_sel in (".fc-consent-root", "#fc-consent-root", "#onetrust-banner-sdk", "div[role='dialog']"):
        try:
            root = page.locator(root_sel)
            if not (await root.count()):
                continue
            if not (await root.first.is_visible()):
                continue
            # Sofascore often shows a GDPR/consent overlay (fc-consent-root) that intercepts clicks.
            # Try common "positive" buttons in priority order.
            for rx in (
                r"^consent$",
                r"accept all|принять все|разрешить все",
                r"confirm choices|подтвердить|подтверждаю|сохранить",
                r"agree|i agree|согласен|согласиться|соглашаюсь|ok|okay|continue|продолжить|понятно",
                r"close|закрыть",
            ):
                try:
                    btn = root.get_by_role("button", name=re.compile(rx, re.I))
                    if await btn.count() and await btn.first.is_visible():
                        await btn.first.click(timeout=2000, force=True)
                        await page.wait_for_timeout(350)
                except Exception:
                    pass

            # If it still blocks clicks, try a last-resort "any visible button" click inside the root.
            # (Some banners render custom buttons without accessible names.)
            try:
                buttons = root.locator("button")
                if await buttons.count():
                    for i in range(min(6, await buttons.count())):
                        b = buttons.nth(i)
                        try:
                            if await b.is_visible():
                                await b.click(timeout=1200, force=True)
                                await page.wait_for_timeout(250)
                                break
                        except Exception:
                            continue
            except Exception:
                pass
        except Exception:
            pass

    # Global fallbacks (non-exact; names sometimes include extra text).
    for rx in (
        r"accept all|принять все",
        r"confirm choices|подтвердить",
        r"agree|согласен|ok",
        r"close|закрыть",
    ):
        try:
            loc = page.get_by_role("button", name=re.compile(rx, re.I))
            if await loc.count():
                await loc.first.click(timeout=1500)
                await page.wait_for_timeout(350)
        except Exception:
            pass

    # Wait a moment for consent overlays to stop intercepting clicks.
    try:
        ov = page.locator(".fc-consent-root, #fc-consent-root, #onetrust-banner-sdk")
        if await ov.count():
            try:
                await ov.first.wait_for(state="hidden", timeout=2500)
            except Exception:
                pass
        # Debug visibility if still there
        try:
            if await ov.count() and await ov.first.is_visible():
                _dbg("consent overlay still visible after dismiss")
                # Last resort: stop it from intercepting pointer events (keeps DOM-only approach).
                try:
                    await page.evaluate(
                        """() => {
                          const el = document.querySelector('.fc-consent-root, #fc-consent-root, #onetrust-banner-sdk');
                          if (el) el.style.pointerEvents = 'none';
                          const overlay = document.querySelector('.fc-dialog-overlay');
                          if (overlay) overlay.style.pointerEvents = 'none';
                        }"""
                    )
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass


async def _click_statistics_tab(page: Page, *, wait_stats_ms: int) -> None:
    await _wait_page_ready(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms + 4_000)
    await _dismiss_overlays(page)
    ov_state = await _consent_overlay_block_state(page)
    if bool(ov_state.get("blocked")):
        raise _dom_err(
            step="open_statistics_tab",
            code="consent_overlay_blocked",
            message="Consent/captcha overlay still blocks interactions before opening statistics tab",
            diag={"overlay": ov_state},
        )
    if await _is_cloudflare_block(page):
        raise DomStatsError("Cloudflare блокирует страницу (нужно реальное окно/куки)")
    _dbg("click_statistics_tab")
    # Sofascore renders match sub-tabs asynchronously.
    # Avoid sequential RU->EN waits (can cost ~12s on each /en-us page).
    try:
        await page.wait_for_function(
            """
            () => {
              const norm = (s) => String(s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
              const want = new Set(['статистика', 'statistics']);
              const els = document.querySelectorAll('a,button,[role="tab"],[role="button"],[role="link"],div,span');
              for (const el of els) {
                if (want.has(norm(el.innerText || el.textContent || ''))) return true;
              }
              return false;
            }
            """,
            timeout=6_000,
            polling=max(80, _DOM_READY_POLL_MS),
        )
    except Exception:
        pass
    await _dismiss_overlays(page)
    ov_state = await _consent_overlay_block_state(page)
    if bool(ov_state.get("blocked")):
        raise _dom_err(
            step="open_statistics_tab",
            code="consent_overlay_blocked",
            message="Consent/captcha overlay still blocks interactions after dismiss",
            diag={"overlay": ov_state},
        )

    rx_ru = re.compile(r"^\\s*статистика\\s*$", re.I)
    rx_en = re.compile(r"^\\s*statistics\\s*$", re.I)

    # Prefer tablists (Detali/Statistika/Matches).
    for rx in (rx_ru, rx_en):
        try:
            tablists = page.get_by_role("tablist")
            if await tablists.count():
                for i in range(min(6, await tablists.count())):
                    tl = tablists.nth(i)
                    tab = tl.get_by_role("tab", name=rx)
                    if await tab.count():
                        await tab.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                        ok = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms)
                        if not ok:
                            raise DomStatsError("step=statistics_tab: stats rows not ready after tab click")
                        return
        except Exception:
            pass

    # Fall back: search within main content for link/button/tab containing "Статистика".
    roots = []
    try:
        main = page.locator("main")
        if await main.count():
            roots.append(main.first)
    except Exception:
        pass
    roots.append(page)

    for root in roots:
        for name_rx in (rx_ru, rx_en):
            for kind in ("tab", "link", "button"):
                try:
                    loc = root.get_by_role(kind, name=name_rx)
                    if not (await loc.count()):
                        continue
                    await loc.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                    ok = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms)
                    if not ok:
                        raise DomStatsError("step=statistics_tab: stats rows not ready after role click")
                    return
                except Exception:
                    continue
            # CSS fallbacks (Sofascore sometimes doesn't expose roles)
            try:
                loc = root.locator("a,button,[role='tab'],[role='button'],[role='link']").filter(has_text=name_rx)
                if await loc.count():
                    await loc.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                    ok = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms)
                    if not ok:
                        raise DomStatsError("step=statistics_tab: stats rows not ready after css click")
                    return
            except Exception:
                pass

    # Final fallback: pure DOM scan for an element whose visible text equals "Статистика"/"Statistics".
    # This avoids accidentally clicking banners that merely contain the word "статистика".
    try:
        clicked = await page.evaluate(
            """
            () => {
              function norm(s){return (s||'').replace(/\\s+/g,' ').trim().toLowerCase();}
              const want = new Set(['статистика','statistics']);
              const els = Array.from(document.querySelectorAll('a,button,[role=\"tab\"],[role=\"button\"],[role=\"link\"],div,span'));
              for (const el of els) {
                const t = norm(el.innerText || el.textContent || '');
                if (!want.has(t)) continue;
                try { el.click(); return true; } catch (e) {}
              }
              return false;
            }
            """
        )
        if clicked:
            ok = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms)
            if not ok:
                raise DomStatsError("step=statistics_tab: stats rows not ready after dom click")
            return
    except Exception:
        pass

    # JS-click fallback (bypasses consent overlays intercepting pointer events).
    # Sofascore often renders the statistics tab as <a href="#tab:statistics">Статистика</a>.
    try:
        _dbg("click_statistics_tab: try js click a[href='#tab:statistics']")
        clicked = await page.evaluate(
            """() => {
              const a = document.querySelector('a[href=\"#tab:statistics\"], [role=\"tab\"][href=\"#tab:statistics\"]');
              if (!a) return false;
              try { a.click(); return true; } catch (e) { return false; }
            }"""
        )
        if clicked:
            ok = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms)
            if not ok:
                raise DomStatsError("step=statistics_tab: stats rows not ready after js hash tab")
            return
    except Exception:
        pass

    # Hash-based fallback: some pages have the tab link but clicks are intercepted by consent overlays.
    # Switching the hash does not require a click and usually reveals #tabpanel-statistics.
    try:
        _dbg("click_statistics_tab: try hash #tab:statistics")
        await page.evaluate(
            """() => {
              try {
                if (location.hash !== '#tab:statistics') location.hash = '#tab:statistics';
                window.dispatchEvent(new HashChangeEvent('hashchange'));
              } catch (e) {}
            }"""
        )
        await _dismiss_overlays(page)
        ok = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms)
        if ok:
            return
    except Exception:
        pass
    ov_state = await _consent_overlay_block_state(page)
    if bool(ov_state.get("blocked")):
        raise _dom_err(
            step="open_statistics_tab",
            code="consent_overlay_blocked",
            message="Consent/captcha overlay blocked statistics tab",
            diag={"overlay": ov_state},
        )
    raise DomStatsError("Не удалось открыть вкладку 'Статистика' (DOM)")


async def _select_period(page: Page, period_code: str, *, wait_stats_ms: int) -> bool:
    await _dismiss_overlays(page)
    if await _is_cloudflare_block(page):
        raise DomStatsError("Cloudflare блокирует страницу (нужно реальное окно/куки)")
    _dbg(f"select_period {period_code}")
    # Period chips/tabs are rendered as "ВСЕ / 1-й / 2-й / 3-й" (RU) or "All / 1st set..." (EN).
    # Text can vary by hyphen type and case, so use regexes and scope to the statistics tabpanel when possible.
    def rx_for(code: str) -> List[re.Pattern]:
        dash = r"[-\u2010\u2011\u2012\u2013\u2014]"
        # Some pages label set tabs as "1-й сет" (or similar); accept optional "сет/set".
        set_word = r"(?:\s*(?:сет|set))?"
        if code == "ALL":
            return [
                re.compile(r"^(все|вcе|all)$", re.I),  # note: sometimes uppercase ВСЕ
                re.compile(r"^all$", re.I),
            ]
        if code == "1ST":
            return [
                re.compile(rf"1\s*{dash}\s*й{set_word}", re.I),
                re.compile(r"1st(?:\s*set)?", re.I),
                re.compile(r"set\s*1", re.I),
                re.compile(r"1\s*set", re.I),
                re.compile(r"^\s*1\s*$", re.I),
            ]
        if code == "2ND":
            return [
                re.compile(rf"2\s*{dash}\s*й{set_word}", re.I),
                re.compile(r"2nd(?:\s*set)?", re.I),
                re.compile(r"set\s*2", re.I),
                re.compile(r"2\s*set", re.I),
                re.compile(r"^\s*2\s*$", re.I),
            ]
        if code == "3RD":
            return [
                re.compile(rf"3\s*{dash}\s*й{set_word}", re.I),
                re.compile(r"3rd(?:\s*set)?", re.I),
                re.compile(r"^\s*3\s*$", re.I),
            ]
        return []

    roots = []
    try:
        tabpanel = page.locator("#tabpanel-statistics")
        if await tabpanel.count():
            roots.append(tabpanel.first)
    except Exception:
        pass
    roots.append(page)

    patterns = rx_for(period_code)
    if not patterns:
        raise DomStatsError(f"step=period_select:{period_code}: unsupported period code")
    if period_code in ("1ST", "2ND"):
        avail = await _probe_stats_availability(page)
        row_count = int(avail.get("rowCount") or 0) if isinstance(avail, dict) else 0
        groups_detected = int(avail.get("groups_detected") or 0) if isinstance(avail, dict) else 0
        if (
            isinstance(avail, dict)
            and bool(avail.get("absent_likely"))
            and row_count < _ROWCOUNT_STATS_PRESENT_THRESHOLD
            and groups_detected <= 0
        ):
            raise _dom_err(
                step=f"select_period_{period_code.lower()}",
                code="stats_not_provided",
                message=f"statistics absent likely: {avail}",
            )

    async def _after_click() -> bool:
        selected = await _wait_period_selected(page, period_code=period_code, timeout_ms=min(4500, _UI_TIMEOUT_MS + 2000))
        ok = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats_ms)
        if not ok:
            raise DomStatsError(f"step=period_select:{period_code}: stats rows not ready")
        if not selected:
            _dbg(f"period {period_code}: active marker not observed; proceeding by rows readiness")
        return True

    async def try_click_within(root) -> bool:
        # 1) role=tab / role=button with regex name
        for kind in ("tab", "button"):
            for rx in patterns:
                try:
                    loc = root.get_by_role(kind, name=rx)
                    if await loc.count():
                        await loc.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                        return await _after_click()
                except Exception:
                    continue
        # 2) plain buttons/tabs filtered by text
        for rx in patterns:
            try:
                loc = root.locator("button, a, [role='tab'], [role='button'], [role='link']").filter(has_text=rx)
                if await loc.count():
                    await loc.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                    return await _after_click()
            except Exception:
                continue
        # 3) any text node (click nearest clickable parent)
        for rx in patterns:
            try:
                txt = root.get_by_text(rx)
                if not (await txt.count()):
                    continue
                el = txt.first
                try:
                    await el.click(timeout=_UI_TIMEOUT_MS, force=True)
                except Exception:
                    # try clicking parent node
                    try:
                        parent = el.locator("xpath=..")
                        await parent.click(timeout=_UI_TIMEOUT_MS, force=True)
                    except Exception:
                        continue
                return await _after_click()
            except Exception:
                continue
        # 4) JS fallback: click any element whose visible text matches.
        # Sofascore sometimes renders these as non-button DIVs with click handlers.
        try:
            clicked = await page.evaluate(
                """
                (arg) => {
                  const { patterns } = arg;
                  function norm(s){return (s||'').replace(/\\s+/g,' ').trim().toLowerCase();}
                  const root = document.querySelector('#tabpanel-statistics') || document.querySelector('main') || document;
                  const els = Array.from(root.querySelectorAll('button,a,[role=\"tab\"],[role=\"button\"],[role=\"link\"],div,span'))
                    .filter(el => el && (el.innerText || el.textContent));
                  for (const el of els) {
                    const t = norm(el.innerText || el.textContent || '');
                    if (!t) continue;
                    for (const pat of patterns) {
                      try {
                        const re = new RegExp(pat, 'i');
                        if (re.test(t)) {
                          el.click();
                          return true;
                        }
                      } catch (e) {}
                    }
                  }
                  return false;
                }
                """,
                {"patterns": [p.pattern for p in patterns]},
            )
            if clicked:
                return await _after_click()
        except Exception:
            pass
        return False

    for root in roots:
        if await try_click_within(root):
            return True

    # Try dropdown/segmented control:
    # many tennis pages show only "ВСЕ/All" as a chip; clicking it reveals "1-й/2-й/3-й".
    if period_code != "ALL":
        base_rx = re.compile(r"^(все|all)$", re.I)
        # Try globally, because the dropdown menu can be rendered in a portal outside the stats container.
        # Also include non-button controls (Sofascore sometimes uses clickable DIVs).
        dropdown = page.locator("button, a, [role='button'], [role='tab'], [role='link'], div, span").filter(has_text=base_rx)
        n = await dropdown.count()
        _dbg(f"select_period {period_code}: dropdown candidates={n}")
        # Prefer the dropdown that is closest to the stats table (near group header like "Подача/Service").
        anchor_y: Optional[float] = None
        try:
            for lab in ("Подача", "Service"):
                h = page.get_by_text(lab, exact=True)
                if await h.count():
                    box = await h.first.bounding_box()
                    if box and isinstance(box.get("y"), (int, float)):
                        anchor_y = float(box["y"])
                        break
        except Exception:
            anchor_y = None

        order = list(range(min(n, 6)))
        if anchor_y is not None and n:
            scored = []
            for i in order:
                try:
                    box = await dropdown.nth(i).bounding_box()
                except Exception:
                    box = None
                if not box or not isinstance(box.get("y"), (int, float)):
                    continue
                scored.append((abs(float(box["y"]) - anchor_y), i))
            if scored:
                scored.sort(key=lambda t: t[0])
                order = [i for _d, i in scored]

        for i in order:
            try:
                cand = dropdown.nth(i)
                if not (await cand.is_visible()):
                    continue
                # Avoid huge container elements that incidentally contain "Все".
                try:
                    txt = (await cand.inner_text()).strip()
                    if len(txt) > 12:
                        continue
                except Exception:
                    pass
                try:
                    _dbg(f"select_period {period_code}: click dropdown[{i}] text={(await cand.inner_text()).strip()!r}")
                except Exception:
                    pass
                await cand.click(timeout=_UI_TIMEOUT_MS, force=True)
                # Wait by condition until period options appear (no fixed sleep).
                try:
                    await page.wait_for_function(
                        """
                        (arg) => {
                          const patterns = (arg?.patterns || []).map((x) => String(x || ''));
                          if (!patterns.length) return false;
                          const els = Array.from(document.querySelectorAll('button, a, [role="button"], [role="tab"], [role="link"], div, span'));
                          const txt = (s) => String(s || '').replace(/\\s+/g, ' ').trim();
                          return els.some((el) => {
                            const t = txt(el.innerText || el.textContent || '');
                            if (!t) return false;
                            return patterns.some((pat) => {
                              try { return new RegExp(pat, 'i').test(t); } catch (e) { return false; }
                            });
                          });
                        }
                        """,
                        {"patterns": [p.pattern for p in patterns]},
                        timeout=1_200,
                        polling=max(80, _DOM_READY_POLL_MS),
                    )
                except Exception:
                    pass
                for rx in patterns:
                    opt = page.locator("button, a, [role='button'], [role='tab'], [role='link'], div, span").filter(has_text=rx)
                    cnt = await opt.count()
                    if not cnt:
                        continue
                    _dbg(f"select_period {period_code}: opt rx={rx.pattern!r} count={cnt}")
                    # Click first visible option.
                    for j in range(min(cnt, 8)):
                        o = opt.nth(j)
                        try:
                            if await o.is_visible():
                                try:
                                    otxt = (await o.inner_text()).strip()
                                    if len(otxt) > 24:
                                        continue
                                except Exception:
                                    pass
                                try:
                                    _dbg(f"select_period {period_code}: click opt[{j}] text={(await o.inner_text()).strip()!r}")
                                except Exception:
                                    pass
                                await o.click(timeout=_UI_TIMEOUT_MS, force=True)
                                return await _after_click()
                        except Exception:
                            continue
                # If the menu opened but we didn't find the option, close it and try next.
                try:
                    await page.keyboard.press("Escape")
                except Exception:
                    pass
            except Exception:
                continue
    return False


async def extract_statistics_dom(
    page: Page,
    *,
    match_url: str,
    event_id: int,
    periods: Tuple[str, ...] = ("1ST", "2ND"),
    nav_timeout_ms: Optional[int] = None,
    wait_stats_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract per-period tennis statistics from Sofascore DOM ("Статистика" tab).

    Returns a pseudo-API JSON compatible with MatchSnapshot/stats_parser:
      { "statistics": [ { "period": "1ST", "groups": [ { "groupName": "...", "statisticsItems": [...] } ] } ] }
    """
    # Ensure we are on the correct match URL (history reuses a single tab).
    # Sofascore can redirect between locales (e.g. /ru/ -> /en-us/), so we match by the
    # locale-independent path `/tennis/match/<slug>/<customId>` rather than full prefix.
    target = match_url + f"#id:{event_id}"
    started_at = asyncio.get_running_loop().time()
    nav_timeout = int(nav_timeout_ms) if isinstance(nav_timeout_ms, int) and nav_timeout_ms > 0 else _NAV_TIMEOUT_MS
    try:
        step_retries = int(os.getenv("THIRDSET_DOM_STEP_RETRIES") or "2")
    except Exception:
        step_retries = 2
    step_retries = max(0, min(6, step_retries))
    try:
        step_backoff_ms = int(os.getenv("THIRDSET_DOM_STEP_BACKOFF_MS") or "350")
    except Exception:
        step_backoff_ms = 350
    step_backoff_ms = max(50, min(5000, step_backoff_ms))
    varnish_seen = False

    def _elapsed_ms() -> int:
        return int(max(0.0, asyncio.get_running_loop().time() - started_at) * 1000.0)

    def _match_identity(url: str) -> str:
        try:
            base = (url or "").split("#", 1)[0]
        except Exception:
            base = url or ""
        m = re.search(r"/tennis/match/[^/]+/[^/?#]+", base)
        return m.group(0) if m else ""

    want_id = _match_identity(match_url)
    have_id = _match_identity(page.url or "")
    on_same_match = bool(want_id) and want_id == have_id
    has_event_fragment = f"id:{event_id}" in (page.url or "")

    if not on_same_match or not has_event_fragment:
        _dbg(f"goto {target}")
        last_err: Optional[Exception] = None
        for attempt in range(step_retries + 1):
            try:
                await _safe_goto(page, target, timeout_ms=nav_timeout)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=4_000)
                except Exception:
                    pass
                await _wait_page_ready(page, timeout_ms=1_500)
                if _HUMAN_PAUSE_MS > 0:
                    await page.wait_for_timeout(_HUMAN_PAUSE_MS)
                if await _is_varnish_503(page):
                    varnish_seen = True
                    wait_ms = step_backoff_ms * (attempt + 1)
                    _dbg(f"varnish_503: retry in {wait_ms}ms (attempt {attempt+1}/{step_retries+1})")
                    if attempt >= step_retries:
                        raise _dom_err(
                            step="goto_match",
                            code="varnish_503",
                            attempt=attempt + 1,
                            message="Sofascore 503 backend read error (Varnish)",
                            diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": True},
                        )
                    await page.wait_for_timeout(wait_ms)
                    continue
                last_err = None
                break
            except DomStatsError:
                raise
            except Exception as ex:
                last_err = ex
                have_id = _match_identity(page.url or "")
                if want_id and have_id == want_id:
                    last_err = None
                    break
                if attempt >= step_retries:
                    raise _dom_err(
                        step="goto_match",
                        code="navigation_failed",
                        attempt=attempt + 1,
                        message=f"{type(ex).__name__}: {ex}",
                        diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                    ) from ex
                await page.wait_for_timeout(step_backoff_ms * (attempt + 1))
                continue
        if last_err is not None:
            raise _dom_err(
                step="goto_match",
                code="navigation_failed",
                message=f"{type(last_err).__name__}: {last_err}",
                diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
            ) from last_err
    else:
        if await _is_varnish_503(page):
            varnish_seen = True
            _dbg("varnish_503 on same match; reloading")
            await _safe_goto(page, target, timeout_ms=nav_timeout)
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=4_000)
            except Exception:
                pass
            await _wait_page_ready(page, timeout_ms=1_500)
            if _HUMAN_PAUSE_MS > 0:
                await page.wait_for_timeout(_HUMAN_PAUSE_MS)
            if await _is_varnish_503(page):
                raise _dom_err(
                    step="goto_match",
                    code="varnish_503",
                    message="Sofascore 503 backend read error (Varnish)",
                    diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": True},
                )

    if await _is_cloudflare_block(page):
        raise _dom_err(
            step="dismiss_overlays",
            code="cloudflare_block",
            message="Cloudflare блокирует страницу (нужно реальное окно/куки)",
            diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
        )

    try:
        await _dismiss_overlays(page)
    except Exception as ex:
        raise _dom_err(
            step="dismiss_overlays",
            code="overlay_blocked",
            message=f"{type(ex).__name__}: {ex}",
            diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
        ) from ex
    wait_stats = int(wait_stats_ms) if isinstance(wait_stats_ms, int) and wait_stats_ms > 0 else _WAIT_STATS_MS
    stats_opened = False
    open_err: Optional[Exception] = None
    for attempt in range(step_retries + 1):
        try:
            await _click_statistics_tab(page, wait_stats_ms=wait_stats)
            ok_rows = await _wait_for_stats_rows(page, timeout_ms=_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats)
            if not ok_rows:
                raise _dom_err(
                    step="open_statistics_tab",
                    code="rows_not_ready",
                    attempt=attempt + 1,
                    message="statistics rows not ready after tab open",
                    diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                )
            stats_opened = True
            open_err = None
            break
        except DomStatsError as ex:
            open_err = ex
            if attempt >= step_retries:
                break
            await page.wait_for_timeout(step_backoff_ms * (attempt + 1))
        except Exception as ex:
            open_err = ex
            if attempt >= step_retries:
                break
            await page.wait_for_timeout(step_backoff_ms * (attempt + 1))
    if not stats_opened:
        msg = f"{type(open_err).__name__}: {open_err}" if open_err is not None else "statistics tab not reachable"
        open_code = "statistics_tab_unreachable"
        if isinstance(open_err, DomStatsError) and isinstance(open_err.code, str) and open_err.code.strip():
            open_code = open_err.code.strip().lower()
        raise _dom_err(
            step="open_statistics_tab",
            code=open_code,
            message=msg,
            diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
        ) from open_err

    try:
        await disable_network_cache(page)
    except Exception:
        pass

    # Expand full stats if Sofascore collapses blocks behind a "Показать больше" button.
    # IMPORTANT: scope this to the statistics tabpanel; otherwise we may click unrelated "Show more" elsewhere on the page.
    try:
        tabpanel = page.locator("#tabpanel-statistics")
        if await tabpanel.count():
            for _ in range(3):
                btn = tabpanel.get_by_text("Показать больше", exact=True)
                if not (await btn.count()):
                    btn = tabpanel.get_by_text("Show more", exact=True)
                if not (await btn.count()):
                    break
                try:
                    await btn.first.click(timeout=2500, force=True)
                except Exception:
                    break
                await _wait_for_stats_rows(page, timeout_ms=min(1_500, _UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats))
    except Exception:
        pass

    async def _scroll_for_full_stats() -> None:
        # Trigger lazy-loading of lower statistic blocks (Return/Games/etc).
        # Run this *after* selecting the desired period; some pages render the period switcher only at the top.
        try:
            for _ in range(3):
                await page.mouse.wheel(0, 1600)
                await _wait_for_stats_rows(page, timeout_ms=max(200, int(_DOM_READY_POLL_MS * 2)))
            # Small scroll back up keeps header sections in view for stable DOM.
            await page.mouse.wheel(0, -2400)
            await _wait_for_stats_rows(page, timeout_ms=max(160, int(_DOM_READY_POLL_MS * 2)))
        except Exception:
            pass

    async def _scroll_to_stats_top() -> None:
        # Ensure period controls (ВСЕ/1-й/2-й/3-й) are in view.
        # If we can see a group header, scroll it into view (places us near the stats block).
        try:
            for lab in ("Подача", "Service"):
                h = page.get_by_text(lab, exact=True)
                if await h.count():
                    await h.first.scroll_into_view_if_needed(timeout=1500)
                    await _wait_for_stats_rows(page, timeout_ms=max(160, int(_DOM_READY_POLL_MS * 2)))
                    break
        except Exception:
            pass
        # Small scroll up helps reveal the segmented control above the first group.
        try:
            for _ in range(2):
                await page.mouse.wheel(0, -1200)
                await _wait_for_stats_rows(page, timeout_ms=max(160, int(_DOM_READY_POLL_MS * 2)))
        except Exception:
            pass

    # Make sure the stats block is in view; some matches lazy-render the period tabs only after scrolling.
    await _scroll_to_stats_top()

    def map_group(title: str) -> Optional[str]:
        return _GROUP_MAP.get((title or "").strip())

    def map_item(group_en: str, label: str) -> Optional[Tuple[str, str]]:
        return _ITEM_MAP.get((group_en, (label or "").strip()))

    async def scrape_current_view() -> List[Dict[str, Any]]:
        _dbg("scrape_current_view evaluate()")
        eval_timeout_s = max(6.0, float((_UI_TIMEOUT_MS + _SLOW_LOAD_MS + wait_stats) / 1000.0))
        if _NO_LIMITS_MODE:
            return await page.evaluate(
                """
            () => {
              const root = document.querySelector('#tabpanel-statistics') || document;

              function normText(s) {
                // Be tolerant to NBSP/zero-width and other layout chars.
                return (s || '')
                  .replace(/[\\u00a0\\u200b\\u200c\\u200d\\ufeff]/g, ' ')
                  .replace(/\\s+/g, ' ')
                  .trim();
              }

              function canonGroup(s) {
                const t = normText(s).toLowerCase();
                if (!t) return null;
                // RU
                if (/^подача\\b/.test(t)) return 'Подача';
                if (/^очки\\b/.test(t)) return 'Очки';
                if (/^(возврат|при[её]м)\\b/.test(t)) return 'Возврат';
                if (/^(игр|игры|геймы|гейм)\\b/.test(t)) return 'Игр';
                if (/^(разное|прочее)\\b/.test(t)) return 'Разное';
                // EN
                if (/^service\\b/.test(t)) return 'Service';
                if (/^points\\b/.test(t)) return 'Points';
                if (/^return\\b/.test(t)) return 'Return';
                if (/^games?\\b/.test(t)) return 'Games';
                if (/^(miscellaneous|misc)\\b/.test(t)) return 'Miscellaneous';
                return null;
              }

              // Assign rows to the closest header above (works for 2-column layout).
              function rectOf(el) {
                try {
                  const r = el.getBoundingClientRect();
                  if (!r) return null;
                  return {
                    x: r.left + window.scrollX,
                    y: r.top + window.scrollY,
                    w: r.width,
                    h: r.height,
                    cx: r.left + window.scrollX + r.width / 2,
                  };
                } catch (e) {
                  return null;
                }
              }

              // Prefer styled group headers (fast) and fall back to broad scan only if needed.
              let headerEls = Array.from(root.querySelectorAll('span[class*=\"textStyle_display\"]'))
                .map((el) => ({ el, t: canonGroup(el.textContent), r: rectOf(el) }))
                .filter((x) => x.t && x.r && typeof x.r.y === 'number');
              if (!headerEls.length) {
                headerEls = Array.from(root.querySelectorAll('h1,h2,h3,h4,span,div,p'))
                  .map((el) => ({ el, t: canonGroup(el.textContent), r: rectOf(el) }))
                  .filter((x) => x.t && x.r && typeof x.r.y === 'number');
              }
              headerEls.sort((a,b) => (a.r.y - b.r.y) || (a.r.cx - b.r.cx));

              const rowSel = [
                'div.d_flex.ai_center.jc_space-between',
                'div.d_flex.jc_space-between',
                'div.jc_space-between',
                'div[class*="statisticsRow"]',
                'div[class*="statRow"]'
              ].join(',');
              const rowsAll = Array.from(root.querySelectorAll(rowSel))
                .map((el) => ({ el, r: rectOf(el) }))
                .filter((x) => x.r && typeof x.r.y === 'number')
                .sort((a,b) => (a.r.y - b.r.y) || (a.r.cx - b.r.cx));

              function parseRow(rowEl) {
                const bdis = Array.from(rowEl.querySelectorAll('bdi'));
                let left = null;
                let right = null;
                if (bdis.length >= 2) {
                  left = bdis[0];
                  right = bdis[bdis.length - 1];
                }
                // Fallback: find first/last numeric-ish text nodes in row.
                if (!left || !right) {
                  const nodes = Array.from(rowEl.querySelectorAll('span,div,p,strong,b'))
                    .map((el) => ({ el, t: normText(el.textContent) }))
                    .filter((x) => x.t);
                  const numeric = nodes.filter((x) => /^\\d+(?:\\s*\\/\\s*\\d+)?(?:\\s*\\(\\d+%\\))?$/.test(x.t));
                  if (numeric.length >= 2) {
                    left = numeric[0].el;
                    right = numeric[numeric.length - 1].el;
                  }
                }
                let label =
                  rowEl.querySelector('span[class*="textStyle_assistive"]') ||
                  rowEl.querySelector('span[class*="textStyle_secondary"]');
                if (!label) {
                  // Fallback: try to find a non-numeric center label.
                  const txts = Array.from(rowEl.querySelectorAll('span,div,p'))
                    .map((el) => ({ el, t: normText(el.textContent) }))
                    .filter((x) => x.t && !/^\\d+(?:\\s*\\/\\s*\\d+)?(?:\\s*\\(\\d+%\\))?$/.test(x.t));
                  if (txts.length) label = txts[0].el;
                }
                if (!left || !right || !label) return null;
                const home = (left.textContent || '').trim();
                const away = (right.textContent || '').trim();
                const name = normText(label.textContent);
                if (!name) return null;
                return { name, home, away };
              }

              function pickHeaderForRow(rowRect) {
                let best = null;
                let bestScore = null;
                for (const h of headerEls) {
                  const dy = rowRect.y - h.r.y;
                  if (dy < -6) continue; // header must be above (allow small overlap)
                  // Score: prefer closest above by y, break ties by x proximity (two columns).
                  const score = (dy * 10.0) + Math.min(600.0, Math.abs(rowRect.cx - h.r.cx));
                  if (bestScore === null || score < bestScore) {
                    bestScore = score;
                    best = h;
                  }
                }
                return best ? best.t : null;
              }

              const byGroup = new Map();
              for (const row of rowsAll) {
                const g = pickHeaderForRow(row.r);
                if (!g) continue;
                const it = parseRow(row.el);
                if (!it) continue;
                if (!byGroup.has(g)) byGroup.set(g, []);
                byGroup.get(g).push(it);
              }

              const groups = [];
              for (const [title, items] of byGroup.entries()) {
                if (items && items.length) groups.push({ title, items });
              }
              return groups;
            }
            """
            )
        return await asyncio.wait_for(
            page.evaluate(
                """
            () => {
              const root = document.querySelector('#tabpanel-statistics') || document;

              function normText(s) {
                // Be tolerant to NBSP/zero-width and other layout chars.
                return (s || '')
                  .replace(/[\\u00a0\\u200b\\u200c\\u200d\\ufeff]/g, ' ')
                  .replace(/\\s+/g, ' ')
                  .trim();
              }

              function canonGroup(s) {
                const t = normText(s).toLowerCase();
                if (!t) return null;
                // RU
                if (/^подача\\b/.test(t)) return 'Подача';
                if (/^очки\\b/.test(t)) return 'Очки';
                if (/^(возврат|при[её]м)\\b/.test(t)) return 'Возврат';
                if (/^(игр|игры|геймы|гейм)\\b/.test(t)) return 'Игр';
                if (/^(разное|прочее)\\b/.test(t)) return 'Разное';
                // EN
                if (/^service\\b/.test(t)) return 'Service';
                if (/^points\\b/.test(t)) return 'Points';
                if (/^return\\b/.test(t)) return 'Return';
                if (/^games?\\b/.test(t)) return 'Games';
                if (/^(miscellaneous|misc)\\b/.test(t)) return 'Miscellaneous';
                return null;
              }

              function rectOf(el) {
                try {
                  const r = el.getBoundingClientRect();
                  if (!r) return null;
                  return {
                    x: r.left + window.scrollX,
                    y: r.top + window.scrollY,
                    w: r.width,
                    h: r.height,
                    cx: r.left + window.scrollX + r.width / 2,
                  };
                } catch (e) {
                  return null;
                }
              }

              let headerEls = Array.from(root.querySelectorAll('span[class*=\"textStyle_display\"]'))
                .map((el) => ({ el, t: canonGroup(el.textContent), r: rectOf(el) }))
                .filter((x) => x.t && x.r && typeof x.r.y === 'number');
              if (!headerEls.length) {
                headerEls = Array.from(root.querySelectorAll('h1,h2,h3,h4,span,div,p'))
                  .map((el) => ({ el, t: canonGroup(el.textContent), r: rectOf(el) }))
                  .filter((x) => x.t && x.r && typeof x.r.y === 'number');
              }
              headerEls.sort((a,b) => (a.r.y - b.r.y) || (a.r.cx - b.r.cx));

              const rowSel = [
                'div.d_flex.ai_center.jc_space-between',
                'div.d_flex.jc_space-between',
                'div.jc_space-between',
                'div[class*=\"statisticsRow\"]',
                'div[class*=\"statRow\"]'
              ].join(',');
              const rowsAll = Array.from(root.querySelectorAll(rowSel))
                .map((el) => ({ el, r: rectOf(el) }))
                .filter((x) => x.r && typeof x.r.y === 'number')
                .sort((a,b) => (a.r.y - b.r.y) || (a.r.cx - b.r.cx));

              function parseRow(rowEl) {
                const bdis = Array.from(rowEl.querySelectorAll('bdi'));
                let left = null;
                let right = null;
                if (bdis.length >= 2) {
                  left = bdis[0];
                  right = bdis[bdis.length - 1];
                }
                if (!left || !right) {
                  const nodes = Array.from(rowEl.querySelectorAll('span,div,p,strong,b'))
                    .map((el) => ({ el, t: normText(el.textContent) }))
                    .filter((x) => x.t);
                  const numeric = nodes.filter((x) => /^\\d+(?:\\s*\\/\\s*\\d+)?(?:\\s*\\(\\d+%\\))?$/.test(x.t));
                  if (numeric.length >= 2) {
                    left = numeric[0].el;
                    right = numeric[numeric.length - 1].el;
                  }
                }
                let label =
                  rowEl.querySelector('span[class*=\"textStyle_assistive\"]') ||
                  rowEl.querySelector('p[class*=\"textStyle_assistive\"]') ||
                  rowEl.querySelector('div[class*=\"textStyle_assistive\"]') ||
                  null;
                if (!label) {
                  const cands = Array.from(rowEl.querySelectorAll('span,div,p'))
                    .map((el) => normText(el.textContent))
                    .filter((t) => t && !/^\\d+(?:\\s*\\/\\s*\\d+)?(?:\\s*\\(\\d+%\\))?$/.test(t));
                  if (cands.length) label = { textContent: cands[Math.floor(cands.length/2)] };
                }
                const labelText = normText(label && (label.textContent || label.innerText || ''));
                if (!labelText) return null;
                const lv = normText(left && (left.textContent || left.innerText || ''));
                const rv = normText(right && (right.textContent || right.innerText || ''));
                if (!lv || !rv) return null;
                return { label: labelText, home: lv, away: rv };
              }

              function nearestGroupForRow(rowRect) {
                if (!rowRect || !headerEls.length) return null;
                let best = null;
                for (const h of headerEls) {
                  if (h.r.y > rowRect.y) break;
                  const dy = rowRect.y - h.r.y;
                  const dx = Math.abs(rowRect.cx - h.r.cx);
                  const score = dy * 1.0 + dx * 0.15;
                  if (!best || score < best.score) best = { t: h.t, score };
                }
                return best ? best.t : null;
              }

              const byGroup = new Map();
              for (const row of rowsAll) {
                const parsed = parseRow(row.el);
                if (!parsed) continue;
                const g = nearestGroupForRow(row.r);
                if (!g) continue;
                const it = { label: parsed.label, home: parsed.home, away: parsed.away };
                if (!byGroup.has(g)) byGroup.set(g, []);
                byGroup.get(g).push(it);
              }

              const groups = [];
              for (const [title, items] of byGroup.entries()) {
                if (items && items.length) groups.push({ title, items });
              }
              return groups;
            }
            """
            ),
            timeout=eval_timeout_s,
        )

    out_periods: List[Dict[str, Any]] = []
    meta_seen: Dict[str, Any] = {}
    meta_unmapped: Dict[str, Any] = {}
    for per in periods:
        groups_raw: List[Dict[str, Any]] = []
        selected = False
        select_err: Optional[Exception] = None
        for attempt in range(step_retries + 1):
            try:
                selected = await _select_period(page, per, wait_stats_ms=wait_stats)
                if not selected:
                    raise _dom_err(
                        step=f"select_period_{per.lower()}",
                        code="period_select_failed",
                        attempt=attempt + 1,
                        message=f"period={per}: not selected",
                        diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                    )
                _dbg(f"period {per}: selected")
                try:
                    await page.wait_for_selector("div.d_flex.ai_center.jc_space-between", timeout=_UI_TIMEOUT_MS + _SLOW_LOAD_MS)
                except Exception:
                    pass
                await _wait_for_stats_rows(page, timeout_ms=max(250, int(_DOM_READY_POLL_MS * 2)))
                groups_raw = await scrape_current_view()
                if not isinstance(groups_raw, list):
                    raise _dom_err(
                        step=f"extract_{per.lower()}",
                        code="scrape_eval_timeout",
                        attempt=attempt + 1,
                        message="scrape returned non-list",
                        diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                    )
                select_err = None
                break
            except asyncio.TimeoutError as ex:
                select_err = _dom_err(
                    step=f"extract_{per.lower()}",
                    code="scrape_eval_timeout",
                    attempt=attempt + 1,
                    message=f"{type(ex).__name__}: {ex}",
                    diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                )
                if attempt >= step_retries:
                    raise select_err from ex
                await page.wait_for_timeout(step_backoff_ms * (attempt + 1))
            except DomStatsError as ex:
                select_err = ex
                ex_code = str(getattr(ex, "code", "") or "")
                if ex_code == "stats_not_provided" or "code=stats_not_provided" in str(ex):
                    raise _dom_err(
                        step=f"select_period_{per.lower()}",
                        code="stats_not_provided",
                        attempt=attempt + 1,
                        message=str(ex),
                        diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                    ) from ex
                if attempt >= step_retries:
                    raise _dom_err(
                        step=f"select_period_{per.lower()}",
                        code="period_select_failed",
                        attempt=attempt + 1,
                        message=str(ex),
                        diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                    ) from ex
                await page.wait_for_timeout(step_backoff_ms * (attempt + 1))
            except Exception as ex:
                select_err = ex
                if attempt >= step_retries:
                    raise _dom_err(
                        step=f"extract_{per.lower()}",
                        code="scrape_eval_timeout",
                        attempt=attempt + 1,
                        message=f"{type(ex).__name__}: {ex}",
                        diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                    ) from ex
                await page.wait_for_timeout(step_backoff_ms * (attempt + 1))
        if select_err is not None and not groups_raw:
            raise _dom_err(
                step=f"extract_{per.lower()}",
                code="scrape_eval_timeout",
                message=str(select_err),
                diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
            ) from select_err
        # If the page is lazy-loading lower blocks, the first scrape may include only the top group.
        # Only scroll if we don't see the key tennis groups (Service/Points/Return).
        try:
            titles = {str(g.get("title") or "").strip() for g in groups_raw if isinstance(g, dict)}
        except Exception:
            titles = set()
        # We intentionally do NOT require "Games"/"Misc" here: Sofascore sometimes renders them far below
        # and scrolling adds a lot of time. Our models rely primarily on Service/Points/Return.
        if len(groups_raw) < 3 or not ({"Подача", "Очки", "Возврат"} & titles):
            await _scroll_for_full_stats()
            try:
                groups_raw = await scrape_current_view()
            except asyncio.TimeoutError as ex:
                raise _dom_err(
                    step=f"extract_{per.lower()}",
                    code="scrape_eval_timeout",
                    message=f"evaluate timeout after scroll: {type(ex).__name__}: {ex}",
                    diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                ) from ex
            except Exception as ex:
                raise _dom_err(
                    step=f"extract_{per.lower()}",
                    code="scrape_eval_timeout",
                    message=f"{type(ex).__name__}: {ex}",
                    diag={"elapsed_ms": _elapsed_ms(), "varnish_503_seen": varnish_seen},
                ) from ex
        # Some pages expose rows but fail to switch period chips reliably (false "no stats").
        # If we only see one top group while rows are present, retry opening statistics tab + period selection once.
        try:
            row_count_now = await page.evaluate(
                """
                () => {
                  const root = document.querySelector('#tabpanel-statistics') || document;
                  return root.querySelectorAll(
                    'div.d_flex.ai_center.jc_space-between, div.d_flex.jc_space-between, div.jc_space-between, div[class*="statisticsRow"], div[class*="statRow"]'
                  ).length;
                }
                """
            )
        except Exception:
            row_count_now = 0
        if isinstance(groups_raw, list) and len(groups_raw) <= 1 and int(row_count_now or 0) >= _ROWCOUNT_STATS_PRESENT_THRESHOLD:
            _dbg(f"period {per}: weak groups_raw={len(groups_raw)} with rowCount={int(row_count_now or 0)}; retry period")
            try:
                await _click_statistics_tab(page, wait_stats_ms=wait_stats)
            except Exception:
                pass
            try:
                await page.wait_for_timeout(max(160, _DOM_READY_POLL_MS))
            except Exception:
                pass
            try:
                _ = await _select_period(page, per, wait_stats_ms=wait_stats)
            except Exception:
                pass
            try:
                groups_raw_retry = await scrape_current_view()
                if isinstance(groups_raw_retry, list) and len(groups_raw_retry) > len(groups_raw):
                    groups_raw = groups_raw_retry
            except Exception:
                pass
        if not isinstance(groups_raw, list):
            continue
        _dbg(f"period {per}: groups_raw={len(groups_raw)}")

        groups_out: Dict[str, Dict[str, Any]] = {}
        # Diagnostics: record raw labels seen and which ones we failed to map.
        seen_labels: Dict[str, List[str]] = {}
        unmapped_labels: Dict[str, List[str]] = {}
        for g in groups_raw:
            if not isinstance(g, dict):
                continue
            title = g.get("title")
            group_en = map_group(str(title)) if title is not None else None
            if not group_en:
                continue
            items = g.get("items") or []
            for it in items:
                if not isinstance(it, dict):
                    continue
                # scrape_current_view returns "label"; keep "name" as legacy fallback.
                label = str(it.get("label") or it.get("name") or "").strip()
                if label:
                    seen_labels.setdefault(group_en, []).append(label)
                mapped = map_item(group_en, label)
                if not mapped:
                    if label:
                        unmapped_labels.setdefault(group_en, []).append(label)
                    continue
                gname, iname = mapped
                home_txt = str(it.get("home") or "").strip()
                away_txt = str(it.get("away") or "").strip()

                item_obj: Dict[str, Any] = {"name": iname, "home": home_txt, "away": away_txt}
                if re.fullmatch(r"\d+", home_txt) and re.fullmatch(r"\d+", away_txt):
                    item_obj["homeValue"] = int(home_txt)
                    item_obj["awayValue"] = int(away_txt)
                else:
                    item_obj["homeValue"] = 0
                    item_obj["awayValue"] = 0

                grp = groups_out.setdefault(gname, {"groupName": gname, "statisticsItems": []})
                grp["statisticsItems"].append(item_obj)

        # Only keep non-empty periods. (Otherwise downstream sees “period present” but nothing inside.)
        if groups_out:
            out_periods.append({"period": per, "groups": list(groups_out.values())})
        # Always keep diagnostics, even if mapping failed.
        if seen_labels:
            # Deduplicate and cap for size.
            meta_seen[per] = {k: sorted(set(v))[:60] for k, v in seen_labels.items()}
        if unmapped_labels:
            meta_unmapped[per] = {k: sorted(set(v))[:60] for k, v in unmapped_labels.items()}

    if not out_periods:
        # Strict mode: no fallbacks. If we can't read 1ST/2ND, fail fast.
        # Provide a small DOM diagnostic for faster tuning (period chips + group headers).
        try:
            diag = await page.evaluate(
                """
                () => {
                  function norm(s){return (s||'').replace(/\\s+/g,' ').trim();}
                  const root = document.querySelector('#tabpanel-statistics') || document;
                  const buttons = Array.from(root.querySelectorAll('button,[role=\"tab\"],[role=\"button\"]'))
                    .map(el=>norm(el.innerText||el.textContent||''))
                    .filter(t=>t)
                    .slice(0,200);
                  const period = buttons.filter(t=>/^(все|all|1\\s*[-\\u2010-\\u2014]\\s*й|2\\s*[-\\u2010-\\u2014]\\s*й|3\\s*[-\\u2010-\\u2014]\\s*й|1st|2nd|3rd)/i.test(t));
                  const groups = Array.from(root.querySelectorAll('h1,h2,h3,h4,span,div,p'))
                    .map(el=>norm(el.textContent||''))
                    .filter(t=>t && /^(Подача|Очки|Возврат|Игр|Разное|Service|Points|Return|Games|Miscellaneous)$/.test(t))
                    .slice(0,20);
                  const rowCount = root.querySelectorAll('div.d_flex.ai_center.jc_space-between, div.d_flex.jc_space-between, div.jc_space-between').length;
                  const consent = !!document.querySelector('.fc-consent-root, #fc-consent-root, #onetrust-banner-sdk');
                  return {period, groups, rowCount, consent};
                }
                """
            )
        except Exception:
            diag = None
        code = "rows_not_ready"
        row_count_diag = 0
        groups_detected = 0
        period_tokens_detected: List[str] = []
        try:
            period_vals = list(diag.get("period") or []) if isinstance(diag, dict) else []
            row_count_diag = int(diag.get("rowCount") or 0) if isinstance(diag, dict) else 0
            groups_detected = len(list(diag.get("groups") or [])) if isinstance(diag, dict) else 0
            if isinstance(diag, dict):
                period_tokens_detected = [str(x) for x in list(diag.get("period") or []) if str(x).strip()]
            joined = " ".join(str(x or "") for x in period_vals).lower()
            has_p1 = bool(re.search(r"(1\s*[-\u2010-\u2014]?\s*й|1st|set\s*1|1\s*set|\b1\b)", joined))
            has_p2 = bool(re.search(r"(2\s*[-\u2010-\u2014]?\s*й|2nd|set\s*2|2\s*set|\b2\b)", joined))
            if (row_count_diag < _ROWCOUNT_STATS_PRESENT_THRESHOLD and groups_detected <= 0) and (not (has_p1 and has_p2)):
                code = "stats_not_provided"
        except Exception:
            code = "rows_not_ready"
        msg = "No per-set statistics found in DOM (missing 1ST/2ND)"
        if isinstance(diag, dict):
            msg += f" | diag={diag}"
        raise _dom_err(
            step="wait_rows_2nd",
            code=code,
            message=msg,
            diag={
                "elapsed_ms": _elapsed_ms(),
                "varnish_503_seen": varnish_seen,
                "diag": diag,
                "rowCount": int(row_count_diag),
                "groups_detected": int(groups_detected),
                "period_tokens_detected": period_tokens_detected[:12],
            },
        )
    diag_meta = {
        "step": f"extract_{str(periods[-1]).lower() if periods else 'unknown'}",
        "code": "ok",
        "attempt": 1,
        "elapsed_ms": _elapsed_ms(),
        "varnish_503_seen": varnish_seen,
    }
    return {
        "statistics": out_periods,
        "_meta": {"seen": meta_seen, "unmapped": meta_unmapped, "diag": diag_meta},
    }
