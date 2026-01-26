from __future__ import annotations

import re
import os
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import Page

from third_set.browser_utils import clear_browser_cache, disable_network_cache, page_is_usable


class DomStatsError(RuntimeError):
    pass


_NAV_TIMEOUT_MS = 25_000
_UI_TIMEOUT_MS = 6_000


def _dbg(msg: str) -> None:
    if os.getenv("THIRDSET_DEBUG") in ("1", "true", "yes"):
        print(f"[dom] {msg}", flush=True)


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
    ("Service", "First serve"): ("Service", "First serve"),
    ("Service", "Поинты первой подачи"): ("Service", "First serve points"),
    ("Service", "First serve points"): ("Service", "First serve points"),
    ("Service", "Поинты второй подачи"): ("Service", "Second serve points"),
    ("Service", "Second serve points"): ("Service", "Second serve points"),
    ("Service", "Брейк-поинты защищены"): ("Service", "Break points saved"),
    ("Service", "Break points saved"): ("Service", "Break points saved"),
    # Return
    ("Return", "Поинты первой подачи"): ("Return", "First serve return points"),
    ("Return", "First serve return points"): ("Return", "First serve return points"),
    ("Return", "Поинты второй подачи"): ("Return", "Second serve return points"),
    ("Return", "Second serve return points"): ("Return", "Second serve return points"),
    ("Return", "Сыграно ответных"): ("Return", "Return games played"),
    ("Return", "Сыграно ответных матчей"): ("Return", "Return games played"),
    ("Return", "Return games played"): ("Return", "Return games played"),
    ("Return", "Брейк-поинты использованы"): ("Return", "Break points converted"),
    ("Return", "Break points converted"): ("Return", "Break points converted"),
    # Points
    ("Points", "Очки на подаче"): ("Points", "Service points won"),
    ("Points", "Service points won"): ("Points", "Service points won"),
    ("Points", "Очки на приёме"): ("Points", "Receiver points won"),
    ("Points", "Очки, выигранные на подаче соперника"): ("Points", "Receiver points won"),
    ("Points", "Receiver points won"): ("Points", "Receiver points won"),
    ("Points", "Очки, выигранные на своей подаче"): ("Points", "Service points won"),
    ("Points", "Очки, выигранные на приёме"): ("Points", "Receiver points won"),
    # Games
    ("Games", "Выигранные подачи"): ("Games", "Service games won"),
    ("Games", "Service games won"): ("Games", "Service games won"),
    ("Games", "Выигранные подачи соперника"): ("Games", "Return games won"),
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


async def _dismiss_overlays(page: Page) -> None:
    _dbg("dismiss_overlays")
    try:
        await page.keyboard.press("Escape")
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


async def _click_statistics_tab(page: Page) -> None:
    await _dismiss_overlays(page)
    if await _is_cloudflare_block(page):
        raise DomStatsError("Cloudflare блокирует страницу (нужно реальное окно/куки)")
    _dbg("click_statistics_tab")
    # Sofascore renders match sub-tabs asynchronously; wait a bit for the UI to hydrate.
    try:
        # IMPORTANT: match exact tab label, not any substring like "...геймы, сеты, статистика".
        await page.wait_for_selector('text="Статистика"', timeout=12_000)
    except Exception:
        try:
            await page.wait_for_selector('text="Statistics"', timeout=12_000)
        except Exception:
            pass
    await _dismiss_overlays(page)

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
                        await page.wait_for_timeout(500)
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
                    await page.wait_for_timeout(500)
                    return
                except Exception:
                    continue
            # CSS fallbacks (Sofascore sometimes doesn't expose roles)
            try:
                loc = root.locator("a,button,[role='tab'],[role='button'],[role='link']").filter(has_text=name_rx)
                if await loc.count():
                    await loc.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                    await page.wait_for_timeout(500)
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
            await page.wait_for_timeout(500)
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
            await page.wait_for_timeout(700)
            if await page.locator("#tabpanel-statistics").count():
                return
            # Even if tabpanel id is missing, allow the extractor to continue (it falls back to document root).
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
        await page.wait_for_timeout(800)
        await _dismiss_overlays(page)
        tabpanel = page.locator("#tabpanel-statistics")
        # Do not require "visible": consent overlays can affect visibility checks.
        if await tabpanel.count():
            return
    except Exception:
        pass
    raise DomStatsError("Не удалось открыть вкладку 'Статистика' (DOM)")


async def _select_period(page: Page, period_code: str) -> bool:
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
            ]
        if code == "2ND":
            return [
                re.compile(rf"2\s*{dash}\s*й{set_word}", re.I),
                re.compile(r"2nd(?:\s*set)?", re.I),
            ]
        if code == "3RD":
            return [
                re.compile(rf"3\s*{dash}\s*й{set_word}", re.I),
                re.compile(r"3rd(?:\s*set)?", re.I),
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

    async def try_click_within(root) -> bool:
        # 1) role=tab / role=button with regex name
        for kind in ("tab", "button"):
            for rx in patterns:
                try:
                    loc = root.get_by_role(kind, name=rx)
                    if await loc.count():
                        await loc.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                        await page.wait_for_timeout(250)
                        return True
                except Exception:
                    continue
        # 2) plain buttons/tabs filtered by text
        for rx in patterns:
            try:
                loc = root.locator("button, a, [role='tab'], [role='button'], [role='link']").filter(has_text=rx)
                if await loc.count():
                    await loc.first.click(timeout=_UI_TIMEOUT_MS, force=True)
                    await page.wait_for_timeout(250)
                    return True
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
                await page.wait_for_timeout(250)
                return True
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
                await page.wait_for_timeout(250)
                return True
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
                # Let the period control render (it may animate / portal-render).
                try:
                    await page.get_by_role("tab", name=re.compile(r"^1", re.I)).first.wait_for(timeout=1200)
                except Exception:
                    await page.wait_for_timeout(1200)
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
                                await page.wait_for_timeout(250)
                                return True
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
    page: Page, *, match_url: str, event_id: int, periods: Tuple[str, ...] = ("1ST", "2ND")
) -> Dict[str, Any]:
    """
    Extract per-period tennis statistics from Sofascore DOM ("Статистика" tab).

    Returns a pseudo-API JSON compatible with MatchSnapshot/stats_parser:
      { "statistics": [ { "period": "1ST", "groups": [ { "groupName": "...", "statisticsItems": [...] } ] } ] }
    """
    # Ensure we are on the correct match URL (history reuses a single tab).
    target = match_url + f"#id:{event_id}"
    try:
        cur_base = (page.url or "").split("#", 1)[0]
    except Exception:
        cur_base = ""
    if not cur_base.startswith(match_url) or f"id:{event_id}" not in (page.url or ""):
        _dbg(f"goto {target}")
        await page.goto(target, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        try:
            await page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass

    if await _is_cloudflare_block(page):
        raise DomStatsError("Cloudflare блокирует страницу (нужно реальное окно/куки)")

    await _dismiss_overlays(page)
    try:
        await _click_statistics_tab(page)
    except DomStatsError as e:
        # Some layouts render the statistics block without an explicit "Статистика" tab
        # (or the tab is hidden behind responsive UI). If the stats rows exist in DOM,
        # scraping can still succeed, so don't hard-fail here.
        _dbg(f"click_statistics_tab skipped: {e}")
    await page.wait_for_timeout(600)

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
                await page.wait_for_timeout(600)
    except Exception:
        pass

    async def _scroll_for_full_stats() -> None:
        # Trigger lazy-loading of lower statistic blocks (Return/Games/etc).
        # Run this *after* selecting the desired period; some pages render the period switcher only at the top.
        try:
            for _ in range(6):
                await page.mouse.wheel(0, 1600)
                await page.wait_for_timeout(350)
            # Small scroll back up keeps header sections in view for stable DOM.
            await page.mouse.wheel(0, -2400)
            await page.wait_for_timeout(250)
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
                    await page.wait_for_timeout(200)
                    break
        except Exception:
            pass
        # Small scroll up helps reveal the segmented control above the first group.
        try:
            for _ in range(2):
                await page.mouse.wheel(0, -1200)
                await page.wait_for_timeout(180)
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

              const rowSel = 'div.d_flex.ai_center.jc_space-between, div.d_flex.jc_space-between, div.jc_space-between';
              const rowsAll = Array.from(root.querySelectorAll(rowSel))
                .map((el) => ({ el, r: rectOf(el) }))
                .filter((x) => x.r && typeof x.r.y === 'number')
                .sort((a,b) => (a.r.y - b.r.y) || (a.r.cx - b.r.cx));

              function parseRow(rowEl) {
                const bdis = Array.from(rowEl.querySelectorAll('bdi'));
                if (bdis.length < 2) return null;
                const left = bdis[0];
                const right = bdis[bdis.length - 1];
                let label =
                  rowEl.querySelector('span[class*="textStyle_assistive"]') ||
                  rowEl.querySelector('span[class*="textStyle_secondary"]');
                if (!label) {
                  // Fallback: try to find a non-numeric center label.
                  const txts = Array.from(rowEl.querySelectorAll('span,div,p'))
                    .map((el) => ({ el, t: normText(el.textContent) }))
                    .filter((x) => x.t && !/^\\d+(?:\\/\\d+)?(?:\\s*\\(\\d+%\\))?$/.test(x.t));
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

    out_periods: List[Dict[str, Any]] = []
    meta_seen: Dict[str, Any] = {}
    meta_unmapped: Dict[str, Any] = {}
    for per in periods:
        ok = await _select_period(page, per)
        if not ok:
            continue
        _dbg(f"period {per}: selected")
        try:
            await page.wait_for_selector("div.d_flex.ai_center.jc_space-between", timeout=_UI_TIMEOUT_MS)
        except Exception:
            pass
        await page.wait_for_timeout(350)
        groups_raw = await scrape_current_view()
        # If the page is lazy-loading lower blocks, the first scrape may include only the top group.
        # Only scroll if we don't see the expected set of groups.
        try:
            titles = {str(g.get("title") or "").strip() for g in groups_raw if isinstance(g, dict)}
        except Exception:
            titles = set()
        if len(groups_raw) < 4 or not ({"Подача", "Очки", "Возврат"} & titles) or not ({"Игр", "Разное"} & titles):
            await _scroll_for_full_stats()
            groups_raw = await scrape_current_view()
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
                label = str(it.get("name") or "").strip()
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
        msg = "No per-set statistics found in DOM (missing 1ST/2ND)"
        if isinstance(diag, dict):
            msg += f" | diag={diag}"
        raise DomStatsError(msg)
    return {
        "statistics": out_periods,
        "_meta": {"seen": meta_seen, "unmapped": meta_unmapped},
    }
