from __future__ import annotations

import json
import re
import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from playwright.async_api import Page, Response

from third_set.dominance import DominanceLivePoints

SOFASCORE_TENNIS_URL = "https://www.sofascore.com/ru/tennis"
SOFASCORE_API_BASE = "https://www.sofascore.com/api/v1"
SOFASCORE_TENNIS_LIVE_API = f"{SOFASCORE_API_BASE}/sport/tennis/events/live"


class SofascoreError(RuntimeError):
    pass


async def _dismiss_overlays_basic(page: Page) -> None:
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass

    # 1) Age verification modal (must select radio, then confirm; confirm is disabled until selection)
    try:
        age_modal = page.locator("text=Age Verification")
        if await age_modal.count():
            # Prefer the explicit 25+ input id used by Sofascore.
            radio = page.locator("input#radiobutton-25_or_older-betting-content-confirm-25_or_older")
            if not (await radio.count()):
                radio = page.locator("label:has-text('I\\'m 25 years or older') input[type=radio]")
            if await radio.count():
                try:
                    await radio.first.check(timeout=1500, force=True)
                except Exception:
                    try:
                        await radio.first.click(timeout=1500, force=True)
                    except Exception:
                        pass
                await page.wait_for_timeout(200)
            # Confirm button (RU/EN). Use locator, not role, because button can be disabled initially.
            confirm = page.locator("button:has-text('Подтвердить'), button:has-text('Confirm')")
            if await confirm.count():
                try:
                    await confirm.first.click(timeout=2000, force=True)
                except Exception:
                    pass
                await page.wait_for_timeout(400)
    except Exception:
        pass

    # 2) Consent modal(s): accept/agree/confirm.
    for rx in (
        r"accept all|принять все|разрешить все|allow all|agree|ok|okay",
        r"соглашаюсь|i agree",
        r"confirm choices|подтвердить|сохранить|save",
    ):
        try:
            btn = page.locator("button").filter(has_text=re.compile(rx, re.I))
            if await btn.count() and await btn.first.is_visible():
                await btn.first.click(timeout=2000, force=True)
                await page.wait_for_timeout(350)
        except Exception:
            pass


async def warm_player_page(page: Page, *, team_id: int, team_slug: str) -> None:
    """
    Open player's page to establish Sofascore cookies/consent in this tab.
    This significantly reduces 403/Forbidden for subsequent /api/v1/team/<id>/... calls,
    especially on servers.
    """
    slug = (team_slug or "").strip()
    if not slug:
        return
    url = f"https://www.sofascore.com/ru/tennis/player/{slug}/{int(team_id)}"
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
    except Exception:
        return
    try:
        await page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass
    try:
        await _dismiss_overlays_basic(page)
    except Exception:
        pass


async def discover_player_profile_urls_from_match(page: Page, *, match_url: str, event_id: int) -> Dict[str, str]:
    """
    Get player profile URLs (DOM) from a match page.
    This is required because event.homeTeam.id is not always the same as the player-profile id,
    and direct /tennis/player/<slug>/<id> can 404.
    Returns: {"home": url, "away": url} when available.
    """
    target = match_url.split("#", 1)[0] + f"#id:{int(event_id)}"
    await page.goto(target, wait_until="domcontentloaded", timeout=25_000)
    try:
        await page.wait_for_load_state("networkidle", timeout=12_000)
    except Exception:
        pass
    await _dismiss_overlays_basic(page)

    try:
        data = await page.evaluate(
            """
            () => {
              function norm(s){return (s||'').replace(/[\\u00a0\\u200b\\u200c\\u200d\\ufeff]/g,' ').replace(/\\s+/g,' ').trim();}
              function isVisible(el){
                if (!el) return false;
                const st = window.getComputedStyle(el);
                if (!st || st.display==='none' || st.visibility==='hidden' || st.opacity==='0') return false;
                const r = el.getClientRects();
                return !!(r && r.length && r[0].width>1 && r[0].height>1);
              }
              const anchors = Array.from(document.querySelectorAll('a[href*=\"/tennis/player/\"]')).filter(isVisible);
              const uniq = [];
              const seen = new Set();
              for (const a of anchors) {
                const href = a.getAttribute('href') || '';
                if (!href.includes('/tennis/player/')) continue;
                const full = (a.href || (location.origin + href)).split('#')[0];
                if (seen.has(full)) continue;
                seen.add(full);
                uniq.push({href: full, text: norm(a.innerText||a.textContent||'')});
                if (uniq.length >= 6) break;
              }
              // Heuristic: the first two profile links on the match header are the two players.
              return uniq.slice(0,2).map(x => x.href);
            }
            """
        )
    except Exception:
        data = []
    out: Dict[str, str] = {}
    if isinstance(data, list):
        if len(data) >= 1 and isinstance(data[0], str) and data[0]:
            out["home"] = data[0]
        if len(data) >= 2 and isinstance(data[1], str) and data[1]:
            out["away"] = data[1]
    return out


async def discover_player_match_links(page: Page, *, team_id: int, team_slug: str, limit: int = 80) -> List[str]:
    """
    Discover match links from the player's page DOM (no /team/<id>/events/last dependency).

    Returns absolute URLs like https://www.sofascore.com/<locale>/tennis/match/<slug>/<customId>
    (without #id fragment). Order is "as displayed" (usually newest first).
    """
    slug = (team_slug or "").strip()
    if not slug:
        return []
    url = f"https://www.sofascore.com/ru/tennis/player/{slug}/{int(team_id)}"
    await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
    try:
        await page.wait_for_load_state("networkidle", timeout=12_000)
    except Exception:
        pass
    await _dismiss_overlays_basic(page)

    # The matches list is often below the fold; scroll to trigger lazy loading.
    try:
        for _ in range(8):
            await page.mouse.wheel(0, 1400)
            await page.wait_for_timeout(250)
    except Exception:
        pass

    hrefs: List[str] = await page.evaluate(
        """
        () => Array.from(document.querySelectorAll('a[href*=\"/tennis/match/\"]'))
          .map(a => a.getAttribute('href'))
          .filter(Boolean)
        """
    )
    out: List[str] = []
    seen: set = set()
    for href in hrefs:
        # Normalize to absolute.
        if href.startswith("http"):
            full = href
        else:
            full = "https://www.sofascore.com" + href
        # Keep only match links.
        if "/tennis/match/" not in full:
            continue
        if full in seen:
            continue
        seen.add(full)
        # Keep fragment "#id:<eventId>" if present (helps stable history resolution).
        out.append(full)
        if len(out) >= int(limit):
            break
    return out


async def discover_player_match_links_from_profile_url(page: Page, *, profile_url: str, limit: int = 80) -> List[str]:
    """
    Discover match links from a player's profile URL (DOM).
    The profile URL is taken from the match page DOM and is the most reliable way to avoid 404s.
    """
    url = (profile_url or "").strip()
    if not url:
        return []
    await page.goto(url, wait_until="domcontentloaded", timeout=25_000)
    try:
        await page.wait_for_load_state("networkidle", timeout=12_000)
    except Exception:
        pass
    await _dismiss_overlays_basic(page)

    # Try to open the "Matches/Матчи" tab if present.
    for rx in (re.compile(r"^\\s*Матчи\\s*$", re.I), re.compile(r"^\\s*Matches\\s*$", re.I)):
        try:
            loc = page.get_by_role("tab", name=rx)
            if await loc.count():
                await loc.first.click(timeout=2500, force=True)
                await page.wait_for_timeout(600)
                break
        except Exception:
            pass
        try:
            loc = page.locator("a,button,[role='tab'],[role='button']").filter(has_text=rx)
            if await loc.count():
                await loc.first.click(timeout=2500, force=True)
                await page.wait_for_timeout(600)
                break
        except Exception:
            pass

    # Prefer singles filter if present.
    for rx in (re.compile(r"Одиноч", re.I), re.compile(r"Singles", re.I)):
        try:
            btn = page.locator("button").filter(has_text=rx)
            if await btn.count():
                await btn.first.click(timeout=2000, force=True)
                await page.wait_for_timeout(450)
                break
        except Exception:
            pass

    # Prefer finished results when available (player pages often default to "Сегодня/Upcoming").
    for rx in (re.compile(r"Законч", re.I), re.compile(r"Finished", re.I), re.compile(r"Результаты", re.I)):
        try:
            btn = page.locator("button").filter(has_text=rx)
            if await btn.count():
                # pick a visible one
                for i in range(min(10, await btn.count())):
                    b = btn.nth(i)
                    try:
                        if await b.is_visible():
                            await b.click(timeout=2000, force=True)
                            await page.wait_for_timeout(450)
                            raise StopIteration
                    except StopIteration:
                        raise
                    except Exception:
                        continue
        except StopIteration:
            break
        except Exception:
            continue

    # Scroll to load matches list.
    try:
        for _ in range(14):
            await page.mouse.wheel(0, 1600)
            await page.wait_for_timeout(250)
    except Exception:
        pass

    # Pull match links with a simple "likely finished" heuristic, so we don't waste time
    # resolving upcoming/live matches first.
    hrefs: List[str] = await page.evaluate(
        """
        () => {
          function norm(s){return (s||'').replace(/[\\u00a0\\u200b\\u200c\\u200d\\ufeff]/g,' ').replace(/\\s+/g,' ').trim();}
          const rxScore = /\\b\\d+\\s*[:\\-]\\s*\\d+\\b/;          // sets score 2:0 / 1-2
          const rxSet = /\\b\\d+\\s+\\d+\\b/;                      // current game points like "30 15" or "6 3"
          const rxFinished = /(Закончил|Finished|FT|ПВ)/i;
          const as = Array.from(document.querySelectorAll('a[href*=\"/tennis/match/\"]'));
          const out = [];
          for (const a of as) {
            const href = a.getAttribute('href') || '';\n            if (!href) continue;\n            const txt = norm(a.innerText || a.textContent || '');\n            const likely = rxFinished.test(txt) || rxScore.test(txt) || rxSet.test(txt);\n            out.push({href, likely});\n          }\n          // Stable-ish: likely finished first, then as-is.\n          out.sort((x,y) => (y.likely?1:0) - (x.likely?1:0));\n          return out.map(x => x.href);\n        }\n        """
    )
    out: List[str] = []
    seen: set = set()
    for href in hrefs:
        if href.startswith("http"):
            full = href
        else:
            full = "https://www.sofascore.com" + href
        if "/tennis/match/" not in full:
            continue
        # Keep fragment "#id:<eventId>" if present.
        if full in seen:
            continue
        seen.add(full)
        out.append(full)
        if len(out) >= int(limit):
            break
    return out


async def fetch_json_via_page(page: Page, url: str) -> Dict[str, Any]:
    result = await page.evaluate(
        """
        async (url) => {
          const ac = new AbortController();
          const t = setTimeout(() => ac.abort(), 15000);
          try {
            const r = await fetch(url, { credentials: "include", signal: ac.signal });
            const text = await r.text();
            return { status: r.status, text };
          } catch (e) {
            return { status: 0, text: String(e && e.message ? e.message : e) };
          } finally {
            clearTimeout(t);
          }
        }
        """,
        url,
    )
    status = int(result["status"])
    text = result["text"]
    if status != 200:
        # Some environments intermittently fail `fetch()` inside the page (status=0: Failed to fetch),
        # while the same request succeeds via Playwright's request context (same cookies/profile).
        try:
            resp = await page.context.request.get(url, timeout=15000)
            if resp.status == 200:
                return await resp.json()
            # Fall through to the original error with some context.
            try:
                body = await resp.text()
            except Exception:
                body = ""
            raise SofascoreError(f"HTTP {resp.status} for {url}: {body[:200]}")
        except SofascoreError:
            raise
        except Exception:
            raise SofascoreError(f"HTTP {status} for {url}: {text[:200]}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SofascoreError(f"Invalid JSON for {url}: {text[:200]}") from e


async def get_live_events(page: Page) -> List[Dict[str, Any]]:
    """
    Returns raw live tennis events from Sofascore API (via browser context).
    """
    # NOTE: Direct fetch() to the API is frequently blocked by Cloudflare (403 challenge),
    # especially on servers. Prefer navigation-captured API via `get_live_events_via_navigation`.
    # Keep this function for local debugging, but callers should not rely on it.
    await page.goto(SOFASCORE_TENNIS_URL, wait_until="domcontentloaded", timeout=25000)
    try:
        await page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    await asyncio.sleep(1.0)
    data = await fetch_json_via_page(page, SOFASCORE_TENNIS_LIVE_API)
    return data.get("events", []) or []


async def get_live_events_via_navigation(page: Page, *, timeout_ms: int = 15000) -> List[Dict[str, Any]]:
    """
    Open tennis page and capture the live events JSON from network responses.
    This avoids direct fetch() calls that can be blocked by CF.
    """
    captured = await _collect_json_via_navigation(
        page,
        url_to_open=SOFASCORE_TENNIS_URL,
        predicates={"live": lambda r: "/api/v1/sport/tennis/events/live" in (r.url or "")},
        required_keys={"live"},
        timeout_ms=timeout_ms,
    )
    data = captured.get("live") or {}
    return data.get("events", []) or []


async def get_live_match_links(page: Page, *, limit: Optional[int] = None) -> List[str]:
    """
    Returns match URLs for live tennis events only.

    Each URL includes a synthetic fragment `#id:<eventId>` so callers can recover the event id.
    """
    # Single source of truth: tennis page DOM, tab "Сейчас (N)".
    # We do NOT filter by API status here: the user wants "exactly what the UI shows".
    # Filtering (BO3/singles/inprogress) is done later by the caller when needed.
    hrefs: List[str] = []
    try:
        hrefs = await discover_match_links(page, limit=limit)
    except Exception:
        hrefs = []
    out: List[str] = []
    seen: set = set()
    for href in hrefs:
        if limit is not None and len(out) >= limit:
            break
        if not isinstance(href, str) or not href:
            continue
        if href in seen:
            continue
        seen.add(href)
        out.append(href)
    return out


async def _collect_json_via_navigation(
    page: Page,
    *,
    url_to_open: str,
    predicates: Dict[str, Callable[[Response], bool]],
    required_keys: Optional[set] = None,
    timeout_ms: int = 15000,
) -> Dict[str, Dict[str, Any]]:
    loop = asyncio.get_running_loop()
    futures: Dict[str, asyncio.Future] = {k: loop.create_future() for k in predicates.keys()}
    required = required_keys if required_keys is not None else set(predicates.keys())

    async def _resp_json_fallback(resp: Response) -> Dict[str, Any]:
        """
        Safely parse response JSON without touching the page execution context.
        This avoids 'Execution context was destroyed' during navigations.
        """
        try:
            return await resp.json()
        except Exception:
            # Some responses can be "unavailable" via resp.json() (cache edge cases).
            # Use the context request API (same cookies/profile) as a fallback.
            req = await page.context.request.get(resp.url, timeout=15000)
            text = await req.text()
            if req.status != 200:
                raise SofascoreError(f"HTTP {req.status} for {resp.url}: {text[:200]}")
            try:
                return json.loads(text)
            except Exception as e:
                raise SofascoreError(f"Invalid JSON for {resp.url}: {text[:200]}") from e

    async def _handle_response(resp: Response) -> None:
        try:
            for key, pred in predicates.items():
                fut = futures[key]
                if fut.done():
                    continue
                try:
                    if not pred(resp):
                        continue
                    if resp.status != 200:
                        if not fut.done():
                            fut.set_exception(SofascoreError(f"HTTP {resp.status} for {resp.url}"))
                        continue
                    data = await _resp_json_fallback(resp)
                    if not fut.done():
                        fut.set_result(data)
                except Exception as e:
                    if not fut.done():
                        fut.set_exception(e)
        except Exception:
            # Never let response-handler tasks raise.
            return

    def _on_response(resp: Response) -> None:
        t = asyncio.create_task(_handle_response(resp))
        # Retrieve exception to avoid "Future exception was never retrieved".
        t.add_done_callback(lambda tt: tt.exception() if not tt.cancelled() else None)

    page.on("response", _on_response)
    try:
        await page.goto(url_to_open, wait_until="domcontentloaded", timeout=25000)
        done: set = set()
        remaining = set(futures.keys())
        # Wait until all required keys are satisfied (or timeout).
        end = loop.time() + timeout_ms / 1000
        while True:
            for k in list(remaining):
                if futures[k].done():
                    remaining.remove(k)
                    done.add(k)
                    # Fail fast if a required future completed with an error.
                    if k in required and futures[k].exception() is not None:
                        raise futures[k].exception()
            if required.issubset(done):
                break
            if loop.time() >= end:
                missing = ", ".join(sorted(required - done))
                raise SofascoreError(f"Timed out collecting: {missing}")
            await asyncio.sleep(0.05)

        result: Dict[str, Dict[str, Any]] = {}
        for k, fut in futures.items():
            if fut.done() and not fut.cancelled():
                err = fut.exception()
                if err is not None:
                    if k in required:
                        raise err
                    continue
                result[k] = fut.result()
        return result
    finally:
        page.remove_listener("response", _on_response)
        # Prevent "Future exception was never retrieved" warnings for non-required keys.
        for fut in futures.values():
            try:
                if fut.done() and not fut.cancelled():
                    _ = fut.exception()
            except Exception:
                pass


@dataclass(frozen=True)
class LiveEvent:
    id: int
    slug: str
    custom_id: str
    home_team_id: int
    home_team_name: str
    away_team_id: int
    away_team_name: str

    @property
    def match_url(self) -> str:
        return f"https://www.sofascore.com/ru/tennis/match/{self.slug}/{self.custom_id}"


async def discover_match_links(page: Page, *, limit: Optional[int] = None) -> List[str]:
    await page.goto(SOFASCORE_TENNIS_URL, wait_until="domcontentloaded", timeout=25000)
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    await _dismiss_overlays_basic(page)
    # Ensure the top filter is "Все" (All). Otherwise Sofascore can show only favorites/competitions,
    # and the LIVE count "(N)" will not match what the user expects from the "Все" screen.
    try:
        all_rx = re.compile(r"^\\s*(Все|All)\\s*$", re.I)
        all_btns = page.locator("button").filter(has_text=all_rx)
        best_all = None
        for i in range(min(await all_btns.count(), 20)):
            cand = all_btns.nth(i)
            try:
                if not await cand.is_visible():
                    continue
            except Exception:
                continue
            try:
                parent_txt = await cand.evaluate(
                    """(el) => (el && el.parentElement && (el.parentElement.innerText || el.parentElement.textContent) || '')"""
                )
            except Exception:
                parent_txt = ""
            p = (parent_txt or "").lower()
            has_fav = ("избран" in p) or ("favorite" in p) or ("favourites" in p)
            has_comp = ("соревн" in p) or ("competition" in p)
            if has_fav and has_comp:
                best_all = cand
                break
            if best_all is None:
                best_all = cand
        if best_all is not None:
            try:
                await best_all.click(timeout=2000, force=True)
                await page.wait_for_timeout(350)
            except Exception:
                pass
    except Exception:
        pass
    # Ensure LIVE tab is selected (RU/EN). The tab includes a count like "Сейчас (28)" / "Live (28)".
    # IMPORTANT: We must match Sofascore UI exactly — the authoritative source is the selected tab:
    #   button[aria-selected="true"] == "Сейчас (N)" (RU) or "Live (N)" (EN).
    expected_n: Optional[int] = None
    chosen_label: Optional[str] = None
    try:
        tab_rx = re.compile(r"^(Сейчас|Live)\s*\(\d+\)\s*$", re.I)
        # Prefer the already-selected LIVE tab (avoids picking unrelated "Live" buttons elsewhere).
        selected = page.locator("button[aria-selected='true']").filter(has_text=tab_rx)
        tab_to_click = None
        if await selected.count():
            tab_to_click = selected.first
        else:
            # Click the LIVE tab in the 3-pill row (Сейчас / Закончился / Предстоящие).
            # There can be other "(N)" buttons on the page; filter by parent text containing the other pills.
            tabs = page.locator("button").filter(has_text=tab_rx)
            best = None
            for i in range(min(await tabs.count(), 30)):
                cand = tabs.nth(i)
                try:
                    if not await cand.is_visible():
                        continue
                except Exception:
                    continue
                try:
                    parent_txt = await cand.evaluate(
                        """(el) => (el && el.parentElement && (el.parentElement.innerText || el.parentElement.textContent) || '')"""
                    )
                except Exception:
                    parent_txt = ""
                p = (parent_txt or "").lower()
                has_finished = ("законч" in p) or ("finished" in p)
                has_upcoming = ("предстоящ" in p) or ("upcoming" in p)
                if has_finished and has_upcoming:
                    best = cand
                    break
                if best is None:
                    best = cand
            tab_to_click = best
        if tab_to_click is not None:
            try:
                await tab_to_click.click(timeout=2500, force=True)
            except Exception:
                pass
            await page.wait_for_timeout(600)
            # After click, the correct LIVE pill should be aria-selected=true.
            try:
                selected2 = page.locator("button[aria-selected='true']").filter(has_text=tab_rx)
                if await selected2.count():
                    tab_to_click = selected2.first
            except Exception:
                pass
            try:
                label = (await tab_to_click.inner_text()).strip()
            except Exception:
                label = ""
            chosen_label = label or None
            m = re.search(r"\((\d+)\)", label)
            if m:
                expected_n = int(m.group(1))
                if expected_n <= 0:
                    expected_n = None
    except Exception:
        expected_n = None
    # Wait for event links to appear.
    try:
        await page.wait_for_selector('a[href*="/tennis/match/"]', timeout=7000)
    except Exception:
        pass
    # Scroll to load more cards until we reach expected count (or stop growing).
    hrefs: List[str] = []
    try:
        prev = 0
        stalled = 0
        # Prefer anchors that contain "#id:" (event cards). This avoids grabbing links from
        # other page areas (rankings, past results, etc.) that also contain "/tennis/match/".
        sel = 'a[href*="/tennis/match/"][href*="#id:"]'
        target = limit
        if target is None and expected_n is not None:
            target = int(expected_n)

        # Sofascore list is often lazy/virtualized; give it time to load as we scroll.
        for _ in range(140):
            hrefs = await page.evaluate(
                f"""
                () => {{
                  function isVisible(el) {{
                    if (!el) return false;
                    const st = window.getComputedStyle(el);
                    if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                    const r = el.getClientRects();
                    if (!r || r.length === 0) return false;
                    const box = r[0];
                    return (box.width > 1 && box.height > 1);
                  }}
                  function findRoot() {{
                    function findLiveBtn() {{
                      const want = /^(Сейчас|Live)\\s*\\(\\d+\\)\\s*$/i;
                      const candidates = Array.from(document.querySelectorAll('button')).filter(b => want.test(((b.innerText||b.textContent||'').trim())));
                      for (const b of candidates) {{
                        const p = (b.parentElement && (b.parentElement.innerText || b.parentElement.textContent) || '').toLowerCase();
                        if ((p.includes('законч') || p.includes('finished')) && (p.includes('предстоящ') || p.includes('upcoming'))) {{
                          return b;
                        }}
                      }}
                      // Fall back: aria-selected=true match
                      const selected = Array.from(document.querySelectorAll('button[aria-selected=\"true\"]'))
                        .find(b => want.test(((b.innerText||b.textContent||'').trim())));
                      return selected || candidates[0] || null;
                    }}
                    const liveBtn = findLiveBtn();
                    if (!liveBtn) return null;
                    // The match list sits "below" the tabs container. We try to locate the first
                    // element after the tabs that contains many visible match links.
                    let tabsRoot = liveBtn.parentElement;
                    while (tabsRoot && tabsRoot.querySelectorAll('button').length < 2) tabsRoot = tabsRoot.parentElement;
                    const main = (tabsRoot && tabsRoot.closest('main')) || document.body;
                    let best = null;
                    let bestN = 0;
                    const walker = document.createTreeWalker(main, NodeFilter.SHOW_ELEMENT);
                    let started = false;
                    while (walker.nextNode()) {{
                      const el = walker.currentNode;
                      if (el === tabsRoot) {{ started = true; continue; }}
                      if (!started) continue;
                      const links = Array.from(el.querySelectorAll({sel!r})).filter(isVisible);
                      const n = links.length;
                      if (n > bestN) {{ bestN = n; best = el; }}
                      if (bestN >= 10) break;
                    }}
                    if (best) return best;
                    let cur = tabsRoot || liveBtn.parentElement;
                    while (cur) {{
                      const n = Array.from(cur.querySelectorAll({sel!r})).filter(isVisible).length;
                      if (n >= 5) return cur;
                      cur = cur.parentElement;
                    }}
                    return null;
                  }}
                  const root = findRoot() || document;
                  return Array.from(root.querySelectorAll({sel!r}))
                    .filter(isVisible)
                    .map(a => a.getAttribute('href'))
                    .filter(Boolean);
                }}
                """
            )
            # Preserve DOM order but dedupe by appearance.
            cur = len({h for h in hrefs if isinstance(h, str)})
            if target is not None and cur >= target:
                break
            if cur <= prev and cur > 0:
                stalled += 1
                # Don't give up too early: new rows often appear after a longer async render.
                # If we're still below the UI count, try a more aggressive "scroll last row into view".
                if target is not None and cur < target and stalled in (6, 12, 18):
                    try:
                        await page.evaluate(
                            f"""
                            () => {{
                              function isVisible(el) {{
                                if (!el) return false;
                                const st = window.getComputedStyle(el);
                                if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                                const r = el.getClientRects();
                                if (!r || r.length === 0) return false;
                                const box = r[0];
                                return (box.width > 1 && box.height > 1);
                              }}
                              // Find last visible match link and scroll it into view.
                              const links = Array.from(document.querySelectorAll({sel!r})).filter(isVisible);
                              const last = links.length ? links[links.length - 1] : null;
                              if (last) {{
                                try {{ last.scrollIntoView({{block: 'end'}}); }} catch (e) {{}}
                              }} else {{
                                window.scrollBy(0, Math.max(1200, window.innerHeight));
                              }}
                            }}
                            """
                        )
                    except Exception:
                        pass
                    await page.wait_for_timeout(650)
                # Hard stop only after many stalls.
                if stalled >= 30 and (target is None or cur >= max(6, int(target * 0.6))):
                    break
            else:
                stalled = 0
            prev = cur
            # Scroll the live list container if possible, else window.
            try:
                await page.evaluate(
                    """
                    () => {
                      const want = /^(Сейчас|Live)\\s*\\(\\d+\\)\\s*$/i;
                      const candidates = Array.from(document.querySelectorAll('button')).filter(b => want.test((b.innerText||b.textContent||'').trim()));
                      let liveBtn = null;
                      for (const b of candidates) {
                        const p = (b.parentElement && (b.parentElement.innerText || b.parentElement.textContent) || '').toLowerCase();
                        if ((p.includes('законч') || p.includes('finished')) && (p.includes('предстоящ') || p.includes('upcoming'))) {
                          liveBtn = b; break;
                        }
                      }
                      if (!liveBtn) {
                        liveBtn = Array.from(document.querySelectorAll('button[aria-selected="true"]')).find(b => want.test((b.innerText||b.textContent||'').trim())) || candidates[0] || null;
                      }
                      let root = null;
                      if (liveBtn) {
                        let cur = liveBtn.parentElement;
                        while (cur) {
                          const n = cur.querySelectorAll('a[href*="/tennis/match/"][href*="#id:"]').length;
                          if (n >= 5) { root = cur; break; }
                          cur = cur.parentElement;
                        }
                      }
                      if (root && root.scrollHeight > root.clientHeight) {
                        root.scrollBy(0, Math.max(800, root.clientHeight * 0.9));
                        return;
                      }
                      window.scrollBy(0, Math.max(800, window.innerHeight * 0.9));
                    }
                    """
                )
            except Exception:
                await page.mouse.wheel(0, 1800)
            # A bit longer wait helps virtualized lists render the next chunk.
            await page.wait_for_timeout(450)
    except Exception:
        pass
    if not hrefs:
        hrefs = await page.evaluate(
            """() => Array.from(document.querySelectorAll('a[href*=\"/tennis/match/\"][href*=\"#id:\"]'))\n              .map(a => a.getAttribute('href'))\n              .filter(Boolean)"""
        )
    seen: set = set()
    out: List[str] = []
    for href in hrefs:
        if href in seen:
            continue
        seen.add(href)
        out.append("https://www.sofascore.com" + href)
        # If limit is not specified, use expected count from "Сейчас (N)".
        lim = limit
        if lim is None and expected_n is not None:
            lim = int(expected_n)
        if lim is not None and len(out) >= lim:
            break
    # Hard-cap to UI count if we managed to read it.
    if expected_n is not None and len(out) > expected_n:
        out = out[:expected_n]
    if os.getenv("THIRDSET_DEBUG_LIVE") in ("1", "true", "yes"):
        try:
            print(
                f"[live-dom] chosen={chosen_label!r} expected_n={expected_n} out={len(out)} limit={limit}",
                flush=True,
            )
        except Exception:
            pass
    return out


def parse_event_id_from_match_link(url: str) -> Optional[int]:
    marker = "#id:"
    if marker not in url:
        return None
    try:
        return int(url.split(marker, 1)[1])
    except Exception:
        return None


async def discover_live_cards_dom(page: Page, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    DOM-only live discovery:
    - opens Sofascore tennis page
    - clicks the "Сейчас (N)"/"Live (N)" pill
    - scrolls until we have N match cards (or stop growing)
    - extracts: eventId (from #id:), match url, home/away names, and a raw score text.

    This is the only method that can be made to exactly match what the user sees in the UI.
    """
    links = await discover_match_links(page, limit=limit)
    if not links:
        return []
    allowed_ids: List[int] = []
    allowed_set: set = set()
    for u in links:
        eid = parse_event_id_from_match_link(str(u))
        if isinstance(eid, int):
            allowed_ids.append(eid)
            allowed_set.add(eid)

    # We already scrolled inside discover_match_links.
    # Build one card per id from the anchor matching "#id:<eventId>".
    try:
        cards = await page.evaluate(
            """
            (allowedIds) => {
              function norm(s){
                return (s || '')
                  .replace(/[\\u00a0\\u200b\\u200c\\u200d\\ufeff]/g, ' ')
                  .replace(/\\s+/g, ' ')
                  .trim();
              }
              function isVisible(el) {
                if (!el) return false;
                const st = window.getComputedStyle(el);
                if (!st || st.display === 'none' || st.visibility === 'hidden' || st.opacity === '0') return false;
                const r = el.getClientRects();
                if (!r || r.length === 0) return false;
                const box = r[0];
                return (box.width > 1 && box.height > 1);
              }
              function findRoot() {
                const want = /^(Сейчас|Live)\\s*\\(\\d+\\)\\s*$/i;
                const candidates = Array.from(document.querySelectorAll('button')).filter(b => want.test((b.innerText||b.textContent||'').trim()));
                let liveBtn = null;
                for (const b of candidates) {
                  const p = (b.parentElement && (b.parentElement.innerText || b.parentElement.textContent) || '').toLowerCase();
                  if ((p.includes('законч') || p.includes('finished')) && (p.includes('предстоящ') || p.includes('upcoming'))) {
                    liveBtn = b; break;
                  }
                }
                if (!liveBtn) {
                  liveBtn = Array.from(document.querySelectorAll('button[aria-selected=\"true\"]'))
                    .find(b => want.test((b.innerText||b.textContent||'').trim())) || candidates[0] || null;
                }
                if (!liveBtn) return document;
                let tabsRoot = liveBtn.parentElement;
                while (tabsRoot && tabsRoot.querySelectorAll('button').length < 2) tabsRoot = tabsRoot.parentElement;
                const main = (tabsRoot && tabsRoot.closest('main')) || document.body;
                let best = null;
                let bestN = 0;
                const walker = document.createTreeWalker(main, NodeFilter.SHOW_ELEMENT);
                let started = false;
                while (walker.nextNode()) {
                  const el = walker.currentNode;
                  if (el === tabsRoot) { started = true; continue; }
                  if (!started) continue;
                  const links = Array.from(el.querySelectorAll('a[href*=\"/tennis/match/\"][href*=\"#id:\"]')).filter(isVisible);
                  const n = links.length;
                  if (n > bestN) { bestN = n; best = el; }
                  if (bestN >= 10) break;
                }
                return best || tabsRoot || main;
              }
              const root = findRoot() || document;
              const res = [];
              for (const id of (allowedIds || [])) {
                const sel = `a[href*=\"/tennis/match/\"][href*=\"#id:${id}\"]`;
                const a = Array.from(root.querySelectorAll(sel)).find(isVisible) || null;
                if (!a) {
                  res.push({ id, url: null, home: null, away: null, score: null });
                  continue;
                }
                const href = a.getAttribute('href') || '';
                const urlAbs = a.href || (location.origin + href);
                const url = urlAbs.split('#')[0];
                let home = null;
                let away = null;
                const rawLines = (a.innerText || a.textContent || '').split('\\n').map(norm).filter(Boolean);
                const nameLines = rawLines
                  .filter(t => /[A-Za-zА-Яа-яЁё]/.test(t))
                  .filter(t => !/^\\d{1,2}:\\d{2}$/.test(t))
                  .filter(t => !/^(ПВ|LIVE|Finished|Закончил(ся|ась)?|Сейчас|Now)$/i.test(t))
                  .filter(t => !/^\\d+(-\\d+)*$/.test(t))
                  .filter(t => !/(сет|set)$/i.test(t));
                if (nameLines.length >= 2) {
                  home = nameLines[0];
                  away = nameLines[1];
                }
                const scoreRoot =
                  a.querySelector('div.d_flex.flex-d_column.ai_flex-end') ||
                  a.querySelector('div[class*=\"ai_flex-end\"][class*=\"flex-d_column\"]') ||
                  null;
                const score = scoreRoot ? norm(scoreRoot.textContent) : null;
                res.push({ id, url, home, away, score });
              }
              return res;
            }
            """,
            allowed_ids,
        )
    except Exception:
        cards = []
    if not isinstance(cards, list):
        return []
    out: List[Dict[str, Any]] = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        eid = c.get("id")
        if not isinstance(eid, int) or eid not in allowed_set:
            continue
        out.append(c)
    return out

async def get_event_from_match_url_via_navigation(
    page: Page, *, match_url: str, event_id: int, timeout_ms: int = 15000
) -> Dict[str, Any]:
    captured = await _collect_json_via_navigation(
        page,
        url_to_open=match_url,
        predicates={"event": lambda r: r.url.endswith(f"/api/v1/event/{event_id}")},
        required_keys={"event"},
        timeout_ms=timeout_ms,
    )
    return captured["event"]


async def get_event_from_match_url_auto(page: Page, match_url: str, *, timeout_ms: int = 15000) -> Dict[str, Any]:
    """
    Navigate to a match URL and capture the first /api/v1/event/<id> response.
    Use when event_id is unknown (DOM-only discovery).
    """
    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    async def _on_response(resp: Response) -> None:
        if fut.done():
            return
        try:
            url = resp.url or ""
            if "/api/v1/event/" not in url:
                return
            if resp.status != 200:
                return
            # Response.json() can fail for cached resources (“No data found for resource…”).
            # Re-fetch inside the same page context to keep this robust.
            data = await fetch_json_via_page(page, url)
            if not fut.done():
                fut.set_result(data)
        except Exception as exc:
            # Ignore transient protocol errors; we'll keep listening until timeout.
            msg = str(exc)
            if "No data found for resource" in msg or "Network.getResponseBody" in msg:
                return
            if not fut.done():
                fut.set_exception(exc)

    page.on("response", _on_response)
    try:
        await page.goto(match_url, wait_until="domcontentloaded", timeout=timeout_ms)
        try:
            await page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except Exception:
            pass
        end = loop.time() + (timeout_ms / 1000.0)
        while loop.time() < end:
            if fut.done():
                break
            await asyncio.sleep(0.05)
        if not fut.done():
            raise SofascoreError("Timed out collecting: event")
        return fut.result()
    finally:
        page.remove_listener("response", _on_response)


def live_event_from_event_payload(payload: Dict[str, Any]) -> LiveEvent:
    e = payload.get("event") or {}
    home = e.get("homeTeam") or {}
    away = e.get("awayTeam") or {}
    return LiveEvent(
        id=int(e["id"]),
        slug=str(e.get("slug") or "").strip(),
        custom_id=str(e.get("customId") or "").strip(),
        home_team_id=int(home["id"]),
        home_team_name=str(home.get("name") or ""),
        away_team_id=int(away["id"]),
        away_team_name=str(away.get("name") or ""),
    )


async def get_event(page: Page, event_id: int) -> Dict[str, Any]:
    # This call is intentionally still available (fetch() inside the page), but
    # the CLI uses navigation-driven captures by default.
    return await fetch_json_via_page(page, f"{SOFASCORE_API_BASE}/event/{event_id}")

async def get_event_via_navigation(page: Page, event_id: int, *, timeout_ms: int = 15000) -> Dict[str, Any]:
    captured = await _collect_json_via_navigation(
        page,
        url_to_open=f"https://www.sofascore.com/event/{event_id}",
        predicates={"event": lambda r: r.url.endswith(f"/api/v1/event/{event_id}")},
        required_keys={"event"},
        timeout_ms=timeout_ms,
    )
    return captured["event"]


async def get_event_statistics(page: Page, event_id: int) -> Dict[str, Any]:
    return await fetch_json_via_page(page, f"{SOFASCORE_API_BASE}/event/{event_id}/statistics")

async def get_event_statistics_via_navigation(page: Page, event_id: int, *, timeout_ms: int = 15000) -> Dict[str, Any]:
    """
    Capture /api/v1/event/<id>/statistics from network while navigating like a real user.
    More CF-resistant than direct fetch().
    """
    captured = await _collect_json_via_navigation(
        page,
        url_to_open=f"https://www.sofascore.com/event/{event_id}",
        predicates={"statistics": lambda r: r.url.endswith(f"/api/v1/event/{event_id}/statistics")},
        required_keys={"statistics"},
        timeout_ms=timeout_ms,
    )
    return captured["statistics"]

async def get_event_votes(page: Page, event_id: int) -> Dict[str, Any]:
    return await fetch_json_via_page(page, f"{SOFASCORE_API_BASE}/event/{event_id}/votes")


async def get_team_last_events(page: Page, team_id: int, *, page_index: int = 0) -> Dict[str, Any]:
    return await fetch_json_via_page(page, f"{SOFASCORE_API_BASE}/team/{team_id}/events/last/{page_index}")

async def collect_match_team_last_events(
    page: Page,
    *,
    match_url: str,
    event_id: int,
    team_ids: List[int],
    timeout_ms: int = 15000,
) -> Dict[str, Dict[str, Any]]:
    predicates: Dict[str, Callable[[Response], bool]] = {
        "event": lambda r: r.url.endswith(f"/api/v1/event/{event_id}"),
    }
    for team_id in team_ids:
        predicates[f"team_last_{team_id}"] = (
            lambda r, tid=team_id: r.url.endswith(f"/api/v1/team/{tid}/events/last/0")
        )

    return await _collect_json_via_navigation(
        page,
        url_to_open=match_url,
        predicates=predicates,
        required_keys={"event"},
        timeout_ms=timeout_ms,
    )


async def collect_match_for_dominance(
    page: Page,
    *,
    match_url: str,
    event_id: int,
    team_ids: List[int],
    timeout_ms: int = 20000,
) -> Dict[str, Dict[str, Any]]:
    predicates: Dict[str, Callable[[Response], bool]] = {
        "event": lambda r: r.url.endswith(f"/api/v1/event/{event_id}"),
        "statistics": lambda r: r.url.endswith(f"/api/v1/event/{event_id}/statistics"),
    }
    for team_id in team_ids:
        predicates[f"team_last_{team_id}"] = (
            lambda r, tid=team_id: r.url.endswith(f"/api/v1/team/{tid}/events/last/0")
        )
    return await _collect_json_via_navigation(
        page,
        url_to_open=match_url,
        predicates=predicates,
        required_keys={"event", "statistics"},
        timeout_ms=timeout_ms,
    )


def pick_last_finished_events(team_last_events: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    for ev in team_last_events.get("events", []) or []:
        status = ev.get("status") or {}
        if (status.get("type") or "").lower() != "finished":
            continue
        picked.append(ev)
        if len(picked) >= limit:
            break
    return picked


async def get_last_finished_events(page: Page, team_id: int, *, limit: int) -> List[Dict[str, Any]]:
    page_index = 0
    picked: List[Dict[str, Any]] = []
    while len(picked) < limit:
        payload = await get_team_last_events(page, team_id, page_index=page_index)
        picked.extend(pick_last_finished_events(payload, limit=limit - len(picked)))
        if not payload.get("hasNextPage"):
            break
        page_index += 1
    return picked[:limit]


def _looks_like_doubles_name(name: str) -> bool:
    # Sofascore uses “A / B” for doubles pairs.
    return "/" in (name or "")


def is_singles_event(ev: Dict[str, Any]) -> bool:
    home = ev.get("homeTeam") or {}
    away = ev.get("awayTeam") or {}
    tournament = ev.get("tournament") or {}
    tournament_name = str(tournament.get("name") or "")

    if _looks_like_doubles_name(str(home.get("name") or "")):
        return False
    if _looks_like_doubles_name(str(away.get("name") or "")):
        return False
    if "doubles" in tournament_name.lower():
        return False
    return True


def pick_last_finished_singles_events(team_last_events: Dict[str, Any], *, limit: int) -> List[Dict[str, Any]]:
    picked: List[Dict[str, Any]] = []
    for ev in team_last_events.get("events", []) or []:
        status = ev.get("status") or {}
        if (status.get("type") or "").lower() != "finished":
            continue
        if not is_singles_event(ev):
            continue
        picked.append(ev)
        if len(picked) >= limit:
            break
    return picked


async def get_last_finished_singles_events(page: Page, team_id: int, *, limit: int) -> List[Dict[str, Any]]:
    page_index = 0
    picked: List[Dict[str, Any]] = []
    while len(picked) < limit:
        payload = await get_team_last_events(page, team_id, page_index=page_index)
        picked.extend(pick_last_finished_singles_events(payload, limit=limit - len(picked)))
        if not payload.get("hasNextPage"):
            break
        page_index += 1
    return picked[:limit]


def summarize_event_for_team(ev: Dict[str, Any], *, team_id: int) -> Dict[str, Any]:
    home = ev.get("homeTeam") or {}
    away = ev.get("awayTeam") or {}
    home_id = home.get("id")
    away_id = away.get("id")

    def score_side(side: str) -> Dict[str, Any]:
        s = ev.get(f"{side}Score") or {}
        return {k: s.get(k) for k in ("current", "display", "period1", "period2", "period3", "period4", "period5")}

    is_home = int(home_id) == int(team_id) if home_id is not None else False
    opponent = away if is_home else home

    winner_code = ev.get("winnerCode")  # 1=home,2=away
    won = None
    if winner_code in (1, 2):
        won = (winner_code == 1 and is_home) or (winner_code == 2 and not is_home)

    return {
        "eventId": ev.get("id"),
        "startTimestamp": ev.get("startTimestamp"),
        "tournament": (ev.get("tournament") or {}).get("name"),
        "opponentName": opponent.get("name"),
        "isHome": is_home,
        "won": won,
        "homeScore": score_side("home"),
        "awayScore": score_side("away"),
    }


def extract_statistics_group(stats_json: Dict[str, Any], *, group_name: str = "Service") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for period in stats_json.get("statistics", []) or []:
        if (period.get("period") or "").upper() != "ALL":
            continue
        for group in period.get("groups", []) or []:
            if group.get("groupName") != group_name:
                continue
            for item in group.get("statisticsItems", []) or []:
                out.append(
                    {
                        "name": item.get("name"),
                        "home": item.get("home"),
                        "away": item.get("away"),
                        "homeValue": item.get("homeValue"),
                        "awayValue": item.get("awayValue"),
                    }
                )
    return out


def extract_points_from_statistics(
    stats_json: Dict[str, Any], *, periods: Tuple[str, ...]
) -> Optional[DominanceLivePoints]:
    spw_home = 0
    spw_away = 0
    rpw_home = 0
    rpw_away = 0
    seen = 0

    wanted = {p.upper() for p in periods}
    for st in stats_json.get("statistics", []) or []:
        if (st.get("period") or "").upper() not in wanted:
            continue
        for group in st.get("groups", []) or []:
            if group.get("groupName") != "Points":
                continue
            for item in group.get("statisticsItems", []) or []:
                name = item.get("name")
                if name == "Service points won":
                    if item.get("homeValue") is None or item.get("awayValue") is None:
                        continue
                    spw_home += int(item.get("homeValue") or 0)
                    spw_away += int(item.get("awayValue") or 0)
                    seen += 1
                elif name == "Receiver points won":
                    if item.get("homeValue") is None or item.get("awayValue") is None:
                        continue
                    rpw_home += int(item.get("homeValue") or 0)
                    rpw_away += int(item.get("awayValue") or 0)
                    seen += 1

    # We expect to see both metrics per period.
    if seen == 0:
        return None
    return DominanceLivePoints(
        spw_home=spw_home,
        rpw_home=rpw_home,
        spw_away=spw_away,
        rpw_away=rpw_away,
    )


def match_is_bo3_from_statistics(stats_json: Dict[str, Any]) -> Optional[bool]:
    # Use presence of 4TH/5TH periods as a BO5 signal.
    periods = {(s.get("period") or "").upper() for s in (stats_json.get("statistics") or [])}
    if "4TH" in periods or "5TH" in periods:
        return False
    if "3RD" in periods:
        return True
    # Could still be BO3 but ended 2 sets; treat as BO3 if only 1ST/2ND exist.
    if "1ST" in periods or "2ND" in periods:
        return True
    return None
