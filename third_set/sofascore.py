from __future__ import annotations

import json
import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from playwright.async_api import Page, Response

from third_set.dominance import DominanceLivePoints

SOFASCORE_TENNIS_URL = "https://www.sofascore.com/ru/tennis"
SOFASCORE_API_BASE = "https://www.sofascore.com/api/v1"
SOFASCORE_TENNIS_LIVE_API = f"{SOFASCORE_API_BASE}/sport/tennis/events/live"


class SofascoreError(RuntimeError):
    pass


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
        raise SofascoreError(f"HTTP {status} for {url}: {text[:200]}")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SofascoreError(f"Invalid JSON for {url}: {text[:200]}") from e


async def get_live_match_links(page: Page, *, limit: Optional[int] = None) -> List[str]:
    """
    Returns match URLs for live tennis events only.

    Each URL includes a synthetic fragment `#id:<eventId>` so callers can recover the event id.
    """
    await page.goto(SOFASCORE_TENNIS_URL, wait_until="domcontentloaded", timeout=25000)
    try:
        await page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    await asyncio.sleep(3)
    try:
        data = await fetch_json_via_page(page, SOFASCORE_TENNIS_LIVE_API)
    except SofascoreError as exc:
        msg = str(exc)
        if "403" in msg and "challenge" in msg.lower():
            _dbg("Live API blocked by challenge; waiting 10s before retry")
            await asyncio.sleep(10)
            data = await fetch_json_via_page(page, SOFASCORE_TENNIS_LIVE_API)
        else:
            raise
    out: List[str] = []
    seen: set = set()
    for ev in data.get("events", []) or []:
        try:
            event_id = int(ev["id"])
            slug = str(ev["slug"])
            custom_id = str(ev["customId"])
        except Exception:
            continue
        url = f"https://www.sofascore.com/ru/tennis/match/{slug}/{custom_id}#id:{event_id}"
        if url in seen:
            continue
        seen.add(url)
        out.append(url)
        if limit is not None and len(out) >= limit:
            break
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

    async def _handle_response(resp: Response) -> None:
        for key, pred in predicates.items():
            fut = futures[key]
            if fut.done():
                continue
            try:
                if not pred(resp):
                    continue
                if resp.status != 200:
                    fut.set_exception(SofascoreError(f"HTTP {resp.status} for {resp.url}"))
                    continue
                # Getting the body from a Playwright Response can fail for cached
                # resources (“No resource with given identifier found”). To keep
                # this robust, re-fetch the JSON inside the same page context.
                fut.set_result(await fetch_json_via_page(page, resp.url))
            except Exception as e:
                if not fut.done():
                    fut.set_exception(e)

    def _on_response(resp: Response) -> None:
        asyncio.create_task(_handle_response(resp))

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
    await page.goto(SOFASCORE_TENNIS_URL, wait_until="networkidle", timeout=25000)
    hrefs: List[str] = await page.evaluate(
        """
        () => Array.from(document.querySelectorAll('a[href*="/ru/tennis/match/"][href*="#id:"]'))
          .map(a => a.getAttribute('href'))
          .filter(Boolean)
        """
    )
    seen: set = set()
    out: List[str] = []
    for href in hrefs:
        if href in seen:
            continue
        seen.add(href)
        out.append("https://www.sofascore.com" + href)
        if limit is not None and len(out) >= limit:
            break
    return out


def parse_event_id_from_match_link(url: str) -> Optional[int]:
    marker = "#id:"
    if marker not in url:
        return None
    try:
        return int(url.split(marker, 1)[1])
    except Exception:
        return None


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
