# Helper utilities derived from the autobet project to keep Chromium sessions
# alive and behaving like a real browser.

from typing import Optional

from playwright.async_api import Page


async def disable_network_cache(page: Page) -> None:
    """
    Reduce cache usage to keep long-running contexts responsive.
    """
    try:
        ctx = page.context
        if ctx is None:
            return
        # The Playwright context exposes new_cdp_session for Chromium.
        new_sess = getattr(ctx, "new_cdp_session", None)
        if not callable(new_sess):
            return
        sess = await new_sess(page)
        try:
            await sess.send("Network.enable")
        except Exception:
            pass
        await sess.send("Network.setCacheDisabled", {"cacheDisabled": True})
    except Exception:
        pass


async def clear_browser_cache(page: Page) -> None:
    """
    Clear HTTP cache and CacheStorage for known origins (best-effort).
    """
    try:
        ctx = page.context
        if ctx is None:
            return
        new_sess = getattr(ctx, "new_cdp_session", None)
        if not callable(new_sess):
            return
        sess = await new_sess(page)
        try:
            await sess.send("Network.enable")
        except Exception:
            pass
        await sess.send("Network.clearBrowserCache")
        origins = (
            "https://chat.openai.com",
            "https://www.sofascore.com",
            "https://www.sport-liga.pro",
        )
        for origin in origins:
            try:
                await sess.send(
                    "Storage.clearDataForOrigin",
                    {"origin": origin, "storageTypes": "cache_storage,service_workers"},
                )
            except Exception:
                pass
    except Exception:
        pass


def page_is_usable(page: Optional[Page]) -> bool:
    if page is None:
        return False
    try:
        is_closed = getattr(page, "is_closed", None)
        if callable(is_closed) and is_closed():
            return False
    except Exception:
        return False
    return True
