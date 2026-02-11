# Helper utilities derived from the autobet project to keep Chromium sessions
# alive and behaving like a real browser.

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

