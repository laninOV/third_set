"""Overlay/consent handling helpers."""

from __future__ import annotations

import re

from playwright.async_api import Page


async def _dismiss_overlays_basic(page: Page) -> None:
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass

    try:
        age_modal = page.locator("text=Age Verification")
        if await age_modal.count():
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
            confirm = page.locator("button:has-text('Подтвердить'), button:has-text('Confirm')")
            if await confirm.count():
                try:
                    await confirm.first.click(timeout=2000, force=True)
                except Exception:
                    pass
                await page.wait_for_timeout(400)
    except Exception:
        pass

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
