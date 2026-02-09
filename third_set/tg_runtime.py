from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional
from urllib import parse as _urlparse
from urllib import request as _urlrequest
from urllib.error import HTTPError


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    chat_id: str


class TelegramClient:
    """
    Minimal Telegram client (sendMessage) borrowed from /Users/lanin/Development/autobet.
    Uses urllib only (no extra deps) and has a simple client-side rate limiter.
    """

    def __init__(self, token: str, chat_id: str):
        self.token = str(token)
        self.chat_id = str(chat_id)
        self._send_call_times: Deque[float] = deque()
        self._poll_call_times: Deque[float] = deque()

    def _resolve_max_rpm(self, kind: str) -> int:
        # Keep backward compatibility for send side with THIRDSET_TG_MAX_RPM.
        if kind == "send":
            raw = os.getenv("THIRDSET_TG_SEND_MAX_RPM")
            if raw in (None, ""):
                raw = os.getenv("THIRDSET_TG_MAX_RPM")
            default = 18
        else:
            raw = os.getenv("THIRDSET_TG_POLL_MAX_RPM")
            default = 0
        try:
            return int(raw) if raw not in (None, "") else default
        except Exception:
            return default

    def _throttle(self, kind: str) -> None:
        max_rpm = self._resolve_max_rpm(kind)
        if max_rpm <= 0:
            return

        now = time.time()
        window = 60.0
        dq = self._send_call_times if kind == "send" else self._poll_call_times
        try:
            while dq and (now - dq[0] > window):
                dq.popleft()
        except Exception:
            dq.clear()

        if len(dq) >= max_rpm:
            oldest = dq[0]
            sleep_for = window - (now - oldest) + 0.1
            if sleep_for > 0:
                time.sleep(min(sleep_for, 5.0))
            now = time.time()
            try:
                while dq and (now - dq[0] > window):
                    dq.popleft()
            except Exception:
                dq.clear()
        dq.append(now)

    def _api(self, method: str, payload: Dict[str, str], timeout: int = 20) -> Dict:
        try:
            self._throttle("send")
            api_base = f"https://api.telegram.org/bot{self.token}/{method}"
            data = _urlparse.urlencode(payload).encode("utf-8")
            req = _urlrequest.Request(api_base, data=data)
            with _urlrequest.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            desc = None
            if isinstance(e, HTTPError):
                try:
                    body = e.read().decode("utf-8", "ignore")
                    try:
                        parsed = json.loads(body)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        desc = body
                except Exception:
                    pass
            return {"ok": False, "description": (desc or str(e))}

    def _api_get(self, method: str, payload: Dict[str, Any], timeout: int = 20) -> Dict:
        try:
            self._throttle("poll")
            qs = _urlparse.urlencode({k: v for k, v in payload.items() if v is not None})
            api_base = f"https://api.telegram.org/bot{self.token}/{method}?{qs}"
            req = _urlrequest.Request(api_base, method="GET")
            with _urlrequest.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            desc = None
            if isinstance(e, HTTPError):
                try:
                    body = e.read().decode("utf-8", "ignore")
                    try:
                        parsed = json.loads(body)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        desc = body
                except Exception:
                    pass
            return {"ok": False, "description": (desc or str(e))}

    def send_text(self, text: str, *, parse_mode: str = "HTML") -> Optional[int]:
        r = self.send_text_result(text, parse_mode=parse_mode)
        if r.get("ok") and isinstance(r.get("result"), dict):
            mid = r["result"].get("message_id")
            return mid if isinstance(mid, int) else None
        return None

    def send_text_result(self, text: str, *, parse_mode: str = "HTML", reply_markup: Optional[Dict[str, Any]] = None) -> Dict:
        payload = {
            "chat_id": self.chat_id,
            "text": (text or "")[:3800],
            "parse_mode": parse_mode,
            "disable_web_page_preview": "true",
        }
        if reply_markup is not None:
            try:
                payload["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False)
            except Exception:
                pass
        r = self._api("sendMessage", payload)
        if r.get("ok"):
            return r
        # Fallback plain text if HTML breaks
        r2 = self._api(
            "sendMessage",
            {"chat_id": self.chat_id, "text": (text or "")[:3800], "disable_web_page_preview": "true"},
        )
        return r2

    def get_updates(self, *, offset: Optional[int] = None, limit: int = 50, timeout: int = 0) -> Dict:
        # Note: timeout is Telegram long-poll seconds; keep 0 by default (instant).
        t = int(timeout)
        return self._api_get(
            "getUpdates",
            {"offset": offset, "limit": int(limit), "timeout": t},
            timeout=max(25, t + 10),
        )

    def get_webhook_info(self) -> Dict:
        return self._api_get("getWebhookInfo", {}, timeout=20)

    def delete_webhook(self, *, drop_pending_updates: bool = False) -> Dict:
        return self._api(
            "deleteWebhook",
            {"drop_pending_updates": "true" if drop_pending_updates else "false"},
            timeout=20,
        )


def get_telegram_config(*, token: str = "", chat_id: str = "") -> Optional[TelegramConfig]:
    t = (token or "").strip() or (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip() or (os.getenv("THIRDSET_TG_TOKEN") or "").strip()
    c = (chat_id or "").strip() or (os.getenv("TELEGRAM_CHAT_ID") or "").strip() or (os.getenv("THIRDSET_TG_CHAT_ID") or "").strip()
    if not t or not c:
        return None
    return TelegramConfig(token=t, chat_id=c)
