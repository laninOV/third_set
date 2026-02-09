#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from third_set.tg_runtime import TelegramClient, get_telegram_config


@dataclass
class Step:
    command: str
    expected_handler: str
    wait_s: float = 6.0
    expect_active: Optional[str] = None


def _read_new_text(path: Path, offset: int) -> Tuple[str, int]:
    if not path.exists():
        return "", offset
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)
        data = f.read()
        new_off = f.tell()
    return data, new_off


def _wait_for_handler(log_path: Path, expected: str, offset: int, timeout_s: float, *, expect_active: Optional[str] = None) -> Tuple[bool, int, str]:
    deadline = time.time() + timeout_s
    buf = ""
    saw_handler = False
    while time.time() < deadline:
        chunk, offset = _read_new_text(log_path, offset)
        if chunk:
            buf += chunk
            if f"-> {expected}" in buf:
                saw_handler = True
                if expect_active is None:
                    return True, offset, buf
            if saw_handler and expect_active is not None:
                if f"active_task={expect_active}" in buf:
                    return True, offset, buf
        time.sleep(0.35)
    return False, offset, buf


def _send(client: TelegramClient, text: str) -> None:
    res = client.send_text_result(text)
    if not isinstance(res, dict) or not res.get("ok"):
        raise RuntimeError(f"send failed for {text!r}: {res}")


def run() -> int:
    ap = argparse.ArgumentParser(description="Telegram bot smoke matrix for third_set tg-bot")
    ap.add_argument("--log", default="third_set.log", help="Path to tg-bot console log file")
    ap.add_argument("--event-id", type=int, default=0, help="Valid event id for 'Анализ по ID' flow")
    ap.add_argument("--timeout", type=float, default=20.0, help="Per-step wait timeout for handler in logs")
    args = ap.parse_args()

    cfg = get_telegram_config(token="", chat_id="")
    if not cfg:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID (or THIRDSET_*)")

    log_path = Path(args.log)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    client = TelegramClient(token=cfg.token, chat_id=cfg.chat_id)
    offset = log_path.stat().st_size

    steps: List[Step] = [
        Step("Список live", "list_live"),
        Step("Список upcoming", "list_upcoming"),
        Step("Анализ upcoming", "analyze_upcoming_prompt"),
        Step("1", "pending_upcoming_hours", wait_s=35.0, expect_active="running"),
        Step("Стоп текущий", "stop_current", wait_s=20.0, expect_active="idle"),
        Step("Анализ всех live", "analyze_all", wait_s=35.0, expect_active="running"),
        Step("Стоп текущий", "stop_current"),
        Step("Выключить бота", "shutdown_prompt"),
        Step("СТОП", "shutdown_confirm"),
    ]

    if args.event_id > 0:
        steps.insert(4, Step("Анализ по ID", "analyze_by_id_prompt"))
        steps.insert(5, Step(str(int(args.event_id)), "pending_id_value", wait_s=25.0))

    print(f"SMOKE start: log={log_path} steps={len(steps)}")
    passed = 0
    failed = 0

    for idx, step in enumerate(steps, 1):
        print(f"[{idx}/{len(steps)}] send: {step.command}")
        _send(client, step.command)
        ok, offset, sample = _wait_for_handler(
            log_path,
            step.expected_handler,
            offset,
            timeout_s=max(3.0, step.wait_s, args.timeout),
            expect_active=step.expect_active,
        )
        if ok:
            passed += 1
            print(f"  OK -> {step.expected_handler}")
        else:
            failed += 1
            hint = ""
            m = re.findall(r"\[TG\] route raw=.*", sample)
            if m:
                hint = f" | recent_route={m[-1]}"
            print(f"  FAIL -> {step.expected_handler}{hint}")

    print(f"SMOKE done: pass={passed} fail={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run())
