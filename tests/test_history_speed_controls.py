import unittest
from unittest.mock import patch

from third_set.analyzer_core.orchestrator import (
    _should_enable_final_attempt_guard,
    _should_restore_wait_stats,
)
from third_set.dom_stats import _resolve_step_retries


class HistorySpeedControlsTests(unittest.TestCase):
    def test_resolve_step_retries_uses_defaults(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            out = _resolve_step_retries(step_retries_default=2)
        self.assertEqual(out["goto_match"], 2)
        self.assertEqual(out["open_statistics_tab"], 2)
        self.assertEqual(out["select_period"], 2)

    def test_resolve_step_retries_env_override(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "THIRDSET_DOM_STEP_RETRIES_GOTO": "2",
                "THIRDSET_DOM_STEP_RETRIES_OPEN_TAB": "1",
                "THIRDSET_DOM_STEP_RETRIES_PERIOD": "1",
            },
            clear=False,
        ):
            out = _resolve_step_retries(step_retries_default=3)
        self.assertEqual(out["goto_match"], 2)
        self.assertEqual(out["open_statistics_tab"], 1)
        self.assertEqual(out["select_period"], 1)

    def test_wait_stats_restore_rule(self) -> None:
        self.assertTrue(
            _should_restore_wait_stats(
                wait_stats_ms=2000,
                wait_stats_base_ms=900,
                success_streak=2,
            )
        )
        self.assertFalse(
            _should_restore_wait_stats(
                wait_stats_ms=2000,
                wait_stats_base_ms=900,
                success_streak=1,
            )
        )
        self.assertFalse(
            _should_restore_wait_stats(
                wait_stats_ms=900,
                wait_stats_base_ms=900,
                success_streak=3,
            )
        )

    def test_final_attempt_guard_rule(self) -> None:
        self.assertTrue(
            _should_enable_final_attempt_guard(
                slow_mode=True,
                no_limits_enabled=False,
                have_valid=4,
                max_history=5,
                remaining_side_s=12.0,
                per_event_timeout_s=14.0,
            )
        )
        self.assertFalse(
            _should_enable_final_attempt_guard(
                slow_mode=True,
                no_limits_enabled=True,
                have_valid=4,
                max_history=5,
                remaining_side_s=12.0,
                per_event_timeout_s=14.0,
            )
        )
        self.assertFalse(
            _should_enable_final_attempt_guard(
                slow_mode=True,
                no_limits_enabled=False,
                have_valid=3,
                max_history=5,
                remaining_side_s=12.0,
                per_event_timeout_s=14.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
