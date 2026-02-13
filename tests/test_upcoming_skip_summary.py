import unittest

from third_set.analyzer_core.history_sources import history_source_unavailable
from third_set.cli import _skip_reason_parts, _sofascore_error_code
from third_set.sofascore import SofascoreError


class UpcomingSkipSummaryTests(unittest.TestCase):
    def test_skip_reason_parts_includes_codes_block(self) -> None:
        parts = _skip_reason_parts(
            {
                "bad_card": 1,
                "analyze_failed": 3,
                "analyze_failed_history_unavailable": 2,
                "analyze_failed_timeout_storm": 1,
            },
            base_keys=("bad_card", "analyze_failed"),
            include_codes=True,
        )
        self.assertIn("bad_card=1", parts)
        self.assertIn("analyze_failed=3", parts)
        self.assertTrue(any(p.startswith("codes[") for p in parts))
        text = " | ".join(parts)
        self.assertIn("history_unavailable=2", text)
        self.assertIn("timeout_storm=1", text)

    def test_skip_reason_parts_without_codes(self) -> None:
        parts = _skip_reason_parts(
            {"analyze_failed": 2, "analyze_failed_timeout_storm": 2},
            base_keys=("analyze_failed",),
            include_codes=False,
        )
        self.assertEqual(parts, ["analyze_failed=2"])

    def test_sofascore_error_code_normalization(self) -> None:
        code, text = _sofascore_error_code(SofascoreError("503 backend read error code=varnish_503"))
        self.assertEqual(code, "source_blocked")
        self.assertNotIn("code=", text.lower())

    def test_sofascore_error_code_history_source_unavailable(self) -> None:
        code, _text = _sofascore_error_code(SofascoreError("code=history_source_unavailable Источник history недоступен"))
        self.assertEqual(code, "history_source_unavailable")

    def test_skip_reason_parts_promotes_history_source_unavailable(self) -> None:
        parts = _skip_reason_parts(
            {"analyze_failed": 4, "analyze_failed_history_source_unavailable": 3},
            base_keys=("analyze_failed",),
            include_codes=True,
        )
        self.assertIn("analyze_failed=4", parts)
        self.assertIn("history_source_unavailable=3", parts)

    def test_history_source_unavailable_detector(self) -> None:
        self.assertTrue(
            history_source_unavailable(
                links_count=0,
                api_history_fetch_failed=1,
                candidates=0,
                valid=0,
            )
        )
        self.assertFalse(
            history_source_unavailable(
                links_count=4,
                api_history_fetch_failed=0,
                candidates=10,
                valid=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
