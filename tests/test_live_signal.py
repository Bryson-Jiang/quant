"""实盘信号日期必须来自正式开市日历，不能把周中最后一条行情当周末。"""

import unittest
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from live_signal import latest_actionable_week


class LiveScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2026-09-14", "2026-10-05")
        is_open = (dates.weekday < 5).astype(int)
        calendar = pd.DataFrame({"is_open": is_open}, index=dates)
        calendar.loc["2026-09-25", "is_open"] = 0  # 中秋节：周四为本周最后交易日
        calendar.loc["2026-10-01":"2026-10-05", "is_open"] = 0
        self.calendar = calendar

    def now(self, day: int, hour: int) -> datetime:
        return datetime(2026, 9, day, hour, tzinfo=ZoneInfo("Asia/Shanghai"))

    def test_holiday_week_thursday_signal_monday_execution(self) -> None:
        self.assertEqual(latest_actionable_week(self.calendar, date(2026, 9, 26), self.now(26, 12)),
                         (date(2026, 9, 24), date(2026, 9, 28), "ready"))

    def test_midweek_is_not_new_signal(self) -> None:
        self.assertEqual(latest_actionable_week(self.calendar, date(2026, 9, 23), self.now(23, 19)),
                         (date(2026, 9, 18), date(2026, 9, 21), "no_new_rebalance"))

    def test_execution_window_closes_and_then_signal_expires(self) -> None:
        self.assertEqual(latest_actionable_week(self.calendar, date(2026, 9, 28),
                                                datetime(2026, 9, 28, 14, 55,
                                                         tzinfo=ZoneInfo("Asia/Shanghai")))[2], "ready")
        self.assertEqual(latest_actionable_week(self.calendar, date(2026, 9, 28), self.now(28, 15))[2],
                         "execution_window_closed")
        self.assertEqual(latest_actionable_week(self.calendar, date(2026, 9, 29), self.now(29, 19))[2],
                         "no_new_rebalance")

    def test_backdated_query_cannot_look_actionable(self) -> None:
        self.assertEqual(latest_actionable_week(self.calendar, date(2026, 9, 26), self.now(29, 19))[2],
                         "historical_only")


if __name__ == "__main__":
    unittest.main()
