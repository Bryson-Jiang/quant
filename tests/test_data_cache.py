"""验证网络获取和离线缓存使用同一份日线。"""

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from etf_rotation.data import _fetch_history, _windows, load_symbol, tushare_code


class DataCacheTests(unittest.TestCase):
    def test_fetch_then_offline_read(self) -> None:
        days = pd.to_datetime(["2024-01-02", "2024-01-03"])
        source = pd.DataFrame({"raw_close": [1.0, 1.0], "adj_factor": [1.0, 1.2],
                               "close": [1.0, 1.2]}, index=pd.DatetimeIndex(days, name="date"))
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            with patch("etf_rotation.data._fetch_history", return_value=source) as fetch:
                online = load_symbol("510300", date(2024, 1, 1), date(2024, 1, 5), cache)
                fetch.assert_called_once()
            offline = load_symbol("510300", date(2024, 1, 1), date(2024, 1, 5), cache, offline=True)
            pd.testing.assert_series_equal(online, offline, check_freq=False)

    def test_extension_only_fetches_recent_window_and_updates_revision(self) -> None:
        days = pd.bdate_range("2024-01-01", "2024-03-05")
        history = pd.DataFrame({"raw_close": [1.0] * len(days),
                                "adj_factor": [1.0] * len(days),
                                "close": [1.0] * len(days)},
                               index=pd.DatetimeIndex(days, name="date"))
        revised = history.copy()
        revised.loc[pd.Timestamp("2024-02-20"), ["adj_factor", "close"]] = 1.2
        calls = []

        def fetch(symbol, start, end, *, allow_empty=False):
            calls.append((start, end, allow_empty))
            source = history if len(calls) == 1 else revised
            return source.loc[(source.index.date >= start) & (source.index.date <= end)]

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            with patch("etf_rotation.data._fetch_history", side_effect=fetch):
                load_symbol("510300", date(2024, 1, 1), date(2024, 2, 29), cache)
                updated = load_symbol("510300", date(2024, 1, 1), date(2024, 3, 5), cache)
                load_symbol("510300", date(2024, 1, 1), date(2024, 3, 5), cache)
            self.assertEqual(len(calls), 2)
            self.assertGreater(calls[1][0], date(2024, 1, 1))
            self.assertEqual(calls[1][1], date(2024, 3, 5))
            self.assertTrue(calls[1][2])
            self.assertAlmostEqual(updated.at[pd.Timestamp("2024-02-20")], 1.2)
            offline = load_symbol("510300", date(2024, 1, 1), date(2024, 3, 5), cache, offline=True)
            pd.testing.assert_series_equal(updated, offline, check_freq=False)

    def test_extension_without_new_trading_day_updates_coverage(self) -> None:
        day = pd.Timestamp("2024-01-05")
        source = pd.DataFrame({"raw_close": [1.0], "adj_factor": [1.0], "close": [1.0]},
                              index=pd.DatetimeIndex([day], name="date"))
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = Path(temp_dir)
            with patch("etf_rotation.data._fetch_history", side_effect=[source, source.iloc[:0]]) as fetch:
                load_symbol("510300", date(2024, 1, 5), date(2024, 1, 5), cache)
                extended = load_symbol("510300", date(2024, 1, 5), date(2024, 1, 7), cache)
                self.assertEqual(fetch.call_count, 2)
            self.assertEqual(extended.index.tolist(), [day])
            load_symbol("510300", date(2024, 1, 5), date(2024, 1, 7), cache, offline=True)

    def test_exchange_suffix_and_adjusted_close(self) -> None:
        self.assertEqual(tushare_code("510300"), "510300.SH")
        self.assertEqual(tushare_code("159915"), "159915.SZ")

        class FakeClient:
            def fund_daily(self, **kwargs):
                self.daily_args = kwargs
                return pd.DataFrame({"trade_date": ["20240103", "20240102"], "close": [0.9, 1.0]})

            def fund_adj(self, **kwargs):
                self.factor_args = kwargs
                return pd.DataFrame({"trade_date": ["20240103", "20240102"],
                                     "adj_factor": [10 / 9, 1.0]})

        client = FakeClient()
        with patch("etf_rotation.data._client", return_value=client):
            adjusted = _fetch_history("510300", date(2024, 1, 2), date(2024, 1, 3))
        self.assertEqual(adjusted["raw_close"].tolist(), [1.0, 0.9])
        self.assertAlmostEqual(adjusted["close"].iloc[0], 1.0)
        self.assertAlmostEqual(adjusted["close"].iloc[1], 1.0)
        self.assertEqual(client.daily_args["ts_code"], "510300.SH")
        self.assertEqual(client.factor_args["start_date"], "20240102")

    def test_missing_factor_fails_instead_of_using_raw_close(self) -> None:
        class FakeClient:
            def fund_daily(self, **kwargs):
                return pd.DataFrame({"trade_date": ["20240102", "20240103"], "close": [1.0, 1.1]})

            def fund_adj(self, **kwargs):
                return pd.DataFrame({"trade_date": ["20240102"], "adj_factor": [1.0]})

        with patch("etf_rotation.data._client", return_value=FakeClient()):
            with self.assertRaisesRegex(ValueError, "缺少复权因子"):
                _fetch_history("510300", date(2024, 1, 2), date(2024, 1, 3))

    def test_long_requests_are_split_without_overlap(self) -> None:
        windows = list(_windows(date(2016, 1, 1), date(2026, 9, 1)))
        self.assertGreater(len(windows), 1)
        self.assertEqual(windows[0][0], date(2016, 1, 1))
        self.assertEqual(windows[-1][1], date(2026, 9, 1))
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous[1].toordinal() + 1, current[0].toordinal())


if __name__ == "__main__":
    unittest.main()
