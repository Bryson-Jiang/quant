"""确认结果文件可落盘，供 PyCharm 和 Excel 查看。"""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.report import save_report


class ReportTests(unittest.TestCase):
    def test_saves_core_outputs(self) -> None:
        days = pd.bdate_range("2024-01-01", periods=3)
        prices = pd.DataFrame({"510300": [1.0, 1.01, 1.02], "A": [1.0, 1.1, 1.2]}, index=days)
        signals = pd.DataFrame({"A": [1.0]}, index=[days[0]])
        result = run_backtest(prices, signals, date(2024, 1, 1), date(2024, 1, 3))
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            paths = save_report(result, signals, output_dir)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertTrue((output_dir / "equity.csv").exists())
            self.assertTrue((output_dir / "trades.csv").exists())
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("max_drawdown", summary)


if __name__ == "__main__":
    unittest.main()
