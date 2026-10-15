"""当前实盘候选的 16 只 ETF 标的池。"""

from __future__ import annotations

# 静态标的池；新上市 ETF 在其真实上市前不会获得动量分数。
# “竣工”暂按可能的笔误“军工”处理，见 README。
ETF_NAMES: dict[str, str] = {
    "510300": "沪深300ETF",
    "159915": "创业板ETF",
    "159941": "纳指ETF",
    "518880": "黄金ETF",
    "511260": "十年国债ETF",
    "159995": "芯片ETF（半导体）",
    "159875": "新能源ETF",
    "159928": "消费ETF",
    "512660": "军工ETF",
    "515070": "人工智能ETF",
    "512880": "证券ETF",
    "512800": "银行ETF（华宝）",
    "513520": "日经ETF",
    "561360": "石油产业ETF",
    "159985": "豆粕期货ETF",
    "159980": "有色金属ETF",
}

BENCHMARK = "510300"
PREVIOUS_STRATEGY_NAME = "weekly_16_etf_fast_40_20_single30"
STRATEGY_NAME = "weekly_16_etf_fast_40_20_vol7_buffer4_top1_single30"
ADJUST = "hfq"  # Tushare 原始价 × 基金复权因子；不是可成交的真实价格。
