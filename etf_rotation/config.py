"""17只核心ETF及独立红利反转卫星；保留旧16只研究定义。"""

from __future__ import annotations

# 静态标的池；新上市 ETF 在其真实上市前不会获得动量分数。
# “竣工”暂按可能的笔误“军工”处理，见 README。
LEGACY_CORE_NAMES: dict[str, str] = {
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

ETF_NAMES = {**LEGACY_CORE_NAMES, "513180": "恒生科技ETF（华夏）"}

BENCHMARK = "510300"
PREVIOUS_STRATEGY_NAME = "weekly_16_etf_fast_40_20_single30"
BUFFERED_MOMENTUM_STRATEGY_NAME = "weekly_16_etf_fast_40_20_vol7_buffer4_top1_single30"
LEGACY_CORE_STRATEGY_NAME = "weekly_16_etf_fast_40_20_vol8_ma20a05_buffer4_top1_single30"
LEGACY_PORTFOLIO_STRATEGY_NAME = "weekly_core16_vol8_ma20a05_div512890_ma200_in0_out4_defense30"
PRE_IVOL_CORE_STRATEGY_NAME = "weekly_17_etf_fast_40_20_vol8_ma20a05_buffer4_top1_single30"
SATELLITE_ETF_NAMES = {"512890": "红利低波ETF（华泰柏瑞）"}
PORTFOLIO_ETF_NAMES = {**ETF_NAMES, **SATELLITE_ETF_NAMES}
PRE_IVOL_PORTFOLIO_STRATEGY_NAME = "weekly_core17_vol8_ma20a05_div512890_ma200_in0_out4_defense30"
CORE_STRATEGY_NAME = PRE_IVOL_CORE_STRATEGY_NAME + "_ivolm120_q75"
STRATEGY_NAME = PRE_IVOL_PORTFOLIO_STRATEGY_NAME + "_ivolm120_q75"
STRATEGY_CODENAME = "澄流120"
STRATEGY_CODE = "CL120"
IVOL_COMPARISON_SYMBOLS = ("510300", "159915", "159995", "159875", "159928",
                           "512660", "515070", "512880", "512800", "561360")
ADJUST = "hfq"  # Tushare 原始价 × 基金复权因子；不是可成交的真实价格。
