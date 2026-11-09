"""有国债/现金防守仓位的 ETF 轮动。所有权重只依赖信号日以前行情。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

import numpy as np
import pandas as pd

from .factor_store import FactorStore
from .factors import FactorSpec, compute_factor, validate_prices


BOND = "511260"


@dataclass(frozen=True)
class RiskConfig:
    """无参构造保留旧研究基线；实盘入口使用 current_default()。"""

    max_risky_positions: int = 3
    target_vol: float = 0.06
    max_risky_weight: float = 0.70
    bear_scale: float = 0.50
    short_momentum_days: int = 60
    long_momentum_days: int = 120
    momentum_skip_days: int = 5
    trend_days: int = 120
    max_single_risky_weight: float | None = None
    inverse_vol_exponent: float = 1.0
    rank_buffer_size: int | None = None
    mandatory_top_rank: int = 0
    ma_reversion_days: int = 20
    ma_reversion_strength: float = 0.0

    @classmethod
    def current_default(cls, *, target_vol: float = 0.08,
                        max_risky_weight: float = 0.70,
                        inverse_vol_exponent: float = 1.0) -> RiskConfig:
        """默认核心参数；澄流120的IVOL资格过滤由portfolio_strategy按启用日期传入。"""
        base = cls.buffered_momentum(target_vol=target_vol, max_risky_weight=max_risky_weight,
                                     inverse_vol_exponent=inverse_vol_exponent)
        return replace(base, ma_reversion_days=20, ma_reversion_strength=0.5)

    @classmethod
    def buffered_momentum(cls, *, target_vol: float = 0.07,
                          max_risky_weight: float = 0.70,
                          inverse_vol_exponent: float = 1.0) -> RiskConfig:
        """冻结2026-10-06的7%纯动量缓冲基线，旧研究显式调用。"""
        return cls(target_vol=target_vol, max_risky_weight=max_risky_weight,
                   short_momentum_days=20, long_momentum_days=40,
                   momentum_skip_days=0, trend_days=40,
                   max_single_risky_weight=0.30,
                   inverse_vol_exponent=inverse_vol_exponent,
                   rank_buffer_size=4, mandatory_top_rank=1)

    @classmethod
    def previous_default(cls, *, target_vol: float = 0.06,
                         max_risky_weight: float = 0.70,
                         inverse_vol_exponent: float = 1.0) -> RiskConfig:
        """保留切换前6%无缓冲方案，供历史研究复现与对照。"""
        return cls(target_vol=target_vol, max_risky_weight=max_risky_weight,
                   short_momentum_days=20, long_momentum_days=40,
                   momentum_skip_days=0, trend_days=40,
                   max_single_risky_weight=0.30,
                   inverse_vol_exponent=inverse_vol_exponent)

    def __post_init__(self) -> None:
        if self.max_risky_positions not in {1, 2, 3}:
            raise ValueError("max_risky_positions 必须在 1 到 3 之间")
        if not 0 < self.target_vol <= 0.30:
            raise ValueError("target_vol 必须在 (0, 0.30] 之间")
        if not 0 < self.max_risky_weight <= 1 or not 0 <= self.bear_scale <= 1:
            raise ValueError("风险权重参数必须在 [0, 1] 之间")
        if not (1 <= self.short_momentum_days < self.long_momentum_days
                and self.momentum_skip_days >= 0 and self.trend_days >= 2):
            raise ValueError("动量及趋势窗口配置无效")
        if (self.max_single_risky_weight is not None
                and not 0 < self.max_single_risky_weight <= 1):
            raise ValueError("单只风险 ETF 仓位上限必须在 (0, 1] 之间")
        if not 0 <= self.inverse_vol_exponent <= 1:
            raise ValueError("逆波动权重指数必须在 [0, 1] 之间")
        if self.ma_reversion_days < 2 or not 0 <= self.ma_reversion_strength <= 1:
            raise ValueError("均线反转窗口必须至少2日，强度必须在[0,1]之间")
        if not isinstance(self.mandatory_top_rank, int) or not 0 <= self.mandatory_top_rank <= self.max_risky_positions:
            raise ValueError("必选排名数量必须在0到风险持仓数量之间")
        if self.rank_buffer_size is None:
            if self.mandatory_top_rank:
                raise ValueError("必选排名配置需要开启排名缓冲")
        elif not isinstance(self.rank_buffer_size, int) or self.rank_buffer_size < self.max_risky_positions:
            raise ValueError("排名缓冲范围不能小于风险持仓数量")


def risk_score_components(short_momentum: pd.DataFrame, long_momentum: pd.DataFrame,
                          volatility: pd.DataFrame, config: RiskConfig,
                          ma_zscore: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """生产与评分解释共用：当日已启用风险ETF原评分离散程度缩放有界反转分。"""
    base = (0.50 * short_momentum + 0.50 * long_momentum) / volatility.clip(lower=0.05)
    adjustment = pd.DataFrame(0.0, index=base.index, columns=base.columns)
    if config.ma_reversion_strength:
        if ma_zscore is None or not ma_zscore.index.equals(base.index) or not ma_zscore.columns.equals(base.columns):
            raise ValueError("均线反转需要与动量评分严格对齐的ma_zscore")
        spread = base.drop(columns=BOND).std(axis=1, ddof=1).fillna(0)
        adjustment = ((-ma_zscore / 3).clip(-1, 1).fillna(0) * config.ma_reversion_strength).mul(spread, axis=0)
        adjustment[BOND] = 0.0
    return {"base_score": base, "ma_adjustment": adjustment, "score": base + adjustment}


def priority_buffer_selection(ranked: pd.Index, previous: list[str], *,
                              mandatory_top: int, buffer_size: int = 4,
                              slots: int = 3) -> list[str]:
    """输入仅含合格候选：先必选最强候选，再保留前4名旧仓，最后补足。"""
    if not 0 <= mandatory_top <= slots <= 3 or slots < 1 or buffer_size < slots:
        raise ValueError("强制入选数、持仓数或缓冲范围无效")
    selected = ranked[:mandatory_top].tolist()
    old = set(previous)
    retained = [s for s in ranked[:buffer_size] if s in old and s not in selected]
    selected += retained[:slots - len(selected)]
    selected += [s for s in ranked if s not in selected][:slots - len(selected)]
    return selected


def weekly_signal_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个交易周的最后一个开市日；周五休市时自然提前。"""
    period = index.to_period("W-FRI")
    return index[~period.duplicated(keep="last")]


def universe_activation_mask(prices: pd.DataFrame, activation_dates: dict[str, date | pd.Timestamp] | None = None) -> pd.DataFrame:
    """保留指标预热价格，仅在启用日前屏蔽评分/资格及截面缩放贡献。"""
    mask = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    for symbol, activation in (activation_dates or {}).items():
        if symbol not in prices or symbol == BOND:
            raise ValueError("标的启用日期包含未知ETF或防守国债")
        day = pd.Timestamp(activation)
        if pd.isna(day) or day.tzinfo is not None:
            raise ValueError("标的启用日期无效")
        mask[symbol] = prices.index >= day
    return mask


def risk_controlled_signals(prices: pd.DataFrame, config: RiskConfig,
                            store: FactorStore | None = None,
                            factor_frames: dict[FactorSpec, pd.DataFrame] | None = None,
                            *, score_overlay: pd.DataFrame | None = None,
                            initial_risky_selection: list[str] | None = None,
                            initial_selection_date: pd.Timestamp | None = None,
                            selection_resets: list[tuple[pd.Timestamp, list[str]]] | None = None,
                            universe_activation_dates: dict | None = None,
                            eligibility_mask: pd.DataFrame | None = None) -> pd.DataFrame:
    """周目标最多三只风险ETF加国债。旧选仓只参与排名筛选，不覆盖绝对资格。

    切换时可指定截至某历史信号日的旧目标；仅在该日之后第一次信号前初始化。
    之前的回测信号不读取这份未来切换状态。以后沿用各周已算出的目标选仓。
    """
    validate_prices(prices)
    if BOND not in prices:
        raise ValueError(f"价格矩阵缺少防守资产 {BOND}")
    if eligibility_mask is not None and (not eligibility_mask.index.equals(prices.index)
            or not eligibility_mask.columns.equals(prices.columns)):
        raise ValueError('资格掩码必须与价格矩阵严格对齐')

    def factor(spec: FactorSpec) -> pd.DataFrame:
        if factor_frames is not None and spec in factor_frames:
            return factor_frames[spec]
        return store.get(prices, spec) if store is not None else compute_factor(prices, spec)

    mom60 = factor(FactorSpec("momentum", config.short_momentum_days,
                              config.momentum_skip_days))
    mom120 = factor(FactorSpec("momentum", config.long_momentum_days,
                               config.momentum_skip_days))
    if universe_activation_dates:
        active = universe_activation_mask(prices, universe_activation_dates)
        mom60 = mom60.where(active)
        mom120 = mom120.where(active)
    trend120 = factor(FactorSpec("trend", config.trend_days))
    vol60 = factor(FactorSpec("volatility", 60))
    bond_mom60 = factor(FactorSpec("momentum", 60))
    market_trend = trend120["510300"]

    ma_zscore = factor(FactorSpec("ma_zscore", config.ma_reversion_days)) if config.ma_reversion_strength else None
    score = risk_score_components(mom60, mom120, vol60, config, ma_zscore)["score"]
    if score_overlay is not None:
        if not score_overlay.index.equals(prices.index) or not score_overlay.columns.equals(prices.columns):
            raise ValueError("score_overlay 必须与价格矩阵严格对齐")
        overlay_values = score_overlay.to_numpy(dtype=float)
        if np.isinf(overlay_values).any():
            raise ValueError("score_overlay 含无穷值")
        score = score + score_overlay.fillna(0.0)

    risk_columns = [symbol for symbol in prices.columns if symbol != BOND]
    if (initial_risky_selection is None) != (initial_selection_date is None):
        raise ValueError("初始化选仓和日期必须同时提供")
    resets = list(selection_resets or [])
    if initial_risky_selection is not None:
        resets.append((initial_selection_date, initial_risky_selection))
    normalized_resets = []
    for reset_date, reset_selection in resets:
        if (config.rank_buffer_size is None
                or len(reset_selection) > config.max_risky_positions
                or len(set(reset_selection)) != len(reset_selection)
                or not set(reset_selection).issubset(risk_columns)):
            raise ValueError("缓冲初始化选仓无效")
        reset_date = pd.Timestamp(reset_date)
        if pd.isna(reset_date) or reset_date.tzinfo is not None:
            raise ValueError("缓冲初始化日期无效")
        if prices.index[-1] > reset_date and reset_date not in prices.index:
            raise ValueError("价格历史必须覆盖缓冲初始化日期，不能从中途重建旧选仓")
        normalized_resets.append((reset_date, list(reset_selection)))
    normalized_resets.sort(key=lambda item: item[0])
    if len({d for d, _ in normalized_resets}) != len(normalized_resets):
        raise ValueError("缓冲初始化日期不能重复")
    rows: list[pd.Series] = []
    previous: list[str] = []
    next_reset = 0
    returns = prices.pct_change(fill_method=None)
    for day in weekly_signal_dates(prices.index):
        row = pd.Series(0.0, index=prices.columns, name=day)
        eligible = (mom120.loc[day, risk_columns] > 0) & (
            trend120.loc[day, risk_columns] > 0)
        if eligibility_mask is not None:
            eligible &= eligibility_mask.loc[day,risk_columns].fillna(False).astype(bool)
        candidate = score.loc[day, risk_columns].where(eligible & (vol60.loc[day, risk_columns] > 0))
        ranked = candidate.dropna().sort_index().sort_values(ascending=False, kind="stable").index
        if config.rank_buffer_size is not None:
            while next_reset < len(normalized_resets) and day > normalized_resets[next_reset][0]:
                previous = list(normalized_resets[next_reset][1])
                next_reset += 1
            chosen = priority_buffer_selection(ranked, previous,
                                               mandatory_top=config.mandatory_top_rank,
                                               buffer_size=config.rank_buffer_size,
                                               slots=config.max_risky_positions)
            # 统一按当前评分顺序计算协方差与权重，与研究实现一致。
            selected = [s for s in ranked if s in chosen]
        else:
            selected = ranked[:config.max_risky_positions].tolist()
        previous = selected
        if selected:
            inverse_vol = vol60.loc[day, selected].pow(-config.inverse_vol_exponent)
            within_risky = inverse_vol / inverse_vol.sum()
            history = returns.loc[:day, selected].tail(60).dropna()
            if len(history) >= 40:
                cov = history.cov().to_numpy(dtype=float) * 252
                estimated_vol = float(np.sqrt(max(within_risky.to_numpy() @ cov @ within_risky.to_numpy(), 0)))
            else:
                estimated_vol = float(np.sqrt(((within_risky * vol60.loc[day, selected]) ** 2).sum()))
            gross = min(config.max_risky_weight, config.target_vol / max(estimated_vol, 0.02))
            if pd.notna(market_trend.loc[day]) and market_trend.loc[day] <= 0:
                gross *= config.bear_scale
            row.loc[selected] = gross * within_risky
            if config.max_single_risky_weight is not None:
                cap = config.max_single_risky_weight
                protected = {symbol for symbol in selected if row[symbol] > cap}
                excess = sum(max(0.0, float(row[symbol] - cap)) for symbol in selected)
                for symbol in protected:
                    row[symbol] = cap
                while excess > 1e-12:
                    available = [symbol for symbol in selected if symbol not in protected
                                 and row[symbol] > 1e-9 and row[symbol] < cap - 1e-9]
                    if not available:
                        break
                    proportions = row.loc[available] / row.loc[available].sum()
                    moved = 0.0
                    for symbol in available:
                        amount = min(float(excess * proportions[symbol]),
                                     float(cap - row[symbol]))
                        row[symbol] += amount
                        moved += amount
                    if moved <= 1e-12:
                        break
                    excess -= moved
        if pd.notna(bond_mom60.at[day, BOND]) and bond_mom60.at[day, BOND] > 0:
            row[BOND] = max(0.0, 1.0 - row.sum())
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=prices.columns, dtype=float)
    signals = pd.DataFrame(rows)
    signals.index.name = "signal_date"
    return signals
