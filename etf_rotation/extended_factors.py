"""量能、波动率反转、IVOL代理及MACD研究因子；不自动接入实盘。"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .factors import validate_prices


@dataclass(frozen=True)
class ExtendedFactorSpec:
    name: str
    window: int = 5
    fast: int = 10
    slow: int = 60
    signal: int = 9
    model: str = 'market'

    def __post_init__(self):
        if self.name not in {'reversal','high_vol_reversal','abnormal_volume','volume_reversal',
                              'ivol','macd_trend','macd_histogram'}:
            raise ValueError('未知扩展因子')
        if any(not isinstance(x,int) or x<1 for x in (self.window,self.fast,self.slow,self.signal)):
            raise ValueError('窗口必须为正整数')
        if self.name in {'high_vol_reversal','macd_trend','macd_histogram'} and self.fast>=self.slow:
            raise ValueError('快窗口必须小于慢窗口')
        if self.name=='ivol' and (self.window<5 or self.model not in {'market','style'}):
            raise ValueError('IVOL窗口至少5日，模型只能是market或style')


def short_move_z(prices,days,vol_days=60):
    r=np.log(prices/prices.shift(1))
    # 短期冲击开始前的风险尺度，不用该冲击自身拟合分母。
    sigma=r.rolling(vol_days,min_periods=vol_days).std(ddof=1).shift(days)
    return np.log(prices/prices.shift(days))/(sigma*np.sqrt(days)).where(sigma>1e-12)


def abnormal_activity(activity,window,lookback):
    """相对自身历史量能；比较基准完全排除当前window日事件窗口。"""
    recent=activity.rolling(window,min_periods=window).mean()
    past=activity.shift(window).rolling(lookback,min_periods=lookback).mean()
    return (np.log(recent/past.where(past>0))/np.log(3)).clip(-1,1).where(recent>0)


def rolling_ivol(prices,days=60,model='market'):
    """过去N日含截距OLS的残差样本标准差×sqrt252；不是FF3或隐含波动。

    market: r_i=alpha+beta*r_510300+epsilon。
    style: 另加(r_159915-r_510300)成长风格代理，非真正SMB/HML。
    禁止将回归自变量对应ETF自身的零残差作为低IVOL优势。
    """
    if '510300' not in prices or (model=='style' and '159915' not in prices):
        raise ValueError('缺少IVOL模型基准')
    r=prices.pct_change(fill_method=None)
    m=r['510300'];mv=m.rolling(days,min_periods=days).var(ddof=1)
    vy=r.rolling(days,min_periods=days).var(ddof=1)
    cm=r.rolling(days,min_periods=days).cov(m)
    if model=='market':
        residual=vy-cm.pow(2).div(mv.where(mv>1e-16),axis=0)
        excluded=['510300']
    elif model=='style':
        g=r['159915']-m
        gv=g.rolling(days,min_periods=days).var(ddof=1)
        mg=m.rolling(days,min_periods=days).cov(g)
        cg=r.rolling(days,min_periods=days).cov(g)
        determinant=mv*gv-mg.pow(2)
        determinant=determinant.where((mv>1e-16)&(gv>1e-16)&(determinant>1e-8*mv*gv))
        bm=(cm.mul(gv,axis=0)-cg.mul(mg,axis=0)).div(determinant,axis=0)
        bg=(cg.mul(mv,axis=0)-cm.mul(mg,axis=0)).div(determinant,axis=0)
        residual=vy-bm*cm-bg*cg
        excluded=['510300','159915']
    else:raise ValueError('未知IVOL代理模型')
    out=np.sqrt(residual.clip(lower=0)*252)
    out.loc[:,excluded]=np.nan
    return out.where(prices.notna())


def compute_extended(prices:pd.DataFrame,spec:ExtendedFactorSpec,
                     activity:pd.DataFrame|None=None) -> pd.DataFrame:
    validate_prices(prices)
    if spec.name in {'reversal','high_vol_reversal'}:
        out=(-short_move_z(prices,spec.window,spec.slow)/3).clip(-1,1)
        if spec.name=='high_vol_reversal':
            r=np.log(prices/prices.shift(1))
            short=r.rolling(spec.fast,min_periods=spec.fast).std(ddof=1)
            long=r.rolling(spec.slow,min_periods=spec.slow).std(ddof=1).shift(spec.window)
            out*=((short/long.where(long>1e-12))-1).clip(0,1)
    elif spec.name in {'abnormal_volume','volume_reversal'}:
        if (activity is None or not activity.index.equals(prices.index)
                or not activity.columns.equals(prices.columns)):
            raise ValueError('量能必须与价格日期及代码严格对齐')
        a=activity.to_numpy(float)
        if np.isinf(a).any() or (a[np.isfinite(a)]<0).any():raise ValueError('量能含负值或无穷')
        if spec.name=='abnormal_volume':out=abnormal_activity(activity,spec.window,spec.slow)
        else:
            event=abnormal_activity(activity,spec.fast,spec.slow).clip(lower=0)
            out=(-short_move_z(prices,spec.window,60)/3).clip(-1,1)*event
    elif spec.name=='ivol':out=rolling_ivol(prices,spec.window,spec.model)
    else:
        fast=prices.ewm(span=spec.fast,min_periods=spec.fast,adjust=False,ignore_na=False).mean()
        slow=prices.ewm(span=spec.slow,min_periods=spec.slow,adjust=False,ignore_na=False).mean()
        difference=fast-slow
        dea=difference.ewm(span=spec.signal,min_periods=spec.signal,adjust=False,ignore_na=False).mean()
        value=difference if spec.name=='macd_trend' else 2*(difference-dea)
        sigma=prices.pct_change(fill_method=None).rolling(60,min_periods=60).std(ddof=1)
        scale_days=spec.slow if spec.name=='macd_trend' else spec.signal
        out=value/(prices*sigma*np.sqrt(scale_days)).where(sigma>1e-12)
        # 不把缺价日当有效EMA报价；缺价后仍要求连续完整预热。
        n=max(60,spec.slow+spec.signal)
        complete=prices.notna().rolling(n,min_periods=n).sum().eq(n)
        out=out.where(complete)
    return out.where(prices.notna()).replace([np.inf,-np.inf],np.nan)


def low_ivol_utility(ivol:pd.DataFrame,members:list[str]) -> pd.DataFrame:
    """本类有效资产至少3只，低IVOL得高分；未评分资产保持NaN、研究叠加时中性。"""
    group=ivol.loc[:,members];count=group.notna().sum(axis=1)
    rank=group.rank(axis=1,method='average')
    score=1-2*(rank-1).div((count-1).where(count>=3),axis=0)
    out=pd.DataFrame(np.nan,index=ivol.index,columns=ivol.columns)
    out.loc[:,members]=score
    return out
