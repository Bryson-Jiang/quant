"""冻结默认策略的过拟合审计；只输出研究结果，不修改实盘状态。

PyCharm/终端：python research_overfitting_audit.py --end 2026-10-20
可选早期数据：python research_overfitting_audit.py --download-history
所有试验与失败结果留在 output/overfitting_audit_20261021。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from statistics import NormalDist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import LEGACY_CORE_NAMES as ETF_NAMES
from etf_rotation.data import _client, load_symbol
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.mean_reversion import ReversionRule, defensive_funding_targets, reversion_states
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.legacy_research import load_portfolio_data, load_rotation_state
from etf_rotation.risk_strategy import BOND, RiskConfig, weekly_signal_dates

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/overfitting_audit_20261021"
START = pd.Timestamp("2022-01-01")
FOCUS_START = pd.Timestamp("2018-01-01")
SAT = "512890"
SEED = 20261021
BASE = RiskConfig.current_default()
RULE = ReversionRule(200, 0, .04, None)
NAMES = {**ETF_NAMES, SAT: "红利低波"}


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def fingerprint():
    files = ["live_signal.py", "run_backtest.py", "etf_rotation/risk_strategy.py",
             "etf_rotation/portfolio_strategy.py", "data/rotation_state.json", "output/live_signal.json"]
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files if (ROOT/p).exists()}


def download_history(end):
    """独立缓存；不覆盖实盘行情，不输出凭证或API异常正文。"""
    cache = ROOT / "data/audit_history"
    cache.mkdir(parents=True, exist_ok=True)
    status = []
    for symbol in [*ETF_NAMES, SAT, "515100", "515450", "588000", "510500", "512100"]:
        try:
            s = load_symbol(symbol, date(2010,1,1), end, cache)
            status.append({"symbol":symbol,"ok":True,"first_quote":str(s.index[0].date()),"rows":len(s)})
            print(f"history {symbol}: {s.index[0].date()}..{s.index[-1].date()}, {len(s)}", flush=True)
        except Exception as exc:
            status.append({"symbol":symbol,"ok":False,"error_type":type(exc).__name__})
            print(f"history {symbol}: {type(exc).__name__} (details withheld)", flush=True)
            # 网络不可用时不连续请求同一失败接口。
            if not any(r["ok"] for r in status):
                write_json(cache/"download_status.json",status)
                raise RuntimeError("早期数据接口不可达；保留现有缓存与失败类型") from None
    try:
        pro = _client()
        frames = [pro.fund_basic(market="E",status=s) for s in ("L","D")]
        pd.concat(frames,ignore_index=True).to_csv(cache/"fund_basic_current_snapshot.csv",index=False,encoding="utf-8-sig")
    except Exception as exc:
        status.append({"metadata_ok":False,"error_type":type(exc).__name__})
    write_json(cache/"download_status.json",status)


def cached(symbol, end, extended=False):
    folder = ROOT/"data/audit_history" if extended else ROOT/"data/cache"
    p = folder/f"tushare_{symbol}_hfq.csv"
    f = pd.read_csv(p,parse_dates=["date"]).set_index("date").sort_index()
    if f.index.has_duplicates or not np.allclose(f.raw_close*f.adj_factor,f.close,rtol=1e-12):
        raise ValueError(f"{symbol} 缓存校验失败")
    return f.loc[:pd.Timestamp(end),"close"].rename(symbol)


@dataclass
class Ledger:
    dates: pd.DatetimeIndex
    returns: np.ndarray
    gross: np.ndarray
    turnover: np.ndarray
    weights: np.ndarray
    skipped: int

    def series(self):
        return pd.Series(self.returns,index=self.dates)


def simulate(prices, signals, start=START, *, cost_bps=5., extra_delay=0):
    """生产收盘引擎的NumPy审计副本；运行前与生产逐日对账，不作免费日再平衡。"""
    columns = prices.columns
    targets = signals.reindex(columns=columns,fill_value=0.)
    positions = np.flatnonzero(prices.index>=pd.Timestamp(start))
    if len(positions)<2:
        raise ValueError("审计窗口过短")
    dates = prices.index[positions]
    raw = prices.to_numpy(float)
    val = prices.ffill(limit=3).to_numpy(float)
    executions = {}
    for day, row in targets.iterrows():
        j = int(prices.index.searchsorted(day,side="right"))+extra_delay
        if j<len(prices):
            executions[j] = row.to_numpy(float)
    held = np.zeros(len(columns))
    gross = np.zeros(len(dates)); turnover = gross.copy(); weights = np.zeros((len(dates),len(columns)))
    skipped = 0
    for k,j in enumerate(positions):
        if k and np.any(held>0):
            owned = held>0
            if np.isnan(val[j-1,owned]).any() or np.isnan(val[j,owned]).any():
                raise ValueError("持仓估值缺价超过3天")
            growth = val[j,owned]/val[j-1,owned]
            gross[k] = float(held[owned] @ (growth-1))
            held[owned] *= growth/(1+gross[k])
        if j in executions:
            desired = executions[j]
            if not np.isfinite(desired).all() or desired.min() < -1e-12 or desired.sum()>1+1e-8:
                raise ValueError("非法研究目标")
            changed = np.abs(desired-held)>1e-10
            if np.isnan(raw[j,changed]).any():
                skipped += 1
            else:
                turnover[k] = np.abs(desired-held).sum()
                held = desired.copy()
        weights[k] = held
    returns = (1+gross)*(1-turnover*cost_bps/10000)-1
    return Ledger(dates,returns,gross,turnover,weights,skipped)


def statistics(r):
    x = np.asarray(r,float)
    if len(x)<2:
        raise ValueError("统计样本过短")
    nav = np.cumprod(1+x)
    vol = x.std(ddof=1)*np.sqrt(252)
    dd = nav/np.maximum.accumulate(np.maximum(nav,1))-1
    return {"days":len(x),"total":float(nav[-1]-1),"annual":float(nav[-1]**(252/len(x))-1),
            "vol":float(vol),"sharpe":float(x.mean()*252/vol) if vol else 0.,
            "drawdown":float(dd.min()),"worst_day":float(x.min()),
            "sharpe_rf2_illustrative":float((x.mean()*252-.02)/vol) if vol else 0.}


class CoreAudit:
    """可调隐含常数的研究内核；默认目标必须与生产函数完全一致。"""
    def __init__(self, prices, state=None):
        self.p = prices
        self.columns = list(prices)
        self.r = prices.pct_change(fill_method=None).to_numpy(float)
        self.factors = {}
        self.covariances = {}
        self.bond = self.columns.index(BOND)
        self.market = self.columns.index("510300")
        self.counts = prices.notna().cumsum().to_numpy()
        self.resets = []
        if state:
            kw = state.signal_kwargs(prices)
            if "initial_selection_date" in kw:
                self.resets.append((kw["initial_selection_date"],kw["initial_risky_selection"]))
            self.resets.extend(kw.get("selection_resets",[]))

    def factor(self,name,window,skip=0):
        key = FactorSpec(name,window,skip)
        if key not in self.factors:
            self.factors[key] = compute_factor(self.p,key).to_numpy(float)
        return self.factors[key]

    def signals(self,cfg=BASE,*,excluded=(),min_age=0,weekday=4,vol_window=60,
                cov_window=60,bond_window=60,short_weight=.5,vol_floor=.05,z_clip=3.,
                decision_dates=None,score_override=None,eligibility_mask=None):
        p = self.p
        for frame in (score_override,eligibility_mask):
            if frame is not None and (not frame.index.equals(p.index) or not frame.columns.equals(p.columns)):
                raise ValueError('研究评分或资格掩码未与行情对齐')
        override=score_override.to_numpy(float) if score_override is not None else None
        admission=eligibility_mask.fillna(False).to_numpy(bool) if eligibility_mask is not None else None
        short = self.factor("momentum",cfg.short_momentum_days,cfg.momentum_skip_days)
        long = self.factor("momentum",cfg.long_momentum_days,cfg.momentum_skip_days)
        trend = self.factor("trend",cfg.trend_days)
        vol = self.factor("volatility",vol_window)
        bm = self.factor("momentum",bond_window)
        z = self.factor("ma_zscore",cfg.ma_reversion_days) if cfg.ma_reversion_strength else None
        excluded = set(excluded)
        allowed = np.array([s!=BOND and s not in excluded for s in self.columns])
        if decision_dates is None:
            # W-FRI与生产一致；其他周末端仅用于调仓星期敏感性。
            periods = p.index.to_period(["W-MON","W-TUE","W-WED","W-THU","W-FRI"][weekday])
            decision_dates = p.index[~periods.duplicated(keep="last")]
        indices = p.index.get_indexer(decision_dates)
        rows = []; previous = []; next_reset = 0
        for day,i in zip(decision_dates,indices):
            if i<0:
                raise ValueError("研究日历未对齐")
            row = np.zeros(len(self.columns))
            mask = allowed & (self.counts[i]>=min_age)
            score = (short_weight*short[i]+(1-short_weight)*long[i])/np.maximum(vol[i],vol_floor)
            if z is not None:
                values = score[mask & np.isfinite(score)]
                spread = np.std(values,ddof=1) if len(values)>1 else 0.
                score = score+cfg.ma_reversion_strength*spread*np.nan_to_num(np.clip(-z[i]/z_clip,-1,1))
            if override is not None:score=override[i]
            eligible = mask & (long[i]>0) & (trend[i]>0) & (vol[i]>0) & np.isfinite(score)
            if admission is not None:eligible &= admission[i]
            # 稳定的代码排序作为平分处理，与生产一致。
            ranked = sorted(np.flatnonzero(eligible),key=lambda j:(-score[j],self.columns[j]))
            if cfg.rank_buffer_size is not None:
                while next_reset<len(self.resets) and day>self.resets[next_reset][0]:
                    previous = [self.columns.index(s) for s in self.resets[next_reset][1] if s in self.columns][:cfg.max_risky_positions]
                    next_reset += 1
                chosen = ranked[:cfg.mandatory_top_rank]
                chosen += [j for j in ranked[:cfg.rank_buffer_size] if j in previous and j not in chosen][:cfg.max_risky_positions-len(chosen)]
                chosen += [j for j in ranked if j not in chosen][:cfg.max_risky_positions-len(chosen)]
                selected = [j for j in ranked if j in chosen]
            else:
                selected = ranked[:cfg.max_risky_positions]
            previous = selected
            if selected:
                inv = vol[i,selected]**(-cfg.inverse_vol_exponent)
                within = inv/inv.sum()
                key = (i,tuple(selected),cov_window)
                if key not in self.covariances:
                    h = self.r[max(0,i-cov_window+1):i+1,selected]
                    h = h[np.isfinite(h).all(axis=1)]
                    self.covariances[key] = np.atleast_2d(np.cov(h,rowvar=False,ddof=1))*252 if len(h)>=40 else None
                cov = self.covariances[key]
                sigma = np.sqrt(max(float(within @ cov @ within),0)) if cov is not None else np.sqrt(np.sum((within*vol[i,selected])**2))
                gross = min(cfg.max_risky_weight,cfg.target_vol/max(sigma,.02))
                if np.isfinite(trend[i,self.market]) and trend[i,self.market]<=0:
                    gross *= cfg.bear_scale
                w = gross*within
                if cfg.max_single_risky_weight is not None:
                    cap = cfg.max_single_risky_weight
                    protected = w>cap
                    excess = np.maximum(w-cap,0).sum()
                    w = np.minimum(w,cap)
                    while excess>1e-12:
                        available = ~protected & (w>1e-9) & (w<cap-1e-9)
                        if not available.any(): break
                        moved = np.minimum(excess*w[available]/w[available].sum(),cap-w[available])
                        w[available] += moved
                        if moved.sum()<=1e-12: break
                        excess -= moved.sum()
                row[selected] = w
            if np.isfinite(bm[i,self.bond]) and bm[i,self.bond]>0:
                row[self.bond] = max(0,1-row.sum())
            rows.append(row)
        return pd.DataFrame(rows,index=decision_dates,columns=self.columns).rename_axis("signal_date")


def satellite_targets(prices,core,history,rule=RULE,cap=.30,*,symbol=SAT,decision_dates=None):
    if cap==0:
        return core.reindex(columns=prices.columns,fill_value=0.)
    decisions = weekly_signal_dates(history.index) if decision_dates is None else decision_dates
    states = reversion_states(history,symbol,rule,decisions)
    return defensive_funding_targets(prices,core,states.reindex(prices.index),symbol,cap)


def global_risk_guard(prices,targets):
    """单一预声明风险对照：全组合预测波动8%/弱市4%，风险仓70%/35%。"""
    guarded=targets.copy()
    asset_r=prices.pct_change(fill_method=None)
    market_trend=compute_factor(prices,FactorSpec("trend",40))["510300"]
    for day,row in guarded.iterrows():
        weak=bool(market_trend.loc[day]<=0)
        risky=row.drop(BOND)
        ceiling=.35 if weak else .70
        if risky.sum()>ceiling:risky*=ceiling/risky.sum()
        revised=row.copy();revised.loc[risky.index]=risky
        revised[BOND]=max(0,1-float(risky.sum())) if row[BOND]>0 else 0.
        names=revised.index[revised>1e-12]
        h=asset_r.loc[:day,names].tail(60).dropna()
        if len(h)>=40:
            w=revised[names].to_numpy();sigma=np.sqrt(max(float(w@(h.cov().to_numpy()*252)@w),0))
            limit=.04 if weak else .08
            if sigma>limit:revised*=limit/sigma
        guarded.loc[day]=revised
    return guarded


def plan():
    # 单参数扫描；不按结果扩大网格。隐含常数也纳入研究副本。
    return {
        "target_vol":[.05,.06,.07,.08,.09,.10,.12],
        "max_risky_weight":[.5,.6,.7,.8,.9],"bear_scale":[0,.25,.5,.75,1.],
        "short_momentum_days":[10,15,20,25,30],"long_momentum_days":[30,40,50,60,80],
        "momentum_skip_days":[0,1,3,5,10],"trend_days":[20,30,40,60,80,120],
        "max_single_risky_weight":[.2,.25,.3,.35,.4,.5],"inverse_vol_exponent":[0,.25,.5,.75,1.],
        "max_risky_positions":[1,2,3],"rank_buffer_size":[3,4,5,6],"mandatory_top_rank":[0,1,2,3],
        "ma_reversion_days":[10,15,20,30,40],"ma_reversion_strength":[0,.25,.5,.75,1.],
        "vol_window":[40,60,90,120],"cov_window":[40,60,90,120],"bond_window":[20,40,60,90,120],
        "short_weight":[0,.25,.5,.75,1.],"vol_floor":[.025,.05,.075,.10],"z_clip":[2.,3.,4.],
        "sat_window":[120,150,180,200,220,250,300],"sat_entry":[-.05,-.03,-.02,-.01,0.],
        "sat_exit":[.02,.03,.04,.05,.06,.08],"sat_cap":[0,.1,.2,.3,.4],
        "sat_max_days":[60,120,252,None],"sat_confirmation":["none","bounce3","slope20"],
    }


def bootstrap_statistics(ledger,core_returns,reps=2000):
    rng = np.random.default_rng(SEED)
    x = ledger.returns; n = len(x); output = {}
    for block in (20,60,120):
        starts = rng.integers(0,n,(reps,math.ceil(n/block)))
        ix = ((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
        samples = x[ix]
        nav = np.cumprod(1+samples,axis=1)
        annual = nav[:,-1]**(252/n)-1
        sharpe = samples.mean(axis=1)/samples.std(axis=1,ddof=1)*np.sqrt(252)
        dd = (nav/np.maximum.accumulate(np.maximum(nav,1),axis=1)-1).min(axis=1)
        diff = samples-np.asarray(core_returns)[ix]
        output[str(block)] = {"annual_95":np.quantile(annual,[.025,.975]).tolist(),
             "sharpe_95":np.quantile(sharpe,[.025,.975]).tolist(),"drawdown_95":np.quantile(dd,[.025,.975]).tolist(),
             "share_resampled_dd_below_minus10":float(np.mean(dd<-.10)),
             "increment_arithmetic_95":np.quantile(diff.mean(axis=1)*252,[.025,.975]).tolist()}
    return output


def family_selection(matrix,default_index,core_returns,reps=2000):
    """仅校正本轮预声明联合家族；不声称覆盖此前所有人工搜索。"""
    x = np.asarray(matrix,float); n,m = x.shape
    diff = x-np.asarray(core_returns)[:,None]
    observed = diff.mean(axis=0)
    centered = diff-observed
    rng = np.random.default_rng(SEED+1); results = {}
    for block in (20,60,120):
        # 预计算循环块和，避免对每次抽样重复复制整个候选矩阵。
        doubled = np.concatenate([centered,centered[:block]],axis=0)
        cs = np.vstack([np.zeros((1,m)),np.cumsum(doubled,axis=0)])
        sums = cs[np.arange(n)+block]-cs[np.arange(n)]
        full,tail = divmod(n,block)
        tail_sums = cs[np.arange(n)+tail]-cs[np.arange(n)] if tail else None
        maxima = []
        for _ in range(math.ceil(reps/100)):
            starts = rng.integers(0,n,(100,full))
            means = sums[starts].sum(axis=1)
            if tail: means += tail_sums[rng.integers(0,n,100)]
            maxima.extend((means/n).max(axis=1))
        maxima = np.asarray(maxima[:reps])
        results[str(block)] = {"current_increment_family_adjusted_p":float((1+(maxima>=observed[default_index]).sum())/(reps+1)),
             "family_best_increment_p":float((1+(maxima>=observed.max()).sum())/(reps+1)),
             "current_arithmetic_increment":float(observed[default_index]*252)}
    # CSCV诊断；非前瞻测试，不将此值解读成当前策略无效的概率。
    splits = list(np.array_split(np.arange(n),10)); pbo = []
    for purge in (0,20):
        records = []
        for comb in itertools.combinations(range(10),5):
            mask = np.zeros(n,bool)
            for k in comb: mask[splits[k]]=True
            if purge:
                keep = np.ones(n,bool)
                borders = np.flatnonzero(mask[1:]!=mask[:-1])+1
                for b in borders: keep[max(0,b-purge):min(n,b+purge)] = False
            else: keep=np.ones(n,bool)
            a=x[mask & keep]; b=x[~mask & keep]
            train=a.mean(axis=0)/a.std(axis=0,ddof=1)*np.sqrt(252)
            test=b.mean(axis=0)/b.std(axis=0,ddof=1)*np.sqrt(252)
            winner=int(np.argmax(train))
            rank=(np.sum(test<test[winner])+.5*np.sum(test==test[winner]))/m
            records.append((train[winner],test[winner],rank))
        a=np.asarray(records)
        pbo.append({"boundary_purge_days":purge,"splits":len(a),"pbo_family":float(np.mean(a[:,2]<=.5)),
                    "median_selected_train_sharpe":float(np.median(a[:,0])),"median_selected_test_sharpe":float(np.median(a[:,1]))})
    corr=np.corrcoef(x,rowvar=False)
    effective=float(m*m/np.sum(corr*corr))
    return {"trials":m,"block_max_mean":results,"cscv":pbo,"return_correlation_participation_ratio":effective,
            "limitation":"局部候选家族、已观察历史；不包含此前选池、因子挖掘及人工试验，PBO不是当前策略失效概率。"}


def dsr_scenarios(x):
    """独立同分布零夏普空假设下的有效试验次数情景；不是已完成的全项目DSR。"""
    x=np.asarray(x);n=len(x);s=x.std(ddof=1); sr=x.mean()/s
    z=(x-x.mean())/s;skew=float(np.mean(z**3));kurt=float(np.mean(z**4))
    nd=NormalDist();den=math.sqrt(max(1-skew*sr+(kurt-1)*sr*sr/4,1e-12))
    output=[]
    for trials in (1,10,100,1000,10000):
        expected=0. if trials==1 else ((1-.5772156649)*nd.inv_cdf(1-1/trials)+.5772156649*nd.inv_cdf(1-1/(trials*math.e)))/math.sqrt(n-1)
        output.append({"assumed_independent_effective_trials":trials,"null_expected_max_annual_sr":expected*math.sqrt(252),
                       "conditional_dsr_style_probability":nd.cdf((sr-expected)*math.sqrt(n-1)/den)})
    hac=[];center=x-x.mean()
    for lag in (5,20,60):
        lrv=float(center@center/n)
        for k in range(1,lag+1): lrv+=2*(1-k/(lag+1))*float(center[k:]@center[:-k]/n)
        hac.append({"lag":lag,"hac_long_run_sharpe":float(x.mean()/math.sqrt(max(lrv,1e-16))*math.sqrt(252)),
                    "annual_mean_se":float(math.sqrt(max(lrv,0)/n)*252)})
    return {"skew":skew,"raw_kurtosis":kurt,"iid_effective_trial_scenarios":output,"hac":hac,
            "limitation":"试验次数是假设而非估计；IID公式忽略非平稳和序列依赖，仅作敏感性情景，不是策略盈利概率。"}


def additional_diagnostics(core,prices,history,engine,base_core,baseline,led):
    """因子预测性、组合风险与起始状态敏感性；不据此再选模型。"""
    risk=[s for s in core if s!=BOND]
    short=compute_factor(core,FactorSpec("momentum",20))
    long=compute_factor(core,FactorSpec("momentum",40))
    vol=compute_factor(core,FactorSpec("volatility",60))
    trend=compute_factor(core,FactorSpec("trend",40))
    z=compute_factor(core,FactorSpec("ma_zscore",20))
    raw=(.5*short+.5*long)/vol.clip(lower=.05)
    final=raw+(-z/3).clip(-1,1).fillna(0).mul(raw[risk].std(axis=1,ddof=1).fillna(0)*.5,axis=0)
    scores={"momentum20":short,"momentum40":long,"risk_adjusted_momentum":raw,
            "ma_reversion":-z,"final_score":final}
    icrows=[]; feature_rows=[]
    dates=base_core.index[base_core.index>=START]
    for horizon in (1,5,20):
        future=core.shift(-(horizon+1))/core.shift(-1)-1
        for day in dates:
            pos=core.index.get_loc(day)
            if pos+horizon+1>=len(core):continue
            end=core.index[pos+horizon+1]
            eligible=(long.loc[day,risk]>0)&(trend.loc[day,risk]>0)&(vol.loc[day,risk]>0)
            for scope in ("all_risk","eligible"):
                for name,frame in scores.items():
                    x=frame.loc[day,risk];y=future.loc[day,risk]
                    valid=x.notna()&y.notna()
                    if scope=="eligible":valid &=eligible
                    if valid.sum()<5:continue
                    a=x[valid].rank();b=y[valid].rank()
                    if a.std()==0 or b.std()==0:continue
                    ic=float(a.corr(b))
                    icrows.append({"signal":day,"label_end":end,"horizon":horizon,"scope":scope,
                                   "factor":name,"n_assets":int(valid.sum()),"rank_ic":ic})
            if horizon==5:
                # 每期只在已有报价的同一截面内标准化特征；未来收益仅用作标签。
                frame=pd.DataFrame({name:values.loc[day,risk] for name,values in scores.items()})
                frame["y"]=future.loc[day,risk]
                frame=frame.dropna()
                if len(frame)>=5:
                    frame["y"]-=frame.y.mean()
                    for name in scores:frame[name]=(frame[name]-frame[name].mean())/max(frame[name].std(ddof=1),1e-12)
                    frame["signal"]=day;frame["label_end"]=end;feature_rows.append(frame.reset_index(names="symbol"))
    ic=pd.DataFrame(icrows)
    ic.to_csv(OUT/"factor_ic_observations.csv",index=False)
    rng=np.random.default_rng(SEED+2);summary=[]
    for (horizon,scope,factor),group in ic.groupby(["horizon","scope","factor"]):
        g=group.sort_values("signal")
        # 20日标签重叠，另给每四个周信号取一个的均值；不假定全部周度IC独立。
        a=g.rank_ic.to_numpy();n=len(a);block=12
        ix=((rng.integers(0,n,(2000,math.ceil(n/block)))[:,:,None]+np.arange(block))%n).reshape(2000,-1)[:,:n]
        ci=np.quantile(a[ix].mean(axis=1),[.025,.975])
        summary.append({"horizon":horizon,"scope":scope,"factor":factor,"observations":n,
                        "mean_rank_ic":float(a.mean()),"median_rank_ic":float(np.median(a)),
                        "positive_share":float((a>0).mean()),"mean_ic_95_low":float(ci[0]),"mean_ic_95_high":float(ci[1]),
                        "thinned_every_four_mean":float(a[::4].mean()) if horizon==20 else float(a.mean())})
    pd.DataFrame(summary).to_csv(OUT/"factor_ic_summary.csv",index=False)
    observations=pd.concat(feature_rows,ignore_index=True)
    regressions=[]
    for year in range(2023,core.index[-1].year+1):
        train=observations[observations.label_end<pd.Timestamp(f"{year}-01-01")]
        test=observations[observations.signal.dt.year.eq(year)]
        if train.empty or test.empty:continue
        for name,features in (("final_score_only",["final_score"]),("momentum_and_reversion",["risk_adjusted_momentum","ma_reversion"])):
            a=np.column_stack([np.ones(len(train)),train[features]])
            b=np.column_stack([np.ones(len(test)),test[features]])
            beta=np.linalg.lstsq(a,train.y.to_numpy(),rcond=None)[0]
            error=test.y.to_numpy()-b@beta
            benchmark=test.y.to_numpy()  # 截面去均值标签的零预测。
            regressions.append({"test_year":year,"model":name,"train_labels_end":str(train.label_end.max().date()),
                                "train_asset_observations":len(train),"test_asset_observations":len(test),
                                "oos_r2_vs_zero_cross_section":float(1-np.sum(error**2)/np.sum(benchmark**2)),
                                "coefficients":json.dumps(beta.tolist())})
    pd.DataFrame(regressions).to_csv(OUT/"factor_forward_r2.csv",index=False)
    risks=[];rr=prices.pct_change(fill_method=None)
    for day,row in baseline.iterrows():
        if day<START:continue
        chosen=row.index[row>1e-12]
        h=rr.loc[:day,chosen].tail(60).dropna()
        if len(h)<40:continue
        w=row[chosen].to_numpy();cov=h.cov().to_numpy()*252
        predicted=np.sqrt(max(float(w@cov@w),0))
        risky=float(row.drop(BOND).sum())
        limit=.04 if trend.at[day,"510300"]<=0 else .08
        risks.append({"signal":day,"risky_target":risky,"portfolio_predicted_vol":predicted,
                      "original_core_vol_target":limit,"above_original_vol_target":bool(predicted>limit+1e-8),
                      "holdings":int((row>1e-12).sum())})
    pd.DataFrame(risks).to_csv(OUT/"risk_budget_diagnostics.csv",index=False)
    # 冷启动与继承历史持仓不同；每个起点从现金执行之后的首个可用周信号。
    cold=[]
    for year in range(2022,2027):
        start=pd.Timestamp(f"{year}-01-01")
        t=baseline.loc[baseline.index>=start]
        if len(t)<2:continue
        ll=simulate(prices,t,start)
        cold.append({"start":str(start.date()),"mode":"cold_cash_then_next_weekly_execution",**statistics(ll.returns)})
    pd.DataFrame(cold).to_csv(OUT/"cold_start.csv",index=False)
    calendar_years=(led.dates[-1]-led.dates[0]).days/365.25
    write_json(OUT/"annualization.json",{"calendar_years":calendar_years,"trading_years_252":len(led.dates)/252,
               "calendar_cagr":float(np.prod(1+led.returns)**(1/calendar_years)-1),
               "trading_252_cagr":statistics(led.returns)["annual"]})


def execute(end, reps=2000):
    OUT.mkdir(parents=True,exist_ok=True)
    before=fingerprint()
    state=load_rotation_state(ROOT/"data/rotation_state.json")
    core,history=load_portfolio_data(START.date(),end,ROOT/"data/cache",offline=True)
    prices=core.copy(); prices[SAT]=history[SAT].reindex(core.index)
    engine=CoreAudit(core,state)
    base_core=engine.signals()
    baseline=satellite_targets(prices,base_core,history)
    manifest={"as_of":str(end),"start":str(START.date()),"frozen_config":asdict(BASE),
              "satellite_rule":asdict(RULE),"satellite_cap":.3,"seed":SEED,"bootstrap_replicates":reps,
              "one_parameter_plan":plan(),"joint_family":"3 vol x 3 reversion strength x 3 satellite MA x 3 exit x 2 cap = 162",
              "random_joint_cases":64,"random_subpools":64,"matched_size_random_pools":64,
              "early_sensitivity_same_plan":True,"all_history_already_observed":True,
              "mode":"hypothetical_full_history","protected_before":before,
              "data_hash":hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
              "source_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    versions=OUT/"run_manifests";versions.mkdir(exist_ok=True)
    previous=OUT/"manifest.json"
    if previous.exists():
        old=json.loads(previous.read_text(encoding="utf-8"))
        (versions/f"{old['source_hash'][:12]}.json").write_bytes(previous.read_bytes())
    (versions/f"{manifest['source_hash'][:12]}.py").write_bytes(Path(__file__).read_bytes())
    write_json(OUT/"manifest.json",manifest)
    (OUT/"rotation_state_snapshot.json").write_bytes((ROOT/"data/rotation_state.json").read_bytes())
    checks=[]
    production=portfolio_signals(core,history,state,hypothetical_satellite_history=True)
    pd.testing.assert_frame_equal(base_core,production.core_signals,check_exact=False,rtol=1e-10,atol=1e-12)
    pd.testing.assert_frame_equal(baseline,production.signals,check_exact=False,rtol=1e-10,atol=1e-12)
    canonical=run_backtest(prices,baseline,START.date(),end)
    led=simulate(prices,baseline)
    np.testing.assert_allclose(led.returns,canonical.equity.daily_return,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(led.weights,canonical.weights.drop(columns="CASH"),rtol=1e-10,atol=1e-12)
    checks.append({"check":"production_signals_and_daily_ledger","passed":True,
                   "max_return_error":float(np.max(np.abs(led.returns-canonical.equity.daily_return.to_numpy())))})
    core_led=simulate(prices,base_core)
    rows=[]; returns={}; targets={}; ledgers={}
    state_cache={}

    def target_for(c,rule=RULE,cap=.3,symbol=SAT,h=history,p=prices):
        if cap==0: return c.reindex(columns=p.columns,fill_value=0.)
        key=(symbol,rule)
        if key not in state_cache:
            state_cache[key]=reversion_states(h,symbol,rule,weekly_signal_dates(h.index))
        return defensive_funding_targets(p,c,state_cache[key].reindex(p.index),symbol,cap)

    def add(name,group,target,p=prices,*,start=START,extra=None):
        result=simulate(p,target,start)
        record={"case":name,"group":group,**statistics(result.returns),
                "turnover_per_year":float(result.turnover.sum()*252/len(result.dates)),"skipped":result.skipped}
        if extra: record.update(extra)
        rows.append(record)
        returns[name]=result.series()
        if group in {"baseline","joint","ablation","schedule","benchmark"}:
            targets[name]=target
        if group in {"baseline","benchmark"}: ledgers[name]=result
        if len(rows)%25==0: print(f"audit cases {len(rows)}; last={name}",flush=True)
        return result

    add("default","baseline",baseline)
    add("core_only","baseline",base_core)
    config_fields=set(asdict(BASE))
    for param,values in plan().items():
        for value in values:
            cfg=BASE;kw={};rule=RULE;cap=.3
            if param in config_fields: cfg=replace(BASE,**{param:value})
            elif param.startswith("sat_"):
                key={"sat_window":"window","sat_entry":"entry","sat_exit":"exit",
                     "sat_max_days":"max_days","sat_confirmation":"confirmation"}.get(param)
                if key: rule=replace(RULE,**{key:value})
                else: cap=value
            else: kw[param]=value
            c=base_core if param.startswith("sat_") else engine.signals(cfg,**kw)
            name=f"oat__{param}__{value}"
            add(name,"oat",target_for(c,rule,cap),extra={"parameter":param,"value":str(value)})

    family=[]
    for vol,strength,ma,exit_level,cap in itertools.product((.06,.08,.10),(0.,.5,1.),(150,200,250),(.03,.04,.05),(.2,.3)):
        cfg=replace(BASE,target_vol=vol,ma_reversion_strength=strength)
        c=engine.signals(cfg)
        rule=replace(RULE,window=ma,exit=exit_level)
        name=f"joint__v{vol:.2f}_s{strength:.1f}_ma{ma}_x{exit_level:.2f}_c{cap:.1f}"
        target=target_for(c,rule,cap)
        add(name,"joint",target,extra={"joint_vol":vol,"joint_strength":strength,"joint_ma":ma,"joint_exit":exit_level,"joint_cap":cap})
        family.append(name)
    current_joint="joint__v0.08_s0.5_ma200_x0.04_c0.3"
    np.testing.assert_allclose(returns[current_joint],returns["default"],rtol=1e-10,atol=1e-12)

    rng=np.random.default_rng(SEED)
    for i in range(64):
        cfg=replace(BASE,target_vol=float(rng.choice([.07,.08,.09])),short_momentum_days=int(rng.choice([15,20,25])),
                    long_momentum_days=int(rng.choice([35,40,50])),trend_days=int(rng.choice([30,40,50])),
                    ma_reversion_days=int(rng.choice([15,20,30])),ma_reversion_strength=float(rng.choice([.25,.5,.75])),
                    rank_buffer_size=int(rng.choice([3,4,5])),max_single_risky_weight=float(rng.choice([.25,.3,.35])),
                    bear_scale=float(rng.choice([.4,.5,.6])))
        rule=replace(RULE,window=int(rng.choice([180,200,220])),exit=float(rng.choice([.03,.04,.05])))
        cap=float(rng.choice([.2,.3,.4]))
        add(f"jitter_{i:02}","jitter",target_for(engine.signals(cfg),rule,cap),
            extra={"settings":json.dumps(asdict(cfg)|{"satellite":asdict(rule),"sat_cap":cap},sort_keys=True)})

    risk=[s for s in core if s!=BOND]
    exclusions={f"without_{s}":[s] for s in risk}
    exclusions.update({"without_gold_nasdaq":["518880","159941"],
        "without_commodities":["518880","159980","159985","561360"],
        "without_overseas":["159941","513520"],"without_tech":["159995","515070","159915"],
        "without_bank_and_gold":["512800","518880"]})
    for name,excluded in exclusions.items():
        add(name,"ablation",target_for(engine.signals(excluded=excluded)),extra={"excluded":",".join(excluded)})
    cash_core=base_core.copy();cash_core[BOND]=0.
    add("bond_to_cash","ablation",target_for(cash_core))
    add("no_core_ma_reversion","ablation",target_for(engine.signals(replace(BASE,ma_reversion_strength=0))))
    add("no_buffer","ablation",target_for(engine.signals(replace(BASE,rank_buffer_size=None,mandatory_top_rank=0))))
    add("no_satellite","ablation",base_core)
    guarded=global_risk_guard(prices,baseline)
    add("global_vol8_and_gross70_guard","risk_guard",guarded)
    for i in range(64):
        excluded=sorted(rng.choice(risk,4,replace=False).tolist())
        add(f"subpool_{i:02}","subpool",target_for(engine.signals(excluded=excluded)),extra={"excluded":",".join(excluded)})

    # 只按观测数量固定2022年初可用子池；缓存左截断使其只能作为保守压力测试。
    before_start=core.loc[:"2021-12-31"].notna().sum()
    excluded=before_start.index[before_start<252].difference([BOND]).tolist()
    add("fixed_pre2022_252_quotes","universe",target_for(engine.signals(excluded=excluded)),extra={"excluded":",".join(excluded)})
    # 科创50、中证500、中证1000等已有缓存的扩池，对齐截止日期，绝不填补缺失的最新价格。
    extras=["588000","510500","512100","159992","513180","513500"]
    series={s:cached(s,end) for s in extras}
    common_end=min(s.index[-1] for s in series.values())
    broad=core.loc[:common_end].copy()
    for symbol,s in series.items(): broad[symbol]=s.reindex(broad.index)
    broad_engine=CoreAudit(broad,state)
    broad_prices=broad.copy();broad_prices[SAT]=history[SAT].reindex(broad.index)
    broad_core=broad_engine.signals()
    add("expanded_pool_common_end","universe",satellite_targets(broad_prices,broad_core,history.loc[:common_end]),broad_prices,
        extra={"comparison_end":str(common_end.date()),"added":",".join(extras)})
    cp=prices.loc[:common_end]
    add("default_common_end","universe",baseline.loc[:common_end],cp,extra={"comparison_end":str(common_end.date())})
    # 等数量换池：保持15只风险候选，从已有21只风险ETF中随机选15只。
    # 原来的删除4只实验同时改变池大小，本实验用于减少这个混淆。
    broad_risk=[s for s in broad if s!=BOND]
    for i in range(64):
        excluded=sorted(rng.choice(broad_risk,len(broad_risk)-len(risk),replace=False).tolist())
        bc=broad_engine.signals(excluded=excluded)
        t=satellite_targets(broad_prices,bc,history.loc[:common_end])
        add(f"matched_size_pool_{i:02}","matched_pool",t,broad_prices,
            extra={"excluded":",".join(excluded),"comparison_end":str(common_end.date()),"risk_candidates":len(risk)})

    for symbol in ("515100","515450"):
        h=cached(symbol,end).reindex(history.index).to_frame(symbol)
        pp=core.copy();pp[symbol]=h[symbol].reindex(core.index)
        t=satellite_targets(pp,base_core,h,symbol=symbol)
        add(f"satellite_swap_{symbol}","replication",t,pp)
    for weekday in range(5):
        c=engine.signals(weekday=weekday)
        period=history.index.to_period(["W-MON","W-TUE","W-WED","W-THU","W-FRI"][weekday])
        decisions=history.index[~period.duplicated(keep="last")]
        t=satellite_targets(prices,c,history,decision_dates=decisions)
        add(f"week_ending_{weekday}","schedule",t)
    for delay in (1,2,4):
        ll=simulate(prices,baseline,extra_delay=delay)
        rows.append({"case":f"execution_extra_delay_{delay}","group":"execution",**statistics(ll.returns),"skipped":ll.skipped})
        returns[f"execution_extra_delay_{delay}"]=ll.series()
    costs=[]
    for cost in (0,5,10,20,30,50):
        costs.append({"single_side_bps":cost,**statistics((1+led.gross)*(1-led.turnover*cost/10000)-1)})
    pd.DataFrame(costs).to_csv(OUT/"costs.csv",index=False)
    stress=run_backtest(prices,baseline,START.date(),end,fee_bps=0,slippage_bps=20)
    np.testing.assert_allclose(stress.equity.daily_return,(1+led.gross)*(1-led.turnover*.002)-1,rtol=1e-10,atol=1e-12)
    checks.append({"check":"production_cost20_reconciliation","passed":True})

    # 基准规则预先固定，未用全期实现波动事后缩放。
    fixed={"510300":.2,"159941":.2,"518880":.2,BOND:.4}
    fixed_targets=pd.DataFrame(0.,index=base_core.index,columns=prices.columns)
    for s,w in fixed.items(): fixed_targets[s]=w
    add("fixed_multiasset_weekly","benchmark",fixed_targets)
    vt=fixed_targets.copy(); rr=prices[list(fixed)].pct_change(fill_method=None)
    fw=np.array(list(fixed.values()))
    for day in vt.index:
        h=rr.loc[:day].tail(60).dropna()
        scale=0.
        if len(h)>=40:
            sigma=np.sqrt(max(float(fw @ (h.cov().to_numpy()*252) @ fw),0))
            scale=min(1.,.08/max(sigma,.02))
        vt.loc[day]*=scale
    add("fixed_multiasset_vol8","benchmark",vt)
    bh=pd.DataFrame(0.,index=base_core.index,columns=prices.columns);bh[BOND]=1.
    eligible=bh.index[bh.index<prices.index[prices.index>=START][0]]
    # 首次回测交易日建仓、之后不再平衡。
    one=bh.loc[[eligible[-1]]] if len(eligible) else bh.iloc[[0]]
    add("treasury_buy_hold","benchmark",one)
    plain=engine.signals(replace(BASE,ma_reversion_strength=0.,rank_buffer_size=None,mandatory_top_rank=0))
    add("simple_momentum_core","benchmark",plain)

    # 年度滚动选参：保留每个候选的因果影子状态，训练仅截止上一年末；切换照常扣费。
    choices=[]; rolling=baseline.copy()
    for year in range(2023,end.year+1):
        train=pd.DataFrame({n:returns[n].loc[:f"{year-1}-12-31"] for n in family})
        score=train.mean()/train.std(ddof=1)*np.sqrt(252)
        winner=score.idxmax()
        dates=rolling.index[rolling.index.year==year]
        rolling.loc[dates]=targets[winner].reindex(dates)
        choices.append({"execution_year":year,"training_end":str(train.index[-1].date()),"training_days":len(train),
                        "selected":winner,"train_sharpe":float(score[winner])})
    add("annual_walk_forward_selection","walk_forward",rolling)
    pd.DataFrame(choices).to_csv(OUT/"walk_forward_choices.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"all_cases.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(returns).to_csv(OUT/"daily_returns.csv",encoding="utf-8-sig",index_label="date")

    # 时间截断和未来报价扰动；截断到完整周信号，避免末尾半周日历差异。
    candidates=core.index[(core.index.weekday==4)&(core.index>="2022-01-01")]
    for cutoff in candidates[np.linspace(5,len(candidates)-3,6,dtype=int)]:
        pp=core.loc[:cutoff];hh=history.loc[:cutoff]
        truncated=portfolio_signals(pp,hh,state,hypothetical_satellite_history=True)
        changed=core.copy(); changed.loc[changed.index>cutoff]*=np.linspace(.2,3.,len(changed.loc[changed.index>cutoff]))[:,None]
        changed_h=history.copy();changed_h.loc[changed_h.index>cutoff]*=2.7
        disturbed=portfolio_signals(changed,changed_h,state,hypothetical_satellite_history=True)
        pd.testing.assert_frame_equal(production.signals.loc[:cutoff],truncated.signals,check_exact=False,atol=1e-12,rtol=1e-10)
        pd.testing.assert_frame_equal(production.signals.loc[:cutoff],disturbed.signals.loc[:cutoff],check_exact=False,atol=1e-12,rtol=1e-10)
        checks.append({"check":"prefix_and_future_perturbation","cutoff":str(cutoff.date()),"passed":True})
    for cfg in (replace(BASE,target_vol=.06,ma_reversion_strength=0),replace(BASE,short_momentum_days=15,long_momentum_days=60),
                replace(BASE,max_risky_positions=1),replace(BASE,max_single_risky_weight=.2)):
        from etf_rotation.risk_strategy import risk_controlled_signals
        kw=state.signal_kwargs(core)
        if "initial_risky_selection" in kw:
            kw["initial_risky_selection"]=kw["initial_risky_selection"][:cfg.max_risky_positions]
        if "selection_resets" in kw:
            kw["selection_resets"]=[(d,s[:cfg.max_risky_positions]) for d,s in kw["selection_resets"]]
        actual=risk_controlled_signals(core,cfg,**kw)
        trial=engine.signals(cfg)
        pd.testing.assert_frame_equal(actual,trial,check_exact=False,atol=1e-12,rtol=1e-10)
        t=target_for(trial)
        slow=run_backtest(prices,t,START.date(),end)
        fast=simulate(prices,t)
        np.testing.assert_allclose(slow.equity.daily_return,fast.returns,atol=1e-12,rtol=1e-10)
        checks.append({"check":"independent_engine_variant_reconciliation","config":asdict(cfg),"passed":True})
    scaled=core.mul(pd.Series(np.linspace(.3,4.,len(core.columns)),index=core.columns),axis=1)
    sb=portfolio_signals(scaled,history*3.7,state,hypothetical_satellite_history=True)
    pd.testing.assert_frame_equal(production.signals,sb.signals,check_exact=False,atol=1e-10,rtol=1e-9)
    checks.append({"check":"per_asset_price_unit_invariance","passed":True})

    table=pd.DataFrame(rows)
    table.to_csv(OUT/"all_cases.csv",index=False,encoding="utf-8-sig")
    daily=pd.DataFrame(returns)
    daily.to_csv(OUT/"daily_returns.csv",encoding="utf-8-sig",index_label="date")
    baseline.to_csv(OUT/"default_targets.csv",encoding="utf-8-sig")
    canonical.equity.to_csv(OUT/"default_equity.csv")
    canonical.weights.to_csv(OUT/"default_weights.csv")
    yearly=[]
    selected=["default","core_only","annual_walk_forward_selection","fixed_multiasset_weekly","fixed_multiasset_vol8","simple_momentum_core"]
    for name in selected:
        for year,values in daily[name].dropna().groupby(daily.index.year):
            if len(values)>1: yearly.append({"case":name,"year":int(year),**statistics(values)})
    pd.DataFrame(yearly).to_csv(OUT/"yearly.csv",index=False)
    segments=[]
    for name in selected:
        for label,left,right in (("2022_2024","2022","2024-12-31"),("2025_2026","2025",str(end)),
                                 ("2023_onward","2023",str(end)),("2026_H1","2026-01-01","2026-06-30")):
            x=daily.loc[left:right,name].dropna()
            segments.append({"case":name,"segment":label,**statistics(x)})
    pd.DataFrame(segments).to_csv(OUT/"segments.csv",index=False)
    windows=[]
    for window in (126,252,504):
        s=daily["default"].dropna()
        vals=[statistics(s.iloc[i-window:i]) for i in range(window,len(s)+1,21)]
        windows.append({"window":window,"overlapping_windows":len(vals),
                        "annual_min":min(v["annual"] for v in vals),"annual_median":float(np.median([v["annual"] for v in vals])),
                        "annual_max":max(v["annual"] for v in vals),"sharpe_min":min(v["sharpe"] for v in vals),
                        "sharpe_median":float(np.median([v["sharpe"] for v in vals])),"drawdown_worst":min(v["drawdown"] for v in vals),
                        "share_negative_windows":float(np.mean([v["total"]<0 for v in vals]))})
    pd.DataFrame(windows).to_csv(OUT/"rolling_windows.csv",index=False)

    print("bootstrap and multiple-selection diagnostics",flush=True)
    boot=bootstrap_statistics(led,core_led.returns,reps)
    write_json(OUT/"bootstrap.json",boot)
    selection=family_selection(daily[family].to_numpy(),family.index(current_joint),core_led.returns,reps)
    write_json(OUT/"selection_bias.json",selection)
    broad_names=[name for name in daily if name not in {"annual_walk_forward_selection","fixed_multiasset_weekly",
                    "fixed_multiasset_vol8","treasury_buy_hold","simple_momentum_core"}]
    broader=daily[broad_names].dropna()
    wide=family_selection(broader.to_numpy(),broad_names.index("default"),returns["core_only"].reindex(broader.index),reps)
    wide.update({"common_start":str(broader.index[0].date()),"common_end":str(broader.index[-1].date()),
                 "scope":"本轮所有同截止日参数、换池、执行时点及简化规则；不含被动基准与年度选参"})
    write_json(OUT/"selection_bias_broad.json",wide)
    write_json(OUT/"sharpe_inference.json",dsr_scenarios(led.returns))
    # 逐ETF财富贡献；使用前日实际权重，费用单列，精确对账总财富变化。
    asset_returns=prices.ffill(limit=3).pct_change(fill_method=None).reindex(led.dates).to_numpy(float)
    previous_weights=np.vstack([np.zeros((1,len(prices.columns))),led.weights[:-1]])
    contribution=np.nan_to_num(asset_returns)*previous_weights
    prevnav=np.r_[1.,np.cumprod(1+led.returns)[:-1]]
    wealth=(contribution*prevnav[:,None]).sum(axis=0)
    fees=float(np.sum(prevnav*(1+led.gross)*led.turnover*.0005))
    np.testing.assert_allclose(wealth.sum()-fees,np.prod(1+led.returns)-1,atol=1e-10)
    holdings=[]
    for j,symbol in enumerate(prices.columns):
        holdings.append({"symbol":symbol,"name":NAMES.get(symbol,symbol),"days_held":int((led.weights[:,j]>1e-10).sum()),
                         "mean_weight":float(led.weights[:,j].mean()),"max_actual_weight":float(led.weights[:,j].max()),
                         "wealth_contribution_initial_capital":float(wealth[j])})
    pd.DataFrame(holdings).to_csv(OUT/"asset_attribution.csv",index=False,encoding="utf-8-sig")
    write_json(OUT/"attribution_reconciliation.json",{"asset_pnl":float(wealth.sum()),"cost":fees,"net_total":float(wealth.sum()-fees)})
    coverage=[];outliers=[]
    for s in [*ETF_NAMES,SAT]:
        f=pd.read_csv(ROOT/f"data/cache/tushare_{s}_hfq.csv",parse_dates=["date"]).set_index("date").loc[:pd.Timestamp(end)]
        rr=f.close.pct_change(fill_method=None)
        coverage.append({"symbol":s,"name":NAMES[s],"first_cached_quote":str(f.index[0].date()),"last_quote":str(f.index[-1].date()),
                         "rows":len(f),"quotes_before_2022":int((f.index<START).sum()),"maximum_abs_adjusted_return":float(rr.abs().max())})
        for day in rr.abs().nlargest(3).index:
            outliers.append({"symbol":s,"date":str(day.date()),"adjusted_return":float(rr[day]),
                             "raw_return":float(f.raw_close.pct_change(fill_method=None)[day]),
                             "factor_ratio":float((f.adj_factor/f.adj_factor.shift(1)).loc[day])})
    pd.DataFrame(coverage).to_csv(OUT/"data_coverage.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(outliers).to_csv(OUT/"largest_quote_moves.csv",index=False)

    additional_diagnostics(core,prices,history,engine,base_core,baseline,led)

    focus_2018_audit(end,checks)
    pd.DataFrame(rows).to_csv(OUT/"all_cases.csv",index=False,encoding="utf-8-sig")
    after=fingerprint()
    if before!=after: raise AssertionError("实盘保护文件意外改变")
    checks.append({"check":"protected_live_files_unchanged","passed":True})
    write_json(OUT/"checks.json",checks)
    summary={"default":statistics(led.returns),"core":statistics(core_led.returns),"total_cases":len(rows),
             "checks":len(checks),"protected_unchanged":True,"bootstrap":boot,"selection":selection}
    write_json(OUT/"summary.json",summary)
    plot_report(pd.DataFrame(rows),daily)
    build_report(end)
    print(json.dumps({"cases":len(rows),"checks":len(checks),"default":summary["default"]},ensure_ascii=False),flush=True)


def focus_2018_audit(end, checks=None):
    """2018为固定主评估起点；此前行情仅预热，不计算其收益用于选参。"""
    OUT.mkdir(parents=True,exist_ok=True)
    before=fingerprint()
    series={s:cached(s,end,True) for s in [*ETF_NAMES,SAT]}
    prices=pd.DataFrame(series,index=series["510300"].index)
    cp=prices.drop(columns=SAT);history=prices[[SAT]]
    state=load_rotation_state(ROOT/"data/rotation_state.json")
    engine=CoreAudit(cp,state);core=engine.signals()
    target=satellite_targets(prices,core,history)
    production=portfolio_signals(cp,history,state,hypothetical_satellite_history=True)
    pd.testing.assert_frame_equal(target,production.signals,check_exact=False,atol=1e-12,rtol=1e-10)
    slow=run_backtest(prices,target,FOCUS_START.date(),end)
    main=simulate(prices,target,FOCUS_START)
    np.testing.assert_allclose(main.returns,slow.equity.daily_return,atol=1e-12,rtol=1e-10)
    variants={"default":target,"core_only":core,"global_vol8_and_gross70_guard":global_risk_guard(prices,target)}
    for age in (120,252):
        c=engine.signals(min_age=age)
        dates=weekly_signal_dates(history.index)
        dates=dates[history[SAT].notna().cumsum().reindex(dates)>=age]
        variants[f"maturity_{age}"]=satellite_targets(prices,c,history,decision_dates=dates)
    fixed=pd.DataFrame(0.,index=core.index,columns=prices.columns)
    for symbol,weight in {"510300":.2,"159941":.2,"518880":.2,BOND:.4}.items():fixed[symbol]=weight
    variants["fixed_multiasset_weekly"]=fixed
    bond=pd.DataFrame(0.,index=core.index,columns=prices.columns);bond[BOND]=1
    first=prices.index[prices.index>=FOCUS_START][0]
    variants["treasury_buy_hold"]=bond.loc[[bond.index[bond.index<first][-1]]]
    records=[];daily={};segments=[]
    for name,t in variants.items():
        ll=simulate(prices,t,FOCUS_START)
        records.append({"case":name,**statistics(ll.returns)})
        daily[name]=ll.series()
        for label,left,right in (("2018_2021","2018","2021-12-31"),("2022_2024","2022","2024-12-31"),("2025_onward","2025",str(end))):
            x=ll.series().loc[left:right]
            if len(x)>1:segments.append({"case":name,"period":label,**statistics(x)})
    pd.DataFrame(records).to_csv(OUT/"focus_2018_cases.csv",index=False)
    pd.DataFrame(daily).to_csv(OUT/"focus_2018_daily_returns.csv",index_label="date")
    pd.DataFrame(segments).to_csv(OUT/"focus_2018_segments.csv",index=False)
    availability=[]
    for year,part in cp.loc[FOCUS_START:].groupby(cp.loc[FOCUS_START:].index.year):
        counts=cp.loc[:part.index[-1]].notna().sum()
        availability.append({"year":int(year),"core_risk_with_60_quotes":int((counts.drop(BOND)>=60).sum())})
    pd.DataFrame(availability).to_csv(OUT/"focus_2018_asset_availability.csv",index=False)
    sensitivity=[];cfg_fields=set(asdict(BASE))
    for param,values in plan().items():
        for value in values:
            cfg=BASE;kw={};rule=RULE;cap=.3
            if param in cfg_fields:cfg=replace(BASE,**{param:value})
            elif param.startswith("sat_"):
                key={"sat_window":"window","sat_entry":"entry","sat_exit":"exit",
                     "sat_max_days":"max_days","sat_confirmation":"confirmation"}.get(param)
                if key:rule=replace(RULE,**{key:value})
                else:cap=value
            else:kw[param]=value
            cc=core if param.startswith("sat_") else engine.signals(cfg,**kw)
            trial=satellite_targets(prices,cc,history,rule,cap)
            ll=simulate(prices,trial,FOCUS_START)
            early=ll.series().loc[:"2021-12-31"];recent=ll.series().loc["2022":]
            sensitivity.append({"parameter":param,"value":str(value),
                 **statistics(ll.returns),**{f"early_{k}":v for k,v in statistics(early).items()},
                 **{f"recent_{k}":v for k,v in statistics(recent).items()}})
        print(f"2018 focus sensitivity {param}",flush=True)
    pd.DataFrame(sensitivity).to_csv(OUT/"focus_2018_sensitivity.csv",index=False)
    if fingerprint()!=before:raise AssertionError("实盘保护文件意外改变")
    write_json(OUT/"focus_2018_manifest.json",{"evaluation_start":"2018-01-01","actual_first_day":str(main.dates[0].date()),
       "end":str(end),"decision":"用户指定2018年后为主要研究范围；此前只预热指标/状态，旧研究保留附录，不作主评分依据",
       "first_evaluation_portfolio":"cash; next available scheduled close execution",
       "no_new_parameter_selection":True,"cases":len(records),"parameter_points":len(sensitivity),
       "source_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"protected_unchanged":True,
       "production_reconciliation_passed":True,"older_selection_bias_not_eliminated":True})
    if checks is not None:checks.append({"check":"2018_focus_production_reconciliation","passed":True})


def extended_audit(end,rows,checks):
    folder=ROOT/"data/audit_history"
    required=[*ETF_NAMES,SAT]
    if not all((folder/f"tushare_{s}_hfq.csv").exists() for s in required):
        write_json(OUT/"extended_status.json",{"available":False,"reason":"独立早期缓存不完整；不使用指数回填ETF上市前行情"})
        return
    series={s:cached(s,end,True) for s in required}
    market=series["510300"]
    p=pd.DataFrame(series,index=market.index)
    cp=p.drop(columns=SAT)
    h=p[[SAT]]
    e=CoreAudit(cp)
    records=[]
    for age in (0,60,120,252):
        c=e.signals(min_age=age)
        decisions=weekly_signal_dates(h.index)
        if age:decisions=decisions[h[SAT].notna().cumsum().reindex(decisions)>=age]
        t=satellite_targets(p,c,h,decision_dates=decisions)
        for start in ("2011-01-01","2015-01-01","2018-01-01","2020-01-01","2022-01-01"):
            if pd.Timestamp(start)>p.index[-2]: continue
            ll=simulate(p,t,pd.Timestamp(start))
            record={"case":f"extended_age{age}_start{start}","group":"extended","min_observed_quotes":age,
                    "requested_start":start,"actual_start":str(ll.dates[0].date()),**statistics(ll.returns),"skipped":ll.skipped}
            records.append(record);rows.append(record)
        if age==0:
            ll=simulate(p,t,pd.Timestamp("2011-01-01"))
            for j,symbol in enumerate(p.columns):
                first=p[symbol].first_valid_index()
                if first is not None and np.any((ll.weights[:,j]>1e-12)&(ll.dates<first)):
                    raise AssertionError(f"{symbol} 出现首次真实报价前的持仓")
            ll.series().to_csv(OUT/"extended_default_returns.csv",index_label="date",header=["return"])
            pd.DataFrame(ll.weights,index=ll.dates,columns=p.columns).to_csv(OUT/"extended_weights.csv",index_label="date")
            t.to_csv(OUT/"extended_targets.csv")
            early=ll.series().loc[:"2021-12-31"]
            records.append({"case":"extended_2011_2021_only","group":"extended",**statistics(early)})
            rows.append(records[-1])
            annual=[]
            for year,x in ll.series().groupby(ll.dates.year):
                if len(x)>1:annual.append({"year":int(year),**statistics(x)})
            pd.DataFrame(annual).to_csv(OUT/"extended_yearly.csv",index=False)
            availability=[]
            for year,part in cp.groupby(cp.index.year):
                prior=cp.loc[:part.index[-1]].notna().sum()
                availability.append({"year":int(year),"core_risk_with_60_quotes":int((prior.drop(BOND)>=60).sum()),
                                     "bond_available":bool(prior[BOND]>=60),"satellite_with_200_quotes":bool(h[SAT].loc[:part.index[-1]].notna().sum()>=200)})
            pd.DataFrame(availability).to_csv(OUT/"extended_asset_availability.csv",index=False)
            for mode,trial in (("core_only",c),("global_guard",global_risk_guard(p,t))):
                shadow=simulate(p,trial,pd.Timestamp("2011-01-01"))
                shadow.series().to_csv(OUT/f"extended_{mode}_returns.csv",index_label="date",header=["return"])
                for left,right in (("2012-01-01",str(end)),("2012-01-01","2021-12-31"),("2018-01-01",str(end)),("2022-01-01",str(end))):
                    x=shadow.series().loc[left:right]
                    rec={"case":f"extended_{mode}_{left}_{right}","group":"extended_comparison",**statistics(x)}
                    records.append(rec);rows.append(rec)
    # 2022前已有252真实报价的固定池；扩展缓存的首条报价仍不必然等于上市日。
    old=cp.loc[:"2021-12-31"].notna().sum()
    excluded=old.index[(old<252)&(old.index!=BOND)].tolist()
    c=e.signals(excluded=excluded)
    t=satellite_targets(p,c,h)
    ll=simulate(p,t)
    rec={"case":"extended_fixed_pre2022_pool","group":"extended","excluded":",".join(excluded),**statistics(ll.returns)}
    rows.append(rec);records.append(rec)
    pd.DataFrame(records).to_csv(OUT/"extended_cases.csv",index=False)
    overlap=[]
    for symbol in required:
        old=cached(symbol,end);new=series[symbol]
        aligned=pd.concat([old,new],axis=1,keys=["production_cache","new_history"],sort=True).dropna()
        error=(aligned.iloc[:,0]/aligned.iloc[:,1]-1).abs()
        overlap.append({"symbol":symbol,"overlap_quotes":len(aligned),"maximum_relative_price_difference":float(error.max()),
                        "different_quotes_above_1e_10":int((error>1e-10).sum())})
    pd.DataFrame(overlap).to_csv(OUT/"extended_data_overlap.csv",index=False)
    # 同一套单参数网格跨期复验；不依据早期表现更改扫描值或默认参数。
    sensitivity=[]
    cfg_fields=set(asdict(BASE)); core0=e.signals()
    for param,values in plan().items():
        for value in values:
            cfg=BASE;kw={};rule=RULE;cap=.3
            if param in cfg_fields:cfg=replace(BASE,**{param:value})
            elif param.startswith("sat_"):
                key={"sat_window":"window","sat_entry":"entry","sat_exit":"exit",
                     "sat_max_days":"max_days","sat_confirmation":"confirmation"}.get(param)
                if key:rule=replace(RULE,**{key:value})
                else:cap=value
            else:kw[param]=value
            cc=core0 if param.startswith("sat_") else e.signals(cfg,**kw)
            trial=satellite_targets(p,cc,h,rule,cap)
            ll=simulate(p,trial,pd.Timestamp("2011-01-01"))
            early=ll.series().loc[:"2021-12-31"];recent=ll.series().loc["2022":]
            sensitivity.append({"parameter":param,"value":str(value),
                                **{f"early_{k}":v for k,v in statistics(early).items()},
                                **{f"recent_{k}":v for k,v in statistics(recent).items()}})
        print(f"early sensitivity {param}",flush=True)
    pd.DataFrame(sensitivity).to_csv(OUT/"extended_sensitivity.csv",index=False)
    write_json(OUT/"extended_status.json",{"available":True,"actual_calendar_start":str(p.index[0].date()),
                "early_history_is_untuned_stress_not_blind_holdout":True,"excluded_fixed_pre2022":excluded,
                "no_historical_liquidity_or_dead_fund_universe":True})
    checks.append({"check":"extended_history_no_prequote_execution", "passed":True,
                   "minimum_maturity_not_liquidity_screen":True})


def plot_report(table,daily):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False,"font.size":10})
    assets=ROOT/"docs/assets";assets.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(2,2,figsize=(15,10))
    for name,label in (("default","当前规则（历史假设）"),("core_only","原核心"),
                       ("annual_walk_forward_selection","年度滚动选参"),("fixed_multiasset_vol8","固定多资产＋8%预算")):
        s=daily[name].dropna();nav=(1+s).cumprod()
        ax[0,0].plot(nav.index,nav,label=label)
        ax[1,0].plot(nav.index,(nav/nav.cummax().clip(lower=1)-1)*100,label=label)
    ax[0,0].set(title="冻结规则与对照：扣费净值",ylabel="净值");ax[0,0].legend(fontsize=8)
    ax[1,0].set(title="收盘回撤",ylabel="%")
    groups=[("oat","单参数"),("joint","联合网格"),("jitter","邻域扰动"),("subpool","随机子池")]
    values=[table.loc[table.group.eq(k),"sharpe"].dropna() for k,_ in groups]
    ax[0,1].boxplot(values,tick_labels=[v for _,v in groups],showfliers=True)
    ax[0,1].axhline(table.loc[table.case.eq("default"),"sharpe"].iloc[0],color="red",ls="--",label="当前规则")
    ax[0,1].set(title="所有结果均保留：零利率夏普分布",ylabel="夏普");ax[0,1].legend()
    grid=table.loc[table.group.eq("joint")]
    matrix=grid.groupby(["joint_ma","joint_exit"]).sharpe.median().unstack()
    im=ax[1,1].imshow(matrix.to_numpy(),aspect="auto",cmap="YlGnBu")
    ax[1,1].set_xticks(range(len(matrix.columns)),[f"{x:.0%}" for x in matrix.columns])
    ax[1,1].set_yticks(range(len(matrix)),matrix.index.astype(int))
    ax[1,1].set(title="红利MA/退出线：其余联合参数取中位数",xlabel="退出偏离",ylabel="MA窗口")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)): ax[1,1].text(j,i,f"{matrix.iloc[i,j]:.2f}",ha="center",va="center")
    fig.colorbar(im,ax=ax[1,1],label="夏普中位数")
    for a in ax.flat: a.grid(alpha=.2)
    fig.suptitle("ETF轮动过拟合审计｜2022—2026-10-20｜非独立盲测",fontsize=16)
    fig.tight_layout();fig.savefig(assets/"overfitting_audit_20261021.png",dpi=160);plt.close(fig)


def md_table(frame,columns,headers=None,percent=(),decimals=3):
    headers=headers or columns
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(columns))+" |"]
    for _,row in frame.iterrows():
        values=[]
        for c in columns:
            v=row[c]
            if pd.isna(v):text="—"
            elif c in percent:text=f"{float(v):.2%}"
            elif isinstance(v,(float,np.floating)):text=f"{v:.{decimals}f}"
            else:text=str(v)
            values.append(text.replace("|","/"))
        lines.append("| "+" | ".join(values)+" |")
    return "\n".join(lines)


def build_report(end):
    t=pd.read_csv(OUT/"all_cases.csv");by=t.set_index("case")
    daily=pd.read_csv(OUT/"daily_returns.csv",index_col=0,parse_dates=True)
    selection=json.loads((OUT/"selection_bias.json").read_text(encoding="utf-8"))
    broad_selection=json.loads((OUT/"selection_bias_broad.json").read_text(encoding="utf-8"))
    boot=json.loads((OUT/"bootstrap.json").read_text(encoding="utf-8"))
    checks=json.loads((OUT/"checks.json").read_text(encoding="utf-8"))
    inference=json.loads((OUT/"sharpe_inference.json").read_text(encoding="utf-8"))
    annualization=json.loads((OUT/"annualization.json").read_text(encoding="utf-8"))
    ic=pd.read_csv(OUT/"factor_ic_summary.csv")
    risks=pd.read_csv(OUT/"risk_budget_diagnostics.csv")
    base=by.loc["default"]
    params={"target_vol":"核心波动预算","max_risky_weight":"核心风险总仓上限","bear_scale":"弱市缩仓比例",
       "short_momentum_days":"短动量窗口","long_momentum_days":"长动量窗口","momentum_skip_days":"动量跳过日数",
       "trend_days":"趋势过滤窗口","max_single_risky_weight":"核心单只上限","inverse_vol_exponent":"逆波动权重指数",
       "max_risky_positions":"核心风险持仓数","rank_buffer_size":"排名缓冲范围","mandatory_top_rank":"必选名次数",
       "ma_reversion_days":"短均线反转窗口","ma_reversion_strength":"短均线反转强度","vol_window":"评分/权重波动窗口",
       "cov_window":"协方差窗口","bond_window":"国债动量窗口","short_weight":"短动量评分占比",
       "vol_floor":"评分波动率下限","z_clip":"反转Z截断尺度","sat_window":"红利MA窗口",
       "sat_entry":"红利入场偏离","sat_exit":"红利退出偏离","sat_cap":"红利仓上限",
       "sat_max_days":"红利最长持有日","sat_confirmation":"红利入场确认"}
    oat=pd.read_csv(OUT/"focus_2018_sensitivity.csv")
    focus_cases=pd.read_csv(OUT/"focus_2018_cases.csv")
    focus_default=focus_cases.set_index("case").loc["default"]
    focus_daily=pd.read_csv(OUT/"focus_2018_daily_returns.csv",index_col=0,parse_dates=True)
    focus_manifest=json.loads((OUT/"focus_2018_manifest.json").read_text(encoding="utf-8"))
    sensitivity=[]
    for param,values in plan().items():
        x=oat[oat.parameter.eq(param)]
        sensitivity.append({"name":params[param],"values":", ".join(str(v) for v in values),
             "annual_range":f"{x.annual.min():.2%}—{x.annual.max():.2%}",
             "sharpe_range":f"{x.sharpe.min():.3f}—{x.sharpe.max():.3f}","worst_dd":x.drawdown.min()})
    sensitivity=pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT/"sensitivity_summary.csv",index=False,encoding="utf-8-sig")
    distributions=[]
    for key,label in (("joint","162组联合网格"),("jitter","64组邻域扰动"),("subpool","64组删除4只子池"),("matched_pool","64组等数量换池")):
        x=t[t.group.eq(key)]
        ref=by.loc["default_common_end"] if key=="matched_pool" else base
        distributions.append({"name":label,"n":len(x),"median_annual":x.annual.median(),"median_sr":x.sharpe.median(),
            "min_sr":x.sharpe.min(),"max_sr":x.sharpe.max(),"worst_dd":x.drawdown.min(),
            "above_current":int((x.sharpe>=ref.sharpe-1e-10).sum())})
    distrib=pd.DataFrame(distributions)
    corecases=["default","core_only","simple_momentum_core","fixed_multiasset_weekly","fixed_multiasset_vol8","treasury_buy_hold","global_vol8_and_gross70_guard"]
    labels={"default":"当前默认（全历史假设）","core_only":"原核心，去掉红利","simple_momentum_core":"简化动量核心，无反转/缓冲",
            "fixed_multiasset_weekly":"固定多资产，每周再平衡","fixed_multiasset_vol8":"固定多资产，8%预测预算",
            "treasury_buy_hold":"十年国债ETF买入持有","global_vol8_and_gross70_guard":"全账户8%/70%风险约束对照"}
    comparison=focus_cases.copy()
    labels.update({"maturity_120":"上市报价满120日准入","maturity_252":"上市报价满252日准入"})
    comparison["label"]=comparison.case.map(labels)
    ablation=t[t.group.eq("ablation")].copy()
    ablation["label"]=ablation.case.map(lambda x:"移除"+NAMES.get(x[8:],x[8:]) if x.startswith("without_") and x[8:].isdigit() else x)
    segments=pd.read_csv(OUT/"segments.csv")
    wf=segments[segments.segment.eq("2023_onward")&segments.case.isin(["default","annual_walk_forward_selection","core_only"])].copy()
    wf["label"]=wf.case.map({"default":"固定当前规则","annual_walk_forward_selection":"每年仅用过去数据选参","core_only":"原核心"})
    diag=t[t.group.isin(["universe","replication","schedule","execution"])].copy()
    attribution=pd.read_csv(OUT/"asset_attribution.csv",dtype={"symbol":str}).sort_values("wealth_contribution_initial_capital",ascending=False)
    short_ic=ic[ic.horizon.eq(5)].copy()
    r2=pd.read_csv(OUT/"factor_forward_r2.csv")
    costs=pd.read_csv(OUT/"costs.csv")
    extended_status={"available":True}
    focus_segments=pd.read_csv(OUT/"focus_2018_segments.csv")
    fs=focus_segments[focus_segments.case.eq("default")].copy()
    fs["label"]=fs.period.map({"2018_2021":"2018—2021","2022_2024":"2022—2024","2025_onward":"2025年至今"})
    early=fs.set_index("period").loc["2018_2021"]
    cross=[]
    for parameter in ("short_momentum_days","long_momentum_days","trend_days","ma_reversion_strength","target_vol",
                      "rank_buffer_size","sat_window","sat_exit","sat_cap"):
        g=oat[oat.parameter.eq(parameter)]
        corr=g.early_sharpe.rank().corr(g.recent_sharpe.rank())
        cross.append({"name":params[parameter],"early_sr_range":f"{g.early_sharpe.min():.3f}—{g.early_sharpe.max():.3f}",
                      "recent_sr_range":f"{g.recent_sharpe.min():.3f}—{g.recent_sharpe.max():.3f}","rank_correlation":corr})
    cross_text=md_table(pd.DataFrame(cross),["name","early_sr_range","recent_sr_range","rank_correlation"],
                  ["同一参数网格","2018—2021夏普范围","2022后夏普范围（继承持仓）","跨期排序相关"])
    ext_text=f"""主研究起点固定为2018年，不再以2012年起的收益作为主评分或选参依据。2018年前的历史仅用于预热滚动因子和因果状态；评估账户从2018年首个交易日现金启动，没有计入此前账户收益。旧研究数据留存为附录参考。

{md_table(fs,["label","total","annual","sharpe","drawdown"],["同一2018账本的区间","累计收益","年化","夏普","最大回撤"],percent=("total","annual","drawdown"))}

**2018—2021年化{early.annual:.2%}、夏普{early.sharpe:.3f}、最大回撤{early.drawdown:.2%}。** 分段都来自同一2018起始账本，2022以后继承前段持仓，和2022现金启动的开发期结果略有差异；这些年份均已被观察，不是独立盲测。

2018年末具备60个真实报价的核心风险ETF为8只，2020年末13只，2024年末15只；国债2018年已有实际报价，红利2019年才开始有实际报价。2018起点更接近本策略的应用环境，仍需处理池扩充及幸存者偏差。上市观察期只处理历史长度，不能代替历史流动性和规模筛选。

最大回撤仍发生在2018年1月24日高点至11月27日低点；把主区间起点改为2018年不会消除这段风险。

![2018年以来净值与回撤](assets/overfitting_extended_20261021.png)
"""
    text=f"""# ETF轮动策略：2018年以来的过拟合与参数审计

研究日期：2026-10-21；行情截止：{end}。完整结果保存在 `output/overfitting_audit_20261021/`。

## 1. 审计结论

**当前策略有继续验证的价值，但证据不足以把夏普2、年化17.57%、回撤4.43%视为可持续实盘水平。**

- 主评估区间固定为2018年至今：年化{focus_default.annual:.2%}、夏普{focus_default.sharpe:.3f}、最大回撤{focus_default.drawdown:.2%}。2018—2021的固定规则表现弱于近年；2018年前的表现不再作为主要判断依据。
- 近年开发期的参数附近多数方案仍盈利，表明并非只有一条盈利曲线；当前20/40日动量组合、40日趋势与0.5反转强度又处于多个2022后单参数扫描的局部高点，存在开发期择优痕迹。
- 当前方案相对原核心的增益，在本轮162组合及更宽家族的多重比较校正后未达到5%显著性；不能据此认定红利无效，也不能声称其增益已经被统计确认。更宽519情景的PBO约39%—41%，比窄参数家族明显更高。
- 因子预测证据较弱，组合收益不能全部归因于稳定的截面选强能力；仓位控制、跨资产配置和具体历史行情均有贡献。
- 全账户8%/70%约束是一个值得保留的风险对照；本轮不更改默认策略、实盘切换状态和信号文件。

## 2. 研究设计与复现

冻结核心：20/40日动量各50%、60日波动、40日趋势、8%核心波动目标、弱市减半、风险总仓70%、核心单只30%、最多3只风险ETF、前4缓冲且必选第1、20日均线反转强度0.5。红利：512890，MA200、入0%、出+4%、不限日、国债现金拨款上限30%。

主评估区间为{focus_manifest['actual_first_day']}—{end}，{len(focus_daily)}个交易日；年化按252交易日，夏普以零无风险利率计算；成本默认单边5bps，信号后下一交易日收盘成交。不是实盘业绩，红利全历史参与属于明确的历史假设。起点由用户按应用范围固定，不依据本轮哪个起点收益最高继续筛选。

2018主区间完成7个配置对照与26组共125个单参数点复验，并拆分2018—2021、2022—2024及2025年至今。此前保存的 **{len(t)}个案例** 包括2022后26组单参数、162联合组合、64邻域扰动、64删除子池、64等数量换池及旧历史参考，按原区间保留，不把它们全部改称2018以来的结果。重复默认点也保留，案例数不等于独立试验数。主区间元数据在 `focus_2018_manifest.json`，原始计划在 `manifest.json`，全部较差结果均保留。

生产与审计引擎逐日收益/实际权重对账、4项不同参数对账、20bps费用对账、6个历史截断及未来扰动、价格单位变换和实盘文件指纹检查共{len(checks)}项通过；现有及新增审计测试共93项通过。这些检查验证覆盖范围内的实现，不能证明没有选池或统计过拟合。

本轮计划及补充诊断均留档，但本项目历史已被观察过；同数量换池、全账户风险对照和早期复验等补充项不构成独立预注册实验。

原2022开发期口径提醒：自然年跨度{annualization['calendar_years']:.3f}年，按自然日计算CAGR为 **{annualization['calendar_cagr']:.2%}**，按252交易日为 **{annualization['trading_252_cagr']:.2%}**。若仅作演示固定无风险利率2%，该2022起始序列夏普约{base.sharpe_rf2_illustrative:.3f}；尚未接入真实同期无风险利率。

## 3. 2018年至今：基准、模块与上市观察期对照

{md_table(comparison,["label","annual","vol","sharpe","drawdown"],["方案","年化","实际年化波动","零利率夏普","最大回撤"],percent=("annual","vol","drawdown"))}

固定多资产为20%沪深300、20%纳指、20%黄金、40%十年国债，每周再平衡；单一固定权重是对照，不代表全部被动配置的最优基准。全账户约束使用信号日前60日协方差，未用全期实现波动事后配平。国债有利率风险，不能当作无风险资产。

下图仍是2022年以来开发期的比较，与本节2018主表区间不同。

![2022开发期净值、回撤、参数分布及红利参数面](assets/overfitting_audit_20261021.png)

## 4. 主区间分段与ETF逐步扩充

{ext_text}

## 5. 2018年至今的参数敏感性：完整扫描摘要

每次只改变一项，其他参数冻结；评分/权重波动窗口与协方差窗口分开扫描。持仓数变小时，已有缓冲种子按记录顺序截取至新上限。以下范围是本轮实际扫描值，不能外推到任意取值。

{md_table(sensitivity,["name","values","annual_range","sharpe_range","worst_dd"],["参数","扫描值","年化范围","夏普范围","最差回撤"],percent=("worst_dd",))}

2022开发期的局部敏感性示例（主区间完整范围见上表）：

- 短动量20改15日，年化从17.57%降至约13.95%、夏普降至1.656；长动量40改50日，年化约15.18%、夏普1.748。并未失效，但局部敏感性值得重视。
- 核心预算5%—12%表现为风险收益有序变化：预算越高收益总体增加、夏普总体下降；当前8%不是夏普最大点。不能在审计后直接把最高夏普点再选为“新最优”。
- 红利MA120—300、退出偏离2%—8%的主样本结果相对平缓，支持进一步验证这个机制；独立周期数少及历史选择偏差仍然存在。
- MA反转强度0时，当前组合仍有约16.82%年化、1.959夏普；0.5的历史增益比较有限，不足以仅凭最高点确认为必要参数。

{md_table(distrib,["name","n","median_annual","median_sr","min_sr","max_sr","worst_dd","above_current"],["家族","案例数","中位年化","中位夏普","最小夏普","最大夏普","最差回撤","夏普不低于当前的个数"],percent=("median_annual","worst_dd"))}

邻域扰动同时更改动量、趋势、反转、仓位及红利参数，因此比单参数扫描严格。它们仍使用同一份已观察历史，不能被当作64次独立验证。

### 同一参数网格在2018—2021及2022后的表现

全部{sum(map(len,plan().values()))}个参数点从2018年现金启动，未根据分段表现修改默认；2018年前行情只用于指标和状态预热。完整结果在 `focus_2018_sensitivity.csv`。以下跨期排序相关只比较有限扫描点，点数少、相互依赖，不提供显著性保证；低或负相关说明“近年最优”未必能迁移。后段继承此前模拟持仓，和2022年现金启动口径略有差异。

{cross_text}

![主要参数的夏普与回撤敏感性](assets/overfitting_sensitivity_20261021.png)

## 6. 2022开发期：标的池依赖与持仓贡献

{md_table(ablation,["label","annual","sharpe","drawdown"],["冻结参数的删除/简化","年化","夏普","最大回撤"],percent=("annual","drawdown"))}

移除单只核心风险ETF后均未变为负年化，但不代表加入该ETF就是样本外有效。黄金+纳指、商品、海外组合的删除影响更大，说明收益来源和市场结构重要。删除子池同时减少候选数量，结果下降也可能来自分散机会变少；等数量换池从已有21只风险候选中随机选15只，减少了这个混淆，但仍局限于今天留下的有限候选集。

扩池和等数量换池统一截止2026-09-30，因为部分候选缓存没有更新至10月20日；只与同截止日原策略比较。扩池增加588000、510500、512100、159992、513180、513500，并非穷尽全市场或历史退市样本。

{md_table(attribution,["symbol","name","days_held","mean_weight","wealth_contribution_initial_capital"],["代码","名称","持有日数","平均实际权重","累计财富贡献/初始资金"],percent=("mean_weight","wealth_contribution_initial_capital"))}

贡献采用前日实际权重乘当日ETF收益，再乘前日账户净值；交易费用单列，合计与组合累计收益精确对账。贡献高不等于删掉它后的收益变化，后者会改变选仓路径。

## 7. 2022开发期：执行时点、成本和风险预算

{md_table(diag,["case","annual","sharpe","drawdown"],["对照","年化","夏普","回撤"],percent=("annual","drawdown"))}

`week_ending_0…4`分别为周一至周五收盘生成信号、下一交易日收盘执行；`execution_extra_delay_1/2/4`为在默认下一交易日基础上再延迟相应交易日。调仓星期改变会影响红利状态和缓冲路径；这些是诊断，不能据此再挑最好星期。

{md_table(costs,["single_side_bps","annual","sharpe","drawdown"],["单边总成本bps","年化","夏普","回撤"],percent=("annual","drawdown"))}

当前最大风险目标总仓{risks.risky_target.max():.0%}、最多{int(risks.holdings.max())}只ETF，最高60日协方差预测账户波动{risks.portfolio_predicted_vol.max():.2%}；{int(risks.above_original_vol_target.sum())}/{len(risks)}个信号点超过原核心8%或弱市4%的账户诊断线。该诊断包括国债，和原核心仅对风险袖套预算的计数口径不同。

红利是股票风险仓；从国债/现金转入后，原8%核心预算不能约束整个账户。全账户风险对照先限制风险仓70%/弱市35%，再按同样过去60日估计把账户缩至8%/弱市4%；没有优化新的阈值，也不保证未来回撤。未模拟盘中回撤、最低佣金、整数手、真实成交价差、涨跌停成交限制、容量及分红资金实际到账。

## 8. 2022开发期：因子IC与时间外线性预测诊断

未来标签从信号后下一交易日收盘开始，持有1/5/20交易日；不把信号日至成交日的不可获得收益计入标签。风险池不含国债及独立红利模块，每期至少5只可比较ETF。区间用12个周观察的循环块抽样，未校正本表所有因子的多重比较。

{md_table(short_ic,["scope","factor","observations","mean_rank_ic","mean_ic_95_low","mean_ic_95_high"],["截面","因子","周观察数","5日平均Rank IC","条件95%下界","条件95%上界"])}

`eligible`仅含通过40日动量和趋势资格的候选。最终评分的平均IC区间跨零；20日均线反转单因子在5/20日标签上平均IC为负，不能将其历史组合增益解读为普适短期反转规律。这与单只红利MA200状态机是不同假说，不能直接互相否定。

另外用上一年末已成熟的5日标签拟合线性模型，预测下一年截面相对收益；所有特征按当时截面标准化，标签截面去均值，零相对收益预测为R²基准。它仅检验选强弱预测性，不是另一个交易策略，也不证明非线性择时无效。

{md_table(r2,["test_year","model","train_labels_end","oos_r2_vs_zero_cross_section"],["测试年","模型","最晚训练标签结束日","时间外R²"],decimals=5)}

## 9. 2022开发期：多重试验、重抽样和滚动选参

162组合仅覆盖本次声明的核心预算、反转强度、红利窗口、退出线及仓位上限，不覆盖此前所有因子/选池研究。

相对原核心的平均日收益增益，循环块20/60/120日、各{json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))['bootstrap_replicates']}次，使用家族最大中心化均值作校正：当前规则对应p值依次 **{selection['block_max_mean']['20']['current_increment_family_adjusted_p']:.3f} / {selection['block_max_mean']['60']['current_increment_family_adjusted_p']:.3f} / {selection['block_max_mean']['120']['current_increment_family_adjusted_p']:.3f}**，均未达到5%。这是对本次增量家族的条件检验，不是整体策略盈利概率。

再把本轮参数、换池、执行时点与简化规则一起纳入更宽的{broad_selection['trials']}个比较，统一到{broad_selection['common_end']}，当前增量的校正p值为 **{broad_selection['block_max_mean']['20']['current_increment_family_adjusted_p']:.3f} / {broad_selection['block_max_mean']['60']['current_increment_family_adjusted_p']:.3f} / {broad_selection['block_max_mean']['120']['current_increment_family_adjusted_p']:.3f}**。这仍未覆盖全部既往研究和人为选择，不能称为全项目校正。

CSCV将历史分成10个连续区块，252种半数训练/半数检验组合：家族PBO为{selection['cscv'][0]['pbo_family']:.2%}；训练/检验边界各去20个交易日后为{selection['cscv'][1]['pbo_family']:.2%}。低PBO提供局部参数排序稳定的支持，但候选回报高度相关，且该划分不是时间向前的实盘；不能解释成当前策略只有这些概率会失败。[PBO原论文](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)

**更宽{broad_selection['trials']}情景的PBO分别升至{broad_selection['cscv'][0]['pbo_family']:.2%}和{broad_selection['cscv'][1]['pbo_family']:.2%}。** 因此不能只报告窄家族较低的PBO。这里衡量的是该候选家族中“训练选出的最佳者在检验段落至家族中位数以下”的频率，不是当前默认策略有约41%概率亏损。20日边界剔除也不足以完全隔离200日均线或不限期持仓状态的影响，仍属于条件诊断。

年度滚动选择仅用截至上一年末的收益选162组中训练夏普最高者，下一年按该候选的因果影子状态交易并扣费；2022年仅用于训练，正式比较从2023年起。未做内层交叉验证，不称为完整嵌套验证。所有这些年已经被观察，因此仍是历史流程诊断。

{md_table(wf,["label","annual","sharpe","drawdown"],["2023年至今，同区间","年化","夏普","回撤"],percent=("annual","drawdown"))}

60日循环块对固定当前曲线抽样的条件95%夏普区间约 **{boot['60']['sharpe_95'][0]:.2f}—{boot['60']['sharpe_95'][1]:.2f}**。这些样本来自已有市场路径，未计入事先选池/选参，不覆盖未出现的危机，不能作为未来收益或回撤置信保证。

HAC长期方差口径的夏普，滞后5/20/60日分别为{inference['hac'][0]['hac_long_run_sharpe']:.3f}/{inference['hac'][1]['hac_long_run_sharpe']:.3f}/{inference['hac'][2]['hac_long_run_sharpe']:.3f}。IID假设下有效试验数1/10/100/1000/10000的DSR式情景另存 `sharpe_inference.json`；试验数不明，未把情景值包装成已完成的全项目DSR或盈利概率。[Lo夏普统计研究](https://rpc.cfainstitute.org/research/financial-analysts-journal/2002/the-statistics-of-sharpe-ratios)、[DSR原论文](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)

## 10. 建议与尚未完成的验证

1. **固定2018研究范围，保留前瞻验证。** 2018年至今夏普{focus_default.sharpe:.3f}、最大回撤{focus_default.drawdown:.2%}；近年表现更好，仍不能把5%回撤当作上限。不要在看过结果后不断将研究起点向后移动，或根据本轮最高点再次升级参数。
2. **优先统一全账户风险约束。** 当前红利替换防守资金后，账户风险目标与最初8%预算已不同。全账户约束对照用于下一步决策，本轮未切换默认。
3. **将20/40窗口和0.5反转强度视为待验证设置。** 可保留简化动量、当前模型、全账户约束三个固定影子组合，记录同期开盘后实际可得信号及成交成本；避免不断增加候选模型。
4. **建立客观历史候选池。** 上市观察期、滞后成交额、历史规模和按指数/资产类别选代表ETF应提前固定。当前数据只足以检验报价可得性，尚未排除幸存者偏差和人工分类偏差。
5. **保留试验台账。** 历史被反复使用后无法恢复盲测；保留成功与失败结果、规则版本、启用日期，未来修改重新建立前瞻记录。实盘收益按外部现金流调整，新增资金不算收益。

用户确认暂时没有额外历史选池资料。本轮已自行取得早期ETF日线及复权因子；仍需的优先数据：

| 优先级 | 数据 | 用途/限制 |
| --- | --- | --- |
| 高 | 覆盖已清盘ETF的完整产品名单、上市/退市日、历史跟踪指数/分类 | 重建当时全市场候选集，避免只看今日存续产品 |
| 高 | 各ETF逐日成交额/成交量，最好上市以来 | 滞后流动性准入和容量评估，不能用今日成交额回填 |
| 高 | 历史份额/规模及实际披露日期 | 使用当时已公布规模，避免规模筛选未来函数 |
| 中 | 历史买卖价差、收盘竞价/临近收盘行情、停牌/涨跌停记录 | 检验真实成交和滑点，不把收盘价假定为必然可成交 |
| 中 | 逐日可得的无风险利率、行情历史版本 | 超额收益夏普与数据修订审计 |

Tushare当前基金列表快照含上市/退市等字段，但当前快照不等于历史规模或分类快照。[Tushare官方说明](https://tushare.pro/document/2?doc_id=19)

## 11. 旧研究附录与复现

2018年前的旧研究结果保留在 `extended_cases.csv`、`extended_sensitivity.csv` 与 `extended_*_returns.csv` 中，仅作历史附录参考，不再用于本轮主要评分和选参。重新运行默认审计会生成2018主区间研究及2022开发期诊断，不再新增2018年前收益扫描。

```powershell
.\\.venv\\Scripts\\python.exe research_overfitting_audit.py --end 2026-10-20
# 只重算2018主区间与125个参数点，然后更新报告
.\\.venv\\Scripts\\python.exe research_overfitting_audit.py --focus-2018 --end 2026-10-20
# 可选：独立缓存下载更早历史；首次需要网络权限
.\\.venv\\Scripts\\python.exe research_overfitting_audit.py --download-history --end 2026-10-20
# 只从已完成结果重新生成报告与图
.\\.venv\\Scripts\\python.exe research_overfitting_audit.py --report-only
```

入口 `research_overfitting_audit.py`；本报告和三张图在 `docs/`；所有逐日收益、完整扫描、指标及验证记录在 `output/overfitting_audit_20261021/`。市场缓存、API凭证和账户数据库未加入研究报告，实盘保护文件指纹保持一致。
"""
    (ROOT/"docs/overfitting_audit_20261021.md").write_text(text,encoding="utf-8")
    write_json(OUT/"report_manifest.json",{"research_source_hash":json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))["source_hash"],
               "focus_source_hash":focus_manifest['source_hash'],"evaluation_start":"2018-01-01",
               "report_source_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"end":str(end)})
    # 参数图：保持全部扫描点，不裁剪差结果。
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False})
    chosen=["short_momentum_days","long_momentum_days","trend_days","ma_reversion_strength","target_vol","rank_buffer_size",
            "cov_window","vol_window","sat_window","sat_entry","sat_exit","sat_cap"]
    fig,axes=plt.subplots(4,3,figsize=(15,15))
    for ax,param in zip(axes.flat,chosen):
        x=oat[oat.parameter.eq(param)].copy();x["numeric"]=x.value.astype(float);x=x.sort_values("numeric")
        ax.plot(x.numeric,x.sharpe,"o-",color="#126e82",label="夏普")
        right=ax.twinx();right.plot(x.numeric,x.drawdown*100,"s--",color="#b65335",alpha=.8,label="回撤%")
        ax.set_title(params[param]);ax.set_ylabel("夏普",color="#126e82");right.set_ylabel("回撤%",color="#b65335")
        ax.grid(alpha=.2)
    fig.suptitle("单参数敏感性｜其他参数固定｜2018—2026-10-20",fontsize=16)
    fig.tight_layout();fig.savefig(ROOT/"docs/assets/overfitting_sensitivity_20261021.png",dpi=150);plt.close(fig)
    if extended_status["available"]:
        fig,axes=plt.subplots(3,1,figsize=(14,11),sharex=True,gridspec_kw={"height_ratios":[2,1,1]})
        for stem,label in (("default","当前规则"),("core_only","去掉红利"),("global_guard","全账户风险约束对照")):
            key={"default":"default","core_only":"core_only","global_guard":"global_vol8_and_gross70_guard"}[stem]
            s=focus_daily[key]
            nav=(1+s).cumprod();axes[0].plot(nav.index,nav,label=label)
            axes[1].plot(nav.index,(nav/nav.cummax().clip(lower=1)-1)*100,label=label)
        availability=pd.read_csv(OUT/"focus_2018_asset_availability.csv")
        axes[2].step(pd.to_datetime(availability.year.astype(str)+"-12-31"),availability.core_risk_with_60_quotes,where="post")
        axes[0].set(title="主要研究区间：2018年以来的净值与回撤",ylabel="净值");axes[0].legend()
        axes[1].set(ylabel="收盘回撤%")
        axes[2].set(ylabel="年末满60报价的核心风险ETF数",xlabel="早期池较稀疏；当前名单仍有后视选择，未完成历史流动性筛选")
        for ax in axes:
            ax.axvline(pd.Timestamp("2022-01-01"),color="gray",ls="--",alpha=.7);ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(ROOT/"docs/assets/overfitting_extended_20261021.png",dpi=160);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end",type=date.fromisoformat,default=date(2026,10,20))
    parser.add_argument("--download-history",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--focus-2018",action="store_true")
    parser.add_argument("--bootstrap-reps",type=int,default=2000)
    args=parser.parse_args()
    if args.download_history: download_history(args.end)
    elif args.focus_2018:
        focus_2018_audit(args.end)
        build_report(args.end)
    elif args.report_only:
        table=pd.read_csv(OUT/"all_cases.csv")
        daily=pd.read_csv(OUT/"daily_returns.csv",index_col=0,parse_dates=True)
        plot_report(table,daily)
        build_report(args.end)
    else: execute(args.end,args.bootstrap_reps)
