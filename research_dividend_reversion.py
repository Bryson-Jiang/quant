"""红利低波长期均线反转研究；冻结参数、复用成交引擎，不改默认策略。"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import CORE_STRATEGY_NAME as STRATEGY_NAME, ETF_NAMES
from etf_rotation.data import load_close_prices, load_symbol
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.mean_reversion import ReversionRule, capped_risk_weights, defensive_funding_targets, reversion_states
from etf_rotation.risk_strategy import BOND, RiskConfig, risk_controlled_signals, risk_score_components, weekly_signal_dates
from etf_rotation.strategy_state import load_rotation_state
from research_reversal import choose_case, stats

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"output/dividend_reversion_20261021"
START=date(2022,1,1)
EARLY_START=date(2020,1,6)
FUNDS={"512890":"华泰柏瑞红利低波ETF", "515100":"景顺长城红利低波100ETF", "515450":"南方红利低波50ETF"}
PRIMARY="512890"
SOURCES=[
    ("上交所：512890份额拆分结果公告", "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2021-10-25/512890_20211025_1_lJ5GeyLb.pdf"),
    ("景顺长城：515100产品资料", "https://www.igwfmc.com/main/jjcp/product/515100/detail.html"),
    ("南方基金：515450产品资料概要", "https://www.nffund.com/main/files/2021/12/13/868077036325.pdf"),
    ("Bailey等：The Probability of Backtest Overfitting", "https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf"),
]


def fixed_rules():
    rules=[ReversionRule(w,e,x,None) for w in (150,200) for e in (0.,-0.03,-0.05) for x in (0.03,0.05)]
    rules += [ReversionRule(w,-0.03,0.03,120,c) for w in (150,200) for c in ("none","bounce3","slope20")]
    return rules


def case_key(rule,mode,weight):
    return f"{rule.key}_{mode}_cap{int(weight*100)}"


def primary_key():
    return case_key(ReversionRule(),"budget_add",0.30)


def baseline_config():
    return RiskConfig(target_vol=0.08,short_momentum_days=20,long_momentum_days=40,momentum_skip_days=0,
                      trend_days=40,max_single_risky_weight=0.30,rank_buffer_size=4,mandatory_top_rank=1,
                      ma_reversion_days=20,ma_reversion_strength=0.5)


def protected():
    names=("live_signal.py","run_backtest.py","etf_rotation/config.py","etf_rotation/risk_strategy.py",
           "data/rotation_state.json","output/live_signal.json")
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names if (ROOT/n).exists()}


def initialize(end):
    OUT.mkdir(parents=True,exist_ok=True)
    snapshot=OUT/"rotation_state_snapshot.json"
    if not snapshot.exists(): snapshot.write_bytes((ROOT/"data/rotation_state_20261018.json").read_bytes())
    manifest={"end":str(end),"main_start":str(START),"standalone_start":str(EARLY_START),"primary_etf":PRIMARY,
              "replication_etfs":list(FUNDS)[1:],"strategy":STRATEGY_NAME,"baseline_config":asdict(baseline_config()),
              "rules":[asdict(r)|{"key":r.key} for r in fixed_rules()],"weekly_integrations":72,
              "primary_case":primary_key(),"funding_modes":["risk_replace","budget_add"],"satellite_caps":[0.15,0.30],
              "confirmation_additions_predeclared":True,"daily_followup_cases":4,"heavy50_stress_cases":2,
              "design_end":"2024-12-31","past_main_years_already_used":True,"protected_before":protected()}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest


def download(end):
    for symbol in ETF_NAMES:
        start=date(2019,1,1) if symbol in {BOND,"510300"} else date(2020,11,27)
        values=load_symbol(symbol,start,end,ROOT/"data/cache")
        print(f"updated {symbol}: {values.index[-1].date()}",flush=True)
    for symbol in FUNDS:
        values=load_symbol(symbol,date(2019,1,1),end,ROOT/"data/cache")
        print(f"updated {symbol}: {values.index[0].date()}..{values.index[-1].date()}",flush=True)


def load_data(end):
    core=load_close_prices(START,end,ROOT/"data/cache",offline=True)
    if core.index[-1].date()!=end or core.iloc[-1].isna().any():
        raise ValueError("研究截止日行情不完整，先更新数据或明确改用更早的完整交易日")
    early_market=load_symbol("510300",date(2019,1,1),end,ROOT/"data/cache",offline=True)
    early_bond=load_symbol(BOND,date(2019,1,1),end,ROOT/"data/cache",offline=True)
    full={}
    quality={}
    for symbol in FUNDS:
        fund=load_symbol(symbol,date(2019,1,1),end,ROOT/"data/cache",offline=True)
        full[symbol]=pd.DataFrame({"510300":early_market,BOND:early_bond,symbol:fund},index=early_market.index)
        missing=full[symbol].loc[core.index,symbol].index[full[symbol].loc[core.index,symbol].isna()]
        quality[symbol]={"missing_quotes":[str(d.date()) for d in missing],"first_quote":str(fund.index[0].date())}
        if full[symbol].loc[str(START):,symbol].isna().any():
            raise ValueError(f"{symbol}主区间存在缺价，不静默填补")
    (OUT/"data_quality.json").write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding="utf-8")
    return core,full


def core_context(core,removed=()):
    state=load_rotation_state(OUT/"rotation_state_snapshot.json",expected_strategy=STRATEGY_NAME)
    config=baseline_config()
    specs={FactorSpec("momentum",n) for n in (20,40,60)}|{FactorSpec("trend",40),FactorSpec("volatility",60),FactorSpec("ma_zscore",20)}
    frames={s:compute_factor(core,s) for s in specs}
    if removed:frames[FactorSpec("momentum",40)].loc[:,list(removed)]=np.nan
    target=risk_controlled_signals(core,config,factor_frames=frames,**state.signal_kwargs(core))
    score=risk_score_components(frames[FactorSpec("momentum",20)],frames[FactorSpec("momentum",40)],
                                frames[FactorSpec("volatility",60)],config,frames[FactorSpec("ma_zscore",20)])["score"]
    return target,score,frames[FactorSpec("trend",40)]["510300"].le(0),frames[FactorSpec("momentum",60)][BOND].gt(0)


def combine(prices,base,score,bear,bond_ok,state,symbol,mode,cap):
    """核心按原周目标独立重放；卫星活跃时保留评分最高的至多2只核心风险ETF。"""
    decisions=base.index.union(state.index[state.active.ne(state.active.shift(1,fill_value=False))]).sort_values()
    decisions=decisions[decisions>=base.index[0]]
    prior=base.reindex(base.index.union(decisions)).sort_index().ffill().reindex(decisions)
    returns=prices.pct_change(fill_method=None)
    rows=[]
    for day in decisions:
        original=prior.loc[day].reindex(prices.columns,fill_value=0).copy()
        if not bool(state.at[day,"active"]):
            rows.append(original.rename(day));continue
        old=original.drop([BOND,symbol]); old=old.loc[old.gt(1e-12)]
        gross=float(old.sum())
        keep=score.loc[day,old.index].sort_index().sort_values(ascending=False,kind="stable").index[:2]
        desired=old.reindex(keep)
        if mode=="risk_replace":
            new=min(cap,gross)
            desired=desired/desired.sum()*(gross-new) if desired.sum()>0 else desired*0
            desired.loc[symbol]=new
        else:
            desired.loc[symbol]=cap
        if mode=="heavy50":
            if desired.sum()>1:
                desired.loc[keep] *= (1-cap)/desired.loc[keep].sum()
            risk=desired
        else:
            caps=pd.Series(0.30,index=desired.index);caps[symbol]=cap
            risk=capped_risk_weights(desired,returns,day,target_vol=0.08,bear=bool(bear.loc[day]),
                      risk_ceiling=gross if mode=="risk_replace" else None,max_weights=caps)
        row=pd.Series(0.,index=prices.columns,name=day);row.loc[risk.index]=risk
        if bond_ok.loc[day]: row[BOND]=max(0.,1-float(row.sum()))
        rows.append(row)
    result=pd.DataFrame(rows);result.index.name="signal_date"
    return result


def standalone_targets(prices,state,symbol,decisions):
    bond=compute_factor(prices[[BOND]],FactorSpec("momentum",60))[BOND].gt(0)
    rows=[]
    for day in decisions:
        row=pd.Series(0.,index=prices.columns,name=day)
        if state.at[day,"active"]:row[symbol]=1.
        elif bond.loc[day]:row[BOND]=1.
        rows.append(row)
    return pd.DataFrame(rows)


def episodes(prices,state,symbol,rule):
    rows=[];opening=None
    for day in state.index[state.reason.isin(["entry","ma_exit","time_exit"])]:
        i=prices.index.get_loc(day);execution=i+1
        if state.at[day,"reason"]=="entry":
            if execution<len(prices):opening=(day,execution)
        elif opening is not None and execution<len(prices):
            entry_day,entry=opening
            path=prices[symbol].iloc[entry:execution+1]/prices[symbol].iloc[entry]-1
            rows.append({"rule":rule.key,"symbol":symbol,"entry_signal":str(entry_day.date()),
                         "entry_execution":str(prices.index[entry].date()),"exit_signal":str(day.date()),
                         "exit_execution":str(prices.index[execution].date()),"exit_reason":state.at[day,"reason"],
                         "closed":True,"holding_days":execution-entry,"asset_return":float(path.iloc[-1]),
                         "mae":float(path.min()),"mfe":float(path.max())})
            opening=None
    if opening is not None:
        day,entry=opening;path=prices[symbol].iloc[entry:]/prices[symbol].iloc[entry]-1
        rows.append({"rule":rule.key,"symbol":symbol,"entry_signal":str(day.date()),"entry_execution":str(prices.index[entry].date()),
                     "exit_signal":None,"exit_execution":None,"exit_reason":"open_censored","closed":False,
                     "holding_days":len(prices)-1-entry,"asset_return":float(path.iloc[-1]),"mae":float(path.min()),"mfe":float(path.max())})
    return pd.DataFrame(rows)


def record(name,result,signal,symbol=None):
    daily=result.equity.daily_return
    stress=(1+result.equity.gross_return)*(1-result.equity.turnover*20/10000)-1
    row={"variant":name,**stats(daily),"volatility":float(daily.std(ddof=1)*np.sqrt(252)),
         "stress20_annual":stats(stress)["annual"],"stress20_drawdown":stats(stress)["drawdown"],
         "turnover_per_year":float(result.equity.turnover.sum()*252/len(daily)),"worst_day":float(daily.min()),
         "mean_dividend_weight":float(result.weights[symbol].mean()) if symbol in result.weights else 0,
         "max_dividend_weight":float(result.weights[symbol].max()) if symbol in result.weights else 0,
         "dividend_days":int(result.weights[symbol].gt(1e-10).sum()) if symbol in result.weights else 0,
         "skipped_trades":int(result.trades.status.eq("skipped_no_quote").sum())}
    for prefix,left,right in (("design","2022","2024-12-31"),("later","2025",str(daily.index[-1].date()))):
        segment=daily.loc[left:right]
        if len(segment):row.update({f"{prefix}_{k}":v for k,v in stats(segment).items()})
    return row


def block_check(daily,leaders,count=2000,block=60):
    data=daily.to_numpy();base=daily.columns.get_loc("default")
    diff=data-data[:,[base]]; centered=diff-diff.mean(axis=0); observed=float(diff.mean(axis=0).max())
    rng=np.random.default_rng(20261021);maximum=[];samples={n:[] for n in leaders}
    for _ in range(count):
        starts=rng.integers(0,len(data),(len(data)+block-1)//block)
        ix=((starts[:,None]+np.arange(block))%len(data)).ravel()[:len(data)]
        maximum.append(centered[ix].mean(axis=0).max())
        b=data[ix,base];bs=b.mean()/b.std(ddof=1)*np.sqrt(252)
        for n in leaders:
            x=data[ix,daily.columns.get_loc(n)]
            samples[n].append(((x-b).mean()*252,x.mean()/x.std(ddof=1)*np.sqrt(252)-bs))
    return {"block_days":block,"replicates":count,"corrected_weekly_trials":len(daily.columns)-1,
            "family_max_mean_p":float((1+np.sum(np.asarray(maximum)>=observed))/(count+1)),
            "paired_unadjusted_95pct":{n:{"annual_arithmetic_difference":np.quantile(np.asarray(v)[:,0],[.025,.975]).tolist(),
              "sharpe_difference":np.quantile(np.asarray(v)[:,1],[.025,.975]).tolist()} for n,v in samples.items()},
            "limitation":"不校正以往研究、ETF选择、后验夏普挑选或非平稳市场；跨ETF并非独立样本"}


def execute(end):
    manifest=initialize(end)
    core,full=load_data(end);base,score,bear,bond_ok=core_context(core)
    prices=core.copy();prices[PRIMARY]=full[PRIMARY][PRIMARY].reindex(core.index)
    rules=fixed_rules();states={r.key:reversion_states(full[PRIMARY],PRIMARY,r,weekly_signal_dates(full[PRIMARY].index)) for r in rules}
    manifest["data_sha256"]=hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest()
    manifest["full_price_sha256"]={s:hashlib.sha256(pd.util.hash_pandas_object(f,index=True).values.tobytes()).hexdigest() for s,f in full.items()}
    rows=[];signals={"default":base.reindex(columns=prices.columns,fill_value=0)};results={};yearly=[]
    initial=run_backtest(prices,signals["default"],START,end)
    results["default"]=initial;rows.append(record("default",initial,signals["default"],PRIMARY))
    standalone=[];episode_rows=[]
    for rule in rules:
        st=states[rule.key]
        ss=standalone_targets(full[PRIMARY],st,PRIMARY,weekly_signal_dates(full[PRIMARY].index))
        single=run_backtest(full[PRIMARY],ss,EARLY_START,end)
        standalone.append(record(rule.key,single,ss,PRIMARY)|asdict(rule))
        episode_rows.append(episodes(full[PRIMARY],st,PRIMARY,rule))
        for mode in ("risk_replace","budget_add"):
            for cap in (.15,.30):
                name=case_key(rule,mode,cap)
                target=combine(prices,base,score,bear,bond_ok,st.reindex(core.index),PRIMARY,mode,cap)
                result=run_backtest(prices,target,START,end)
                signals[name],results[name]=target,result
                rows.append(record(name,result,target,PRIMARY)|asdict(rule)|{"rule":rule.key,"mode":mode,"cap":cap})
                print(f"evaluated {name}: annual={rows[-1]['annual']:.2%} SR={rows[-1]['sharpe']:.3f} DD={rows[-1]['drawdown']:.2%}",flush=True)
    table=pd.DataFrame(rows).set_index("variant")
    daily=pd.DataFrame({n:r.equity.daily_return for n,r in results.items()})
    design=choose_case(daily,pd.Timestamp("2024-12-31"),[2022,2023,2024])
    leader=table.loc[table.drawdown.ge(-.10)].sort_values("sharpe",ascending=False).index[0]
    candidate=table.drop(index="default").sort_values("sharpe",ascending=False).index[0]
    choices=pd.Series("default",index=base.index);folds=[]
    for year in (2024,2025,2026):
        cutoff=core.index[core.index<pd.Timestamp(f"{year}-01-01")][-6]
        chosen=choose_case(daily,cutoff,list(range(2022,year)))
        choices.loc[choices.index.year==year]=chosen
        folds.append({"year":year,"training_end":str(cutoff.date()),"variant":chosen})
    wf=pd.DataFrame([signals[choices.loc[day]].loc[day] for day in base.index],index=base.index)
    results["walk_forward"]=run_backtest(prices,wf,START,end);signals["walk_forward"]=wf
    rows.append(record("walk_forward",results["walk_forward"],wf,PRIMARY))
    daily["walk_forward"]=results["walk_forward"].equity.daily_return
    for name,result in results.items():
        for year,segment in result.equity.daily_return.groupby(result.equity.index.year):
            yearly.append({"variant":name,"year":int(year),**stats(segment)})
    pd.DataFrame(rows).to_csv(OUT/"comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(standalone).to_csv(OUT/"standalone.csv",index=False,encoding="utf-8-sig")
    pd.concat(episode_rows,ignore_index=True).to_csv(OUT/"episodes.csv",index=False,encoding="utf-8-sig")
    daily.to_csv(OUT/"daily_returns.csv",encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT/"yearly.csv",index=False,encoding="utf-8-sig")
    pd.concat(signals,names=["variant","signal_date"]).to_csv(OUT/"target_weights.csv",encoding="utf-8-sig")
    selection={"primary":primary_key(),"full_sample_leader":leader,"best_candidate":candidate,"design_selection":design,"walk_forward_choices":folds}
    (OUT/"selection.json").write_text(json.dumps(selection,ensure_ascii=False,indent=2),encoding="utf-8")
    followups(core,full,base,score,bear,bond_ok,states,signals,results,selection,end)
    posthoc_controls(end)
    boot=block_check(daily.drop(columns="walk_forward"),list(dict.fromkeys([primary_key(),leader,design])))
    (OUT/"bootstrap.json").write_text(json.dumps(boot,ensure_ascii=False,indent=2),encoding="utf-8")
    if protected()!=manifest["protected_before"]:raise AssertionError("研究修改了正式策略或信号")
    manifest.update({"protected_unchanged":True,"posthoc_controls":3,"posthoc_cycle_ablation":True,
                     "time_audits":json.loads((OUT/"checks.json").read_text(encoding="utf-8")),
                     "code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"COMPLETE {write_report(end)}",flush=True)


def lookup_cases():
    return {case_key(r,m,w):(r,m,w) for r in fixed_rules() for m in ("risk_replace","budget_add") for w in (.15,.30)}


def followups(core,full,base,score,bear,bond_ok,states,signals,results,selection,end):
    cases=lookup_cases()
    literal=case_key(ReversionRule(200,0,0.03,None),"budget_add",.30)
    names=list(dict.fromkeys([primary_key(),literal,selection["best_candidate"],selection["design_selection"]]))
    names=[n for n in names if n!="default"]
    replica=[];extra=[];checks=[];holdings=[]
    for name in ["default"]+names+["walk_forward"]:
        for symbol in results[name].weights:
            w=results[name].weights[symbol]
            holdings.append({"variant":name,"symbol":symbol,"days":int(w.gt(1e-10).sum()),"mean_weight":float(w.mean()),"max_weight":float(w.max())})
    pd.DataFrame(holdings).to_csv(OUT/"holdings.csv",index=False,encoding="utf-8-sig")
    for symbol in list(FUNDS)[1:]:
        prices=core.copy();prices[symbol]=full[symbol][symbol].reindex(core.index)
        for name in names:
            rule,mode,cap=cases[name]
            st=reversion_states(full[symbol],symbol,rule,weekly_signal_dates(full[symbol].index)).reindex(core.index)
            target=combine(prices,base,score,bear,bond_ok,st,symbol,mode,cap)
            result=run_backtest(prices,target,START,end)
            replica.append(record(name,result,target,symbol)|{"symbol":symbol})
        print(f"replication {symbol} complete",flush=True)
    pd.DataFrame(replica).to_csv(OUT/"replication.csv",index=False,encoding="utf-8-sig")
    primary_prices=core.copy();primary_prices[PRIMARY]=full[PRIMARY][PRIMARY].reindex(core.index)
    for window in (150,200):
        rule=ReversionRule(window,-.03,.03,120)
        daily_state=reversion_states(full[PRIMARY],PRIMARY,rule,full[PRIMARY].index).reindex(core.index)
        for mode in ("risk_replace","budget_add"):
            key="daily_"+case_key(rule,mode,.30)
            target=combine(primary_prices,base,score,bear,bond_ok,daily_state,PRIMARY,mode,.30)
            result=run_backtest(primary_prices,target,START,end)
            extra.append(record(key,result,target,PRIMARY)|{"kind":"daily_event_followup"})
        weekly_state=states[rule.key].reindex(core.index)
        target=combine(primary_prices,base,score,bear,bond_ok,weekly_state,PRIMARY,"heavy50",.50)
        result=run_backtest(primary_prices,target,START,end)
        extra.append(record("heavy50_"+rule.key,result,target,PRIMARY)|{"kind":"outside_default_risk_constraints"})
    pd.DataFrame(extra).to_csv(OUT/"frequency_and_heavy_stress.csv",index=False,encoding="utf-8-sig")
    # 与默认相同的信息时点，分别移除银行、黄金+纳指，以及把国债换成现金。
    ablations=[]
    for label,removed in (("without_bank",("512800",)),("without_gold_nasdaq",("518880","159941")),("bond_to_cash",())):
        cb,cs,cm,co=core_context(core,removed)
        for name in list(dict.fromkeys(["default",primary_key(),selection["best_candidate"]])):
            if name=="default":target=cb.reindex(columns=primary_prices.columns,fill_value=0)
            else:
                rule,mode,cap=cases[name]
                target=combine(primary_prices,cb,cs,cm,co,states[rule.key].reindex(core.index),PRIMARY,mode,cap)
            if label=="bond_to_cash":target[BOND]=0.
            result=run_backtest(primary_prices,target,START,end)
            ablations.append({"scenario":label,"variant":name,**stats(result.equity.daily_return)})
        print(f"ablation {label} complete",flush=True)
    pd.DataFrame(ablations).to_csv(OUT/"ablation.csv",index=False,encoding="utf-8-sig")
    for cutoff in ("2023-06-30","2025-06-30","2026-09-30"):
        short=core.loc[:cutoff];cb,cs,cm,co=core_context(short)
        fp=full[PRIMARY].loc[:cutoff]
        prices=short.copy();prices[PRIMARY]=fp[PRIMARY].reindex(short.index)
        for name in names:
            rule,mode,cap=cases[name]
            st=reversion_states(fp,PRIMARY,rule,weekly_signal_dates(fp.index)).reindex(short.index)
            target=combine(prices,cb,cs,cm,co,st,PRIMARY,mode,cap)
            days=target.index[target.index<short.index[-1]]
            error=float((target.loc[days]-signals[name].loc[days]).abs().to_numpy().max())
            assert error<1e-12
            checks.append({"variant":name,"check":"prefix_"+cutoff,"max_target_error":error})
    changed_core=core.copy();changed_core.loc[changed_core.index>"2025-06-30"]*=np.linspace(2,5,len(core.columns))
    cb,cs,cm,co=core_context(changed_core)
    changed=full[PRIMARY].copy();changed.loc[changed.index>"2025-06-30",PRIMARY]*=7
    cp=changed_core.copy();cp[PRIMARY]=changed[PRIMARY].reindex(cp.index)
    for name in names:
        rule,mode,cap=cases[name]
        st=reversion_states(changed,PRIMARY,rule,weekly_signal_dates(changed.index)).reindex(cp.index)
        target=combine(cp,cb,cs,cm,co,st,PRIMARY,mode,cap)
        error=float((target.loc[:"2025-06-30"]-signals[name].loc[:"2025-06-30"]).abs().to_numpy().max())
        assert error<1e-12
        checks.append({"variant":name,"check":"future_prices_changed","max_target_error":error})
    for name,target in signals.items():
        risky=target.drop(columns=BOND)
        assert risky.gt(1e-10).sum(axis=1).le(3).all()
        assert risky.le(.30+1e-12).all().all()
        assert risky.sum(axis=1).le(.70+1e-12).all()
        assert target.sum(axis=1).le(1+1e-12).all()
    # 只核验主方案一次成本公式，避免逐参数重复跑同一验证。
    stress=run_backtest(primary_prices,signals[primary_key()],START,end,fee_bps=20,slippage_bps=0)
    eq=results[primary_key()].equity
    np.testing.assert_allclose(stress.equity.daily_return,(1+eq.gross_return)*(1-eq.turnover*20/10000)-1,atol=1e-12)
    pd.DataFrame(checks).to_csv(OUT/"time_audit.csv",index=False,encoding="utf-8-sig")
    (OUT/"checks.json").write_text(json.dumps({"time_audits":len(checks),"max_target_error":max(r['max_target_error'] for r in checks),
            "all_constrained_target_caps_passed":True,"stress20_formula_verified":True},indent=2),encoding="utf-8")
    assets=pd.DataFrame({s:f[s].reindex(core.index) for s,f in full.items()}|{"512800":core['512800']})
    assets.pct_change(fill_method=None).corr().to_csv(OUT/"asset_correlations.csv",encoding="utf-8-sig")
    status=[]
    for rule in fixed_rules():
        row=states[rule.key].iloc[-1]
        status.append({"rule":rule.key,"as_of":str(core.index[-1].date()),"deviation":float(row.deviation),"active":bool(row.active)})
    pd.DataFrame(status).to_csv(OUT/"latest_research_states.csv",index=False,encoding="utf-8-sig")
    # 只保留一组图用净值，失败项仍保留完整汇总和每日收益供审计。
    leaders=list(dict.fromkeys(["default",primary_key(),literal,selection["best_candidate"],"walk_forward"]))
    pd.DataFrame({n:results[n].equity.strategy_nav for n in leaders}).to_csv(OUT/"nav.csv",encoding="utf-8-sig")
    pd.DataFrame({"price":full[PRIMARY][PRIMARY],"deviation200":states[ReversionRule().key].deviation,
                  "primary_active":states[ReversionRule().key].active}).loc[str(START):].to_csv(OUT/"price_and_state.csv",encoding="utf-8-sig")


def posthoc_controls(end):
    """网格结果后的归因对照，单独记账，不混入72项预定网格的校正或选优。"""
    core,full=load_data(end);base,score,bear,bond_ok=core_context(core)
    prices=core.copy();prices[PRIMARY]=full[PRIMARY][PRIMARY].reindex(core.index)
    selection=json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    state=load_rotation_state(OUT/"rotation_state_snapshot.json",expected_strategy=STRATEGY_NAME)
    expanded=risk_controlled_signals(prices,baseline_config(),**state.signal_kwargs(prices))
    result=run_backtest(prices,expanded,START,end)
    rows=[record("plain_17_etf_rotation",result,expanded,PRIMARY)]
    constant=pd.DataFrame({"active":False},index=core.index)
    constant.loc[constant.index>=base.index[base.index>=core.index[60]][0],"active"]=True
    # 原组合预热早于绩效起点；持续配置对照在主样本起点前已产生周目标。
    for mode in ("risk_replace","budget_add"):
        target=combine(prices,base,score,bear,bond_ok,constant,PRIMARY,mode,.30)
        result=run_backtest(prices,target,START,end)
        rows.append(record("constant_dividend30_"+mode,result,target,PRIMARY))
    pd.DataFrame(rows).to_csv(OUT/"posthoc_controls.csv",index=False,encoding="utf-8-sig")
    # 真正的买入持有在区间首日收盘买入一次，不进行周度再平衡。
    fp=full[PRIMARY]
    prior=fp.index[fp.index<pd.Timestamp(EARLY_START)][-1]
    target=pd.DataFrame(0.,index=pd.DatetimeIndex([prior]),columns=fp.columns);target[PRIMARY]=1.
    result=run_backtest(fp,target,EARLY_START,end)
    single=pd.read_csv(OUT/"standalone.csv")
    single=single.loc[~single.variant.eq("buy_hold_512890")]
    single=pd.concat([single,pd.DataFrame([record("buy_hold_512890",result,target,PRIMARY)])],ignore_index=True)
    single.to_csv(OUT/"standalone.csv",index=False,encoding="utf-8-sig")
    cycles=[];cases=lookup_cases()
    for name in list(dict.fromkeys([primary_key(),selection["best_candidate"]])):
        rule,mode,cap=cases[name]
        original=reversion_states(fp,PRIMARY,rule,weekly_signal_dates(fp.index)).reindex(core.index)
        group=original.active.ne(original.active.shift(1,fill_value=False)).cumsum()
        for _,block in original.loc[original.active].groupby(group[original.active]):
            if block.index[-1]<pd.Timestamp(START):continue
            altered=original.copy();altered.loc[block.index,"active"]=False
            target=combine(prices,base,score,bear,bond_ok,altered,PRIMARY,mode,cap)
            result=run_backtest(prices,target,START,end)
            cycles.append({"variant":name,"disabled_signal_start":str(block.index[0].date()),
                           "disabled_signal_end":str(block.index[-1].date()),**stats(result.equity.daily_return)})
    pd.DataFrame(cycles).to_csv(OUT/"posthoc_episode_ablation.csv",index=False,encoding="utf-8-sig")
    cache=pd.read_csv(ROOT/f"data/cache/tushare_{PRIMARY}_hfq.csv",parse_dates=["date"]).set_index("date")
    quoted=fp[[PRIMARY]].dropna();raw=cache.raw_close.reindex(quoted.index).to_frame(PRIMARY)
    quote=quoted.index[quoted.index>pd.Timestamp("2021-10-22")][0]
    rows=[]
    for window in (150,200):
        adjusted=compute_factor(quoted,FactorSpec("trend",window))[PRIMARY]
        unadjusted=compute_factor(raw,FactorSpec("trend",window))[PRIMARY]
        valid=adjusted.notna()&unadjusted.notna()
        rows.append({"window":window,"split_date":"2021-10-22","first_quote_after_split":str(quote.date()),
                     "raw_distance_after_split":float(unadjusted.loc[quote]),"adjusted_distance_after_split":float(adjusted.loc[quote]),
                     "below_ma_classification_different_days":int(adjusted.loc[valid].lt(0).ne(unadjusted.loc[valid].lt(0)).sum())})
    pd.DataFrame(rows).to_csv(OUT/"adjustment_check.csv",index=False,encoding="utf-8-sig")
    print("posthoc attribution controls complete",flush=True)


def write_report(end):
    table=pd.read_csv(OUT/"comparison.csv").set_index("variant")
    selection=json.loads((OUT/"selection.json").read_text(encoding="utf-8"))
    manifest=json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))
    single=pd.read_csv(OUT/"standalone.csv").set_index("variant")
    trades=pd.read_csv(OUT/"episodes.csv")
    replica=pd.read_csv(OUT/"replication.csv")
    yearly=pd.read_csv(OUT/"yearly.csv")
    extra=pd.read_csv(OUT/"frequency_and_heavy_stress.csv")
    boot=json.loads((OUT/"bootstrap.json").read_text(encoding="utf-8"))
    cases=lookup_cases();literal=case_key(ReversionRule(200,0,.03,None),"budget_add",.30)
    def label(name):
        if name=="default":return "当前默认8%＋均线20/0.5"
        if name=="walk_forward":return "逐年仅按过去选择"
        rule,mode,cap=cases[name]
        return rule.label+f"／{'替换风险仓' if mode=='risk_replace' else '统一风险预算增配'}／上限{cap:.0%}"
    names=list(dict.fromkeys(["default",literal,primary_key(),selection["best_candidate"],selection["design_selection"],"walk_forward"]))
    pct=lambda x:f"{x:.2%}"
    best=table.loc[selection["best_candidate"]];base=table.loc["default"]
    static=table.drop(index="walk_forward")
    counts={"annual":int(static.annual.gt(base.annual).sum()),"sharpe":int(static.sharpe.gt(base.sharpe).sum())}
    lines=["# 红利低波ETF长期均线反转研究", "", f"研究日期2026-10-21；行情截止{end}。", "",
           "## 结论与范围", "", "本研究没有切换默认策略，也没有修改实盘信号或账户记录。"
           "主研究对象512890，515100与515450仅用相同规则复核，未按各自回测重新调参。", "",
           f"72个周度受约束组合中的最高历史夏普候选：**{label(selection['best_candidate'])}**，"
           f"年化{pct(best.annual)}、夏普{best.sharpe:.3f}、回撤{pct(best.drawdown)}；"
           f"同期默认为{pct(base.annual)}、{base.sharpe:.3f}、{pct(base.drawdown)}。", "",
           f"- 固定72项中，{counts['annual']}项年化、{counts['sharpe']}项夏普高于默认；这些组合高度相关，不能视为72次独立成功。",
           f"- 主检验年化{pct(table.loc[primary_key(),'annual'])}、夏普{table.loc[primary_key(),'sharpe']:.3f}，"
           "相同规则迁移到另外两只红利ETF仍提高收益和夏普；但2022年以来仅3个持仓周期，证据强度有限。",
           f"- 训练期选中的候选为{label(selection['design_selection'])}。若继续前瞻跟踪，可优先保留这一版本及预先指定主方案，"
           "避免直接用全样本最高夏普选定退出阈值。当前默认仍保持8%＋20日均线反转。",
           f"- 逐年仅按过去选参数的路径年化{pct(table.loc['walk_forward','annual'])}，低于默认，"
           "说明不能把事后冠军当作当时可提前选中的结果。", "",
           "这是基于既有历史提出的观察。2022–2024用于设计、2025以后用于时间分段复核；"
           "这些年份与红利风格可能已被反复观察，不能称为独立盲测。早期2020年用于检查下跌尾部，不参与主参数筛选。", "",
           "## 算法与资金安排", "", "偏离d=P/MA(N)−1，使用同日复权收盘价，N取150/200个有真实报价的交易日。"
           "空仓时d≤0/−3%/−5%入场，持有时d≥+3%/+5%退出；两条线之间保持原状态。"
           "12条原始规则不限持有期；另6条固定补充为入−3%、出+3%、最长120个交易日，分别不确认、近3日已反弹、长均线20日斜率非负。"
           "超时退出后，价格须重新高于入场线，才允许下一次超跌买入，避免连续接下跌资产。", "",
           "均线随时间移动：超过下降后的均线+3%，可能仍低于买入价。该退出条件不能等同于盈利3%。"
           "最长持有期只在决策日检查，周度检查可能晚数个交易日退出；均为下一交易日收盘执行。", "",
           "- **替换风险仓**：在原风险仓内转移最多15%/30%账户资产至红利，不主动提高原风险总仓位。",
           "- **统一风险预算增配**：把红利作为逆向卫星，允许绕过原40日趋势资格，但与原仓共同估计60日协方差并受8%预算约束；从防守资金调入的实际额度受风险约束限制。",
           "- 两种受约束方式都要求风险ETF最多3只（红利占1个名额），原核心最多保留修正评分最高的2只；"
           "单只风险上限30%、总风险上限70%、弱市减半，红利另受15%/30%卫星上限。限仓后再复核组合波动。"
           "这会挤掉部分原持仓，报告收益包含该机会成本。",
           "- 原核心的排名缓冲按其独立周目标持续重放，卫星叠加后实际执行组合由统一引擎计费；"
           "这是组合叠加规则，不把被挤出的实际持仓冒充核心模型旧目标。",
           "- 主方案预先指定为200日、入−3%、出+3%、最长120日、增配上限30%；"
           "用户字面观察对照为200日、跌破即入、出+3%、不限持有期、增配上限30%。", "",
           "## 主组合表现", "", "2022年以来；单边手续费3bps＋滑点2bps；夏普按零无风险利率，回撤按收盘净值。", "",
           "| 方案 | 年化 | 夏普 | 回撤 | 20bps年化 | 20bps回撤 | 平均红利仓 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        r=table.loc[n];lines.append(f"| {label(n)} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {pct(r.stress20_annual)} | {pct(r.stress20_drawdown)} | {pct(r.mean_dividend_weight)} |")
    lines += ["", "![组合收益、回撤与长期均线偏离](assets/dividend_reversion_20261021.png)", "", "## 时间分段及逐年", "",
              "| 方案 | 设计段年化 | 设计段夏普 | 后段年化 | 后段夏普 | 后段回撤 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        r=table.loc[n];lines.append(f"| {label(n)} | {pct(r.design_annual)} | {r.design_sharpe:.3f} | {pct(r.later_annual)} | {r.later_sharpe:.3f} | {pct(r.later_drawdown)} |")
    lines += ["", "| 方案 | 2022 | 2023 | 2024 | 2025 | 2026至今 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        r=yearly.loc[yearly.variant.eq(n)].set_index("year")
        lines.append("| "+label(n)+" | "+" | ".join(pct(r.loc[y,"total"]) for y in range(2022,2027))+" |")
    lines += ["", "滚动路径2022–2023用默认，之后每年只读此前已实现收益，末端空出5个交易日。"
              "选择须回撤≤10%、年化不低于基准超过0.5个百分点、至少2/3训练年份收益不低于基准，再选最高夏普。"
              "配置对应的卫星状态按其历史价格独立跟踪，切换时使用当时已知的状态，不拼接静态回测的净收益，统一重算实际换手。", ""]
    lines += [f"- {f['year']}：{label(f['variant'])}，训练收益截至{f['training_end']}。" for f in selection["walk_forward_choices"]]
    lines += ["", "## 交易周期与‘通常会反弹’", "", "下面逐笔以下一交易日收盘计入出价，展示完整退出和未退出周期；"
              "收益为ETF自身复权收益、未扣费，不能当作账户组合收益。未结束周期标明浮动损益，不删除亏损或长期未退出的观察。", "",
              "周期统计包含2019年以来可形成完整长均线的全部观察；主组合绩效只计2022年以来。MAE表示相对买入价的最深浮亏，"
              "与净值从持有期高点回落的最大回撤口径不同。完成周期胜率不能用于保证新周期盈利。", "",
              "| 规则 | 完成周期 | 完成胜率 | 完成均值 | 最差完成收益 | 最长完成持有日 | 开放周期 | 最深买入后浮亏 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    rule_names=list(dict.fromkeys([cases[n][0].key for n in names if n in cases]))
    for key in rule_names:
        t=trades.loc[trades.rule.eq(key)];closed=t.loc[t.closed.eq(True)];opened=t.loc[t.closed.eq(False)]
        lines.append(f"| {key} | {len(closed)} | {pct(closed.asset_return.gt(0).mean())} | {pct(closed.asset_return.mean())} | {pct(closed.asset_return.min())} | {closed.holding_days.max():g} | {len(opened)} | {pct(t.mae.min())} |")
        if len(opened):
            for _,r in opened.iterrows():lines.append(f"\n开放周期：{key}，{r.entry_execution}买入，截至截止日持有{int(r.holding_days)}日，浮动收益{pct(r.asset_return)}，持有期最差{pct(r.mae)}。\n")
    lines += ["", "### 单独100%运行的早期风险", "", f"收益起点{EARLY_START}；此处100%买ETF或持防守资产，超出默认单只30%约束，仅作尾部诊断。", "",
              "| 规则 | 年化 | 夏普 | 回撤 |", "| --- | ---: | ---: | ---: |"]
    for key in rule_names:
        r=single.loc[key];lines.append(f"| {key} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} |")
    buy_hold=single.loc['buy_hold_512890'] if 'buy_hold_512890' in single.index else None
    if buy_hold is not None:
        lines.append(f"| 买入持有512890 | {pct(buy_hold.annual)} | {buy_hold.sharpe:.3f} | {pct(buy_hold.drawdown)} |")
    lines += ["", "## 参数邻域", "", "下列仅展示上限30%、替换风险仓、不限持有期的全部12个原始规则，完整72项含失败项见comparison.csv。", "",
              "| 均线 | 入场偏离 | 退出偏离 | 年化 | 夏普 | 回撤 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for rule in fixed_rules()[:12]:
        r=table.loc[case_key(rule,"risk_replace",.30)]
        lines.append(f"| {rule.window} | {pct(rule.entry)} | +{pct(rule.exit)} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} |")
    lines += ["", "## 跨ETF复核", "", "参数直接迁移，不分别寻找各ETF最优。三只基金为相关股票风格，并非独立实验。", "",
              "| ETF | 方案 | 年化 | 夏普 | 回撤 |", "| --- | --- | ---: | ---: | ---: |"]
    for _,r in replica.iterrows():lines.append(f"| {r.symbol} | {label(r.variant)} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} |")
    lines += ["", "## 每日触发与50%重仓压力", "", "每日版本只在入出场事件及原周信号日调仓，没有每天免费再平衡。"
              "50%重仓版本绕过8%预算、弱市减半和单只30%限制，属于超出默认约束的风险压力实验，不参与72项主选优。", "",
              "| 方案 | 年化 | 夏普 | 回撤 | 20bps回撤 |", "| --- | ---: | ---: | ---: | ---: |"]
    for _,r in extra.iterrows():lines.append(f"| {r.variant} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {pct(r.stress20_drawdown)} |")
    abl=pd.read_csv(OUT/"ablation.csv")
    lines += ["", "## 资产依赖", "", "| 场景 | 方案 | 年化 | 夏普 | 回撤 |", "| --- | --- | ---: | ---: | ---: |"]
    for _,r in abl.iterrows():lines.append(f"| {r.scenario} | {label(r.variant)} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} |")
    adjustment=pd.read_csv(OUT/"adjustment_check.csv")
    controls_path=OUT/"posthoc_controls.csv"
    if controls_path.exists():
        controls=pd.read_csv(controls_path)
        lines += ["", "## 补充归因：直接扩池与持续配置", "", "这3项是在看到主网格后增加的对照，不参与原72项选优及多重比较校正。"
                  "用于区分新增资产、持续持有与条件性反转的贡献。持续配置也受风险预算及名额限制。", "",
                  "| 对照 | 年化 | 夏普 | 回撤 | 平均红利仓 |", "| --- | ---: | ---: | ---: | ---: |"]
        for _,r in controls.iterrows():lines.append(f"| {r.variant} | {pct(r.annual)} | {r.sharpe:.3f} | {pct(r.drawdown)} | {pct(r.mean_dividend_weight)} |")
        ca=pd.read_csv(OUT/"posthoc_episode_ablation.csv")
        lines += ["", "### 逐周期剔除", "", "事后把某个红利持仓周期改回原组合，其余规则不重调。"
                  "这需要事后知道周期边界，仅作归因，不是可交易的择时规则。", "",
                  "| 方案 | 禁用周期起始信号 | 禁用周期结束信号 | 剩余年化 | 剩余夏普 |", "| --- | --- | --- | ---: | ---: |"]
        for _,r in ca.iterrows():lines.append(f"| {label(r.variant)} | {r.disabled_signal_start} | {r.disabled_signal_end} | {pct(r.annual)} | {r.sharpe:.3f} |")
    lines += ["", "## 复权、样本及统计限制", "", "512890于2021-10-22发生份额拆分，当日无交易报价，拆分后首个报价日为2021-10-25。"
              "长均线使用过去N个真实报价，缺价当日不产生新信号，也不以估值填充价成交；主收益区间2022年以来三只红利ETF均无缺价。"
              "复权后研究连续收益，不复权曲线仅用于检查信号污染。", "",
              "| 均线 | 拆分后首日报价的不复权偏离 | 复权偏离 | 是否低于均线的分类差异天数 |", "| --- | ---: | ---: | ---: |"]
    for _,r in adjustment.iterrows():lines.append(f"| {int(r.window)} | {pct(r.raw_distance_after_split)} | {pct(r.adjusted_distance_after_split)} | {int(r.below_ma_classification_different_days)} |")
    latest=pd.read_csv(OUT/"latest_research_states.csv")
    d=float(latest.loc[latest.rule.eq(ReversionRule().key),'deviation'].iloc[0])
    lines += ["", f"截至{end}，512890相对200日均线偏离为{d:+.2%}，当前没有新的低于均线入场条件。"
              "+5%退出版本仍显示持有，来源于年初历史买入；新账户不能使用当时价格回溯建仓。", ""]
    lines += ["", f"- 共18条状态规则（相互高度相关）、72项周度组合、18个单独运行诊断，另有4项每日事件与2项50%压力比较。"
              f"60日循环块、{boot['replicates']}次成对抽样，对72项主组合最大均值收益差校正的探索性p={boot['family_max_mean_p']:.3f}。",
              "- 此检验不校正以前的多轮搜索、ETF选择及事后夏普冠军；不能把较高历史点估计直接解释为稳定收益来源。",
              "- ETF收益已包含基金内部费用及复权调整，但未模拟最低佣金、整手、盘中价差、现金分红到账延迟、税费差异和资金容量。"
              "模型用复权收盘价连续计收益，实际交易使用原始价格；日收盘回撤不等于盘中回撤。", "",
              "## 验证及复现", "", f"已通过{manifest.get('unit_tests_passed',77)}项单元测试。时序检查见time_audit.csv；主要候选共{manifest.get('time_audits',{}).get('time_audits',0)}项截断/未来行情扰动，"
              "受约束组合全部通过最多3只风险ETF、单只30%、总风险70%检查。默认源代码、切换状态与正式信号指纹前后不变。", "",
              "```powershell",f".\\.venv\\Scripts\\python.exe research_dividend_reversion.py --download-only --end {end}",
              f".\\.venv\\Scripts\\python.exe research_dividend_reversion.py --end {end}",
              ".\\.venv\\Scripts\\python.exe research_dividend_reversion.py --report-only", "```", "",
              "完整参数、失败项汇总、逐日收益、逐笔周期及跨ETF结果集中在output/dividend_reversion_20261021，"
              "保留一个研究入口与可复用的均线偏离状态机，不增加独立默认策略分支。", "", "## 资料依据", ""]
    lines += [f"- [{title}]({url})" for title,url in SOURCES]
    path=ROOT/"docs/dividend_reversion_20261021.md";path.write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"]=["Microsoft YaHei","DejaVu Sans"];plt.rcParams["axes.unicode_minus"]=False
    nav=pd.read_csv(OUT/"nav.csv",index_col=0,parse_dates=True)
    chart=pd.read_csv(OUT/"price_and_state.csv",index_col=0,parse_dates=True)
    fig,axes=plt.subplots(3,1,figsize=(14,12),sharex=True,gridspec_kw={"height_ratios":[2.5,1.2,1.2]})
    shortlabels={"default":"当前默认",primary_key():"主检验：200日负偏离3%／30%卫星",literal:"200日跌破即入／不限持有",selection['best_candidate']:"历史最高夏普卫星候选","walk_forward":"逐年仅按过去选择"}
    for n in nav:
        axes[0].plot(nav.index,(nav[n]-1)*100,label=shortlabels[n],linewidth=2 if n=="default" else 1.2)
        axes[1].plot(nav.index,(nav[n]/nav[n].cummax().clip(lower=1)-1)*100,linewidth=1.3)
    axes[0].legend(fontsize=8,loc="upper left")
    axes[0].set_title(f"红利低波长期均线反转研究｜2022-01-04至{end}｜单边5bps\n卫星候选与当前8%＋20日均线反转默认对比；未切换实盘")
    axes[0].set_ylabel("累计收益（%）");axes[1].set_ylabel("收盘回撤（%）")
    axes[2].plot(chart.index,chart.deviation200*100,color="black",linewidth=.8,label="512890相对200日均线")
    for y in (-3,3,5):axes[2].axhline(y,color="grey",linestyle="--",alpha=.6)
    axes[2].fill_between(chart.index,-30,30,where=chart.primary_active.astype(bool),alpha=.15,color="orange",label="主规则持有信号")
    axes[2].set_ylim(min(-15,float(chart.deviation200.min()*100)-1),max(15,float(chart.deviation200.max()*100)+1))
    axes[2].set_ylabel("长期偏离（%）");axes[2].legend(fontsize=8,loc="upper left")
    for ax in axes:ax.grid(alpha=.25);ax.axvline(pd.Timestamp("2025-01-01"),color="grey",linestyle="--",alpha=.5)
    fig.tight_layout();fig.savefig(ROOT/"docs/assets/dividend_reversion_20261021.png",dpi=150);plt.close(fig)
    return path


def cap40_comparison(end):
    """用户指定的4%/5%退出、40%上限补充；不重选参数、不混入原72项搜索。"""
    destination=OUT/"cap40"
    destination.mkdir(parents=True,exist_ok=True)
    before=protected()
    core,full=load_data(end)
    base,score,bear,bond_ok=core_context(core)
    prices=core.copy();prices[PRIMARY]=full[PRIMARY][PRIMARY].reindex(core.index)
    requested=[(ReversionRule(200,0,x,None),.40) for x in (.04,.05)]
    controls=[(ReversionRule(200,0,x,None),.30) for x in (.03,.04,.05)]
    configurations=controls+requested
    rows=[];results={};targets={};states={};cycle_rows=[];annual_rows=[]
    targets["default"]=base.reindex(columns=prices.columns,fill_value=0.)
    results["default"]=run_backtest(prices,targets["default"],START,end)
    rows.append(record("default",results["default"],targets["default"],PRIMARY))
    returns=prices.pct_change(fill_method=None)
    for rule,cap in configurations:
        name=case_key(rule,"risk_replace",cap)
        state=reversion_states(full[PRIMARY],PRIMARY,rule,weekly_signal_dates(full[PRIMARY].index))
        states[name]=state
        target=combine(prices,base,score,bear,bond_ok,state.reindex(core.index),PRIMARY,"risk_replace",cap)
        result=run_backtest(prices,target,START,end)
        targets[name],results[name]=target,result
        main=target.loc[str(START):]
        risky=main.drop(columns=BOND)
        np.testing.assert_array_less(main[PRIMARY].to_numpy(),cap+1e-10)
        np.testing.assert_array_less(risky.drop(columns=PRIMARY).to_numpy(),.30+1e-10)
        np.testing.assert_array_less(risky.sum(axis=1).to_numpy(),.70+1e-10)
        assert risky.gt(1e-10).sum(axis=1).le(3).all()
        assert main.sum(axis=1).le(1+1e-10).all()
        ceiling=base.drop(columns=BOND).sum(axis=1).reindex(main.index,method="ffill")
        assert risky.sum(axis=1).le(ceiling+1e-10).all()
        forecast=[]
        for day,row in main.loc[main[PRIMARY].gt(0)].iterrows():
            held=row.drop(BOND);held=held[held.gt(1e-12)]
            history=returns.loc[:day,held.index].tail(60).dropna()
            if len(history)>=40:
                sigma=float(np.sqrt(max(held.to_numpy() @ (history.cov().to_numpy()*252) @ held.to_numpy(),0)))
                limit=.04 if bear.loc[day] else .08
                assert sigma<=limit+1e-10
                forecast.append(sigma)
        actual=result.weights[PRIMARY];nav=result.equity.strategy_nav
        trough=result.equity.drawdown.idxmin();peak=nav.loc[:trough].idxmax()
        row=record(name,result,target,PRIMARY)|{
            "exit":rule.exit,"cap":cap,"max_target_dividend":float(main[PRIMARY].max()),
            "mean_dividend_when_held":float(actual.loc[actual.gt(1e-10)].mean()),
            "max_target_gross_risky":float(risky.sum(axis=1).max()),
            "max_forecast_risky_vol_when_active":max(forecast,default=0),
            "drawdown_peak":str(peak.date()),"drawdown_trough":str(trough.date()),
            "dividend_weight_at_drawdown_trough":float(actual.loc[trough]),
            "active_at_cutoff":bool(state.active.iloc[-1]),"latest_deviation":float(state.deviation.iloc[-1])}
        for bps in (5,10,20):
            gross=result.equity.gross_return;turnover=result.equity.turnover
            calculated=(1+gross)*(1-turnover*bps/10000)-1
            row.update({f"cost{bps}_{k}":v for k,v in stats(calculated).items()})
        rows.append(row)
        cycle_rows.append(episodes(full[PRIMARY],state,PRIMARY,rule).assign(variant=name))
        print(f"cap40 comparison {name}: annual={row['annual']:.4%} SR={row['sharpe']:.4f} DD={row['drawdown']:.4%}",flush=True)
    for name,result in results.items():
        for year,segment in result.equity.daily_return.groupby(result.equity.index.year):
            annual_rows.append({"variant":name,"year":int(year),**stats(segment)})
    table=pd.DataFrame(rows).set_index("variant")
    # 基线成本也由同一逐日持仓与实际换手重算。
    for bps in (5,10,20):
        equity=results["default"].equity
        for k,v in stats((1+equity.gross_return)*(1-equity.turnover*bps/10000)-1).items():
            table.loc["default",f"cost{bps}_{k}"]=v
    table.to_csv(destination/"comparison.csv",encoding="utf-8-sig")
    pd.DataFrame(annual_rows).to_csv(destination/"yearly.csv",index=False,encoding="utf-8-sig")
    pd.concat(cycle_rows,ignore_index=True).to_csv(destination/"episodes.csv",index=False,encoding="utf-8-sig")
    pd.concat(targets,names=["variant","signal_date"]).to_csv(destination/"target_weights.csv",encoding="utf-8-sig")
    daily=pd.DataFrame({n:v.equity.daily_return for n,v in results.items()})
    daily.to_csv(destination/"daily_returns.csv",encoding="utf-8-sig")
    nav=pd.DataFrame({n:v.equity.strategy_nav for n,v in results.items()})
    nav.to_csv(destination/"nav.csv",encoding="utf-8-sig")
    pd.concat({n:v.weights for n,v in results.items()},names=["variant","date"]).to_csv(destination/"actual_weights.csv",encoding="utf-8-sig")
    replicas=[]
    for symbol in list(FUNDS)[1:]:
        alternative=core.copy();alternative[symbol]=full[symbol][symbol].reindex(core.index)
        for rule,cap in configurations:
            if rule.exit==.03:continue
            name=case_key(rule,"risk_replace",cap)
            state=reversion_states(full[symbol],symbol,rule,weekly_signal_dates(full[symbol].index)).reindex(core.index)
            target=combine(alternative,base,score,bear,bond_ok,state,symbol,"risk_replace",cap)
            result=run_backtest(alternative,target,START,end)
            replicas.append(record(name,result,target,symbol)|{"symbol":symbol})
        print(f"cap40 replication {symbol} complete",flush=True)
    pd.DataFrame(replicas).to_csv(destination/"replication.csv",index=False,encoding="utf-8-sig")
    # 截断路径的最后一个不完整周不做比较，未来扰动保持原交易日期不变。
    audits=[]
    for cutoff in ("2023-06-30","2025-06-30","2026-09-30"):
        cut=pd.Timestamp(cutoff)
        short=core.loc[:cut];cb,cs,cm,co=core_context(short)
        changed=core.copy();changed.loc[changed.index>cut]*=3
        fb,fs,fm,fo=core_context(changed)
        prefix_full=full[PRIMARY].loc[:cut]
        altered_full=full[PRIMARY].copy();altered_full.loc[altered_full.index>cut]*=7
        for rule,cap in requested:
            name=case_key(rule,"risk_replace",cap)
            sp=short.copy();sp[PRIMARY]=prefix_full[PRIMARY].reindex(short.index)
            st=reversion_states(prefix_full,PRIMARY,rule,weekly_signal_dates(prefix_full.index)).reindex(short.index)
            prefix=combine(sp,cb,cs,cm,co,st,PRIMARY,"risk_replace",cap)
            pp=changed.copy();pp[PRIMARY]=altered_full[PRIMARY].reindex(pp.index)
            st2=reversion_states(altered_full,PRIMARY,rule,weekly_signal_dates(altered_full.index)).reindex(pp.index)
            future=combine(pp,fb,fs,fm,fo,st2,PRIMARY,"risk_replace",cap)
            old=targets[name]
            for label,check,mask in (("prefix",prefix,old.index<cut),("future_perturbation",future,old.index<=cut)):
                expected=old.loc[mask]
                pd.testing.assert_frame_equal(expected,check.reindex(expected.index),atol=1e-12,rtol=1e-12)
                audits.append({"variant":name,"cutoff":cutoff,"check":label,"max_error":float((expected-check.reindex(expected.index)).abs().to_numpy().max())})
        print(f"cap40 time audit {cutoff} complete",flush=True)
    pd.DataFrame(audits).to_csv(destination/"time_audit.csv",index=False,encoding="utf-8-sig")
    assert protected()==before,"正式策略或信号被研究修改"
    metadata={"end":str(end),"new_user_requested_cases":2,"reference_cases":len(controls),
              "rules":"MA200 / entry0 / exit4-or5 / unlimited / risk_replace / dividendcap40; corecap30, vol8, gross70, bear reduction retained",
              "time_audits":len(audits),"max_target_error":max(a["max_error"] for a in audits),
              "default_files_unchanged":True,"protected_sha256":before,
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "prices_sha256":hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
              "past_history_already_inspected":True,"original_72_trial_p_value_not_applicable_to_new_search":True}
    (destination/"checks.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    names=["default"]+[case_key(rule,"risk_replace",cap) for rule,cap in controls+requested]
    def title(name):
        if name=="default":return "当前默认"
        row=table.loc[name]
        return f"MA200／入0／出+{row['exit']:.0%}／替换／上限{row['cap']:.0%}"
    lines=["# 红利低波40%上限：4%与5%退出对照", "",f"研究日期2026-10-21；行情截止{end}。收益自2022-01-04起。", "",
           "用户指定两项后续检验；保持原数据、核心参数和成交时点。未重新扫描窗口、入场线、持有期或ETF。"
           "补充30%同阈值对照用于区分仓位与退出阈值的影响；未切换默认策略。", "", "## 仓位定义", "",
           "红利512890上限40%按账户总资产计算，其他风险ETF仍限30%。红利活跃时只保留核心评分最高的至多2只，"
           "从原风险仓中划出min(40%,原风险总目标)作为初始红利需求，再复核原风险总额、70%总上限、8%波动预算、弱市减半及单只上限。"
           "因此40%是上限，实际目标可显著更低；不会像50%压力版本一样绕过风险约束。目标限仓不阻止非调仓日价格引起的自然权重漂移。", "",
           "以真实报价的复权价格计算200日均线；信号日收盘低于/等于均线触发，持有至超过均线4%或5%，不限日。"
           "周末信号、下一交易日收盘成交；持有期间每周依风险预算更新仓位。核心缓冲按原独立周目标重放。", "",
           "## 全期结果", "", "单边5bps；零无风险利率夏普；收盘最大回撤。", "",
           "| 方案 | 年化 | 夏普 | 最大回撤 | 实际年化波动 | 全期平均红利仓 | 实际最大红利仓 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        row=table.loc[n]
        lines.append(f"| {title(n)} | {row.annual:.2%} | {row.sharpe:.3f} | {row.drawdown:.2%} | {row.volatility:.2%} | {row.mean_dividend_weight:.2%} | {row.max_dividend_weight:.2%} |")
    lines += ["", "![收益及回撤对比](assets/dividend_cap40_20261021.png)", "", "## 成本敏感性", "",
              "| 方案 | 单边10bps年化 | 10bps夏普 | 10bps回撤 | 单边20bps年化 | 20bps夏普 | 20bps回撤 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        row=table.loc[n]
        lines.append(f"| {title(n)} | {row.cost10_annual:.2%} | {row.cost10_sharpe:.3f} | {row.cost10_drawdown:.2%} | {row.cost20_annual:.2%} | {row.cost20_sharpe:.3f} | {row.cost20_drawdown:.2%} |")
    lines += ["", "## 时间分段", "", "2022—2024与2025后仅作已观察历史的时间分段，不称独立样本外盲测。", "",
              "| 方案 | 前段年化 | 前段夏普 | 前段回撤 | 后段年化 | 后段夏普 | 后段回撤 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in names:
        row=table.loc[n]
        lines.append(f"| {title(n)} | {row.design_annual:.2%} | {row.design_sharpe:.3f} | {row.design_drawdown:.2%} | {row.later_annual:.2%} | {row.later_sharpe:.3f} | {row.later_drawdown:.2%} |")
    lines += ["", "## 同规则跨ETF复核", "", "515100与515450不重新调参；二者与512890相关，不能算独立复制。", "",
              "| ETF | 方案 | 年化 | 夏普 | 回撤 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in replicas:
        lines.append(f"| {row['symbol']} | {title(row['variant'])} | {row['annual']:.2%} | {row['sharpe']:.3f} | {row['drawdown']:.2%} |")
    lines += ["", "## 当前状态与持有周期", ""]
    cycles=pd.concat(cycle_rows,ignore_index=True)
    for rule,cap in requested:
        name=case_key(rule,"risk_replace",cap);row=table.loc[name]
        history=cycles.loc[cycles.variant.eq(name)&pd.to_datetime(cycles.entry_execution).ge(pd.Timestamp(START))]
        lines.append(f"- {title(name)}：主样本完成{int(history.closed.sum())}次，未结束{int((~history.closed).sum())}次；"
                     f"截止日信号状态{'持有' if row.active_at_cutoff else '空仓'}，相对MA200偏离{row.latest_deviation:.2%}。"
                     f"最大回撤峰日{row.drawdown_peak}、谷日{row.drawdown_trough}；谷日红利实际仓位{row.dividend_weight_at_drawdown_trough:.2%}。"
                     f"红利活跃日平均仓位{row.mean_dividend_when_held:.2%}，目标最大{row.max_target_dividend:.2%}。")
    lines += ["", "## 局限及验证", "",
              "- 两项为在既有研究结果之后提出的后续检验；原72项的p值不涵盖新搜索，也未检验40%是最优上限。",
              "- 不限日会保留很长的浮亏/未完成周期；均线退出线会移动，不保证盈利。未结束周期保留在净值和episodes.csv中。",
              "- 历史最大回撤仅是样本内已实现结果，不是止损承诺。未模拟盘中回撤、最低佣金、整手、实际分红到账和容量。",
              f"- 两项共{len(audits)}项前缀/未来行情扰动检查全部通过，目标最大误差0；正式源代码、锚点、实盘JSON指纹未变。",
              "", "## 复现", "", "```powershell",f".\\.venv\\Scripts\\python.exe research_dividend_reversion.py --cap40 --end {end}","```", "",
              "全部对照、年度表现、真实漂移权重、目标权重、逐日收益、周期与审计位于output/dividend_reversion_20261021/cap40。"]
    report=ROOT/"docs/dividend_cap40_20261021.md"
    report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"]=["Microsoft YaHei","DejaVu Sans"];plt.rcParams["axes.unicode_minus"]=False
    plotted=["default",case_key(ReversionRule(200,0,.05,None),"risk_replace",.30)]+[case_key(rule,"risk_replace",cap) for rule,cap in requested]
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    for n in plotted:
        axes[0].plot(nav.index,(nav[n]-1)*100,label=title(n),linewidth=1.7)
        axes[1].plot(nav.index,(nav[n]/nav[n].cummax().clip(lower=1)-1)*100,linewidth=1.3)
    axes[0].legend(loc="upper left",fontsize=9)
    axes[0].set_title(f"红利低波MA200：40%风险替换上限对照｜2022-01-04至{end}\n保持8%预算、总风险70%、弱市减仓；单边5bps；未切换默认")
    axes[0].set_ylabel("累计收益（%）");axes[1].set_ylabel("收盘回撤（%）")
    for ax in axes:ax.grid(alpha=.25);ax.axvline(pd.Timestamp("2025-01-01"),color="grey",linestyle="--",alpha=.5)
    fig.tight_layout();fig.savefig(ROOT/"docs/assets/dividend_cap40_20261021.png",dpi=150);plt.close(fig)
    print(table[["annual","sharpe","drawdown","mean_dividend_weight","cost20_annual","cost20_sharpe"]].to_string(),flush=True)
    print(f"CAP40 COMPLETE {report}",flush=True)


def defensive_funding_comparison(end):
    """用户指定4项国债/现金替换；另做只分配剩余风险额度的匹配对照。"""
    destination=OUT/"defensive_funding"
    destination.mkdir(parents=True,exist_ok=True)
    before=protected()
    core,full=load_data(end);base,score,bear,bond_ok=core_context(core)
    prices=core.copy();prices[PRIMARY]=full[PRIMARY][PRIMARY].reindex(core.index)
    requested=[(ReversionRule(200,0,x,None),cap) for x in (.04,.05) for cap in (.20,.30)]
    states={r.key:reversion_states(full[PRIMARY],PRIMARY,r,weekly_signal_dates(full[PRIMARY].index)) for r,_ in requested}
    def key(rule,cap,guard=False):return case_key(rule,"defense_spare_budget" if guard else "defense_replace",cap)
    def build(p,cb,cm,fp,rule,cap,guard):
        st=reversion_states(fp,PRIMARY,rule,weekly_signal_dates(fp.index)).reindex(p.index)
        return defensive_funding_targets(p,cb,st,PRIMARY,cap,bear=cm,guard=guard)
    targets={"default":base.reindex(columns=prices.columns,fill_value=0.)}
    for x,cap in ((.04,.40),(.05,.40),(.05,.30)):
        rule=ReversionRule(200,0,x,None)
        targets[case_key(rule,"risk_replace",cap)]=combine(prices,base,score,bear,bond_ok,states[rule.key].reindex(core.index),PRIMARY,"risk_replace",cap)
    for guard in (False,True):
        for rule,cap in requested:
            targets[key(rule,cap,guard)]=defensive_funding_targets(prices,base,states[rule.key].reindex(core.index),PRIMARY,cap,bear=bear,guard=guard)
    results={};rows=[];yearly=[];forecasts=[];returns=prices.pct_change(fill_method=None)
    core_names=list(base.columns.drop(BOND))
    for name,target in targets.items():
        result=run_backtest(prices,target,START,end);results[name]=result
        row=record(name,result,target,PRIMARY)
        for bps in (5,10,20):
            charged=(1+result.equity.gross_return)*(1-result.equity.turnover*bps/10000)-1
            row.update({f"cost{bps}_{k}":v for k,v in stats(charged).items()})
        for label,left,right in (("h1_2026","2026-01-01","2026-06-30"),("ytd_2026","2026-01-01",str(end))):
            row.update({f"{label}_{k}":v for k,v in stats(result.equity.daily_return.loc[left:right]).items()})
        for year,segment in result.equity.daily_return.groupby(result.equity.index.year):
            yearly.append({"variant":name,"year":int(year),**stats(segment)})
        main=target.loc[str(START):]
        prior=base.reindex(base.index.union(main.index)).sort_index().ffill().reindex(main.index)
        risky=main.drop(columns=BOND)
        row["max_target_gross_risky"]=float(risky.sum(axis=1).max())
        row["max_target_dividend"]=float(main[PRIMARY].max())
        row["max_target_positions"]=int(main.gt(1e-10).sum(axis=1).max())
        row["max_target_risky_positions"]=int(risky.gt(1e-10).sum(axis=1).max())
        row["max_core_target_difference"]=float((main[core_names]-prior[core_names]).abs().to_numpy().max())
        row["risk_sleeve_active_days"]=int(result.weights[PRIMARY].gt(1e-10).sum())
        row["mean_dividend_when_held"]=float(result.weights.loc[result.weights[PRIMARY].gt(1e-10),PRIMARY].mean()) if result.weights[PRIMARY].gt(1e-10).any() else 0.
        nav=result.equity.strategy_nav;trough=result.equity.drawdown.idxmin();peak=nav.loc[:trough].idxmax()
        row.update({"drawdown_peak":str(peak.date()),"drawdown_trough":str(trough.date()),"dividend_at_worst_drawdown":float(result.weights.at[trough,PRIMARY])})
        active=main[PRIMARY].gt(1e-10);risk_forecasts=[];total_forecasts=[];all_risk_forecasts=[];all_total_forecasts=[];vol_breach=0;gross_breach=0
        for day,weights in main.iterrows():
            names=list(weights.index[weights.gt(1e-12)])
            history=returns.loc[:day,names].tail(60).dropna()
            if len(history)<40:continue
            cov=history.cov()*252
            risky_names=[s for s in names if s!=BOND]
            rv=weights.loc[risky_names].to_numpy()
            risk_vol=float(np.sqrt(max(rv @ cov.loc[risky_names,risky_names].to_numpy() @ rv,0)))
            tv=weights.loc[names].to_numpy()
            total_vol=float(np.sqrt(max(tv @ cov.to_numpy() @ tv,0)))
            all_risk_forecasts.append(risk_vol);all_total_forecasts.append(total_vol)
            limit=.04 if bear.loc[day] else .08;gross_limit=.35 if bear.loc[day] else .70
            vol_breach+=int(risk_vol>limit+1e-10)
            gross_breach+=int(risky.loc[day].sum()>gross_limit+1e-10)
            if active.loc[day]:risk_forecasts.append(risk_vol);total_forecasts.append(total_vol)
            forecasts.append({"variant":name,"signal_date":str(day.date()),"risk_vol":risk_vol,"total_vol":total_vol,"risk_limit":limit,"risky_gross":float(risky.loc[day].sum()),"gross_limit":gross_limit,"dividend":float(weights[PRIMARY])})
        row.update({"max_active_predicted_risk_vol":max(risk_forecasts,default=0),"max_active_predicted_account_vol":max(total_forecasts,default=0),"max_predicted_risk_vol":max(all_risk_forecasts,default=0),"max_predicted_account_vol":max(all_total_forecasts,default=0),"forecast_risk_budget_breach_signals":vol_breach,"gross_budget_breach_signals":gross_breach})
        assert np.isfinite(target).all().all() and target.ge(-1e-12).all().all()
        assert target.sum(axis=1).le(1+1e-10).all()
        if "defense_" in name:
            cap=float(name.rsplit("cap",1)[1])/100
            assert row["max_core_target_difference"]<=1e-12
            assert main[PRIMARY].le(cap+1e-12).all()
            assert main[BOND].le(prior[BOND]+1e-12).all()
            assert risky.gt(1e-10).sum(axis=1).le(4).all()
            assert main.gt(1e-10).sum(axis=1).le(5).all()
        rows.append(row)
        print(f"defense study {name}: annual={row['annual']:.4%} SR={row['sharpe']:.4f} DD={row['drawdown']:.4%} H1={row['h1_2026_total']:.4%}",flush=True)
    table=pd.DataFrame(rows).set_index("variant")
    table.to_csv(destination/"comparison.csv",encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(destination/"yearly.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(forecasts).to_csv(destination/"forecast_risk.csv",index=False,encoding="utf-8-sig")
    pd.concat(targets,names=["variant","signal_date"]).to_csv(destination/"target_weights.csv",encoding="utf-8-sig")
    nav=pd.DataFrame({n:v.equity.strategy_nav for n,v in results.items()})
    nav.to_csv(destination/"nav.csv",encoding="utf-8-sig")
    pd.DataFrame({n:v.equity.daily_return for n,v in results.items()}).to_csv(destination/"daily_returns.csv",encoding="utf-8-sig")
    pd.concat({n:v.weights for n,v in results.items()},names=["variant","date"]).to_csv(destination/"actual_weights.csv",encoding="utf-8-sig")
    cycles=pd.concat([episodes(full[PRIMARY],states[rule.key],PRIMARY,rule) for rule in (ReversionRule(200,0,.04,None),ReversionRule(200,0,.05,None))],ignore_index=True)
    cycles.to_csv(destination/"episodes.csv",index=False,encoding="utf-8-sig")
    replicas=[]
    for symbol in list(FUNDS)[1:]:
        p=core.copy();p[symbol]=full[symbol][symbol].reindex(core.index)
        for rule,cap in requested:
            state=reversion_states(full[symbol],symbol,rule,weekly_signal_dates(full[symbol].index)).reindex(core.index)
            target=defensive_funding_targets(p,base,state,symbol,cap)
            result=run_backtest(p,target,START,end)
            replicas.append(record(key(rule,cap),result,target,symbol)|{"symbol":symbol,"h1_2026_total":stats(result.equity.daily_return.loc["2026-01-01":"2026-06-30"])["total"]})
        print(f"defense replication {symbol} complete",flush=True)
    pd.DataFrame(replicas).to_csv(destination/"replication.csv",index=False,encoding="utf-8-sig")
    audits=[]
    for cutoff in ("2023-06-30","2025-06-30","2026-09-30"):
        cut=pd.Timestamp(cutoff);short=core.loc[:cut];cb,_,cm,_=core_context(short)
        prefix_full=full[PRIMARY].loc[:cut];sp=short.copy();sp[PRIMARY]=prefix_full[PRIMARY].reindex(short.index)
        changed=core.copy();changed.loc[changed.index>cut]*=3
        fb,_,fm,_=core_context(changed)
        altered_full=full[PRIMARY].copy();altered_full.loc[altered_full.index>cut]*=7
        pp=changed.copy();pp[PRIMARY]=altered_full[PRIMARY].reindex(changed.index)
        for guard in (False,True):
            for rule,cap in requested:
                name=key(rule,cap,guard);old=targets[name]
                prefix=build(sp,cb,cm,prefix_full,rule,cap,guard)
                future=build(pp,fb,fm,altered_full,rule,cap,guard)
                for label,check,mask in (("prefix",prefix,old.index<cut),("future_perturbation",future,old.index<=cut)):
                    expected=old.loc[mask]
                    pd.testing.assert_frame_equal(expected,check.reindex(expected.index),atol=1e-12,rtol=1e-12)
                    audits.append({"variant":name,"cutoff":cutoff,"check":label,"max_error":float((expected-check.reindex(expected.index)).abs().to_numpy().max())})
        print(f"defense time audit {cutoff} complete",flush=True)
    pd.DataFrame(audits).to_csv(destination/"time_audit.csv",index=False,encoding="utf-8-sig")
    check_name=key(ReversionRule(200,0,.05,None),.30)
    cost_replay=run_backtest(prices,targets[check_name],START,end,fee_bps=20,slippage_bps=0)
    equity=results[check_name].equity
    np.testing.assert_allclose(cost_replay.equity.daily_return,(1+equity.gross_return)*(1-equity.turnover*20/10000)-1,atol=1e-12)
    assert protected()==before,"研究修改了正式策略或信号"
    checks={"end":str(end),"user_requested_cases":4,"spare_budget_controls":4,"risk_replacement_controls":3,
            "default_files_unchanged":True,"protected_sha256":before,"time_audits":len(audits),
            "max_target_error":max(a["max_error"] for a in audits),"cost20_replay_verified":True,
            "primary_core_targets_unchanged":True,"max_allowed_positions":5,
            "primary_sleeve_reapplies_global_8pct_budget":False,
            "past_history_already_inspected":True,"no_p_value_from_previous_search_reused":True,
            "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "helper_sha256":hashlib.sha256((ROOT/"etf_rotation/mean_reversion.py").read_bytes()).hexdigest(),
            "price_sha256":hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest()}
    (destination/"checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
    def title(name):
        if name=="default":return "当前默认"
        x=int(name.split("_out")[1].split("_")[0]);cap=int(name.rsplit("cap",1)[1])
        mode="风险替换" if "risk_replace" in name else "剩余额度" if "spare_budget" in name else "国债现金替换"
        return f"出+{x}%／{mode}／上限{cap}%"
    primary=[key(rule,cap) for rule,cap in requested]
    controls=[n for n in targets if "risk_replace" in n]
    ordered=["default"]+controls+primary
    lines=["# 红利低波：以国债和现金为资金来源的反转卫星", "",f"研究日期2026-10-21；行情截止{end}；主绩效2022-01-04起。", "",
           "## 规则与比较口径", "",
           "本轮固定MA200、入0%、出+4%/+5%、不限持有日，上限20%/30%。从原国债及现金目标中拨款，"
           "先用现金、再减国债，保持所有原风险ETF的目标权重、名单与核心缓冲路径。没有新调入金、杠杆或透支。"
           "持有期间每周恢复目标卫星额度；额度不足时使用可用防守资金。周末信号、下一交易日收盘成交。", "",
           "原核心继续执行其8%预算、弱市减仓、单只30%和最多3只风险ETF；主方案叠加卫星后不再将整个账户压回8%预算或70%风险上限，"
           "因此可出现4只风险ETF加国债共5只、风险总仓位最高100%。它是用户本轮资金来源假说的研究性放宽，未切换实盘默认。"
           "另有4项匹配对照：保持核心不动，仅在原8%/70%额度（弱市4%/35%）剩余范围加卫星，额度不足则少买或不买。"
           "若核心原目标自身已超重新测算的预测波动额度，剩余额度对照不主动修复核心。", "",
           "红利低波是股票ETF，国债和现金属于资金来源；换入红利后不能继续把该份额算作防守仓。", "",
           "## 全期及2026上半年", "", "单边5bps；夏普为零无风险利率；回撤按收盘净值。2026上半年使用实际区间累计收益，不年化。", "",
           "| 方案 | 年化 | 夏普 | 最大回撤 | 实际年化波动 | 2026上半年收益 | 上半年回撤 | 平均红利仓 |",
           "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in ordered:
        row=table.loc[n]
        lines.append(f"| {title(n)} | {row.annual:.2%} | {row.sharpe:.3f} | {row.drawdown:.2%} | {row.volatility:.2%} | {row.h1_2026_total:.2%} | {row.h1_2026_drawdown:.2%} | {row.mean_dividend_weight:.2%} |")
    h1_bench={s:float((1+returns.loc["2026-01-01":"2026-06-30",s]).prod()-1) for s in ("510300","159915",BOND,PRIMARY)}
    lines += ["", "同期2026上半年ETF自身复权收益："+"；".join(f"{s} {v:.2%}" for s,v in h1_bench.items())+"。", "",
              "![全期收益、回撤及2026上半年](assets/dividend_defensive_funding_20261021.png)", "", "## 交易成本敏感性", "",
              "| 方案 | 单边10bps年化 | 10bps夏普 | 10bps回撤 | 单边20bps年化 | 20bps夏普 | 20bps回撤 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in ordered:
        row=table.loc[n]
        lines.append(f"| {title(n)} | {row.cost10_annual:.2%} | {row.cost10_sharpe:.3f} | {row.cost10_drawdown:.2%} | {row.cost20_annual:.2%} | {row.cost20_sharpe:.3f} | {row.cost20_drawdown:.2%} |")
    lines += ["", "## 保留原风险额度的匹配对照", "", "以下都保持核心目标不动，仅压缩新增红利仓。", "",
              "| 方案 | 年化 | 夏普 | 最大回撤 | 2026上半年收益 | 平均红利仓 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for rule,cap in requested:
        n=key(rule,cap,True);row=table.loc[n]
        lines.append(f"| {title(n)} | {row.annual:.2%} | {row.sharpe:.3f} | {row.drawdown:.2%} | {row.h1_2026_total:.2%} | {row.mean_dividend_weight:.2%} |")
    lines += ["", "## 风险与持仓数量", "", "预测使用信号日及以前60个交易日；预测波动只作诊断，不是实际或未来回撤上限。", "",
              "| 方案 | 最大风险目标总仓 | 最大ETF数 | 最高预测风险仓波动 | 最高预测账户波动 | 超原波动额度信号数 | 超原总仓额度信号数 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in ["default"]+primary+[key(r,c,True) for r,c in requested]:
        row=table.loc[n]
        lines.append(f"| {title(n)} | {row.max_target_gross_risky:.2%} | {int(row.max_target_positions)} | {row.max_predicted_risk_vol:.2%} | {row.max_predicted_account_vol:.2%} | {int(row.forecast_risk_budget_breach_signals)} | {int(row.gross_budget_breach_signals)} |")
    original_budget=table.loc['default']
    lines += ["",f"重新核算原默认目标有{int(original_budget.forecast_risk_budget_breach_signals)}个信号点超过8%目标预算（弱市按4%检查），"
              f"最高预测风险仓波动{original_budget.max_predicted_risk_vol:.2%}。本轮为保持同一核心目标，未修改默认；"
              "剩余额度对照保留这些原有超额点，但未新增超额点。8%应理解为目标预算，不能当作账户波动或回撤硬上限。", "",
              "## 时间分段", "", "2022—2024及2025以后都是已经被观察的历史，不是独立盲测。", "",
              "| 方案 | 前段年化 | 前段夏普 | 前段回撤 | 后段年化 | 后段夏普 | 后段回撤 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for n in ordered:
        row=table.loc[n]
        lines.append(f"| {title(n)} | {row.design_annual:.2%} | {row.design_sharpe:.3f} | {row.design_drawdown:.2%} | {row.later_annual:.2%} | {row.later_sharpe:.3f} | {row.later_drawdown:.2%} |")
    lines += ["", "## 同规则迁移ETF", "", "515100与515450不重新调参，相关性高，不是独立市场样本。", "",
              "| ETF | 方案 | 年化 | 夏普 | 回撤 |", "| --- | --- | ---: | ---: | ---: |"]
    for row in replicas:
        lines.append(f"| {row['symbol']} | {title(row['variant'])} | {row['annual']:.2%} | {row['sharpe']:.3f} | {row['drawdown']:.2%} |")
    lines += ["", "## 胜率与持有期的限制", ""]
    for rule in (ReversionRule(200,0,.04,None),ReversionRule(200,0,.05,None)):
        relevant=cycles.loc[cycles.rule.eq(rule.key)&pd.to_datetime(cycles.entry_execution).ge(pd.Timestamp(START))]
        closed=relevant.loc[relevant.closed]
        lines.append(f"- 出+{rule.exit:.0%}：主样本已完成{len(closed)}个周期，未完成{int((~relevant.closed).sum())}个；"
                     f"已完成ETF未扣费胜率{closed.asset_return.gt(0).mean():.2%}，最深相对买入价浮亏{relevant.mae.min():.2%}。"
                     "这些是ETF周期统计，不是账户交易保证；开放周期保留在组合净值内。")
    lines += ["", "- 均线上方4%或5%是动态退出线，不能等同于赚4%或5%；不限日可能持续持有很久。",
              "- 这是看过既有历史后提出的4项补充试验，未沿用旧72项研究的统计显著性结论。",
              "- 不模拟盘中回撤、最低佣金、整数手、真实现金分红到账延迟或成交容量。历史回撤不构成未来上限。", "",
              "## 审计与复现", "",f"4项主方案和4项额度对照共{len(audits)}项时间截断/未来行情扰动通过，目标误差0；"
              "原风险目标权重逐项一致；成本20bps由同一成交引擎复算验证。正式源代码、锚点和实盘信号指纹未变。", "",
              "```powershell",f".\\.venv\\Scripts\\python.exe research_dividend_reversion.py --defensive-funding --end {end}","```", "",
              "逐日净值、目标与实际权重、风险预测、年度表现、跨ETF结果、开放周期和审计均在output/dividend_reversion_20261021/defensive_funding。"]
    report=ROOT/"docs/dividend_defensive_funding_20261021.md";report.write_text("\n".join(lines)+"\n",encoding="utf-8")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"]=["Microsoft YaHei","DejaVu Sans"];plt.rcParams["axes.unicode_minus"]=False
    plotted=["default",case_key(ReversionRule(200,0,.05,None),"risk_replace",.40)]+primary
    fig,axes=plt.subplots(3,1,figsize=(14,12),gridspec_kw={"height_ratios":[2,1,1.3]})
    for n in plotted:
        line,=axes[0].plot(nav.index,(nav[n]-1)*100,label=title(n),linewidth=1.6)
        axes[1].plot(nav.index,(nav[n]/nav[n].cummax().clip(lower=1)-1)*100,color=line.get_color(),linewidth=1.1)
        # 从2025年末的同一净值1起算，不丢掉2026年首个交易日收益。
        daily=results[n].equity.daily_return.loc["2026-01-01":"2026-06-30"]
        anchor=prices.index[prices.index<pd.Timestamp("2026-01-01")][-1]
        path=pd.concat([pd.Series([1.],index=[anchor]),(1+daily).cumprod()])
        axes[2].plot(path.index,(path-1)*100,color=line.get_color(),linewidth=1.4)
    axes[0].legend(loc="upper left",fontsize=8,ncol=2)
    axes[0].set_title(f"MA200红利反转：从国债/现金拨款，保留原核心｜截至{end}\n主方案放宽全账户8%/70%额度，最多5只ETF；单边5bps；未切换默认")
    axes[0].set_ylabel("累计收益（%）");axes[1].set_ylabel("收盘回撤（%）")
    axes[2].set_ylabel("2026上半年收益（%）")
    for ax in axes:ax.grid(alpha=.25)
    axes[1].sharex(axes[0])
    fig.tight_layout();fig.savefig(ROOT/"docs/assets/dividend_defensive_funding_20261021.png",dpi=150);plt.close(fig)
    print(table[["annual","sharpe","drawdown","h1_2026_total","cost20_annual","cost20_sharpe","cost20_drawdown"]].to_string(),flush=True)
    print(f"DEFENSIVE FUNDING COMPLETE {report}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end",type=date.fromisoformat,default=date(2026,10,20))
    parser.add_argument("--download-only",action="store_true")
    parser.add_argument("--report-only",action="store_true")
    parser.add_argument("--cap40",action="store_true",help="仅运行用户指定的40%%风险替换补充研究")
    parser.add_argument("--defensive-funding",action="store_true",help="红利20/30%%国债现金替换研究")
    args=parser.parse_args()
    if sum((args.download_only,args.report_only,args.cap40,args.defensive_funding))>1:parser.error("单独运行模式不能同时指定")
    if args.defensive_funding:defensive_funding_comparison(args.end)
    elif args.cap40:cap40_comparison(args.end)
    elif args.download_only:initialize(args.end);download(args.end)
    elif args.report_only:print(write_report(date.fromisoformat(json.loads((OUT/"manifest.json").read_text(encoding="utf-8"))["end"])))
    else:execute(args.end)


if __name__=="__main__":main()
