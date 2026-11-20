"""CL120的IVOL实际选仓影响、事件回放及ETF持仓时间；只读既有行情。

python research_ivol_impact.py
python research_ivol_impact.py --report-only
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.data import load_symbol
from etf_rotation.factors import FactorSpec,compute_factor
from etf_rotation.risk_strategy import risk_controlled_signals,risk_score_components
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.mean_reversion import defensive_funding_targets
from etf_rotation.backtest import run_backtest
from research_cl120_pool_expansion import CORE_NAMES,BASE,END,SAT,state,protected,markdown_table

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/ivol_impact_20261110';CACHE=ROOT/'data/factor_extensions_20261109'
NAMES={**CORE_NAMES,SAT:'红利低波卫星','CASH':'现金'}
RISK=[s for s in CORE_NAMES if s!='511260']
EPS=1e-8

def json_file(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')

def read_prices():
    values={s:load_symbol(s,date(2010,1,1),END,CACHE,offline=True) for s in [*CORE_NAMES,SAT]}
    return pd.DataFrame(values,index=values['510300'].index).rename_axis('date')

def selected(row,symbols=RISK):
    return tuple(sorted(s for s in symbols if row.get(s,0)>EPS))

def with_cash(frame):
    out=frame.copy();out['CASH']=1-out.sum(axis=1)
    return out

def selection_states(prices,signals,rotation_state):
    """每周决策前的缓冲状态，含原锚点干预；仅用前次已形成的目标。"""
    kwargs=rotation_state.signal_kwargs(prices);anchors=[]
    if 'initial_selection_date' in kwargs:
        anchors.append((kwargs['initial_selection_date'],kwargs['initial_risky_selection']))
    anchors+=kwargs.get('selection_resets',[]);anchors.sort(key=lambda x:x[0])
    result={};resets=[];previous=[];next_reset=0
    for day,row in signals.iterrows():
        while next_reset<len(anchors) and day>pd.Timestamp(anchors[next_reset][0]):
            previous=list(anchors[next_reset][1]);next_reset+=1
        result[day]=tuple(previous)
        pos=prices.index.get_loc(day)
        if pos>0:resets.append((prices.index[pos-1],list(previous)))
        previous=list(selected(row))
    return result,resets,anchors

def conditional_replay(prices,signals,start,end,initial_weights,cost_bps=5.):
    """从start收盘已知仓位继续回放；不虚构用于建仓的更早信号。"""
    dates=prices.index[(prices.index>=start)&(prices.index<=end)]
    columns=prices.columns;held=initial_weights.reindex(columns).fillna(0.).copy()
    if held.min()<-1e-10 or held.sum()>1+1e-8:raise ValueError('Invalid initial state')
    value=prices.ffill(limit=3);executions={}
    for day,row in signals.loc[signals.index>=start].iterrows():
        pos=prices.index.searchsorted(day,side='right')
        if pos<len(prices):executions[prices.index[pos]]=row.reindex(columns,fill_value=0.)
    nav=1.;rows=[{'date':dates[0],'nav':nav,'gross_return':0.,'cost':0.}]
    for i,day in enumerate(dates[1:],1):
        owned=held.gt(0)
        growth=value.loc[day,owned]/value.loc[dates[i-1],owned]
        if growth.isna().any():raise ValueError('Missing held quote in conditional replay')
        gross=float((held[owned]*(growth-1)).sum())
        held.loc[owned]=held[owned]*growth/(1+gross);nav*=1+gross;cost=0.
        if day in executions:
            target=executions[day]
            if not np.isfinite(target).all() or target.min()<-1e-10 or target.sum()>1+1e-8:raise ValueError('Invalid conditional target')
            changed=(target-held).abs().gt(1e-10)
            if prices.loc[day,changed].notna().all():
                cost=float((target-held).abs().sum())*cost_bps/10000
                nav*=1-cost;held=target.copy()
        rows.append({'date':day,'nav':nav,'gross_return':gross,'cost':cost})
    return pd.DataFrame(rows).set_index('date')

def cycle_curve(prices,target,pretrade,execution,finish,cost_bps=5.):
    """同一执行收盘前财富和旧仓下，本次换仓至下一次换仓前的局部净值。"""
    target=target.reindex(prices.columns,fill_value=0.)
    cost=float((target-pretrade).abs().sum())*cost_bps/10000
    quotes=prices.ffill(limit=3).loc[execution:finish]
    owned=target.gt(0)
    if quotes.loc[:,owned].isna().any().any():raise ValueError('Local holding valuation unavailable')
    gains=quotes.loc[:,owned].div(quotes.loc[execution,owned],axis=1)-1
    return (1+gains.mul(target[owned],axis=1).sum(axis=1))*(1-cost),cost

def holding_summary(result,start,model):
    rows=[];n=len(result.weights)
    for s in result.weights:
        w=result.weights[s];held=w.gt(EPS);groups=held.ne(held.shift(fill_value=False)).cumsum()
        lengths=held.loc[held].groupby(groups.loc[held]).size()
        rows.append({'start':start,'model':model,'symbol':s,'name':NAMES[s],
                     'average_weight':float(w.mean()),'held_days':int(held.sum()),'held_fraction':float(held.mean()),
                     'mean_weight_when_held':float(w[held].mean()) if held.any() else 0.,
                     'episodes_including_open':len(lengths),'median_episode_days':float(lengths.median()) if len(lengths) else 0.,
                     'max_episode_days':int(lengths.max()) if len(lengths) else 0,'open_at_end':bool(held.iloc[-1]),'trading_days':n})
    return rows

def pick_examples(events,calendar):
    """事后讲解样本：最大周受益/拖累、最强被滤龙头20日上涨、接近中位。"""
    pool=events.loc[events.start.eq('2022-01-01')&events.direct_target_changed&events.complete_cycle
                    &events.complete_20d&events.counterfactual_executable].copy()
    chosen=[]
    def add(kind,frame,ascending):
        for _,r in frame.sort_values(ascending[0],ascending=ascending[1]).iterrows():
            if any(abs(calendar.get_loc(r.execution_date)-calendar.get_loc(x['execution_date']))<20 for x in chosen):continue
            chosen.append({**r.to_dict(),'kind':kind});return
        for _,r in frame.sort_values(ascending[0],ascending=ascending[1]).iterrows():
            if r.signal_date not in [x['signal_date'] for x in chosen]:chosen.append({**r.to_dict(),'kind':kind});return
    add('最大当周受益',pool,('cycle_net_gap',False))
    add('最大当周拖累',pool,('cycle_net_gap',True))
    add('错过强势龙头后续上涨',pool.loc[pool.rank1_filtered&pool.rank1_forward20.notna()],('rank1_forward20',False))
    median=float(pool.cycle_net_gap.median());typical=pool.assign(distance_to_median=(pool.cycle_net_gap-median).abs())
    add('接近直接影响中位数',typical,('distance_to_median',True))
    return chosen

def describe_weights(row):
    parts=[f'{s} {float(w):.1%}' for s,w in row.items() if w>EPS]
    cash=1-float(row.sum())
    if 'CASH' not in row.index and cash>EPS:parts.append(f'CASH {cash:.1%}')
    return '、'.join(parts)

def next_holding_execution(weights,symbol,after,end):
    future=weights.loc[(weights.index>after)&(weights.index<=end),symbol]
    active=future.index[future.gt(EPS)]
    return active[0] if len(active) else None

def execute():
    OUT.mkdir(parents=True,exist_ok=True);before=protected()
    json_file(OUT/'plan.json',{'end':str(END),'cash_starts':['2018-01-01','2022-01-01'],'cost_bps':[5.,20.],
                'comparison':'相同17只核心+红利，唯一开关为IVOL；全历史假设，不是真实采纳记录',
                'frequency':'以分析区间内实际下一交易日执行的周目标为分母，末尾未执行信号不计',
                'direct_effect':'以CL120当周决策前的同一缓冲状态，分别加/关当前IVOL再选仓',
                'local_cycle':'从同一执行收盘前旧仓开始，含本次费用，不含下一次调仓费用',
                'case_followup_window':'从信号日收盘的共同旧仓出发，至首次执行收盘后第20交易日；含共同的执行日旧仓收益，合计21个日收益区间',
                'case_selection':['最大当周净受益','最大当周净拖累','被滤原评分第1名后续20交易日上涨（优先避开已选例20日内重叠）','接近直接影响中位数'],
                'cases_are_posthoc':'只作机制解释，使用后续结果选例，未输入任何交易信号',
                'holdings':'收盘后仓位的全期均值，持有日为收盘后仓位>1e-8；非持有期间平均仓位',
                'protected_before':before})
    prices=read_prices();core=prices[list(CORE_NAMES)];rotation=state()
    on=portfolio_signals(core,prices[[SAT]],rotation,BASE,hypothetical_satellite_history=True,
                         hypothetical_universe_history=True,hypothetical_ivol_history=True)
    off=portfolio_signals(core,prices[[SAT]],replace(rotation,ivol_policy=None),BASE,
                          hypothetical_satellite_history=True,hypothetical_universe_history=True)
    previous,resets,anchors=selection_states(core,on.core_signals,rotation)
    same_state_on=risk_controlled_signals(core,BASE,eligibility_mask=on.ivol_filter.eligibility_mask,selection_resets=resets)
    error=float((same_state_on-on.core_signals).abs().to_numpy().max())
    checks=[{'check':'common_predecision_state_reproduces_CL120','error':error,'passed':error<1e-10}]
    shadow_core=risk_controlled_signals(core,BASE,selection_resets=resets)
    shadow=defensive_funding_targets(prices,shadow_core,on.satellite_states,SAT,.30)
    on.signals.to_csv(OUT/'signals_ivol.csv');off.signals.to_csv(OUT/'signals_no_ivol.csv');shadow.to_csv(OUT/'signals_same_state_no_ivol.csv')
    short=compute_factor(core,FactorSpec('momentum',20));long=compute_factor(core,FactorSpec('momentum',40))
    vol=compute_factor(core,FactorSpec('volatility',60));trend=compute_factor(core,FactorSpec('trend',40))
    score=risk_score_components(short,long,vol,BASE,compute_factor(core,FactorSpec('ma_zscore',20)))['score']
    qualified=long.gt(0)&trend.gt(0)&vol.gt(0)
    results={};performance=[];holdings=[]
    for start in ('2018-01-01','2022-01-01'):
        results[start]={}
        for model,bundle in [('CL120',on),('no_ivol',off)]:
            for cost in (5.,20.):
                result=run_backtest(prices,bundle.signals,date.fromisoformat(start),END,fee_bps=0,slippage_bps=cost)
                performance.append({'start':start,'model':model,'cost_bps':cost,**result.metrics})
                if cost==5:
                    results[start][model]=result;holdings.extend(holding_summary(result,start,model))
                    result.equity.to_csv(OUT/f'equity_{model}_{start}.csv');result.weights.to_csv(OUT/f'weights_{model}_{start}.csv')
    pd.DataFrame(performance).to_csv(OUT/'performance.csv',index=False);pd.DataFrame(holdings).to_csv(OUT/'holdings.csv',index=False,encoding='utf-8-sig')
    rows=[];asset_rows=[];calendar=prices.index;weeks=on.signals.index
    for start,models in results.items():
        account=models['CL120'];trades=account.trades.set_index('signal_date')
        for s in trades.index:
            execution=pd.Timestamp(trades.at[s,'execution_date']);e=calendar.get_loc(execution)
            pos=weeks.get_loc(s);next_pos=calendar.searchsorted(weeks[pos+1],side='right') if pos+1<len(weeks) else len(calendar)
            finish=calendar[next_pos] if next_pos<len(calendar) else pd.NaT
            ranked=score.loc[s,RISK].where(qualified.loc[s,RISK]).dropna().sort_index().sort_values(ascending=False,kind='stable')
            rank1=ranked.index[0] if len(ranked) else None
            target=on.signals.loc[s];reference=off.signals.loc[s];same=shadow.loc[s]
            actual_changed=selected(target)!=selected(reference)
            direct_changed=selected(target)!=selected(same)
            excluded=[x for x in RISK if same[x]>EPS and not on.ivol_filter.eligibility_mask.at[s,x]]
            blocked_eligible=[x for x in RISK if qualified.at[s,x] and not on.ivol_filter.eligibility_mask.at[s,x]]
            y={'start':start,'signal_date':s,'execution_date':execution,'next_execution_date':finish,
               'independent_selection_changed':actual_changed,'direct_target_changed':direct_changed,
               'inherited_only':bool(actual_changed and not direct_changed),
               'any_qualified_filtered':bool(blocked_eligible),'rank1':rank1,
               'rank1_filtered':bool(rank1 is not None and not on.ivol_filter.eligibility_mask.at[s,rank1]),
               'directly_excluded':','.join(excluded),'all_eligible_filtered':','.join(blocked_eligible),
               'target_distance':float((with_cash(target.to_frame().T).iloc[0]-with_cash(reference.to_frame().T).iloc[0]).abs().sum()/2),
               'ivol_target':describe_weights(target),'no_ivol_target':describe_weights(reference),
               'same_state_no_ivol_target':describe_weights(same),'complete_cycle':bool(pd.notna(finish)),
               'complete_20d':bool(e+20<len(calendar)),'rank1_forward20':np.nan,
               'cycle_net_on':np.nan,'cycle_net_off':np.nan,'cycle_net_gap':np.nan,'counterfactual_executable':False}
            j=account.weights.index.get_loc(execution)
            prior=account.weights.iloc[j-1].reindex(prices.columns).fillna(0.) if j else pd.Series(0.,index=prices.columns)
            if j:
                prev_day=account.weights.index[j-1];owned=prior.gt(0)
                growth=prices.ffill(limit=3).loc[execution,owned]/prices.ffill(limit=3).loc[prev_day,owned]
                gross=float((prior[owned]*(growth-1)).sum());prior.loc[owned]*=growth/(1+gross)
            changed=(same-prior).abs().gt(1e-10)
            can_trade=bool(prices.loc[execution,changed].notna().all() and trades.at[s,'status']=='executed')
            y['counterfactual_executable']=can_trade
            if pd.notna(finish) and can_trade:
                a,_=cycle_curve(prices,target,prior,execution,finish)
                b,_=cycle_curve(prices,same,prior,execution,finish)
                y.update(cycle_net_on=float(a.iloc[-1]-1),cycle_net_off=float(b.iloc[-1]-1),cycle_net_gap=float(a.iloc[-1]-b.iloc[-1]))
                expected=(1-float(trades.at[s,'cost_fraction']))*float((1+account.equity.loc[(account.equity.index>execution)&(account.equity.index<=finish),'gross_return']).prod())
                error=abs(float(a.iloc[-1])-expected)
                checks.append({'check':'local_curve_'+start+'_'+str(s.date()),'error':error,'passed':error<1e-10})
            if rank1 is not None and e+20<len(calendar):
                p0=prices.at[execution,rank1];p1=prices.iloc[e+20][rank1]
                if pd.notna(p0) and pd.notna(p1):y['rank1_forward20']=float(p1/p0-1)
            for x in excluded:
                row={'start':start,'signal_date':s,'execution_date':execution,'symbol':x,
                     'ivol_annualized':float(on.ivol_filter.ivol.at[s,x]),'raw_rank':int(ranked.index.get_loc(x)+1),
                     'CL120_weight':float(target[x]),'same_state_no_ivol_weight':float(same[x])}
                for h in (5,20):
                    row[f'forward_{h}d']=float(prices.iloc[e+h][x]/prices.at[execution,x]-1) if e+h<len(calendar) and pd.notna(prices.iloc[e+h][x]) and pd.notna(prices.at[execution,x]) else np.nan
                asset_rows.append(row)
            rows.append(y)
    events=pd.DataFrame(rows);events.to_csv(OUT/'all_weekly_events.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(asset_rows).to_csv(OUT/'excluded_asset_followups.csv',index=False,encoding='utf-8-sig')
    frequency=[];annual=[]
    for start,g in events.groupby('start'):
        on_w=results[start]['CL120'].weights;off_w=results[start]['no_ivol'].weights
        daily_changed=on_w[RISK].gt(EPS).ne(off_w[RISK].gt(EPS)).any(axis=1)
        local=g.loc[g.direct_target_changed&g.cycle_net_gap.notna()]
        row={'start':start,'executable_weeks':len(g),'qualified_filtered_weeks':int(g.any_qualified_filtered.sum()),
             'rank1_filtered_weeks':int(g.rank1_filtered.sum()),'direct_changed_weeks':int(g.direct_target_changed.sum()),
             'independent_changed_weeks':int(g.independent_selection_changed.sum()),'inherited_only_weeks':int(g.inherited_only.sum()),
             'daily_selection_diff_days':int(daily_changed.sum()),'daily_days':len(daily_changed),
             'mean_target_distance_if_changed':float(g.loc[g.independent_selection_changed,'target_distance'].mean()),
             'complete_direct_cycles':len(local),'positive_direct_cycles':int(local.cycle_net_gap.gt(1e-10).sum()),
             'negative_direct_cycles':int(local.cycle_net_gap.lt(-1e-10).sum()),
             'median_direct_cycle_gap':float(local.cycle_net_gap.median()),'mean_direct_cycle_gap':float(local.cycle_net_gap.mean())}
        frequency.append(row)
        for year,z in g.groupby(g.execution_date.dt.year):
            annual.append({'start':start,'year':int(year),'weeks':len(z),'direct':int(z.direct_target_changed.sum()),
                           'independent':int(z.independent_selection_changed.sum()),'rank1_filtered':int(z.rank1_filtered.sum())})
    json_file(OUT/'frequency.json',frequency);pd.DataFrame(annual).to_csv(OUT/'frequency_by_year.csv',index=False)
    examples=pick_examples(events,calendar);case_rows=[];window_curves=[];cycle_curves=[]
    for case in examples:
        s=case['signal_date'];execution=case['execution_date'];e=calendar.get_loc(execution);last=calendar[e+20]
        initial=results['2022-01-01']['CL120'].weights.loc[s].reindex(prices.columns)
        prior_day=calendar[calendar.get_loc(s)-1]
        later=[(d,list(x)) for d,x in anchors if pd.Timestamp(d)>prior_day]
        branch_core=risk_controlled_signals(core.loc[:last],BASE,initial_risky_selection=list(previous[s]),
                                            initial_selection_date=prior_day,selection_resets=later)
        branch=defensive_funding_targets(prices.loc[:last],branch_core,on.satellite_states.loc[:last],SAT,.30)
        error=float((branch.loc[s]-shadow.loc[s]).abs().max())
        checks.append({'check':'case_same_initial_decision_'+str(s.date()),'error':error,'passed':error<1e-10})
        a=conditional_replay(prices,on.signals,s,last,initial)
        b=conditional_replay(prices,branch,s,last,initial)
        expected=results['2022-01-01']['CL120'].equity.strategy_nav.loc[s:last]
        error=float((a.nav-expected/expected.iloc[0]).abs().max())
        checks.append({'check':'conditional_replay_matches_production_'+str(s.date()),'error':error,'passed':error<1e-10})
        row={k:v for k,v in case.items() if k not in ['distance_to_median']}
        row.update(conditional_end=last,conditional_return_on=float(a.nav.iloc[-1]-1),conditional_return_off=float(b.nav.iloc[-1]-1),
                   conditional_net_gap=float(a.nav.iloc[-1]-b.nav.iloc[-1]))
        removed=case['directly_excluded'].split(',') if case['directly_excluded'] else []
        reentry=[]
        for x in removed:
            returned=next_holding_execution(results['2022-01-01']['CL120'].weights,x,execution,last)
            reentry.append(x+':'+(str(returned.date()) if returned is not None else '20日内未重入'))
        row['reentry']=','.join(reentry);case_rows.append(row)
        window_curves.extend({'kind':case['kind'],'signal_date':s,'date':d,'CL120':float(a.nav.loc[d]),'same_state_no_ivol':float(b.nav.loc[d])} for d in a.index)
        # 局部周收益曲线以执行前财富归一：从已校验的局部计算重建同一旧仓。
        account=results['2022-01-01']['CL120'];j=account.weights.index.get_loc(execution)
        pre=account.weights.iloc[j-1].reindex(prices.columns).fillna(0.)
        growth=prices.ffill(limit=3).loc[execution]/prices.ffill(limit=3).loc[account.weights.index[j-1]]
        owned=pre.gt(0);gross=float((pre[owned]*(growth[owned]-1)).sum());pre.loc[owned]*=growth[owned]/(1+gross)
        local_a,_=cycle_curve(prices,on.signals.loc[s],pre,execution,case['next_execution_date'])
        local_b,_=cycle_curve(prices,shadow.loc[s],pre,execution,case['next_execution_date'])
        cycle_curves.extend({'kind':case['kind'],'signal_date':s,'date':d,'CL120':float(local_a.loc[d]),'same_state_no_ivol':float(local_b.loc[d])} for d in local_a.index)
    pd.DataFrame(case_rows).to_csv(OUT/'representative_cases.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(window_curves).to_csv(OUT/'case_20d_curves.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cycle_curves).to_csv(OUT/'case_week_curves.csv',index=False,encoding='utf-8-sig')
    checks.append({'check':'default_and_live_files_unchanged','passed':before==protected()})
    json_file(OUT/'checks.json',checks)
    json_file(OUT/'manifest.json',{'end':str(END),'all_checks_passed':all(x['passed'] for x in checks),
                'protected_before':before,'protected_after':protected(),'default_changed':False,
                'price_hash':hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
                'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if not all(x['passed'] for x in checks):raise AssertionError('IVOL impact validation failed')
    print(pd.DataFrame(frequency).to_string(index=False),flush=True)
    print(pd.DataFrame(case_rows)[['kind','signal_date','directly_excluded','cycle_net_gap','rank1_forward20','conditional_net_gap']].to_string(index=False),flush=True)

def display_target(text):
    import re
    return re.sub(r'\b([0-9]{6}|CASH)\b',lambda m:NAMES[m.group(1)]+'('+m.group(1)+')',str(text))

def figures():
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    assets=ROOT/'docs/assets';assets.mkdir(exist_ok=True,parents=True)
    holdings=pd.read_csv(OUT/'holdings.csv',dtype={'symbol':str})
    for start in ('2022-01-01','2018-01-01'):
        rows=holdings.loc[holdings.start.eq(start)]
        a=rows.loc[rows.model.eq('CL120')].set_index('symbol')
        b=rows.loc[rows.model.eq('no_ivol')].set_index('symbol')
        order=a.average_weight.sort_values(ascending=False).index.tolist()
        labels=[NAMES[s]+' '+s for s in order];y=np.arange(len(order));height=.34
        fig,axes=plt.subplots(1,2,figsize=(18,12.8),sharey=True)
        for ax,key,title,scale in [(axes[0],'average_weight','全期平均账户仓位',100),
                                   (axes[1],'held_days','持有交易日数',1)]:
            off=b.loc[order,key].to_numpy()*scale;on=a.loc[order,key].to_numpy()*scale
            ax.barh(y-height/2,off,height,color='#b8bec7',label='无IVOL，同池同规则')
            ax.barh(y+height/2,on,height,color='#2364aa',label='CL120，有IVOL')
            maximum=float(max(off.max(),on.max()));padding=maximum*.012
            for k,(x0,x1) in enumerate(zip(off,on)):
                ax.text(x0+padding,k-height/2,f'{x0:.2f}%' if scale==100 else f'{int(x0)}',va='center',fontsize=8)
                ax.text(x1+padding,k+height/2,f'{x1:.2f}%' if scale==100 else f'{int(x1)}',va='center',fontsize=8,color='#144475')
            ax.set_title(title);ax.set_xlim(0,maximum*1.23);ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',fontsize=9)
        axes[0].set_yticks(y,labels,fontsize=10);axes[0].invert_yaxis()
        axes[0].set_xlabel('平均权重 %');axes[1].set_xlabel('交易日')
        n=int(a.trading_days.iloc[0])
        fig.suptitle(f'IVOL开关对比｜{start[:4]}现金起点至{END}｜同一18只ETF＋现金',fontsize=15)
        fig.text(.18,.015,f'平均仓位包含未持有日的0；持有日按收盘后仓位>0计算。共{n}个交易日，全历史假设，非实盘。',fontsize=10)
        fig.tight_layout(rect=[0,.03,1,.97]);fig.savefig(assets/f'ivol_holdings_{start[:4]}_20261110.png',dpi=150);plt.close(fig)
    annual=pd.read_csv(OUT/'frequency_by_year.csv');annual=annual.loc[annual.start.eq('2022-01-01')]
    events=pd.read_csv(OUT/'all_weekly_events.csv',parse_dates=['signal_date','execution_date'])
    direct=events.loc[events.start.eq('2022-01-01')&events.direct_target_changed&events.cycle_net_gap.notna()]
    fig,axes=plt.subplots(2,1,figsize=(13,9))
    x=np.arange(len(annual));width=.35
    for offset,column,color,name in [(-width/2,'direct','#2364aa','当周直接改变选仓'),(width/2,'independent','#d97a32','完整两条路径持仓不同')]:
        values=annual[column]/annual.weeks*100
        axes[0].bar(x+offset,values,width,color=color,label=name)
        for k,(v,count,n) in enumerate(zip(values,annual[column],annual.weeks)):
            axes[0].text(k+offset,v+.5,f'{int(count)}/{int(n)}',ha='center',fontsize=9)
    axes[0].set(xticks=x,xticklabels=annual.year.astype(str),ylabel='占该年周信号 %',title='IVOL多久改变一次持仓？分母含零换手周目标')
    axes[0].legend(fontsize=9);axes[0].grid(axis='y',alpha=.2);axes[0].set_ylim(0,75)
    colors=np.where(direct.cycle_net_gap>=0,'#2364aa','#bc4141')
    axes[1].scatter(direct.execution_date,direct.cycle_net_gap*100,c=colors,s=25,alpha=.8)
    axes[1].axhline(0,color='#666666',linewidth=.8)
    axes[1].set(title='全部直接改变周：同一旧仓下，到下一次换仓前的净收益差',ylabel='保留IVOL－关闭IVOL / 百分点')
    axes[1].grid(alpha=.2);fig.tight_layout()
    fig.savefig(assets/'ivol_change_frequency_20261110.png',dpi=150);plt.close(fig)
    cases=pd.read_csv(OUT/'representative_cases.csv',parse_dates=['signal_date','execution_date','conditional_end'])
    curves=pd.read_csv(OUT/'case_20d_curves.csv',parse_dates=['signal_date','date'])
    fig,axes=plt.subplots(2,2,figsize=(16,10))
    for i,(ax,case) in enumerate(zip(axes.flat,cases.itertuples())):
        c=curves.loc[curves.signal_date.eq(case.signal_date)]
        ax.plot(c.date,(c.CL120-1)*100,color='#2364aa',linewidth=2,label='保留IVOL')
        ax.plot(c.date,(c.same_state_no_ivol-1)*100,color='#8b929b',linewidth=1.7,label='从同一旧仓关闭IVOL')
        ax.axvline(case.execution_date,color='#999999',linestyle=':',linewidth=.9)
        names='、'.join(NAMES[s].replace('ETF','') for s in str(case.directly_excluded).split(','))
        ax.set_title(f'{chr(65+i)} {case.kind}｜{case.signal_date.date()}信号\n过滤{names}；20日净差{case.conditional_net_gap*100:+.2f}个百分点',fontsize=11)
        ax.set_ylabel('从信号日收盘起累计净收益 %');ax.grid(alpha=.2);ax.legend(fontsize=9)
        ax.tick_params(axis='x',rotation=25)
    fig.suptitle('代表案例：从信号收盘共同旧仓出发，跟踪至执行后第20交易日；后续正常周度调仓',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(assets/'ivol_representative_cases_20261110.png',dpi=150);plt.close(fig)

def report():
    performance=pd.read_csv(OUT/'performance.csv');holdings=pd.read_csv(OUT/'holdings.csv',dtype={'symbol':str})
    freq=json.loads((OUT/'frequency.json').read_text(encoding='utf-8'));checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    f={r['start']:r for r in freq};r=f['2022-01-01'];old=f['2018-01-01']
    annual=pd.read_csv(OUT/'frequency_by_year.csv')
    annual=annual.loc[annual.start.eq('2022-01-01')]
    def pct(x):return f'{float(x):.2%}'
    def count(x,n):return f'{x}/{n}（{x/n:.1%}）'
    frequency_rows=[]
    for row in freq:
        n=row['executable_weeks']
        frequency_rows.append([row['start'][:4],str(n),count(row['qualified_filtered_weeks'],n),
                 count(row['rank1_filtered_weeks'],n),count(row['direct_changed_weeks'],n),count(row['independent_changed_weeks'],n),
                 count(row['inherited_only_weeks'],n)])
    performance_rows=[]
    for p in performance.itertuples():
        performance_rows.append([p.start[:4],'CL120' if p.model=='CL120' else '同池无IVOL',f'{p.cost_bps:g}',pct(p.annual_return),
                                  f'{p.sharpe_zero_rf:.3f}',pct(p.max_drawdown)])
    def holdings_table(start):
        h=holdings.loc[holdings.start.eq(start)];a=h.loc[h.model.eq('CL120')].set_index('symbol');b=h.loc[h.model.eq('no_ivol')].set_index('symbol')
        order=a.average_weight.sort_values(ascending=False).index
        rows=[[s,NAMES[s],pct(b.at[s,'average_weight']),pct(a.at[s,'average_weight']),str(int(b.at[s,'held_days'])),
               str(int(a.at[s,'held_days'])),pct(a.at[s,'held_fraction']),f'{a.at[s,"median_episode_days"]:.1f}'] for s in order]
        return markdown_table(['代码','资产','无IVOL平均仓位','CL120平均仓位','无IVOL持有日','CL120持有日','CL120日占比','CL120周期中位日'],rows)
    cases=pd.read_csv(OUT/'representative_cases.csv',dtype={'directly_excluded':str})
    assets=pd.read_csv(OUT/'excluded_asset_followups.csv',dtype={'symbol':str})
    case_sections=[]
    for i,c in enumerate(cases.itertuples()):
        a=assets.loc[assets.start.eq('2022-01-01')&assets.signal_date.eq(c.signal_date)]
        price_table=markdown_table(['被过滤ETF','当时原评分名次','当时年化IVOL','执行收盘后5日收益','执行收盘后20日收益'],
                    [[NAMES[x.symbol]+' '+x.symbol,str(int(x.raw_rank)),pct(x.ivol_annualized),pct(x.forward_5d),pct(x.forward_20d)] for x in a.itertuples()])
        case_sections.append(f'''### {chr(65+i)}．{c.kind}：{str(c.signal_date)[:10]}信号

计划于{str(c.execution_date)[:10]}收盘执行，下一次调仓为{str(c.next_execution_date)[:10]}。以下是历史假设交易，不是券商记录。

- 有IVOL目标：{display_target(c.ivol_target)}。
- 同一旧仓、当次无IVOL目标：{display_target(c.same_state_no_ivol_target)}。
- 本次执行收盘前相同财富和旧仓，到下一次调仓前，含本次单边5bps综合成本：有IVOL {pct(c.cycle_net_on)}，无IVOL {pct(c.cycle_net_off)}，净差{c.cycle_net_gap*100:+.3f}个百分点。
- 从同一信号收盘旧仓起，跟踪至首次执行收盘后第20个交易日，两条路径各自继续正常周调仓：有IVOL {pct(c.conditional_return_on)}，无IVOL {pct(c.conditional_return_off)}，净差{c.conditional_net_gap*100:+.3f}个百分点。窗口还包含首次执行当日相同旧仓的收益，合计21个日收益区间。
- 被滤ETF实际再次持有日期：{c.reentry}。该日期来自模拟账户的收盘后持仓，而非尚未执行的信号。

{price_table}
''')
    report=f'''# CL120的IVOL有多大实际影响：频率、换仓案例及全池持仓

研究日期2026-11-10；价格固定截至{END}，沿用前轮相同快照。对比同一17只核心＋512890卫星，仅开关IVOL。两者均为全历史假设回放、独立现金账本，不是历史真实采用记录；CL120规则于2026-11-09启用，首次计划该日收盘执行，不代表已确认券商成交。未修改默认策略、实盘计划或真实账户数据库。

## 改变持仓的频率

2022起点共有{r['executable_weeks']}次可执行周目标，IVOL当周直接改变选仓{r['direct_changed_weeks']}次（{r['direct_changed_weeks']/r['executable_weeks']:.1%}），约每4次信号影响一次。若两套规则各自沿历史连续运行，持仓名单不同{r['independent_changed_weeks']}周（{r['independent_changed_weeks']/r['executable_weeks']:.1%}）；逐日持仓不同{r['daily_selection_diff_days']}/{r['daily_days']}日（{r['daily_selection_diff_days']/r['daily_days']:.1%}）。

{markdown_table(['现金起点','周目标数','原趋势合格但被过滤','原评分第1名被过滤','当周直接改变','完整路径名单不同','仅历史缓冲延续差异'],frequency_rows)}

“直接影响”让两条决策使用CL120当时完全相同的前次选择，再仅切换当前过滤器；“完整路径差异”让两条策略从历史分别运行，因此还包含旧仓缓冲的连锁影响。两项不能相加：{r['inherited_only_weeks']}周属于当周没有直接影响、但前期选择不同造成的延续；另有极少数周当前直接作用被不同历史状态抵消。

每周名义上过滤一个IVOL尾部，未必会改变原本实际选中的ETF。{r['qualified_filtered_weeks']}周至少挡住一只原趋势合格标的，真正改变当次选择仅{r['direct_changed_weeks']}周。当前原评分冠军被滤{r['rank1_filtered_weeks']}周；“第1名必选”适用于过滤后的合格池。

名单不同的周，两套方案含现金的目标权重距离平均为{r['mean_target_distance_if_changed']:.2%}：定义为两套目标绝对差之和的一半，表示约多少账户权重要重新分配，不是额外换手率或收益。信号日与执行日分开统计；按分析期内的下一交易日执行日期取样，11月6日尚无样本内执行的末尾信号不计。

影响频率随年份变化：2026截至本轮快照，43次周目标中直接改变17次（39.5%），完整路径名单不同23次（53.5%），高于全期均值。不能把“约四周一次”当作固定间隔；策略仍按原周度日历交易。

{markdown_table(['执行年份','周目标数','直接改变次数/占比','完整路径不同次数/占比'],[[str(int(a.year)),str(int(a.weeks)),count(int(a.direct),int(a.weeks)),count(int(a.independent),int(a.weeks))] for a in annual.itertuples()])}

![影响频率及全部周事件](assets/ivol_change_frequency_20261110.png)

## 影响并非每次都有利

在{r['complete_direct_cycles']}个完整且直接改变的周期中，以同一执行收盘前旧仓为起点、仅比较当次决策到下次换仓前：有利{r['positive_direct_cycles']}次，不利{r['negative_direct_cycles']}次，正向比例{r['positive_direct_cycles']/r['complete_direct_cycles']:.1%}。净差中位数{r['median_direct_cycle_gap']*100:+.3f}个百分点，均值{r['mean_direct_cycle_gap']*100:+.3f}个百分点。多数单次影响较小，少数尾部事件会明显拉动平均值；这不是完整ETF交易回合胜率。

全部周期结果保留在`all_weekly_events.csv`，没有只保留盈利案例。局部收益含本次费用，不含下一次调仓费用；该指标不能直接逐次相加解释全历史收益差，因为后续缓冲状态与仓位会相互影响。完整两条策略的绩效如下：

{markdown_table(['起点','方案','单边成本bps','年化','零利率夏普','最大收盘回撤'],performance_rows)}

## 四次代表性变化与后续发展

按公开保留在脚本中的事后解释规则，选2022起点内最大当周受益、最大当周拖累、原评分第1名被滤后仍上涨、接近中位影响四类。优先让案例间隔至少20个交易日，避免全是同一轮行情。使用后续收益挑解释样本，是事后案例选择，没有输入交易信号；这些极端案例不能冒充典型胜率。

![同一旧仓起点至执行后第20交易日的正常调仓对照](assets/ivol_representative_cases_20261110.png)

{''.join(case_sections)}

A例说明过滤可以避开短期追涨风险，但芯片后20日又上涨，不能只展示第一周下跌。B例展示明确踏空：芯片随后5日上涨约16.6%，这一周过滤有代价。C例新能源随后上涨，但完整组合有替代收益且7月4日重新入选，20日净差接近零，不能将标的涨幅全部算成账户损失。D例接近单次影响中位数，虽然被滤芯片20日大跌，账户增益受原权重和替代资产表现共同决定。

## 各ETF平均仓位与持有时间

平均仓位是全部交易日收盘后权重的算术均值，未持有日记0；不是持有期间平均仓位，也不是期末建议仓位。持有日指收盘后仓位>1e-8，执行日收盘后开始计持有，实际新仓收益从下一交易日起计。周期数按连续持有区间，包含期末开放周期；未将每次加减仓算成一次新交易回合。现金单列为非ETF。

### 2022独立现金起点：1172个交易日

![2022以来全池仓位与时间](assets/ivol_holdings_2022_20261110.png)

{holdings_table('2022-01-01')}

最明显的改变是减少芯片、AI、军工，部分转向银行、宽基、新能源、黄金和境外资产。此处是本样本的结果，不是规则写死行业偏好。国债平均仓位从约47.03%降至46.02%，现金约5.46%降至5.15%；全账户风险资产平均仓位从约47.51%升至48.83%，因此IVOL并非简单增加防守仓。

红利卫星的周目标规则独立，目标启停一致，持有日同为390；平均仓位的极小差异来自其他资产收益导致的自然权重漂移。芯片39日、AI111日、军工52日说明本样本的直接股票科技敞口不高，但纳指、创业板、恒生科技仍有间接科技敞口；不能把科技行业标签简单相加。低平均仓位既可能来自短周期，也可能来自ETF上市较晚，不能单凭这一列删除资产来继续优化历史。

### 2018独立现金起点：2145个交易日

![2018以来全池仓位与时间](assets/ivol_holdings_2018_20261110.png)

{holdings_table('2018-01-01')}

两次起点分别从现金开始，不把2018账户的既有持仓或收益直接截到2022。上市前不会买入，2018年前价格仅作因子预热；当前池与卫星全历史回放仍有后视和存续选择偏差。

## 核对与复现

{len(checks)}项数值/时序核对通过，包括用相同决策前状态重建CL120目标、每个完整周期局部收益与正式引擎对账、代表案例的条件回放与正式逐日净值一致、默认文件指纹未变。事件解释窗口从已知信号收盘旧仓出发，跟踪至首次执行收盘后第20个交易日，包含共同的执行日旧仓收益（合计21个日收益区间）。两条路径继续正常周度调仓并支付各自费用，没有固定持仓20日或虚构更早的建仓信号。单只ETF未来5/20日收益只用于事后解释，未输入选仓。

```powershell
python research_ivol_impact.py
python research_ivol_impact.py --report-only
```

读取`data/factor_extensions_20261109/`的18只独立行情缓存，不联网也不需要Token；新环境先运行`research_factor_extensions.py --download-only`。全部信号、逐日权重、完整周事件、被滤ETF后续收益、代表案例、条件回放与检查放在`output/ivol_impact_20261110/`，均Git忽略。报告和图保留在`docs/`。
'''
    (ROOT/'docs/ivol_impact_20261110.md').write_text(report,encoding='utf-8')
    print('Four figures and IVOL impact report saved.',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if not args.report_only:execute()
    figures();report()

if __name__=='__main__':main()
