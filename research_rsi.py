"""固定CL120的RSI评分研究；不改实盘规则、状态或账户。

python research_rsi.py
python research_rsi.py --download-latest
python research_rsi.py --report-only
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_symbol, _normalize_history, _normalize_table
from etf_rotation.factor_store import FactorStore
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.ivol_filter import market_ivol_filter
from etf_rotation.mean_reversion import defensive_funding_targets, reversion_states, ReversionRule
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.risk_strategy import risk_controlled_signals, weekly_signal_dates, risk_score_components
from research_cl120_pool_expansion import CORE_NAMES, BASE, SAT, A_BASE, state, protected, Client
from research_overfitting_audit import CoreAudit, simulate, statistics
from research_pool_expansion import paired_bootstrap, markdown_table

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'output/rsi_20261120'
CACHE = ROOT/'data/factor_extensions_20261109'
LATEST_CACHE = ROOT/'data/rsi_20261120'
END = date(2026, 11, 6)
EXTEND = date(2026, 11, 20)
RISK = [s for s in CORE_NAMES if s != '511260']
MODES = ('trend', 'reversal', 'cooling', 'acceleration')
MODE_NAMES = {'trend':'趋势强度', 'reversal':'趋势内反转', 'cooling':'仅惩罚过热', 'acceleration':'RSI改善'}
SOURCES = ['https://www.tradingview.com/support/solutions/43000502338-relative-strength-index-rsi/',
           'https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI']


def write_json(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


@dataclass(frozen=True)
class Case:
    name: str
    method: str = 'wilder'
    window: int = 14
    mode: str = 'trend'
    strength: float = 0.
    scope: str = 'all'
    replace_ma: bool = False


def cases():
    result = [Case('default')]
    for method in ('sma', 'wilder'):
        for window in (7, 14, 28):
            for mode in MODES:
                for strength in (.25, .5, 1.):
                    result.append(Case(f'{method}{window}_{mode}_a{strength:g}', method, window, mode, strength))
    for mode in MODES:
        for strength in (.25, .5):
            result.append(Case(f'wilder14_{mode}_ashare_a{strength:g}', 'wilder', 14, mode, strength, 'ashare'))
    for strength in (.25, .5, 1.):
        result.append(Case(f'wilder14_reversal_replace_ma_a{strength:g}', 'wilder', 14, 'reversal', strength, replace_ma=True))
    return result


def label(case):
    if case.name == 'default': return 'CL120默认'
    return f'{case.method.upper()}{case.window} {MODE_NAMES[case.mode]} a={case.strength:g}' + (' A股' if case.scope=='ashare' else '') + (' 替换MA' if case.replace_ma else '')


def plan():
    return {'research_date':'2026-11-20', 'main_end':END, 'extension_end':EXTEND,
            'case_count':len(cases()), 'cases':[asdict(c) for c in cases()],
            'base_config':asdict(BASE), 'universe':CORE_NAMES, 'satellite':SAT,
            'scope':'主网格作用于16只非国债核心ETF，A股限制与替换MA另列；红利、IVOL、日历和风控参数保持固定',
            'formula':'新评分=CL120评分+a*当日原动量评分截面标准差D*U；缺因子加0，不新增资格过滤',
            'utilities':{'trend':'clip((RSI-50)/25,-1,1)', 'reversal':'-clip((RSI-50)/25,-1,1)',
                         'cooling':'-clip((RSI-70)/30,0,1)', 'acceleration':'clip((RSI-RSI[t-5])/20,-1,1)'},
            'wilder':'价格变动分正负；首个完整N变动以SMA种子、其后alpha=1/N；缺报价重新预热；平盘50',
            'cash_starts':['2018-01-01','2022-01-01'], 'one_way_cost_bps':[5,20],
            'periods':['2018-2021','2022-2024','2025-2026-11-06'],
            'nomination':'两个现金起点、两费用下年化和夏普均不低于默认；5bp两起点回撤<=10%，且2022回撤不比默认恶化0.5个百分点以上；通过者取2018夏普最高。无通过则默认。只登记研究结果，不自动改实盘',
            'walk':'2022起每年按2018至年初前20交易日的20bp历史表现选；至少504日、回撤<=10%、年化不低于默认且夏普至少高0.05；统一重放评分路径与缓冲，不拼接虚拟持仓',
            'prediction':'未来收益从下一交易日收盘开始；5/20日RankIC；按年扩展训练、只用已到期标签预测，比较基准评分与加RSI模型的预测R2',
            'multiplicity':'全部83种新增场景配对块抽样20/60/120日；仅覆盖本轮搜索，不覆盖此前研究；历史各段已观察',
            'extension':'先冻结主区间代表与提名，再获取截至11/20增量；仅追加观察，不据此改选参数',
            'sources':SOURCES, 'protected_before':protected()}


def read_prices(end=END, folder=CACHE):
    values = {s:load_symbol(s, date(2010,1,1), end, folder, offline=True) for s in [*CORE_NAMES,SAT]}
    return pd.DataFrame(values, index=values['510300'].index).rename_axis('date')


def rsi_utility(raw, mode):
    if mode == 'trend': return ((raw-50)/25).clip(-1,1)
    if mode == 'reversal': return -((raw-50)/25).clip(-1,1)
    if mode == 'cooling': return -((raw-70)/30).clip(0,1)
    if mode == 'acceleration': return ((raw-raw.shift(5))/20).clip(-1,1)
    raise ValueError('Unknown RSI scoring mode')


def feature_frames(core, *, use_cache=False):
    store=FactorStore(OUT/'factor_cache.sqlite') if use_cache else None
    result={}
    for method in ('sma','wilder'):
        for window in (7,14,28):
            spec=FactorSpec('rsi' if method=='sma' else 'rsi_wilder',window)
            raw=store.get(core,spec) if store else compute_factor(core,spec)
            for mode in MODES: result[(method,window,mode)]=rsi_utility(raw,mode)
    return result


def components(core):
    values={name:compute_factor(core,FactorSpec(kind,n)) for name,kind,n in [
        ('short','momentum',20),('long','momentum',40),('vol','volatility',60),('ma','ma_zscore',20),('trend40','trend',40)]}
    score=risk_score_components(values['short'],values['long'],values['vol'],BASE,values['ma'])
    return {**score, **values}


def case_score(comp, frames, case):
    original=comp['base_score'] if case.replace_ma else comp['score']
    if case.name=='default': return original.copy()
    u=frames[(case.method,case.window,case.mode)].copy()
    u['511260']=np.nan
    if case.scope=='ashare': u.loc[:,[s for s in u if s not in A_BASE]]=np.nan
    spread=comp['base_score'].drop(columns='511260').std(axis=1,ddof=1).fillna(0.)
    return original + u.fillna(0.).mul(spread*case.strength,axis=0)


def compose(prices, core_targets, sat_states):
    return defensive_funding_targets(prices,core_targets,sat_states,SAT,.30)


def context(prices, *, use_cache=False):
    core=prices[list(CORE_NAMES)];rotation=state()
    bundle=portfolio_signals(core,prices[[SAT]],rotation,BASE,hypothetical_satellite_history=True,
                             hypothetical_universe_history=True,hypothetical_ivol_history=True)
    return core,rotation,bundle,CoreAudit(core,rotation),feature_frames(core,use_cache=use_cache),components(core)


def add_check(checks,name,error,tolerance=1e-10):
    checks.append({'check':name,'error':float(error),'passed':bool(error<tolerance)})


def select_nominee(table):
    base=table.loc[table.case.eq('default')].set_index(['start','cost'])
    passing=[]
    for name,g in table.groupby('case'):
        if name=='default': continue
        x=g.set_index(['start','cost']).reindex(base.index)
        if (x.annual>=base.annual-1e-12).all() and (x.sharpe>=base.sharpe-1e-12).all() and (x.loc[pd.IndexSlice[:,5.],'drawdown']>=-.10).all() and x.at[('2022-01-01',5.),'drawdown']>=base.at[('2022-01-01',5.),'drawdown']-.005:
            passing.append(name)
    if not passing:return 'default',passing
    main=table.loc[table.start.eq('2018-01-01')&table.cost.eq(5)].set_index('case')
    return main.loc[passing].sharpe.idxmax(),passing


def walk_score(core,comp,frames,entries,daily_stress):
    result=comp['score'].copy();choices=[]
    for year in range(2022,2027):
        before=core.index[core.index<pd.Timestamp(year,1,1)]
        cut=before[-21]
        train=daily_stress.loc[:cut]
        stats=pd.DataFrame({name:statistics(train[name]) for name in train}).T
        b=stats.loc['default']
        eligible=stats.loc[(stats.days>=504)&(stats.drawdown>=-.10)&(stats.annual>=b.annual)&(stats.sharpe>=b.sharpe+.05)]
        chosen=eligible.sharpe.idxmax() if len(eligible) else 'default'
        case=next(c for c in entries if c.name==chosen)
        use=(core.index>=pd.Timestamp(year,1,1))&(core.index<pd.Timestamp(year+1,1,1))
        result.loc[use]=case_score(comp,frames,case).loc[use]
        choices.append({'year':year,'train_end':cut,'case':chosen,'training_days':len(train),
                        'training_sharpe_20bp':float(stats.at[chosen,'sharpe'])})
    return result,choices


def diagnostic_samples(prices,bundle,comp,utility,horizon,scope):
    core=prices[list(CORE_NAMES)]
    future=core.shift(-(horizon+1))/core.shift(-1)-1
    days=weekly_signal_dates(core.index)
    rows=[];ics=[]
    for day in days[days>=pd.Timestamp('2018-01-01')]:
        valid=utility.loc[day,RISK].notna()&future.loc[day,RISK].notna()&comp['score'].loc[day,RISK].notna()
        if scope=='eligible':
            valid &= comp['long'].loc[day,RISK].gt(0)&comp['vol'].loc[day,RISK].gt(0)
            valid &= comp['trend40'].loc[day,RISK].gt(0)
            valid &= bundle.ivol_filter.eligibility_mask.loc[day,RISK]
        names=valid.index[valid]
        if len(names)<4:continue
        x=utility.loc[day,names];y=future.loc[day,names];base=comp['score'].loc[day,names]
        if x.std()<1e-12 or y.std()<1e-12:continue
        spread=float(comp['base_score'].loc[day,RISK].std(ddof=1))
        augmented=base+.25*spread*x
        ics.append({'date':day,'rank_ic':float(x.rank().corr(y.rank())),
                    'base_ic':float(base.rank().corr(y.rank())),
                    'augmented_ic_a025':float(augmented.rank().corr(y.rank())),
                    'score_correlation':float(x.rank().corr(base.rank())),
                    'ma_correlation':float(x.rank().corr((-comp['ma'].loc[day,names]).rank()))})
        end=prices.index[prices.index.get_loc(day)+horizon+1]
        rows.extend({'date':day,'label_end':end,'symbol':s,'base':float(base[s]),'x':float(x[s]),'y':float(y[s])} for s in names)
    return pd.DataFrame(rows),pd.DataFrame(ics)


def predictive_metrics(sample):
    sse_b=sse_a=null=0.;observations=0
    for year in range(2019,2027):
        first=pd.Timestamp(year,1,1);last=pd.Timestamp(year+1,1,1)
        train=sample.loc[sample.label_end<first];test=sample.loc[sample.date.ge(first)&sample.date.lt(last)]
        if len(train)<200 or test.empty:continue
        for columns in (['base'],['base','x']):
            center=train[columns].mean();scale=train[columns].std().replace(0,1)
            x=np.column_stack([np.ones(len(train)),(train[columns]-center)/scale])
            xt=np.column_stack([np.ones(len(test)),(test[columns]-center)/scale])
            coef=np.linalg.lstsq(x,train.y,rcond=None)[0]
            error=float(np.square(test.y-xt@coef).sum())
            if len(columns)==1:sse_b+=error
            else:sse_a+=error
        null+=float(np.square(test.y-train.y.mean()).sum());observations+=len(test)
    return {'base_predictive_r2':1-sse_b/null if null else None,
            'augmented_predictive_r2':1-sse_a/null if null else None,
            'incremental_predictive_r2':1-sse_a/sse_b if sse_b else None,'oos_observations':observations}


def diagnostics(prices,bundle,comp,frames):
    rows=[];weekly=[]
    for i,(key,u) in enumerate(frames.items()):
        for horizon in (5,20):
            for scope in ('all','eligible'):
                sample,ic=diagnostic_samples(prices,bundle,comp,u,horizon,scope)
                if sample.empty:continue
                rows.append({'method':key[0],'window':key[1],'mode':key[2],'horizon':horizon,'scope':scope,
                    'weeks':len(ic),'mean_rank_ic':float(ic.rank_ic.mean()),'base_mean_rank_ic':float(ic.base_ic.mean()),
                    'augmented_mean_rank_ic_a025':float(ic.augmented_ic_a025.mean()),
                    'original_score_rank_correlation':float(ic.score_correlation.mean()),
                    'ma_reversion_rank_correlation':float(ic.ma_correlation.mean()),**predictive_metrics(sample)})
                weekly.extend({'method':key[0],'window':key[1],'mode':key[2],'horizon':horizon,'scope':scope,**r} for r in ic.to_dict('records'))
        print(f'diagnostic {i+1}/{len(frames)} {key}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'factor_diagnostics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(weekly).to_csv(OUT/'weekly_ic.csv',index=False,encoding='utf-8-sig')


def paired_intervals(daily,name):
    rng=np.random.default_rng(20261120);a=daily['default'].to_numpy();b=daily[name].to_numpy();rows=[]
    for block in (20,60,120):
        differences=[]
        for _ in range(1000):
            starts=rng.integers(0,len(a),size=int(np.ceil(len(a)/block)))
            idx=((starts[:,None]+np.arange(block))%len(a)).ravel()[:len(a)]
            x=statistics(a[idx]);y=statistics(b[idx]);differences.append([y['annual']-x['annual'],y['sharpe']-x['sharpe']])
        d=np.asarray(differences)
        rows.append({'case':name,'block':block,'reps':1000,'annual_delta_p025':float(np.quantile(d[:,0],.025)),
                     'annual_delta_p975':float(np.quantile(d[:,0],.975)),'sharpe_delta_p025':float(np.quantile(d[:,1],.025)),
                     'sharpe_delta_p975':float(np.quantile(d[:,1],.975))})
    return rows


def ablations(prices,rotation,bundle,case):
    """对历史最好者与默认作相同删资产诊断，不再据此挑参数。"""
    rows=[]
    for label_name,removed in [('drop_gold',['518880']),('drop_bank',['512800']),('drop_nasdaq',['159941']),
                               ('drop_chip_AI',['159995','515070']),('no_satellite',[])]:
        names=[s for s in CORE_NAMES if s not in removed];core=prices[names]
        eligible=market_ivol_filter(core,rotation.ivol_policy,hypothetical_history=True).eligibility_mask
        engine=CoreAudit(core,rotation);frames=feature_frames(core);comp=components(core)
        for c in (Case('default'),case):
            t=engine.signals(BASE,score_override=case_score(comp,frames,c),eligibility_mask=eligible)
            t=t.reindex(columns=prices.columns,fill_value=0.) if label_name=='no_satellite' else compose(prices,t,bundle.satellite_states)
            for start in ('2018-01-01','2022-01-01'):
                led=simulate(prices,t,pd.Timestamp(start),cost_bps=5.)
                rows.append({'ablation':label_name,'case':c.name,'start':start,**statistics(led.returns)})
    pd.DataFrame(rows).to_csv(OUT/'asset_ablations.csv',index=False)


def execute():
    OUT.mkdir(exist_ok=True,parents=True)
    if not (OUT/'plan.json').exists():write_json(OUT/'plan.json',plan())
    before=protected();prices=read_prices()
    core,rotation,bundle,engine,frames,comp=context(prices,use_cache=True)
    entries=cases();targets={};records=[];daily={};recent={};stress={};checks=[];holdings=[]
    for i,case in enumerate(entries):
        scores=case_score(comp,frames,case)
        ct=engine.signals(BASE,score_override=scores,eligibility_mask=bundle.ivol_filter.eligibility_mask)
        target=compose(prices,ct,bundle.satellite_states);targets[case.name]=target
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                ledger=simulate(prices,target,pd.Timestamp(start),cost_bps=cost)
                records.append({'case':case.name,'start':start,'cost':cost,'label':label(case),**asdict(case),
                                'annual_turnover':float(ledger.turnover.sum()*252/len(ledger.dates)),**statistics(ledger.returns)})
                if start=='2018-01-01': (daily if cost==5 else stress)[case.name]=ledger.series()
                elif cost==5:recent[case.name]=ledger.series()
                if start=='2022-01-01' and cost==5:
                    w=pd.DataFrame(ledger.weights,index=ledger.dates,columns=prices.columns)
                    holdings.extend({'case':case.name,'symbol':s,'average_weight':float(w[s].mean()),'held_days':int(w[s].gt(1e-8).sum())} for s in prices)
        risky=target[RISK]
        checks.append({'check':case.name+'_constraints','passed':bool(target.sum(axis=1).max()<=1+1e-9 and risky.sum(axis=1).max()<=.70+1e-9 and risky.max().max()<=.30+1e-9 and risky.gt(1e-8).sum(axis=1).max()<=3)})
        print(f'case {i+1}/{len(entries)} {case.name}',flush=True)
    table=pd.DataFrame(records);table.to_csv(OUT/'all_results.csv',index=False,encoding='utf-8-sig')
    daily=pd.DataFrame(daily);recent=pd.DataFrame(recent);stress=pd.DataFrame(stress)
    daily.to_csv(OUT/'daily_2018.csv');recent.to_csv(OUT/'daily_2022.csv');stress.to_csv(OUT/'daily_2018_20bp.csv')
    pd.DataFrame(holdings).to_csv(OUT/'holdings.csv',index=False)
    main=table.loc[table.start.eq('2018-01-01')&table.cost.eq(5)].set_index('case')
    best=main.drop('default').sharpe.idxmax()
    best_recent=table.loc[table.start.eq('2022-01-01')&table.cost.eq(5)&table.case.ne('default')].sort_values('sharpe',ascending=False).iloc[0]['case']
    nominee,passing=select_nominee(table)
    family_best={mode:main.loc[main['mode'].eq(mode)&main.index.to_series().ne('default')].sharpe.idxmax() for mode in MODES}
    representatives=list(dict.fromkeys(['default',best,best_recent,nominee,*family_best.values(),
                         *[f'wilder14_{m}_a0.25' for m in MODES]]))
    selection={'best_2018_sharpe':best,'best_2022_sharpe':best_recent,'nominee':nominee,'passing':passing,
               'family_best':family_best,'representatives':representatives,'selected_using_data_through':str(END)}
    write_json(OUT/'frozen_selection.json',selection)
    # 不同起点的现金账本独立；以下分段仅用于对完整2018路径作归因。
    segments=[{'case':n,'period':period,**statistics(daily.loc[a:b,n])} for n in daily for period,a,b in
              [('2018-2021','2018','2021'),('2022-2024','2022','2024'),('2025+','2025',str(END))]]
    pd.DataFrame(segments).to_csv(OUT/'segments.csv',index=False)
    for name in representatives:
        case=next(c for c in entries if c.name==name)
        scores=case_score(comp,frames,case)
        production=risk_controlled_signals(core,BASE,**rotation.signal_kwargs(core),
                    score_overlay=scores-comp['score'],eligibility_mask=bundle.ivol_filter.eligibility_mask)
        target=compose(prices,production,bundle.satellite_states)
        add_check(checks,name+'_production_targets',np.nanmax(np.abs((targets[name]-target).to_numpy())))
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                formal=run_backtest(prices,target,date.fromisoformat(start),END,fee_bps=0,slippage_bps=cost)
                fast=simulate(prices,targets[name],pd.Timestamp(start),cost_bps=cost)
                add_check(checks,name+'_ledger_'+start+'_'+str(cost),np.max(np.abs(formal.equity.daily_return.to_numpy()-fast.returns)))
        target.to_csv(OUT/f'targets_{name}.csv')
    add_check(checks,'default_matches_frozen_CL120',np.max(np.abs((targets['default']-bundle.signals).to_numpy())))
    # 非周末截断不能把尾部未执行目标当成过去信号。
    for cut in (pd.Timestamp('2020-08-19'),pd.Timestamp('2024-09-27'),pd.Timestamp('2026-06-26')):
        pc=prices.loc[:cut];cc,ss,bb,ee,ff,co=context(pc)
        executable=targets['default'].index[prices.index.searchsorted(targets['default'].index,side='right')<=prices.index.get_loc(cut)]
        for name in list(dict.fromkeys(['default',best,nominee])):
            case=next(c for c in entries if c.name==name)
            prefix=compose(pc,ee.signals(BASE,score_override=case_score(co,ff,case),eligibility_mask=bb.ivol_filter.eligibility_mask),bb.satellite_states)
            add_check(checks,name+'_prefix_'+str(cut.date()),np.max(np.abs((prefix.loc[executable]-targets[name].loc[executable]).to_numpy())))
    shocked=prices.copy();cut=pd.Timestamp('2024-09-27');shocked.loc[shocked.index>cut]*=np.linspace(.3,2.8,len(shocked.columns))
    sc,sr,sb,se,sf,sp=context(shocked)
    for name in list(dict.fromkeys(['default',best,nominee])):
        case=next(c for c in entries if c.name==name)
        changed=compose(shocked,se.signals(BASE,score_override=case_score(sp,sf,case),eligibility_mask=sb.ivol_filter.eligibility_mask),sb.satellite_states)
        add_check(checks,name+'_future_shock',np.max(np.abs((changed.loc[:cut]-targets[name].loc[:cut]).to_numpy())))
    # 年度选参保留同一缓冲状态并完整重新计算，避免拼接候选的虚拟持仓。
    score,choices=walk_score(core,comp,frames,entries,stress)
    walk=compose(prices,engine.signals(BASE,score_override=score,eligibility_mask=bundle.ivol_filter.eligibility_mask),bundle.satellite_states)
    write_json(OUT/'walk_choices.json',choices);walk.to_csv(OUT/'targets_walk.csv')
    walk_rows=[]
    for cost in (5.,20.):
        led=simulate(prices,walk,pd.Timestamp('2022-01-01'),cost_bps=cost)
        walk_rows.append({'case':'walk','start':'2022-01-01','cost':cost,**statistics(led.returns)})
        if cost==5:pd.DataFrame({'default':recent['default'],'walk':led.series()}).to_csv(OUT/'walk_daily.csv')
    pd.DataFrame(walk_rows).to_csv(OUT/'walk_results.csv',index=False)
    write_json(OUT/'bootstrap.json',paired_bootstrap(daily[['default',*[c.name for c in entries if c.name!='default']]].to_numpy(),reps=1500))
    write_json(OUT/'paired_intervals.json',paired_intervals(daily,best))
    ablations(prices,rotation,bundle,next(c for c in entries if c.name==best))
    diagnostics(prices,bundle,comp,frames)
    checks.append({'check':'default_state_live_unchanged','passed':before==protected()})
    write_json(OUT/'checks.json',checks)
    write_json(OUT/'manifest.json',{'end':END,'all_checks_passed':all(c['passed'] for c in checks),'checks':len(checks),
        'protected_before':before,'protected_after':protected(),'default_changed':False,
        'price_hash':hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
        'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    if not all(c['passed'] for c in checks):raise AssertionError('RSI numeric validation failed')
    print(json.dumps(selection,ensure_ascii=False,indent=2),flush=True)


def download_latest():
    if not (OUT/'frozen_selection.json').exists():raise ValueError('Run and freeze the main study first')
    LATEST_CACHE.mkdir(exist_ok=True,parents=True);client=None;status=[]
    for symbol in [*CORE_NAMES,SAT]:
        old=_normalize_history(pd.read_csv(CACHE/f'tushare_{symbol}_hfq.csv'),symbol)
        old=old.loc[:pd.Timestamp(END)]
        path=LATEST_CACHE/f'tushare_{symbol}_hfq.csv';meta=path.with_suffix('.json')
        if path.exists() and meta.exists() and json.loads(meta.read_text(encoding='utf-8'))['end']==str(EXTEND):
            status.append({'symbol':symbol,'cached':True});continue
        if client is None:client=Client()
        params={'ts_code':symbol+('.SH' if symbol.startswith('5') else '.SZ'),
                'start_date':(END-timedelta(days=21)).strftime('%Y%m%d'),'end_date':EXTEND.strftime('%Y%m%d')}
        raw=_normalize_table(client.query('fund_daily',params,'ts_code,trade_date,close'),'close',symbol)
        adj=_normalize_table(client.query('fund_adj',params,'ts_code,trade_date,adj_factor'),'adj_factor',symbol)
        common=old.index.intersection(raw.index).intersection(adj.index)
        if not len(common) or not np.allclose(old.loc[common,'raw_close'],raw.loc[common],rtol=0,atol=1e-9) or not np.allclose(old.loc[common,'adj_factor'],adj.loc[common],rtol=1e-9):
            raise ValueError(f'{symbol}: historical snapshot revision; preserve main data')
        new=raw.loc[raw.index>pd.Timestamp(END)];factor=adj.reindex(new.index)
        if factor.isna().any() or len(new)==0:raise ValueError(f'{symbol}: incomplete extension factors/prices')
        added=pd.DataFrame({'raw_close':new,'adj_factor':factor,'close':new*factor});added.index.name='date'
        joined=pd.concat([old,added]);joined.to_csv(path)
        write_json(meta,{'source':'tushare','schema_version':2,'symbol':symbol,'adjust':'hfq','start':'2010-01-01','end':str(EXTEND)})
        status.append({'symbol':symbol,'first_new':new.index[0],'last_new':new.index[-1],'new_quotes':len(new),'past_unchanged':True})
        write_json(OUT/'download_status.json',status);print(f'extension {symbol}: {len(new)} new quotes to {new.index[-1].date()}',flush=True)
    write_json(OUT/'download_status.json',status)
    prices=read_prices(EXTEND,LATEST_CACHE)
    if prices.index[-1].date()!=EXTEND:raise ValueError('Latest market close not available')
    core,rotation,bundle,engine,frames,comp=context(prices)
    selected=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'));by_name={c.name:c for c in cases()}
    rows=[];daily={};checks=[]
    for name in selected['representatives']:
        target=compose(prices,engine.signals(BASE,score_override=case_score(comp,frames,by_name[name]),eligibility_mask=bundle.ivol_filter.eligibility_mask),bundle.satellite_states)
        old=pd.read_csv(OUT/f'targets_{name}.csv',index_col=0,parse_dates=True)
        add_check(checks,name+'_unchanged_past',np.max(np.abs((target.loc[old.index]-old).to_numpy())))
        for cost in (5.,20.):
            led=simulate(prices,target,pd.Timestamp('2018-01-01'),cost_bps=cost)
            x=led.series().loc[led.dates>pd.Timestamp(END)]
            rows.append({'case':name,'cost':cost,'start':str(x.index[0].date()),'end':str(x.index[-1].date()),**statistics(x)})
            if cost==5:daily[name]=x
    pd.DataFrame(rows).to_csv(OUT/'extension_results.csv',index=False);pd.DataFrame(daily).to_csv(OUT/'extension_daily.csv')
    write_json(OUT/'extension_checks.json',checks)
    if not all(c['passed'] for c in checks):raise AssertionError('Extension altered historical targets')


def behavior_audit():
    """事后解释交易变化集中度，保持84项既定搜索及冻结提名不变。"""
    from research_ivol_impact import selection_states
    selection=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    winner=next(c for c in cases() if c.name==selection['best_2018_sharpe'])
    prices=read_prices();core,rotation,bundle,engine,frames,comp=context(prices)
    a=pd.read_csv(OUT/'targets_default.csv',index_col=0,parse_dates=True)
    b=pd.read_csv(OUT/f'targets_{winner.name}.csv',index_col=0,parse_dates=True)
    previous,resets,_=selection_states(core,bundle.core_signals,rotation)
    rebuild=risk_controlled_signals(core,BASE,eligibility_mask=bundle.ivol_filter.eligibility_mask,selection_resets=resets)
    score=case_score(comp,frames,winner)
    shadow=risk_controlled_signals(core,BASE,score_overlay=score-comp['score'],
                                   eligibility_mask=bundle.ivol_filter.eligibility_mask,selection_resets=resets)
    checks=[];add_check(checks,'same_previous_selection_rebuilds_CL120',np.max(np.abs((rebuild-bundle.core_signals).to_numpy())))
    execution_pos=prices.index.searchsorted(a.index,side='right')
    usable=execution_pos<len(prices)
    on=pd.Series(False,index=a.index);on.loc[usable]=prices.index[execution_pos[usable]]>=pd.Timestamp('2018-01-01')
    different=(a-b).abs().max(axis=1)>1e-10
    direct=shadow[RISK].gt(1e-10).ne(bundle.core_signals[RISK].gt(1e-10)).any(axis=1)
    raw=compute_factor(core,FactorSpec('rsi' if winner.method=='sma' else 'rsi_wilder',winner.window))
    def weights(row):
        return '、'.join(CORE_NAMES.get(s,'红利低波')+' '+s+f' {w:.1%}' for s,w in row.items() if w>1e-8)
    events=[]
    for signal in a.index[on&(different|direct)]:
        pos=prices.index.searchsorted(signal,side='right')
        hot=[s for s in RISK if pd.notna(raw.at[signal,s]) and raw.at[signal,s]>70]
        events.append({'signal_date':signal,'execution_date':prices.index[pos],'direct_selection_changed':bool(direct.at[signal]),
             'independent_target_changed':bool(different.at[signal]),'hot_assets':','.join(s+f':{raw.at[signal,s]:.2f}' for s in hot),
             'default_target':weights(a.loc[signal]),'RSI_target':weights(b.loc[signal])})
    pd.DataFrame(events).to_csv(OUT/'behavior_events.csv',index=False,encoding='utf-8-sig')
    daily=pd.read_csv(OUT/'daily_2018.csv',index_col=0,parse_dates=True)
    equivalent=[n for n in daily if n!='default' and float((daily[n]-daily['default']).abs().max())<1e-12]
    years=[]
    for year,g in daily.groupby(daily.index.year):
        base=float((1+g['default']).prod()-1);new=float((1+g[winner.name]).prod()-1)
        years.append({'year':int(year),'default_return':base,'RSI_return':new,'net_difference':new-base})
    pd.DataFrame(years).to_csv(OUT/'yearly_contribution.csv',index=False)
    all_counts=[]
    for start in ('2018-01-01','2022-01-01'):
        window=on.copy();window.loc[usable] &= prices.index[execution_pos[usable]]>=pd.Timestamp(start)
        all_counts.append({'start':start,'executable_weekly_targets':int(window.sum()),
                           'independent_different_targets':int((window&different).sum()),
                           'same_state_direct_selection_changes':int((window&direct).sum())})
    write_json(OUT/'behavior_summary.json',{'winner':winner.name,'counts':all_counts,'return_identical_to_default':equivalent,
               'nonzero_gain_years':[r['year'] for r in years if abs(r['net_difference'])>1e-10],
               'direct_signal_dates':[str(d.date()) for d in a.index[on&direct]]})
    write_json(OUT/'behavior_checks.json',checks)
    if not all(c['passed'] for c in checks):raise AssertionError('Behavior audit reconstruction failed')


def figures():
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    assets=ROOT/'docs/assets';assets.mkdir(exist_ok=True,parents=True)
    selection=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    by_name={c.name:c for c in cases()}
    daily=pd.read_csv(OUT/'daily_2022.csv',index_col=0,parse_dates=True)
    walk=pd.read_csv(OUT/'walk_daily.csv',index_col=0,parse_dates=True)
    names=list(dict.fromkeys(['default',selection['best_2018_sharpe'],selection['best_2022_sharpe'],selection['nominee']]))
    fig,axes=plt.subplots(2,1,figsize=(13,9),sharex=True)
    for name in names:
        nav=(1+daily[name]).cumprod();dd=nav/nav.cummax().clip(lower=1)-1
        axes[0].plot(nav.index,(nav-1)*100,label=label(by_name[name]),linewidth=2 if name=='default' else 1.4)
        axes[1].plot(dd.index,dd*100,label=label(by_name[name]),linewidth=1.4)
    nav=(1+walk['walk']).cumprod();dd=nav/nav.cummax().clip(lower=1)-1
    axes[0].plot(nav.index,(nav-1)*100,label='按过去年度选参',linestyle='--',color='#777777')
    axes[1].plot(dd.index,dd*100,linestyle='--',color='#777777')
    axes[0].set(ylabel='累计净收益 %',title=f'同一CL120池与风控｜2022现金起点至{END}｜单边5bp')
    axes[1].set(ylabel='收盘回撤 %',xlabel='日期')
    for ax in axes:ax.grid(alpha=.2)
    axes[0].legend(fontsize=9);fig.tight_layout();fig.savefig(assets/'rsi_equity_20261120.png',dpi=160);plt.close(fig)
    table=pd.read_csv(OUT/'all_results.csv');main=table.loc[table.start.eq('2018-01-01')&table.cost.eq(5)].set_index('case')
    baseline=float(main.at['default','sharpe'])
    fig,axes=plt.subplots(2,4,figsize=(16,8.4))
    cells=[]
    for i,method in enumerate(('sma','wilder')):
        for j,mode in enumerate(MODES):
            matrix=np.array([[main.at[f'{method}{w}_{mode}_a{a:g}','sharpe']-baseline for a in (.25,.5,1.)] for w in (7,14,28)])
            cells.append((axes[i,j],matrix,method,mode))
    maximum=max(abs(v).max() for _,v,_,_ in cells)
    for ax,matrix,method,mode in cells:
        im=ax.imshow(matrix,cmap='RdYlGn',vmin=-maximum,vmax=maximum,aspect='auto')
        for i in range(3):
            for j in range(3):ax.text(j,i,f'{matrix[i,j]:+.3f}',ha='center',va='center',fontsize=10)
        ax.set(xticks=range(3),xticklabels=['0.25','0.5','1.0'],yticks=range(3),yticklabels=['7日','14日','28日'],
               xlabel='叠加强度 a',title=method.upper()+' '+MODE_NAMES[mode])
    fig.suptitle('RSI参数敏感性｜2018起点夏普相对CL120的变化｜其余规则固定',fontsize=14)
    fig.subplots_adjust(left=.05,right=.91,bottom=.08,top=.9,hspace=.38,wspace=.27)
    fig.colorbar(im,cax=fig.add_axes([.93,.16,.014,.65]),label='夏普差值')
    fig.savefig(assets/'rsi_sensitivity_20261120.png',dpi=160);plt.close(fig)
    diagnostic=pd.read_csv(OUT/'factor_diagnostics.csv')
    d=diagnostic.loc[diagnostic.method.eq('wilder')&diagnostic.window.eq(14)&diagnostic.scope.eq('eligible')]
    fig,axes=plt.subplots(1,2,figsize=(13,5.3));x=np.arange(4);width=.34
    for offset,horizon,color in [(-width/2,5,'#2364aa'),(width/2,20,'#d97a32')]:
        rows=d.loc[d.horizon.eq(horizon)].set_index('mode').loc[list(MODES)]
        for ax,key in [(axes[0],'mean_rank_ic'),(axes[1],'incremental_predictive_r2')]:
            values=rows[key].to_numpy()*100
            ax.bar(x+offset,values,width,label=f'{horizon}日',color=color)
            for k,value in enumerate(values):ax.text(k+offset,value,f'{value:.2f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=8)
    for ax in axes:
        ax.set_xticks(x,[MODE_NAMES[m] for m in MODES]);ax.axhline(0,color='#666666',linewidth=.8);ax.grid(axis='y',alpha=.2);ax.legend()
    axes[0].set(title='RSI本身的截面Rank IC',ylabel='IC × 100')
    axes[1].set(title='在默认评分之外增加RSI的预测R²',ylabel='增量预测R² %')
    fig.suptitle('Wilder RSI14｜仅在原趋势与IVOL合格池检验｜按年只用已到期标签训练',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(assets/'rsi_predictive_20261120.png',dpi=160);plt.close(fig)


def report():
    table=pd.read_csv(OUT/'all_results.csv');by_name={c.name:c for c in cases()}
    selected=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    bootstrap=json.loads((OUT/'bootstrap.json').read_text(encoding='utf-8'))
    intervals=json.loads((OUT/'paired_intervals.json').read_text(encoding='utf-8'))
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    for name in ('behavior_checks.json','extension_checks.json'):
        if (OUT/name).exists():checks+=json.loads((OUT/name).read_text(encoding='utf-8'))
    diagnostics_table=pd.read_csv(OUT/'factor_diagnostics.csv');segments=pd.read_csv(OUT/'segments.csv')
    p=lambda v:f'{v:.2%}'
    names=list(dict.fromkeys(['default',selected['best_2018_sharpe'],selected['best_2022_sharpe'],selected['nominee'],*selected['family_best'].values()]))
    rows=[]
    for start in ('2018-01-01','2022-01-01'):
        for name in names:
            for cost in (5.,20.):
                row=table.loc[table.case.eq(name)&table.start.eq(start)&table.cost.eq(cost)].iloc[0]
                rows.append([start[:4],label(by_name[name]),f'{cost:g}',p(row.annual),f'{row.sharpe:.3f}',p(row.drawdown),p(row.annual_turnover)])
    performance=markdown_table(['现金起点','方案','单边成本bp','年化','零利率夏普','最大收盘回撤','年化双边换手'],rows)
    segment_table=markdown_table(['方案','区间','年化','夏普','最大回撤'],[[label(by_name[r['case']]),r['period'],p(r['annual']),f'{r["sharpe"]:.3f}',p(r['drawdown'])] for r in segments.loc[segments.case.isin(names)].to_dict('records')])
    diag=diagnostics_table.loc[diagnostics_table.method.eq('wilder')&diagnostics_table.window.eq(14)&diagnostics_table.scope.eq('eligible')]
    diag_table=markdown_table(['RSI用法','预测跨度','有效周数','RSI Rank IC','原评分IC','叠加0.25后的IC','原评分预测R²','加RSI预测R²','增量预测R²'],
          [[MODE_NAMES[r['mode']],str(r['horizon']),str(r['weeks']),f'{r["mean_rank_ic"]:.4f}',f'{r["base_mean_rank_ic"]:.4f}',f'{r["augmented_mean_rank_ic_a025"]:.4f}',p(r['base_predictive_r2']),p(r['augmented_predictive_r2']),p(r['incremental_predictive_r2'])] for r in diag.to_dict('records')])
    walk=pd.read_csv(OUT/'walk_results.csv');choices=json.loads((OUT/'walk_choices.json').read_text(encoding='utf-8'))
    walk_table=markdown_table(['成本bp','年化','夏普','最大回撤'],[[str(r['cost']),p(r['annual']),f'{r["sharpe"]:.3f}',p(r['drawdown'])] for r in walk.to_dict('records')])
    choice_table=markdown_table(['使用年份','训练截止','采用方案'],[[str(r['year']),str(r['train_end'])[:10],label(by_name[r['case']])] for r in choices])
    boot_table=markdown_table(['块长交易日','抽样次数','本轮家族最大均值增益p'],[[str(r['block']),str(r['reps']),f'{r["family_mean_gain_p"]:.3f}'] for r in bootstrap])
    ci_table=markdown_table(['块长','年化增量95%区间（百分点）','夏普增量95%区间'],[[str(r['block']),f'{r["annual_delta_p025"]*100:.2f} 至 {r["annual_delta_p975"]*100:.2f}',f'{r["sharpe_delta_p025"]:.3f} 至 {r["sharpe_delta_p975"]:.3f}'] for r in intervals])
    ablation=pd.read_csv(OUT/'asset_ablations.csv');ab_rows=[]
    for (a,start),g in ablation.groupby(['ablation','start']):
        b=g.loc[g.case.eq('default')].iloc[0];v=g.loc[g.case.eq(selected['best_2018_sharpe'])].iloc[0]
        ab_rows.append([a,start[:4],p(v.annual-b.annual),f'{v.sharpe-b.sharpe:+.3f}',p(b.drawdown),p(v.drawdown)])
    ab_table=markdown_table(['共同删减','现金起点','RSI年化增量','夏普增量','删减后默认回撤','删减后RSI回撤'],ab_rows)
    if (OUT/'extension_results.csv').exists():
        ext=pd.read_csv(OUT/'extension_results.csv')
        ext_table=markdown_table(['方案','成本bp','新增交易日','累计收益','局部最大回撤'],[[label(by_name[r['case']]),str(r['cost']),str(r['days']),p(r['total']),p(r['drawdown'])] for r in ext.to_dict('records')])
        extension=f'主区间代表已先写入冻结名单，再取11月9日至11月20日追加行情；延续2018账户真实模拟旧仓，不从11月9日清零、不据此改选参数。短窗不能证明稳定性。\n\n{ext_table}'
        eb=ext.loc[ext.case.eq('default')&ext.cost.eq(5)].iloc[0]
        en=ext.loc[ext.case.eq(selected['best_2018_sharpe'])&ext.cost.eq(5)].iloc[0]
        extension+=f'\n\n默认与Wilder28过热候选在这{int(eb.days)}日内的5bp净收益分别为{eb.total:.4%}、{en.total:.4%}。11月20日收盘信号尚未在本观察窗口内执行，未将其未来收益计入。'
    else:extension='主区间提名已经冻结，追加观察结果尚未生成；不以缺少追加结果认定候选有效。'
    nominee=selected['nominee'];winner=selected['best_2018_sharpe']
    behavior=json.loads((OUT/'behavior_summary.json').read_text(encoding='utf-8'))
    gain_years='、'.join(str(y) for y in behavior['nonzero_gain_years'])
    behavior_rows=pd.read_csv(OUT/'behavior_events.csv')
    behavior_table=markdown_table(['信号日','执行日','同旧仓下直接改变','原目标','RSI目标'],
        [[r['signal_date'][:10],r['execution_date'][:10],'是' if r['direct_selection_changed'] else '否',r['default_target'],r['RSI_target']] for r in behavior_rows.to_dict('records')])
    equal_pass=[n for n in selected['passing'] if n in behavior['return_identical_to_default']]
    decision=(f'共有{len(selected["passing"])}种方案通过事先固定的跨区间和成本门槛，研究提名为{label(by_name[nominee])}。'
              if nominee!='default' else '没有新增方案同时通过事先固定的跨区间、费用、收益、夏普及回撤门槛，研究结论保留CL120。')
    text=f'''# RSI评分对CL120的增量研究

研究日期2026年11月20日。{decision} 但其中{len(equal_pass)}种与默认逐日收益完全相同，不构成新增效果。2018历史夏普最高的新增方案是{label(by_name[winner])}，变化集中在{gain_years}年；参数邻域、预测证据和年度选参未形成可靠的稳定增益证据，建议继续观察、保留当前默认。本轮不改变CL120实盘评分、调仓状态或已发布计划。

## 算法和研究范围

固定当前17只核心ETF及512890红利卫星，主比较截至2026年11月6日。84个场景包括默认、72个主网格、8个仅A股叠加及3个替换原MA反转的对照。窗口7/14/28日，SMA与Wilder两种算法，叠加强度0.25/0.5/1。所有参数在运行前记录在本地plan.json，没有按结果追加网格。

RSI=100×平均上涨价格变动/(平均上涨变动＋平均下跌变动)。简单均值版沿用原因子；Wilder版新增为`FactorSpec("rsi_wilder", N)`，以首个完整N日价格变动的简单均值初始化，其后用alpha=1/N递推。平盘取50；只涨取100、只跌取0；缺报价不填充，重新预热。`FactorStore.get`按名称和窗口缓存，两种定义不混用。

主评分为`S_new=S_CL120+a×D_t×U_t`，D是当日原动量评分在非国债核心池的截面样本标准差，与原MA反转的量纲一致。RSI缺值加0。U分别为：趋势强度clip((RSI−50)/25,−1,1)，趋势内反转取其相反数，仅惩罚过热为−clip((RSI−70)/30,0,1)，RSI改善为clip((RSI−RSI[t−5])/20,−1,1)。不新增RSI硬资格、不提高风险预算，国债和红利不参加评分。替换MA组先移除原0.5MA反转加分，其他组均保留它。

Wilder RSI的平滑计算和14日惯例见[TradingView官方定义]({SOURCES[0]})。强趋势中RSI可长期保持高位，见[Fidelity指标说明]({SOURCES[1]})；因此“高RSI加分”和“过热扣分”是两个需用数据分别检验的假设，资料本身不证明本策略有超额收益。

## 收益和成本对照

全部为当前池、红利及IVOL的全历史假设回放，保留原公共缓冲锚点；2018/2022各自从现金开始。周末端收盘信号、下一交易日收盘执行，新仓只参与此后的收益。费用是单边综合成本5或20bp，含每次实际调仓成本与仓位漂移，不作免费每日再平衡。下表换手为买卖绝对权重之和的年化，即双边口径。

{performance}

![RSI收益和回撤对照](assets/rsi_equity_20261120.png)

完整336行结果保留在本地`all_results.csv`。筛选门槛要求两个现金起点、两个费用下年化和夏普均不低于默认；5bp两个起点回撤不超过10%，且2022回撤最多恶化0.5个百分点。满足者只按2018夏普提名，不自动切换策略。

## 改善集中在哪些交易

历史最好者在2018以来{behavior['counts'][0]['executable_weekly_targets']}次可执行周目标中，仅{behavior['counts'][0]['independent_different_targets']}次与默认目标不同；让决策前旧仓缓冲相同时，当前RSI直接改变选仓{behavior['counts'][0]['same_state_direct_selection_changes']}次，其余差异还包含缓冲延续。所有收益差异集中在{gain_years}年，两个回测起点都包含该年，不能将它们视为两个独立验证样本。2024年累计收益由29.46%提高至31.35%；其余年份逐日收益相同。

{behavior_table}

Wilder28过热惩罚的0.25和0.5两个版本与默认逐日收益完全相同；本样本只有强度1跨过选仓边界。它不是一片稳定改善的参数平台。上述事件及过热标的的RSI值保留在`behavior_events.csv`；未在看到这些年份后加入新参数。

两次直接变化分别是：3月8日黄金RSI28约71.85，轻微扣分使原边界发生变化，3月11日由黄金切换至沪深300；9月27日沪深300RSI28约73.52、创业板约70.35，两者均受惩罚，前者受罚更大，9月30日改选创业板。9月30日信号10月8日执行的后续差异来自缓冲路径，不能当成第三次独立RSI成功事件。过热扣分仍允许高RSI资产入选，没有“超过70必须卖出”的硬规则。

## 参数与行情阶段稳定性

![RSI参数敏感性](assets/rsi_sensitivity_20261120.png)

热图每格是2018起点夏普相对默认的变化。周边窗口或强度同步改善，才比孤立最高点更值得观察；下列分段沿用2018完整路径，不能与2022独立现金起点混为同一账本。所有历史阶段已经参与前期研究，均不能声称全新盲测。

{segment_table}

## IC和预测R方

未来收益从信号后的下一交易日收盘开始，分别观察其后5和20个交易日。IC是各周截面秩相关，合格池只含当时通过原趋势、动量、波动及IVOL资格的资产；至少4只且横截面有变化才取样。两种RSI方法、全部窗口、全池与合格池的96项统计均保留在本地，下面展示常用Wilder14。高IC表示排序关联，不等同于可实现组合收益。

{diag_table}

![RSI的预测证据](assets/rsi_predictive_20261120.png)

预测模型对照`未来收益~常数+原评分`与`未来收益~常数+原评分+RSI效用`。每年只用此前已经到期的标签拟合OLS，标准化也只从训练集计算；下一年逐点预测。预测R²以训练集历史均值为基准；增量预测R²=1−SSE_加RSI/SSE_原评分。负数表示增添RSI反而增加预测误差。它不是同期价格拟合R²，也没有把未来收益或未来转折点用于交易。

Wilder14趋势分在合格池的5/20日IC约0.056/0.066，但在原评分外增加它后，预测误差略增。它与原评分的秩相关约0.54，与MA偏离反转方向约−0.78，信息有明显重叠。趋势和反转效用互为相反数，OLS可通过系数符号转换，所以这两者预测R²相同，固定方向交易的结果仍不同。过热与RSI改善的取样周数也不同，应按各自匹配样本评价。

历史最好者使用的Wilder28过热效用，在5/20日的增量预测R²分别为1.36%/2.44%，是值得留存的正面线索；但每个跨度只有58个具备截面差异的合格周，且其5日增广模型绝对预测R²仍为−0.80%。OLS可自行选择正负系数，正R²仅提示变量可能含有信息，并未验证固定“过热扣分”方向；该效用20日IC略负，约−0.0065。有限预测线索与仅两次直接改变的交易样本，尚不足以认定稳定新增优势。

## 按过去选参与多重比较

从2022起每年仅使用年初前20个交易日之前的历史20bp净收益；至少504日、回撤<=10%、年化不低于默认且夏普至少高0.05，才从既定84项中选择，否则继续默认。年度切换后统一重放评分及缓冲，保留上期状态，不拼接各候选自己的虚拟持仓。

{choice_table}

{walk_table}

本轮83个新增场景共同做配对循环块抽样，1500次，块长20/60/120交易日。下表p值检验家族内最大日均收益增益，不是夏普显著性，也未覆盖此前全部研究搜索。历史最好者的普通配对增量区间在挑选之后计算，仍有选择偏差。

{boot_table}

{ci_table}

家族p值约0.988—0.996，本轮最大均值增益没有通过常见5%显著门槛；这不等于“无效概率99%”。普通区间的下界接近0，部分原因是可重抽的独立改善事件很少，很多抽样根本未覆盖它们；也不能把这一结果当作稳健的正向收益保证。年度选参版本的2022起点年化19.29%、夏普2.196，均低于默认19.65%、2.235。

## 资产依赖诊断

对历史夏普最高的新增方案{label(by_name[winner])}与默认做相同删资产对照，评分离散度和IVOL比较也随剩余池重算；不据此再选参数。以下增量仅比较同一删减后的两个方案，不能直接当成原策略资产收益归因。

{ab_table}

共同删去黄金、银行或纳指后，RSI仍留下少量增益，说明本样本改善并非完全由单一资产决定；这些对照仍共享2024年行情，不能补足独立事件和年度的数量。

## 主比较之外的追加行情

{extension}

## 实施与复现

本轮登记研究因子和结果，CL120仍使用原实盘评分。SMA RSI继续可用，Wilder RSI加入因子库便于后续调用。历史高收益、低回撤和一次参数搜索的胜者都不足以保证未来；当前池仍有存续与选池后视偏差。

数值与时序核对{len(checks)}项通过，包括原CL120目标完全重现、代表方案的生产选仓及正式回测收益对账、前缀截断与未来行情扰动、仓位上限、默认和实盘文件指纹不变。新增测试另覆盖Wilder手算种子、缺价重置、两种RSI缓存隔离、标签到期以及年度选择的未来收益隔离。

```powershell
python research_rsi.py
python research_rsi.py --download-latest
python research_rsi.py --report-only
```

第一条离线读取`data/factor_extensions_20261109/`既有18只行情，新环境先运行`research_factor_extensions.py --download-only`。第二条在冻结主比较后使用自己的Tushare权限增量下载至独立`data/rsi_20261120/`并做追加观察；第三条只重画已有结果。结果、因子缓存、所有场景和检查在`output/rsi_20261120/`，不共享Token、行情或账户资料；报告与三张图保留在`docs/`。
'''
    (ROOT/'docs/rsi_research_20261120.md').write_text(text,encoding='utf-8')
    print('RSI report and three figures saved.',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download-latest',action='store_true');p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    if args.download_latest:download_latest()
    elif not args.report_only:execute()
    # 报告绘图在固定主区间完成后生成，后续追加观察不会修改提名。
    behavior_audit()
    figures();report()


if __name__=='__main__':main()
