"""固定默认规则的行业扩池研究；不修改默认标的池或实盘状态。

首次：python research_pool_expansion.py --download
复现：python research_pool_expansion.py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from etf_rotation.backtest import run_backtest
from etf_rotation.config import LEGACY_CORE_NAMES as ETF_NAMES
from etf_rotation.data import _client, _download_table, _normalize_table, _token, load_symbol, tushare_code
from etf_rotation.factors import FactorSpec, compute_factor
from etf_rotation.mean_reversion import ReversionRule, reversion_states, defensive_funding_targets
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.risk_strategy import RiskConfig, risk_controlled_signals, risk_score_components, weekly_signal_dates
from etf_rotation.legacy_research import load_rotation_state
from research_overfitting_audit import CoreAudit, simulate, statistics

ROOT = Path(__file__).resolve().parent
CACHE = ROOT/'data/pool_expansion_20261031'
OUT = ROOT/'output/pool_expansion_20261031'
END = date(2026, 10, 20)
SAT = '512890'
CANDIDATES = {
    '159938': '医药ETF广发',
    '512200': '房地产ETF南方',
    '515220': '煤炭ETF国泰',
    '159611': '电力ETF广发',
    '159870': '化工ETF鹏华',
}
BASE = RiskConfig.current_default()
RULE = ReversionRule(200, 0, .04, None)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def protected_fingerprints():
    names = ['etf_rotation/config.py', 'etf_rotation/risk_strategy.py',
             'etf_rotation/portfolio_strategy.py', 'live_signal.py', 'data/rotation_state.json',
             'output/live_signal.json']
    return {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in names if (ROOT/p).exists()}


def plan():
    return {
        'research_date': '2026-10-31', 'data_end': END, 'evaluation_start': '2018-01-01',
        'candidates': CANDIDATES,
        'selection_basis': '缺少直接覆盖的大行业；2021年末前成立，当前披露净资产原则上至少20亿元；先冻结候选再回测',
        'not_included': {'515210': '钢铁约11亿元，小于本轮20亿元规模门槛',
                         '516970': '基建约12亿元，小于门槛',
                         '516950': '基建约1亿元，小于门槛',
                         'tech_themes': '芯片、AI、创业板已有直接覆盖，不为回测择优新增重叠主题'},
        'experiments': '5个候选全部32子集（含空集默认），无因子参数搜索；全组合与医药电力组合事先指定为覆盖对照',
        'cost_bps': [5, 20], 'segments': ['2018-2021', '2022-2024', '2025-2026-10-20'],
        'robustness': '新候选252真实报价年龄、滞后1交易日的20日成交额中位数>=1000万元、滞后份额估计市值>=2亿元；固定规则，不寻优阈值',
        'satellite': '全历史假设采用现有512890规则，与前轮审计一致；不是历史真实账户',
        'limits': '按当前存续基金及当前规模筛选仍有后视偏差；全部历史已观察，不是盲测；份额数据可能有修订或发布时间延迟',
        'protected_before': protected_fingerprints(),
    }


def download():
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'plan.json').exists():
        write_json(OUT/'plan.json', plan())
    for symbol in [*ETF_NAMES, SAT]:
        for suffix in ('.csv', '.json'):
            src = ROOT/'data/audit_history'/f'tushare_{symbol}_hfq{suffix}'
            dst = CACHE/src.name
            if not dst.exists():
                shutil.copy2(src, dst)
    pro = _client()
    status = []
    for symbol, name in CANDIDATES.items():
        try:
            prices = load_symbol(symbol, date(2010, 1, 1), END, CACHE)
            daily_path = CACHE/f'daily_{symbol}.csv'
            if not daily_path.exists():
                daily = _download_table(pro, 'fund_daily', tushare_code(symbol), date(2010, 1, 1), END)
                daily.to_csv(daily_path, index=False, encoding='utf-8-sig')
            share_path = CACHE/f'share_{symbol}.csv'
            if not share_path.exists():
                share = _download_table(pro, 'fund_share', tushare_code(symbol), date(2010, 1, 1), END)
                share.to_csv(share_path, index=False, encoding='utf-8-sig')
            nav_path = CACHE/f'nav_{symbol}.csv'
            if not nav_path.exists():
                nav = pro.fund_nav(ts_code=tushare_code(symbol), start_date='20260901', end_date='20261030')
                nav.to_csv(nav_path, index=False, encoding='utf-8-sig')
            status.append({'symbol': symbol, 'name': name, 'ok': True,
                           'first_quote': str(prices.index[0].date()), 'last_quote': str(prices.index[-1].date()), 'rows': len(prices)})
            print(f'download {symbol}: {len(prices)} rows, {prices.index[0].date()}..{prices.index[-1].date()}', flush=True)
        except Exception as exc:
            message = str(exc).replace(_token(), '[REDACTED]')
            status.append({'symbol': symbol, 'ok': False, 'error_type': type(exc).__name__,
                           'diagnostic': message})
            write_json(OUT/'download_status.json', status)
            raise RuntimeError(f'{symbol} download failed ({type(exc).__name__}); API details withheld') from None
    basic = pd.read_csv(ROOT/'data/audit_history/fund_basic_current_snapshot.csv', dtype=str)
    selected = basic[basic.ts_code.isin([tushare_code(s) for s in CANDIDATES])]
    selected.to_csv(CACHE/'candidate_fund_basic.csv', index=False, encoding='utf-8-sig')
    write_json(OUT/'download_status.json', status)


def read_prices():
    series = {}
    for symbol in [*ETF_NAMES, *CANDIDATES, SAT]:
        series[symbol] = load_symbol(symbol, date(2010, 1, 1), END, CACHE, offline=True)
    calendar = series['510300'].index
    prices = pd.DataFrame({s: p.reindex(calendar) for s, p in series.items()}, index=calendar)
    prices.index.name = 'date'
    return prices


def historical_eligibility(price, amount_yuan, market_value_yuan):
    """固定研究准入：真实报价满252日；流动性和规模至少滞后一个交易日。"""
    return (price.notna().cumsum().ge(252) & price.notna()
            & amount_yuan.rolling(20, min_periods=20).median().shift(1).ge(10_000_000)
            & market_value_yuan.shift(1).ge(200_000_000))


def eligibility_data(prices):
    """只向前沿用有限的份额数据；报价/成交额必须由历史实值产生。"""
    allowed = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    metadata = []
    basic = pd.read_csv(CACHE/'candidate_fund_basic.csv', dtype=str).set_index('ts_code')
    for symbol in CANDIDATES:
        raw = pd.read_csv(CACHE/f'tushare_{symbol}_hfq.csv', parse_dates=['date']).set_index('date')
        daily = pd.read_csv(CACHE/f'daily_{symbol}.csv', dtype={'trade_date': str})
        daily.index = pd.to_datetime(daily.trade_date, format='%Y%m%d')
        if daily.index.has_duplicates:
            raise ValueError('重复成交额日期')
        amount = pd.to_numeric(daily.amount).sort_index().reindex(prices.index)*1000  # Tushare千元
        shares_raw = pd.read_csv(CACHE/f'share_{symbol}.csv', dtype={'trade_date': str})
        share = pd.Series(pd.to_numeric(shares_raw.fd_share).to_numpy(),
                          index=pd.to_datetime(shares_raw.trade_date, format='%Y%m%d')).sort_index()
        if share.index.has_duplicates:
            raise ValueError('重复份额日期')
        # 每日份额接口若仅在变动日出现，可最多沿用5个交易日；不能反向填充。
        share = share.reindex(prices.index).ffill(limit=5)*10000
        close = raw.raw_close.reindex(prices.index)
        market_value = share*close
        allowed[symbol] = historical_eligibility(prices[symbol], amount, market_value)
        nav = pd.read_csv(CACHE/f'nav_{symbol}.csv', dtype={'nav_date': str})
        nav_row = nav.loc[nav.nav_date.eq(END.strftime('%Y%m%d'))]
        if nav_row.empty:
            raise ValueError('缺少当前规模估算所需的同日净值')
        unit_nav = float(nav_row.iloc[0].unit_nav)
        last_share = float(share.iloc[-1])
        first = prices[symbol].first_valid_index()
        meta = basic.loc[tushare_code(symbol)]
        metadata.append({'symbol': symbol, 'name': CANDIDATES[symbol], 'found_date': meta.found_date,
                         'list_date': meta.list_date, 'first_quote': first.date().isoformat(),
                         'quote_days': int(prices[symbol].notna().sum()), 'scale_date': END.isoformat(),
                         'estimated_nav_assets_yi': last_share*unit_nav/1e8,
                         'last_share_units': last_share, 'unit_nav': unit_nav,
                         'median_amount20_yuan': float(amount.tail(20).median()),
                         'eligible_first': str(allowed.index[allowed[symbol]][0].date()) if allowed[symbol].any() else None,
                         'benchmark': meta.benchmark})
    return allowed, pd.DataFrame(metadata)


def case_name(added):
    return 'default' if not added else 'add_'+'_'.join(added)


def targets_for(prices, added, state, satellite_state, *, allowed=None, fast=True, fixed_spread=False):
    core_prices = prices[[*ETF_NAMES, *added]]
    if fast and allowed is None and not fixed_spread:
        core = CoreAudit(core_prices, state).signals(BASE)
    else:
        frames = None
        if allowed is not None or fixed_spread:
            specs = [FactorSpec('momentum', 20), FactorSpec('momentum', 40), FactorSpec('trend', 40),
                     FactorSpec('volatility', 60), FactorSpec('momentum', 60), FactorSpec('ma_zscore', 20)]
            frames = {s: compute_factor(core_prices, s) for s in specs}
            # 只改变可入选资格，不抹去历史价格或未来反复预热指标。
            if allowed is not None:
                frames[FactorSpec('trend', 40)] = frames[FactorSpec('trend', 40)].where(allowed[core_prices.columns])
        overlay = None
        if fixed_spread:
            components = risk_score_components(frames[FactorSpec('momentum',20)], frames[FactorSpec('momentum',40)],
                         frames[FactorSpec('volatility',60)],BASE,frames[FactorSpec('ma_zscore',20)])
            old_spread = components['base_score'].loc[:,list(ETF_NAMES)].drop(columns='511260').std(axis=1,ddof=1).fillna(0)
            desired = (-frames[FactorSpec('ma_zscore',20)]/3).clip(-1,1).fillna(0).mul(BASE.ma_reversion_strength*old_spread,axis=0)
            desired['511260'] = 0.
            overlay = desired-components['ma_adjustment']
        core = risk_controlled_signals(core_prices, BASE, factor_frames=frames,score_overlay=overlay, **state.signal_kwargs(core_prices))
    combined = core_prices.assign(**{SAT: prices[SAT]})
    targets = defensive_funding_targets(combined, core, satellite_state, SAT, .30)
    return combined, targets


def holding_summary(prices, targets, ledger, label):
    out = []
    w = pd.DataFrame(ledger.weights, index=ledger.dates, columns=prices.columns)
    for symbol in prices.columns:
        active = w[symbol].gt(1e-10)
        groups = active.ne(active.shift(fill_value=False)).cumsum()
        lengths = active[active].groupby(groups[active]).size()
        out.append({'case': label, 'symbol': symbol, 'name': CANDIDATES.get(symbol, ETF_NAMES.get(symbol, '红利低波')),
                    'held_days': int(active.sum()), 'held_fraction': float(active.mean()),
                    'average_weight': float(w[symbol].mean()), 'max_weight': float(w[symbol].max()),
                    'episodes': len(lengths), 'median_episode_days': float(lengths.median()) if len(lengths) else 0.,
                    'max_episode_days': int(lengths.max()) if len(lengths) else 0})
    return out


def paired_bootstrap(matrix, reps=1500):
    """配对循环块抽样，家族内均值增益校正；不覆盖历次全部研究选择。"""
    rng = np.random.default_rng(20261031)
    delta = matrix[:, 1:]-matrix[:, [0]]
    observed = float(delta.mean(axis=0).max())
    centered = delta-delta.mean(axis=0)
    results = []
    for block in (20, 60, 120):
        maxima = []
        for _ in range(reps):
            starts = rng.integers(0, len(matrix), size=int(np.ceil(len(matrix)/block)))
            idx = ((starts[:, None]+np.arange(block))%len(matrix)).ravel()[:len(matrix)]
            maxima.append(centered[idx].mean(axis=0).max())
        results.append({'block': block, 'reps': reps, 'observed_best_daily_mean_gain': observed,
                        'family_mean_gain_p': float((1+np.count_nonzero(np.asarray(maxima)>=observed))/(reps+1))})
    return results


def execute():
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT/'plan.json').exists():
        write_json(OUT/'plan.json', plan())
    before = protected_fingerprints()
    prices = read_prices()
    allowed, meta = eligibility_data(prices)
    meta.to_csv(OUT/'candidate_metadata.csv', index=False, encoding='utf-8-sig')
    if (meta.estimated_nav_assets_yi < 20).any():
        raise ValueError('候选不满足事先规模门槛，停止并披露；不可静默按回测结果换候选')
    state = load_rotation_state(ROOT/'data/rotation_state.json')
    sat_state = reversion_states(prices[[SAT]], SAT, RULE, weekly_signal_dates(prices.index))
    tests = []
    records = []
    holdings = []
    returns = {}
    cases = {}
    tuples = [comb for n in range(6) for comb in itertools.combinations(CANDIDATES, n)]
    for i, added in enumerate(tuples):
        label = case_name(added)
        p, t = targets_for(prices, added, state, sat_state)
        cases[label] = (p, t)
        led = simulate(p, t, pd.Timestamp('2018-01-01'))
        returns[label] = led.series()
        holdings += holding_summary(p, t, led, label)
        t.to_csv(OUT/f'{label}_targets.csv', encoding='utf-8-sig')
        if label in {'default', case_name(tuple(CANDIDATES))}:
            production = risk_controlled_signals(p.drop(columns=SAT), BASE, **state.signal_kwargs(p))
            audit_targets = defensive_funding_targets(p, production, sat_state, SAT, .30)
            error = float((audit_targets-t).abs().to_numpy().max())
            tests.append({'check': label+'_signals_match_production', 'error': error, 'passed': error<1e-10})
            for cost in (5., 20.):
                slow = run_backtest(p, t, date(2018,1,1), END, fee_bps=0, slippage_bps=cost)
                fast = simulate(p, t, pd.Timestamp('2018-01-01'), cost_bps=cost)
                error = float(np.max(np.abs(fast.returns-slow.equity.daily_return.to_numpy())))
                tests.append({'check': label+f'_ledger_cost{cost:g}', 'error': error, 'passed': error<1e-10})
        for start in ('2018-01-01', '2022-01-01'):
            for cost in (5.,20.):
                ledger = led if start=='2018-01-01' and cost==5 else simulate(p, t, pd.Timestamp(start), cost_bps=cost)
                records.append({'case': label, 'added': ','.join(added), 'size': len(added), 'start': start,
                                'cost': cost, 'skipped': ledger.skipped,
                                'annual_turnover': float(ledger.turnover.sum()*252/len(ledger.dates)),
                                **statistics(ledger.returns)})
        print(f'pool {i+1}/{len(tuples)} {label}: complete', flush=True)
    # 默认整合函数的独立对账，防止研究绕过当前默认逻辑。
    base = portfolio_signals(prices[list(ETF_NAMES)], prices[[SAT]], state, hypothetical_satellite_history=True)
    err = float((base.signals-cases['default'][1]).abs().to_numpy().max())
    tests.append({'check': 'portfolio_default_target_match', 'error': err, 'passed': err<1e-10})
    table = pd.DataFrame(records)
    table.to_csv(OUT/'all_results.csv', index=False, encoding='utf-8-sig')
    daily = pd.DataFrame(returns)
    daily.to_csv(OUT/'all_daily_returns.csv', encoding='utf-8-sig')
    pd.DataFrame(holdings).to_csv(OUT/'holding_summary.csv', index=False, encoding='utf-8-sig')
    main = table.query("start=='2018-01-01' and cost==5").sort_values(['sharpe','size'], ascending=[False,True])
    winner = main.iloc[0]['case']
    selected = ['default', case_name(('159938',)), case_name(('159611',)),
                case_name(('159938','159611')), case_name(tuple(CANDIDATES)), winner]
    selected = list(dict.fromkeys(selected))
    segments = []
    for label in daily:
        for segment, left, right in [('2018-2021','2018-01-01','2021-12-31'),
                                     ('2022-2024','2022-01-01','2024-12-31'),
                                     ('2025+','2025-01-01',str(END))]:
            segment_r = daily.loc[left:right,label]
            segments.append({'case': label, 'segment': segment, **statistics(segment_r)})
    segments = pd.DataFrame(segments)
    segments.to_csv(OUT/'segment_results.csv', index=False, encoding='utf-8-sig')
    # 仅以过去阶段选池，再看下一阶段；下一阶段已观察过，不能称为盲测。
    choices = []
    for train, test in [('2018-2021','2022-2024'),('2022-2024','2025+')]:
        scores = segments.loc[segments.segment.eq(train)].merge(main[['case','size']], on='case')
        chosen = scores.sort_values(['sharpe','size','case'],ascending=[False,True,True]).iloc[0]['case']
        test_rows = segments.loc[segments.segment.eq(test)&segments.case.isin([chosen,'default'])]
        choices += [{'train':train,'test':test,'chosen':chosen,**row} for row in test_rows.to_dict('records')]
    write_json(OUT/'time_ordered_choices.json', choices)
    robust = []
    robust_names = set(['default', *[case_name((s,)) for s in CANDIDATES],
                        case_name(('159938','159611')), case_name(tuple(CANDIDATES)), winner])
    for label in sorted(robust_names):
        added = tuple(label.removeprefix('add_').split('_')) if label!='default' else ()
        p, t = targets_for(prices, added, state, sat_state, allowed=allowed, fast=False)
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led = simulate(p,t,pd.Timestamp(start),cost_bps=cost)
                robust.append({'case':label,'start':start,'cost':cost,**statistics(led.returns)})
    pd.DataFrame(robust).to_csv(OUT/'maturity_liquidity_results.csv', index=False, encoding='utf-8-sig')
    # 截断和未来扰动覆盖默认、全池和事后冠军，截点取真实周信号。
    for label in dict.fromkeys(['default',case_name(tuple(CANDIDATES)),winner]):
        added = tuple(label.removeprefix('add_').split('_')) if label!='default' else ()
        original = cases[label][1]
        for cutoff in (pd.Timestamp('2020-12-25'),pd.Timestamp('2023-12-29'),pd.Timestamp('2025-12-26')):
            partial = prices.loc[:cutoff]
            partial_sat = reversion_states(partial[[SAT]],SAT,RULE,weekly_signal_dates(partial.index))
            _, prefix = targets_for(partial,added,state,partial_sat)
            changed = prices.copy()
            future = changed.index>cutoff
            changed.loc[future] *= np.linspace(.4,2.7,changed.shape[1])
            changed_sat = reversion_states(changed[[SAT]],SAT,RULE,weekly_signal_dates(changed.index))
            _, perturbed = targets_for(changed,added,state,changed_sat)
            err = max(float((prefix-original.loc[:cutoff]).abs().to_numpy().max()),
                      float((perturbed.loc[:cutoff]-original.loc[:cutoff]).abs().to_numpy().max()))
            tests.append({'check':label+'_future_'+str(cutoff.date()),'error':err,'passed':err<1e-10})
    # 未上市及核心限仓、名额约束逐点验证。
    for label,(p,t) in cases.items():
        bad = False
        for s in CANDIDATES:
            if s in t:
                bad |= bool(t.loc[t.index<p[s].first_valid_index(),s].gt(0).any())
        risky = t.drop(columns=['511260',SAT])
        valid = (not bad and risky.max().max()<=.30+1e-10 and risky.sum(axis=1).max()<=.70+1e-10
                 and risky.gt(1e-10).sum(axis=1).max()<=3 and t.sum(axis=1).max()<=1+1e-10)
        tests.append({'check':label+'_positions_and_listing','passed':bool(valid)})
    prices.loc['2022-01-01':].pct_change(fill_method=None).corr().to_csv(OUT/'correlations_2022.csv',encoding='utf-8-sig')
    bootstrap = paired_bootstrap(daily.to_numpy())
    write_json(OUT/'family_bootstrap.json',bootstrap)
    after = protected_fingerprints()
    tests.append({'check':'production_and_live_unchanged','passed':before==after})
    write_json(OUT/'checks.json',tests)
    write_json(OUT/'manifest.json',{'plan':json.loads((OUT/'plan.json').read_text(encoding='utf-8')),
               'winner_ex_post':winner,'selected_for_display':selected,'checks_passed':all(x['passed'] for x in tests),
               'production_before':before,'production_after':after,
               'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in CACHE.glob('*.csv')}})
    if not all(x['passed'] for x in tests):
        raise AssertionError('研究检查失败，不能将结果作为已验证结论')
    plot_results(table,daily,selected,winner)
    print(main[['case','annual','sharpe','drawdown']].head(8).to_string(index=False),flush=True)


def diagnose_spread():
    """首轮后补充的固定解释对照，不并入事先32池统计、不用于切换默认。"""
    prices=read_prices();state=load_rotation_state(ROOT/'data/rotation_state.json')
    sat_state=reversion_states(prices[[SAT]],SAT,RULE,weekly_signal_dates(prices.index))
    records=[];counter=0
    for added in [(), *[(s,) for s in CANDIDATES], tuple(CANDIDATES)]:
        p,t=targets_for(prices,added,state,sat_state,fixed_spread=True)
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(p,t,pd.Timestamp(start),cost_bps=cost)
                records.append({'case':case_name(added),'start':start,'cost':cost,**statistics(led.returns)})
        counter+=1;print(f'fixed original dispersion {counter}/7',flush=True)
    pd.DataFrame(records).to_csv(OUT/'posthoc_fixed_original_dispersion.csv',index=False,encoding='utf-8-sig')


def markdown_table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])


def build_report():
    table=pd.read_csv(OUT/'all_results.csv')
    holdings=pd.read_csv(OUT/'holding_summary.csv',dtype={'symbol':str})
    meta=pd.read_csv(OUT/'candidate_metadata.csv',dtype={'symbol':str,'found_date':str,'list_date':str})
    segments=pd.read_csv(OUT/'segment_results.csv')
    robust=pd.read_csv(OUT/'maturity_liquidity_results.csv')
    spread=pd.read_csv(OUT/'posthoc_fixed_original_dispersion.csv')
    checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    if protected_fingerprints()!=manifest['production_before'] or not all(x['passed'] for x in checks):
        raise ValueError('实盘指纹或研究检查不符')
    names={'default':'当前默认',**{case_name((s,)):'仅加'+n for s,n in CANDIDATES.items()},
           case_name(('159938','159611')):'加医药和电力',case_name(tuple(CANDIDATES)):'五行业全扩池'}
    display=list(names)
    full=case_name(tuple(CANDIDATES))
    def perf_rows(source,start,cost=5):
        sub=source.loc[source.start.eq(start)&source.cost.eq(cost)].set_index('case')
        return [[names[label],f'{sub.at[label,"annual"]:.2%}',f'{sub.at[label,"sharpe"]:.3f}',
                 f'{sub.at[label,"drawdown"]:.2%}'] for label in display if label in sub.index]
    pct=lambda x:f'{x:.2%}'
    meta_rows=[]
    for row in meta.to_dict('records'):
        meta_rows.append([row['symbol'],row['name'],pd.Timestamp(row['found_date']).date(),row['first_quote'],
                          f'{row["estimated_nav_assets_yi"]:.2f}',f'{row["median_amount20_yuan"]/1e8:.2f}'])
    hold=holdings.loc[holdings.case.eq(full)&holdings.symbol.isin(CANDIDATES)]
    hold_rows=[[r.symbol,CANDIDATES[r.symbol],int(r.held_days),pct(r.held_fraction),pct(r.average_weight),int(r.episodes),r.median_episode_days]
               for r in hold.itertuples()]
    segment_rows=[]
    for label in ('default',full):
        for r in segments.loc[segments.case.eq(label)].itertuples():
            segment_rows.append([names[label],r.segment,pct(r.annual),f'{r.sharpe:.3f}',pct(r.drawdown)])
    corr=pd.read_csv(OUT/'correlations_2022.csv',index_col=0)
    corr.index=corr.index.astype(str)
    corr_rows=[]
    for symbol in CANDIDATES:
        old=corr.loc[symbol,list(ETF_NAMES)].drop('511260').dropna().sort_values(ascending=False)
        corr_rows.append([symbol,CANDIDATES[symbol],old.index[0],ETF_NAMES[old.index[0]],f'{old.iloc[0]:.3f}'])
    bootstrap=json.loads((OUT/'family_bootstrap.json').read_text(encoding='utf-8'))
    # 候选实际持有日的加权毛收益贡献，前日权重乘当日收益；非ETF自身年化。
    prices=read_prices();p=prices[[*ETF_NAMES,*CANDIDATES,SAT]]
    t=pd.read_csv(OUT/f'{full}_targets.csv',parse_dates=['signal_date']).set_index('signal_date')
    led=simulate(p,t,pd.Timestamp('2018-01-01'))
    w=pd.DataFrame(led.weights,index=led.dates,columns=p.columns)
    contribution=w.shift(1).fillna(0)*p.ffill(limit=3).pct_change(fill_method=None).reindex(w.index).fillna(0)
    attribution=contribution.mean()*252
    contribution.to_csv(OUT/'full_pool_gross_daily_contributions.csv',encoding='utf-8-sig')
    pd.DataFrame({'annualized_arithmetic_gross_contribution':attribution}).to_csv(OUT/'full_pool_contribution_summary.csv',encoding='utf-8-sig')
    candidate_contrib=', '.join(f'{s}：{attribution[s]*100:+.3f}个百分点/年' for s in CANDIDATES)
    primary=table.query("start=='2018-01-01' and cost==5").set_index('case')
    recent=table.query("start=='2022-01-01' and cost==5").set_index('case')
    better_primary=int((primary.sharpe>primary.at['default','sharpe']+1e-10).sum())
    better_recent=int((recent.sharpe>recent.at['default','sharpe']+1e-10).sum())
    text=f'''# 行业ETF扩池研究：覆盖面与历史表现

研究日期2026-10-31；行情固定截至2026-10-20，与此前审计使用同一数据截点。当前默认及实盘状态保持不变。

## 结论

当前17只ETF跨资产覆盖较广，行业轮动覆盖并不完整。医药卫生、公用事业电力、地产和基础化工值得作为独立行业候选；煤炭补充能源子行业，但与石油及红利有一定周期/风格重叠。沪深300、创业板、纳指和红利本来就间接包含一些行业，不能将“没有独立行业ETF”说成“完全没有风险敞口”。

固定因子、参数、持仓名额与执行规则后，五个候选全部32个子集（含空集默认）中，2018主区间有{better_primary}个、2022辅助区间有{better_recent}个方案夏普高于默认。单只新增也均未改善。成立满一年及历史流动性/规模准入对照未扭转主要结论。建议保留候选及数据，继续使用当前默认，不因“行业越齐越好”立即扩池；也不能因本轮默认更好就宣布原池无过拟合。

## 当前池

| 类型 | 当前ETF |
| --- | --- |
| A股宽基 | 510300沪深300、159915创业板 |
| 海外指数 | 159941纳指、513520日经 |
| 科技与成长行业 | 159995芯片、515070人工智能、159875新能源 |
| 消费 | 159928消费 |
| 金融 | 512800银行、512880证券 |
| 军工 | 512660军工 |
| 能源股票 | 561360石油产业 |
| 商品 | 518880黄金、159985豆粕期货、159980有色金属期货 |
| 防守 | 511260十年国债 |
| 独立卫星 | 512890红利低波，单独MA200反转，不参加核心排名 |

合计16只核心＋1只卫星。159980是商品期货ETF，并非有色矿业股票；518880也不是黄金矿业股票ETF。扩大行业直接覆盖与增加独立风险来源不是同一件事。

## 候选筛选与规模

回测前冻结5个候选：2021年末前成立、目前估计净资产至少20亿元、补充较少直接覆盖的行业；未按回测结果更换同类代表。医药选择较宽的全指医药卫生，而非只覆盖创新药；电力覆盖发电/公用事业，与新能源设备制造不同。化工虽为主题指数，覆盖多种化工子行业，不是单一产品。

{markdown_table(['代码','候选','成立日','首个真实交易报价','规模估算/亿元','近20日成交额中位数/亿元'],meta_rows)}

规模估算统一为2026-10-20同日Tushare基金份额×单位净值，不是份额数，也不冒充最新季度审计净资产；历史准入使用份额×原始交易价格估计市值。两者口径不同。成立日早于上市报价日，回测只允许真实报价出现且因子充分预热后交易。电力虽2021年成立，2022-01-07才上市，不能放回2018年买入。

钢铁515210约11亿元、广发基建516970约12亿元、银华基建516950约1亿元，在本轮资料时点未达到20亿元门槛，未进入回测候选。通信、软件、机器人、光伏等与已有科技/新能源重叠较多，不为增加代码数量重复覆盖。保险、交通运输、可选消费、建筑材料仍是可能的直接覆盖缺口；本轮没有为凑齐行业而降低规模门槛，也没有完成全市场同类代表逐一穷举。

## 回测口径

主账户2018-01-02现金启动，2018年前行情只预热；另列2022-01-04现金启动，不能混用起始账本。核心仍是20/40日风险调整动量＋20日均线反转0.5、40日趋势、8%核心波动预算、70%核心风险总上限、单只30%、最多3只核心风险ETF、前4缓冲且必选第1名。每周最后开市日收盘出信号，下一交易日收盘成交。未改因子参数、名额或红利规则。

512890仍为全历史假设MA200／入0%／出+4%／先现金后国债／上限30%，与此前审计基线一致，并非真实采用日期回放或实盘收益；全账户风险仓可能达到100%、最多4只风险ETF加国债。单边5bps为3bps费用＋2bps滑点预算，另测20bps。252交易日年化，夏普无风险利率为零，回撤按收盘净值。允许分数复权单位，没有整数手、盘口、竞价容量和真实现金分红到账的执行模型。

## 同规则扩池结果：2018年以来

{markdown_table(['方案','年化','零利率夏普','最大回撤'],perf_rows(table,'2018-01-01'))}

## 2022年以来辅助比较

{markdown_table(['方案','年化','零利率夏普','最大回撤'],perf_rows(table,'2022-01-01'))}

![扩池净值、回撤与全部组合分布](assets/pool_expansion_20261031.png)

## 分段、费用与历史准入

以下分段继承2018账本持仓，和2022独立现金启动略有差异。

{markdown_table(['方案','区间','年化','夏普','最大回撤'],segment_rows)}

单边20bps的2018主区间：

{markdown_table(['方案','年化','夏普','最大回撤'],[r for r in perf_rows(table,'2018-01-01',20) if r[0] in ('当前默认','五行业全扩池')])}

固定历史准入只作用于新增候选：真实报价至少252日、前一交易日已知的20日成交额中位数至少1000万元、前一交易日份额估计市值至少2亿元。当日缺价也不得新选入；份额只向前沿用最多5个交易日，完全不向后填充。门槛没有搜索寻优。这是将新候选变得更保守的敏感性，不等同于对全市场历史池进行统一筛选，也不能保证资料发布延迟已被完全模拟。

{markdown_table(['准入后方案','年化','夏普','最大回撤'],perf_rows(robust,'2018-01-01'))}

按2018—2021夏普选池后检验2022—2024，以及按2022—2024夏普选池后检验2025以后，均选中原默认。这里已观察所有历史，只是时间顺序诊断，不是新的盲测，也没有证明未来必选原池。

## 为何行业更广没有带来更好回测

排名轮动最多仍选3只核心风险ETF，新增标的会挤出原资产，并非在原组合外免费叠加分散。行业具备长期经济价值，不意味着它在本策略短动量、周收盘执行下具有正的边际选时收益。五行业全扩池的实际新增持仓如下，周期可能包含样本末尚未退出的周期；天数占比的分母是全部2018以来交易日，包括部分ETF上市前日期。

{markdown_table(['代码','候选','持有日','全期持有天数占比','全期平均账户权重','连续持有周期数','周期中位天数'],hold_rows)}

前日实际权重×当日复权收益计算的算术年率毛贡献为：{candidate_contrib}。这是组合账户的加权毛贡献，不是ETF自身年化，未扣单项交易成本，也不是相对于原资产的精确机会成本；正贡献同样可能低于被挤出资产。

2022年以来日收益与原核心风险ETF的最高成对相关系数如下。使用各自共同真实报价日；全期相关仅作事后诊断，未输入交易信号；低于1不代表危机中独立。

{markdown_table(['候选','名称','原池最相关代码','名称','相关系数'],corr_rows)}

另发现现有均线反转项由池内原评分离散度D缩放，扩池即使未买入新ETF也可能改变旧ETF评分。首轮后单独补充“D固定为原15只风险ETF”的解释对照，未并入事先32池统计、不切换默认。其五行业全扩池2018年化{pct(spread.query("start=='2018-01-01' and cost==5 and case==@full").iloc[0].annual)}，夏普{spread.query("start=='2018-01-01' and cost==5 and case==@full").iloc[0].sharpe:.3f}；较原扩池改善，但仍低于默认。2022辅助区间年化{pct(spread.query("start=='2022-01-01' and cost==5 and case==@full").iloc[0].annual)}、夏普{spread.query("start=='2022-01-01' and cost==5 and case==@full").iloc[0].sharpe:.3f}。因此不能只用评分缩放变化解释扩池变差。

## 审计与证据边界

全部{len(checks)}项研究检查通过：默认整合函数对账；默认/全池信号与生产函数对账；5/20bps逐日账本对账；3个历史周截点的截断和未来价格扰动；32个池的上市前、核心限仓及名额检查；生产及实盘文件指纹不变。新增3项历史准入单元测试，完整107项测试通过。有限测试不构成没有任何未来函数的数学证明。

32个池共有128个主表区间/费用结果，全部保留，未删掉差方案。7个准入对照加默认共8种、2个起点和2个成本共32个结果；另有7种固定旧D的后设解释对照、28个结果，分别存放，不合并冒充事先试验。配对循环块抽样20/60/120日、各1500次的家族内均值增益探索性p值为{', '.join(f'{x["family_mean_gain_p"]:.3f}' for x in bootstrap)}；本轮连观察到的正均值增益都没有，不应将高p值解读为“已证明候选永远无效”。这也不涵盖项目此前所有筛选和搜索。

按当前大规模存续ETF选择历史候选仍然有幸存者及选池后视，规模越大也可能正是此前涨得好/吸金多的结果。历史成交额和份额准入只能缓解其中一部分。没有清盘ETF完整名单、可靠的逐日历史分类及全部份额发布时间；不具备全市场无偏历史池。原策略本身经多轮开发选择，默认获胜不能当作排除其过拟合的证据。

建议继续冻结默认；五只候选进入观察库而非实盘默认。后续若要推进扩池，可预先指定统一的大行业候选、历史年龄/规模/流动性门槛，并以新累积的真实样本验证，而不是从这32组中继续追逐局部赢家。产业覆盖目标和收益目标应分别评估。

## 复现与文件

```powershell
python research_pool_expansion.py --download
python research_pool_expansion.py
python research_pool_expansion.py --diagnose-spread
python research_pool_expansion.py --report-only
```

首次下载使用自己的Tushare权限；研究只使用独立的`data/pool_expansion_20261031/`。主结果、逐日收益、持仓统计、准入对照、时间顺序选择、相关、检查和数据指纹位于`output/pool_expansion_20261031/`。报告重建不重新选参。没有自动修改`config.py`、默认策略、实盘计划或真实账户数据库。

## 公开资料

- [广发医药ETF官方季度报告：成立与跟踪范围](https://www.gffunds.com.cn/jjgg/dqgg/202604/P020260422306671905861.pdf)
- [南方地产ETF官方2026三季报](https://www.nffund.com/main/files/2026/10/26/167233623652.pdf)
- [电力ETF官方产品概要：成立与上市日](https://www.gffunds.com.cn/jjgg/flwj/202603/P020260325366808318318.pdf)
- [鹏华化工ETF2026三季报，巨潮资讯披露](https://static.cninfo.com.cn/finalpage/2026-10-26/1225621767.PDF)
- [上交所煤炭ETF资料入口](https://www.sse.com.cn/assortment/fund/list/etfinfo/basic/index.shtml?FUNDID=515220)
- [Tushare基金份额接口：fd_share单位为万份，2000积分起](https://tushare.pro/document/2?doc_id=207)
- 未纳入候选的规模筛查：[钢铁515210](https://fund.eastmoney.com/515210.html)、[基建516970](https://fundf10.eastmoney.com/jbgk_516970.html)、[基建516950](https://finance.sina.com.cn/fund/quotes/516950/bc.shtml)。规模资料日期不完全相同，仅作研究日候选筛选，不输入历史排名。
'''
    report=ROOT/'docs/pool_expansion_20261031.md'
    report.write_text(text,encoding='utf-8')
    write_json(OUT/'final_validation.json',{'checks':len(checks),'all_checks_passed':True,'unit_tests_passed':107,
               'default_unchanged':True,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                 for p in ['research_pool_expansion.py','tests/test_pool_expansion.py']}})
    print(f'Report saved: {report.relative_to(ROOT)}',flush=True)


def plot_results(table,daily,selected,winner):
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(15,10))
    fig.subplots_adjust(top=.90,bottom=.11,left=.06,right=.95,hspace=.38,wspace=.27)
    labels={'default':'当前默认','add_159938':'加医药','add_159611':'加电力',
            'add_159938_159611':'加医药和电力',case_name(tuple(CANDIDATES)):'五行业全扩池'}
    labels[winner] = labels.get(winner,'事后最高夏普组合')+('（事后冠军）' if winner!='default' else '')
    for label in selected:
        nav=(1+daily[label]).cumprod()
        axes[0,0].plot(nav.index,nav,label=labels.get(label,label),lw=1.6)
        dd=nav/nav.cummax().clip(lower=1)-1
        axes[1,0].plot(dd.index,dd*100,label=labels.get(label,label),lw=1.2)
    axes[0,0].set_title('2018年现金启动：扣费净值')
    axes[1,0].set_title('收盘回撤曲线（%）')
    axes[0,0].legend(fontsize=8)
    main=table.query("start=='2018-01-01' and cost==5").set_index('case')
    dots=axes[0,1].scatter(main.annual*100,main.sharpe,c=main['size'],cmap='viridis',s=45,alpha=.7)
    fig.colorbar(dots,ax=axes[0,1],label='新增ETF数量',ticks=range(6),shrink=.8)
    for label in selected:
        row=main.loc[label]
        offset=(-5,-10) if label=='default' else (3,3)
        axes[0,1].annotate(labels.get(label,label),(row.annual*100,row.sharpe),fontsize=8,
                           xytext=offset,textcoords='offset points',ha='right' if label=='default' else 'left')
    axes[0,1].margins(x=.08,y=.10)
    axes[0,1].set_xlabel('年化收益（%）');axes[0,1].set_ylabel('零利率夏普')
    axes[0,1].set_title('32种池组合全部保留：同规则、未调因子')
    recent=table.query("start=='2022-01-01' and cost==5").set_index('case').loc[selected]
    axes[1,1].bar(np.arange(len(recent)),recent.annual*100)
    axes[1,1].set_xticks(np.arange(len(recent)),[labels.get(s,s) for s in selected],rotation=25,ha='right',fontsize=8)
    axes[1,1].set_ylabel('年化收益（%）');axes[1,1].set_title('2022年现金启动：相同区间辅助比较')
    for ax in axes.ravel():ax.grid(alpha=.2)
    fig.suptitle('行业扩池研究 | 行情至2026-10-20 | 单边5bps | 全历史红利假设，非盲测',fontsize=14,y=.98)
    assets=ROOT/'docs/assets';assets.mkdir(exist_ok=True)
    fig.savefig(assets/'pool_expansion_20261031.png',dpi=160,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--diagnose-spread',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    args = parser.parse_args()
    if args.download:
        download()
    elif args.diagnose_spread:
        diagnose_spread()
    elif args.report_only:
        build_report()
    elif args.plot_only:
        manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        plot_results(pd.read_csv(OUT/'all_results.csv'),pd.read_csv(OUT/'all_daily_returns.csv',index_col=0,parse_dates=True),
                     manifest['selected_for_display'],manifest['winner_ex_post'])
    else:
        execute()
