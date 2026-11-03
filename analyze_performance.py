"""当前默认策略七项指标；只读本地行情和已对账Backtrader订单，不改实盘。

python analyze_performance.py
python analyze_performance.py --end 2026-10-20
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.config import ETF_NAMES,STRATEGY_NAME
from etf_rotation.data import load_symbol
from etf_rotation.performance import return_metrics,outcome_metrics,rebalance_periods,holding_episodes,turnover_metrics
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.strategy_state import load_rotation_state

ROOT=Path(__file__).resolve().parent


def write_json(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def fingerprints():
    names=['etf_rotation/config.py','etf_rotation/risk_strategy.py','etf_rotation/portfolio_strategy.py',
           'live_signal.py','data/rotation_state.json','output/live_signal.json']
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names if (ROOT/n).exists()}


def percent(x):return '—' if x is None else f'{x:.2%}'
def number(x):return '—' if x is None else f'{x:.3f}'
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,row))+' |' for row in rows]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end',type=date.fromisoformat,default=date(2026,10,20))
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'data/audit_history')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output/performance_metrics_current')
    parser.add_argument('--backtrader-dir',type=Path,default=ROOT/'output/backtrader_comparison_current')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.end<=date(2022,1,1):raise ValueError('该对照要求同时覆盖2018和2022两个起点')
    before=fingerprints()
    values={s:load_symbol(s,date(2010,1,1),args.end,args.cache_dir,offline=True) for s in [*ETF_NAMES,'512890']}
    prices=pd.DataFrame(values,index=values['510300'].index).rename_axis('date')
    state=load_rotation_state(ROOT/'data/rotation_state.json')
    bundle=portfolio_signals(prices[list(ETF_NAMES)],prices[['512890']],state,
                             hypothetical_satellite_history=True,hypothetical_universe_history=True)
    meta=json.loads((args.backtrader_dir/'manifest.json').read_text(encoding='utf-8'))
    if (meta['strategy']!=STRATEGY_NAME or meta['end']!=str(args.end)
            or meta['simulation_mode']!='hypothetical_full_history'):
        raise ValueError('Backtrader订单与本次策略或日期不符，请先运行run_backtrader.py相同截止日')
    saved_signals=pd.read_csv(args.backtrader_dir/'signal_weights.csv',index_col=0,parse_dates=True)
    pd.testing.assert_frame_equal(bundle.signals,saved_signals,atol=1e-10,rtol=1e-10,check_freq=False)
    summaries={};checks=[];rows=[];all_episodes=[];annual_rows=[]
    for start in ('2018-01-01','2022-01-01'):
        result=run_backtest(bundle.prices,bundle.signals,date.fromisoformat(start),args.end)
        e=result.equity
        account=return_metrics(e.daily_return);account.update(turnover_metrics(e))
        b=e.benchmark_nav.pct_change().fillna(0.)
        benchmark=return_metrics(b)
        periods=rebalance_periods(e.daily_return,result.trades.loc[result.trades.status.eq('executed'),'execution_date'])
        weekly=outcome_metrics(periods.loc[periods.complete,'net_return'])
        btmeta=json.loads((args.backtrader_dir/f'backtrader_metadata_{start}_cost5.json').read_text(encoding='utf-8'))
        orders=pd.read_csv(args.backtrader_dir/f'backtrader_orders_{start}_cost5.csv',dtype={'symbol':str})
        native=pd.read_csv(args.backtrader_dir/f'backtrader_equity_{start}_cost5.csv',index_col=0,parse_dates=True)
        difference=float((native.strategy_nav-e.strategy_nav).abs().max())
        checks.append({'check':start+'_native_nav_match','error':difference,'passed':difference<1e-4})
        initial=float(btmeta['initial_cash'])
        episodes=holding_episodes(orders,prices.loc[e.index[-1]],initial_cash=initial)
        closed=episodes.loc[episodes.closed];risky=closed.loc[closed.symbol.ne('511260')]
        trade=outcome_metrics(closed.net_pnl);risk_trade=outcome_metrics(risky.net_pnl)
        trade_roi=outcome_metrics(closed.roi_on_cumulative_buys)
        pnl_error=abs(float(episodes.net_pnl.sum())-float(native.strategy_nav.iloc[-1]-1))
        checks.append({'check':start+'_episode_cash_and_mark_pnl_reconcile','error':pnl_error,'passed':pnl_error<1e-8})
        fee_error=abs(float(episodes.fees.sum())-float(native.commission.sum())/initial)
        checks.append({'check':start+'_episode_fee_reconcile','error':fee_error,'passed':fee_error<1e-10})
        before_incomplete=periods.loc[~periods.complete,'net_return']
        period_product=float(np.prod(1+periods.net_return)-1)
        # 首个成交之前保持现金，完整及未完整区块必须恰好覆盖净值变化。
        checks.append({'check':start+'_periods_cover_net_nav','error':abs(period_product-account['total_return']),
                       'passed':abs(period_product-account['total_return'])<1e-10})
        stress=run_backtest(bundle.prices,bundle.signals,date.fromisoformat(start),args.end,fee_bps=0,slippage_bps=20)
        gross=run_backtest(bundle.prices,bundle.signals,date.fromisoformat(start),args.end,fee_bps=0,slippage_bps=0)
        summaries[start]={'actual_start':str(e.index[0].date()),'end':str(e.index[-1].date()),'account':account,
           'benchmark_510300':benchmark,'closed_etf_episodes':trade,'closed_risk_etf_episodes':risk_trade,
           'closed_episode_return_basis':trade_roi,'portfolio_rebalance_periods':weekly,
           'open_episodes':int((~episodes.closed).sum()),'incomplete_account_periods':len(before_incomplete),
           'cost20':return_metrics(stress.equity.daily_return),'cost0':return_metrics(gross.equity.daily_return),
           'native_maximum_nav_difference':difference,'episode_pnl_reconcile_error':pnl_error,
           'rebalance_count':len(result.trades)}
        episodes.to_csv(args.output_dir/f'holding_episodes_{start}.csv',index=False,encoding='utf-8-sig')
        periods.to_csv(args.output_dir/f'rebalance_periods_{start}.csv',index=False,encoding='utf-8-sig')
        e.to_csv(args.output_dir/f'equity_{start}.csv',encoding='utf-8-sig')
        result.trades.to_csv(args.output_dir/f'rebalances_{start}.csv',index=False,encoding='utf-8-sig')
        for year in sorted(e.index.year.unique()):
            block=e.loc[e.index.year==year]
            annual_rows.append({'start':start,'year':int(year),'year_incomplete':year==args.end.year,
                'total_return':float((1+block.daily_return).prod()-1),
                'benchmark_total_return':float((1+b.loc[block.index]).prod()-1),
                'maximum_drawdown_within_year':return_metrics(block.daily_return)['max_drawdown'],
                'two_sided_turnover_during_year':float(block.turnover.sum())})
        for label,sample in [('all',closed),('risk_only',risky)]:
            stat=outcome_metrics(sample.net_pnl)
            rows.append({'start':start,'scope':label,**stat})
        all_episodes.append(episodes.assign(start=start))
        print(f'{start}: CAGR={account["annual_return"]:.4%}, Sharpe={account["sharpe_zero_rf"]:.4f}, Calmar={account["calmar"]:.4f}, episode win={trade["win_rate"]:.2%}, payoff={trade["payoff_ratio"]:.4f}, turnover(two sides)={account["annual_two_sided_turnover"]:.2%}',flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'closed_episode_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(annual_rows).to_csv(args.output_dir/'annual_results.csv',index=False,encoding='utf-8-sig')
    checks.append({'check':'strategy_state_live_unchanged','passed':before==fingerprints()})
    if not all(c['passed'] for c in checks):raise AssertionError('Performance calculation reconciliation failed')
    write_json(args.output_dir/'metrics.json',summaries);write_json(args.output_dir/'checks.json',checks)
    write_json(args.output_dir/'manifest.json',{'strategy':STRATEGY_NAME,'end':args.end,'hypothetical_universe':True,
        'hypothetical_satellite':True,'risk_free_rate_primary':0,'cost_bps_primary':5,'protected_before':before,
        'protected_after':fingerprints(),'all_checks_passed':True,'checks':len(checks),
        'data_hash':hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
        'source_hashes':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
                         ['analyze_performance.py','etf_rotation/performance.py']}})
    render_report(args,summaries,annual_rows)


def render_report(args,summaries,annual_rows):
    columns=['指标','2018年以来','2022年以来','截图参考线']
    a=summaries['2018-01-01'];b=summaries['2022-01-01']
    items=[('年化收益率',lambda s:percent(s['account']['annual_return']),'高于基准5个百分点'),
           ('夏普（Rf=0）',lambda s:number(s['account']['sharpe_zero_rf']),'>1可接受，>1.5优秀'),
           ('卡玛',lambda s:number(s['account']['calmar']),'>1可接受，>2优秀'),
           ('最大回撤',lambda s:percent(s['account']['max_drawdown']),'小于基准回撤幅度'),
           ('完整ETF回合胜率',lambda s:percent(s['closed_etf_episodes']['win_rate']),'>45%'),
           ('ETF回合金额盈亏比',lambda s:number(s['closed_etf_episodes']['payoff_ratio']),'>1.5可接受，>2优秀'),
           ('年化单边等效换手',lambda s:percent(s['account']['annual_half_two_sided_equivalent']),'<500%（需统一口径）')]
    rows=[[label,fn(a),fn(b),target] for label,fn,target in items]
    benchmark_rows=[];outcome_rows=[];cost_rows=[]
    for name,s in [('2018以来',a),('2022以来',b)]:
        for label,k in [('策略','account'),('沪深300ETF持有','benchmark_510300')]:
            r=s[k];benchmark_rows.append([name,label,percent(r['annual_return']),number(r['sharpe_zero_rf']),
                                          number(r['calmar']),percent(r['max_drawdown'])])
        for label,key in [('完整ETF回合，含国债','closed_etf_episodes'),('完整风险ETF回合，不含国债','closed_risk_etf_episodes'),
                           ('完整账户调仓周期','portfolio_rebalance_periods')]:
            x=s[key];outcome_rows.append([name,label,x['count'],x['wins'],x['losses'],percent(x['win_rate']),
                                         number(x['payoff_ratio']),number(x['profit_factor'])])
        for label,k in [('不计成本','cost0'),('单边5bps','account'),('单边20bps','cost20')]:
            r=s[k];cost_rows.append([name,label,percent(r['annual_return']),number(r['sharpe_zero_rf']),percent(r['max_drawdown'])])
    annual=[r for r in annual_rows if r['start']=='2018-01-01']
    text=f'''# 当前默认ETF轮动策略：七项指标与评价

2026-10-31计算；统一行情截止{args.end}。默认为17只核心＋512890红利卫星（含513180恒科、不含513500标普），RSRS继续观察，不增加实盘规则。本报告只计算指标，不调参，不读取真实账户。

## 区间与假设

主区间{a['actual_start']}至{a['end']}，{a['account']['days']}个交易日；另从{b['actual_start']}现金开始独立回测。以当前池及红利规则全历史可用为假设，但ETF实际上市前及因子预热不足时不能入选，不是此前实际账户业绩或真实采用当前规则的记录。普通回测入口仍默认按真实启用日期，本报告明确采用两个全历史假设以与此前策略研究一致。

每周最后交易日收盘信号，下一交易日收盘执行；单边费用3bps＋滑点预算2bps，共5bps，所有指标均扣除该成本，另列0/20bps压力。沿用允许分数复权单位、按成交日参考价确定目标数量的理想执行假设，不代表实盘保证成交；年化252交易日。沪深300基准使用510300后复权买入持有，不扣其买入手续费，非000300价格指数。

## 七项指标

{md(columns,rows)}

截图中的数值是评价参考线，不是收益保证或统一行业标准。主区间夏普及卡玛超过1，但未达到截图各自1.5/2的优秀线；2022后更强，不能用较短开发区间替代较长样本。胜率与盈亏比采用完整ETF持仓回合，下文另列账户周期口径。

## 与沪深300比较

{md(['区间','对象','年化收益','零利率夏普','卡玛','最大回撤'],benchmark_rows)}

年化差分别为{(a['account']['annual_return']-a['benchmark_510300']['annual_return'])*100:.2f}和{(b['account']['annual_return']-b['benchmark_510300']['annual_return'])*100:.2f}个百分点，均超过截图的5个百分点；这是年化收益差，不是回归alpha，也不证明与多资产持有组合相比存在稳定超额收益。

## 各指标的计算口径

- 年化收益：CAGR=(期末净值/期初本金)^(252/T)−1。保持项目252日惯例；按首末自然日期（含首日）换算的年化为{percent(a['account']['calendar_annual_return'])}和{percent(b['account']['calendar_annual_return'])}，不与252日口径混用。
- 夏普：sqrt(252)×mean(日收益−日无风险收益)/std(日超额收益,ddof=1)，不是直接用CAGR除波动率。主表Rf=0与历史报告一致。示例若固定年化无风险2%（按复利换算每日），夏普分别为{number(a['account']['sharpe_rf2_illustrative'])}、{number(b['account']['sharpe_rf2_illustrative'])}；2%只是敏感性假设，不代表当时或当前真实利率。国债ETF本身有价格风险，不等同无风险收益。[Sharpe原文](https://web.stanford.edu/~wfsharpe/art/sr/sr.htm)
- 卡玛：CAGR/abs(全区间最大回撤)。全区间口径与截图相同，不是仅滚动36个月。主表卡玛的分子与回撤来自同一现金起点和同一扣费曲线。
- 最大回撤：min(NAV/过去最高NAV−1)，峰值至少为初始本金1，因此首日建仓成本不被忽略。是每日收盘回撤，不能视为盘中最大回撤上限。
- 胜率：盈利已完成回合数/全部已完成回合数；零盈亏计入分母，未平仓排除。账户净值仍包含所有未实现收益和损失。
- 盈亏比：已完成回合平均净盈利金额/abs(平均净亏损金额)，不是总盈利/总亏损（后者是Profit Factor）。这与[TradingView的指标定义](https://www.tradingview.com/support/solutions/43000681719-average-profit-average-loss/)一致。仓位规模和复利会影响金额口径，另按每回合累计买入成本归一化，ROI均值盈亏比为{number(a['closed_episode_return_basis']['payoff_ratio'])}、{number(b['closed_episode_return_basis']['payoff_ratio'])}；它不是主表的金额比。
- 换手：双边sum(abs(目标权重−调仓前漂移权重))×252/T，分别为{percent(a['account']['annual_two_sided_turnover'])}、{percent(b['account']['annual_two_sided_turnover'])}。主表“单边等效”仅为双边的一半，并非基金财报按买卖较小金额/平均投资资产计算的法定口径；本策略含现金变化，不能把两者视为完全相同。

## 胜率、盈亏比与持仓回合

同一ETF从0建仓至全部清仓算一个回合，加仓和减仓不拆成多个胜负；累计全部真实模拟成交额及费用，清仓后现金流净额决定盈亏。使用已与自写引擎对账的Backtrader原生target_close订单，金额按假设初始资金归一化，不涉及个人本金。

账户收益/回撤/换手使用生产自写引擎；持仓回合使用原生经纪商订单，两个引擎最大净值差分别为{a['native_maximum_nav_difference']:.3g}、{b['native_maximum_nav_difference']:.3g}，来自微小的成本求解差异。每笔买卖费用全部计入；闭合回合PnL＋开放回合已实现/未实现PnL与原生终值逐项对账。

{md(['区间','统计对象','样本数','盈利','亏损','胜率','平均盈利/平均亏损','总盈利/总亏损'],outcome_rows)}

2018/2022现金起点期末分别有{a['open_episodes']}、{b['open_episodes']}个未平仓ETF回合，未强制卖出，不进入闭合回合胜率。按回合合并部分加减仓后，持有期间内已经实现的中途盈亏也要等整个回合结束才进入该表，避免把减仓盈利提前算作多次胜利。

“账户调仓周期”是从调仓日到下一调仓日前一天的互不重叠日收益区块，含本次调仓费用及当日旧仓收盘前收益；完整区块与期末未完成区块分开，最后未完成区块不参与胜负比例。它是账户结果，不是将同一ETF的买入/卖出看成一笔交易。日胜率另外为{percent(a['account']['daily_win_rate'])}、{percent(b['account']['daily_win_rate'])}，不拿日胜率套交易胜率阈值。

## 换手与成本评价

即使按双边的一半，两个区间仍高于500%的参考线，属于当前明显短板。周调仓不代表每周只交易一个小仓位：换ETF要卖旧买新，核心权重变化也会使国债和现金仓重新分配。高胜率和较好回撤不抵消交易摩擦。

{md(['区间','成本假设','年化收益','夏普','最大回撤'],cost_rows)}

双边换手乘以5bps得到每年按当日NAV加总的成本约{percent(a['account']['annual_additive_cost_fraction'])}、{percent(b['account']['annual_additive_cost_fraction'])}；这是名义相加成本率，不等同CAGR降幅。实际从0到5bps成本的年化降幅为{(a['cost0']['annual_return']-a['account']['annual_return'])*100:.2f}、{(b['cost0']['annual_return']-b['account']['annual_return'])*100:.2f}个百分点，复利会放大差别。实际券商最低佣金、临收盘滑点或QDII溢价可能使模型成本偏低。

## 年份表现及评价边界

{md(['年份','策略区间收益','510300持有收益','年内最大回撤','当年双边换手'],[
    [str(r['year'])+('（截至'+str(args.end)+'）' if r['year_incomplete'] else ''),percent(r['total_return']),
     percent(r['benchmark_total_return']),percent(r['maximum_drawdown_within_year']),percent(r['two_sided_turnover_during_year'])] for r in annual])}

各年回撤在年初重新从1计峰，不是跨年持续回撤；2026未满年收益没有冒充完整年化。当前可以评价为：较沪深300持有有明显历史收益/回撤优势；长区间风险调整收益合格但未达到截图的全部优秀线；近年显著更强，存在区间依赖；换手偏高，需要持续跟踪真实成交成本。胜率和盈亏比要一起看，并且不同持仓周期之间相关，不能把样本数当作独立试验次数。

这些指标描述已经观察过、并经历多轮选池调参的历史表现，不能据此判定已消除过拟合；2018区间回撤略超过10%，也不能承诺实盘5%—10%上限。保持默认，继续观察RSRS过滤而不接入实盘。

## 本地复现与输出

```powershell
python analyze_performance.py
```

默认离线使用`data/audit_history`，当前截止2026-10-20；如更换截止日，先用`run_backtrader.py --end YYYY-MM-DD`生成匹配的订单/净值，再运行本入口同一截止日。输出`output/performance_metrics_current/metrics.json`、`closed_episode_metrics.csv`、`holding_episodes_起点.csv`、`rebalance_periods_起点.csv`、逐日净值、调仓记录、年度汇总及检查清单。

本次{len(json.loads((args.output_dir/'checks.json').read_text(encoding='utf-8')))}项对账通过：新计算目标与保存Backtrader目标一致、两个引擎净值差在容差内、闭合及开放回合现金流/费用还原终值、周期收益覆盖净值、默认配置与实盘文件指纹不变。没有修改交易规则或读取个人账户。
'''
    (ROOT/'docs/performance_metrics_current.md').write_text(text,encoding='utf-8')
    print('Report: docs/performance_metrics_current.md',flush=True)


if __name__=='__main__':main()
