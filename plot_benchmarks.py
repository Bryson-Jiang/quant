"""将默认策略回测与主要 A 股宽基 ETF 的买入持有收益画在同一张图。"""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.config import STRATEGY_NAME
from etf_rotation.data import load_symbol


ROOT = Path(__file__).resolve().parent
BENCHMARKS: dict[str, tuple[str, str]] = {
    "510050": ("上证50", "#9467bd"),
    "510300": ("沪深300", "#1f77b4"),
    "510500": ("中证500", "#ff7f0e"),
    "512100": ("中证1000", "#d62728"),
    "159915": ("创业板", "#2ca02c"),
    "588000": ("科创50", "#8c564b"),
}
TECH_WEIGHTS={"588000":.40,"159995":.25,"515070":.15,"510300":.20}


def buy_hold_nav(prices: pd.DataFrame, weights: dict[str,float], fee_bps: float = 5.) -> pd.Series:
    """首日收盘买入后不再平衡；各标的用固定份额、同一初始交易成本。"""
    w=pd.Series(weights)
    if not (np.isfinite(w).all() and w.ge(0).all() and abs(float(w.sum())-1)<1e-12):
        raise ValueError("买入持有初始权重必须非负且和为1")
    quoted=prices.loc[:,w.index].ffill(limit=3)
    if quoted.isna().any().any() or quoted.le(0).any().any():
        raise ValueError("比较区间缺少有效ETF复权价，不用未来报价补齐")
    return (quoted/quoted.iloc[0]).mul(w,axis=1).sum(axis=1)*(1-fee_bps/10000)


def incremental_attribution(prices, with_dividend, without_dividend, symbol="512890", bond="511260"):
    """精确分解净收益差，再按原核心收益复投，合计等于两组合的净值差。"""
    import numpy as np
    dates=with_dividend.equity.index
    ret=prices.ffill(limit=3).pct_change(fill_method=None).reindex(dates)
    new=with_dividend.weights.reindex(columns=prices.columns).shift(1).fillna(0.)
    old=without_dividend.weights.reindex(columns=prices.columns).shift(1).fillna(0.)
    core=[s for s in prices if s not in {symbol,bond}]
    daily=pd.DataFrame(index=dates)
    daily['dividend']=new[symbol]*ret[symbol].fillna(0.)
    daily['displaced_bond']=(new[bond]-old[bond])*ret[bond].fillna(0.)
    daily['core_weight_drift']=((new[core]-old[core])*ret[core].fillna(0.)).sum(axis=1)
    a,b=with_dividend.equity,without_dividend.equity
    daily['extra_cost']=-(a.cost_fraction*(1+a.gross_return)-b.cost_fraction*(1+b.gross_return))
    np.testing.assert_allclose(daily.sum(axis=1),a.daily_return-b.daily_return,atol=1e-12)
    accumulated=pd.DataFrame(0.,index=dates,columns=daily.columns)
    carried=np.zeros(len(daily.columns));previous_nav=a.strategy_nav.shift(1).fillna(1.)
    for day,row in daily.iterrows():
        carried=carried*(1+b.at[day,'daily_return'])+previous_nav.loc[day]*row.to_numpy()
        accumulated.loc[day]=carried
    accumulated['net_nav_difference']=a.strategy_nav-b.strategy_nav
    np.testing.assert_allclose(accumulated.drop(columns='net_nav_difference').sum(axis=1),accumulated.net_nav_difference,atol=1e-11)
    return daily,accumulated


def dividend_comparison(end, cache_dir, output_dir, offline):
    """新默认全历史假设、科技重仓示例与红利边际归因；不修改实盘。"""
    import numpy as np
    from etf_rotation.backtest import run_backtest
    from etf_rotation.config import ETF_NAMES
    from etf_rotation.portfolio_strategy import load_portfolio_data, portfolio_signals
    from etf_rotation.strategy_state import load_rotation_state
    protected_names=('data/rotation_state.json','output/live_signal.json','etf_rotation/config.py','live_signal.py','run_backtest.py')
    def fingerprints():return {s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in protected_names if (ROOT/s).exists()}
    before=fingerprints()
    state=load_rotation_state(ROOT/'data/rotation_state.json')
    core,history=load_portfolio_data(date(2022,1,1),end,cache_dir,offline=offline)
    bundle=portfolio_signals(core,history,state,hypothetical_satellite_history=True)
    first=date(2022,1,4)
    new=run_backtest(bundle.prices,bundle.signals,first,end)
    core_target=bundle.core_signals.reindex(columns=bundle.prices.columns,fill_value=0.)
    old=run_backtest(bundle.prices,core_target,first,end)
    dates=new.equity.index
    prices=bundle.prices.reindex(dates).copy();missing={}
    for symbol in dict.fromkeys([*BENCHMARKS,*TECH_WEIGHTS]):
        if symbol not in prices:
            prices[symbol]=load_symbol(symbol,dates[0].date(),end,cache_dir,offline=offline).reindex(dates)
        missing[symbol]=prices.index[prices[symbol].isna()].strftime('%Y-%m-%d').tolist()
        print(f"comparison quote {symbol} checked",flush=True)
    prices=prices.ffill(limit=3)
    if prices[list(set(BENCHMARKS)|set(TECH_WEIGHTS))].isna().any().any():
        raise ValueError('比较行情仍存在连续缺价')
    labels={'new':'新默认（全历史假设）','core':'不加红利的原核心','tech':'科技重仓示例（80%科技）'}
    nav=pd.DataFrame({labels['new']:new.equity.strategy_nav,labels['core']:old.equity.strategy_nav})
    for symbol,(label,_) in BENCHMARKS.items():nav[label]=buy_hold_nav(prices,{symbol:1.})
    nav[labels['tech']]=buy_hold_nav(prices,TECH_WEIGHTS)
    daily_returns=nav.pct_change();daily_returns.iloc[0]=nav.iloc[0]-1
    dd=nav/nav.cummax().clip(lower=1)-1
    metrics=[]
    for label in nav:
        x=daily_returns[label];sigma=x.std(ddof=1)
        metrics.append({'name':label,'total_return':float(nav[label].iloc[-1]-1),'annual_return':float(nav[label].iloc[-1]**(252/len(nav))-1),'annual_volatility':float(sigma*np.sqrt(252)),'sharpe_zero_rf':float(x.mean()/sigma*np.sqrt(252)),'max_drawdown':float(dd[label].min())})
    metrics=pd.DataFrame(metrics).set_index('name')
    output_dir.mkdir(parents=True,exist_ok=True)
    nav.to_csv(output_dir/'comparison_nav.csv',encoding='utf-8-sig')
    dd.to_csv(output_dir/'drawdown_curves.csv',encoding='utf-8-sig')
    dd.cummin().to_csv(output_dir/'running_max_drawdown.csv',encoding='utf-8-sig')
    metrics.to_csv(output_dir/'comparison_summary.csv',encoding='utf-8-sig')
    daily,attribution=incremental_attribution(bundle.prices,new,old)
    daily.to_csv(output_dir/'incremental_daily_returns.csv',encoding='utf-8-sig')
    attribution.to_csv(output_dir/'incremental_wealth_attribution.csv',encoding='utf-8-sig')
    # 每个持仓周期都列出，不因窗口效果差而删除；市场下跌窗口仅作事后归因。
    events=[];opening=None
    s=bundle.satellite_states
    for signal,row in s.loc[s.reason.isin(['entry','ma_exit'])].iterrows():
        loc=bundle.prices.index.searchsorted(signal,side='right')
        if loc>=len(bundle.prices):continue
        execution=bundle.prices.index[loc]
        if row.reason=='entry':opening=execution
        elif opening is not None:
            if opening>=dates[0]:events.append((opening,execution,True))
            opening=None
    if opening is not None and opening>=dates[0]:events.append((opening,dates[-1],False))
    event_rows=[]
    def segment_stats(label,left,right):
        part=daily_returns.loc[left:right,label]
        path=np.r_[1.,(1+part).cumprod().to_numpy()]
        return float(path[-1]-1),float((path/np.maximum.accumulate(path)-1).min())
    for left,right,closed in events:
        i=dates.get_loc(left);anchor=dates[max(i-1,0)]
        line=nav.loc[anchor:right,labels['tech']]
        decline=line/line.cummax()-1
        trough=decline.idxmin();peak=line.loc[:trough].idxmax()
        a,ad=segment_stats(labels['new'],left,right);b,bd=segment_stats(labels['core'],left,right)
        market_return=float(nav.at[right,'沪深300']/nav.at[anchor,'沪深300']-1)
        tech_return=float(nav.at[right,labels['tech']]/nav.at[anchor,labels['tech']]-1)
        window_new=float(nav.at[trough,labels['new']]/nav.at[peak,labels['new']]-1)
        window_core=float(nav.at[trough,labels['core']]/nav.at[peak,labels['core']]-1)
        event_rows.append({'entry':str(left.date()),'exit':str(right.date()),'closed':closed,'new_return':a,'core_return':b,'return_increment':a-b,'new_local_drawdown':ad,'core_local_drawdown':bd,'drawdown_improvement':ad-bd,'csi300_return':market_return,'tech_return':tech_return,'tech_decline_peak':str(peak.date()),'tech_decline_trough':str(trough.date()),'tech_decline':float(decline.min()),'new_return_in_tech_decline':window_new,'core_return_in_tech_decline':window_core,'increment_in_tech_decline':window_new-window_core})
    event_table=pd.DataFrame(event_rows)
    event_table.to_csv(output_dir/'dividend_period_attribution.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    colors={labels['new']:'#00695c',labels['core']:'#6b7280',labels['tech']:'#c2185b',**{label:color for label,color in BENCHMARKS.values()}}
    assets=ROOT/'docs/assets';assets.mkdir(parents=True,exist_ok=True)
    chart=assets/'default_benchmark_20261021.png'
    fig,axes=plt.subplots(2,1,figsize=(15,10),sharex=True,gridspec_kw={'height_ratios':[2.1,1.5]})
    for label in nav:
        prominent=label in labels.values()
        for ax,series in ((axes[0],(nav[label]-1)*100),(axes[1],dd[label]*100)):
            ax.plot(nav.index,series,color=colors[label],label=label,linewidth=2.8 if label==labels['new'] else 2 if prominent else 1.1,alpha=1 if prominent else .6,linestyle='--' if label==labels['core'] else '-')
    axes[0].legend(ncol=3,loc='upper left',fontsize=9,framealpha=.95)
    axes[0].set_title(f"新默认策略、A股常见宽基与科技重仓示例｜2022-01-04至{end}\n全历史假设回测；红利计划首次启用信号为2026-10-23",fontsize=15,pad=12)
    axes[0].set_ylabel('累计收益（%）');axes[1].set_ylabel('相对历史峰值的回撤（%）')
    for label in (labels['new'],labels['tech']):
        axes[0].annotate(f"{nav[label].iloc[-1]-1:+.1%}",(nav.index[-1],(nav[label].iloc[-1]-1)*100),xytext=(8,0),textcoords='offset points',color=colors[label],fontsize=10)
    for ax in axes:ax.grid(alpha=.18);ax.axhline(0,color='#9ca3af',linewidth=.6)
    fig.text(.5,.018,'科技示例：初始40%科创50＋25%芯片＋15%人工智能＋20%沪深300，买入后固定份额；全部初始买入扣5bp，策略每次交易单边5bp。',ha='center',fontsize=9,color='#4b5563')
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(chart,dpi=160);plt.close(fig)
    # 红利对冲与反弹增强分开展示；每个周期的回撤改善也包括负值。
    attribution_chart=assets/'dividend_hedge_attribution_20261021.png'
    fig,axes=plt.subplots(4,1,figsize=(15,14),gridspec_kw={'height_ratios':[1.8,1.4,1.5,1.4]})
    for label in (labels['new'],labels['core']):
        axes[0].plot(nav.index,(nav[label]-1)*100,label=label,color=colors[label],linewidth=2)
        axes[1].plot(nav.index,dd[label]*100,color=colors[label],linewidth=1.6)
    axes[0].set_title('红利反转的边际作用：收益增强不等于降低回撤\n全历史假设；绿色标出科技下跌窗口中增益至少0.5个百分点的区间，均为事后归因',fontsize=15)
    for row in event_rows:
        for ax in axes[:3]:ax.axvspan(pd.Timestamp(row['entry']),pd.Timestamp(row['exit']),color='#fbbf24',alpha=.10)
        if row['tech_decline']<=-.03 and row['increment_in_tech_decline']>=.005:
            left=pd.Timestamp(row['tech_decline_peak']);right=pd.Timestamp(row['tech_decline_trough'])
            for ax in axes[:3]:ax.axvspan(left,right,color='#16a34a',alpha=.13)
            middle=left+(right-left)/2
            axes[0].annotate(f"{left:%Y.%m}—{right:%Y.%m}\n下跌窗口增益+{row['increment_in_tech_decline']*100:.2f}个百分点",(middle,float(((nav[labels['new']]-1)*100).asof(middle))),xytext=(0,28),textcoords='offset points',ha='center',fontsize=8,color='#166534',arrowprops={'arrowstyle':'-','color':'#166534'})
    axes[0].legend(loc='upper left',fontsize=9)
    axes[0].set_ylim(-5,float(((nav[labels['new']]-1)*100).max())*1.25+8)
    axes[0].set_ylabel('累计收益（%）');axes[1].set_ylabel('全程收盘回撤（%）')
    axes[2].plot(attribution.index,attribution.dividend*100,color='#7c3aed',label='红利价格收益的复投贡献',linewidth=1.4)
    axes[2].plot(attribution.index,attribution.displaced_bond*100,color='#64748b',label='减少国债的机会成本',linewidth=1.2)
    axes[2].plot(attribution.index,attribution.net_nav_difference*100,color='#111827',label='模块净增资产（含漂移与成本）',linewidth=2.3)
    axes[2].set_ylabel('相对初始本金的增量（%）');axes[2].legend(loc='upper left',fontsize=8,ncol=3)
    values=event_table.drawdown_improvement.to_numpy()*100
    event_labels=[f"{pd.Timestamp(r['entry']):%Y.%m}—{pd.Timestamp(r['exit']):%Y.%m}" for r in event_rows]
    bars=axes[3].barh(event_labels,values,color=['#16a34a' if x>0 else '#dc2626' for x in values],alpha=.85)
    axes[3].axvline(0,color='#6b7280',linewidth=.8);axes[3].set_xlabel('该持仓周期内最大回撤的改善（百分点；负数表示回撤加深）')
    for bar,x in zip(bars,values):axes[3].annotate(f'{x:+.2f}',(x,bar.get_y()+bar.get_height()/2),xytext=(5 if x>=0 else -5,0),textcoords='offset points',ha='left' if x>=0 else 'right',va='center',fontsize=9)
    for ax in axes:ax.grid(alpha=.16)
    for ax in axes[1:3]:ax.sharex(axes[0])
    fig.text(.5,.012,'淡黄色＝红利持仓期；绿色＝预定口径下的科技下跌窗口正向缓冲。所有周期均保留；事后区间不能作为可提前知道的交易条件。',ha='center',fontsize=9,color='#4b5563')
    fig.tight_layout(rect=(0,.03,1,1));fig.savefig(attribution_chart,dpi=160);plt.close(fig)
    # 另存最大回撤柱状图，便于直接看各曲线最深值。
    fig,ax=plt.subplots(figsize=(11,6))
    sorted_metrics=metrics.sort_values('max_drawdown')
    ax.barh(sorted_metrics.index,sorted_metrics.max_drawdown*100,color=[colors[n] for n in sorted_metrics.index])
    for i,(n,row) in enumerate(sorted_metrics.iterrows()):ax.text(row.max_drawdown*100-.6,i,f'{row.max_drawdown:.2%}',ha='right',va='center',fontsize=9)
    ax.set_xlim(min(-65,float(sorted_metrics.max_drawdown.min()*100)-8),2);ax.set_xlabel('全期最大回撤（%）');ax.set_title('统一时间区间的最大回撤｜全历史假设与ETF买入持有')
    ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(assets/'default_benchmark_drawdowns_20261021.png',dpi=160);plt.close(fig)
    checks={'period':{'start':str(dates[0].date()),'end':str(dates[-1].date())},'strategy':STRATEGY_NAME,'simulation_mode':'hypothetical_full_history','actual_satellite_signal_start':state.dividend_policy.effective_signal_date.isoformat(),'technology_initial_weights':TECH_WEIGHTS,'technology_is_example_not_real_user':True,'buy_hold_initial_fee_bps':5,'strategy_one_way_fee_and_slippage_bps':5,'benchmark_quote_gaps':missing,'missing_valuation':'past quote only, maximum3 trading days','full_period_drawdown_is_not_a_risk_promise':True,'daily_increment_attribution_error':float((daily.sum(axis=1)-(new.equity.daily_return-old.equity.daily_return)).abs().max()),'wealth_attribution_error':float((attribution.drop(columns='net_nav_difference').sum(axis=1)-attribution.net_nav_difference).abs().max()),'all_six_periods_included':len(event_rows)==6,'protected_default_unchanged':fingerprints()==before,'historical_highlight_is_not_an_entry_signal':True}
    assert checks['protected_default_unchanged']
    (output_dir/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 新默认策略、宽基与科技重仓对比及红利归因','',f'数据区间2022-01-04至{end}；研究生成于2026-10-21。','',
           '**所有新默认历史曲线是假设从2022年以来均已执行新规则；红利计划首个启用信号为2026-10-23。截至本图数据末日尚无该模块的实盘对冲记录。**','',
           '## 比较口径','',
           '宽基用代表ETF复权买入持有，上证50/沪深300/中证500/中证1000/创业板/科创50，不是官方指数总收益序列。科技重仓是预先固定的示例：40%588000科创50、25%159995芯片、15%515070人工智能、20%510300沪深300，首日买入后固定份额，不调参、不再平衡，不代表真实用户账户。所有买入持有组合初始成本5bp；新默认和原核心每次交易单边5bp，均未模拟整数手、最低佣金、期末卖出或盘中回撤。','',
           '## 收益与风险','', '| 方案 | 累计收益 | 年化 | 夏普 | 最大回撤 |','| --- | ---: | ---: | ---: | ---: |']
    for n,row in metrics.iterrows():lines.append(f'| {n} | {row.total_return:.2%} | {row.annual_return:.2%} | {row.sharpe_zero_rf:.3f} | {row.max_drawdown:.2%} |')
    lines+=['','![统一区间收益与回撤](assets/default_benchmark_20261021.png)','', '![最大回撤](assets/default_benchmark_drawdowns_20261021.png)','',
            '## 红利模块与原核心的逐周期对照','',
            '入场及退出区间由既有策略信号确定，下一交易日收盘执行。每个周期计算从买入执行日至卖出执行日的组合收益和局部回撤，包含相关交易费用。回撤改善为新默认回撤减原核心回撤，正值表示缓冲、负值表示加深。','',
            '| 周期 | 新默认收益 | 原核心收益 | 新增收益 | 新默认局部回撤 | 原核心局部回撤 | 回撤改善 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in event_rows:lines.append(f"| {row['entry']}至{row['exit']} | {row['new_return']:.2%} | {row['core_return']:.2%} | {row['return_increment']*100:+.2f}个百分点 | {row['new_local_drawdown']:.2%} | {row['core_local_drawdown']:.2%} | {row['drawdown_improvement']*100:+.2f}个百分点 |")
    lines+=['','## 科技下跌窗口中的缓冲','',
            '为了避免只挑有利日期，逐个持仓周期固定选取科技重仓示例在该周期内最大的峰谷下跌区间，六个窗口全部列出。这些峰谷只能事后知道，仅用于归因，不进入策略或参数选择。图中绿色仅标出科技跌幅至少3%、同时新默认相对原核心的窗口收益增益至少0.5个百分点的区间；它不等同于持仓全周期回撤降低或严格金融对冲。','',
            '| 科技峰日→谷日 | 科技下跌 | 新默认区间收益 | 原核心区间收益 | 红利模块净增益 |','| --- | ---: | ---: | ---: | ---: |']
    for row in event_rows:lines.append(f"| {row['tech_decline_peak']}至{row['tech_decline_trough']} | {row['tech_decline']:.2%} | {row['new_return_in_tech_decline']:.2%} | {row['core_return_in_tech_decline']:.2%} | {row['increment_in_tech_decline']*100:+.2f}个百分点 |")
    better=int(event_table.drawdown_improvement.gt(0).sum())
    lines+=['','![红利边际归因](assets/dividend_hedge_attribution_20261021.png)','',
            f'六个完整周期中，{better}个周期的局部最大回撤降低，{len(event_rows)-better}个加深；因此红利不能被概括为持续有效的避险资产。全期新默认回撤{metrics.at[labels["new"],"max_drawdown"]:.2%}、原核心{metrics.at[labels["core"],"max_drawdown"]:.2%}。原策略本身的国债、现金及风险控制仍是防守的重要来源。','',
            f'期末相对不加红利的净资产增加{attribution.net_nav_difference.iloc[-1]*100:.2f}个初始本金百分点。精确归因包括红利收益、减少国债的机会成本、核心权重自然漂移及额外费用；按原核心收益复投后各项合计严格等于净值差。不能直接把红利ETF涨幅乘30%当作全期组合增益。','',
            '所有历史已被观察，多轮策略选择存在过拟合风险；过去完成周期胜率不保证未来，历史回撤不是未来上限。','',
            '## 复现','', '```powershell',f'.\\.venv\\Scripts\\python.exe plot_benchmarks.py --dividend-attribution --offline --end {end}','```','',
            '完整净值、逐日回撤、截至各日最深回撤、周期归因、增量收益与资产分解、行情缺口和指纹检查保存在output/default_benchmark_20261021/。','',
            '## 产品资料','',
            '- [华夏科创50 ETF 588000](https://fund.chinaamc.com/fund/588000/)','- [华夏人工智能ETF 515070](https://www.chinaamc.com/fund_en/515070/index.shtml?source=click)']
    report=ROOT/'docs/default_benchmark_attribution_20261021.md';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(metrics.to_string(),flush=True)
    print(event_table.to_string(index=False),flush=True)
    print(f'CHARTS AND REPORT COMPLETE {report}',flush=True)


def make_comparison(equity: pd.DataFrame, benchmarks: dict[str, pd.Series]) -> pd.DataFrame:
    """起点都归一为 1；行情缺失时停止，不用未来价格或长时填补。"""
    if "strategy_nav" not in equity or equity.empty:
        raise ValueError("回测净值缺少 strategy_nav")
    if not isinstance(equity.index, pd.DatetimeIndex) or equity.index.has_duplicates:
        raise ValueError("回测日期索引无效")
    out = pd.DataFrame(index=equity.index)
    strategy = pd.to_numeric(equity["strategy_nav"], errors="coerce")
    if strategy.isna().any() or (strategy <= 0).any():
        raise ValueError("策略净值包含无效数值")
    # 回测首日已经扣除了建仓费用，保留该费用而不重新归一化。
    out["策略"] = strategy
    for code, (label, _) in BENCHMARKS.items():
        price = benchmarks[code].reindex(equity.index)
        # 停牌或份额折算造成的短暂缺报价，只能沿用已知旧价估值。
        price = price.ffill(limit=3)
        if price.isna().any() or (price <= 0).any():
            missing = price.index[price.isna()][:3].strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{code} 在回测日缺少有效复权价：{missing}")
        out[label] = price / price.iloc[0]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-dir", type=Path, default=ROOT / "output" / "backtest")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "benchmark_comparison")
    parser.add_argument("--offline", action="store_true", help="只读本地 Tushare 缓存")
    parser.add_argument("--dividend-attribution",action="store_true",help="新默认全历史假设与宽基/科技示例对比，并归因红利模块")
    parser.add_argument("--end",type=date.fromisoformat,default=date(2026,10,20),help="归因研究截止日")
    args = parser.parse_args()
    if args.dividend_attribution:
        folder=ROOT/'output/default_benchmark_20261021' if args.output_dir==ROOT/'output/benchmark_comparison' else args.output_dir
        dividend_comparison(args.end,args.cache_dir,folder,args.offline)
        return

    summary = json.loads((args.backtest_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("strategy", {}).get("name") != STRATEGY_NAME:
        raise ValueError("回测摘要不是当前默认策略；请先运行 run_backtest.py")
    equity = pd.read_csv(args.backtest_dir / "equity.csv", index_col=0, parse_dates=True,
                         encoding="utf-8-sig")
    equity.index = pd.DatetimeIndex(equity.index, name="date")
    first, last = equity.index[0].date(), equity.index[-1].date()
    prices = {code: load_symbol(code, first, last, args.cache_dir, offline=args.offline)
              for code in BENCHMARKS}
    missing_quotes = {code: prices[code].reindex(equity.index).index[
        prices[code].reindex(equity.index).isna()].strftime("%Y-%m-%d").tolist()
        for code in BENCHMARKS}
    nav = make_comparison(equity, prices)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nav.to_csv(args.output_dir / "comparison_nav.csv", encoding="utf-8-sig", index_label="date")
    (args.output_dir / "data_checks.json").write_text(json.dumps({
        "strategy": STRATEGY_NAME,
        "source": "Tushare fund_daily close × same-day fund_adj factor",
        "proxy": "buy-and-hold broad-index ETFs, not official index total-return series",
        "benchmark_missing_quote_days": missing_quotes,
        "valuation_of_missing_quote": "prior available close; at most 3 trading days",
        "period": {"start": str(first), "end": str(last)},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    days = len(nav)
    rows = []
    for label in nav:
        series = nav[label]
        rows.append({"name": label, "total_return": float(series.iloc[-1] - 1),
                     "annual_return": float(series.iloc[-1] ** (252 / days) - 1),
                     "max_drawdown": float((series / series.cummax().clip(lower=1) - 1).min())})
    metrics = pd.DataFrame(rows).sort_values("total_return", ascending=True)
    metrics.to_csv(args.output_dir / "comparison_summary.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, (ax, bars) = plt.subplots(2, 1, figsize=(13, 8.5),
                                   gridspec_kw={"height_ratios": [3, 1.35]})
    colors = {"策略": "#111827", **{label: color for label, color in BENCHMARKS.values()}}
    for label in nav:
        ax.plot(nav.index, (nav[label] - 1) * 100, label=label,
                color=colors[label], linewidth=2.6 if label == "策略" else 1.5,
                alpha=1.0 if label == "策略" else 0.85)
    ax.axhline(0, color="#94a3b8", linewidth=0.8)
    ax.set_title(f"默认策略与主要 A 股宽基 ETF 收益对比  |  {first} 至 {last}",
                 fontsize=15, pad=12)
    ax.set_ylabel("累计收益率（%）")
    ax.grid(alpha=0.2)
    ax.legend(ncol=4, loc="upper left", frameon=False)

    values = metrics["total_return"].to_numpy() * 100
    labels = metrics["name"].tolist()
    bars.barh(labels, values, color=[colors[label] for label in labels], alpha=0.88)
    bars.axvline(0, color="#64748b", linewidth=0.8)
    bars.set_xlabel("全区间累计收益率（%）")
    bars.grid(axis="x", alpha=0.2)
    for y, value in enumerate(values):
        bars.annotate(f"{value:+.1f}%", (value, y),
                      xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                      va="center", ha="left" if value >= 0 else "right", fontsize=9)
    bars.set_xlim(min(-15, float(values.min()) - 10), max(65, float(values.max()) + 10))
    fig.text(0.5, 0.015,
             "宽基曲线为代表 ETF 的后复权收盘价买入持有；策略含单边 3bp 手续费及 2bp 滑点，ETF 买入持有未另计交易费。",
             ha="center", color="#475569", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    chart = args.output_dir / "broad_market_comparison.png"
    fig.savefig(chart, dpi=170)
    plt.close(fig)

    drawdowns = metrics.sort_values("max_drawdown")
    fig_dd, ax_dd = plt.subplots(figsize=(10, 5.5))
    dd_labels = drawdowns["name"].tolist()
    dd_values = drawdowns["max_drawdown"].to_numpy() * 100
    ax_dd.barh(dd_labels, dd_values, color=[colors[label] for label in dd_labels], alpha=0.9)
    ax_dd.axvline(0, color="#64748b", linewidth=0.8)
    ax_dd.set_xlim(min(-60, float(dd_values.min()) - 6), 4)
    ax_dd.set_xlabel("最大回撤（%）")
    ax_dd.set_title(f"默认策略与主要 A 股宽基 ETF 最大回撤  |  {first} 至 {last}",
                    fontsize=14, pad=12)
    ax_dd.grid(axis="x", alpha=0.18)
    for y, value in enumerate(dd_values):
        ax_dd.text(value - 0.8, y, f"{value:.1f}%", va="center", ha="right", fontsize=10)
    fig_dd.text(0.5, 0.018,
                "按统一起点后的净值计算历史最高点至其后最低点；宽基为代表 ETF 后复权买入持有。",
                ha="center", color="#475569", fontsize=9)
    fig_dd.tight_layout(rect=(0, 0.04, 1, 1))
    dd_chart = args.output_dir / "drawdown_comparison.png"
    fig_dd.savefig(dd_chart, dpi=170)
    plt.close(fig_dd)
    print(metrics.sort_values("total_return", ascending=False).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Chart: {chart}")
    print(f"Drawdown chart: {dd_chart}")


if __name__ == "__main__":
    main()
