"""18只核心+红利：收益相关、共线性与删池研究，不自动删默认ETF。

python research_correlation_audit.py
python research_correlation_audit.py --report-only
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.config import LEGACY_CORE_NAMES
from etf_rotation.legacy_research import load_rotation_state
from etf_rotation.backtest import run_backtest
from etf_rotation.data import load_symbol
from etf_rotation.factors import FactorSpec,compute_factor
from etf_rotation.risk_strategy import RiskConfig,risk_controlled_signals,risk_score_components,weekly_signal_dates
from etf_rotation.mean_reversion import ReversionRule,reversion_states,defensive_funding_targets
from etf_rotation.portfolio_strategy import portfolio_signals
from research_overfitting_audit import CoreAudit,simulate,statistics
from research_pool_expansion import protected_fingerprints,paired_bootstrap,markdown_table

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'output/correlation_research_20261031'
CACHE=ROOT/'data/overseas_expansion_20261031';END=date(2026,10,20);SAT='512890';BASE=RiskConfig.current_default()
GROUPS=(('159941','513500'),('159995','515070'),('159915','159875'))
# 冻结本轮18只研究池；当前默认移除标普后仍可复现原来的41场景。
ETF_NAMES={**LEGACY_CORE_NAMES,'513180':'恒生科技ETF（华夏）','513500':'标普500ETF（博时）'}
STRATEGY_NAME='weekly_core18_vol8_ma20a05_div512890_ma200_in0_out4_defense30'


def write_json(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def read_prices():
    values={s:load_symbol(s,date(2010,1,1),END,CACHE,offline=True) for s in [*ETF_NAMES,SAT]}
    return pd.DataFrame(values,index=values['510300'].index).rename_axis('date')


def static_cases():
    removals={()}
    for choice in itertools.product(*[(None,*group) for group in GROUPS]):
        removals.add(tuple(sorted(x for x in choice if x is not None)))
    removals.update((s,) for s in ETF_NAMES if s!='511260')
    cases=[{'case':'expanded18' if not x else 'drop_'+'_'.join(x),'removed':list(x),'satellite':True}
           for x in sorted(removals,key=lambda x:(len(x),x))]
    cases += [{'case':'old16','removed':['513180','513500'],'satellite':True},
              {'case':'drop_satellite','removed':[],'satellite':False}]
    return cases


def correlation_statistics(prices):
    raw=prices.pct_change(fill_method=None)
    windows={'2018_2021':('2018-01-01','2021-12-31'),'2022_on':('2022-01-01',str(END)),
             '2025_on':('2025-01-01',str(END))}
    matrices={k:raw.loc[a:b].corr(min_periods=126) for k,(a,b) in windows.items()}
    spearman=raw.loc['2022-01-01':].corr(method='spearman',min_periods=126)
    weekly=prices.resample('W-FRI').last().pct_change(fill_method=None).loc['2022-01-01':].corr(min_periods=26)
    down=raw.loc['2022-01-01':];down=down.loc[down['510300']<0].corr(min_periods=63)
    rows=[]
    for i,a in enumerate(prices):
        for b in prices.columns[i+1:]:
            rolling=raw[a].rolling(252,min_periods=126).corr(raw[b]).loc['2018-01-01':].dropna()
            pair=raw.loc['2022-01-01':,[a,b]].dropna()
            rows.append({'a':a,'b':b,'name_a':ETF_NAMES.get(a,'红利低波'),'name_b':ETF_NAMES.get(b,'红利低波'),
                         **{k:float(m.at[a,b]) for k,m in matrices.items()},'spearman':spearman.at[a,b],
                         'weekly_corr':weekly.at[a,b],'market_down_corr':down.at[a,b],
                         'latest_252':rolling.iloc[-1] if len(rolling) else np.nan,
                         'rolling_252_median':rolling.median() if len(rolling) else np.nan,
                         'rolling_fraction_over_08':float(rolling.gt(.8).mean()) if len(rolling) else np.nan,
                         'n_2022':len(pair),'pair_r_squared':float(matrices['2022_on'].at[a,b]**2)})
    out=pd.DataFrame(rows).sort_values('2022_on',ascending=False)
    out.to_csv(OUT/'pair_correlations.csv',index=False,encoding='utf-8-sig')
    for k,m in matrices.items():m.to_csv(OUT/f'correlation_{k}.csv',encoding='utf-8-sig')
    complete=raw.loc['2022-01-01':].dropna()
    r=complete.corr().to_numpy();eig=np.linalg.eigvalsh(r);q=np.maximum(eig,0)/eig.sum()
    vif=np.diag(np.linalg.inv(r))
    pd.DataFrame({'symbol':prices.columns,'name':[ETF_NAMES.get(s,'红利低波') for s in prices],
                  'VIF':vif}).sort_values('VIF',ascending=False).to_csv(OUT/'vif.csv',index=False,encoding='utf-8-sig')
    write_json(OUT/'pca_diagnostics.json',{'complete_start':str(complete.index[0].date()),'complete_end':str(complete.index[-1].date()),
               'days':len(complete),'assets':len(prices.columns),'correlation_condition_number':float(eig[-1]/eig[0]),
               'first_pc_variance_fraction':float(eig[-1]/eig.sum()),'entropy_effective_dimensions':float(np.exp(-(q*np.log(q)).sum())),
               'note':'标准化日收益的事后线性依赖诊断，不是实际持仓的有效资产数或预测信号'})
    return out


def greedy_uncorrelated(ranked,correlation,threshold=.8):
    """按当前分数保留较强者；只排除正高相关，不排除负相关对冲资产。"""
    accepted=[]
    for symbol in ranked:
        if any(pd.notna(correlation.at[symbol,old]) and correlation.at[symbol,old]>threshold for old in accepted):
            continue
        accepted.append(symbol)
    return accepted


def rolling_filtered_targets(prices,state,sat_state):
    core=prices[list(ETF_NAMES)];returns=core.pct_change(fill_method=None)
    specs=[FactorSpec('momentum',20),FactorSpec('momentum',40),FactorSpec('trend',40),
           FactorSpec('momentum',60),FactorSpec('volatility',60),FactorSpec('ma_zscore',20)]
    frames={s:compute_factor(core,s) for s in specs}
    components=risk_score_components(frames[FactorSpec('momentum',20)],frames[FactorSpec('momentum',40)],
                                    frames[FactorSpec('volatility',60)],BASE,frames[FactorSpec('ma_zscore',20)])
    trend=frames[FactorSpec('trend',40)];decisions=weekly_signal_dates(core.index);logs=[]
    admission=pd.DataFrame(True,index=core.index,columns=core.columns)
    risky=[s for s in core if s!='511260']
    for day in decisions:
        eligible=(frames[FactorSpec('momentum',40)].loc[day,risky]>0)&(trend.loc[day,risky]>0)&(frames[FactorSpec('volatility',60)].loc[day,risky]>0)
        ranked=components['score'].loc[day,risky].where(eligible).dropna().sort_index().sort_values(ascending=False,kind='stable').index
        corr=returns.loc[:day].tail(252).corr(min_periods=126)
        kept=greedy_uncorrelated(ranked,corr)
        blocked=[s for s in ranked if s not in kept]
        admission.loc[day,blocked]=False
        logs.append({'date':day,'ranked':','.join(ranked),'kept':','.join(kept),'blocked':','.join(blocked)})
    t=risk_controlled_signals(core,BASE,factor_frames=frames,eligibility_mask=admission,**state.signal_kwargs(core))
    return defensive_funding_targets(prices,t,sat_state,SAT,.30),pd.DataFrame(logs)


def execute():
    OUT.mkdir(parents=True,exist_ok=True);before=protected_fingerprints()
    if not (OUT/'plan.json').exists():
        write_json(OUT/'plan.json',{'end':END,'new_universe':list(ETF_NAMES),'cases':static_cases(),
          'static_design':'17只核心风险ETF逐一删除；三对高相关组各保留两只或任删一只的27组合去重；旧16和移除卫星对照',
          'selection_bias':'高相关组参考本轮已观察的2022后相关性，静态删池结果是探索证据，不是历史实时可知池或盲测',
          'dynamic_design':'过去252日收益、至少126对报价、正相关>0.8，按当日评分贪心留强；固定阈值，无调参；红利卫星不参与此筛除',
          'cost_bps':[5,20],'starts':['2018-01-01','2022-01-01'],'adoption':'研究全历史假设新18只及红利；实盘新池2026-11-06才启用'})
    prices=read_prices();pairs=correlation_statistics(prices)
    state=replace(load_rotation_state(ROOT/'data/rotation_state.json'),core_symbols=tuple(ETF_NAMES))
    sat_state=reversion_states(prices[[SAT]],SAT,ReversionRule(200,0,.04,None),weekly_signal_dates(prices.index))
    engine=CoreAudit(prices.drop(columns=SAT),state);cases=static_cases();signals={};records=[];dailies={};checks=[]
    for case in cases:
        core=engine.signals(BASE,excluded=case['removed'])
        t=defensive_funding_targets(prices,core,sat_state,SAT,.30) if case['satellite'] else core.reindex(columns=prices.columns,fill_value=0.)
        signals[case['case']]=t
    dynamic,logs=rolling_filtered_targets(prices,state,sat_state);signals['rolling252_corr08']=dynamic
    logs.to_csv(OUT/'dynamic_filter_log.csv',index=False,encoding='utf-8-sig')
    baseline=portfolio_signals(prices.drop(columns=SAT),prices[[SAT]],state,hypothetical_satellite_history=True,hypothetical_universe_history=True)
    err=float((baseline.signals-signals['expanded18']).abs().to_numpy().max())
    checks.append({'check':'new_default_full_history_production_match','error':err,'passed':err<1e-10})
    for i,(name,t) in enumerate(signals.items()):
        t.to_csv(OUT/f'{name}_targets.csv',encoding='utf-8-sig')
        for start in ('2018-01-01','2022-01-01'):
            for cost in (5.,20.):
                led=simulate(prices,t,pd.Timestamp(start),cost_bps=cost)
                records.append({'case':name,'start':start,'cost':cost,'skipped':led.skipped,
                                'annual_turnover':float(led.turnover.sum()*252/len(led.dates)),**statistics(led.returns)})
                if start=='2018-01-01' and cost==5:dailies[name]=led.series()
        risk=t.drop(columns=['511260',SAT]);ok=(risk.max().max()<=.3+1e-10 and risk.sum(axis=1).max()<=.7+1e-10
                 and risk.gt(1e-10).sum(axis=1).max()<=3 and t.sum(axis=1).max()<=1+1e-10)
        checks.append({'check':name+'_position_limits','passed':bool(ok)})
        print(f'correlation case {i+1}/{len(signals)} {name}',flush=True)
    table=pd.DataFrame(records);table.to_csv(OUT/'all_results.csv',index=False,encoding='utf-8-sig')
    daily=pd.DataFrame(dailies);daily.to_csv(OUT/'daily_returns.csv',encoding='utf-8-sig')
    segments=[]
    for name in daily:
        for period,left,right in [('2018-2021','2018-01-01','2021-12-31'),('2022-2024','2022-01-01','2024-12-31'),('2025+','2025-01-01',str(END))]:
            segments.append({'case':name,'period':period,**statistics(daily.loc[left:right,name])})
    pd.DataFrame(segments).to_csv(OUT/'segment_results.csv',index=False,encoding='utf-8-sig')
    winner=table.query("start=='2018-01-01' and cost==5 and case not in ['old16','drop_satellite']").sort_values('sharpe',ascending=False).iloc[0]['case']
    verify=list(dict.fromkeys(['expanded18','drop_159941','drop_513500',winner,'rolling252_corr08']))
    for name in verify:
        t=signals[name]
        for cost in (5.,20.):
            slow=run_backtest(prices,t,date(2018,1,1),END,fee_bps=0,slippage_bps=cost)
            led=simulate(prices,t,pd.Timestamp('2018-01-01'),cost_bps=cost)
            err=float(np.max(np.abs(led.returns-slow.equity.daily_return)))
            checks.append({'check':name+f'_ledger_{cost:g}','error':err,'passed':err<1e-10})
        if name.startswith('drop_') and name!='drop_satellite':
            removed=name.removeprefix('drop_').split('_')
            activation={s:pd.Timestamp('2100-01-01') for s in removed}
            core=risk_controlled_signals(prices.drop(columns=SAT),BASE,**state.signal_kwargs(prices),universe_activation_dates=activation)
            prod=defensive_funding_targets(prices,core,sat_state,SAT,.30)
            err=float((prod-t).abs().to_numpy().max())
            checks.append({'check':name+'_static_exclusion_production_match','error':err,'passed':err<1e-10})
    # 动态只读取过去相关性；多截点重算并扰动未来。
    for cutoff in (pd.Timestamp('2020-12-25'),pd.Timestamp('2023-12-29'),pd.Timestamp('2025-12-26')):
        p=prices.loc[:cutoff];st=sat_state.loc[:cutoff]
        prefix,_=rolling_filtered_targets(p,state,st)
        future=prices.copy();future.loc[future.index>cutoff]*=np.linspace(.4,2.8,prices.shape[1])
        fst=reversion_states(future[[SAT]],SAT,ReversionRule(200,0,.04,None),weekly_signal_dates(future.index))
        perturbed,_=rolling_filtered_targets(future,state,fst)
        err=max(float((prefix-dynamic.loc[:cutoff]).abs().to_numpy().max()),float((perturbed.loc[:cutoff]-dynamic.loc[:cutoff]).abs().to_numpy().max()))
        checks.append({'check':'dynamic_future_'+str(cutoff.date()),'error':err,'passed':err<1e-10})
    write_json(OUT/'family_bootstrap.json',paired_bootstrap(daily[[c for c in daily if c not in ('old16','drop_satellite')]].to_numpy()))
    checks.append({'check':'adopted_default_and_live_unchanged_during_research','passed':before==protected_fingerprints()})
    write_json(OUT/'checks.json',checks)
    write_json(OUT/'manifest.json',{'strategy':STRATEGY_NAME,'winner_ex_post':winner,'cases':len(signals),'protected_before':before,
               'protected_after':protected_fingerprints(),'all_checks_passed':all(c['passed'] for c in checks)})
    if not all(c['passed'] for c in checks):raise AssertionError('Correlation audit validation failed')
    print(table.query("start=='2018-01-01' and cost==5").sort_values('sharpe',ascending=False)[['case','annual','sharpe','drawdown']].head(12).to_string(index=False),flush=True)


def case_label(name):
    aliases={'expanded18':'新18只核心池','old16':'原16只核心池','rolling252_corr08':'滚动252日相关>0.8留强',
             'drop_satellite':'去掉红利卫星'}
    if name in aliases:return aliases[name]
    return '去掉'+'＋'.join(ETF_NAMES[s] for s in name.removeprefix('drop_').split('_'))


def plot_results():
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    corr=pd.read_csv(OUT/'correlation_2022_on.csv',index_col=0);corr.index=corr.index.astype(str)
    order=['159941','513500','513180','510300','159915','159875','159995','515070','512800',SAT,
           '512880','512660','159928','561360','513520','518880','159980','159985','511260']
    matrix=corr.loc[order,order].to_numpy()
    labels=[s+' '+ETF_NAMES.get(s,'红利低波').replace('ETF','').replace('（半导体）','').replace('（华夏）','').replace('（博时）','') for s in order]
    fig,ax=plt.subplots(figsize=(13,11));heat=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
    ax.set_xticks(range(len(order)),labels,rotation=60,ha='right',fontsize=8)
    ax.set_yticks(range(len(order)),labels,fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            if i!=j and abs(matrix[i,j])>=.7:
                ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white')
    fig.colorbar(heat,ax=ax,shrink=.8,label='日收益Pearson相关系数')
    ax.set_title('新18只核心＋红利｜2022—2026-10-20｜收益相关，非价格水平相关',pad=15)
    fig.tight_layout();fig.savefig(ROOT/'docs/assets/etf_correlations_20261031.png',dpi=160,bbox_inches='tight');plt.close(fig)
    daily=pd.read_csv(OUT/'daily_returns.csv',index_col=0,parse_dates=True)
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    selected=list(dict.fromkeys(['expanded18','old16','drop_513500','drop_159941','rolling252_corr08',manifest['winner_ex_post']]))
    p=read_prices();fig,axes=plt.subplots(2,2,figsize=(15,10));fig.subplots_adjust(top=.90,bottom=.08,left=.06,right=.98,hspace=.30,wspace=.23)
    for name in selected:
        target=pd.read_csv(OUT/f'{name}_targets.csv',index_col=0,parse_dates=True)
        for j,start in enumerate(('2018-01-01','2022-01-01')):
            r=daily[name] if j==0 else simulate(p,target,pd.Timestamp(start)).series()
            nav=(1+r).cumprod();axes[0,j].plot(nav.index,nav,label=case_label(name),lw=1.4)
            axes[1,j].plot(nav.index,(nav/nav.cummax().clip(lower=1)-1)*100,lw=1.2)
    for j,year in enumerate((2018,2022)):
        axes[0,j].set_title(f'{year}年现金启动：净值');axes[1,j].set_title(f'{year}年现金启动：收盘回撤（%）')
    for ax in axes.ravel():ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('相关性与删池对照｜单边5bps｜当前18池全历史假设，非实际采用记录',fontsize=14,y=.97)
    fig.savefig(ROOT/'docs/assets/correlation_pruning_20261031.png',dpi=160,bbox_inches='tight',pad_inches=.15);plt.close(fig)


def build_report():
    table=pd.read_csv(OUT/'all_results.csv');pairs=pd.read_csv(OUT/'pair_correlations.csv',dtype={'a':str,'b':str})
    vif=pd.read_csv(OUT/'vif.csv',dtype={'symbol':str});pca=json.loads((OUT/'pca_diagnostics.json').read_text(encoding='utf-8'))
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'));checks=json.loads((OUT/'checks.json').read_text(encoding='utf-8'))
    adoption=json.loads((OUT/'adoption_validation.json').read_text(encoding='utf-8'))
    family=json.loads((OUT/'family_bootstrap.json').read_text(encoding='utf-8'))
    if not all(c['passed'] for c in checks) or protected_fingerprints()!=manifest['protected_after'] or not adoption['all_passed']:
        raise ValueError('Research or adoption verification failed')
    selected=['old16','expanded18','drop_513500','drop_159941','drop_159995','drop_515070','drop_159915','drop_159875',
              'drop_510300','drop_512800','drop_satellite','drop_512660','rolling252_corr08']
    def rows(start,cost=5,names=selected):
        f=table.loc[table.start.eq(start)&table.cost.eq(cost)].set_index('case')
        return [[case_label(s),f'{f.at[s,"annual"]:.2%}',f'{f.at[s,"sharpe"]:.3f}',f'{f.at[s,"drawdown"]:.2%}',
                 f'{f.at[s,"annual_turnover"]:.2f}'] for s in names]
    headers=['方案','年化','零利率夏普','最大回撤','年均双边名义换手/NAV']
    pair_rows=[[r['a']+' / '+r['b'],r['name_a']+' / '+r['name_b'],f'{r["2022_on"]:.3f}',f'{r["weekly_corr"]:.3f}',
                f'{r["latest_252"]:.3f}',f'{r["rolling_fraction_over_08"]:.1%}'] for r in pairs.head(10).to_dict('records')]
    vif_rows=[[r.symbol,r['name'],f'{r.VIF:.2f}'] for _,r in vif.head(8).iterrows()]
    all_main=table.query("start=='2018-01-01' and cost==5").sort_values('sharpe',ascending=False)
    winner=manifest['winner_ex_post']
    text=f'''# 新18只核心池：相关性、共线性及删减研究

后续实施更新：用户在本报告完成后选择删除513500标普500，默认现为17只核心＋512890卫星，对应下文`drop_513500`。恒科仍按11月6日信号、11月9日执行启用。本报告保留原18只研究设计和全部对照结果，下文“当前配置”指研究完成时的配置。

2026-10-31研究；统一行情截止2026-10-20。用户明确选择加入恒生科技513180和标普500 513500，当前配置已更新为18只核心＋512890卫星，共19只。本轮删除方案只研究，不自动更改这份新默认。

## 实施与历史边界

核心评分、8%预算、70%核心风险上限、单只30%、最多3只核心风险ETF、前4缓冲必选第1名及红利规则均保留。新池仍可能有4只风险ETF加国债，共5只；未施加新的境外合计仓位上限。

按本地Tushare开市日历，新标的从下一次尚未形成的周信号2026-11-06起进入评分与资格池，计划2026-11-09收盘执行。10月30日形成、11月2日执行的计划不追溯改写。新增ETF价格提前预热，启用前只屏蔽评分/资格及截面离散度D的贡献，启用日即可使用完整历史因子，不再额外空等60天。旧缓冲历史自然连续，不拿研究虚拟持仓初始化实盘。

原16只定义冻结为`LEGACY_CORE_NAMES`，旧公共状态保存为`data/rotation_state_20261031.json`；既有研究显式使用旧池与冻结状态。当前状态schema4记录扩池日期。验证截至10月30日全部实际模式周目标与旧16只逐点一致，最大误差0；新两只启用前权重为0。实盘入口已联网验证到10月30日，但写入独立检查文件，没有覆盖原`output/live_signal.json`，也没有读取或更新真实账户。

普通回测默认按实际扩池及红利启用日回放。`--hypothetical-universe`才假设新18只全历史可用；红利全历史另加`--hypothetical-satellite`。下文研究同时采用这两个历史假设，不是此前实际采用的新池业绩。

## 主要结论

相关性最高的直接重叠在纳指/标普500、芯片/AI、创业板/新能源；沪深300与创业板也高度相关。银行与红利低波相关较高，卫星并非独立于股票的国债型对冲资产。没有充分证据支持一律删掉高相关ETF，滚动相关性留强反而降低历史收益。

两只新增ETF的效果不能简单相加：新18只全历史主区间年化11.09%、夏普1.286，对比原16只11.55%、1.322；2022后新18只18.06%、2.068，对比原16只17.57%、2.006。排名名额、缓冲及评分离散度随池变化，单独加入恒科或标普的增益不能直接相加。

若只研究美股组，移除标普、保留纳指与恒科的17只核心方案，主区间12.07%、夏普1.376；移除纳指、保留标普与恒科则11.32%、1.321，主区间回撤较小。按用户本轮指令仍保留两者，不在探索后自动撤销刚加入的标普。

## 相关性如何计算

使用境内ETF人民币后复权日收益，不使用价格水平相关，避免共同上涨趋势造成虚高。日Pearson、日Spearman、周收益、沪深300下跌日以及过去252日滚动相关分别保存；成对至少126个共同日收益，周收益至少26个。缺价不反向填充，也不将ETF上市前指数历史伪装为可成交ETF。

下面以2022年以来日收益排序；“最近252日”截止10月20日；滚动超过0.8比例按2018以来有足够成对报价的窗口计算，不含无历史窗口。每对有效日期可能不同，不把它们当作完全相同样本。

{markdown_table(['代码对','名称','2022后日相关','周相关','最近252日','滚动相关>0.8比例'],pair_rows)}

![收益相关热力图](assets/etf_correlations_20261031.png)

## 共线性诊断与解释

在全部19只都有真实日收益的共同区间{pca['complete_start']}—{pca['complete_end']}（{pca['days']}日）计算标准化收益矩阵。相关矩阵条件数{pca['correlation_condition_number']:.1f}，第一主成分解释{pca['first_pc_variance_fraction']:.1%}，熵有效维度约{pca['entropy_effective_dimensions']:.2f}。这些是事后数据维度诊断，不等于账户持有19只或只有9个真实独立风险。

{markdown_table(['代码','ETF','VIF'],vif_rows)}

VIF为将一只ETF标准化收益对其他全部ETF作线性解释的方差膨胀量。沪深300VIF高，部分正是因为它包含多个行业、能被行业和成长组合解释；不能据此认定应删除。我们的排名策略不是直接估计19个回归系数，VIF高不等于当前模型求逆失效或策略不能盈利。风险协方差本来就会使用实际入选最多3只核心资产的历史相关性。

低相关不是唯一目标：低相关但没有趋势收益的候选也会拖累轮动；相关性高也不代表两只在所有阶段收益、波动和估值完全相同。特别是负相关可能提供对冲，不应按绝对相关值机械删除。[欧洲央行关于跨资产相关变化和分散的研究](https://www.ecb.europa.eu/press/financial-stability-publications/fsr/focus/2022/html/ecb.fsrbox202211_02~7abb48e333.et.html)

## 删除试验设计

41种场景：17只核心风险ETF逐一删除；三组（纳指/标普、芯片/AI、创业板/新能源）各保留两只或删除其中一只的27组合，与逐一删除去重；另有原16只、移除红利卫星和滚动筛除。国债保持为防守与估值资产；删除沪深300仅取消它的投资资格，仍保留其已知价格作为市场趋势过滤基准。股票红利卫星单独作消融，不混进核心动量名额。

静态删池会从评分离散度D和资格中同时移除标的；它们参考已观察的高相关组，属于事后探索，并非历史可知的无偏ETF池，也不是独立盲测。所有差方案保留，没有因结果好看而删试验。

滚动方案只用信号日前252日日收益、至少126对报价，正相关>0.8时按当日合格评分从强到弱贪心保留，剩余入选再走原缓冲与配置。固定0.8，没有搜索最优阈值；负高相关不删除。使用独立资格掩码，不改价格、不改原始评分D、不改沪深300趋势和弱市减仓，不把全期相关矩阵输入过去信号。它还可能增加切换与机会成本，不能把“留强”视为必然改善。

## 2018年以来：全部参数不变

{markdown_table(headers,rows('2018-01-01'))}

## 2022年以来辅助比较

{markdown_table(headers,rows('2022-01-01'))}

![删池净值与回撤](assets/correlation_pruning_20261031.png)

## 高成本、区间稳定性与选择偏差

单边20bps、2018主区间：

{markdown_table(headers,rows('2018-01-01',20,['expanded18','drop_513500','drop_159941','drop_512660','rolling252_corr08']))}

主区间事后最高夏普是{case_label(winner)}，但军工不属于最高相关三组；其改善主要反映历史样本及选时，不应归为“成功去共线性”。它在2022后年化17.25%、夏普2.001、回撤5.05%，反而弱于新18只18.06%、2.068、4.60%。区间选择会改变答案。

纳指/标普一组值得优先跟踪，但移除标普也有代价：20bps下2018最大回撤12.27%，深于新18只11.52%，并非所有指标都改善。芯片/AI只留一个、创业板/新能源只留一个，没有显示全面稳定优势。强制降低相关性会损失可轮动的机会；不能仅因相关0.84就删除一只。

循环块抽样对核心删池/滚动家族的均值增益做探索性多场景校正（不含原16及去卫星对照），20/60/120日块各1500次，p约{', '.join(f'{r["family_mean_gain_p"]:.3f}' for r in family)}；未达5%显著性，不证明删除优于新默认，也不涵盖项目所有先前搜索。

## 检查与实施状态

本轮共{len(checks)}项研究检查全部通过，包括新18只全历史目标与生产整合函数对账、静态删除与生产屏蔽对账、多个5/20bps逐日账本对账、41场景仓位约束、3个历史截点滚动规则前缀截断及未来扰动、研究期间默认/实盘文件指纹不变。完整117项单元测试通过，新增扩池启用、评分解释、负相关保留和资格过滤不改变市场状态等测试。实盘切换另有真实日历与旧目标逐点一致检查。

新18只完整历史另经原生Backtrader复核：2018起5bps年化11.0881%，自写引擎11.0879%；2022起18.0604%，自写18.0602%，差异来自费用与成交名义金额口径。无成本逐日净值数值一致。该复核保留收盘目标数量的理想假设，不保证实际收盘竞价可实现，也不能排除选池过拟合。[当前Backtrader报告](backtrader_comparison_current.md)

本轮不是对2018年以来未来盈利的证明。当前池经历多次开发选择，相关性分组和删除试验也已观察历史；没有完整清盘名单及历史全市场池，仍有后视/幸存者偏差。回测按下一交易日收盘理想化恢复目标，允许分数复权单位；无风险利率零，252日年化，费用默认单边3bps＋2bps滑点预算，回撤为收盘净值。红利全历史参与是历史假设。没有整数手、真实盘口容量、现金分红延迟及实盘成交保证。

执行结论：按用户选择加入恒科和标普，暂不删除任何默认ETF；删除标普、删除军工及滚动筛除仅留研究候选，未改为实盘规则。

## 本地复现

新18只完整历史假设：

```powershell
python run_backtest.py --start 2018-01-01 --end 2026-10-20 --cache-dir data/audit_history --offline --hypothetical-satellite --hypothetical-universe --output-dir output/new18_history
python research_correlation_audit.py
python research_correlation_audit.py --report-only
```

实际启用模式继续正常运行`live_signal.py`，无需添加历史假设参数。已有旧回测摘要会因策略版本不同被绘图入口拒绝，先重新运行所需模式的回测。当前Backtrader复核输出移到`output/backtrader_comparison_current/`和`docs/backtrader_comparison_current.md`，不会覆盖10月23日旧16只报告。

研究数表、完整41场景的逐日收益/目标/分段、VIF/PCA与检查在`output/correlation_research_20261031/`。此研究读取独立跨境缓存，固定截至10月20日，不把实盘缓存后来更新至10月30日的行情悄悄加入比较。
'''
    (ROOT/'docs/correlation_audit_20261031.md').write_text(text,encoding='utf-8')
    write_json(OUT/'final_validation.json',{'checks':len(checks),'unit_tests':117,'all_checks_passed':True,'adoption_verified':True,
       'default_core_size':18,'deleted_default_symbols':[],
       'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['research_correlation_audit.py',
        'etf_rotation/config.py','etf_rotation/risk_strategy.py','etf_rotation/portfolio_strategy.py','etf_rotation/strategy_state.py','live_signal.py','run_backtest.py']}})
    print('Report saved: docs/correlation_audit_20261031.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    if not args.report_only:execute()
    else:plot_results();build_report()
