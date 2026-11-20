"""候选行业ETF与CL120当前18只ETF的相关性热力图；只读独立研究缓存。

python analyze_pool_correlations.py
需要先完成 research_cl120_pool_expansion.py 的筛选、数据及主回测。
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.factors import validate_prices
from research_cl120_pool_expansion import (read_data,CORE_NAMES,SAT,SCREEN,END,OUT as STUDY_OUT,
                                         experiment_cases,protected)

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/pool_correlations_20261109'
ASSETS=ROOT/'docs/assets'
# 按经济类别展示现有18只，包括独立红利卫星；不是仅看本周实际三只持仓。
GROUPS=[('A股宽基',['510300','159915']),('成长行业',['159995','159875','515070']),
        ('金融/消费/军工',['512800','512880','159928','512660']),('能源股票',['561360']),
        ('商品',['159980','159985','518880']),('境外指数',['159941','513520','513180']),
        ('国债',['511260']),('红利卫星',[SAT])]
EXISTING=[s for _,symbols in GROUPS for s in symbols]
RISKY=[s for s in EXISTING if s!='511260']
NAMES={**CORE_NAMES,SAT:'红利低波卫星'}
SHORT={'510300':'沪深300','159915':'创业板','159995':'芯片','159875':'新能源','515070':'人工智能',
       '512800':'银行','512880':'证券','159928':'消费','512660':'军工','561360':'石油股票',
       '159980':'有色期货','159985':'豆粕期货','518880':'黄金','159941':'纳指','513520':'日经',
       '513180':'恒生科技','511260':'十年国债',SAT:'红利低波'}

def json_file(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def daily_returns(prices):
    validate_prices(prices)
    return prices.pct_change(fill_method=None)

def correlation_matrix(returns,candidates,existing,min_pairs):
    """仅用每一对共同有效的日收益；缺价不填充，缺样本不假装相关性为0。"""
    symbols=list(dict.fromkeys([*candidates,*existing]))
    values=returns.loc[:,symbols].replace([np.inf,-np.inf],np.nan)
    n=values[candidates].notna().astype(np.int64).T.dot(values[existing].notna().astype(np.int64))
    rho=values.corr(method='pearson',min_periods=min_pairs).reindex(index=candidates,columns=existing)
    return rho,n

def window_definitions(returns):
    long=returns.loc['2022-01-01':]
    return {'long':(long,252),'recent':(returns.tail(252),200),
            'down':(long.loc[long['510300'].lt(0)],120),
            'crash':(long.loc[long['510300'].le(-.01)],60)}

def highest_overlap(row,risky):
    x=row.reindex(risky).dropna()
    if x.empty:return float('nan'),None
    symbol=x.idxmax()
    return float(x[symbol]),symbol

def priority_table(matrices,candidates,risky):
    """按长期/最近一年/下跌日中最大的同向相关排序；不是收益预测或选参。"""
    rows=[]
    for s in candidates:
        row={'symbol':s}
        for key in ('long','recent','down','crash'):
            rho=matrices[key].loc[s]
            value,peer=highest_overlap(rho,risky)
            valid=rho.reindex(risky).dropna()
            row[key+'_max']=value;row[key+'_peer']=peer
            row[key+'_top3_mean']=float(valid.nlargest(3).mean()) if len(valid)>=3 else float('nan')
            row[key+'_mean']=float(valid.mean()) if len(valid) else float('nan')
        x=[row[k+'_max'] for k in ('long','recent','down')]
        row['worst_of_three']=max(x) if np.isfinite(x).all() else float('nan')
        rows.append(row)
    return pd.DataFrame(rows).sort_values(['worst_of_three','long_max','symbol'],na_position='last').reset_index(drop=True)

def draw_matrix(ax,rho,title):
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dedede')
    values=np.ma.masked_invalid(rho.to_numpy(float))
    plot=ax.imshow(values,cmap=cmap,vmin=-1,vmax=1,aspect='auto')
    ax.set_xticks(range(len(rho.columns)),[SHORT[s]+'\n'+s for s in rho.columns],rotation=45,ha='right',fontsize=10)
    labels=[SCREEN[s][0].replace('ETF','')+' '+s for s in rho.index]
    ax.set_yticks(range(len(rho)),labels,fontsize=11)
    ax.set_title(title,fontsize=14,pad=18)
    for i in range(len(rho)):
        for j in range(len(rho.columns)):
            value=rho.iloc[i,j]
            ax.text(j,i,'—' if pd.isna(value) else f'{value:.2f}',ha='center',va='center',
                    color='white' if pd.notna(value) and abs(value)>.70 else '#202020',fontsize=10)
    offset=0
    for group,symbols in GROUPS:
        if offset:ax.axvline(offset-.5,color='#666666',linewidth=1)
        offset+=len(symbols)
    ax.set_xticks(np.arange(-.5,len(rho.columns),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(rho.index),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.5);ax.tick_params(which='minor',bottom=False,left=False)
    return plot

def figures(matrices,counts,order,dates):
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
    ASSETS.mkdir(exist_ok=True,parents=True)
    fig,ax=plt.subplots(figsize=(20.5,7.6))
    rho=matrices['long'].reindex(order)
    im=draw_matrix(ax,rho,f'候选行业ETF × CL120现有18只｜2022年至{END}日收益相关性')
    bar=fig.colorbar(im,ax=ax,pad=.015,fraction=.025);bar.set_label('Pearson相关系数',fontsize=11)
    fig.text(.15,.015,'行按跨窗口最高相关性由低到高排列；红色同向、蓝色反向、灰色样本不足。含农业/基建观察层。',fontsize=11)
    fig.tight_layout(rect=[0,.04,1,1])
    fig.savefig(ASSETS/'pool_correlation_heatmap_20261109.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(20.5,13.5))
    for ax,key,title in [(axes[0],'recent',f'最近252个交易日：{dates["recent"][0]}至{END}'),
                          (axes[1],'down','下跌日：2022年以来沪深300当天收益<0')]:
        im=draw_matrix(ax,matrices[key].reindex(order),title)
        fig.colorbar(im,ax=ax,pad=.015,fraction=.025)
    fig.suptitle('相关性稳定性复核｜各面板统一−1至+1色标，行列顺序相同',fontsize=16,y=.995)
    fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(ASSETS/'pool_correlation_regimes_20261109.png',dpi=150);plt.close(fig)

def render_report(summary,counts,checks,top_pair,dates):
    def fmt(x):return '—' if pd.isna(x) else f'{x:.3f}'
    head=['候选','层级','长期最高ρ','最相关现有ETF','最近252日最高ρ','下跌日最高ρ','三窗口最差','126日滚动最高ρ的90%分位']
    body=[]
    for r in summary.itertuples():
        body.append([SCREEN[r.symbol][0]+' '+r.symbol,r.tier,fmt(r.long_max),SHORT[r.long_peer]+' '+r.long_peer,
                     fmt(r.recent_max),fmt(r.down_max),fmt(r.worst_of_three),fmt(r.rolling_max_p90)])
    table='\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |',
                    *['| '+' | '.join(row)+' |' for row in body]])
    risk_rows=[]
    for r in summary.itertuples():
        risk_rows.append([r.symbol,fmt(r.crash_max),str(int(r.crash_pairs_for_max)),fmt(r.correlation_to_CL120)])
    stress='\n'.join(['| 候选 | 沪深300≤−1%日最高ρ | 对应共同样本数 | 与CL120净收益ρ |',
                      '| --- | --- | --- | --- |',*['| '+' | '.join(row)+' |' for row in risk_rows]])
    assetnames='、'.join(SCREEN[s][0]+s for s in top_pair['symbols'])
    backtest=top_pair['backtest']
    report=f'''# 候选ETF与CL120现有池的相关性热力图

日期2026-11-09；沿用[行业扩池研究](cl120_pool_expansion_20261109.md)截至{END}的同一行情快照。9只候选对比当前配置的18只ETF，包括独立红利卫星，不是仅看本周3只目标持仓。默认策略、标的池、参数和实盘JSON均未修改。

## 结果与排序

从同向重复风险看，相对低相关的主层候选优先为{assetnames}。这是对已观察历史的分散价值排序，并不意味着已找到能提高CL120收益的扩池方案。相关性低、波动率低、下跌保护和具有正预期收益是不同属性。

![长期相关性热力图](assets/pool_correlation_heatmap_20261109.png)

{table}

最高ρ及前3高相关的均值均只针对17只现有风险ETF（含红利），国债仍在图内展示但不参与该排序；不能因为多数股票与国债相关较低，就把存在高度行业重叠的ETF误判为有分散价值。主排序取长期、最近一年、下跌日三个最高ρ中的最大值；没有把负相关取绝对值后当作重复多头风险。强负相关仍是紧密关系，只是潜在用途更接近对冲。

本轮把红利卫星512890也作为现有成员：电力与原核心ETF的最高ρ约0.496，但与红利低波约0.581，后者才是完整现有池中的最高值。因此和上轮只看核心池的数值存在差别。其他微小差异来自先计算完整日收益再截取2022区间，保留首个交易日相对上一年末的收益。

## 最近一年与下跌环境

![近期与下跌日](assets/pool_correlation_regimes_20261109.png)

{stress}

下跌日定义为沪深300当天收益<0，大跌日为≤−1%；条件只用于事后风险诊断，不是提前预测市场状态。条件相关衡量在该类日期中两种收益变化是否同向，不能据较低数值断言候选在这些日期不亏损，也不是对组合最大回撤的保证。滚动126日最高相关90%分位反映历史上较高的重复风险，近期126日值、完整时序及每对有效样本数量另存CSV。

## 与已有扩池回测衔接

按上述分散排序挑出的两只主层候选为{assetnames}，对应之前已经测试的`{top_pair['case']}`。原35场景中的该组合，2018现金起点、单边5bps，年化{backtest['annual']:.2%}、夏普{backtest['sharpe']:.3f}、回撤{backtest['drawdown']:.2%}；CL120默认为13.50%、1.556、9.62%。2022起点该组合为{top_pair['recent']['annual']:.2%}、{top_pair['recent']['sharpe']:.3f}、{top_pair['recent']['drawdown']:.2%}，默认19.65%、2.235、4.25%。

这里仅引用已有回测，没有根据本次全样本相关性重新构造一条回测并冒充样本外结果。选择候选参考整个历史，属于事后观察；若未来将低相关条件变成交易规则，应在每个当时的信号日，仅使用此前的滚动数据，重新核对费用、持仓挤出和IVOL比较集合变化。

电力、地产可以作为相对低相关观察方向；煤炭补充能源周期但与石油股票有关；医药的近期重复度下降但长期仍受A股市场影响。证券保险与现有券商高度相关，新增它主要是扩展金融行业内部覆盖，不能算新增一个独立风险来源。有色股票与现有有色期货不是同一产品，不过仍有显著的权益市场相关。

## 方法、样本与限制

- 输入为同日基金复权因子计算的收盘价，相关性用简单日收益，不用价格水平。
- 不向前/向后填补缺价；某日或其前一日无报价，则该日收益缺失。每对资产仅使用共同有效收益，缺少样本的相关性留空，绝不填成0。
- 长期为2022年至11月6日，至少252个共同收益；最近252个交易日至少200个共同收益；下跌日至少120个共同收益；大跌日至少60个共同收益。新ETF、停牌和短样本会使每格数量不同，已另存样本数矩阵，图中灰格表示不足。
- 本轮长期每格实际共同样本为731—1172日，最近一年所有格均为252日，后者可在完全相同日期上直接横向比较；下跌日353—596日、大跌日86—154日，短条件样本须谨慎解释。
- 所有信号和收益都来自境内ETF价格，没有拼接在A股收盘后才形成的同日境外指数收盘。
- 与CL120的相关系数使用同区间的历史假设净收益，不是个人账户记录。它只反映历史动态持仓；和配置池的最高相关性用途不同。
- 当前规模筛选仍有后视与存续偏差；样本已经观察，相关性随市场状态变化。低ρ可帮助分散，最终组合风险仍取决于权重、波动率和协方差，不能替代收益与风险预算验证。

{len(checks)}项分析核对通过，包括矩阵与逐对Pearson对账、样本数、已有回测引用、输入快照及默认文件指纹。没有启用相关性交易门槛或修改CL120。

## 复现与输出

```powershell
python analyze_pool_correlations.py
```

依赖刚才扩池研究的独立行情缓存与主结果；无需联网或Token。输出在`output/pool_correlations_20261109/`：四组相关性及样本数矩阵、排序、126日滚动最高相关、分散优先两只的已有回测引用、检查及输入指纹。两张图保存在`docs/assets/`。

组合分散思想的原始文献参考：[Markowitz（1952），Portfolio Selection](https://doi.org/10.1111/j.1540-6261.1952.tb01525.x)。本轮数值是本地ETF行情计算结果，论文没有验证本项目候选或CL120。
'''
    (ROOT/'docs/pool_correlations_20261109.md').write_text(report,encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);before=protected()
    prices,spec=read_data();candidates=spec['main']+spec['observation']
    assert set(EXISTING)==set(CORE_NAMES)|{SAT}
    input_hash=hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest()
    old=json.loads((STUDY_OUT/'manifest.json').read_text(encoding='utf-8'))
    assert input_hash==old['price_hash'],'Underlying research price snapshot changed'
    returns=daily_returns(prices)
    matrices={};counts={};windows=window_definitions(returns);checks=[];dates={}
    for key,(window,minimum) in windows.items():
        rho,n=correlation_matrix(window,candidates,EXISTING,minimum)
        matrices[key]=rho;counts[key]=n;dates[key]=[str(window.index[0].date()),str(window.index[-1].date())]
        rho.to_csv(OUT/f'correlations_{key}.csv',encoding='utf-8-sig');n.to_csv(OUT/f'pair_counts_{key}.csv',encoding='utf-8-sig')
        for candidate in candidates:
            for existing in EXISTING:
                pair=window[[candidate,existing]].dropna()
                assert len(pair)==n.at[candidate,existing]
                expected=pair[candidate].corr(pair[existing]) if len(pair)>=minimum else float('nan')
                observed=rho.at[candidate,existing]
                assert (pd.isna(expected) and pd.isna(observed)) or abs(expected-observed)<1e-10
        checks.append({'check':key+'_all_pair_values_and_counts','passed':True,'cells':rho.size,
                       'minimum_pairs':int(n.min().min()),'maximum_pairs':int(n.max().max()),'selected_dates':len(window)})
    summary=priority_table(matrices,candidates,RISKY)
    rolling={}
    strategy=pd.read_csv(STUDY_OUT/'daily_2022.csv',index_col=0,parse_dates=True)['default']
    for s in candidates:
        pairwise=pd.concat({other:returns[s].rolling(126,min_periods=100).corr(returns[other]) for other in RISKY},axis=1)
        values=pairwise.replace([np.inf,-np.inf],np.nan).max(axis=1).loc['2022':]
        rolling[s]=values
        valid=values.dropna();idx=summary.index[summary.symbol.eq(s)][0]
        summary.loc[idx,'rolling_max_p90']=float(valid.quantile(.9))
        summary.loc[idx,'latest_rolling_max']=float(valid.iloc[-1])
        summary.loc[idx,'tier']='主层' if s in spec['main'] else '观察'
        summary.loc[idx,'correlation_to_CL120']=returns.loc[strategy.index,s].corr(strategy)
        peer=summary.loc[idx,'crash_peer']
        summary.loc[idx,'crash_pairs_for_max']=int(counts['crash'].at[s,peer]) if isinstance(peer,str) else 0
    pd.DataFrame(rolling).to_csv(OUT/'rolling_126_max_correlation.csv',encoding='utf-8-sig')
    summary.to_csv(OUT/'candidate_priority.csv',index=False,encoding='utf-8-sig')
    selected=summary.loc[summary.symbol.isin(spec['main'])].head(2).symbol.tolist()
    entry=next(c for c in experiment_cases(spec) if set(c.added)==set(selected) and not c.removed)
    old_results=pd.read_csv(STUDY_OUT/'all_results.csv')
    selected_results=old_results.loc[old_results.case.eq(entry.name)]
    pair={'symbols':selected,'case':entry.name,
          'backtest':selected_results.loc[selected_results.start.eq('2018-01-01')&selected_results.cost.eq(5)].iloc[0][['annual','sharpe','drawdown']].to_dict(),
          'recent':selected_results.loc[selected_results.start.eq('2022-01-01')&selected_results.cost.eq(5)].iloc[0][['annual','sharpe','drawdown']].to_dict()}
    json_file(OUT/'lowest_overlap_pair_existing_results.json',pair)
    checks.append({'check':'two_low_overlap_names_reference_existing_trial_not_new_backtest','passed':True})
    figures(matrices,counts,summary.symbol.tolist(),dates)
    checks.append({'check':'production_and_live_files_unchanged','passed':before==protected()})
    render_report(summary,counts,checks,pair,dates)
    assert before==protected()
    assert all(c['passed'] for c in checks)
    json_file(OUT/'checks.json',checks)
    json_file(OUT/'manifest.json',{'end':str(END),'candidates':candidates,'existing':EXISTING,
             'price_hash':input_hash,'default_changed':False,'protected_before':before,'protected_after':protected(),
             'all_checks_passed':True,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'ranking':'max of long/recent/down maximum positive-direction correlation to all existing risky ETFs incl dividend',
             'windows':{k:{'dates':len(w),'min_pairs':n} for k,(w,n) in windows.items()}})
    print(summary[['symbol','long_max','long_peer','recent_max','down_max','worst_of_three','rolling_max_p90']].to_string(index=False))
    print(f'PASS: {len(checks)} grouped checks; correlation heatmaps and report saved.')

if __name__=='__main__':main()
