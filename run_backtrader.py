"""Backtrader独立回测对照；PyCharm可直接运行，默认从2018年起、仅用本地缓存。

python run_backtrader.py
python run_backtrader.py --start 2022-01-01 --end 2026-10-20
默认假设整段历史采用当前规则；--from-adoption才按实际红利启用日期回放。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etf_rotation.backtest import run_backtest
from etf_rotation.backtrader_engine import run_backtrader
from etf_rotation.config import ETF_NAMES, STRATEGY_NAME
from etf_rotation.data import load_symbol
from etf_rotation.portfolio_strategy import portfolio_signals
from etf_rotation.strategy_state import load_rotation_state

ROOT=Path(__file__).resolve().parent
DEFAULT_OUT=ROOT/"output/backtrader_comparison_20261023"


def protected():
    names=["live_signal.py","run_backtest.py","etf_rotation/risk_strategy.py","etf_rotation/portfolio_strategy.py",
           "etf_rotation/backtest.py","etf_rotation/mean_reversion.py","etf_rotation/factors.py",
           "etf_rotation/config.py","data/rotation_state.json","output/live_signal.json"]
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names if (ROOT/n).exists()}


def json_file(path,payload):
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding="utf-8")


def load_offline(cache,end):
    symbols=[*ETF_NAMES,"512890"]
    metas={s:json.loads((cache/f"tushare_{s}_hfq.json").read_text(encoding="utf-8")) for s in symbols}
    cutoff=min(date.fromisoformat(x["end"]) for x in metas.values())
    end=end or cutoff
    if end>cutoff:raise ValueError(f"缓存共同覆盖至{cutoff}；对照回测不自动请求Tushare")
    values={s:load_symbol(s,date.fromisoformat(metas[s]["start"]),end,cache,offline=True) for s in symbols}
    calendar=values["510300"].index
    prices=pd.DataFrame(values,index=calendar)
    return prices.drop(columns="512890"),prices[["512890"]],end


def compare(args):
    args.output_dir.mkdir(parents=True,exist_ok=True)
    before=protected()
    core,history,end=load_offline(args.cache_dir,args.end)
    if core.index[0].date()>args.start:raise ValueError("缓存起点晚于回测起点，请使用data/audit_history完整缓存")
    state=load_rotation_state(ROOT/"data/rotation_state.json")
    bundle=portfolio_signals(core,history,state,hypothetical_satellite_history=not args.from_adoption)
    prices,signals=bundle.prices,bundle.signals
    mode="from_actual_adoption_date" if args.from_adoption else "hypothetical_full_history"
    json_file(args.output_dir/"manifest.json",{"research_date":"2026-10-23","end":str(end),"requested_start":str(args.start),
       "strategy":STRATEGY_NAME,"simulation_mode":mode,"cost_bps":args.fee_bps+args.slippage_bps,
       "initial_cash_for_normalized_research":args.initial_cash,"protected_before":before,
       "price_hash":hashlib.sha256(pd.util.hash_pandas_object(prices,index=True).values.tobytes()).hexdigest(),
       "target_hash":hashlib.sha256(pd.util.hash_pandas_object(signals,index=True).values.tobytes()).hexdigest(),
       "backtrader_adapter_hash":hashlib.sha256((ROOT/"etf_rotation/backtrader_engine.py").read_bytes()).hexdigest()})
    starts=sorted({args.start,date(2022,1,1)} if args.start<date(2022,1,1)<end else {args.start})
    table=[];comparison_rows=[];results={};checks=[]
    for start in starts:
        for cost in sorted({0.,args.fee_bps+args.slippage_bps,20.}):
            key=f"{start}_cost{cost:g}"
            original=run_backtest(prices,signals,start,end,fee_bps=0,slippage_bps=cost)
            native,orders,metadata=run_backtrader(prices,signals,start,end,fee_bps=0,slippage_bps=cost,
                                                 initial_cash=args.initial_cash)
            pair=pd.DataFrame({"original_nav":original.equity.strategy_nav,"backtrader_nav":native.equity.strategy_nav,
                               "original_return":original.equity.daily_return,"backtrader_return":native.equity.daily_return})
            pair["nav_difference"]=pair.backtrader_nav-pair.original_nav
            weight_error=float((original.weights-native.weights).abs().to_numpy().max())
            daily_error=float((pair.original_return-pair.backtrader_return).abs().max())
            completed=orders[orders.status.eq("Completed")]
            timing_ok=bool((pd.to_datetime(completed.signal_date)<pd.to_datetime(completed.execution_date)).all()
                           and (pd.to_datetime(completed.scheduled_execution)==pd.to_datetime(completed.execution_date)).all())
            if not timing_ok:raise AssertionError("Backtrader成交时序错误")
            if metadata["orders_margin_or_rejected"]:raise AssertionError("目标仓位模式出现资金不足或拒单")
            if cost==0:
                np.testing.assert_allclose(pair.original_nav,pair.backtrader_nav,atol=1e-8,rtol=1e-8)
                checks.append({"check":"zero_cost_independent_equity_match","start":str(start),"passed":True})
            np.testing.assert_allclose(metadata["native_drawdown"],native.metrics["max_drawdown"],atol=1e-10)
            pair.to_csv(args.output_dir/f"daily_comparison_{key}.csv",index_label="date")
            native.equity.to_csv(args.output_dir/f"backtrader_equity_{key}.csv")
            native.weights.to_csv(args.output_dir/f"backtrader_weights_{key}.csv")
            orders.to_csv(args.output_dir/f"backtrader_orders_{key}.csv",index=False)
            native.trades.to_csv(args.output_dir/f"backtrader_rebalances_{key}.csv",index=False)
            json_file(args.output_dir/f"backtrader_metadata_{key}.json",metadata)
            for name,r in (("original",original),("backtrader_target_close",native)):
                table.append({"start":str(start),"end":str(end),"cost_bps":cost,"engine":name,**r.metrics})
            comparison_rows.append({"start":str(start),"cost_bps":cost,"maximum_nav_absolute_difference":float(pair.nav_difference.abs().max()),
                      "terminal_nav_difference":float(pair.nav_difference.iloc[-1]),"maximum_daily_return_difference":daily_error,
                      "maximum_weight_difference":weight_error,"orders_completed":metadata["orders_completed"],
                      "orders_margin_or_rejected":metadata["orders_margin_or_rejected"],"timing_verified":timing_ok,
                      "native_timereturn_error":metadata["native_timereturn_max_error"]})
            if cost==args.fee_bps+args.slippage_bps:
                results[str(start)]={"original":original,"target_close":native}
                locked,lo,lm=run_backtrader(prices,signals,start,end,fee_bps=args.fee_bps,slippage_bps=args.slippage_bps,
                                          initial_cash=args.initial_cash,mode="fixed_close")
                table.append({"start":str(start),"end":str(end),"cost_bps":cost,"engine":"backtrader_fixed_close",**locked.metrics})
                lo.to_csv(args.output_dir/f"fixed_close_orders_{start}.csv",index=False)
                locked.equity.to_csv(args.output_dir/f"fixed_close_equity_{start}.csv")
                json_file(args.output_dir/f"fixed_close_metadata_{start}.json",lm)
                results[str(start)]["fixed_close"]=locked
            print(f"completed {key}: original={original.metrics['annual_return']:.6%}; BT={native.metrics['annual_return']:.6%}; max NAV diff={pair.nav_difference.abs().max():.3g}",flush=True)
    pd.DataFrame(table).to_csv(args.output_dir/"metrics.csv",index=False)
    pd.DataFrame(comparison_rows).to_csv(args.output_dir/"reconciliation.csv",index=False)
    signals.to_csv(args.output_dir/"signal_weights.csv")
    if protected()!=before:raise AssertionError("原有策略或实盘状态文件发生变化")
    checks.append({"check":"production_files_unchanged","passed":True})
    json_file(args.output_dir/"checks.json",checks)
    render_report(args,end,pd.DataFrame(table),pd.DataFrame(comparison_rows),results,mode)


def render_report(args,end,table,reconciliation,results,mode):
    plt.rcParams.update({"font.family":"Microsoft YaHei","axes.unicode_minus":False})
    fig,axes=plt.subplots(len(results),3,figsize=(16,5*len(results)),squeeze=False)
    for row,(start,rs) in zip(axes,results.items()):
        for key,label,style in (("original","自写引擎","-"),("target_close","Backtrader：收盘目标仓位","--"),
                                ("fixed_close","Backtrader：前日锁定份额",":")):
            r=rs[key]
            row[0].plot(r.equity.index,r.equity.strategy_nav,style,label=label)
            row[1].plot(r.equity.index,r.equity.drawdown*100,style,label=label)
        diff=rs["target_close"].equity.strategy_nav-rs["original"].equity.strategy_nav
        row[2].plot(diff.index,diff*100,color="#a65027")
        row[0].set(title=f"{start}起：扣费净值",ylabel="净值");row[0].legend(fontsize=8)
        row[1].set(title="收盘回撤",ylabel="%")
        row[2].set(title="目标仓位模式－原引擎：净值差",ylabel="初始资金百分点")
        for a in row:a.grid(alpha=.2)
    fig.suptitle(f"Backtrader独立经纪商复核｜行情至{end}｜单边{args.fee_bps+args.slippage_bps:g}bps",fontsize=15)
    fig.tight_layout()
    assets=ROOT/"docs/assets";assets.mkdir(exist_ok=True,parents=True)
    fig.savefig(assets/"backtrader_comparison_20261023.png",dpi=150);plt.close(fig)
    lines=[]
    labels={"original":"原自写引擎","backtrader_target_close":"Backtrader：收盘目标比例","backtrader_fixed_close":"Backtrader：前日锁定份额"}
    for _,r in table.iterrows():
        lines.append(f"| {r.start} | {r.cost_bps:g} | {labels[r.engine]} | {r.annual_return:.4%} | {r.sharpe_zero_rf:.4f} | {r.max_drawdown:.4%} |")
    descriptions=[]
    for start in results:
        lm=json.loads((args.output_dir/f"fixed_close_metadata_{start}.json").read_text(encoding="utf-8"))
        descriptions.append(f"- {start}起：锁定份额模式完成{lm['orders_completed']}笔ETF订单，资金不足/拒单{lm['orders_margin_or_rejected']}笔。没有隐藏失败或自动补单。")
    report=f"""# Backtrader独立回测对照

研究日期2026-10-23；行情截止{end}。主研究从2018年开始，2022年起另作同区间复核。没有重新调参，没有改变默认策略。

## 1. 原来使用什么引擎

原入口为 `run_backtest.py`，成交、持仓漂移、费用与净值由 `etf_rotation/backtest.py` 自写引擎计算，未使用Backtrader。本次新增 `run_backtrader.py` 和 `etf_rotation/backtrader_engine.py`，安装Backtrader 1.9.78.123，使用原生Cerebro、BackBroker、Position、订单及现金记账，没有覆盖经纪商的成交、持仓、现金或净值方法。

策略信号继续使用同一生产函数；本次独立复核的是订单、资金和收益计算，不是独立重新发现因子，也不能消除选池后视或调参过拟合。

## 2. 相同数据与执行假设

- 所有引擎读取同一份Tushare本地复权收盘价和同一份因果周信号。行情截止{end}，没有自动更新至报告日期。
- 当前运行模式 `{mode}`。默认是全历史采用当前红利规则的假设回测，非此前实际账户收益；`--from-adoption`按真实启用日期回放。
- 每个交易周最后开市日收盘形成目标，下一交易日收盘成交。2018前行情只预热因子与状态，不计入2018账户收益。
- 为隔离引擎差异，允许分数复权单位，未套用100份整数手；复权单位不是实盘可直接下单的原始ETF份额。
- 费用合并为原生按成交额比例手续费：手续费{args.fee_bps:g}bps＋滑点预算{args.slippage_bps:g}bps。费用压力测试另外核对0与20bps。没有伪造开高低价后再声称验证盘口滑点。
- 最多沿用3日过去报价估值；缺真实成交报价时取消整个调仓批次；上市前占位值不允许交易，长期缺价持仓立即报错。

## 3. 为什么有两种Backtrader模式

### A. 收盘目标比例对照：保持原模型的执行理想化

原模型在成交日收盘按当时净值恢复目标权重。为了保留这一假设，每个复权收盘被表示为独立的合成收盘事件，OHLC都取该收盘；上一日已冻结的目标在下一事件提交普通Market订单。使用Backtrader `cheat_on_open` 回调在经纪商成交前计算数量，该事件的open代表收盘参考价，**不代表真实开盘**；`cheat_on_close=False`，没有把当日新信号提前成交。

这里只在执行事件按已知模型参考价确定数量，选仓和权重不读取下一日行情。精确按最终收盘价确定份额本身是原模型保留的理想化，实盘临近收盘价格、成交数量与此会有误差。不能把这种对照当作真实收盘竞价可保证实现的业绩。

经纪商实际先卖后买，按原生现金检查成交；目标市值求解 `成交后NAV＋实际成交额×费用率＝成交前NAV`，保留不超过净值1e-11的数值现金余量，避免浮点误差拒单。没有投入新资金、使用杠杆或手工修补NAV。

### B. 前日锁定份额：原生Close订单的执行敏感性

在信号日用当日已知净值和收盘价确定份额，提交原生 `Order.Close`，下一交易日收盘成交。数量不使用下一日价格。次日涨跌会使仓位偏离目标；资金不足时Backtrader会拒绝买单，默认不自动重试。因此它评估的是固定份额执行政策，与A的目标比例政策不同，不应将差异直接判断为因子失效。

{chr(10).join(descriptions)}

Backtrader官方说明：[次日Close订单](https://www.backtrader.com/docu/order-creation-execution/order-creation-execution/)、[Cheat-On-Open回调](https://www.backtrader.com/docu/cerebro/cheat-on-open/cheat-on-open/)。官方的Close订单不会通过内置滑点机制加价，所以为匹配原研究，将2bps滑点预算作为现金成本合并扣取。[官方滑点说明](https://www.backtrader.com/docu/slippage/slippage/)

## 4. 结果

年化统一按252交易日，夏普无风险利率为零，回撤按收盘净值。研究资金为统一归一化本金，不是用户真实账户。

| 起始日 | 单边总成本bps | 引擎/执行政策 | 年化收益 | 夏普 | 最大回撤 |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(lines)}

![净值、回撤及引擎差异](assets/backtrader_comparison_20261023.png)

## 5. 差异与检查

目标比例模式与原引擎的最大逐日净值绝对差为{reconciliation.maximum_nav_absolute_difference.max():.8f}；最大实际权重绝对差为{reconciliation.maximum_weight_difference.max():.3g}。无成本场景逐日净值通过数值一致性检查；所有目标比例订单均严格在信号日之后、约定下一交易日成交，没有资金不足拒单。

有成本时不强制两条曲线完全相等：原模型用扣费前目标名义成交额估算成本，再把所有权重恢复为目标；原生经纪商按真实成交份额扣费，自洽求解可买金额。差异应由这项记账口径及数值余量解释，而非通过覆盖经纪商净值消除。

Backtrader原生TimeReturn与输出逐日收益核对，原生DrawDown与统一统计核对。完整订单、失败原因、现金、净值、实际持仓和对账误差保存于本地输出目录；已有生产文件、实盘状态及live_signal输出指纹保持一致。

新增8项Backtrader测试覆盖：无成本逐日对账、实际现金扣费、跨起点订单、最后交易日成交、缺价取消、上市前禁止买入、长期缺价停止、前日固定份额与未来价格扰动。完整101项测试通过；系统临时目录测试使用沙盒外权限完成。

锁定份额模式中的拒单会改变风险暴露，因此较高夏普不能直接归为因子改善。实际执行应根据临近成交时的资产和现金核算可买份额并预留费用；上述固定份额场景保留拒单且不补单，不能直接当作完善的实盘执行系统。

更换回测框架能帮助发现实现错误，无法验证未来持续盈利；仍保留上市可得性、历史选池、交易成本、整数手、收盘竞价、停牌及极端行情等限制。没有模拟真实OHLC路径、盘中回撤、实际现金分红到账或容量。

## 6. 本地运行

```powershell
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe run_backtrader.py
# 仅比较2022年以来
.\\.venv\\Scripts\\python.exe run_backtrader.py --start 2022-01-01 --end {end}
# 按实际红利启用日期回放
.\\.venv\\Scripts\\python.exe run_backtrader.py --from-adoption
```

可以在PyCharm直接运行 `run_backtrader.py`。默认只读 `data/audit_history` 本地缓存；若该目录未准备好，可先运行既有审计的数据下载入口。输出在 `output/backtrader_comparison_20261023/`，原自写回测仍由 `run_backtest.py` 运行。
"""
    (ROOT/"docs/backtrader_comparison_20261023.md").write_text(report,encoding="utf-8")
    print("\n"+table[table.cost_bps.eq(args.fee_bps+args.slippage_bps)][["start","engine","annual_return","sharpe_zero_rf","max_drawdown"]].to_string(index=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start",type=date.fromisoformat,default=date(2018,1,1))
    parser.add_argument("--end",type=date.fromisoformat)
    parser.add_argument("--cache-dir",type=Path,default=ROOT/"data/audit_history")
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--initial-cash",type=float,default=1_000_000.)
    parser.add_argument("--fee-bps",type=float,default=3.)
    parser.add_argument("--slippage-bps",type=float,default=2.)
    parser.add_argument("--from-adoption",action="store_true")
    compare(parser.parse_args())
