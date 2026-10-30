"""使用原生Backtrader经纪商验证收盘再平衡，不覆盖生产回测引擎。

target_close：前日冻结目标，次日收盘事件按成交参考价确定份额。
fixed_close：前日收盘即锁定份额，使用原生Close订单在次日收盘成交。
只提供收盘数据，不能用于验证真实开盘成交、盘口或整手资金约束。
"""
from __future__ import annotations

from datetime import date, time

import backtrader as bt
import numpy as np
import pandas as pd

from .backtest import BacktestResult, _metrics
from .config import BENCHMARK
from .factors import validate_prices


class CloseEventFeed(bt.feeds.PandasData):
    """各日OHLC同为复权收盘，仅用于收盘事件；实盘原始报价有效性另存。"""
    lines=("real_quote","valid_valuation")
    params=(("real_quote","real_quote"),("valid_valuation","valid_valuation"))


def funded_target_values(nav, held_values, weights, rate):
    """求V + 成本率*sum(abs(V*w-原持仓市值)) = 成交前NAV。"""
    if nav<=0 or not 0<=rate<.1:
        raise ValueError("NAV或交易成本无效")
    w=np.asarray(weights,float)
    h=np.asarray(held_values,float)
    if not np.isfinite(w).all() or (w<0).any() or w.sum()>1+1e-8:
        raise ValueError("目标权重必须有限、非负且总和不超过1")
    # 避免满仓时浮点舍入造成经纪商资金不足；最多保留NAV的1e-11。
    budget=nav*(1-1e-11)
    lo,hi=0.,budget
    for _ in range(60):
        mid=(lo+hi)/2
        if mid+rate*np.abs(mid*w-h).sum()>budget:hi=mid
        else:lo=mid
    return ((lo+hi)/2)*w


class RotationReplay(bt.Strategy):
    params=(("targets",None),("evaluation_start",None),("evaluation_end",None),
            ("cost_rate",.0005),("initial_cash",1_000_000.),("mode","target_close"))

    def __init__(self):
        self.schedule={pd.Timestamp(k).normalize():v for k,v in self.p.targets.iterrows()}
        self.days=pd.DatetimeIndex(self.p.targets.attrs["market_calendar"])
        self.eval_start=pd.Timestamp(self.p.evaluation_start)
        self.eval_end=pd.Timestamp(self.p.evaluation_end)
        self.pending=None
        self.open_orders=[]
        self.equity_rows=[];self.weight_rows=[];self.order_rows=[];self.rebalance_rows=[]
        self.current_fee=0.;self.current_notional=0.;self.current_order_count=0
        self.previous_nav=float(self.p.initial_cash)
        self.pretrade_nav=None

    def _day(self):
        return pd.Timestamp(self.datas[0].datetime.date(0))

    def _positions(self):
        return np.array([self.getposition(d).size for d in self.datas],dtype=float)

    def _validate_valuation(self):
        for d in self.datas:
            if abs(self.getposition(d).size)>1e-12 and not d.valid_valuation[0]:
                raise ValueError(f"持仓{d._name}在{self._day().date()}缺价超过估值容忍期")

    def next_open(self):
        self.current_fee=0.;self.current_notional=0.;self.current_order_count=0
        self._validate_valuation()
        marks=np.array([d.open[0] for d in self.datas])
        self.pretrade_nav=float(self.broker.getcash()+self._positions()@marks)
        if self.pending is None:return
        signal_day,execution_day,target=self.pending
        if self._day()!=execution_day:return
        if signal_day>=execution_day:raise AssertionError("信号日不得等于或晚于成交日")
        weights=target.to_numpy(float)
        held_values=self._positions()*marks
        current=held_values/max(self.pretrade_nav,1e-12)
        changed=np.abs(weights-current)>1e-10
        missing=[d._name for i,d in enumerate(self.datas) if changed[i] and not d.real_quote[0]]
        if self.p.mode=="fixed_close":
            missing=sorted({o.data._name for o in self.open_orders if o.alive() and not o.data.real_quote[0]})
        if missing:
            for order in self.open_orders:
                if order.alive():self.cancel(order)
            self.rebalance_rows.append({"signal_date":signal_day,"execution_date":execution_day,
                 "status":"skipped_no_quote","unavailable":",".join(missing)})
        else:
            self.rebalance_rows.append({"signal_date":signal_day,"execution_date":execution_day,
                 "status":"submitted","unavailable":""})
            if self.p.mode=="target_close":
                self._send_orders(weights,marks,self.pretrade_nav,signal_day,execution_day,bt.Order.Market)
        self.pending=None

    def prenext_open(self):self.next_open()
    def nextstart_open(self):self.next_open()

    def _send_orders(self,weights,marks,nav,signal_day,execution_day,order_type):
        sizes=self._positions()
        values=funded_target_values(nav,sizes*marks,weights,self.p.cost_rate)
        differences=values/marks-sizes
        # 由同一经纪商先处理卖单释放资金，再处理买单。
        for i in np.argsort(differences,kind="stable"):
            delta=float(differences[i])
            if abs(delta*marks[i])<nav*1e-12:continue
            method=self.buy if delta>0 else self.sell
            order=method(data=self.datas[i],size=abs(delta),exectype=order_type)
            order.addinfo(signal_date=str(signal_day.date()),scheduled_execution=str(execution_day.date()))
            self.open_orders.append(order)

    def notify_order(self,order):
        if order.status in (order.Submitted,order.Accepted):return
        execution=None
        if order.executed.dt:
            execution=pd.Timestamp(bt.num2date(order.executed.dt)).tz_localize(None).normalize()
        completed=order.status==order.Completed
        if completed:
            notional=abs(float(order.executed.size*order.executed.price))
            self.current_fee+=float(order.executed.comm)
            self.current_notional+=notional
            self.current_order_count+=1
        self.order_rows.append({"reference":order.ref,"symbol":order.data._name,
              "signal_date":order.info.get("signal_date"),"scheduled_execution":order.info.get("scheduled_execution"),
              "execution_date":execution,"notification_date":self._day(),"status":order.getstatusname(),
              "size":float(order.executed.size),"price":float(order.executed.price),
              "commission":float(order.executed.comm),"is_buy":order.isbuy()})

    def next(self):
        day=self._day()
        if self.eval_start<=day<=self.eval_end:
            nav=float(self.broker.getvalue())
            if not np.isfinite(nav) or nav<=0:raise ValueError("Backtrader NAV无效")
            marks=np.array([d.close[0] for d in self.datas])
            held=self._positions()*marks/nav
            cash=float(self.broker.getcash())
            if cash < -self.p.initial_cash*1e-9:raise AssertionError("出现透支现金")
            self.equity_rows.append({"date":day,"strategy_nav":nav/self.p.initial_cash,
               "daily_return":nav/self.previous_nav-1,"cash":cash,"value":nav,
               "commission":self.current_fee,"traded_notional":self.current_notional,
               "turnover":self.current_notional/max(self.pretrade_nav or nav,1e-12),
               "cost_fraction":self.current_fee/max(self.pretrade_nav or nav,1e-12)})
            self.weight_rows.append({"date":day,**{d._name:float(held[i]) for i,d in enumerate(self.datas)},"CASH":cash/nav})
            self.previous_nav=nav
        if day in self.schedule:
            j=int(self.days.searchsorted(day,side="right"))
            if j<len(self.days) and self.eval_start<=self.days[j]<=self.eval_end:
                target=self.schedule[day]
                self.pending=(day,self.days[j],target)
                if self.p.mode=="fixed_close":
                    marks=np.array([d.close[0] for d in self.datas])
                    nav=float(self.broker.getvalue())
                    self._send_orders(target.to_numpy(float),marks,nav,day,self.days[j],bt.Order.Close)

    def prenext(self):self.next()
    def nextstart(self):self.next()


def run_backtrader(prices:pd.DataFrame,signals:pd.DataFrame,start:date,end:date,*,
                   fee_bps=3.,slippage_bps=2.,initial_cash=1_000_000.,
                   mode="target_close",max_stale_days=3):
    """返回标准化逐日结果、原生订单日志及经纪商元数据。

target_close保留原模型的成交时点目标比例理想假设，不代表能预知最终收盘价。
fixed_close锁定信号日确定的份额；次日涨跌可导致资金不足拒单，完整保留日志。
"""
    validate_prices(prices)
    if mode not in {"target_close","fixed_close"}:raise ValueError("未知Backtrader执行模式")
    if fee_bps<0 or slippage_bps<0 or initial_cash<=0 or max_stale_days<0:
        raise ValueError("费用、资本或缺价参数无效")
    if not isinstance(signals.index,pd.DatetimeIndex) or signals.index.has_duplicates or not signals.index.isin(prices.index).all():
        raise ValueError("信号必须来自价格矩阵的唯一交易日")
    if len(signals.columns.difference(prices.columns)):raise ValueError("信号包含未知ETF")
    targets=signals.reindex(columns=prices.columns,fill_value=0.).copy()
    values=targets.to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values.sum(axis=1)>1+1e-8).any():
        raise ValueError("目标权重无效")
    dates=prices.index[(prices.index>=pd.Timestamp(start))&(prices.index<=pd.Timestamp(end))]
    if len(dates)<2:raise ValueError("回测区间至少需要两个交易日")
    if BENCHMARK not in prices or pd.isna(prices.at[dates[0],BENCHMARK]):raise ValueError("基准首日无报价")
    first=max(0,int(prices.index.get_loc(dates[0]))-1)
    # 只多提供一个前置交易日，供跨回测起点的真实次日订单排队；不持有预热收益。
    frame=prices.iloc[first:prices.index.get_loc(dates[-1])+1]
    valuation=prices.ffill(limit=max_stale_days) if max_stale_days else prices
    targets.attrs["market_calendar"]=prices.index
    cerebro=bt.Cerebro(cheat_on_open=True,stdstats=False)
    cerebro.broker.setcash(initial_cash)
    # 原始模型把手续费和滑点都按成交额扣现金。合并为原生手续费，避免Close单忽略滑点。
    rate=(fee_bps+slippage_bps)/10000
    if rate>=.1:raise ValueError("交易成本过高")
    cerebro.broker.setcommission(commission=rate,commtype=bt.CommInfoBase.COMM_PERC,
                                stocklike=True,percabs=True,leverage=1.)
    cerebro.broker.set_coc(False)
    # 立即接受待执行订单，次日可原生取消缺价批次；实际成交仍由经纪商检查现金。
    cerebro.broker.set_checksubmit(False)
    for symbol in prices:
        raw=frame[symbol]
        bounded=valuation[symbol].reindex(frame.index)
        # 占位值仅使零持仓估值有限；真实报价/估值有效性标记禁止占位成交或无限停牌估值。
        mark=prices[symbol].ffill().reindex(frame.index).fillna(1.)
        data=pd.DataFrame({k:mark for k in ("open","high","low","close")})
        data["volume"]=0.;data["openinterest"]=0.
        data["real_quote"]=raw.notna().astype(float)
        data["valid_valuation"]=bounded.notna().astype(float)
        data.index=data.index+pd.Timedelta(hours=15)
        feed=CloseEventFeed(dataname=data,timeframe=bt.TimeFrame.Days,sessionend=time(15,0))
        cerebro.adddata(feed,name=symbol)
    cerebro.addstrategy(RotationReplay,targets=targets,evaluation_start=dates[0],evaluation_end=dates[-1],
                        cost_rate=rate,initial_cash=initial_cash,mode=mode)
    cerebro.addanalyzer(bt.analyzers.TimeReturn,_name="native_returns",timeframe=bt.TimeFrame.Days,fund=False)
    cerebro.addanalyzer(bt.analyzers.DrawDown,_name="native_drawdown",fund=False)
    strategy=cerebro.run(runonce=False,preload=True)[0]
    equity=pd.DataFrame(strategy.equity_rows).set_index("date")
    equity["benchmark_nav"]=prices.loc[equity.index,BENCHMARK]/prices.at[dates[0],BENCHMARK]
    equity["drawdown"]=equity.strategy_nav/equity.strategy_nav.cummax().clip(lower=1)-1
    weights=pd.DataFrame(strategy.weight_rows).set_index("date")
    orders=pd.DataFrame(strategy.order_rows,columns=["reference","symbol","signal_date","scheduled_execution",
               "execution_date","notification_date","status","size","price","commission","is_buy"])
    trades=pd.DataFrame(strategy.rebalance_rows,columns=["signal_date","execution_date","status","unavailable"])
    trades["turnover"]=[float(equity.at[d,"turnover"]) for d in trades.execution_date]
    metrics=_metrics(equity,trades)
    native_returns=pd.Series(strategy.analyzers.native_returns.get_analysis(),dtype=float)
    native_returns.index=pd.to_datetime(native_returns.index).normalize()
    native_error=float((native_returns.reindex(equity.index)-equity.daily_return).abs().max())
    if native_error>1e-10:raise AssertionError("Backtrader原生TimeReturn与输出净值不一致")
    metadata={"engine":"backtrader","version":bt.__version__,"mode":mode,"initial_cash":initial_cash,
              "combined_cost_as_native_commission":rate,"synthetic_close_events":True,
              "orders_completed":int(orders.status.eq("Completed").sum()),
              "orders_margin_or_rejected":int(orders.status.isin(["Margin","Rejected"]).sum()),
              "native_broker_no_position_or_nav_overrides":True,"zero_risk_free_rate":True,
              "fractional_adjusted_price_units":True,"cheat_on_close":False,
              "broker_actual_fill_cash_checks_enabled":True,
              "native_timereturn_max_error":native_error,
              "native_drawdown":-float(strategy.analyzers.native_drawdown.get_analysis().max.drawdown)/100,
              "cheat_on_open_only_for_previous_signal_close_event":True}
    return BacktestResult(equity,weights,trades,metrics),orders,metadata
