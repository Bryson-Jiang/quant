"""账户收益、完整持仓回合及换手指标；只统计，不参与选仓。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def return_metrics(returns: pd.Series, *, annual_days: int = 252) -> dict:
    r=returns.astype(float)
    if (len(r)<2 or not isinstance(r.index,pd.DatetimeIndex) or r.index.has_duplicates
            or not r.index.is_monotonic_increasing or not np.isfinite(r).all() or r.le(-1).any()):
        raise ValueError('收益序列必须按唯一日期排序、有限且大于-100%')
    nav=(1+r).cumprod();dd=nav/nav.cummax().clip(lower=1)-1
    vol=float(r.std(ddof=1)*np.sqrt(annual_days))
    annual=float(nav.iloc[-1]**(annual_days/len(r))-1)
    rf_daily=1.02**(1/annual_days)-1
    maxdd=float(dd.min())
    calendar_years=((r.index[-1]-r.index[0]).days+1)/365.2425
    return {'days':len(r),'total_return':float(nav.iloc[-1]-1),'annual_return':annual,
            'calendar_annual_return':float(nav.iloc[-1]**(1/calendar_years)-1),
            'annual_volatility':vol,'sharpe_zero_rf':float(r.mean()*annual_days/vol) if vol else None,
            'sharpe_rf2_illustrative':float((r-rf_daily).mean()*annual_days/vol) if vol else None,
            'max_drawdown':maxdd,'calmar':annual/abs(maxdd) if maxdd<0 else None,
            'daily_win_rate':float(r.gt(1e-12).mean())}


def outcome_metrics(pnl) -> dict:
    """平均盈利/平均亏损与总盈利/总亏损分别返回，不把它们混为一谈。"""
    values=np.asarray(pnl,dtype=float)
    if not np.isfinite(values).all():raise ValueError('盈亏含非有限值')
    wins=values[values>1e-12];losses=values[values < -1e-12]
    avg_win=float(wins.mean()) if len(wins) else None
    avg_loss=float(-losses.mean()) if len(losses) else None
    return {'count':len(values),'wins':len(wins),'losses':len(losses),
            'flats':int(len(values)-len(wins)-len(losses)),
            'win_rate':float(len(wins)/len(values)) if len(values) else None,
            'average_win':avg_win,'average_loss':avg_loss,
            'payoff_ratio':avg_win/avg_loss if avg_win is not None and avg_loss else None,
            'profit_factor':float(wins.sum()/(-losses.sum())) if len(losses) else None,
            'mean_pnl':float(values.mean()) if len(values) else None}


def rebalance_periods(returns: pd.Series, execution_dates) -> pd.DataFrame:
    """划分互不重叠的净收益区块：调仓日到下次调仓日前一日。

    每段含本次调仓成本及该日旧持仓到收盘的收益，是账户期间结果，
    不是单只ETF的买卖收益。最后未完成区块保留，但胜负统计应排除。
    """
    positions=returns.index.get_indexer(pd.DatetimeIndex(execution_dates).drop_duplicates().sort_values())
    if (positions<0).any():raise ValueError('调仓日不在账户收益日历')
    rows=[]
    for i,start in enumerate(positions):
        stop=positions[i+1] if i+1<len(positions) else len(returns)
        block=returns.iloc[start:stop]
        rows.append({'start':block.index[0],'end':block.index[-1],'days':len(block),
                     'complete':i+1<len(positions),'net_return':float((1+block).prod()-1)})
    return pd.DataFrame(rows,columns=['start','end','days','complete','net_return'])


def holding_episodes(orders: pd.DataFrame, final_prices: pd.Series,
                      *, initial_cash: float) -> pd.DataFrame:
    """原生成交明细从0建仓到清零为一回合；加减仓持续累计现金流。

    pnl按初始资金归一化，只用于统计盈亏金额；ROI另按该回合累计买入
    成本计算。期末未清仓只计算标记市值，不假造卖出费用或算入闭合胜率。
    """
    required={'execution_date','symbol','status','size','price','commission'}
    if not required.issubset(orders) or initial_cash<=0:raise ValueError('成交字段或初始资金无效')
    done=orders.loc[orders.status.eq('Completed')].copy()
    done['execution_date']=pd.to_datetime(done.execution_date)
    done=done.sort_values('execution_date',kind='stable')
    active={};records=[]
    def finish(symbol,item,day,closed):
        mark=0. if closed else item['quantity']*float(final_prices[symbol])
        profit=item['sell_net']+mark-item['buy_cost']
        return {'symbol':symbol,'entry_date':item['entry'],'last_trade_date':day,'closed':closed,
                'quantity':item['quantity'],'buy_cost':item['buy_cost']/initial_cash,
                'sell_net':item['sell_net']/initial_cash,'mark_value':mark/initial_cash,
                'net_pnl':profit/initial_cash,'roi_on_cumulative_buys':profit/item['buy_cost'],
                'fees':item['fees']/initial_cash,'orders':item['orders']}
    for row in done.itertuples():
        symbol=str(row.symbol);qty=float(row.size);price=float(row.price);fee=float(row.commission)
        if not np.isfinite([qty,price,fee]).all() or price<=0 or fee<0:
            raise ValueError('成交数值无效')
        if symbol not in active:
            if qty<=0:raise ValueError('不能在未建仓时卖出')
            active[symbol]={'quantity':0.,'peak':0.,'buy_cost':0.,'sell_net':0.,'fees':0.,
                            'entry':row.execution_date,'last':row.execution_date,'orders':0}
        item=active[symbol]
        item['last']=row.execution_date;item['orders']+=1;item['fees']+=fee
        item['quantity']+=qty;item['peak']=max(item['peak'],item['quantity'])
        if qty>0:item['buy_cost']+=qty*price+fee
        else:item['sell_net']+=-qty*price-fee
        tolerance=1e-9*max(1,item['peak'])
        if item['quantity'] < -tolerance:raise ValueError('卖出超过持仓')
        if abs(item['quantity'])<=tolerance:
            item['quantity']=0.;records.append(finish(symbol,item,row.execution_date,True));del active[symbol]
    for symbol,item in active.items():
        if symbol not in final_prices or not np.isfinite(final_prices[symbol]):raise ValueError('期末估值缺价')
        records.append(finish(symbol,item,item['last'],False))
    return pd.DataFrame(records,columns=['symbol','entry_date','last_trade_date','closed','quantity',
       'buy_cost','sell_net','mark_value','net_pnl','roi_on_cumulative_buys','fees','orders'])


def turnover_metrics(equity: pd.DataFrame) -> dict:
    rate=float(equity.turnover.sum()*252/len(equity))
    return {'annual_two_sided_turnover':rate,'annual_half_two_sided_equivalent':rate/2,
            'annual_additive_cost_fraction':float(equity.cost_fraction.sum()*252/len(equity))}
