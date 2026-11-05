"""Explainable strategy discovery with a simple fee-aware historical simulation."""
import math

STRATEGIES=[
 ("EMA_8_21_TREND","Fast EMA above slow EMA with positive momentum"),
 ("EMA_13_34_TREND","Medium EMA alignment with positive momentum"),
 ("MOMENTUM_6","Positive six-candle momentum"),
 ("BREAKOUT_20","Close breaks the previous 20-candle high"),
 ("RSI_RECOVERY","RSI recovery from a weak zone"),
 ("VOLUME_BREAKOUT","Positive candle with above-average volume"),
 ("VWAP_RECLAIM","Close reclaims rolling volume-weighted price"),
 ("RANGE_EXPANSION","Positive candle with expanding range"),
]
FEE_PCT=0.10
SLIPPAGE_PCT=0.05

def _ema(values,period):
    if not values:return 0.0
    k=2.0/(period+1);v=values[0]
    for x in values[1:]:v=x*k+v*(1-k)
    return v

def _rsi(values,period=14):
    d=[values[i]-values[i-1] for i in range(max(1,len(values)-period),len(values))]
    g=sum(max(0,x) for x in d);l=sum(max(0,-x) for x in d)
    return 100.0 if l==0 and g else (50.0 if l==0 else 100-100/(1+g/l))

def _flags(candles):
    c=[float(x["c"]) for x in candles];v=[float(x.get("vol",0)) for x in candles]
    if len(c)<40:return {}
    last=c[-1];mom=last/c[-7]-1 if c[-7] else 0
    avgv=sum(v[-21:-1])/20
    typical=[(float(x["h"])+float(x["l"])+float(x["c"]))/3 for x in candles[-20:]]
    v20=v[-20:];vwap=sum(p*q for p,q in zip(typical,v20))/max(1e-12,sum(v20))
    prev_range=float(candles[-2]["h"])-float(candles[-2]["l"])
    last_range=float(candles[-1]["h"])-float(candles[-1]["l"])
    return {
      "EMA_8_21_TREND":_ema(c[-40:],8)>_ema(c[-40:],21) and mom>0,
      "EMA_13_34_TREND":_ema(c[-40:],13)>_ema(c[-40:],34) and mom>0,
      "MOMENTUM_6":mom>0.002,
      "BREAKOUT_20":last>max(float(x["h"]) for x in candles[-21:-1]),
      "RSI_RECOVERY":35<_rsi(c)<58 and c[-1]>c[-2],
      "VOLUME_BREAKOUT":c[-1]>c[-2] and avgv>0 and v[-1]>1.4*avgv,
      "VWAP_RECLAIM":c[-2]<=vwap and last>vwap,
      "RANGE_EXPANSION":last>c[-2] and last_range>1.25*max(prev_range,1e-12),
    }

def _backtest(candles,name):
    trades=[];fee=(2*FEE_PCT)+SLIPPAGE_PCT;i=40
    # One paper position at a time per strategy; never count overlapping entries.
    while i<len(candles)-1:
        flags=_flags(candles[:i+1])
        if not flags.get(name):
            i+=1;continue
        entry=float(candles[i]["c"]);tp=entry*1.012;sl=entry*0.993;exit_price=None;exit_index=None
        for j in range(i+1,min(len(candles),i+49)):
            bar=candles[j];hi=float(bar["h"]);lo=float(bar["l"])
            if lo<=sl:
                exit_price=sl;exit_index=j;break
            if hi>=tp:
                exit_price=tp;exit_index=j;break
        if exit_price is None:
            # The simulated position remains occupied for its full 48-candle evaluation window.
            i=min(len(candles)-1,i+48);continue
        trades.append((exit_price/entry-1)*100-fee)
        i=exit_index+1
    wins=[x for x in trades if x>0];losses=[x for x in trades if x<0]
    gross_win=sum(wins);gross_loss=abs(sum(losses))
    pf=gross_win/gross_loss if gross_loss else (99.0 if wins else 0.0)
    return {"trades":len(trades),"wins":len(wins),"losses":len(losses),
            "win_rate_pct":round(100*len(wins)/len(trades),2) if trades else None,
            "profit_factor":round(pf,3),"expectancy_pct":round(sum(trades)/len(trades),4) if trades else None,
            "eligible":len(trades)>=20 and pf>1.0 and sum(trades)>0}

def evaluate_strategies(candles):
    flags=_flags(candles);closes=[float(x["c"]) for x in candles]
    if not flags:return []
    mom=closes[-1]/closes[-7]-1 if closes[-7] else 0
    rsi=_rsi(closes);results=[]
    for name,description in STRATEGIES:
        active=bool(flags.get(name))
        bt=_backtest(candles,name)
        score=50+min(25,abs(mom)*2500)+(10 if _ema(closes[-40:],8)>_ema(closes[-40:],21) else 0)
        if bt["trades"]>=20:
            score+=max(-20,min(20,(bt["expectancy_pct"] or 0)*8))
        if not active:score=min(score,49)
        results.append({"name":name,"description":description,"active":active,
                        "score":round(max(0,min(100,score)),2),**bt})
    return sorted(results,key=lambda x:(x["eligible"],x["score"],x["profit_factor"]),reverse=True)

def select_strategy(candles):
    ranked=evaluate_strategies(candles)
    active=[x for x in ranked if x["active"]]
    return (active[0] if active else None),ranked
