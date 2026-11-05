"""Persistent XRP paper-signal lifecycle with AI-selected pending entry methods; paper only."""
import json, os, time
STATE="xrp_state.json"
FEE_PCT=float(os.getenv("PAPER_FEE_PCT","0.10"))
SLIPPAGE_PCT=float(os.getenv("PAPER_SLIPPAGE_PCT","0.05"))

def read():
    try:
        with open(STATE,encoding="utf-8") as f:s=json.load(f)
    except (OSError,ValueError):
        s={"cycle":0,"total_signals":0,"tp_count":0,"sl_count":0,"open_signal":None,"pending_signal":None,"history":[]}
    s.setdefault("history",[]);s.setdefault("total_signals",len(s["history"]))
    s.setdefault("tp_count",sum(1 for x in s["history"] if x.get("status")=="TP"))
    s.setdefault("sl_count",sum(1 for x in s["history"] if x.get("status")=="SL"))
    s.setdefault("open_signal",None);s.setdefault("pending_signal",None)
    s.setdefault("strategy_stats",{});s.setdefault("entry_method_stats",{})
    return s

def write(s):
    tmp=STATE+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:json.dump(s,f,ensure_ascii=False,separators=(",",":"))
    os.replace(tmp,STATE)

def _record_entry_result(a,s,net):
    method=a.get("entry_method","MARKET")
    row=s.setdefault("entry_method_stats",{}).setdefault(method,{"trades":0,"wins":0,"losses":0,"net_return_pct":0.0,"avg_net_return_pct":0.0,"win_rate_pct":0.0})
    row["trades"]+=1
    if net>0:row["wins"]+=1
    else:row["losses"]+=1
    row["net_return_pct"]=round(float(row.get("net_return_pct",0))+net,5)
    row["avg_net_return_pct"]=round(row["net_return_pct"]/row["trades"],5)
    row["win_rate_pct"]=round(100*row["wins"]/row["trades"],2)

def _close(a,status,price,now,s):
    entry=float(a.get("entry",price));gross=(price/entry-1)*100 if entry else 0.0
    high=float(a.get("high_water",entry));low=float(a.get("low_water",entry))
    mfe=max(0.0,(high/entry-1)*100) if entry else 0.0
    mae=max(0.0,(1-low/entry)*100) if entry else 0.0
    net=gross-(2*FEE_PCT)-SLIPPAGE_PCT
    a.update(status=status,exit_price=price,closed_at=now,gross_return_pct=round(gross,5),
             fees_pct=2*FEE_PCT,slippage_pct=SLIPPAGE_PCT,net_return_pct=round(net,5),
             close_reason=status,mfe_pct=round(mfe,5),mae_pct=round(mae,5))
    s["history"].append(a)
    if status=="TP":s["tp_count"]+=1
    else:s["sl_count"]+=1
    stats=s.setdefault("strategy_stats",{}).setdefault(a.get("strategy_name","BASELINE"),{"trades":0,"wins":0,"losses":0,"net_return_pct":0.0})
    stats["trades"]+=1;stats["wins" if net>0 else "losses"]+=1
    stats["net_return_pct"]=round(stats["net_return_pct"]+net,5)
    _record_entry_result(a,s,net)

def _open(p,price,now,s):
    s["total_signals"]+=1
    entry=float(price)
    # Rebase learned exit distances to the actual fill, not the earlier trigger.
    tp_distance=float(p.get("tp_distance_pct",1.2) or 1.2)
    sl_distance=float(p.get("sl_distance_pct",0.7) or 0.7)
    tp=entry*(1+tp_distance/100.0);sl=entry*(1-sl_distance/100.0)
    a={"id":s["total_signals"],"symbol":"XRP-USDT","side":"BUY","entry":entry,"tp":tp,"sl":sl,
       "original_tp":tp,"original_sl":sl,"tp_old":None,"tp_new":tp,"sl_old":None,"sl_new":sl,
       "high_water":entry,"low_water":entry,"opened_at":now,"status":"OPEN",
       "features":p.get("features",{}),"feature_version":p.get("feature_version","xrp-1200-v2"),
       "strategy_name":p.get("strategy_name","BASELINE_MOMENTUM"),
       "strategy_description":p.get("strategy_description","AI-selected entry setup"),
       "strategy_score":float(p.get("strategy_score",50)),"ai_win_probability":p.get("ai_win_probability"),
       "exit_policy_source":p.get("exit_policy_source","FIXED_FALLBACK"),
       "exit_policy_samples":p.get("exit_policy_samples",0),"predicted_mfe_pct":p.get("predicted_mfe_pct"),
       "predicted_mae_pct":p.get("predicted_mae_pct"),"tp_distance_pct":tp_distance,"sl_distance_pct":sl_distance,
       "entry_method":p.get("entry_method","MARKET"),"entry_method_scores":p.get("entry_method_scores",{}),
       "entry_policy_source":p.get("entry_policy_source","ONLINE_LEARNED_ENTRY_POLICY"),
       "entry_trigger_price":p.get("entry_trigger_price"),"entry_reason":p.get("entry_reason","AI entry confirmation"),
       "timeframe":p.get("timeframe","5m"),"market_regime":p.get("market_regime","unknown"),
       "opened_price":price,"trailing_updates":0}
    s["open_signal"]=a;s["pending_signal"]=None

def _pending_trigger(p,price):
    method=p.get("entry_method","MARKET")
    if method=="MARKET":return True
    invalid=float(p.get("entry_invalidation_price",0) or 0)
    if invalid>0 and price<invalid:return "CANCEL"
    if method=="BREAKOUT":
        return price>=float(p.get("entry_trigger_price",float("inf")))
    if method=="RETEST":
        trigger=float(p.get("entry_trigger_price",float("inf")))
        level=float(p.get("entry_retest_price",0) or 0)
        if p.get("entry_stage")!="WAIT_RETEST":
            if price>=trigger:p["entry_stage"]="WAIT_RETEST"
            return False
        return level>0 and price<=level*1.003 and price>=level*0.997
    if method=="SUPPORT_BOUNCE":
        support=float(p.get("entry_support_price",0) or 0)
        if support<=0:return False
        if price<=support*1.01:p["support_touched"]=True
        return bool(p.get("support_touched")) and price>=support*1.002
    return False

class SignalManager:
    def cycle(self,price,candidate=None,ranked_strategies=None,features=None,market_context=None):
        s=read();s["cycle"]+=1;now=int(time.time());a=s.get("open_signal")
        if a:
            a.setdefault("original_tp",a.get("tp"));a.setdefault("original_sl",a.get("sl"))
            a.setdefault("high_water",float(a.get("entry",price)));a.setdefault("low_water",float(a.get("entry",price)))
            previous_high=float(a["high_water"]);a["high_water"]=max(previous_high,float(price))
            a["low_water"]=min(float(a["low_water"]),float(price))
            if price>=float(a["tp"]):_close(a,"TP",price,now,s);s["open_signal"]=None
            elif price<=float(a["sl"]):_close(a,"SL",price,now,s);s["open_signal"]=None
            elif price>previous_high:
                old=float(a["sl"]);proposed=price*0.993
                if proposed>old:
                    log=a.setdefault("risk_history",[])
                    log.append({"at":now,"price":price,"sl_old":old,"sl_new":proposed,"reason":"trailing_stop_tightened"})
                    a.update(sl_old=old,sl=proposed,sl_new=proposed,sl_updated_at=now,
                             change_reason="trailing_stop_tightened",trailing_updates=int(a.get("trailing_updates",0))+1)
        if not s.get("open_signal"):
            pending=s.get("pending_signal")
            if pending:
                outcome=_pending_trigger(pending,float(price))
                if outcome=="CANCEL":s["pending_signal"]=None
                elif outcome is True:_open(pending,float(price),now,s)
            elif candidate and candidate.get("side")=="BUY":
                pending=dict(candidate);pending["features"]=features or candidate.get("features",{})
                pending["market_regime"]=(market_context or {}).get("regime","unknown")
                pending["pending_since"]=now;pending["entry_stage"]="WAIT_BREAKOUT"
                s["pending_signal"]=pending
                outcome=_pending_trigger(pending,float(price))
                if outcome is True:_open(pending,float(price),now,s)
                elif outcome=="CANCEL":s["pending_signal"]=None
        s["price"]=price;s["updated_at"]=now;s["ranked_strategies"]=(ranked_strategies or [])[:20]
        s["market_context"]=market_context or {}
        write(s);self.write_summary(s)
    def write_summary(self,s):
        source=s.get("open_signal")
        allowed=("id","signal_id","symbol","side","entry","tp","sl","original_tp","original_sl","tp_old","tp_new","sl_old","sl_new","high_water","low_water","opened_at","status","strategy_name","strategy_description","strategy_score","ai_win_probability","exit_policy_source","exit_policy_samples","predicted_mfe_pct","predicted_mae_pct","tp_distance_pct","sl_distance_pct","entry_method","entry_method_scores","entry_policy_source","entry_trigger_price","entry_reason","timeframe","market_regime","opened_price","trailing_updates","risk_history","sl_updated_at","change_reason")
        a={key:source[key] for key in allowed if key in source} if isinstance(source,dict) else None
        if a and isinstance(a.get("risk_history"),list):a["risk_history"]=a["risk_history"][-20:]
        pending=s.get("pending_signal")
        p=None
        if isinstance(pending,dict):
            p={key:pending[key] for key in ("entry_method","entry_policy_source","entry_trigger_price","entry_retest_price","entry_support_price","entry_stage","pending_since","strategy_name","strategy_score","entry_reason") if key in pending}
        closed=[x for x in s.get("history",[]) if x.get("status") in ("TP","SL")]
        net_values=[float(x.get("net_return_pct",0) or 0) for x in closed]
        profitable=sum(1 for x in net_values if x>0)
        summary={"price":s.get("price"),"current_price":s.get("price"),"closed_signals":len(closed),
          "net_profit_pct":round(sum(net_values),5),"win_rate_pct":round(100*profitable/len(net_values),2) if net_values else None,
          "total_signals":s["total_signals"],"tp_count":s["tp_count"],"sl_count":s["sl_count"],"cycle":s["cycle"],"updated_at":s.get("updated_at"),
          "open_signal":a,"pending_signal":p,"entry_method_stats":s.get("entry_method_stats",{}),
          "strategy_count":len(s.get("ranked_strategies",[])),"top_strategies":s.get("ranked_strategies",[])[:20],
          "feature_count":1205,"ai_gate":s.get("ai_gate",{}),"last_training":s.get("last_training",{}),
          "market_context":s.get("market_context",{})}
        tmp="xrp_summary.json.tmp"
        with open(tmp,"w",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,separators=(",",":"))
        os.replace(tmp,"xrp_summary.json")
