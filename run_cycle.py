"""One complete persistent XRP/USDT research and paper-signal cycle."""
import json,time
from state_storage import restore
from xrp_engine import analyze
from signal_manager import SignalManager,read,write
from news_engine import fetch_news,sentiment_score
from ai_model import gate_status,train,predict_probability,predict_exit_policy
from okx_live import process_live

def main():
    restore()
    news=fetch_news();sentiment=sentiment_score(news)
    state=read()
    closed_history=[x for x in state.get("history",[]) if x.get("status") in ("TP","SL")]
    recent=closed_history[-10:]
    recent_stats={
        "closed":len(recent),
        "win_rate_pct":(100*sum(1 for x in recent if float(x.get("net_return_pct",0) or 0)>0)/len(recent)) if recent else 50.0,
    }
    price,candidate,features,ranked,context=analyze(
        sentiment,state.get("entry_method_stats",{}),recent_stats
    )
    state["news_items"]=news[:250];state["news_sentiment"]=sentiment;state["news_updated_at"]=int(time.time())
    write(state)
    gate=gate_status(state.get("history",[]))
    probability=predict_probability(features) if gate["enabled"] else None
    atr_pct=float((candidate or {}).get("atr_pct", 0.0) or 0.0)
    exit_policy=predict_exit_policy(features,price,atr_pct=atr_pct) if gate["enabled"] else None
    if candidate and probability is not None:
        candidate["ai_win_probability"]=round(probability,5)
        # Early-stage probabilities are noisy. The independent technical/risk gate
        # remains primary; only an extreme model warning can veto an otherwise valid setup.
        candidate["ai_probability_role"]="secondary_risk_check"
        if probability<0.40:
            candidate=None
            context.setdefault("entry_diagnostics",{})["reason"]="ai_extreme_low_confidence_veto"
            context["entry_diagnostics"]["ai_probability"]=round(probability,5)
    if candidate and candidate.get("strategy_type") == "SCALPING":
        # Keep the dedicated low-distance scalp TP/SL; the swing exit model is not
        # trained for this separate strategy profile.
        candidate["exit_policy_source"] = "SCALPING_ATR_FEE_AWARE"
        candidate["exit_policy_samples"] = 0
    elif candidate:
        if exit_policy:
            candidate.update(exit_policy)
            candidate["exit_policy_source"]="AI_LEARNED_EXIT_POLICY"
            candidate["exit_policy_samples"]=exit_policy.get("model_samples",0)
            candidate["exit_distance_source"]="AI_LEARNED_PLUS_ATR_GUARD"
        else:
            # ATR-adaptive fallback until the learned exit model is ready.
            candidate["tp"]=price*(1+float(candidate.get("tp_distance_pct",0.80))/100.0)
            candidate["sl"]=price*(1-float(candidate.get("sl_distance_pct",0.45))/100.0)
            candidate["exit_policy_source"]="ATR_VOLATILITY_ADAPTIVE"
            candidate["exit_policy_samples"]=gate.get("exit_policy_samples",0)
    SignalManager().cycle(price,candidate,ranked,features,context)
    # Live execution is separately opt-in and observes only newly generated signals.
    live_status=process_live(read())
    state=read();gate=gate_status(state.get("history",[]));state["ai_gate"]=gate
    closed=[x for x in state.get("history",[]) if x.get("status") in ("TP","SL")]
    last_trained=int(state.get("last_training_samples",0))
    # Incremental update after every additional compatible closed signal.
    if gate["enabled"] and gate.get("compatible_signals",0)>last_trained:
        result=train(closed,epochs=2);state["last_training"]=result
        # Record the attempted sample boundary even when the guard rejects a model.
        # Retry only after genuinely new compatible closed signals arrive.
        state["last_training_samples"]=int(gate.get("compatible_signals",0))
    state["strategy_discovery"]={"evaluated":len(ranked),"eligible":sum(1 for x in ranked if x.get("eligible")),
                                  "top":ranked[:20],"updated_at":int(time.time())}
    write(state)
    SignalManager().write_summary(state)
    with open("strategy_history.json","w",encoding="utf-8") as f:
        json.dump({"cycle":state["cycle"],"updated_at":state["updated_at"],"strategies":ranked[:20]},f,ensure_ascii=False,separators=(",",":"))
    print(json.dumps({"cycle":state["cycle"],"price":price,"open":bool(state.get("open_signal")),
                      "pending":bool(state.get("pending_signal")),"entry_diagnostics":context.get("entry_diagnostics",{}),"scalping_diagnostics":context.get("scalping_diagnostics",{}),
                      "news":len(news),"features":len(features),"strategies":len(ranked),
                      "ai_gate":gate,"ai_probability":probability,"ai_exit_policy":exit_policy,"live_execution":live_status},ensure_ascii=False))

if __name__=="__main__":main()
