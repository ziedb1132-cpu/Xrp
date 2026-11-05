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
    price,candidate,features,ranked,context=analyze(sentiment,state.get("entry_method_stats",{}))
    state["news_items"]=news[:250];state["news_sentiment"]=sentiment;state["news_updated_at"]=int(time.time())
    write(state)
    gate=gate_status(state.get("history",[]))
    probability=predict_probability(features) if gate["enabled"] else None
    exit_policy=predict_exit_policy(features,price) if gate["enabled"] else None
    if candidate and probability is not None:
        candidate["ai_win_probability"]=round(probability,5)
        # Preserve the existing model confidence filter; TP/SL policy is handled independently.
        if probability<0.52:candidate=None
    if candidate:
        if exit_policy:
            candidate.update(exit_policy)
            candidate["exit_policy_source"]="AI_LEARNED_EXIT_POLICY"
            candidate["exit_policy_samples"]=exit_policy.get("model_samples",0)
        else:
            # Safe initial levels until the exit model has enough observed excursions.
            candidate["tp"]=price*1.012
            candidate["sl"]=price*0.993
            candidate["exit_policy_source"]="FIXED_FALLBACK"
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
        if result.get("trained"):
            state["last_training_samples"]=int(result.get("samples",gate.get("compatible_signals",0)))
    state["strategy_discovery"]={"evaluated":len(ranked),"eligible":sum(1 for x in ranked if x.get("eligible")),
                                  "top":ranked[:20],"updated_at":int(time.time())}
    write(state)
    SignalManager().write_summary(state)
    with open("strategy_history.json","w",encoding="utf-8") as f:
        json.dump({"cycle":state["cycle"],"updated_at":state["updated_at"],"strategies":ranked[:20]},f,ensure_ascii=False,separators=(",",":"))
    print(json.dumps({"cycle":state["cycle"],"price":price,"open":bool(state.get("open_signal")),
                      "news":len(news),"features":len(features),"strategies":len(ranked),
                      "ai_gate":gate,"ai_probability":probability,"ai_exit_policy":exit_policy,"live_execution":live_status},ensure_ascii=False))

if __name__=="__main__":main()
