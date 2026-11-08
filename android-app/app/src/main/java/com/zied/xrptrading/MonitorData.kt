package com.zied.xrptrading
import org.json.JSONObject
import org.json.JSONArray
import java.net.HttpURLConnection
import java.net.URL

data class SignalInfo(val id:String,val symbol:String,val side:String,val entry:String,val tp:String,val sl:String,val strategy:String,val status:String)
data class EventInfo(val id:String,val type:String,val title:String,val message:String,val at:Long,val cycle:Int)
data class MonitorData(
    val price:String="—",val cycle:Int=0,val updatedAt:Long=0,val closed:Int=0,val winRate:String="—",
    val net:String="—",val profitFactor:String="—",val aiEnabled:Boolean=false,val aiWins:Int=0,val aiLosses:Int=0,
    val open:SignalInfo?=null,val pending:SignalInfo?=null,val events:List<EventInfo> = emptyList()
)

object SummaryRepository {
    private const val DATA_URL="https://raw.githubusercontent.com/ziedb1132-cpu/Xrp/main/xrp_summary.json"
    fun load():MonitorData {
        val connection=URL(DATA_URL).openConnection() as HttpURLConnection
        try {
            connection.connectTimeout=12000; connection.readTimeout=12000
            connection.setRequestProperty("Cache-Control","no-cache"); connection.setRequestProperty("Pragma","no-cache")
            val body=connection.inputStream.bufferedReader().use{it.readText()}
            return parse(JSONObject(body))
        } finally { connection.disconnect() }
    }
    private fun number(o:JSONObject,key:String,fallback:String="—"):String {
        if(o.isNull(key)) return fallback
        val v=o.optDouble(key,Double.NaN)
        return if(v.isNaN()) fallback else String.format(java.util.Locale.US,"%.2f",v)
    }
    private fun signal(o:JSONObject?):SignalInfo? {
        if(o==null)return null
        return SignalInfo(o.opt("id")?.toString()?:"—",o.optString("symbol","XRP-USDT"),o.optString("side","BUY"),
            number(o,"entry"),number(o,"tp"),number(o,"sl"),o.optString("strategy_name",o.optString("strategy_type","—")),o.optString("status","OPEN"))
    }
    private fun parse(root:JSONObject):MonitorData {
        val gate=root.optJSONObject("ai_gate")?:JSONObject()
        val raw=root.optJSONArray("recent_events")?:JSONArray()
        val events=(0 until raw.length()).mapNotNull{i->
            val e=raw.optJSONObject(i)?:return@mapNotNull null
            val type=e.optString("type","EVENT"); val symbol=e.optString("symbol","XRP-USDT")
            val id=e.opt("event_id")?.toString()?:"${type}-${e.optLong("at")}"
            val sid=e.opt("signal_id")?.toString()?:"—"
            val price=if(e.has("price")) number(e,"price","") else ""
            val message=when(type){
                "SIGNAL_OPENED"->"تم فتح $symbol · إشارة #$sid · الدخول $price"
                "SIGNAL_CLOSED"->"إغلاق $symbol · ${e.optString("status","—")} · العائد ${number(e,"net_return_pct")}%"
                "SL_MOVED"->"تحريك SL · ${number(e,"sl_old")} ← ${number(e,"sl_new")} · السعر $price"
                else->"$symbol · $type"
            }
            EventInfo(id,type,when(type){"SIGNAL_OPENED"->"فتح إشارة";"SIGNAL_CLOSED"->"إغلاق إشارة";"SL_MOVED"->"تحريك وقف الخسارة";else->"حدث"},
                message,e.optLong("at"),e.optInt("cycle"))
        }.sortedByDescending{it.at}
        return MonitorData(
            price=number(root,"current_price"),cycle=root.optInt("cycle"),updatedAt=root.optLong("updated_at"),
            closed=root.optInt("closed_signals"),winRate=number(root,"win_rate_pct"),net=number(root,"net_profit_pct"),
            profitFactor=number(root,"profit_factor"),aiEnabled=gate.optBoolean("enabled"),aiWins=gate.optInt("wins"),
            aiLosses=gate.optInt("losses"),open=signal(root.optJSONObject("open_signal")),
            pending=signal(root.optJSONObject("pending_signal")),events=events
        )
    }
}
