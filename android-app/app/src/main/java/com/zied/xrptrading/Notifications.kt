package com.zied.xrptrading
import android.Manifest
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters

object LocalEvents {
    private const val PREFS="xrp_event_notifications"
    private const val CHANNEL="xrp_signal_events"
    fun ensureChannel(context:Context) {
        if(Build.VERSION.SDK_INT>=Build.VERSION_CODES.O) {
            val manager=context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(NotificationChannel(CHANNEL,"إشعارات إشارات XRP",NotificationManager.IMPORTANCE_DEFAULT)
                .apply{description="فتح وإغلاق الإشارات وتحريك وقف الخسارة"})
        }
    }
    fun consume(context:Context,events:List<EventInfo>) {
        ensureChannel(context)
        val prefs=context.getSharedPreferences(PREFS,Context.MODE_PRIVATE)
        val seen=prefs.getStringSet("seen_ids",emptySet())?.toMutableSet()?:mutableSetOf()
        if(!prefs.getBoolean("initialized",false)) {
            seen.addAll(events.map{it.id})
            prefs.edit().putStringSet("seen_ids",seen).putBoolean("initialized",true).apply()
            return
        }
        if(events.isEmpty())return
        val canNotify=Build.VERSION.SDK_INT<33 || ContextCompat.checkSelfPermission(context,Manifest.permission.POST_NOTIFICATIONS)==PackageManager.PERMISSION_GRANTED
        val fresh=events.filter{it.id !in seen}.sortedBy{it.at}
        fresh.forEach{event->
            if(canNotify) {
                val n=NotificationCompat.Builder(context,CHANNEL).setSmallIcon(android.R.drawable.ic_dialog_info)
                    .setContentTitle("XRP AI · ${event.title}").setContentText(event.message)
                    .setStyle(NotificationCompat.BigTextStyle().bigText(event.message)).setAutoCancel(true)
                    .setPriority(NotificationCompat.PRIORITY_DEFAULT).build()
                NotificationManagerCompat.from(context).notify(event.id.hashCode(),n)
            }
            if(canNotify) seen.add(event.id)
        }
        if(seen.size>300) seen.retainAll(events.map{it.id}.toSet())
        prefs.edit().putStringSet("seen_ids",seen).apply()
    }
}
class XrpRefreshWorker(context:Context,params:WorkerParameters):CoroutineWorker(context,params) {
    override suspend fun doWork():Result=try {
        val data=kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO){SummaryRepository.load()}
        LocalEvents.consume(applicationContext,data.events); Result.success()
    } catch(_:Exception){Result.retry()}
}
