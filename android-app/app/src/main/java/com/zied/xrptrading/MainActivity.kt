package com.zied.xrptrading
import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Dashboard
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.ShowChart
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.work.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

class MainActivity:ComponentActivity() {
    private val permissionLauncher=registerForActivityResult(ActivityResultContracts.RequestPermission()){}
    override fun onCreate(savedInstanceState:Bundle?){
        super.onCreate(savedInstanceState)
        LocalEvents.ensureChannel(this)
        if(Build.VERSION.SDK_INT>=33 && ContextCompat.checkSelfPermission(this,Manifest.permission.POST_NOTIFICATIONS)!=PackageManager.PERMISSION_GRANTED)
            permissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        val work=PeriodicWorkRequestBuilder<XrpRefreshWorker>(15,TimeUnit.MINUTES)
            .setConstraints(Constraints.Builder().setRequiredNetworkType(NetworkType.CONNECTED).build()).build()
        WorkManager.getInstance(this).enqueueUniquePeriodicWork("xrp-background-refresh",ExistingPeriodicWorkPolicy.KEEP,work)
        setContent{XrpApp()}
    }
    @Composable private fun XrpApp(){
        var data by remember{mutableStateOf(MonitorData())}
        var loading by remember{mutableStateOf(true)}
        var error by remember{mutableStateOf<String?>(null)}
        var tab by remember{mutableIntStateOf(0)}
        suspend fun refresh(){
            try{
                val fresh=withContext(Dispatchers.IO){SummaryRepository.load()}
                data=fresh;error=null;LocalEvents.consume(this@MainActivity,fresh.events)
            }catch(_:Exception){error="تعذر الاتصال بـ GitHub. تحقق من الإنترنت."}
            finally{loading=false}
        }
        LaunchedEffect(Unit){while(true){refresh();delay(60_000)}}
        MaterialTheme(colorScheme=lightColorScheme(primary=Color(0xFF3157C7),background=Color(0xFFF5F7FB),surface=Color.White)){
            Scaffold(topBar={
                Column(Modifier.fillMaxWidth().padding(horizontal=20.dp,vertical=14.dp)){
                    Text("XRP AI Trading",style=MaterialTheme.typography.titleLarge,fontWeight=FontWeight.Bold)
                    Text("مراقبة المحرك · GitHub",style=MaterialTheme.typography.bodySmall,color=MaterialTheme.colorScheme.onSurfaceVariant)
                }
            },bottomBar={
                NavigationBar{
                    NavigationBarItem(selected=tab==0,onClick={tab=0},icon={Icon(Icons.Default.Dashboard,null)},label={Text("الرئيسية")})
                    NavigationBarItem(selected=tab==1,onClick={tab=1},icon={Icon(Icons.Default.ShowChart,null)},label={Text("الإشارة")})
                    NavigationBarItem(selected=tab==2,onClick={tab=2},icon={Icon(Icons.Default.Notifications,null)},label={Text("الأحداث")})
                }
            }){padding->
                when(tab){0->DashboardScreen(data,loading,error,Modifier.padding(padding));1->SignalScreen(data,Modifier.padding(padding));else->EventsScreen(data,Modifier.padding(padding))}
            }
        }
    }
}
@Composable private fun DashboardScreen(data:MonitorData,loading:Boolean,error:String?,modifier:Modifier=Modifier){
    LazyColumn(modifier.fillMaxSize().padding(horizontal=16.dp),verticalArrangement=Arrangement.spacedBy(12.dp),contentPadding=PaddingValues(vertical=12.dp)){
        item{
            Card(colors=CardDefaults.cardColors(containerColor=MaterialTheme.colorScheme.primary),shape=RoundedCornerShape(22.dp)){
                Column(Modifier.fillMaxWidth().padding(20.dp)){
                    Text("سعر XRP-USDT",color=MaterialTheme.colorScheme.onPrimary.copy(alpha=.8f))
                    Text("${data.price} USDT",color=MaterialTheme.colorScheme.onPrimary,style=MaterialTheme.typography.headlineMedium,fontWeight=FontWeight.Bold)
                    Text("الدورة #${data.cycle} · ${timeText(data.updatedAt)}",color=MaterialTheme.colorScheme.onPrimary.copy(alpha=.8f))
                }
            }
        }
        item{Row(horizontalArrangement=Arrangement.spacedBy(10.dp)){MetricCard("الإشارات المغلقة",data.closed.toString(),Modifier.weight(1f));MetricCard("نسبة النجاح","${data.winRate}%",Modifier.weight(1f))}}
        item{Row(horizontalArrangement=Arrangement.spacedBy(10.dp)){MetricCard("صافي العائد","${data.net}%",Modifier.weight(1f));MetricCard("Profit Factor",data.profitFactor,Modifier.weight(1f))}}
        item{Card{Column(Modifier.fillMaxWidth().padding(16.dp)){Text("حالة الذكاء الاصطناعي",fontWeight=FontWeight.Bold);Text(if(data.aiEnabled)"النموذج مفعّل" else "النموذج في مرحلة التعلم");Text("انتصارات: ${data.aiWins} · خسائر: ${data.aiLosses}",color=MaterialTheme.colorScheme.onSurfaceVariant)}}}
        item{Text("الإشارة المفتوحة",style=MaterialTheme.typography.titleMedium,fontWeight=FontWeight.Bold);SignalCard(data.open)}
        item{Text("آخر الأحداث",style=MaterialTheme.typography.titleMedium,fontWeight=FontWeight.Bold)}
        items(data.events.take(3)){EventCard(it)}
        item{Text(error?:if(loading)"جارٍ تحميل البيانات…" else "تحديث تلقائي كل دقيقة أثناء فتح التطبيق · مزامنة خلفية دورية",style=MaterialTheme.typography.bodySmall,color=if(error!=null)MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)}
    }
}
@Composable private fun SignalScreen(data:MonitorData,modifier:Modifier=Modifier){
    LazyColumn(modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)){
        item{Text("الإشارة المفتوحة",style=MaterialTheme.typography.titleLarge,fontWeight=FontWeight.Bold);SignalCard(data.open)}
        item{Text("الإشارة المعلقة",style=MaterialTheme.typography.titleLarge,fontWeight=FontWeight.Bold);SignalCard(data.pending)}
    }
}
@Composable private fun EventsScreen(data:MonitorData,modifier:Modifier=Modifier){
    LazyColumn(modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(10.dp)){
        item{Text("سجل الأحداث",style=MaterialTheme.typography.titleLarge,fontWeight=FontWeight.Bold)}
        if(data.events.isEmpty())item{Text("لا توجد أحداث مسجلة بعد.",color=MaterialTheme.colorScheme.onSurfaceVariant)}
        items(data.events){EventCard(it)}
    }
}
@Composable private fun MetricCard(title:String,value:String,modifier:Modifier=Modifier){
    Card(modifier){Column(Modifier.padding(14.dp)){Text(title,style=MaterialTheme.typography.bodySmall,color=MaterialTheme.colorScheme.onSurfaceVariant);Text(value,style=MaterialTheme.typography.titleLarge,fontWeight=FontWeight.Bold)}}
}
@Composable private fun SignalCard(signal:SignalInfo?){
    Card{if(signal==null)Text("لا توجد إشارة حالياً",Modifier.padding(16.dp),color=MaterialTheme.colorScheme.onSurfaceVariant)
    else Column(Modifier.fillMaxWidth().padding(16.dp),verticalArrangement=Arrangement.spacedBy(8.dp)){
        Row(verticalAlignment=Alignment.CenterVertically,horizontalArrangement=Arrangement.SpaceBetween,modifier=Modifier.fillMaxWidth()){
            Text("${signal.symbol} · ${signal.side}",fontWeight=FontWeight.Bold);AssistChip(onClick={},label={Text(signal.status)})
        }
        Text("الدخول: ${signal.entry}   TP: ${signal.tp}   SL: ${signal.sl}")
        Text("الاستراتيجية: ${signal.strategy}",color=MaterialTheme.colorScheme.onSurfaceVariant)
    }}
}
@Composable private fun EventCard(event:EventInfo){
    Card{Column(Modifier.fillMaxWidth().padding(14.dp),verticalArrangement=Arrangement.spacedBy(5.dp)){
        Text(event.title,fontWeight=FontWeight.Bold);Text(event.message,style=MaterialTheme.typography.bodyMedium)
        Text("الدورة #${event.cycle} · ${timeText(event.at)}",style=MaterialTheme.typography.bodySmall,color=MaterialTheme.colorScheme.onSurfaceVariant)
    }}
}
private fun timeText(epoch:Long):String=if(epoch<=0)"—" else SimpleDateFormat("dd/MM HH:mm",Locale("ar","TN")).format(Date(epoch*1000))
