"""XRP and Ripple news intake with a strict 24-hour publication window and duplicate filtering."""
import time,requests,xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
FEEDS=[
 "https://news.google.com/rss/search?q=XRP+Ripple+crypto&hl=en-US&gl=US&ceid=US:en",
 "https://news.google.com/rss/search?q=XRP+Ripple+payments&hl=en-US&gl=US&ceid=US:en"
]
def fetch_news():
    cutoff=time.time()-86400;seen=set();items=[]
    for url in FEEDS:
        try:
            response=requests.get(url,timeout=15,headers={"User-Agent":"XrpResearchBot/1.0"})
            response.raise_for_status();root=ET.fromstring(response.text)
            for it in root.findall(".//item"):
                title=(it.findtext("title") or "").strip();link=(it.findtext("link") or "").strip()
                pub=(it.findtext("pubDate") or "").strip()
                if not title or not link or link in seen or not pub:continue
                try:published=parsedate_to_datetime(pub).timestamp()
                except (TypeError,ValueError,OverflowError):continue
                if published<cutoff or published>time.time()+300:continue
                seen.add(link)
                items.append({"title":title,"url":link,"published":pub,"published_ts":int(published),"source":"Google News RSS"})
        except Exception:continue
    return sorted(items,key=lambda x:x["published_ts"],reverse=True)
def sentiment_score(items):
    positive={"surge","rise","gain","bull","adoption","launch","upgrade","growth","integrates","partnership"}
    negative={"fall","drop","hack","exploit","lawsuit","bear","outage","crash","attack","liquidation"}
    scores=[]
    for item in items:
        words=set(item.get("title","").lower().replace(","," ").replace(":"," ").split())
        scores.append((len(words&positive)-len(words&negative))/max(1,len(words)))
    return sum(scores)/len(scores) if scores else 0.0
