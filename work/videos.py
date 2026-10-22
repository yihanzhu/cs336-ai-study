import json,pathlib,re,urllib.request
out={}
for year,pid in [(2025,'PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_'),(2026,'PLoROMvodv4rMqXOcazWaTUHhq-yembLCV')]:
    url='https://www.youtube.com/playlist?list='+pid
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        text=urllib.request.urlopen(req,timeout=35).read().decode()
        match=re.search(r'var ytInitialData = (.*?);</script>',text)
        data=json.loads(match.group(1)); entries=[]
        def walk(x):
            if isinstance(x,dict):
                if 'playlistVideoRenderer' in x:
                    r=x['playlistVideoRenderer']; entries.append({'title':''.join(t['text'] for t in r.get('title',{}).get('runs',[])),'url':'https://www.youtube.com/watch?v='+r['videoId']})
                for v in x.values(): walk(v)
            elif isinstance(x,list):
                for v in x: walk(v)
        walk(data); out[str(year)]={'url':url,'videos':entries}
    except Exception as e: out[str(year)]={'url':url,'error':str(e)}
pathlib.Path('work/videos.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
