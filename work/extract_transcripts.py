import concurrent.futures,datetime,hashlib,json,pathlib,re,time
from youtube_transcript_api import YouTubeTranscriptApi
ROOT=pathlib.Path('work/transcripts');ROOT.mkdir(exist_ok=True)
jobs=[]
for year in (2025,2026):
    p=json.loads(pathlib.Path(f'work/playlist-{year}.json').read_text())
    for i,v in enumerate(p['entries'],1):jobs.append({'year':year,'playlist_order':i,'id':v['id'],'title':v['title'],'url':'https://www.youtube.com/watch?v='+v['id']})

def extract(v):
    p=ROOT/f'{v["year"]}-{v["playlist_order"]:02}-{v["id"]}.json'
    if p.exists():
        d=json.loads(p.read_text())
        if d.get('segments'):
            return {k:d[k] for k in d if k!='segments'}
    try:
        tracks=list(YouTubeTranscriptApi().list(v['id']))
        en=[t for t in tracks if t.language_code in ('en','en-US','en-GB') or t.language_code.startswith('en-')]
        if not en:raise RuntimeError('No English caption track available')
        en.sort(key=lambda t:(t.is_generated,t.language_code!='en-US',t.language_code!='en'))
        track=en[0]; data=track.fetch().to_raw_data()
        if not data:raise RuntimeError('Empty caption track')
        duration=max(x['start']+x['duration'] for x in data)
        blob=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode()
        d={**v,'status':'extracted','track_language':track.language_code,'is_generated':track.is_generated,'available_tracks':[{'language':t.language_code,'is_generated':t.is_generated} for t in tracks],'segments_count':len(data),'last_caption_end_seconds':duration,'word_count':sum(len(s['text'].split()) for s in data),'segments_sha256':hashlib.sha256(blob).hexdigest(),'extracted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_status':'not_yet_compared_with_lecture_material','segments':data}
        p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
        def stamp(s):
            n=int(s);return f'{n//3600:02}:{n%3600//60:02}:{n%60:02}'
        text='\n'.join(f'[{stamp(x["start"])}] {x["text"]}' for x in data)
        p.with_suffix('.txt').write_text(v['title']+'\n'+v['url']+'\n\n'+text)
        return {k:d[k] for k in d if k!='segments'}
    except Exception as e:
        return {**v,'status':'failed','error_type':type(e).__name__,'error':str(e)}

results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    for v in pool.map(extract,jobs):
        results.append(v);print(v['year'],v['playlist_order'],v['status'],v.get('segments_count',v.get('error_type')),flush=True)
        (ROOT/'index.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
print(json.dumps({'total':len(results),'extracted':sum(x['status']=='extracted' for x in results),'failed':[{'year':x['year'],'id':x['id'],'error':x.get('error_type')} for x in results if x['status']=='failed'],'caption_hours':sum(x.get('last_caption_end_seconds',0) for x in results)/3600},ensure_ascii=False,indent=2))
