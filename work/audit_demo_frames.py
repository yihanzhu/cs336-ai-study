"""Fetch frames for every identified demo context, without equating frames with full video verification."""
import concurrent.futures,hashlib,json,pathlib,re,subprocess,sys
import yt_dlp
from review_notes import FINDINGS
from full_reading_notes import FULL_READINGS
W=pathlib.Path(__file__).resolve().parent;D=W/'demo-audit';D.mkdir(exist_ok=True)
V={f"{v['year']}-{v['playlist_order']:02}":v for v in json.loads((W/'transcripts/index.json').read_text())}
contexts={}
for f in FINDINGS:
 if re.search('演示|现场|输出|profiler|图中|图表|画面|屏幕|曲线|截图|表格|显示',f['finding']):contexts.setdefault(f['lecture'],[]).append((f['start'],f['end'],f['finding']))
for key,notes in FULL_READINGS.items():
 for a,b,text in notes:
  if re.search('现场|演示|trace|profiler|实际执行|没有完成|未成功|运行结果',text,re.I):contexts.setdefault(key,[]).append((a,b,text))
class Quiet:
 def debug(self,*a):pass
 def warning(self,*a):pass
 def error(self,*a):pass
def run(job):
 key,cs=job;v=V[key];records=[]
 try:
  with yt_dlp.YoutubeDL({'quiet':True,'no_warnings':True,'logger':Quiet(),'skip_download':True,'socket_timeout':20,'retries':0,'extractor_retries':0}) as y:info=y.extract_info(v['url'],download=False)
  formats=[x for x in info['formats'] if x.get('vcodec','none').startswith('avc') and x.get('height') in [720,480,360] and x.get('protocol')=='https'];formats.sort(key=lambda x:-x['height']);f=formats[0]
  times={}
  for i,(a,b,text) in enumerate(cs):
   for t in [int((a+b)/2),int(min(b-3,a+25))]:
    if t>=0 and t<info['duration']:times.setdefault(t,[]).append(i)
  for t,ids in sorted(times.items()):
   p=D/f'{key}-{t}.png';success=p.exists()
   if not success:
    r=subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',f['url'],'-frames:v','1','-an','-vf','scale=-2:720','-y',str(p)],capture_output=True,timeout=60);success=r.returncode==0 and p.exists()
   row={'lecture':key,'video_id':v['id'],'timestamp':t,'success':success,'context_ids':ids}
   if success:row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
   records.append(row)
 except Exception as e:records.append({'lecture':key,'success':False,'error_type':type(e).__name__})
 (D/(key+'-index.json')).write_text(json.dumps({'lecture':key,'contexts':[{'start':a,'end':b,'finding':t} for a,b,t in cs],'frames':records,'visually_reviewed':False},ensure_ascii=False,indent=2));print(json.dumps({'lecture':key,'frames':len(records),'success':sum(r['success'] for r in records)}),flush=True);return records
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:allrows=sum(list(pool.map(run,sorted(contexts.items()))),[])
(D/'index.json').write_text(json.dumps({'frames':allrows,'visually_reviewed':False},ensure_ascii=False,indent=2));print('DONE',len(allrows),flush=True)
