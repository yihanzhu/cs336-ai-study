"""Full original-audio acquisition and ASR/caption comparison. No human-listening claim."""
import concurrent.futures,hashlib,json,os,pathlib,re,difflib,time,threading
W=pathlib.Path(__file__).resolve().parent;A=W/'audio-audit';A.mkdir(exist_ok=True);M=A/'original';M.mkdir(exist_ok=True)
os.environ['HF_HOME']=str(W/'asr-cache');os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import yt_dlp
from faster_whisper import WhisperModel
class Quiet:
 def debug(self,*a):pass
 def warning(self,*a):pass
 def error(self,*a):pass
model=WhisperModel('base.en',device='cpu',compute_type='int8',cpu_threads=2,num_workers=2,download_root=str(W/'asr-cache/models'),local_files_only=True)
lock=threading.Lock()
def emit(x):
 with lock:print(json.dumps(x,ensure_ascii=False),flush=True)
def atomic(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2));tmp.replace(p)
def toks(t):return re.findall(r"[a-z]+|\d+(?:\.\d+)?",t.lower())
def run(v):
 key=f"{v['year']}-{v['playlist_order']:02}";dest=A/(key+'-full.json');started=time.time()
 if dest.exists():
  old=json.loads(dest.read_text())
  if old.get('status')=='full_audio_automatically_compared':emit({'lecture':key,'status':'reused_completed'});return old
 result={'lecture':key,'video_id':v['id'],'url':v['url'],'status':'started','manual_listening_complete':False}
 try:
  opts={'quiet':True,'no_warnings':True,'logger':Quiet(),'skip_download':True,'socket_timeout':20,'retries':0,'extractor_retries':0,'noprogress':True}
  with yt_dlp.YoutubeDL(opts) as y:info=y.extract_info(v['url'],download=False)
  assert info['channel_id']=='UCBa5G_ESCn8Yd4vw5U-gIcg','Unexpected publisher'
  candidates=[f for f in info['formats'] if f.get('vcodec')=='none' and f.get('ext')=='m4a' and f.get('protocol')=='https' and (f.get('language') or 'en').startswith('en') and 'dub' not in f.get('format_note','').lower()]
  candidates.sort(key=lambda f:('original' not in f.get('format_note','').lower(),f.get('abr') or 999))
  f=candidates[0];audio=M/(key+'.m4a')
  if not audio.exists():
   dl={**opts,'skip_download':False,'format':f['format_id'],'http_chunk_size':1048576,'outtmpl':str(M/(key+'.%(ext)s')),'max_filesize':100*1024*1024,'overwrites':False}
   with yt_dlp.YoutubeDL(dl) as y:y.download([v['url']])
  assert audio.exists() and audio.stat().st_size>100000
  result.update(audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),audio_bytes=audio.stat().st_size,duration_seconds=info['duration'],format_id=f['format_id'],format_note=f.get('format_note'),language=f.get('language'),publisher=info['channel_id'])
  emit({'lecture':key,'status':'audio_acquired','duration':info['duration']})
  segs,_=model.transcribe(str(audio),language='en',beam_size=1,temperature=0,condition_on_previous_text=False,vad_filter=True)
  items=[]
  for s in segs:
   items.append({'start':s.start,'duration':s.end-s.start,'text':s.text.strip(),'avg_logprob':s.avg_logprob,'no_speech_prob':s.no_speech_prob})
   if len(items)%600==0:emit({'lecture':key,'status':'transcribing','audio_seconds':round(s.end),'elapsed':round(time.time()-started)})
  caption=next((W/'transcripts').glob(key+'-*.json'));c=json.loads(caption.read_text())['segments']
  windows=[]
  for t in range(0,int(info['duration']),30):
   end=min(t+30,info['duration']);x=' '.join(z['text'] for z in c if t<=z['start']<end);y=' '.join(z['text'] for z in items if t<=z['start']<end)
   if not(x or y):continue
   score=difflib.SequenceMatcher(None,toks(x),toks(y),autojunk=False).ratio()
   low=[z for z in items if t<=z['start']<end and (z['avg_logprob']<-.75 or z['no_speech_prob']>.5)]
   windows.append({'start':t,'end':end,'similarity':round(score,4),'caption':x,'asr':y,'low_confidence_segments':len(low)})
  atomic(A/(key+'-comparison.json'),{'lecture':key,'windows':windows})
  result.update(status='full_audio_automatically_compared',caption_file_sha256=hashlib.sha256(caption.read_bytes()).hexdigest(),model='faster-whisper base.en int8 CPU beam1',segments=items,asr_last_end=items[-1]['start']+items[-1]['duration'],comparison_windows=len(windows),windows_below_075=sum(w['similarity']<.75 for w in windows),elapsed_seconds=round(time.time()-started),warning='Full-audio machine comparison only. Caption and ASR may both be wrong; flagged windows require source-bound review.')
 except Exception as e:result.update(status='failed',error_type=type(e).__name__,elapsed_seconds=round(time.time()-started))
 atomic(dest,result);emit({k:v for k,v in result.items() if k not in ['segments','warning','url','publisher','format_note','audio_sha256','caption_file_sha256']});return result
videos=[v for v in json.loads((W/'transcripts/index.json').read_text()) if v['status']=='extracted']
# Newest course first; both years are processed. Two jobs share at most four CPU worker threads.
videos.sort(key=lambda v:(-v['year'],v['playlist_order']))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,videos))
atomic(A/'batch-summary.json',{'checked_date':'2026-10-22','lectures':[{k:v for k,v in r.items() if k!='segments'} for r in results],'manual_listening_complete':False})
emit({'batch_finished':True,'complete':sum(r['status']=='full_audio_automatically_compared' for r in results),'failed':sum(r['status']=='failed' for r in results)})
