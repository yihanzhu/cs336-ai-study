"""Second-model review candidates for inaudible and discrepant regions; never auto-accept wording."""
import hashlib,json,os,pathlib,re,time,subprocess
W=pathlib.Path(__file__).resolve().parent;A=W/'audio-audit';R=A/'repairs';R.mkdir(exist_ok=True)
os.environ['HF_HOME']=str(W/'asr-cache');os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from faster_whisper import WhisperModel
model=WhisperModel('small.en',device='cpu',compute_type='int8',cpu_threads=2,num_workers=1,download_root=str(W/'asr-cache/models'),local_files_only=True)
def work(p):
 d=json.loads(p.read_text());key=d['lecture'];dest=R/(key+'.json')
 if dest.exists() or d['status']!='full_audio_automatically_compared':return
 audio=A/'original'/(key+'.m4a');captions=json.loads(next((W/'transcripts').glob(key+'-*.json')).read_text())['segments'];windows=json.loads((A/(key+'-comparison.json')).read_text())['windows'];regions=[]
 for s in captions:
  if re.search(r'inaudible|unintelligible',s['text'],re.I):regions.append((max(0,s['start']-7),min(d['duration_seconds'],s['start']+s['duration']+8),'inaudible'))
 for w in windows:
  if w['similarity']<.75 or w['low_confidence_segments']:
   regions.append((max(0,w['start']-5),min(d['duration_seconds'],w['end']+5),'disagreement_or_low_confidence'))
 regions.sort();merged=[]
 for x,y,why in regions:
  if merged and x<=merged[-1][1]+2 and y-merged[-1][0]<=120:merged[-1][1]=max(y,merged[-1][1]);merged[-1][2].append(why)
  else:merged.append([x,y,[why]])
 rows=[];tmp=R/(key+'-clip.wav')
 for x,y,reasons in merged:
  subprocess.run(['ffmpeg','-v','error','-ss',str(x),'-i',str(audio),'-t',str(y-x),'-ar','16000','-ac','1','-y',str(tmp)],capture_output=True,check=True)
  segs,_=model.transcribe(str(tmp),language='en',beam_size=3,temperature=0,condition_on_previous_text=False,vad_filter=True)
  out=[{'start':x+s.start,'end':x+s.end,'text':s.text.strip(),'avg_logprob':s.avg_logprob,'no_speech_prob':s.no_speech_prob} for s in segs]
  rows.append({'start':x,'end':y,'reasons':sorted(set(reasons)),'small_en_segments':out,'caption':' '.join(s['text'] for s in captions if x<=s['start']<y),'base_en':' '.join(s['text'] for s in d['segments'] if x<=s['start']<y)})
 if tmp.exists():tmp.unlink()
 result={'lecture':key,'source_audio_sha256':d['audio_sha256'],'primary_asr_result_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'model':'small.en CPU int8 beam3','regions':rows,'status':'second_model_candidates_not_human_listened','manual_listening_complete':False}
 dest.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({'lecture':key,'second_model_regions':len(rows)}),flush=True)
while True:
 files=sorted(A.glob('*-full.json'))
 for p in files:work(p)
 if (A/'batch-summary.json').exists() and all((R/(p.name[:7]+'.json')).exists() or json.loads(p.read_text())['status']!='full_audio_automatically_compared' for p in A.glob('*-full.json')):break
 time.sleep(5)
print('ALL_REPAIR_CANDIDATES_READY',flush=True)
