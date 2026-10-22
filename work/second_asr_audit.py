"""Independent small.en transcription of L08, preserving base.en and official-caption gap."""
import os,pathlib,json,time,hashlib
W=pathlib.Path(__file__).resolve().parent
os.environ['HF_HOME']=str(W/'asr-cache');os.environ['HF_HUB_DISABLE_XET']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from faster_whisper import WhisperModel
out=W/'audio-audit';out.mkdir(exist_ok=True)
source=W/'missing-audio/2026-08-original.m4a';started=time.time()
print('Loading small.en for independent full-audio automatic comparison',flush=True)
try:
 model=WhisperModel('small.en',device='cpu',compute_type='int8',cpu_threads=4,num_workers=1,download_root=str(W/'asr-cache/models'))
 segments,info=model.transcribe(str(source),language='en',beam_size=1,temperature=0,condition_on_previous_text=False,vad_filter=True)
 items=[]
 for s in segments:
  items.append({'start':s.start,'duration':s.end-s.start,'text':s.text.strip(),'avg_logprob':s.avg_logprob,'no_speech_prob':s.no_speech_prob})
  if len(items)%100==0:
   print(json.dumps({'segments':len(items),'audio_seconds':round(s.end),'elapsed_seconds':round(time.time()-started)}),flush=True)
   (out/'2026-08-small-en-partial.json').write_text(json.dumps({'status':'running','segments':items},ensure_ascii=False))
 record={'status':'automatic_transcription_complete_not_listened','model':'faster-whisper small.en CPU int8 beam=1','source_audio_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'source_seconds':4810,'last_end':items[-1]['start']+items[-1]['duration'],'segments':items,'elapsed_seconds':round(time.time()-started),'warning':'A second machine transcription is not manual listening, not official captions, and does not resolve inaudible questions by itself.'}
 (out/'2026-08-small-en.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print('COMPLETE',len(items),round(record['last_end']),flush=True)
except Exception as e:
 (out/'2026-08-small-en-error.json').write_text(json.dumps({'status':'failed','error_type':type(e).__name__,'elapsed_seconds':round(time.time()-started)}))
 print('FAILED',type(e).__name__,flush=True);raise
