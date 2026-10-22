import hashlib,json,os,pathlib,time
root=pathlib.Path('work');cache=root/'asr-cache';cache.mkdir(exist_ok=True)
os.environ['HF_HOME']=str(cache.resolve())
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
from faster_whisper import WhisperModel
source=json.loads((root/'missing-audio/source-original.json').read_text());assert source['success'] and source['language'].startswith('en')
print('Loading local base.en recognizer',flush=True)
model=WhisperModel('base.en',device='cpu',compute_type='int8',cpu_threads=4,num_workers=1,download_root=str(cache/'models'))
segments,info=model.transcribe(str(root/'missing-audio/2026-08-original.m4a'),language='en',beam_size=1,temperature=0,condition_on_previous_text=False,vad_filter=True)
items=[];started=time.time();p=root/'transcripts/2026-08-local-asr.json'
for s in segments:
 items.append({'start':s.start,'duration':s.end-s.start,'text':s.text.strip(),'avg_logprob':s.avg_logprob,'no_speech_prob':s.no_speech_prob})
 if len(items)%100==0:
  print('segments',len(items),'audio_seconds',round(s.end),'elapsed',round(time.time()-started),flush=True)
  p.write_text(json.dumps({'status':'in_progress','model':'base.en','segments':items},ensure_ascii=False,indent=2))
d={'year':2026,'lecture':8,'video_id':'6-cXp-aOmdg','status':'local_asr_unverified','model':'faster-whisper base.en CPU int8 beam=1','original_audio_sha256':source['sha256'],'source_duration':source['duration'],'segments':items,'segment_count':len(items),'last_end':items[-1]['start']+items[-1]['duration'],'warning':'Local automatic transcription, not an official caption; technical terms and formulas require verification.'}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2));print('FINISHED',len(items),round(d['last_end']),flush=True)
