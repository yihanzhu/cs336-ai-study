import hashlib,json,pathlib,subprocess
from batch_audio_notes import BATCH_AUDIO_NOTES
W=pathlib.Path(__file__).parent;O=W.parent/'outputs';A=W/'audio-audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.load(open(A/'batch-summary.json'));entries=[]
assert len(summary['lectures'])==34
for row in summary['lectures']:
 k=row['lecture'];p=A/(k+'-full.json');d=json.load(open(p));rpath=A/'repairs'/(k+'.json');rep=json.load(open(rpath));comp=json.load(open(A/(k+'-comparison.json')))
 assert d['status']=='full_audio_automatically_compared' and sha(A/'original'/(k+'.m4a'))==d['audio_sha256']
 assert rep['source_audio_sha256']==d['audio_sha256'] and rep['primary_asr_result_sha256']==sha(p)
 assert k in BATCH_AUDIO_NOTES and 'all' in str(BATCH_AUDIO_NOTES[k]['windows_read'])
 for w in comp['windows']:
  if w['similarity']<.75 or w['low_confidence_segments']:
   assert any(x['start']<=w['start'] and x['end']>=w['end'] for x in rep['regions']),(k,w['start'])
 physical=float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(A/'original'/(k+'.m4a'))],capture_output=True,text=True,check=True).stdout)
 assert abs(physical-d['duration_seconds'])<2,(k,physical,d['duration_seconds'])
 out={k:v for k,v in d.items() if k not in ['segments','warning']}
 out.update(track_verification=('publisher_default_English_track_no_original_label' if 'original' not in d.get('format_note','').lower() else 'publisher_English_original_labeled_track'),physical_audio_duration_seconds=physical,source_result_sha256=sha(p),comparison_sha256=sha(A/(k+'-comparison.json')),repair_sha256=sha(rpath),repair_region_count=len(rep['regions']),textual_candidate_review_complete=True,manual_listening_complete=False,findings=[{'start':a,'end':b,'finding':t} for a,b,t in BATCH_AUDIO_NOTES[k]['findings']],regions=[{'start':x['start'],'end':x['end'],'reasons':x['reasons'],'caption_text_sha256':hashlib.sha256(x['caption'].encode()).hexdigest(),'second_asr_text_sha256':hashlib.sha256(' '.join(z['text'] for z in x['small_en_segments']).encode()).hexdigest()} for x in rep['regions']],out_of_duration_asr_segments=sum(s['start']>=d['duration_seconds'] or s['start']+s['duration']>d['duration_seconds']+1 for s in d['segments']),effective_end_seconds=min(d['duration_seconds'],d['asr_last_end']))
 track_meta=A/(k+'-track-check.json')
 if track_meta.exists():out['track_metadata_sha256']=sha(track_meta)
 entries.append(out)
old=json.load(open(O/'CS336-音频核验记录.json'))
entries.append({'lecture':'2026-08','source_audio_sha256':old['source_audio_sha256'],'legacy_ledger':'CS336-音频核验记录.json','manual_listening_complete':False,'method':'Full original audio twice transcribed; 161 comparison windows, largest 35 windows textually reviewed, remaining inaudible questions documented.','findings':old['findings']})
assert len(entries)==35
(O/'CS336-原始音频比对.json').write_text(json.dumps({'date':'2026-10-22','lectures':entries,'manual_listening_complete':False,'scope':'Full original English audio machine transcription/caption comparison with all flagged region second-model candidates textually reviewed (34 recordings), plus existing full dual-ASR L08 audit. Not an independent human listening signoff; both Whisper variants may share errors. Out-of-duration ASR text is invalid and excluded from teaching.','counts':{'recordings':35,'new_full_audio_machine_comparisons':34,'flagged_regions_reviewed':sum(x.get('repair_region_count',0) for x in entries),'comparison_windows':sum(x.get('comparison_windows',0) for x in entries)+161,'duration_seconds':sum(x.get('duration_seconds',0) for x in entries)+4810}},ensure_ascii=False,indent=2))
print('35 original-audio source-bound audit entries built')
