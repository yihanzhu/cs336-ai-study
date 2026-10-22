"""Surface disagreements; similarity is a review aid, never a correctness certificate."""
import difflib,json,pathlib,re,hashlib
from audio_findings import AUDIO_FINDINGS,AUDIO_REVIEW_SOURCE_SHA256
W=pathlib.Path(__file__).resolve().parent;O=W.parent/'outputs';A=W/'audio-audit'
a=json.loads((W/'transcripts/2026-08-local-asr.json').read_text());b=json.loads((A/'2026-08-small-en.json').read_text())
assert b['status']=='automatic_transcription_complete_not_listened'
assert hashlib.sha256((A/'2026-08-small-en.json').read_bytes()).hexdigest()==AUDIO_REVIEW_SOURCE_SHA256,'Audio evidence changed; review must be repeated before reusing the findings.'
def tokens(s):return re.findall(r"[a-z]+|\d+(?:\.\d+)?",s.lower())
def chunk(segments,start,end):return ' '.join(x['text'] for x in segments if start<=x['start']<end)
windows=[]
for start in range(0,4810,30):
 end=min(start+30,4810);x=chunk(a['segments'],start,end);y=chunk(b['segments'],start,end)
 if not(x or y):continue
 sim=difflib.SequenceMatcher(None,tokens(x),tokens(y),autojunk=False).ratio()
 windows.append({'start':start,'end':end,'normalized_word_similarity':round(sim,4),'base_en':x,'small_en':y})
(A/'comparison-windows.json').write_text(json.dumps(windows,ensure_ascii=False,indent=2))
low=sorted(windows,key=lambda x:x['normalized_word_similarity'])[:35]
(A/'largest-differences.txt').write_text('\n\n'.join(f"@ {r['start']}–{r['end']} similarity={r['normalized_word_similarity']}\nBASE: {r['base_en']}\nSMALL: {r['small_en']}" for r in low))
report={'lecture':'2026-08','status':'second_automatic_transcription_complete_largest_disagreements_reviewed','scope':'Two Whisper sizes on full original audio. Correlated recognizers can agree on errors; no manual-listening sign-off. Similarity is navigation only.','source_duration_seconds':4810,'first_model':'base.en','second_model':'small.en','first_segments':len(a['segments']),'second_segments':len(b['segments']),'second_last_end':b['last_end'],'source_audio_sha256':b['source_audio_sha256'],'second_result_sha256':hashlib.sha256((A/'2026-08-small-en.json').read_bytes()).hexdigest(),'window_count':len(windows),'windows_below_0_7':sum(x['normalized_word_similarity']<.7 for x in windows),'largest_disagreement_windows_read':35,'findings':AUDIO_FINDINGS,'review_limits':['35 largest-disagreement windows read as text; the remaining windows were compared mechanically, not independently listened.','The recognizers belong to the same Whisper family. Agreement is not correctness.','Unresolved questions remain unresolved; other 34 lecture audio tracks are not audited by this run.'],'full_audio_listening_complete':False,'official_caption_obtained':False}
(O/'CS336-音频核验记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
