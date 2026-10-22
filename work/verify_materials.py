import hashlib,importlib.util,json,math,pathlib,subprocess,zipfile
from html.parser import HTMLParser
ROOT=pathlib.Path(__file__).parent.parent;OUT=ROOT/'outputs';W=ROOT/'work'
catalog=json.loads((OUT/'CS336-来源索引.json').read_text())
for f in catalog['files']:
 ext='pdf' if f['url'].endswith('.pdf') else 'py';p=W/'sources'/f'{f["year"]}-lecture-{f["number"]:02}.{ext}'
 assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
for f in catalog['assignment_files']:
 assert hashlib.sha256((W/'sources'/f['name']).read_bytes()).hexdigest()==f['sha256']
assert sum(len(x['lectures']) for x in catalog['years'].values())==57
videos=json.loads((W/'transcripts/index.json').read_text());uploaded=[v for v in videos if v['status']=='extracted']
assert len(videos)==35 and len(uploaded)==34
count=0
for v in uploaded:
 p=W/'transcripts'/f'{v["year"]}-{v["playlist_order"]:02}-{v["id"]}.json';d=json.loads(p.read_text());s=d['segments']
 blob=json.dumps(s,ensure_ascii=False,separators=(',',':')).encode()
 assert hashlib.sha256(blob).hexdigest()==v['segments_sha256'] and len(s)==v['segments_count'];count+=len(s)
assert count==54825
asr=json.loads((W/'transcripts/2026-08-local-asr.json').read_text());assert asr['status']=='local_asr_unverified' and len(asr['segments'])==1702
assert asr['source_duration']-asr['last_end']<15
assert all(s['text'].strip() and math.isfinite(s['start']) and s['start']>=0 and s['duration']>=0 for s in asr['segments'])
assert all(b['start']>=a['start'] for a,b in zip(asr['segments'],asr['segments'][1:]))
audio=json.loads((W/'missing-audio/source-original.json').read_text());assert audio['success'] and audio['language'].startswith('en') and 'original' in audio['format_note']
assert hashlib.sha256((W/'missing-audio/2026-08-original.m4a').read_bytes()).hexdigest()==audio['sha256']
review=json.loads((OUT/'CS336-逐讲对照记录.json').read_text());assert len(review['lecture_entries'])==35 and len(review['frame_checks'])==158
for f in review['frame_checks']:
 p=ROOT/f['local_evidence_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==f['frame_sha256']
nav=json.loads((OUT/'CS336-课堂片段索引.json').read_text());assert nav['segments_accounted']==56527 and nav['window_count']==1131
assert sum(len(x['windows']) for x in nav['transcripts'])==1131
assert len(nav['transcripts'])==35 and sum(x['source_type']=='local_asr_unverified' for x in nav['transcripts'])==1
spec=importlib.util.spec_from_file_location('cpu_demos',OUT/'CS336-教学小实验.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
actual_demos=module.run_all()
assert actual_demos==json.loads((OUT/'CS336-实验结果.json').read_text())
class Parse(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.refs=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  if 'id' in d:self.ids.append(d['id'])
  if t=='a' and d.get('href','').startswith('#'):self.refs.append(d['href'][1:])
for name in ['CS336-AI陪学资料.html','CS336-AI陪学目录.html']:
 p=Parse();p.feed((OUT/name).read_text());assert len(p.ids)==len(set(p.ids)) and all(r in p.ids for r in p.refs)
with zipfile.ZipFile(OUT/'CS336-AI陪学资料包.zip') as z:
 assert z.testzip() is None and len(z.namelist())==38
 names={pathlib.PurePosixPath(n).name for n in z.namelist()}
 assert '学习进度.md' not in names
 assert {'换电脑继续学习.md','学习进度模板.md'} <= names
 for name in z.namelist():
  path=pathlib.PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts
  assert z.read(name)==(OUT/path.name).read_bytes()
  assert path.suffix not in {'.wav','.mp4','.m4a','.pdf','.png'}
assert catalog['learning_materials']['full_caption_text_review_complete'] and not catalog['learning_materials']['full_audio_listening_review_complete'] and not catalog['learning_materials']['visual_review_complete']
ledger=json.loads((OUT/'CS336-全文校读记录.json').read_text())
assert len(ledger['entries'])==35 and sum(x['segments'] for x in ledger['entries'])==56527
for x in ledger['entries']:
 path=next((W/'transcripts').glob(x['lecture']+'-*.json'))
 assert hashlib.sha256(path.read_bytes()).hexdigest()==x['source_file_sha256']
 assert x['previous_context_segments']+x['chronological_remaining_segments']==x['segments']
 assert x['full_caption_text_review_complete'] and not x['full_audio_listening_review_complete'] and x['notes']
 end=max(t['start']+t['duration'] for t in json.loads(path.read_text())['segments'])
 assert all(0<=n['start']<n['end']<=end+5 and n['finding'] for n in x['notes'])
assert sum(len(x['notes']) for x in ledger['entries'])==247
assert len(actual_demos)==10
from review_notes import FINDINGS
assert len(FINDINGS)==334
pdf_inventory=json.loads((OUT/'CS336-PDF逐页核验记录.json').read_text())
assert pdf_inventory['total_pages']==1889 and len(pdf_inventory['pages'])==1889
assert sum(x['visual_overview_complete'] for x in pdf_inventory['pages'] if 'visual_overview_complete' in x)==1889
assert sum(x['semantic_review_complete'] for x in pdf_inventory['pages'])<1889
assert len({(x['file'],x['page']) for x in pdf_inventory['pages']})==1889
for x in pdf_inventory['pages']:
 txt=W/'pdf-audit'/pathlib.Path(x['file']).stem/f"{x['page']:03}.txt"
 assert hashlib.sha256(txt.read_bytes()).hexdigest()==x['text_sha256']
audio_review=json.loads((OUT/'CS336-音频核验记录.json').read_text())
assert audio_review['source_audio_sha256']==audio['sha256']
assert not audio_review['full_audio_listening_complete'] and not audio_review['official_caption_obtained']
assert hashlib.sha256((W/'audio-audit/2026-08-small-en.json').read_bytes()).hexdigest()==audio_review['second_result_sha256']
assert catalog['verification_completion_status']=='publicly_accessible_audit_complete_with_documented_gaps'
report_pdf=json.loads((OUT/'CS336-PDF内容核对.json').read_text())
for r in report_pdf['reviews']:
 assert hashlib.sha256((W/'sources'/r['file']).read_bytes()).hexdigest()==r['source_pdf_sha256']
 for page in r['rendered_pages']:
  path=ROOT/page['path']
  assert hashlib.sha256(path.read_bytes()).hexdigest()==page['sha256']
traces=json.loads((W/'traces/index.json').read_text())
assert len(traces)==21
assert review['counts']['full_caption_text_reviews']==35 and review['counts']['full_reading_findings']==247
# Source-bound added image, PDF and original-audio evidence.
assert len(report_pdf['reviews'])==42 and sum(len(r['pages_viewed']) for r in report_pdf['reviews'])==1889
for x in pdf_inventory['pages']:
 assert hashlib.sha256((ROOT/x['render_path']).read_bytes()).hexdigest()==x['render_sha256']
 assert any(r['file']==x['file'] and any(x['page'] in f['pages'] for f in r['findings']) for r in report_pdf['reviews'])
fig=json.loads((OUT/'CS336-讲义图像核验.json').read_text())
assert len(fig['sources'])==312 and len(fig['unique_images'])==231
assert len(fig['animation_frames'])==266
from PIL import Image
for idx in (22,63):
 x=fig['unique_images'][idx];im=Image.open(ROOT/x['file']);frames=[f for f in fig['animation_frames'] if f['unique_image']==idx]
 assert im.n_frames==len(frames)
 for f in frames:
  im.seek(f['frame']);rgba=im.convert('RGBA');bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(rgba)
  assert hashlib.sha256(bg.convert('RGB').tobytes()).hexdigest()==f['pixel_sha256']

for x in fig['sources']:assert hashlib.sha256((ROOT/x['file']).read_bytes()).hexdigest()==x['sha256'] and x['success']
assert set(n for r in fig['notes'] for n in r['unique_indices'])==set(range(231))
values=json.loads((OUT/'CS336-演示数组核验.json').read_text())
assert len(values['records'])==405 and values['checks']['finite_compound_values']==405 and len(values['notes'])==12
for x in values['records']:assert hashlib.sha256((W/'traces'/x['file']).read_bytes()).hexdigest()==x['source_trace_sha256']
audios=json.loads((OUT/'CS336-原始音频比对.json').read_text());assert len(audios['lectures'])==35 and not audios['manual_listening_complete']
for x in audios['lectures']:
 if x['lecture']=='2026-08':continue
 k=x['lecture'];assert x['textual_candidate_review_complete'] and not x['manual_listening_complete'] and x['findings']
 assert hashlib.sha256((W/'audio-audit/original'/(k+'.m4a')).read_bytes()).hexdigest()==x['audio_sha256']
 for suffix,field in [('-full.json','source_result_sha256'),('-comparison.json','comparison_sha256')]:assert hashlib.sha256((W/'audio-audit'/(k+suffix)).read_bytes()).hexdigest()==x[field]
 assert hashlib.sha256((W/'audio-audit/repairs'/(k+'.json')).read_bytes()).hexdigest()==x['repair_sha256']
 assert len(x['regions'])==x['repair_region_count']
from batch_audio_notes import BATCH_AUDIO_NOTES
assert len(BATCH_AUDIO_NOTES)==34
report={'result':'pass','core_source_hashes':51,'assignment_source_hashes':17,'schedule_entries':57,'uploaded_captions':34,'local_automatic_transcriptions':1,'text_segments':56527,'navigation_windows':1131,'lecture_notes':35,'inspected_frames':158,'executed_cpu_demos':10,'package_members':len(names),'full_caption_text_reviews':35,'full_reading_findings':247,'pdf_pages_enumerated':1889,'pdf_visual_overview_pages':1889,'publicly_accessible_audit_complete':True,'full_classroom_capture_complete':False,'limits':['Full caption text reading is complete; full audio listening is not complete','Not a full frame-by-frame verification','HTML checked structurally; no browser visual layout acceptance','No GPU performance reproduction','Local ASR is not official caption evidence']}
report.update(original_audio_recordings=35,flagged_audio_regions=audios['counts']['flagged_regions_reviewed'],trace_images=231,animation_frames=266,trace_compound_values=405,pdf_content_findings=sum(len(r['findings']) for r in report_pdf['reviews']))
(W/'verification-final.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
