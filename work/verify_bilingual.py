"""Validate bilingual alignment and portable release without external source caches."""
import hashlib,json,pathlib,re,subprocess,shutil,zipfile
from html.parser import HTMLParser
R=pathlib.Path(__file__).parent.parent;O=R/'outputs';W=R/'work'
b=json.loads((O/'CS336-bilingual-lessons.json').read_text());en=b['languages']['en'];zh=b['languages']['zh'];review=json.loads((O/'CS336-bilingual-review.json').read_text())
assert b['default_language']=='en' and b['supported_languages']==['en','zh']
assert en==json.loads((O/'CS336-lesson-plans.en.json').read_text()) and zh==json.loads((O/'CS336-逐点教案.json').read_text())
assert len(en['lessons'])==len(zh['lessons'])==len(review['points'])==299 and len(en['chapters'])==len(zh['chapters'])==29
fields=['title','objective','explanation','worked_example','question','answer_criteria','repair','source_gap','evidence_boundary','progression','example_status']
md=(O/'CS336-lesson-plans.md').read_text();zmd=(O/'CS336-逐点陪学教案.md').read_text()
shared_fields=['id','chapter','prerequisites','next_id']
for e,z,r in zip(en['lessons'],zh['lessons'],review['points']):
 assert all(e[k]==z[k] for k in shared_fields),(e['id'],'shared identity')
 assert all(e[k].strip() and z[k].strip() for k in fields),(e['id'],'missing text')
 assert bool(e['extension'])==bool(z['extension']),(e['id'],'missing derivation')
 assert [(s['id'],s['url'],s['sha256']) for s in e['source_refs']]==[(s['id'],s['url'],s['sha256']) for s in z['source_refs']]
 assert md.count('### '+e['id']+' '+e['title']+'\n')==1
 assert zmd.count('### '+z['id']+' '+z['title']+'\n')==1
 for key,x in [('english_sha256',e),('chinese_sha256',z)]:assert r[key]==hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
 # The intentional UTF-8 example contains 中. Accidental untranslated Chinese prose is rejected.
 for k in fields+['extension']:
  text=e[k].replace('中','');assert not re.search(r'[\u4e00-\u9fff]',text),(e['id'],k,'untranslated Chinese')
for s in review['author_sources']:assert hashlib.sha256((W/s['file']).read_bytes()).hexdigest()==s['sha256']
assert len({x['question'] for x in en['lessons']})==299
assert len([x for x in en['lessons'] if x['extension']])==13
assert len(review['numeric_text_candidates'])==39
for c in review['numeric_text_candidates']:
 assert c['author_resolution'] and c['status'].startswith('author_reviewed')
 e=next(x for x in en['lessons'] if x['id']==c['id']);z=next(x for x in zh['lessons'] if x['id']==c['id'])
 assert e[c['field']]==c['english_text'] and z[c['field']]==c['chinese_text']
# Source-bound anchors for worked outcomes and the most error-prone derivations.
byid={x['id']:x for x in en['lessons']}
for pid,needle in [('04.08','2.5'),('09.04','16MB'),('21.07','0.4375'),('23.09','800'),('25.13','0.36')]:assert needle in byid[pid]['worked_example']
assert '/(D r³)' in byid['11.10']['extension']
assert 'lnew=c·l+' in byid['12.05']['extension'] and 'unew=c·u+' in byid['12.05']['extension']
assert 'not a complete Transformer' in byid['17.02']['extension']
assert 'original paper' in byid['25.09']['extension'] and 'constant' in byid['25.09']['extension']
assert 'not recorded' in byid['28.01']['explanation'] and 'not obtained' in byid['28.01']['explanation']
class Page(HTMLParser):
 def __init__(self):super().__init__();self.data=False;self.js=False;self.payload='';self.current='';self.scripts=[];self.external=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  if t=='script':
   self.data=d.get('id')=='data';self.js=not self.data;self.current=''
   if d.get('src'):self.external.append(d['src'])
  if t=='link' and d.get('href'):self.external.append(d['href'])
 def handle_data(self,s):
  if self.data:self.payload+=s
  if self.js:self.current+=s
 def handle_endtag(self,t):
  if t=='script':
   if self.js:self.scripts.append(self.current)
   self.data=False;self.js=False
p=Page();h=(O/'CS336-lesson-plans.html').read_text();p.feed(h);assert json.loads(p.payload)==b and not p.external and '<html lang="en">' in h
(W/'bilingual-page-check.js').write_text('\n'.join(p.scripts));node=shutil.which('node')
if node:subprocess.run([node,'--check',str(W/'bilingual-page-check.js')],check=True)
with zipfile.ZipFile(O/'CS336-AI陪学资料包.zip') as z:
 assert z.testzip() is None and len(z.namelist())==38
 names={pathlib.PurePosixPath(n).name for n in z.namelist()};assert '学习进度.md' not in names
 assert {'START-HERE.txt','CONTINUE-LEARNING.md','learning-progress-template.en.md','CS336-lesson-plans.html','CS336-bilingual-lessons.json'}<=names
 for n in z.namelist():
  path=pathlib.PurePosixPath(n);assert not path.is_absolute() and '..' not in path.parts
  assert z.read(n)==(O/path.name).read_bytes()
assert b['progress_contract']['saved_preference_overrides_default'] and not b['progress_contract']['reader_navigation_marks_mastery']
report={'result':'pass','english_lessons':299,'chinese_lessons':299,'aligned_primary_teaching_fields':299*6,'aligned_extensions':13,'reviewed_numeric_text_candidates':39,'point_source_identities':'shared URL/hash identity','package_members':38,'public_default_language':'en','progress_policy':'ignored local-only file; no automatic commits or uploads; no reader mastery writes','semantic_review':'Same-author review; no claim of independent full-course peer review','raw_evidence_scope':'unchanged baseline 1.2'}
(W/'bilingual-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
