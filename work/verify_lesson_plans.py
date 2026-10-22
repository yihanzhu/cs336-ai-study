"""Cache-free checks of coverage, source accounting, authored formulas and package contents.
These checks do not certify every semantic claim or replace external review.
"""
import hashlib,json,math,pathlib,re,zipfile,shutil,subprocess
from html.parser import HTMLParser
R=pathlib.Path(__file__).parent.parent;O=R/'outputs'
doc=json.loads((O/'CS336-逐点教案.json').read_text()); rows=doc['lessons'];cov=json.loads((O/'CS336-教案覆盖核对.json').read_text())
expected_hash='96623a851f66975c48b5b82575af71640c85340fbff565d9a2a0643ab1f2e2b0'
identity=[(x['id'],x['title']) for x in rows]
assert hashlib.sha256(json.dumps(identity,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==expected_hash
assert len(cov['author_review_notes'])==29
assert len(rows)==299 and len(doc['chapters'])==29 and len(set(x['id'] for x in rows))==299
keys=['objective','explanation','worked_example','question','answer_criteria','repair']
ids={x['id'] for x in rows};md=(O/'CS336-逐点陪学教案.md').read_text()
for i,x in enumerate(rows):
 assert all(isinstance(x[k],str) and x[k].strip() for k in keys)
 assert all(p in ids for p in x['prerequisites'])
 assert x['next_id']==(rows[i+1]['id'] if i+1<len(rows) else None)
 assert x['source_refs'] and all(r['url'].startswith('https://') for r in x['source_refs'])
 assert x['source_gap'] and x['example_status'] and x['progression']
 assert md.count('### '+x['id']+' '+x['title']+'\n')==1
 assert cov['points'][i]['sha256']==hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
for x in cov['author_sources']:assert hashlib.sha256((R/'work'/x['file']).read_bytes()).hexdigest()==x['sha256']
assert len({r['question'] for r in rows})==299 and len({r['explanation'] for r in rows})==299
# Each referenced acquired lecture SHA must match the retained public-source ledger.
catalog=json.loads((O/'CS336-来源索引.json').read_text());lecture_shas={f"{f['year']}-{f['number']:02}":f['sha256'] for f in catalog['files']}
for x in rows:
 for s in x['source_refs']:
  if s['id'] in lecture_shas:assert s['sha256']==lecture_shas[s['id']]
# Detect accidentally circular foundation recommendations.
byid={x['id']:x for x in rows}
def walk(pid,path):
 assert pid not in path,('circular prerequisite',path,pid)
 for p in byid[pid]['prerequisites']:walk(p,path+[pid])
for pid in ids:walk(pid,[])
class Page(HTMLParser):
 def __init__(self):super().__init__();self.data=False;self.js=False;self.payload='';self.scripts=[];self.current='';self.external=[]
 def handle_starttag(self,t,a):
  d=dict(a)
  if t=='script':
   self.data=d.get('id')=='data';self.js=not self.data;self.current=''
   if d.get('src'):self.external.append(d['src'])
  if t=='link' and d.get('href'):self.external.append(d['href'])
 def handle_data(self,d):
  if self.data:self.payload+=d
  if self.js:self.current+=d
 def handle_endtag(self,t):
  if t=='script':
   if self.js:self.scripts.append(self.current)
   self.data=False;self.js=False
p=Page();p.feed((O/'CS336-lesson-plans.html').read_text());assert json.loads(p.payload)['languages']['zh']==doc and not p.external
(R/'work/lesson-page-check.js').write_text('\n'.join(p.scripts))
node=shutil.which('node')
if node:subprocess.run([node,'--check',str(R/'work/lesson-page-check.js')],check=True)
# Independent arithmetic and calculus checks of key worked examples.
checks={}
def check(name,ok):assert ok,name;checks[name]='pass'
check('sequence_probability',math.isclose(.5*.4*.2,.04))
check('weighted_microbatch_mean',math.isclose((2*1+6*3)/8,2.5))
check('gradient_clip',math.isclose(math.hypot(.6,.8),1))
check('roofline',min(1000,10*20)==200)
check('ring_per_rank_send',2*(4-1)*100/4==150)
check('lsh_candidate_probability',math.isclose(1-(1-.5**2)**2,.4375))
check('pass_at_two_under_independence',math.isclose(1-(1-.2)**2,.36))
check('lora_parameter_count',4*(100+100)==800)
# RMSNorm analytical input gradient versus centered finite differences, nontrivial gain/upstream.
x=[1.3,-.7,2.1];g=[.8,1.2,-.5];a=[.4,-.2,.9];eps=1e-5;n=len(x)
def loss(v):
 r=math.sqrt(sum(z*z for z in v)/n+eps);return sum(ai*gi*xi/r for ai,gi,xi in zip(a,g,v))
r=math.sqrt(sum(z*z for z in x)/n+eps);dot=sum(ai*gi*xi for ai,gi,xi in zip(a,g,x));errs=[]
for j in range(n):
 xp=x.copy();xm=x.copy();xp[j]+=1e-6;xm[j]-=1e-6
 analytic=a[j]*g[j]/r-x[j]*dot/(n*r**3);numeric=(loss(xp)-loss(xm))/2e-6;errs.append(abs(analytic-numeric))
check('rmsnorm_backward_finite_difference',max(errs)<1e-8)
# Online stable softmax block merging versus whole-row weighted mean.
s=[1000.,1002.,999.,1005.,1001.];v=[2.,-1.,4.,.5,-3.]
big=max(s);ew=[math.exp(z-big) for z in s];target=sum(w*z for w,z in zip(ew,v))/sum(ew);errs2=[]
for bs in [1,2,3,5]:
 m=None;l=0.;u=0.
 for start in range(0,len(s),bs):
  sb=s[start:start+bs];vb=v[start:start+bs];mn=max(sb) if m is None else max(m,max(sb));c=0 if m is None else math.exp(m-mn)
  w=[math.exp(z-mn) for z in sb];l=c*l+sum(w);u=c*u+sum(wi*vi for wi,vi in zip(w,vb));m=mn
 errs2.append(abs(u/l-target))
check('online_softmax_block_merge',max(errs2)<1e-12)
check('chain_rule',2*(2*1-3)*2==-4)
check('discrete_stability_counterexample',abs(1+3*(-1))>1)
with zipfile.ZipFile(O/'CS336-AI陪学资料包.zip') as z:
 names={pathlib.PurePosixPath(n).name for n in z.namelist()}
 assert len(names)==38 and '学习进度.md' not in names
 assert {'CS336-逐点陪学教案.md','CS336-逐点陪学教案.html','CS336-逐点教案.json','CS336-教案覆盖核对.json'} <= names
 for n in z.namelist():assert z.read(n)==(O/pathlib.PurePosixPath(n).name).read_bytes()
report={'result':'pass','original_curriculum_identity_sha256':expected_hash,'lessons':299,'chapters':29,'authored_fields_per_lesson':6,'unique_questions':299,'unique_explanations':299,'extensions':sum(bool(x['extension']) for x in rows),'numeric_checks':checks,'rmsnorm_max_gradient_error':max(errs),'online_softmax_max_error':max(errs2),'package_members':len(names),'personal_progress_in_share_package':False,'semantic_review':'Author review; not independent external peer review; structural checks do not prove all semantic accuracy','browser_checks':'Recorded separately when executed'}
(R/'work/lesson-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
