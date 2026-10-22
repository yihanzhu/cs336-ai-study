import collections,concurrent.futures,hashlib,json,pathlib,urllib.request
from PIL import Image,ImageDraw
W=pathlib.Path('work');D=W/'trace-figures';D.mkdir(exist_ok=True);sources=json.load(open(W/'traces/index.json'));jobs={}
for src in sources:
 if not src['path'].endswith('.json'):continue
 p=W/'traces'/(str(src['year'])+'-'+pathlib.Path(src['path']).name)
 for s in json.load(open(p))['steps']:
  for r in s.get('renderings',[]):
   if r['type']!='image':continue
   path=r['data'];url='https://raw.githubusercontent.com/stanford-cs336/'+src['repo']+'/'+src['sha']+'/'+path
   jobs.setdefault(url,{'url':url,'repo_path':path,'lectures':[]})['lectures'].append(p.stem)
rows=[]
def fetch(row):
 row['lectures']=sorted(set(row['lectures']));name=hashlib.sha256(row['url'].encode()).hexdigest()[:20];p=D/name
 try:
  if not p.exists():
   with urllib.request.urlopen(row['url'],timeout=25) as f:data=f.read()
   p.write_bytes(data)
  im=Image.open(p);im.seek(0);row.update(success=True,file=str(p),width=im.width,height=im.height,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 except Exception as e:row.update(success=False,error=type(e).__name__)
 return row
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(fetch,jobs.values()))
(D/'index.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
# De-duplicate identical image bytes. Preserve every source link.
unique={}
for r in rows:
 if r['success']:unique.setdefault(r['sha256'],r)
ordered=list(unique.values());(D/'unique.json').write_text(json.dumps(ordered,ensure_ascii=False,indent=2))
for offset in range(0,len(ordered),4):
 batch=ordered[offset:offset+4];im=Image.new('RGB',(2200,1600),'#ddd');draw=ImageDraw.Draw(im)
 for j,r in enumerate(batch):
  pic=Image.open(r['file']).convert('RGB');pic.thumbnail((1100,750));x=j%2*1100;y=j//2*800;im.paste(pic,(x,y+45));draw.text((x+5,y+5),str(offset+j)+' '+r['lectures'][0]+' '+r['repo_path'][:55],fill='black')
 im.save(D/('sheet-%02d.jpg'%(offset//4+1)),quality=92)
print(json.dumps({'sources':len(rows),'success':sum(r['success'] for r in rows),'unique_images':len(ordered),'failed':[r for r in rows if not r['success']]},ensure_ascii=False),flush=True)
