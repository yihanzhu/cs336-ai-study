import concurrent.futures,hashlib,json,pathlib,urllib.request
out=pathlib.Path('work/traces');out.mkdir(exist_ok=True);jobs=json.loads(pathlib.Path('work/review/trace-manifest.json').read_text())
def get(x):
 year=2025 if x['repo'].startswith('spring2025') else 2026;p=out/f'{year}-{pathlib.PurePosixPath(x["path"]).name}';u=f'https://raw.githubusercontent.com/stanford-cs336/{x["repo"]}/{x["sha"]}/{x["path"]}'
 try:
  if p.exists():b=p.read_bytes()
  else:
   with urllib.request.urlopen(u,timeout=45) as r:b=r.read()
   p.write_bytes(b)
  assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==x['blob_sha']
  result={**x,'year':year,'url':u,'sha256':hashlib.sha256(b).hexdigest(),'success':True}
  if p.suffix=='.json':
   d=json.loads(b);source=d.get('files',{});result['steps']=len(d['steps']);result['functions']=list(dict.fromkeys(s['stack'][-1]['function_name'] for s in d['steps'] if s['stack']));vals=[]
   for s in d['steps']:
    if not s['stack']:continue
    st=s['stack'][-1]
    for k,v in s.get('env',{}).items():
     if (isinstance(v,(int,float,bool,str)) and len(str(v))<180) or (isinstance(v,dict) and v.get('type') in ['int','float','bool','str'] and len(str(v.get('contents','')))<180):
      row={'function':st['function_name'],'line':st['line_number'],'variable':k,'value':v.get('contents') if isinstance(v,dict) else v}
      if not vals or row!=vals[-1]:vals.append(row)
   result['scalar_outputs']=vals
   for name,txt in source.items():
    if '/' not in name and name.endswith('.py'):(out/f'{year}-{p.stem}-{name}').write_text(txt)
  return result
 except Exception as e:return {**x,'success':False,'error':type(e).__name__}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(get,jobs))
(out/'index.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2));print(json.dumps({'files':len(rows),'success':sum(x['success'] for x in rows),'scalar_outputs':sum(len(x.get('scalar_outputs',[])) for x in rows),'failed':[x for x in rows if not x['success']]},indent=2))
