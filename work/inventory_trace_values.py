import collections,hashlib,json,math,pathlib
W=pathlib.Path('work');records=[]
for p in sorted((W/'traces').glob('20*-lecture_*.json')):
 seen={}
 for s in json.load(open(p))['steps']:
  for name,v in s.get('env',{}).items():
   key=json.dumps([name,v],sort_keys=True)
   if key in seen:continue
   seen[key]=True;kind=type(v).__name__;value=v.get('contents') if isinstance(v,dict) and 'contents' in v else v
   if isinstance(value,(int,float,str,bool,type(None))):continue
   st=s.get('stack',[]);r={'file':p.name,'function':st[-1].get('function_name') if st else None,'line':st[-1].get('line_number') if st else None,'variable':name,'kind':kind,'shape':v.get('shape') if isinstance(v,dict) else None,'value':value}
   records.append(r)
(W/'trace-values.json').write_text(json.dumps(records,ensure_ascii=False,indent=2));print('unique compound values',len(records))
for p in sorted(set(r['file'] for r in records)):
 rows=[r for r in records if r['file']==p];print(p,len(rows),collections.Counter(r['variable'] for r in rows))
