import sys,json,pathlib,re
year=int(sys.argv[1]);lo=int(sys.argv[2]);hi=int(sys.argv[3]);mode=sys.argv[4] if len(sys.argv)>4 else 'outline'
for p in sorted(pathlib.Path('work/alignment').glob(f'{year}-*.json')):
 d=json.loads(p.read_text());n=d['lecture']
 if not lo<=n<=hi:continue
 print('\n===',year,n,d['title'],'===')
 if mode=='outline':
  for w in d['windows']:
   print(f'{int(w["start"]):4}: '+','.join(w['topics'][:5])+' | '+(w['material_candidates'][0]['heading'][:85] if w['material_candidates'] else 'NO SLIDE MATCH'))
 else:
  seg=json.loads(next(pathlib.Path('work/transcripts').glob(f'{year}-{n:02}-*.json')).read_text())['segments']
  if mode=='qa':
   hits=[i for i,x in enumerate(seg) if re.search(r'(?:(?:so|yeah|sorry|repeat).{0,25}question (?:is|was)|you.re asking|you asked)',x['text'],re.I)]
  elif mode=='examples':hits=[i for i,x in enumerate(seg) if re.search(r'for example|imagine|suppose|let.s say|analogy|toy',x['text'],re.I)]
  elif mode=='demo':hits=[i for i,x in enumerate(seg) if re.search(r'let.s run|run this|print out|on the screen|let.s look at (?:the|this) (?:code|plot)|demo',x['text'],re.I)]
  elif mode.startswith('time:'):
   a,b=map(float,mode.split(':')[1:]);hits=[i for i,x in enumerate(seg) if a<=x['start']<=b][:1]
  else:hits=[]
  selected=[]
  for i in hits:
   if selected and seg[i]['start']-seg[selected[-1]]['start']<110:continue
   selected.append(i)
  if mode in ('examples','demo'):selected=selected[::max(1,len(selected)//4)][:4]
  if mode=='qa':selected=selected[:6]
  if len(sys.argv)>5:selected=selected[:int(sys.argv[5])]
  for i in selected:
   arr=[x for x in seg[max(0,i-2):] if x['start']<=seg[i]['start']+(120 if mode.startswith('time:') else 65)]
   print(f'@{int(seg[i]["start"])} '+ ' '.join(x['text'].replace('\n',' ') for x in arr))
