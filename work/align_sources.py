"""Deterministic navigation aid, not a semantic-review or completeness claim."""
import ast,collections,hashlib,json,math,pathlib,re
ROOT=pathlib.Path('work');OUT=ROOT/'alignment';OUT.mkdir(exist_ok=True)
STOP=set('a an the and or of to in on for from with by as is are was were be been being it its this that these those we i you they he she them our your their so if but not no yes do does did have has had can could would should will may might just about like actually basically really very also right okay ok now then here there what how why when where which who all some any more most much many little lot kind sort thing things go going get gets got let lets know think see look looks looking say says said want need use using used make makes made one two three first second next last example examples question questions good great yeah sure well up down out into over even only still there something because different same at than number put bit point thats dont im youre ill ive uh um ah'.split())
def terms(text):
    words=re.findall(r'[a-z][a-z0-9_-]{2,}',text.lower())
    return [w[:-1] if w.endswith('s') and len(w)>5 else w for w in words if w not in STOP]
def headings(path):
    if path.suffix=='.py':
        tree=ast.parse(path.read_text());result=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='text' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):
                s=n.args[0].value
                if s.startswith('#'):result.append((None,s.lstrip('# ')))
        return result
    result=[]
    for i,page in enumerate(path.read_text().split('\f'),1):
        lines=[x.strip() for x in page.splitlines() if x.strip()]
        if lines:result.append((i,' '.join(lines[:2])))
    return result

index=json.loads((ROOT/'transcripts/index.json').read_text());records=[]
for v in index:
    if v['status']!='extracted' and not v.get('local_asr'):continue
    year=v['year'];order=v['playlist_order']
    p=ROOT/v['local_asr']['path'] if v.get('local_asr') else ROOT/'transcripts'/f'{year}-{order:02}-{v["id"]}.json';d=json.loads(p.read_text())
    group=collections.defaultdict(list)
    for s in d['segments']:group[int(s['start']//150)].append(s)
    paper=ROOT/'sources'/f'{year}-lecture-{order:02}.py'
    if not paper.exists():paper=paper.with_suffix('.txt')
    if year==2026 and order==18:paper=None
    source=paper.read_text() if paper and paper.exists() else ''
    hs=headings(paper) if paper and paper.exists() else []
    hc=[collections.Counter(terms(x[1])) for x in hs]
    df=collections.Counter()
    for arr in group.values():df.update(set(terms(' '.join(s['text'] for s in arr))))
    windows=[]
    for k,arr in sorted(group.items()):
        text=' '.join(x['text'].replace('\n',' ') for x in arr);tc=collections.Counter(terms(text))
        scored=sorted(tc,key=lambda w:tc[w]*math.log(1+len(group)/(1+df[w])),reverse=True)
        overlap=[]
        for (page,h),c in zip(hs,hc):
            score=sum(min(tc[w],n) for w,n in c.items())/(math.sqrt(sum(c.values()) or 1))
            if score:overlap.append((score,page,h))
        overlap.sort(reverse=True,key=lambda x:x[0])
        # Candidate identification is explicit; an occurrence is not proof of novelty.
        markers=[]
        for category,pattern in [('问答',r'\?|question|asking|you asked|yeah, so'),('例子',r'for example|imagine|suppose|let.s say|toy|analogy'),('演示',r'let.s run|run this|you can see|let.s look|print|plot|code|demo|screen')]:
            hits=[i for i,x in enumerate(arr) if re.search(pattern,x['text'],re.I)]
            chosen=[]
            for i in hits:
                if chosen and i-chosen[-1]<10:continue
                chosen.append(i)
                if len(chosen)>=2:break
            for i in chosen:
                excerpt=' '.join(x['text'].replace('\n',' ') for x in arr[max(0,i-3):min(len(arr),i+13)])
                markers.append({'type':category,'start':arr[max(0,i-3)]['start'],'end':arr[min(len(arr)-1,i+12)]['start']+arr[min(len(arr)-1,i+12)]['duration'],'text':excerpt})
        windows.append({'start':arr[0]['start'],'end':arr[-1]['start']+arr[-1]['duration'],'segments':len(arr),'topics':scored[:9],'material_candidates':[{'page':p,'heading':h,'lexical_score':round(sc,2)} for sc,p,h in overlap[:3]],'review_candidates':markers,'text':text})
    rec={'year':year,'lecture':order,'video_id':v['id'],'title':v['title'],'video_url':v['url'],'source':str(paper) if paper else None,'source_type':'local_asr_unverified' if v.get('local_asr') else 'uploaded_caption','alignment_method':'150-second transcript windows; lexical heading matches are navigation candidates, not semantic findings','semantic_status':'awaiting_review','windows':windows}
    (OUT/f'{year}-{order:02}.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2))
    records.append({k:r for k,r in rec.items() if k!='windows'}|{'windows':len(windows),'segments_accounted':sum(w['segments'] for w in windows)})
    assert sum(w['segments'] for w in windows)==(v['local_asr']['segments_count'] if v.get('local_asr') else v['segments_count'])
(OUT/'index.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print('Aligned',len(records),'transcripts;',sum(r['windows'] for r in records),'windows; all retrieved segments accounted for.')
