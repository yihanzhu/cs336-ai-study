import json,pathlib,sys
from review_notes import FINDINGS
Y,N,P=map(int,sys.argv[1:4]);key=f'{Y}-{N:02}';d=json.loads(next(pathlib.Path('work/transcripts').glob(key+'-*.json')).read_text());s=d['segments'];ranges=[(x['start'],x['end']) for x in FINDINGS if x['lecture']==key];remaining=[(i,x) for i,x in enumerate(s) if not any(a<=x['start']<=b for a,b in ranges)];chunks=[];chunk=[];words=0
for i,x in remaining:
 w=len(x['text'].split())
 if words+w>3400 and chunk:chunks.append(chunk);chunk=[];words=0
 chunk.append((i,x));words+=w
if chunk:chunks.append(chunk)
if P==0:print(key,'total_segments',len(s),'previous_context',len(s)-len(remaining),'remaining_segments',len(remaining),'chunks',len(chunks));raise SystemExit
print('REMAINING CAPTION TEXT',key,'chunk',P,'of',len(chunks),'segments',len(chunks[P-1]))
prev=-2;timeblock=-1
for i,x in chunks[P-1]:
 if i!=prev+1 or int(x['start']//150)!=timeblock:print('\n@',round(x['start']), '[original segment',i,']')
 print(x['text'].replace('\n',' '),end=' ');prev=i;timeblock=int(x['start']//150)
print('\nEND CHUNK',P)
