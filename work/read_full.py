import json,pathlib,sys
Y,N,P=map(int,sys.argv[1:4]);key=f'{Y}-{N:02}';path=next(pathlib.Path('work/transcripts').glob(key+'-*.json'));d=json.loads(path.read_text());seg=d['segments'];size=len(seg);a=size*(P-1)//3;b=size*P//3
print('FULL CAPTION READING',key,'part',P,'segments',a,b,'of',size)
for i in range(a,b):
 s=seg[i]
 if i==a or int(s['start']//150)!=int(seg[i-1]['start']//150):print('\n@',int(s['start']))
 print(s['text'].replace('\n',' '),end=' ')
print('\nEND part',P,'last caption time',seg[b-1]['start']+seg[b-1]['duration'])
