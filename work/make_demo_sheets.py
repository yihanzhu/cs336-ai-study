import json,pathlib
from PIL import Image,ImageDraw
D=pathlib.Path('work/demo-audit');out=D/'sheets';out.mkdir(exist_ok=True)
rows=[f for f in json.load(open(D/'index.json'))['frames'] if f['success']]
for offset in range(0,len(rows),4):
 batch=rows[offset:offset+4];im=Image.new('RGB',(2000,1200),'#dddddd');dr=ImageDraw.Draw(im)
 for i,r in enumerate(batch):
  p=D/(r['lecture']+'-'+str(r['timestamp'])+'.png');x=(i%2)*1000;y=(i//2)*600;pic=Image.open(p);pic.thumbnail((1000,562));im.paste(pic,(x,y+35));dr.text((x+10,y+10),r['lecture']+' @ '+str(r['timestamp'])+'s',fill='black')
 im.save(out/('sheet-%02d.jpg'%(offset//4+1)),quality=90)
(out/'index.json').write_text(json.dumps(rows,ensure_ascii=False))
print('sheets',len(list(out.glob('*.jpg'))),'frames',len(rows))
