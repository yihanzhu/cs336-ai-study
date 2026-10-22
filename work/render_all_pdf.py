"""Read-only source rendering and exact pixel-identity inventory; no review claim."""
import concurrent.futures,hashlib,json,pathlib,re,subprocess
from PIL import Image,ImageDraw
W=pathlib.Path(__file__).resolve().parent;R=W/'pdf-audit';R.mkdir(exist_ok=True)
def run(p):
 folder=R/p.stem;folder.mkdir(exist_ok=True)
 n=int(re.search(r'^Pages:\s+(\d+)',subprocess.run(['pdfinfo',str(p)],text=True,capture_output=True,check=True).stdout,re.M)[1])
 existing=list(folder.glob('page-*.png'))
 if len(existing)!=n:subprocess.run(['pdftoppm','-png','-scale-to','1200',str(p),str(folder/'page')],capture_output=True,check=True)
 pages=sorted(folder.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]));assert len(pages)==n
 rows=[]
 for q in pages:
  im=Image.open(q).convert('RGB');rows.append({'file':p.name,'page':int(q.stem.split('-')[-1]),'path':str(q.relative_to(W.parent)),'pixel_sha256':hashlib.sha256(im.tobytes()).hexdigest(),'width':im.width,'height':im.height})
 for i in range(0,n,6):
  batch=pages[i:i+6];sheet=Image.new('RGB',(2400,2115),'#ddd');d=ImageDraw.Draw(sheet)
  for j,q in enumerate(batch):
   im=Image.open(q).convert('RGB');im.thumbnail((1190,675));x=j%2*1200;y=j//2*705;sheet.paste(im,(x,y+26));d.text((x+12,y+5),p.name+' p'+str(rows[i+j]['page']),fill='black')
  sheet.save(folder/f'sheet-{i//6+1:02}.jpg',quality=90)
 (folder/'render-index.json').write_text(json.dumps(rows,indent=2));print(json.dumps({'file':p.name,'rendered':n}),flush=True);return rows
pdfs=sorted((W/'sources').glob('*.pdf'),key=lambda p:(not p.name.startswith('2026'),p.name))
allrows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for rows in pool.map(run,pdfs):allrows+=rows
unique={}
for row in allrows:unique.setdefault(row['pixel_sha256'],[]).append({'file':row['file'],'page':row['page']})
(R/'all-render-index.json').write_text(json.dumps({'pages':allrows,'exact_pixel_groups':list(unique.values()),'review_complete':False},indent=2))
print(json.dumps({'pages_rendered':len(allrows),'exact_unique_images':len(unique)}),flush=True)
