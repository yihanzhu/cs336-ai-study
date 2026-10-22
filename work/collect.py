import concurrent.futures, hashlib, json, pathlib, re, subprocess, urllib.request
from html.parser import HTMLParser
from urllib.parse import urljoin, unquote

ROOT=pathlib.Path(__file__).parent/'sources'
ROOT.mkdir(exist_ok=True)
class Page(HTMLParser):
    def __init__(self):
        super().__init__(); self.rows=[]; self.links=[]; self.row=None; self.cell=None; self.a=None
    def handle_starttag(self,t,a):
        d=dict(a)
        if t=='tr': self.row=[]
        if t in ('td','th') and self.row is not None: self.cell={'text':'','links':[]}
        if t=='a': self.a={'text':'','url':d.get('href','')}
    def handle_data(self,s):
        if self.cell is not None: self.cell['text']+=s
        if self.a is not None: self.a['text']+=s
    def handle_endtag(self,t):
        if t=='a' and self.a is not None:
            self.links.append(self.a)
            if self.cell is not None: self.cell['links'].append(self.a)
            self.a=None
        if t in ('td','th') and self.cell is not None:
            self.cell['text']=' '.join(self.cell['text'].split()); self.row.append(self.cell); self.cell=None
        if t=='tr' and self.row is not None: self.rows.append(self.row); self.row=None

def get(url):
    req=urllib.request.Request(url,headers={'User-Agent':'CS336-learning-source-audit/1.0'})
    with urllib.request.urlopen(req,timeout=45) as r: return r.read()

def raw_url(u):
    return u.replace('https://github.com/','https://raw.githubusercontent.com/').replace('/blob/','/')

manifest={'checked':'2026-10-22','years':{},'files':[]}
tasks=[]
for year,base,repo in [(2024,'https://cs336.stanford.edu/spring2024/','spring2024-lectures'),(2025,'https://cs336.stanford.edu/spring2025/','spring2025-lectures'),(2026,'https://cs336.stanford.edu/','lectures')]:
    data=get(base); (ROOT/f'{year}-website.html').write_bytes(data)
    p=Page(); p.feed(data.decode())
    lectures=[]
    for cells in p.rows:
        if len(cells)<4 or not cells[0]['text'].isdigit(): continue
        n=int(cells[0]['text']); item={'number':n,'date':cells[1]['text'],'title':cells[2]['text'],'links':[]}
        for cell in cells[2:4]:
            for l in cell['links']: item['links'].append({'label':l['text'],'url':urljoin(base,l['url'])})
        for l in item['links']:
            if re.search(r'\.(pdf|py)$',unquote(l['url']),re.I) and 'github.com' in l['url']:
                ext='pdf' if '.pdf' in l['url'] else 'py'; tasks.append((year,n,raw_url(l['url']),ext))
        if n<=17 and not any(x[0]==year and x[1]==n for x in tasks):
            tasks.append((year,n,f'https://raw.githubusercontent.com/stanford-cs336/{repo}/main/lecture_{n:02}.py','py'))
        lectures.append(item)
    links=[{'label':l['text'],'url':urljoin(base,l['url'])} for l in p.links]
    manifest['years'][str(year)]={'url':base,'lectures':lectures,'links':links}

def download(task):
    y,n,u,ext=task; path=ROOT/f'{y}-lecture-{n:02}.{ext}'
    try:
        data=get(u); path.write_bytes(data)
        info={'year':y,'number':n,'url':u,'path':str(path.relative_to(ROOT.parent.parent)),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'status':'ok'}
        if ext=='pdf':
            out=path.with_suffix('.txt'); subprocess.run(['pdftotext','-layout',str(path),str(out)],check=True,capture_output=True)
        return info
    except Exception as e: return {'year':y,'number':n,'url':u,'status':'error','error':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    manifest['files']=list(ex.map(download,tasks))
(ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'lectures':{y:len(x['lectures']) for y,x in manifest['years'].items()},'files':len(tasks),'errors':[x for x in manifest['files'] if x['status']!='ok']},ensure_ascii=False,indent=2))
