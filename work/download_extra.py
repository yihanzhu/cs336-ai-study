import concurrent.futures,json,pathlib,urllib.request,subprocess,hashlib,sys
base=pathlib.Path(__file__).parent
manifest=base/(sys.argv[1] if len(sys.argv)>1 else 'assignments_all.json')
items=json.loads(manifest.read_text())
def run(x):
    p=base/'sources'/x['name']
    with urllib.request.urlopen(x['url'],timeout=45) as r: b=r.read()
    p.write_bytes(b)
    subprocess.run(['pdftotext','-layout',str(p),str(p.with_suffix('.txt'))],check=True,capture_output=True)
    x.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='ok')
    return x
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex: items=list(ex.map(run,items))
manifest.write_text(json.dumps(items,ensure_ascii=False,indent=2))
print('Downloaded and extracted',len(items),'assignment PDFs')
