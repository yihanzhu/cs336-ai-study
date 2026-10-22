"""Inventory every source PDF page. Mechanical checks are not semantic sign-off."""
import hashlib,json,pathlib,re,subprocess
from pdf_findings import PDF_REVIEWS
W=pathlib.Path(__file__).resolve().parent;O=W.parent/'outputs';cache=W/'pdf-audit';cache.mkdir(exist_ok=True)
for review in PDF_REVIEWS:
 assert hashlib.sha256((W/'sources'/review['file']).read_bytes()).hexdigest()==review['source_pdf_sha256_expected'],'PDF evidence changed; repeat review before reusing the findings.'
entries=[]
for p in sorted((W/'sources').glob('*.pdf')):
 info=subprocess.run(['pdfinfo',str(p)],capture_output=True,text=True,check=True).stdout
 count=int(re.search(r'^Pages:\s+(\d+)',info,re.M)[1])
 result=subprocess.run(['pdftotext','-layout',str(p),'-'],capture_output=True,text=True,check=True)
 pages=result.stdout.split('\f')
 if not pages[-1].strip():pages.pop()
 assert len(pages)==count,(p.name,len(pages),count)
 pdf_hash=hashlib.sha256(p.read_bytes()).hexdigest()
 for n,t in enumerate(pages,1):
  entry={'file':p.name,'page':n,'pdf_sha256':pdf_hash,'text_sha256':hashlib.sha256(t.encode()).hexdigest(),'words':len(t.split()),'text_extraction_complete':True,'visual_review_complete':False,'semantic_review_complete':False}
  entries.append(entry)
  dest=cache/p.stem;dest.mkdir(exist_ok=True);(dest/f'{n:03}.txt').write_text(t)
for review in PDF_REVIEWS:
 for item in entries:
  if item['file']==review['file'] and item['page'] in review['pages_viewed']:
   item['visual_overview_complete']=True
   item['visual_review_scope']=review['scope']
   item['semantic_findings_file']='CS336-PDF内容核对.json'
summary={'date':'2026-10-22','pdf_files':42,'core_lecture_pdf_pages':1439,'assignment_pdf_pages':450,'total_pages':len(entries),'page_texts_extracted':len(entries),'pages_with_zero_words':sum(x['words']==0 for x in entries),'scope':'Page enumeration, PDF readability, text extraction and checksums only. No semantic or visual sign-off implied.','pages':entries}
assert len(entries)==1889
(O/'CS336-PDF逐页核验记录.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k!='pages'},ensure_ascii=False))

records=[]
for review in PDF_REVIEWS:
 r=dict(review);r['source_pdf_sha256']=hashlib.sha256((W/'sources'/r['file']).read_bytes()).hexdigest()
 r['rendered_pages']=[{'page':n,'sha256':hashlib.sha256((cache/pathlib.Path(r['file']).stem/f'page-{n:02}.png').read_bytes()).hexdigest()} for n in r['pages_viewed']]
 records.append(r)
(O/'CS336-PDF内容核对.json').write_text(json.dumps({'date':'2026-10-22','scope':'Authored content findings from rendered-page inspection; not a claim of all-PDF semantic correctness.','reviews':records},ensure_ascii=False,indent=2))
