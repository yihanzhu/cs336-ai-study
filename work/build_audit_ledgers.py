"""Build source-bound audit ledgers from authored notes; do not infer semantic proof from hashes."""
import hashlib,json,pathlib
from full_pdf_notes import PDF_FULL_NOTES
from assignment_figure_notes import ASSIGNMENT_FIGURE_NOTES
from trace_figure_notes import TRACE_FIGURE_NOTES
from trace_value_notes import TRACE_VALUE_NOTES
R=pathlib.Path(__file__).parent.parent;W=R/'work';O=R/'outputs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,d):(O/name).write_text(json.dumps(d,ensure_ascii=False,indent=2))
inv=json.load(open(O/'CS336-PDF逐页核验记录.json'));old=json.load(open(O/'CS336-PDF内容核对.json'))
renders=json.load(open(W/'pdf-audit/all-render-index.json'))['pages'];by={(x['file'],x['page']):x for x in renders}
reviews=[x for x in old['reviews'] if x['file']=='2026-lecture-08.pdf']
for file,n in {**PDF_FULL_NOTES,**ASSIGNMENT_FIGURE_NOTES}.items():
 findings=[{'pages':list(range(a,b+1)),'finding':t} for a,b,t in n['findings']]
 assert set(sum((x['pages'] for x in findings),[]))==set(range(1,n['pages']+1))
 reviews.append({'file':file,'source_pdf_sha256':sha(W/'sources'/file),'pages_viewed':list(range(1,n['pages']+1)),'scope':n.get('limits'),'primary_sources':n.get('primary_sources',[]),'semantic_signoff':False,'findings':findings})
for x in reviews:
 x['source_pdf_sha256']=sha(W/'sources'/x['file'])
 x['rendered_pages']=[{k:v for k,v in by[(x['file'],p)].items() if k!='file'} for p in x['pages_viewed']]
 for a in x['rendered_pages']:a['sha256']=sha(R/a['path'])
assert len(reviews)==42 and sum(len(x['pages_viewed']) for x in reviews)==1889
for x in inv['pages']:
 r=by[(x['file'],x['page'])];x.update(visual_overview_complete=True,visual_review_complete=False,semantic_review_complete=False,render_pixel_sha256=r['pixel_sha256'],render_sha256=sha(R/r['path']),render_path=r['path'],content_review_file='CS336-PDF内容核对.json')
inv['scope']='All 1889 page layouts/main figures visually inspected, with authored page-range notes and selected dense formulas cross-read. Not every tiny label/cited experiment independently verified; not a full semantic signoff.'
save('CS336-PDF逐页核验记录.json',inv);save('CS336-PDF内容核对.json',{'date':'2026-10-22','scope':inv['scope'],'reviews':reviews})
figs=json.load(open(W/'trace-figures/index.json'));uniq=json.load(open(W/'trace-figures/unique.json'));notes=[]
for a,b,t in TRACE_FIGURE_NOTES:
 notes.append({'unique_indices':list(range(a,b+1)),'finding':t})
assert set(sum((x['unique_indices'] for x in notes),[]))==set(range(len(uniq)))
for i,x in enumerate(uniq):
 assert sha(R/x['file'])==x['sha256'];x.update(unique_index=i,visual_overview_complete=True,findings=[n['finding'] for n in notes if i in n['unique_indices']])
save('CS336-讲义图像核验.json',{'date':'2026-10-22','versioned_source_count':len(figs),'unique_image_count':len(uniq),'scope':'All unique image layouts/main diagrams inspected; selected equations and axes crosschecked. Small text, each raw benchmark point, and every cited original experiment not independently reproduced. Images excluded from distribution.','sources':figs,'unique_images':uniq,'notes':notes,'animation_frames':json.load(open(W/'trace-figures/animation-index.json')),'animation_scope':'Both animated assets fully expanded: image22 six frames and image63 260 frames visually inspected. Image22 counts 6 multiply/add FLOPs per weight for linear forward+two backward paths; image63 splits outlier features into FP16 and quantizes the regular branch, dequantizes int32 accumulation then adds FP16 outputs; tiny labels remain limited by source resolution.'})
vals=json.load(open(W/'trace-values.json'));checks=json.load(open(W/'trace-value-checks.json'));records=[]
for i,x in enumerate(vals):
 record={k:v for k,v in x.items() if k!='value'};record.update(index=i,value_sha256=hashlib.sha256(json.dumps(x['value'],sort_keys=True,ensure_ascii=False).encode()).hexdigest(),source_trace_sha256=sha(W/'traces'/x['file']));records.append(record)
save('CS336-演示数组核验.json',{'date':'2026-10-22','scope':checks['scope'],'checks':checks['checks'],'records':records,'notes':[{'file':f,'topic':t,'finding':s} for f,t,s in TRACE_VALUE_NOTES]})
print({'pdfs':len(reviews),'pages':len(inv['pages']),'unique_images':len(uniq),'compound_values':len(records)})
