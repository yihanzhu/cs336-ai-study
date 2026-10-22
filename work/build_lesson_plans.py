"""Build the 299 authored tutoring plans; standard library only, no source cache required."""
import ast,hashlib,html,json,pathlib,re,zipfile
from lesson_cards_a import CARDS as A
from lesson_cards_b import CARDS as B
from lesson_cards_c import CARDS as C
from lesson_cards_d import CARDS as D
from lesson_cards_e import CARDS as E
from lesson_card_extensions import EXTENSIONS,EXTRA_SOURCES
from lesson_review_notes import REVIEW_NOTES
R=pathlib.Path(__file__).parent.parent;O=R/'outputs'
tree=ast.parse((R/'work/build_guide.py').read_text())
modules=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='modules' for t in n.targets))
catalog=json.loads((O/'CS336-来源索引.json').read_text()); raw=A|B|C|D|E
KEYS=['objective','explanation','worked_example','question','answer_criteria','repair']
PREREQ={
'00':[], '01':['00.05'], '02':['01.01'], '03':['00.03','00.04'], '04':['00.06','00.07','03.07'],
'05':['02.03','03.01','04.01'], '06':['05.10'], '07':['05.05','09.01'], '08':['05.07','04.03'],
'09':['00.04','03.02'], '10':['09.07','09.08'], '11':['10.04','09.09'], '12':['05.03','09.07','10.04'],
'13':['03.01'], '14':['13.02','13.03','09.04'], '15':['01.05','05.12','09.08'], '16':['00.06','09.03'],
'17':['04.04','04.05','16.02'], '18':['00.05','00.08','04.02'], '19':['00.08'], '20':['19.01','00.05'],
'21':['20.01','00.05'], '22':['19.04','16.02'], '23':['01.06','04.02'], '24':['23.01','00.05'],
'25':['24.05','00.07'], '26':['25.03','03.07'], '27':['05.01','23.03'], '28':['18.10']}
GLOBAL_LIMIT='原创教学重建；不是课堂逐字译稿。继承1.2核验边界：未人工逐句听校或逐帧审片，GPU训练未复跑。示例为教学构造，除明确链接实验记录外不作实测声明。'
rows=[]; chapters=[]
def refs_for(keys):
 result=[]
 for key in keys.split():
  y,n=key.split('-'); l=catalog['years'][y]['lectures'][int(n)-1]
  f=next((f for f in catalog['files'] if str(f['year'])==y and f['number']==int(n)),None)
  result.append({'id':key,'label':y+' L'+n,'url':l.get('download') or l.get('video_url') or 'https://cs336.stanford.edu/'+('spring'+y+'/' if y!='2026' else '')+'#schedule','sha256':f['sha256'] if f else None,'scope':'章节级官方来源；非本段逐字引文或自动逐句蕴含证明'})
 return result
for mid,title,goal,points,refs,check in modules:
 cards=raw[mid].strip().splitlines(); titles=points.split('；'); assert len(cards)==len(titles)
 chapters.append({'id':mid,'title':title,'goal':goal,'count':len(cards)})
 for i,(name,line) in enumerate(zip(titles,cards),1):
  values=line.split('|'); assert len(values)==6,(mid,i)
  pid=f'{mid}.{i:02}'; d=dict(zip(KEYS,values))
  gap='本点已有可教学的公开来源。原课堂音视频仍受全局核验限制；不声称恢复所有学生问答。'
  if mid=='00':gap='本章为补基础的原创教学重排，非官方单独一讲；不伪造课堂例子。'
  if mid in ('08','14'):gap+=' 2026 L08官方字幕未取得，现有本地转写中部分学生原问无法恢复。'
  if pid=='22.06':gap+=' 官方2026 L16未给出完整长上下文扩展配方，本点不补写成已验证配方。'
  if mid=='28':gap='六场具体缺口见28.01和逐讲对照记录guest_checks；未取得的课堂不重建问答/演示。Dan Fu有录像字幕，独立本场幻灯片未取得。'
  d.update(id=pid,title=name,chapter=mid,prerequisites=PREREQ[mid],source_refs=refs_for(refs),source_gap=gap,extension=EXTENSIONS.get(pid,''),evidence_boundary=GLOBAL_LIMIT,
   progression='达到判断依据且能解释原因才可建议下一点；答对选项但理由不清时，沿本例追问原因。答错时使用误解修复，仍停在本点。学习者明确要求跳过时记录为未确认掌握。',
   example_status='原创教学构造；非未经说明的课堂实录或新实测')
  for label,url in EXTRA_SOURCES.get(pid,[]): d['source_refs'].append({'id':'paper-'+pid,'label':label,'url':url,'sha256':None,'scope':'一手补充；不是缺失嘉宾课堂的替代记录'})
  if pid=='25.09':
   for f in catalog['assignment_files']:
    if '2026' in str(f) and 'assignment5' in str(f.get('url','')):
     d['source_refs'].append({'id':'2026-A5','label':'2026 A5：§5.1/5.3/6.3','url':f['url'],'sha256':f.get('sha256'),'scope':'变体公式与作业/论文差异'});break
  rows.append(d)
assert len(rows)==299 and len({x['id'] for x in rows})==299
ids=[x['id'] for x in rows]
for i,x in enumerate(rows):x['next_id']=ids[i+1] if i+1<len(ids) else None
PROTOCOL='''先读取仅本地的 local/learning-progress.md，核实当前点及未解决问题；不存在时使用空白模板。不要把本文件一次性展示给学习者。
每点按“目标→讲解→例子逐步推演→一个理解问题”展开。遇到术语先用前置点补，不让学习者先看视频。
提问后停止本轮，不泄露导师参考。学习者作答后才读判断依据并反馈：先指出回答中成立的部分，再处理一项误解。
若答错，用本点的修复方式再教一次，并沿原问题提一个更小的追问；仍不前进。若正确但无理由，问例子中为何如此。
若已能解释原因和适用条件，可建议下一点；学习者可以跳过，进度标“跳过/未确认”。不要求大型作业。
扩展推导按需要分多轮：同一知识点可以停留多轮，不把一张卡当一次讲完的配额。
每次结束只更新被Git忽略的 local/learning-progress.md，绝不提交或自动推送个人记录：当前位置、回答证据、疑问、待回答问题、下一点。不得依据讲过或参考答案已展开而标掌握。
材料事实与原创例子分开；原始材料缺失、未经运行和数值假设必须说明。即时价格、法律、榜单和软件支持需当场查一手来源。
'''
doc={'version':'2.0','date':'2026-10-22','scope':'原目录299点逐点原创陪学教案齐备；不代表所有原课堂材料完整恢复或外部独立审校通过','evidence_baseline':'1.2-public-audit','protocol':PROTOCOL,'chapters':chapters,'lessons':rows}
(O/'CS336-逐点教案.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n')
md=['# CS336 逐点陪学教案','', '版本2.0 · 2026-10-22 · 29章299点。每点均有讲解、推演、理解问题和答后分支。','',
'原课程公开材料与核验记录沿用1.2版。教案齐备不等于缺失课堂已恢复；本文件是中文原创教学重建。第28章明确教材料边界，不编造未知嘉宾讲座内容。','',
'## 导师使用规则','',PROTOCOL,'## 章节导航','']
for c in chapters:md.append(f'- [{c["id"]} · {c["title"]}（{c["count"]}点）](#c{c["id"]})')
for c in chapters:
 md+=['',f'<a id="c{c["id"]}"></a>',f'## {c["id"]} · {c["title"]}','',c['goal'],'']
 for x in (r for r in rows if r['chapter']==c['id']):
  md += [f'<a id="p{x["id"].replace(".","-")}"></a>',f'### {x["id"]} {x["title"]}','',f'**学习目标：** {x["objective"]}。',f'**需要时回补：** '+('、'.join(x['prerequisites']) or '无需前置知识')+'。','',f'**讲解：** {x["explanation"]}','',f'**例子逐步走：** {x["worked_example"]}','',f'**只问这一个问题，然后等待：** {x["question"]}','', '<details><summary>导师参考：收到回答后再展开</summary>','',f'**判断依据：** {x["answer_criteria"]}','',f'**答错或含混时：** {x["repair"]}','',x['progression'],'','</details>','']
  if x['extension']:md+=['<details><summary>需要时分轮展开的推导</summary>','',x['extension'],'','</details>','']
  md += ['**来源：** '+' · '.join(f'[{r["label"]}]({r["url"]})' for r in x['source_refs']),'',f'**原始材料与边界：** {x["source_gap"]} {x["evidence_boundary"]}','']
(O/'CS336-逐点陪学教案.md').write_text('\n'.join(md).rstrip()+'\n')
# HTML has no network dependencies; JSON is embedded, answers collapse on every navigation.
payload=json.dumps(doc,ensure_ascii=False).replace('<','\\u003c')
page=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CS336 · 299点AI陪学教案</title>
<style>
:root{color-scheme:light;--bg:#f5f3ed;--ink:#24352e;--line:#d9dfd7;--accent:#216447}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif}header{padding:25px 5vw 22px;border-bottom:1px solid var(--line);background:#e9efe6}h1{font-size:clamp(24px,4vw,36px);margin:0 0 8px}h2{font-size:27px;line-height:1.5;margin:10px 0 18px}p{margin:0 0 16px}header p{max-width:960px;margin:0}.tag{font-size:13px;letter-spacing:.08em;color:#476853}.layout{display:grid;grid-template-columns:285px minmax(0,850px);gap:34px;max-width:1240px;margin:auto;padding:28px 25px}aside{min-width:0}#search{width:100%;padding:11px;border:1px solid #91a992;border-radius:8px;font:inherit;background:white}#nav{height:72vh;overflow:auto;padding:12px 2px}#nav details{border:0;padding:5px 0}summary{cursor:pointer;font-weight:600}button{font:inherit;cursor:pointer;border:1px solid #9bae9f;border-radius:7px;background:#fff;padding:8px 14px;color:var(--ink)}button:hover{background:#e3eee3}button:disabled{opacity:.4;cursor:default}.point{display:block;width:100%;text-align:left;font-size:14px;line-height:1.6;border:0;background:transparent;padding:7px}.point[aria-current=true]{background:#dceadb;color:#174c33;border-left:3px solid var(--accent)}main{min-width:0;background:#fff;border:1px solid var(--line);border-radius:14px;padding:30px 38px}.small{font-size:14px;color:#596b5d}section{margin:20px 0;padding-bottom:10px}h3{font-size:16px;margin:0 0 9px;color:var(--accent)}.question{padding:20px;border-left:4px solid var(--accent);background:#f0f5ef}.question p{font-size:19px;margin:0}.stop{font-size:14px;color:#476853;margin-top:10px}main details{border-top:1px solid var(--line);padding:15px 0}main details p{margin-top:14px}.actions{display:flex;justify-content:space-between;gap:10px;margin-top:26px;flex-wrap:wrap}.source a{display:block;overflow-wrap:anywhere;color:var(--accent)}#notice{font-size:13px;color:#60705e;min-height:25px}.formula{white-space:pre-wrap;overflow-wrap:anywhere}a{color:var(--accent)}@media(max-width:780px){.layout{grid-template-columns:1fr;padding:15px;gap:15px}#nav{height:220px}main{padding:23px 20px}header{padding:20px}h2{font-size:24px}}
</style>
<header><div class="tag">STANFORD CS336 · 中文原创陪学教案 · 2.0</div><h1>一次走通一个知识点</h1><p>29章 · 299点教案齐备。先讲解，再提问，等你回答后再反馈。原始课堂的缺口仍保留；这里不会自动判断你是否理解。</p></header>
<div class="layout"><aside><label for="search">查找知识点</label><input id="search" type="search" placeholder="编号或标题，如 12.05"><nav id="nav" aria-label="章节知识点"></nav></aside><main>
<div class="small" id="counter"></div><h2 id="title"></h2><p id="objective"></p><p class="small" id="prereq"></p>
<section><h3>先理解这个关系</h3><p id="explanation" class="formula"></p></section><section><h3>一起走一个例子</h3><p id="example" class="formula"></p></section>
<section class="question"><h3>轮到你解释</h3><p id="question"></p><div class="stop">此处停下。把你的回答交给AI，再由它依据教案反馈。</div></section>
<details id="answer"><summary>导师参考 · 收到回答后再展开</summary><p id="criteria"></p><h3>答错或含混时</h3><p id="repair"></p><p id="progression" class="small"></p></details>
<details id="extension"><summary>需要时分轮展开的推导</summary><p id="deep" class="formula"></p></details>
<details id="sources"><summary>来源与原始材料缺口</summary><p id="gap"></p><div id="refs" class="source"></div><p id="boundary" class="small"></p></details>
<details><summary>怎样让AI陪学与换电脑接续</summary><p>把逐点教案、目录、核验说明和个人学习进度交给能读文件的AI。每轮只教当前点，提问后停止。正确理解后才前进。每次结束只更新本地 local/learning-progress.md，不提交或自动推送；本页面的翻页不等于掌握，也不自动同步进度。</p></details>
<div class="actions"><button id="prev">上一点</button><button id="copy">复制本点开课请求</button><button id="next">浏览下一点</button></div><div id="notice" role="status"></div>
</main></div><script id="data" type="application/json">__DATA__</script><script>
const D=JSON.parse(document.getElementById('data').textContent),L=D.lessons;let pos=0;const $=id=>document.getElementById(id);
function nav(){const term=$('search').value.trim().toLowerCase();$('nav').replaceChildren();D.chapters.forEach(c=>{const found=L.filter(x=>x.chapter===c.id&&(x.id+' '+x.title).toLowerCase().includes(term));if(!found.length)return;const group=document.createElement('details');group.open=!!term||L[pos].chapter===c.id;const s=document.createElement('summary');s.textContent=c.id+' '+c.title;group.append(s);found.forEach(x=>{const b=document.createElement('button');b.className='point';b.textContent=x.id+' '+x.title;b.setAttribute('aria-current',String(x.id===L[pos].id));b.onclick=()=>show(L.indexOf(x));group.append(b)});$('nav').append(group)})}
function show(i){pos=i;const x=L[i];$('counter').textContent='知识点 '+(i+1)+' / '+L.length+' · 当前位置仅供浏览，不是学习进度';$('title').textContent=x.id+' '+x.title;$('objective').textContent='目标：'+x.objective;$('prereq').textContent='需要时回补：'+(x.prerequisites.join('、')||'无需前置知识');[['explanation','explanation'],['example','worked_example'],['question','question'],['criteria','answer_criteria'],['repair','repair'],['deep','extension'],['gap','source_gap'],['boundary','evidence_boundary'],['progression','progression']].forEach(([id,key])=>$(id).textContent=x[key]);$('extension').hidden=!x.extension;['answer','extension','sources'].forEach(id=>$(id).open=false);$('refs').replaceChildren();x.source_refs.forEach(r=>{const a=document.createElement('a');a.href=r.url;a.textContent=r.label+' · '+r.scope;a.target='_blank';a.rel='noopener noreferrer';$('refs').append(a)});$('prev').disabled=i===0;$('next').disabled=i===L.length-1;$('notice').textContent='';history.replaceState(null,'','#p'+x.id);nav()}
$('prev').onclick=()=>show(pos-1);$('next').onclick=()=>show(pos+1);$('search').oninput=nav;
$('copy').onclick=async()=>{const x=L[pos],t='请按CS336 2.0教案陪我学习'+x.id+' '+x.title+'。先读取我的学习进度。一次只讲这个点，按下面内容展开一个例子，再问一个理解问题，等我回答。不要先给参考答案。\n讲解：'+x.explanation+'\n例子：'+x.worked_example+'\n问题：'+x.question+'\n材料边界：'+x.source_gap;try{await navigator.clipboard.writeText(t);$('notice').textContent='已复制。本页面不会自动发送给AI。'}catch(e){$('notice').textContent='浏览器未允许复制，请手动选择本点内容。'}};
window.addEventListener('hashchange',()=>{const i=L.findIndex(x=>'#p'+x.id===location.hash);if(i>=0&&i!==pos)show(i)});
const initial=L.findIndex(x=>'#p'+x.id===location.hash);show(initial<0?0:initial);
</script></html>'''
(O/'CS336-逐点陪学教案.html').write_text(page.replace('__DATA__',payload))
# Source-bound accounting; identities and fields, not a substitute for semantic review.
coverage={'version':'2.0','expected':299,'actual':len(rows),'chapters':chapters,'method':'Exact ID/title comparison with the pre-existing curriculum; six authored teaching fields plus source/boundary mapping. Author review, not independent external peer review. Structural checks cannot establish all semantic claims.','fields':KEYS,'author_sources':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((R/'work').glob('lesson_cards_[a-e].py'))]+[{'file':'lesson_card_extensions.py','sha256':hashlib.sha256((R/'work/lesson_card_extensions.py').read_bytes()).hexdigest()},{'file':'lesson_review_notes.py','sha256':hashlib.sha256((R/'work/lesson_review_notes.py').read_bytes()).hexdigest()}],'points':[{'id':x['id'],'title':x['title'],'authored_fields':list(KEYS),'source_ids':[r['id'] for r in x['source_refs']],'gap':x['source_gap'],'sha256':hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True).encode()).hexdigest()} for x in rows],'limits':['No assertion that missing guest classes were recovered','No external independent review of all 299 lessons','No new GPU experiments','Mechanical acceptance is not proof of complete semantic accuracy']}
coverage['author_review_notes']=REVIEW_NOTES
coverage['author_review_role']='Same author; not independent external review'
(O/'CS336-教案覆盖核对.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n')
catalog['tutoring_plans']={'version':'2.0','authored_lessons':299,'chapters':29,'file':'CS336-逐点教案.json','coverage_file':'CS336-教案覆盖核对.json','scope':'Original curriculum lesson plans complete; not complete capture of missing original classes; not independently peer reviewed'}
(O/'CS336-来源索引.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
# Flat share package stays free of personal learning progress and raw sources.
old=O/'CS336-AI陪学资料包.zip'
materials_tree=ast.parse((R/'work/build_materials.py').read_text())
names=next(ast.literal_eval(n.value) for n in materials_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='files' for t in n.targets))
for n in ['CS336-逐点陪学教案.md','CS336-逐点陪学教案.html','CS336-逐点教案.json','CS336-教案覆盖核对.json']:
 if n not in names:names.append(n)
assert '学习进度.md' not in names
with zipfile.ZipFile(old,'w',zipfile.ZIP_DEFLATED) as z:
 for n in names:z.write(O/n,'CS336-AI陪学/'+n)
print(json.dumps({'authored_lessons':len(rows),'chapters':len(chapters),'package_members':len(names),'lesson_markdown_bytes':(O/'CS336-逐点陪学教案.md').stat().st_size},ensure_ascii=False))
