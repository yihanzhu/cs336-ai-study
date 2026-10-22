"""Cache-free bilingual release builder. Chinese facts and source identities remain shared."""
import ast,copy,hashlib,json,pathlib,re,subprocess,sys,zipfile,shutil
from lesson_cards_en_a import CARDS as A
from lesson_cards_en_b import CARDS as B
from lesson_cards_en_c import CARDS as C
from lesson_cards_en_d import CARDS as D
from lesson_cards_en_e import CARDS as E
from lesson_extensions_en import EXTENSIONS,CHAPTERS,PROTOCOL
from bilingual_page import PAGE
from bilingual_review_notes import NUMERIC_NOTES
R=pathlib.Path(__file__).parent.parent;O=R/'outputs';W=R/'work'
subprocess.run([sys.executable,str(W/'build_lesson_plans.py')],check=True,cwd=R)
zh=json.loads((O/'CS336-逐点教案.json').read_text());en=copy.deepcopy(zh);raw=A|B|C|D|E
keys=['title','objective','explanation','worked_example','question','answer_criteria','repair']
limit='Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.'
progression='Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.'
for ch in en['chapters']:
 title,goal=CHAPTERS[int(ch['id'])];ch.update(title=title,goal=goal)
labels={'Gated Delta Networks 原论文':'Gated Delta Networks paper','μTransfer 原论文':'μTransfer paper','QLoRA 原论文':'QLoRA paper','DPO 原论文':'DPO paper','GSPO 原论文':'GSPO paper','Dr. GRPO 原论文':'Dr. GRPO paper','Parcae 原论文':'Parcae paper','Megakernels 作者代码':'Megakernels author code','2026 A5：§5.1/5.3/6.3':'2026 A5: sections 5.1/5.3/6.3'}
for ch in en['chapters']:
 erows=raw[ch['id']].strip().splitlines(); selected=[x for x in en['lessons'] if x['chapter']==ch['id']];assert len(erows)==len(selected)
 for x,line in zip(selected,erows):
  values=line.split('|');assert len(values)==7;x.update(zip(keys,values));pid=x['id']
  gap='Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered.'
  if x['chapter']=='00':gap='This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions.'
  if x['chapter'] in ('08','14'):gap+=' Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR.'
  if pid=='22.06':gap+=' Official 2026 L16 did not fully develop a complete long-context extension recipe; no verified full recipe is invented here.'
  if x['chapter']=='28':gap='See 28.01 and guest_checks in the lecture-comparison ledger for six individual statuses. Unavailable classes are not reconstructed as Q&A or demos. Dan Fu has video/captions, but standalone slides for that class were not obtained.'
  x.update(source_gap=gap,extension=EXTENSIONS.get(pid,''),evidence_boundary=limit,progression=progression,example_status='Original teaching construction, not an unattributed classroom transcript or new measurement')
  for s in x['source_refs']:
   s['label']=labels.get(s['label'],s['label'])
   s['scope']='Chapter-level official source, not a verbatim quotation or automated sentence-level entailment proof'
   if s['id'].startswith('paper-'):s['scope']='Primary supplementary source, not a substitute record of an unavailable guest class'
   if s['id']=='2026-A5':s['scope']='Variant formulas and differences between the assignment and original papers'
for doc,language in ((zh,'zh'),(en,'en')):
 doc.update(version='3.1',language=language)
zh['protocol']='公开页面默认英文；陪学以同一份进度中的preferred_language为准，无偏好才默认英文。切换语言不重置知识点或理解证据。\n'+zh['protocol']
en.update(scope='Complete English counterparts for the original 299 teaching points; original classroom gaps remain; no full independent external review',protocol=PROTOCOL)
(O/'CS336-逐点教案.json').write_text(json.dumps(zh,ensure_ascii=False,indent=2)+'\n')
(O/'CS336-lesson-plans.en.json').write_text(json.dumps(en,ensure_ascii=False,indent=2)+'\n')
bundle={'version':'3.1','date':'2026-10-22','default_language':'en','supported_languages':['en','zh'],'progress_contract':{'shared_file':'local/learning-progress.md','storage':'local_only','tracked':False,'auto_push':False,'key':'knowledge-point ID','public_default':'en','saved_preference_overrides_default':True,'reader_navigation_marks_mastery':False},'languages':{'en':en,'zh':zh}}
(O/'CS336-bilingual-lessons.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2)+'\n')
md=['# CS336: AI tutoring lesson plans','', 'Version 3.1 · 2026-10-22 · 29 chapters, 299 points in each language. English is the public default; saved learner preference controls tutoring language.','',
'These are original teaching plans with examples, questions, answer criteria, and repair branches. They do not recover missing guest classes. Original-material evidence retains the 1.2 audit boundaries. Historical Chinese preparation notes remain available as supplementary audit context; the full 299-point teaching layer is bilingual.','', '## Tutor protocol','',PROTOCOL,'## Chapters','']
for c in en['chapters']:md.append(f'- [{c["id"]} · {c["title"]} ({c["count"]} points)](#c{c["id"]})')
for c in en['chapters']:
 md+=['',f'<a id="c{c["id"]}"></a>',f'## {c["id"]} · {c["title"]}','',c['goal'],'']
 for x in (r for r in en['lessons'] if r['chapter']==c['id']):
  md += [f'<a id="p{x["id"].replace(".","-")}"></a>',f'### {x["id"]} {x["title"]}','',f'**Goal:** {x["objective"]}.', '**Review if needed:** '+(', '.join(x['prerequisites']) or 'No prerequisites')+'.','',f'**Explanation:** {x["explanation"]}','',f'**Worked example:** {x["worked_example"]}','',f'**Ask this, then wait:** {x["question"]}','', '<details><summary>Tutor reference: open after the learner answers</summary>','',f'**Answer criteria:** {x["answer_criteria"]}','',f'**Repair:** {x["repair"]}','',x['progression'],'','</details>','']
  if x['extension']:md+=['<details><summary>Derivation: unfold across turns if needed</summary>','',x['extension'],'','</details>','']
  md+=['**Sources:** '+' · '.join(f'[{s["label"]}]({s["url"]})' for s in x['source_refs']),'',f'**Source gaps and limits:** {x["source_gap"]} {x["evidence_boundary"]}','']
(O/'CS336-lesson-plans.md').write_text('\n'.join(md).rstrip()+'\n')
p=O/'CS336-逐点陪学教案.md';p.write_text(p.read_text().replace('版本2.0','版本3.1（完整双语版，公开入口默认英文）'))
oldpage=(O/'CS336-逐点陪学教案.html').read_text();css=re.search(r'<style>(.*?)</style>',oldpage,re.S)[1]
payload=json.dumps(bundle,ensure_ascii=False).replace('<','\\u003c')
(O/'CS336-lesson-plans.html').write_text(PAGE.replace('__CSS__',css).replace('__DATA__',payload))
# Keep old Chinese links usable without duplicating the full embedded dataset.
(O/'CS336-逐点陪学教案.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>CS336 中文入口</title><p><a id="go" href="CS336-lesson-plans.html?lang=zh">打开中文教案 / Open Chinese lessons</a></p><script>const target='CS336-lesson-plans.html?lang=zh'+location.hash;document.getElementById('go').href=target;location.replace(target);</script></html>''')
curr=['# CS336 curriculum and sources','', 'Version 3.1 · 299 teaching points · shared IDs across English and Chinese.','', 'Source links below are chapter-level navigation. File identities and hashes are retained in CS336-来源索引.json. Teaching order is an original reorganization, not official lecture numbering.','']
for c in en['chapters']:
 curr += [f'## {c["id"]} · {c["title"]}','',c['goal'],'']
 for x in (x for x in en['lessons'] if x['chapter']==c['id']):curr.append(f'- {x["id"]} {x["title"]}')
 first=next(x for x in en['lessons'] if x['chapter']==c['id']);curr+=['','Sources: '+' · '.join(f'[{s["label"]}]({s["url"]})' for s in first['source_refs']),'']
(O/'CS336-curriculum.en.md').write_text('\n'.join(curr).rstrip()+'\n')
(O/'START-HERE.txt').write_text('''CS336 AI tutoring · 3.1 · English default, Chinese supported

Open CS336-lesson-plans.html for the offline bilingual reader. Switching language keeps the same point. Browsing does not record mastery.
For an AI tutor, provide CS336-lesson-plans.md (English) or CS336-逐点陪学教案.md (Chinese), the curriculum, CS336-evidence-and-gaps.en.md, and your ONE current progress file.
First-time learners: copy a blank template into local/learning-progress.md, or in a repository checkout run python3 work/init_learning.py --language en (or zh). Existing learners: never overwrite local progress. The release ZIP offers templates; it does not include your personal history.

Send this to your tutor:
Read my local progress first, including teaching language. Use the saved language; otherwise default to English. Read the current point and its source limits. Treat me as a beginner who needs foundations refreshed. Teach one small concept with a worked example and one understanding question. Wait for my answer. Only then use answer criteria and repair guidance. Do not require videos or large assignments. Do not invent missing original material. Use point IDs for progress across languages. At session end save only to ignored local/learning-progress.md. Never commit or automatically push personal progress to this curriculum repository. Without file tools, give me the full updated text and do not claim it was saved.

Full original evidence context is retained in the historical Chinese preparation/audit files; core 299-point teaching plans and all point-level source limits are available in both languages.
''')
(O/'CONTINUE-LEARNING.md').write_text('''# Continue on another computer

Repository: https://github.com/yihanzhu/cs336-ai-study. Public curriculum files contain no personal progress.

1. Sign in to GitHub and clone or download the latest repository.
2. Give the new AI START-HERE.txt, your preferred-language lesson plans, the curriculum, evidence notes, and your existing progress file.
3. Continue the pending question by point ID. Switching language never resets progress or establishes mastery.

Keep ONE local file at local/learning-progress.md, including preferred_language, current ID, demonstrated understanding, open questions, and the next action. Start from a blank template; public Git history and release ZIPs must not contain personal progress or a shared owner profile.

Git updates the curriculum only. Inspect changes before pulling and do not overwrite conflicts. Save learner progress only under ignored local/. Never commit it or automatically upload it to the curriculum repository. On another computer, download the curriculum and separately carry your local progress file; a new clone does not recover private progress.

Without file tools: ask for complete updated progress text and save it yourself. Private cross-device storage is optional and must be chosen explicitly by the learner, separately from this repository. The offline reader neither calls an AI nor stores learner answers. Chat retention and remote model processing depend on the chosen agent/provider; local progress does not imply local model execution.

Public sharing excludes personal learning records, including Git history. The general release ZIP contains templates only. The repository ignores local/; this is not permission to force-add or upload those files. Original captions, audio/video, official PDFs, and local evidence caches are not redistributed.
''')
(O/'learning-progress-template.en.md').write_text('''# My CS336 learning progress

Material version: 3.0 (evidence baseline 1.2)
preferred_language: en
Updated:
Current point: not started; choose the starting point after an actual foundation check.
Demonstrated understanding, with actual answer evidence: none.
Foundations needing review: not diagnosed.
Unresolved questions:
Pending understanding question:
Next action: teach one point, give one example and one question, then wait.

Save as ignored local/learning-progress.md. Never commit or automatically upload it to the curriculum repository. Keep this ONE local file across languages; carry it separately when changing computers. Only actual answers establish demonstrated understanding. Reading or opening an answer does not.
''')
(O/'CS336-evidence-and-gaps.en.md').write_text('''# Evidence scope and missing originals

Version 3.1 · 2026-10-22. The teaching layer has 299 English and 299 Chinese plans with shared IDs and source identities. This means complete plans for the defined curriculum; it does not mean every original classroom detail was recovered.

| Material | Evidence baseline 1.2 |
|---|---|
| Course schedules | 57 entries across 2024–2026, mapped to 29 teaching chapters |
| Core sources and assignments | 51 core files and 17 assignment PDFs |
| Captions | 34 uploaded tracks plus local ASR for 2026 L08; 56527 segments, 35 chronological text reviews |
| Original audio | 35 recordings; machine comparison plus 267 flagged-region textual checks; not full manual listening |
| PDFs | 42 files, 1889 page layouts/main figures viewed; not every tiny label or original experiment independently checked |
| Images and sampled video | 231 unique lecture images; 266 GIF frames; 158 sampled video times, not all frames |
| Publisher execution records | 19 JSON traces, 2 stdout files; 405 compound outputs inspected, selected relationships recomputed; not local GPU replay |
| Teaching experiments | 10 executed CPU reconstructions; not large-model/GPU performance reproduction |

## Explicit gaps

- No confirmed official 2024 video playlist was obtained. Official 2024 L17 page 2 says the upcoming Ce Zhang and Aakanksha Chowdhery guest classes were not recorded; their standalone class slides were not obtained.
- Full 2025 Junyang Lin and Mike Lewis and 2026 Daniel Selsam classes were not obtained. Qwen3/Llama3 publications are related reading, not recovered classroom Q&A or demos.
- Dan Fu 2026 has class video/captions plus related Parcae/Megakernel sources; standalone slides for that class were not obtained. GPU execution was not reproduced.
- Official 2026 L08 captions remain missing. Local dual-ASR and source comparisons help, but some distant student questions cannot be recovered. ASR systems can share errors; guesses are not quoted as original speech.
- Video inspection is sampled. Cropped screens and intervening changes remain unverified. All-page layout inspection does not independently establish every footnote or experimental datapoint.
- 2026 L16 did not fully develop a complete long-context extension recipe. No complete verified recipe is invented.
- Current model ranks, prices, legal conclusions, hardware specifications, and software support require primary-source checks when taught. This package is not an exhaustive rolling survey after spring 2026.

## Bilingual checks

Point IDs, prerequisite IDs, next-point IDs, source URLs and hashes, extension coverage, and all six teaching fields are checked across languages. Examples, questions, judgment criteria, formulas, and boundaries were reviewed by the same author; this is not independent external peer review. Numeric-text differences include translated number words, units, and simplified wording; candidate reviews are recorded in CS336-bilingual-review.json. Mechanical counts and hashes do not prove semantic correctness.

The main teaching layer and point-level limits are bilingual. Historical detailed preparation/audit narratives remain in Chinese; retained JSON and original primary-source links provide the evidence trail. Full raw captions, PDFs, audio/video, local learner state and recovery caches are excluded from Git and the share package.
''')
# Upgrade personal-facing instructions without overwriting actual learning evidence.
(O/'开始陪学.txt').write_text('3.1完整双语版：公开默认英文，支持完整中文。\n页面：CS336-lesson-plans.html；语言切换保留知识点，不记录掌握。\n\n首次使用：复制空白模板到 local/learning-progress.md，或在仓库运行 python3 work/init_learning.py --language zh。已有本地文件不得覆盖。\n向agent提供当前点教案、核验边界与本地进度。它能否保存聊天由所用平台决定。\n\n启动提示：你是我的CS336陪学导师。先读本地进度及相应语言的当前点教案，按以下规则陪学：\n\n'+zh['protocol'].rstrip()+'\n')
cov=json.loads((O/'CS336-教案覆盖核对.json').read_text());cov['version']='3.1';cov['bilingual']={'default_language':'en','lessons_per_language':299,'languages':['en','zh'],'shared_sources':True,'shared_progress':True};(O/'CS336-教案覆盖核对.json').write_text(json.dumps(cov,ensure_ascii=False,indent=2)+'\n')
# Produce a reproducible parity ledger with source hashes and per-point bilingual identity.
review={'version':'3.1','reviewer_role':'Same author; not independent external review','scope':'All 299 counterparts reviewed for example meaning, question intent, answer criteria, repair, formulas, and source limits. Numeric text candidates are navigation for author review, not automatic semantic proof.','author_sources':[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(W.glob('lesson_cards_en_*.py'))]+[{'file':'lesson_extensions_en.py','sha256':hashlib.sha256((W/'lesson_extensions_en.py').read_bytes()).hexdigest()},{'file':'bilingual_review_notes.py','sha256':hashlib.sha256((W/'bilingual_review_notes.py').read_bytes()).hexdigest()}],'points':[]}
for e,z in zip(en['lessons'],zh['lessons']):
 review['points'].append({'id':e['id'],'title_en':e['title'],'title_zh':z['title'],'english_sha256':hashlib.sha256(json.dumps(e,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'chinese_sha256':hashlib.sha256(json.dumps(z,ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'review_scope':['explanation','worked_example','question','answer_criteria','repair','extension','source_gap'],'review_note':'Equivalent teaching aim, example outcome, question intent and acceptance conditions; compression of wording is not a claim of verbatim translation.'})
# Candidate accounting is computed from source rows; the author reviewed the categories below.
issues=[]
for ch in raw:
 zr=[x for x in zh['lessons'] if x['chapter']==ch]
 for e,z in zip([x for x in en['lessons'] if x['chapter']==ch],zr):
  for field in ['objective','explanation','worked_example','question','answer_criteria','repair']:
   nums=lambda t:set(re.findall(r'(?<![A-Za-z0-9])\d+(?:\.\d+)?',t))
   diff=nums(z[field])-nums(e[field])
   if diff:issues.append({'id':e['id'],'field':field,'chinese_numeric_tokens_not_literal_in_english':sorted(diff),'chinese_text':z[field],'english_text':e[field],'status':'author_reviewed_number_words_units_or_equivalent_context','limitation':'Text matching is not a numeric or semantic equivalence proof.'})
for candidate in issues:
 key=candidate['id']+':'+candidate['field'];assert key in NUMERIC_NOTES,key;candidate['author_resolution']=NUMERIC_NOTES[key]
review['numeric_text_candidates']=issues
review['reviewed_categories']=['Chinese 万 versus English full numbers','English number words: one/two/four, ordinal and fold forms','digits inside dtype or algorithm identifiers','equivalent repeated facts carried in the same example or adjacent explanation','explicit additions made to retain probability normalization, error size, output counts, masks and zero-reward advantage']
(O/'CS336-bilingual-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
catalog=json.loads((O/'CS336-来源索引.json').read_text());catalog['tutoring_plans'].update(version='3.1',default_language='en',languages=['en','zh'],authored_lessons_per_language=299,bilingual_file='CS336-bilingual-lessons.json',bilingual_review_file='CS336-bilingual-review.json');(O/'CS336-来源索引.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
with zipfile.ZipFile(O/'CS336-AI陪学资料包.zip') as z:names=[pathlib.PurePosixPath(n).name for n in z.namelist()]
shutil.copyfile(R/'LICENSE',O/'LICENSE.txt')
shutil.copyfile(R/'THIRD_PARTY_NOTICES.md',O/'THIRD-PARTY-NOTICES.md')
extras=['LICENSE.txt','THIRD-PARTY-NOTICES.md','CS336-lesson-plans.md','CS336-lesson-plans.html','CS336-lesson-plans.en.json','CS336-bilingual-lessons.json','CS336-curriculum.en.md','START-HERE.txt','CONTINUE-LEARNING.md','CS336-evidence-and-gaps.en.md','learning-progress-template.en.md','CS336-bilingual-review.json']
for n in extras:
 if n not in names:names.append(n)
assert '学习进度.md' not in names
with zipfile.ZipFile(O/'CS336-AI陪学资料包.zip','w',zipfile.ZIP_DEFLATED) as z:
 for n in names:z.write(O/n,'CS336-AI-study/'+n)
print(json.dumps({'version':'3.1','english_lessons':len(en['lessons']),'chinese_lessons':len(zh['lessons']),'package_members':len(names),'numeric_text_candidates_reviewed':len(issues)},ensure_ascii=False))
