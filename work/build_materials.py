import ast,hashlib,html,json,pathlib,re,zipfile
from lesson_notes import NOTES_26,NOTES_25
from supplement_notes import SUPPLEMENTS
from frame_findings import FRAME_FINDINGS
from review_notes import FINDINGS,TRACE_NOTES
from guest_notes import GUESTS
from full_reading_notes import FULL_READINGS
from pdf_findings import PDF_REVIEWS
from release_sections import append_audit
from demo_visual_notes import DEMO_VISUAL_NOTES

ROOT=pathlib.Path(__file__).parent.parent;OUT=ROOT/'outputs'
catalog=json.loads((OUT/'CS336-来源索引.json').read_text());videos=json.loads((ROOT/'work/transcripts/index.json').read_text())
by_video={(v['year'],v['playlist_order']):v for v in videos}
demo_results=json.loads((ROOT/'work/demo-results.json').read_text())

def stamp(n):
 n=int(n);return f'{n//3600:02}:{n%3600//60:02}:{n%60:02}'
def video_link(v,start,end=None):
 return f'[{stamp(start)}'+(f'–{stamp(end)}' if end else '')+f']({v["url"]}&t={int(start)}s)'
def source(key):
 if key.startswith('A'):
  num=int(key[1]);kind=key.split('-')[1]
  arr=catalog['assignment_files']
  filt=[a for a in arr if f'assignment{num}' in a['repo'] and ('supplement' not in a['name']) and (('spring2024' in a['repo']) if kind=='2024' else ('/spring2025/' in a['url']) if kind=='2025' else ('spring2024' not in a['repo'] and '/spring2025/' not in a['url']))]
  a=filt[0];return f'[{key}]({a["html"]})'
 y,n=key.split('-');l=catalog['years'][y]['lectures'][int(n)-1]
 if l.get('download'):
  u=l['download'].replace('https://raw.githubusercontent.com/','https://github.com/')
  parts=u.split('/');u='/'.join(parts[:5])+'/blob/'+'/'.join(parts[5:])
 else:u=catalog['years'][y]['url']+'#schedule'
 return f'[{key}]({u})'

INTRO='''# CS336 AI 陪学资料

版本 3.1（完整双语教案版，公开默认英文） · 299点教案配套备课资料；原材料核验基线1.2 · 2026-10-22 · 独立原创学习资料，非 Stanford 官方教材或官方翻译。

本资料供 AI 陪学使用。学习者不需要先看课或完成大作业。导师每轮只讲一个点，等学习者解释后再决定下一点。本文件保存讲解底稿、课堂补充和提问素材，不应一次性展示给学习者。完整逐点教学入口是同目录《CS336-逐点陪学教案.md/html》，结构化内容在《CS336-逐点教案.json》。

## 资料范围和完成边界

| 部分 | 本版实际范围 | 限制 |
|---|---|---|
| 历年来源 | 2024–2026三年57个日程、51份核心讲义、17份作业PDF | 未公开课堂材料无法凭目录重建 |
| 录像文字 | 34份上传字幕及L08本地转写，35份文字轨道已按时间顺序校读 | 本地转写不是官方字幕 |
| 音频 | 35场原始英文音频；34场全程自动转写比对字幕并复核全部疑难窗口；L08完整双模型转写 | 属机器比对与文本语义检查，未人工逐句听校；仍有不可恢复提问 |
| PDF图表 | 42份1889页均查看版面与主图，逐页来源/渲染哈希、页范围内容说明齐全 | 未对每个小字、原始实验及作业实现作独立复现 |
| 可执行讲义 | 19份trace、2份stdout；312个图像来源、231张不同图；405个复合输出检查 | GPU计时和随机模型输出为作者缓存，不是本机复跑 |
| 演示画面 | 初轮22帧与追加136帧均查看，逐条保留只拍讲者/裁切等限制 | 关键时间点取样，不是逐帧完整视频审片 |
| 教学内容 | 299点目录、67核心解释块、334上下文说明、247全文校读补充、年度及基础补充，新增音频/PDF/图像纠错 | 用于逐点陪学，不能一次性倒给初学者 |
| 实验 | 10组CPU教学重建已运行 | 不等于GPU/大模型训练或性能复现 |
| 嘉宾 | 六场逐项追查；Dan Fu录像已纳入，Qwen/Llama等官方相关阅读已补 | 2024两场官方说明不录制；另三场完整课堂未取得，详见清单 |

本版已完成上表公开可得材料的补取和所列范围核验。完整课堂捕获仍有明确缺口，不称“所有发言、所有帧、所有细节已验证”。核验记录中的哈希只确认来源身份；内容判断另有原创说明。本公开资料不包含任何学习者的个人进度。

主线：概率与分词 → 张量和训练 → Transformer → 硬件和并行 → 推理 → Scaling → 评估与数据 → 后训练 → 多模态。历年重叠部分合并，年度新增与旧版独有部分保留。

## 给 AI 导师的使用约定

1. 先读本文件和课程目录，再确认当前知识点。把学习者视为需要补基础的人；不要要求先去看视频。
2. 每轮一个主要概念。先说要解决什么问题，再给具体例子。新术语先解释，公式先说明每个变量。
3. 只问一个理解问题，并等待回答。不要同一轮把答案和下一小节一起发出。判断依据是学习者的解释与迁移，不能只凭“懂了”或选择题命中。
4. 将原课例子、资料新增例子、已运行实验、未运行演示分开。不要把说明性数值称为实测。
5. 对有争议、听不清、图中不可读或未取得的内容，标出缺口。课程当年的判断保留年份；需要讲当前进展时查询一手新来源。
6. 不一次讲完一整讲。按 299 点目录定位，再从本文件相应解释、课堂条目与来源中选择一小段展开。
7. 大作业可以跳过实现，但其知识点保留。实验由导师按需要演示，不能强迫初学者先掌握框架安装与 GPU 运维。
8. 每次收尾记录：当前位置、学习者已能解释的内容、仍有疑问的内容、下一点。不要把“讲过”记录成“已掌握”。

## 如何读逐讲对照

“讲义主线”把对应知识改写为原创中文解释。“课堂展开”附可回到原视频的时间点；它可能扩展讲义已有概念，不保证是讲义完全没有的内容。“问答”只有在字幕有足够上下文时才复述。字幕缺失或问题听不清时不编造。

“演示重建”给出教学用途。CPU 运行结果可在实验文件中复核。其他图、调度表和 GPU 演示只说明可重建步骤；未核验画面的结论不当作实测证据。时间区间用于定位，不表示区间内每一句已完成听校。

若原课口述、初轮摘点与“字幕全文校读补充与纠错”不一致，按后者列明的依据纠正，不把口误作为课程结论。

## 一、2026 年主讲义与课堂对照
'''
lines=INTRO.strip().splitlines();review=[]
for x in NOTES_26:
 n=x['n'];v=by_video.get((2026,n));key=f'2026-{n:02}' if n<=17 else '2026-19'
 lines += ['',f'### 2026 L{n:02} · {x["title"]}','',f'对应陪学章节：{x["chapters"]}。官方讲义：{source(key)}。']
 if v:lines += ['',f'官方录像：[{v["title"]}]({v["url"]})。字幕状态：'+('已取得，未完成逐句听校。' if v['status']=='extracted' else '官方字幕未取得；已有双模型完整转写、讲义与疑难窗口交叉核对，不能恢复的原问保留缺口。' if v.get('local_asr') else '未取得，不代写本年度课堂内容。')]
 if n==18:lines += ['','编号说明：这是播放列表第 18 项，官网日程第 19 讲；不是 Daniel Selsam 那场日程条目。']
 lines += ['','讲义主线：','']
 for label,text in x['core']:lines += [f'- **{label}**：{text}']
 lines += ['','课堂展开、例子和讨论：','']
 refs=[]
 for start,end,kind,text in x['oral']:
  assert v and (v['status']=='extracted' or v.get('local_asr')) and start<end<=(v['local_asr']['last_end'] if v.get('local_asr') else v['last_caption_end_seconds'])+5
  lines += [f'- {video_link(v,start,end)} · {kind}：{text}'];refs.append({'start':start,'end':end,'type':kind,'finding':text})
 if not refs:lines += ['- 本讲字幕缺失；仅整理官方 PDF 的知识主线。']
 lines += ['', '二轮问答、例子与演示核对：', '']
 for f in [f for f in FINDINGS if f['lecture']==f'2026-{n:02}']:
  lines += [f'- {video_link(v,f["start"],f["end"])} · {f["finding"]}']
 lines += ['', '字幕全文校读补充与纠错：', '']
 for a,b,t in FULL_READINGS[f'2026-{n:02}']:lines += [f'- {video_link(v,a,b)} · {t}']
 lines += ['',f'问答处理：{x["qa"]}','',f'演示：{x["demo"]}','',f'理解检查：{x["check"]}','',f'缺口：{x["gap"]}','']
 review.append({'year':2026,'lecture_or_playlist_order':n,'schedule_lecture':19 if n==18 else n,'title':x['title'],'chapters':x['chapters'],'video_id':v['id'] if v else None,'status':'local_asr_and_lecture_material_crosscheck' if v and v.get('local_asr') else 'context_candidates_and_question_screen_reviewed' if refs else 'lecture_material_only_missing_captions','reference_windows':refs,'visual_status':'not_frame_verified','gap':x['gap']})

lines += ['## 二、2025 年逐讲补充：保留年度差异','',
'本节不是把 2025 课程简单删掉。17 份字幕均已建立分段导航；下列条目是与主讲义合并时保留的指定片段和教学差异。已读完全部 17 份字幕文字，并保留下面的时间段补充；听不清的发言和未显示画面仍不猜补。','']
for n,title,start,end,label,text,demo in NOTES_25:
 v=by_video[(2025,n)]
 assert start<end<=v['last_caption_end_seconds']+5
 lines += [f'### 2025 L{n:02} · {title}','',f'来源：{source(f"2025-{n:02}")} · {video_link(v,start,end)}。','',f'**{label}**：{text}','',f'例子／演示／合并处理：{demo}','',
 '核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。','']
 lines += ['二轮问答、例子与演示核对：','']
 for f in [f for f in FINDINGS if f['lecture']==f'2025-{n:02}']:
  lines += [f'- {video_link(v,f["start"],f["end"])} · {f["finding"]}']
 lines += ['', '字幕全文校读补充与纠错：', '']
 for a,b,t in FULL_READINGS[f'2025-{n:02}']:lines += [f'- {video_link(v,a,b)} · {t}']
 lines += ['']
 review.append({'year':2025,'lecture_or_playlist_order':n,'title':title,'video_id':v['id'],'status':'earlier_year_context_candidates_and_question_screen_reviewed','reference_windows':[{'start':start,'end':end,'type':'课堂补充','finding':text}],'visual_status':'not_frame_verified','gap':'未完成逐句听校和完整画面复核'})

lines += ['## 三、基础、旧版与作业中的必要补充','',
'以下部分让陪学内容不依赖完成大作业。旧版较细的内容按主题保留，不能只看新版标题就略去。2024 没有取得官方录像，所有 2024 说明均来自讲义／作业，不写成视频问答。','']
for sid,title,chapters,paras,refs,demo,check in SUPPLEMENTS:
 lines += [f'### {sid} · {title}','',f'对应章节：{chapters}。来源：'+ ' · '.join(source(k) for k in refs),'']
 for p in paras:lines += [p,'']
 lines += [f'演示入口：{demo}。','',f'理解检查：{check}','']

lines += ['## 四、演示文件与实际运行结果','',
'随包的 CS336-教学小实验.py 仅用 Python 标准库，无需网络、GPU、模型或训练数据。运行 python3 CS336-教学小实验.py 会打印结果，不写文件。输出也保存在 CS336-实验结果.json。以下是重新计算的结果，不是视频原始屏幕输出。','',
'| 实验 | 做什么 | 已运行的关键结果 | 边界 |','|---|---|---|---|']
exp=[
('D01','字节对合并',f'{demo_results["D01_BPE"]["initial_byte_tokens"]} 个初始字节 token → {demo_results["D01_BPE"]["final_tokens"]} 个 token，可逆解码','极小语料、允许跨空格合并，非生产分词器'),
('D02','因果 Attention','改变未来位置的 V 后，第一个位置输出仍为 [10,0]','固定 Q、K、V，非训练出的模型'),
('D03','softmax 与交叉熵','[1000,1001] 的稳定 softmax 约为 [0.268941,0.731059]；真实概率 0.2 的损失大于 0.8','自然对数，单样本示意'),
('D04','梯度累积','大 batch 与累积后一次更新均得 w=0.6；逐小批次更新得 0.56','均值损失、固定参数累积、无依赖 batch 的层'),
('D05','通信原语语义','四份输入 all-reduce 得 [6,10,14,18]，reduce-scatter 加 all-gather 同结果','列表计算，没有真实通信或带宽测量'),
('D06','LSH 与数据轮数','3×4 band／row 设置下，相似度 0.2 的候选概率约 0.00479，0.8 约 0.79420','理想独立哈希假设；不是确定去重阈值'),
('D07','组内优势与旧策略冻结','回报 [0,1,1,2] 中心化为 [-1,0,0,1]；固定旧分母时比率为 1 但导数约 0.119203','标量概率示例，非完整 PPO／GRPO 训练'),
('D08','递推稳定性','a=−2 时 8 次递推从 1 变 256；负号本身不保证收敛','不等于 Parcae 真实递推，也不证明非线性模型稳定'),
('D09','奖励漏洞反例','[1,1,2] 的错误排序 [1,2,2] 与正确排序都获包含性＋顺序奖励 5 分','教材新增反例，不声称讲者已经演示')]
for row in exp:lines.append('| '+' | '.join(row)+' |')

lines += ['','### 已核对的具体演示帧','',
'初轮以下22个时间点已读取并检查实际视频帧；追加136帧见第十一节。核对范围限于下述可见内容，不等于整段视频或所有屏幕变化已验证。原帧只保留在本地备课区，不随分享包分发。','',
'| 讲次与时间 | 画面确认的内容与边界 |','|---|---|']
frame_records=[]
for label,t,finding in FRAME_FINDINGS:
 year,n=map(int,label.split('-'));v=by_video[(year,n)]
 p=ROOT/'work/video-frames'/f'{label}-{t}.png';assert p.exists()
 sha=hashlib.sha256(p.read_bytes()).hexdigest()
 frame_records.append({'lecture':label,'video_id':v['id'],'timestamp':t,'frame_sha256':sha,'local_evidence_path':str(p.relative_to(ROOT)),'finding':finding,'scope':'single_frame_only'})
 lines.append(f'| {label} · {video_link(v,t)} | {finding} |')
extra_frames=json.loads((ROOT/'work/demo-audit/index.json').read_text())['frames']
seen_frames={(f['lecture'],f['timestamp']) for f in frame_records}
for f in extra_frames:
 key=(f['lecture'],f['timestamp'])
 if key in seen_frames:continue
 seen_frames.add(key)
 p=ROOT/'work/demo-audit/frames'/f"{f['lecture']}-{f['timestamp']}.png"
 if not p.exists():p=ROOT/'work/demo-audit'/f"{f['lecture']}-{f['timestamp']}.png"
 assert p.exists(),p
 assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
 frame_records.append({'lecture':f['lecture'],'video_id':f['video_id'],'timestamp':f['timestamp'],'frame_sha256':f['sha256'],'local_evidence_path':str(p.relative_to(ROOT)),'finding':DEMO_VISUAL_NOTES[f['lecture']]['finding'],'scope':'single_frame_only'})
for r in review:
 matches=[f for f in frame_records if f['lecture']==f'{r["year"]}-{r["lecture_or_playlist_order"]:02}']
 if matches:r['visual_status']='selected_frames_verified';r['frame_checks']=matches

lines += ['','## 五、必须明确保留的缺口','',
'| 缺口 | 范围与处理 |','|---|---|',
 '| 2026 L08 官方字幕 | 官方轨道仍返回 HTTP 429；已取得原始英文音频，并补本地自动转写。术语和问答听校仍是缺口，不用自动转写冒充官方字幕。 |',
'| 嘉宾一手资料 | 2024 Ce Zhang、Aakanksha Chowdhery：2024 L17第2页明确嘉宾不录制，独立讲义未取得；2025 Junyang Lin、Mike Lewis与2026 Daniel Selsam完整课堂未取得。相关官方论文不能代替本场问答。 |',
'| 2024 视频 | 官网未提供本次可确认的官方播放列表，不能声称该年录像已整理。 |',
'| 音频听校与字幕漏记 | 35 份现有字幕文字已顺序读完，补充与纠错逐讲列出；已对可取得原始音频做完整自动转写比对，疑难窗口再转写并检查语义；尚未人工逐句听校。INAUDIBLE等听不清片段、未被字幕记录的提问仍有缺口。文字读完不等于所有发言已恢复。 |',
'| 板书、屏幕与所有问答 | 文字能确认的操作已整理。关键演示帧、全部PDF版面主图和可执行讲义图像已检查；未显示/被裁切/细小不可读部分继续标缺，不由取样推出全部视频画面无遗漏。 |',
'| 源讲义逐页细节 | 已做结构与文本线索核对，并核对本文涉及的重点；PDF 共 1,889 页（核心讲义 1,439 页，作业 450 页）。每页版面主图已查看并有页范围说明；密集公式重点交叉读，未宣称全部脚注小字及引用实验已独立验证。 |',
'| GPU 复现 | 没有启动 GPU 训练、集群或大型 RL。10 个小实验只验证自己的教学重建，不验证课堂的加速倍数。 |',
'| 完整长上下文扩展配方 | 2026 L16 讲者明确未充分展开。已有概念不等于已有完整可复现教程。 |',
'| 后续发展与时点信息 | 2026 春季之后未做全领域更新审查。硬件、价格、榜单、法律和模型规格需要按教学需要另查一手来源。 |','',
'这些缺口不能由 AI 根据常识猜补。若问题落在缺口内，先说明证据边界，再取得来源或换用已核实的小例子。','',
'## 六、来源、版本与分享','',
'课程目录与来源索引保存三年逐讲链接、作业版本、字幕提取元数据与文件校验值。逐讲对照记录另存每个课堂条目的时间范围和核对级别，便于后续补缺。自动按词匹配的候选标题只用于导航，没有被称为人工语义验收。','',
'原始公开视频字幕和官方讲义保留在本地备课区，不打包分发全文。本资料包提供原创中文教学解释、来源链接、小实验与核对状态。分享不代表获得 Stanford 的背书或原资料再分发授权。','']

lines += ['## 七、官方执行记录、现场展示与纠错','',
'19 份官方 JSON trace 和 2 份多进程 stdout 已取得并核对 Git 对象哈希。JSON 内展示的数值、源码和图是作者缓存的执行过程。它们能补足单帧未显示的输出，但不能证明本机已经运行原始 GPU 程序。838个短标量/字符串已逐批检查；全部图像版面/主图、405个复合输出的检查与边界见第十二节。','',
'| 对应讲次 | 演示与处理 |','|---|---|']
for label,path,note in TRACE_NOTES:
 year=2025 if label.startswith('2025') else 2026
 repo='spring2025-lectures' if year==2025 else 'lectures'
 url=f'https://github.com/stanford-cs336/{repo}/blob/main/var/traces/{path}' if path.endswith('.json') else 'https://cs336.stanford.edu/spring2025/' if year==2025 else 'https://www.youtube.com/watch?v=9EEm4iMAF5s'
 lines += [f'| [{label}]({url}) | {note} |']
lines += ['', '### 新增 D10：在线 Attention 的分块重缩放','',
'为避免把“在线分母”误讲为“已经得到全部注意力输出”，小实验同时维护当前最大值 m、分母 l 和加权 V 的分子 o。遇到更大分数 m_new 时，旧 l 与旧 o 都先乘 exp(m−m_new)，再加入当前块。最终输出是 o/l。','',
'对分数 [1000,1002,999,1005,1001]，分块大小取 1、2、3、5 都与整行稳定 softmax 加权结果约 [-0.8810706584,3.7736455752] 一致，误差在浮点舍入范围内。这是固定一个 query 的 CPU 数学重建，不是 GPU 加速测试。','',
'## 八、六场嘉宾逐项核查','',
'检查日为 2026-10-22。未取得表示本次在所列公开入口没有拿到足够一手资料，不表示证明其从未公开。相关论文／其他课程演讲只作补充，不能替代本场问答。','',
'| 日程 | 嘉宾 | 材料状态与处理 |','|---|---|']
for g in GUESTS:
 rel=f' [相关一手入口]({g["related"]})。' if g['related'] else ''
 lines += [f'| [{g["year"]} L{g["number"]}]({g["source"]}) | {g["speaker"]} | {g["status"]}。{g["note"]}{rel} |']
lines += ['', '检索路径：三年官网、两个官方播放列表、三份官方仓库完整文件树、可确认的讲者主页、姓名＋课程号的公开视频／讲义搜索。未使用第三方 AI 笔记填补嘉宾的原始课堂内容。','']
lines += ['## 九、追加补源与逐页核验','',
'42份PDF的1889页均已查看版面和主图，并有逐页身份记录与页范围内容说明。2026 L08的73页说明如下；其余1816页见第十三节。细小图例和原论文实验的独立验证仍有限制。完整记录见 CS336-PDF逐页核验记录.json 与 CS336-PDF内容核对.json。','',
'补源纠错：二手文章给出的疑似 Mike Lewis 视频 r3DC_gjFCSA，经实际 YouTube 元数据核对，是 Joe Spisak 在 2024 年的会议演讲，不能纳入本课。第 8 讲官方字幕此次重查仍返回 IpBlocked，没有取得新官方字幕。','']
ar=json.loads((OUT/'CS336-音频核验记录.json').read_text())
lines += ['第 8 讲完整原始音频已用 small.en 二次转写，得到 1,826 段，末尾到 4,805.38 秒；与原 base.en 的 1,702 段作 161 个时间窗比对，并读取差异最大的 35 窗。相似度只帮助定位，不证明准确。两模型属于同一家族；课堂提问仍有不能恢复的部分，尚未完成逐句听校，其他34讲已另做全程自动比对与疑难窗口语义复核，见第十节。','']
for f in ar['findings']:lines += [f'- {video_link(by_video[(2026,8)],f["start"],f["end"])} · '+f['finding']]
lines += ['']
for pr in PDF_REVIEWS:
 lines += [f'### {pr["file"]} · 实际页面核对','']
 for f in pr['findings']:
  refs=' · '.join(f'[原始依据]({u})' for u in f.get('sources',[]))
  lines += [f'- 第 '+ '、'.join(map(str,f['pages']))+' 页：'+f['finding']+(' '+refs if refs else '')+(' 待核：'+f['remaining'] if f.get('remaining') else '')]
 lines += ['']

for r in review:
 key=f'{r["year"]}-{r["lecture_or_playlist_order"]:02}'
 r['context_review_findings']=[f for f in FINDINGS if f['lecture']==key]
 r['full_caption_text_review_complete']=True
 r['full_audio_listening_review_complete']=False
 r['full_audio_machine_comparison_complete']=True
 r['audio_comparison_ledger']='CS336-原始音频比对.json'
 r['full_reading_findings']=[{'start':a,'end':b,'finding':t} for a,b,t in FULL_READINGS[key]]
 r['status']='full_caption_text_reviewed_local_asr_low_confidence' if key=='2026-08' else 'full_uploaded_caption_text_reviewed'
 r['context_review_scope']='Original audio additionally machine-compared and flagged regions textually reviewed; details in separate audio ledger. All 278 extracted interaction/demo candidate contexts reviewed across 35 recordings; second pass screened 1252 uncovered question-mark positions and inspected 56 follow-up contexts. Combined with chronological reading of all remaining caption segments. Not full-audio review.'

append_audit(lines,ROOT,video_link,by_video)
md='\n'.join(lines).rstrip()+'\n';(OUT/'CS336-AI陪学资料.md').write_text(md)
review_doc={'version':'1.2-public-audit','checked_date':'2026-10-22','method':'Full retrieved-caption navigation; review of all 278 extracted interaction/demo contexts, screening of 1252 uncovered question-mark locations and 56 follow-up contexts; 334 authored context findings, initial and added frames, full PDF layouts/main figures, original-audio machine comparison and official execution-record checks. Complete chronological caption-text reading adds per-lecture corrections; not full audiovisual verification.','lecture_entries':review,'counts':{'retrieved_uploaded_captions':34,'local_asr_drafts':1,'all_text_segments':56527,'full_caption_text_reviews':len(FULL_READINGS),'full_reading_findings':sum(map(len,FULL_READINGS.values())),'segmentation_windows':1131,'new_year_lecture_entries':18,'earlier_year_entries':17,'core_explanations':67,'new_year_timestamped_entries':60,'supplement_groups':8,'executed_cpu_demos':10,'inspected_demo_frames':len(frame_records),'context_findings':len(FINDINGS),'official_trace_json':19,'official_stdout':2},'frame_checks':frame_records,'guest_checks':GUESTS}
(OUT/'CS336-逐讲对照记录.json').write_text(json.dumps(review_doc,ensure_ascii=False,indent=2))
(OUT/'CS336-实验结果.json').write_text(json.dumps(demo_results,ensure_ascii=False,indent=2))

prompt='''你是我的 CS336 陪学导师。先读取资料包中的《CS336-逐点陪学教案.md》当前点、《CS336-AI陪学资料.md》相关记录和《CS336-AI陪学目录.md》，遵守其中的使用约定和证据边界。把我视为学过入门课但需要从基础重学的人。你负责备课，我无需先看视频或做大型作业。每轮只讲一个小点，用一个具体例子，再问一个理解问题，等待我回答后再继续。不要一次总结整讲，不要提前展示问题答案。分清原课内容、资料新增例子、实际运行的小实验和未验证部分。遇到缺口先说明，不编造。先读取仓库根目录下的 local/learning-progress.md；不存在时从空白进度模板做基础诊断，不猜测旧聊天。接着待回答的问题继续；仅按实际回答确认掌握。3.1版299点双语教案齐备，按本点讲解与例子教学，提问后停止，参考答案与修复在作答后才使用。扩展推导可分多轮，材料缺口不编造。需要深入或更新时查所列一手来源。每次结束只更新被Git忽略的 local/learning-progress.md；绝不提交或自动推送个人进度。没有文件工具时给出完整更新文本，不声称已保存。换电脑由学习者自行携带该文件或明确选择独立私有存储。参照《换电脑继续学习.md》。'''
(OUT/'开始陪学.txt').write_text('使用方法：\n1. 解压资料包。\n2. 按《换电脑继续学习.md》交给AI目录、备课资料、核验说明和个人学习进度；首次学习用空白模板。需要实验时再交给它代码和结果。\n3. 发送下面这段话。公开可得材料核验版已完成，先读 CS336-核验进度.md 与缺口清单。\n\n'+prompt+'\n\n首次学习请复制《学习进度模板.md》为 local/learning-progress.md（仅本地、被Git忽略）。已有本地进度则保留，不能覆盖。\n')

def inline(s):
 s=html.escape(s)
 s=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',s)
 return re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',lambda m:f'<a href="{m[2]}" target="_blank" rel="noopener noreferrer">{m[1]}</a>',s)
parts=[];table=False;ul=False;anchors=[]
def close():
 global table,ul
 if table:parts.append('</tbody></table></div>');table=False
 if ul:parts.append('</ul>');ul=False
for line in lines:
 if line.startswith('|'):
  if re.match(r'^\|[-|: ]+\|$',line):continue
  cells=[x.strip() for x in line.strip('|').split('|')]
  if not table:
   close();table=True;parts.append('<div class="table"><table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in cells)+'</tr></thead><tbody>')
  else:parts.append('<tr>'+''.join('<td>'+inline(x)+'</td>' for x in cells)+'</tr>')
 elif line.startswith('- ') or re.match(r'^\d+\. ',line):
  if table:close()
  if not ul:parts.append('<ul>');ul=True
  parts.append('<li>'+inline(re.sub(r'^- |^\d+\. ','',line))+'</li>')
 else:
  close()
  if not line:continue
  m=re.match(r'^(#{1,3}) (.*)',line)
  if m:
   level=len(m[1]);anchor='sec-'+str(len(anchors));anchors.append((anchor,level,m[2]));parts.append(f'<h{level} id="{anchor}">{inline(m[2])}</h{level}>')
  else:parts.append('<p>'+inline(line)+'</p>')
close()
nav=''.join(f'<a class="level{level}" href="#{a}">{html.escape(t)}</a>' for a,level,t in anchors if level>=2)
css='''*{box-sizing:border-box}body{margin:0;color:#243a33;background:#f6f4ee;font:16px/1.85 system-ui,-apple-system,"PingFang SC",sans-serif}.layout{display:grid;grid-template-columns:270px minmax(0,1000px);max-width:1360px;margin:auto}aside{padding:28px 18px;position:sticky;top:0;height:100vh;overflow:auto;font-size:13px}aside a{display:block;padding:4px 0;text-decoration:none}.level3{padding-left:12px}main{padding:42px 35px 100px;min-width:0}h1{font-size:34px;line-height:1.4}h2{margin-top:60px;padding-top:14px;border-top:2px solid #c5d3c8;font-size:25px}h3{margin-top:42px;color:#205947;font-size:22px}h2,h3{scroll-margin-top:20px}a{color:#216b58;text-underline-offset:3px}p{max-width:85ch}li{margin:8px 0}ul{padding-left:24px}.table{overflow:auto;background:#fff;border:1px solid #dce3d9;border-radius:9px;margin:20px 0}table{width:100%;min-width:640px;border-collapse:collapse;font-size:14px;line-height:1.7}td,th{text-align:left;padding:12px;border-bottom:1px solid #e3e8df;vertical-align:top}th{background:#e6eee5}.status{padding:12px 16px;background:#e7e4d8;border-radius:8px;font-size:14px}@media(max-width:850px){.layout{display:block}aside{position:static;height:auto;max-height:260px;border-bottom:1px solid #d1dbce}main{padding:25px 18px}h1{font-size:27px}}@media print{aside{display:none}.layout{display:block}main{padding:0}body{background:white}table{font-size:10px;min-width:0}h2,h3{break-after:avoid}}'''
page='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CS336 AI 陪学资料 · 课堂对照版</title><style>'+css+'</style></head><body><div class="layout"><aside aria-label="资料导航">'+nav+'</aside><main><div class="status">公开可得材料核验版 · 35场音频机器比对 · 1889页版面主图 · 231张讲义图像 · 保留不可恢复/未公开项目</div>'+''.join(parts)+'</main></div></body></html>'
(OUT/'CS336-AI陪学资料.html').write_text(page)

catalog['learning_materials']={'version':'1.2-public-audit','file':'CS336-AI陪学资料.md','semantic_review_level':'complete_retrieved_caption_text_reading_plus_lecture_material_and_selected_demo_checks','full_caption_text_review_complete':True,'full_audio_listening_review_complete':False,'visual_review_complete':False,'pdf_visual_overview_pages':1889,'original_audio_machine_comparisons':35,'trace_unique_images_reviewed':231,'trace_compound_values_checked':405,'demo_results':'CS336-实验结果.json'}
for v in catalog.get('video_sources',[]):
    match=next((r for r in review if r['year']==v['year'] and r['lecture_or_playlist_order']==v['playlist_order']),None)
    if match:v['review_status']=match['status']
for r in review:
 y=str(r['year']);n=r.get('schedule_lecture',r['lecture_or_playlist_order']);l=catalog['years'][y]['lectures'][n-1]
 l['video_comparison_status']=r['status'];l['review_detail']='字幕全文已校读，逐讲结构、问答/演示候选和纠错均有时间戳；非完整音频听校或全画面核验'
execution_index=json.loads((ROOT/'work/traces/index.json').read_text())
execution_meta=[{k:v for k,v in x.items() if k!='scalar_outputs'} for x in execution_index]
(OUT/'CS336-执行记录索引.json').write_text(json.dumps({'checked_date':'2026-10-22','sources':execution_meta,'scope':'Official cached executions, not locally reproduced GPU runs; hashes checked against Git blob ids.','scalar_outputs_screened':sum(len(x.get('scalar_outputs',[])) for x in execution_index)},ensure_ascii=False,indent=2))
catalog['execution_sources']=execution_meta
catalog['audio_comparison_summary']={'ledger':'CS336-原始音频比对.json','counts':json.loads((OUT/'CS336-原始音频比对.json').read_text())['counts'],'manual_listening_complete':False}
catalog['additional_visual_evidence']={'pdf_ledger':'CS336-PDF内容核对.json','trace_images':'CS336-讲义图像核验.json','compound_values':'CS336-演示数组核验.json'}
catalog['guest_checks']=GUESTS
catalog['context_review_findings_count']=len(FINDINGS)
catalog['verification_completion_status']='publicly_accessible_audit_complete_with_documented_gaps'
catalog['pdf_page_inventory']={'total':1889,'core_lectures':1439,'assignments':450,'mechanical_inventory':'CS336-PDF逐页核验记录.json','content_findings':'CS336-PDF内容核对.json','visual_overview_pages':1889,'full_semantic_verification_complete':False}
catalog['coverage_limits']=['全部已取得字幕文字已校读；音频尚未完整听校，字幕漏记发言无法由文字恢复','关键帧取样不等于全帧验收；231张讲义图像主图已看，405复合输出检查不等于所有GPU程序复跑','2026 L08 官方字幕未取得；本地自动转写需听校','2024 录像及五场嘉宾完整一手课堂材料未取得','全部1889页版面/主图已看，密集公式重点核对；小字、引用实验未全部独立验证','尚未做 2026 春季之后全领域更新审查']
(OUT/'CS336-来源索引.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))

full_ledger=[]
for r in review:
 key=f"{r['year']}-{r['lecture_or_playlist_order']:02}"
 path=next((ROOT/'work/transcripts').glob(key+'-*.json'))
 raw=json.loads(path.read_text());segments=raw['segments']
 ranges=[(f['start'],f['end']) for f in FINDINGS if f['lecture']==key]
 prior=sum(any(a<=x['start']<=b for a,b in ranges) for x in segments)
 full_ledger.append({'lecture':key,'video_id':r['video_id'],'source_type':'local_asr_unverified' if key=='2026-08' else 'uploaded_caption','source_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'segments':len(segments),'previous_context_segments':prior,'chronological_remaining_segments':len(segments)-prior,'full_caption_text_review_complete':True,'full_audio_listening_review_complete':False,'notes':r['full_reading_findings']})
assert len(full_ledger)==35 and sum(x['segments'] for x in full_ledger)==56527
(OUT/'CS336-全文校读记录.json').write_text(json.dumps({'version':'1.2-public-audit','checked_date':'2026-10-22','method':'Previously reviewed interaction contexts plus chronological reading of every remaining caption segment. This ledger records authored review attestations and source identity; hash/count checks do not independently prove semantic correctness. No claim of full listening or all-frame inspection.','entries':full_ledger},ensure_ascii=False,indent=2))

files=['开始陪学.txt','换电脑继续学习.md','学习进度模板.md','CS336-AI陪学资料.md','CS336-AI陪学资料.html','CS336-AI陪学目录.md','CS336-AI陪学目录.html','CS336-来源索引.json','CS336-逐讲对照记录.json','CS336-课堂片段索引.json','CS336-教学小实验.py','CS336-实验结果.json','CS336-执行记录索引.json','CS336-全文校读记录.json','CS336-核验进度.md','CS336-PDF逐页核验记录.json','CS336-PDF内容核对.json','CS336-补源记录.json','CS336-音频核验记录.json','CS336-原始音频比对.json','CS336-讲义图像核验.json','CS336-演示数组核验.json']
with zipfile.ZipFile(OUT/'CS336-AI陪学资料包.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(OUT/f,'CS336-AI陪学/'+f)
print(json.dumps({'lesson_entries':len(review),'core_explanations':sum(len(x['core']) for x in NOTES_26),'classroom_entries':sum(len(x['oral']) for x in NOTES_26)+len(NOTES_25),'supplement_groups':len(SUPPLEMENTS),'cpu_demos':len(demo_results),'inspected_demo_frames':len(frame_records),'markdown_bytes':len(md.encode()),'package_files':len(files),'known_review_limits':True},ensure_ascii=False,indent=2))
