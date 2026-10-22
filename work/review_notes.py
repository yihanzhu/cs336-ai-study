"""Review findings authored from read transcript contexts, separate from mechanical navigation."""
import hashlib,json,pathlib
W=pathlib.Path(__file__).parent
FINDINGS=[];seen=set()
for p in sorted((W/'review').glob('findings*.tsv')):
 for line in p.read_text().splitlines():
  key,cid,finding=line.split('\t',2);cid=int(cid);d=json.loads((W/'review'/f'{key}.json').read_text());c=d['candidates'][cid-1];assert c['id']==cid and (key,cid) not in seen;seen.add((key,cid))
  FINDINGS.append({'lecture':key,'candidate_id':cid,'start':c['start'],'end':c['end'],'finding':finding,'evidence':'read_uploaded_caption_context' if key!='2026-08' else 'read_local_asr_context_crosschecked_with_handout','source_excerpt_sha256':hashlib.sha256(c['text'].encode()).hexdigest()})
for p in (W/'review').glob('20*.json'):
 d=json.loads(p.read_text());assert all((p.stem,c['id']) in seen for c in d['candidates'])
extras=json.loads((W/'review/extra-context.json').read_text());extra_lines=(W/'review/extra_findings.tsv').read_text().splitlines();assert len(extras)==len(extra_lines)
for line in extra_lines:
 i,finding=line.split('\t',1);i=int(i);c=extras[i-1]
 FINDINGS.append({**{k:c[k] for k in ['lecture','start','end']},'candidate_id':'followup-'+str(i),'finding':finding,'evidence':'read_uploaded_caption_context' if c['lecture']!='2026-08' else 'read_local_asr_context_crosschecked_with_handout','source_excerpt_sha256':hashlib.sha256(c['text'].encode()).hexdigest()})
FINDINGS.sort(key=lambda x:(x['lecture'],x['start']))
assert len(FINDINGS)==334
TRACE_NOTES=[
('2026 L01','lecture_01.json','字节、字符和 BPE 的编码/解码；示例 the cat in the hat 训练 3 次合并，之后编码新句子。当前词表示例与 2025 的 token 数不同，压缩率是具体 tokenizer 的结果。'),
('2025 L01','lecture_01.json','同一 Hello/地球 emoji/你好 字符串的字节与 token 个数不同；词级例子有 UNK 风险。预分词的空格边界与 BPE 合并规则分别讲解。'),
('2026 L02','lecture_02_recording.json','录制版实际使用 CPU，get_promised_flop_per_sec 在无 CUDA 时回传占位数 1，导致显示极大的 mfu。此数不具有 GPU 利用率含义，不能当硬件跑分。'),
('2026 L02','lecture_02.json','当前版与录制版完整源码 diff 已读。当前版补两周预算中的 ×2、修正 ReLU 算术强度注释 1/4、按参数个数计梯度/状态字节、给 backward 标量化、改 checkpoint 复杂度 O(L)。同一网页的新版内容不是录像原文。'),
('2026 L02','lecture_02.json','当前 trace 的 activation_checkpointing 中 memory=202114048、checkpointed_memory=202900480，示例并未显示显存下降。只能据此说明一次记录；测量包含常驻/临时开销、模型小和切分方法等，不能宣称已经验证理想节省。'),
('2025 L02','lecture_02.json','录制例子同样为 CPU 占位峰值。16 bytes/parameter 的状态配方与 2026 的 12 bytes 配方不同；都还未计全激活等显存，不可直接用作最大可训练参数量。'),
('2026 L06','lecture_06.json','完整列入 GELU、softmax、row sum、matmul+ReLU 的源码路径与检查步骤。trace 中朴素、内置、compiled GELU 的排序与口述一致；只保留本次排序，不给跨设备速度保证。'),
('2025 L06','lecture_06.json','旧版还有手写 CUDA GELU、Nsight 与 torch.compile/Triton 比较。该 trace 中手写 CUDA/Triton GELU 仍慢于 built-in；softmax 的 compiled/内置/手写排序也不同。先检查正确性，再用同形状同精度计时。'),
('2026 L07','lecture_07.json','单步 trace 与多进程 stdout 分开。四 rank 的 all-reduce 与 reduce-scatter/all-gather 数值一致；DDP 归约梯度，TP/PP 示例主要是前向，不称完整分布式反向训练。'),
('2025 L08','lecture_08.json','旧版同样保留 communication、DDP、TP 与 PP 示例；局部 loss 不同但同步后参数一致。stdout 只是作者记录，不能证明本机有相同吞吐。'),
('2026 L10','lecture_10.json','prefill/decode 算术强度、KV 缓存、GQA/MLA、剪枝、speculative sampling、continuous batching、paged attention 都有讲稿入口；其中部分是图或公式推演，不是完整服务系统跑通。'),
('2025 L10','lecture_10.json','公式示例 batch=256 的预估 memory=240779264000 字节，已超过所说单张 80GB H100 的容量。它是代入模型的外推点，不能当可部署配置或实测吞吐。'),
('2026 L12／2025 L12','lecture_12.json','评估课的网页、示例题、Arena、agent、安全图是原理与案例展示。不是本资料重跑过的榜单；题目提取、工具权限、采样预算和模型版本都会改变结果。'),
('2026 L13／2025 L13','lecture_13.json','数据集和网页举例保留来源与当年状态；看到仓库或论文链接不表示下载并审计过整个训练语料。版权和访问条款要另外核实。'),
('2026 L14','lecture_14.json','变换、过滤、Jaccard/MinHash/LSH、混合和后训练数据都有对应函数。Jaccard 示意 3/5；LSH 3×4 分组图与后续 5×10 参数实验不是同一次设置。小来源 50 轮是风险例子。'),
('2025 L14','lecture_14.json','KenLM、fastText、DSIR、语言/毒性过滤、Bloom、MinHash/LSH 的示例逐项登记。DSIR 的一次概率为 0；分类器低分和数值近 1 的输出都不代表经过语义真值确认。固定随机种子/稳定哈希和平滑需另外定义。'),
('2025 L15','lecture 15.pdf','课堂成对判断练习由字幕确认：较长回答可能隐藏事实或数学错误。没有取得可完整核对的题目—投票—答案键，保留该画面细节缺口，不能补造 27 人完整投票表。'),
('2025 L17','lecture_17.json','两状态例子实际调用 torch.std，因此 5.3229 与 1.1547 是样本标准差，非方差。变量名/口述需纠正。冻结分母示例的梯度从 0 变约 0.119203，与 D07 相符。'),
('2025 L17','lecture_17.json','排序教学模型各位置独立输出，不能当作自回归 Transformer。两种奖励、baseline、归一化、旧概率停止梯度、重要性比、clipping、reference KL 和训练日志均保留。D09 是本文新增的奖励漏洞反例。'),
('2026 嘉宾 Dan Fu','视频及相关作者论文/代码','请求生命周期、缓存卸载、Megakernel 时间线和循环模型是这场录像的四段主线。已核对时间线和线性递推帧；Parcae 论文与 Megakernels 代码另作一手补充，不冒充缺失幻灯片 PDF。')]
