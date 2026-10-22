import json,pathlib,re,html,ast
BASE=pathlib.Path(__file__).parent.parent
OUT=BASE/'outputs'; SRC=BASE/'work/sources'
manifest=json.loads((SRC/'manifest.json').read_text())
assign=json.loads((BASE/'work/assignments_all.json').read_text())+json.loads((BASE/'work/assignments25.json').read_text())
video_index=BASE/'work/transcripts/index.json'
videos=json.loads(video_index.read_text()) if video_index.exists() else []
video_by_lecture={}
for v in videos:
    match=re.search(r'(?:Lecture|Lec\.?)\s*(\d+)',v['title'],re.I)
    n=int(match.group(1)) if match else (19 if v['year']==2026 and 'Dan Fu' in v['title'] else None)
    if n is not None:video_by_lecture[(str(v['year']),n)]=v

# Each source key is YEAR-LECTURE. Chapters may reorder the lecture for a beginner.
modules=[
('00','随用随补的基础','读懂课程中必要的数学和程序表示。','数字与变量；函数和参数；向量与矩阵；矩阵乘法；概率分布与条件概率；对数与指数；导数、梯度与链式法则；训练集、验证集和测试集','2024-02 2025-02 2026-02','为什么概率最大的结果也可能没有发生？'),
('01','语言模型在做什么','区分预测概率、生成文字和训练。','下一 token 预测；条件概率；序列概率；训练与推理；自回归生成；预训练与后训练；模型历史和开放程度；固定数据与算力下的效率问题；小模型经验的适用边界','2024-01 2025-01 2026-01','预测一种续写的概率，与保证它会出现，有什么区别？'),
('02','文字如何变成 token','理解分词的作用和取舍。','Unicode 与 UTF-8；字符、字节、词和子词；token ID 与词表；BPE 的训练与编码、解码；特殊 token 和预分词；Unigram 与 SentencePiece；规范化、空格、数字和多语言；压缩率与词表大小；字节级和无分词器路线','2024-01 2024-03 2024-04 2025-01 2026-01','为什么一个汉字不一定恰好对应一个 token？'),
('03','张量与最小训练程序','能跟随模型中的数据形状。','形状、维度与 batch；dtype 和设备；存储、stride、view 与切片；广播、逐元素运算和矩阵乘法；einsum、einops 和形状标注；参数、Embedding 与 Module；自动求导；数据加载、随机种子与复现；保存与恢复 checkpoint','2024-02 2025-02 2026-02','两个张量形状相同，为什么仍可能有不同的内存布局？'),
('04','模型怎样从错误中学习','走通一次参数更新。','logits 与 softmax；交叉熵及数值稳定；梯度下降与反向传播；SGD、动量、AdamW；初始化；学习率 warmup、cosine 与 WSD；梯度裁剪；batch 和梯度累积；训练循环；验证曲线、过拟合与调参；消融实验','2024-02 2025-02 2026-01 2026-02 2026-03 2026-11','怎样区分模型能力不足和训练方法没有调好？'),
('05','从零理解 Transformer','跟随一个 token 通过整个模型。','Embedding 与线性层；Q、K、V；点积、缩放和 softmax；因果 mask；多头自注意力；位置表示；MLP／FFN；残差连接；归一化；Transformer block 的堆叠；输出投影与概率；并行训练与逐步生成','2024-03 2025-03 2026-03','为什么训练时必须遮住未来的 token？'),
('06','现代模型结构与超参数','理解每种设计解决的问题。','pre-norm 与 post-norm；LayerNorm 与 RMSNorm；double norm 和非残差 post-norm；ReLU、GELU、GLU 与 SwiGLU；移除 bias；串行与并行 block；RoPE；模型深度、宽度、头数和 FFN 比例；词表大小；dropout 与 weight decay；z-loss、QK norm 和 logit soft-capping','2024-03 2024-04 2025-03 2026-03','一个结构修改提升了速度，是否一定也减少了 FLOPs？'),
('07','Attention 的成本与替代路线','理解长上下文带来的问题。','全注意力的计算和缓存成本；MQA 与 GQA；MLA 与跨层缓存共享；滑动窗口和稀疏注意力；全局与局部层交错；线性注意力及递推形式；Mamba／状态空间模型；Gated DeltaNet；混合结构；稀疏索引器与动态稀疏注意力；质量、内存和速度的取舍','2024-03 2024-04 2025-04 2025-10 2026-03 2026-04 2026-10','线性注意力减少计算后，为什么还需要比较模型质量？'),
('08','混合专家 MoE','区分总参数和每次激活的参数。','稠密与稀疏计算；router、top-k 与专家选择；token-choice 和 expert-choice；细粒度和共享专家；路由噪声与可微训练；负载均衡辅助损失；无辅助损失的偏置调整；capacity、通信和专家并行；路由稳定性；upcycling；DeepSeek 等公开案例及经验边界','2024-04 2025-04 2026-04 2026-08','总参数翻倍，单个 token 的计算量是否一定翻倍？'),
('09','计算量、显存与性能','能判断主要资源花在哪里。','参数量与 FLOPs；前向和反向的成本；训练计算量估算；参数、梯度、优化器状态和激活内存；FP32、FP16、BF16、FP8 和 FP4；混合精度；算术强度；带宽瓶颈与算力瓶颈；roofline；MFU；重计算和梯度累积的成本','2024-02 2025-02 2026-02 2026-05','GPU 还有空闲算力，为什么程序仍然可能很慢？'),
('10','GPU、TPU 与硬件执行','建立硬件性能的直觉。','CPU 与 GPU；SM、warp、thread 和 block；SIMT 和分支分歧；寄存器、共享内存、缓存和 HBM；Tensor Core；TPU 与矩阵计算；低精度硬件；算力与带宽增长不一致；内存合并访问；tiling、对齐和 wave quantization','2024-05 2025-05 2026-05','为什么更少的显存读写，可能比更少的算术运算更有价值？'),
('11','性能分析、编译与 GPU kernel','理解怎样定位并改善慢操作。','可靠计时、预热和同步；端到端基准与 profiler；Nsight、火焰图和内存剖析；算子融合；CUDA、PTX 与 Triton 的层次；torch.compile；GELU、softmax、行求和和矩阵乘示例；分块与 mask；自定义 autograd；RMSNorm 前向与反向；正确性和速度分开验证','2024-06 2025-06 2026-06','单个 kernel 更快，为什么端到端训练不一定明显变快？'),
('12','FlashAttention 与重计算','理解精确注意力怎样节省内存。','普通 Attention 的中间矩阵；HBM 与片上内存；分块；在线 softmax；数值稳定的增量统计；融合前向计算；保存必要统计量；反向重计算；FlashAttention-2 的工作划分；精确计算与近似注意力的区别','2024-05 2025-05 2026-05 2026-06','FlashAttention 为什么可以不保存整个注意力矩阵？'),
('13','多设备通信','先理解设备怎样交换数据。','rank、world size 和进程组；broadcast、reduce、all-reduce；all-gather、reduce-scatter 和 all-to-all；环式通信；延迟与带宽；NCCL 和 torch.distributed；同步与异步；通信和计算重叠；节点内与节点间互连；拓扑与 TPU／GPU 网络差异','2024-07 2024-08 2025-07 2025-08 2026-07 2026-08','all-reduce 与 reduce-scatter 加 all-gather 有什么关系？'),
('14','分布式训练','理解每种并行方式分割什么。','数据并行 DDP；梯度平均与通信 bucket；ZeRO 1／2／3 与 FSDP；优化器、梯度和参数分片；张量并行的行列切分；流水线、microbatch 和气泡；1F1B 与 zero-bubble；序列和上下文并行；专家并行；激活重计算；组合并行和公开模型案例；通信量与显存估算','2024-07 2024-08 2025-07 2025-08 2026-07 2026-08','增加 GPU 后训练不再加速，你会先检查什么？'),
('15','推理与服务','理解模型生成答案时的成本。','自回归采样；temperature、top-k 和 top-p；prefill 与 decode；KV cache；延迟与吞吐量；批量大小和内存带宽；量化及 AWQ；剪枝；缓存压缩与结构改动；推测采样及验证；连续批处理；PagedAttention；动态请求管理；vLLM 的用途','2025-10 2026-10','为什么输入处理和逐 token 输出可能需要不同优化？'),
('16','Scaling laws 的基础','会读曲线，并理解外推的限制。','幂律和对数坐标；数据、参数和计算量的关系；样本复杂度和均值估计；内在维度解释；分布变化；数据重复和有限数据；模型结构、深宽比及优化器的 scaling；临界 batch；Kaplan 与 Chinchilla；最小包络、IsoFLOPs 和联合拟合；拟合误差；训练最优与部署最优','2024-09 2025-09 2026-09','同一训练预算下，为什么更大的模型不一定更好？'),
('17','把小实验扩展到大模型','理解超参数迁移与实验设计。','初始化、学习率和 batch 的共同变化；μP 的基本条件与推导；宽度迁移；标准参数化对照；RMSNorm、优化器和 weight decay 的适用边界；MiniCPM、DeepSeek、CerebrasGPT 等公开配方；WSD 和训练复用；Muon 的尺度依赖；多尺度实验与预算分配；误差和外推风险','2024-10 2025-11 2026-11','小模型上的最好设置，为什么不能直接照搬到大模型？'),
('18','怎样评价模型','先定义能力，再选评价方法。','负对数似然、perplexity 和分词依赖；选择题、知识、数学和代码测试；few-shot 与提示敏感性；指令遵循、聊天偏好和 Arena；LLM-as-judge；长度、位置和风格偏差；推理和智能体评测；安全、偏见与公平；训练测试污染及检测局限；饱和、效度、真实任务和统计不确定性','2024-17 2025-12 2026-12','两个模型的 perplexity，什么情况下不能直接比较？'),
('19','训练数据从哪里来','认识数据来源及其局限。','网页、Common Crawl 与 WARC；Wikipedia、代码、论文、书籍和问答；WebText、C4、The Pile、MassiveText、RefinedWeb、Dolma、DCLM、Nemotron-CC、Common Pile 与 The Stack；来源、语言、领域和时间；开放权重与开放数据的区别；版权、许可、隐私与可追溯性问题','2024-11 2024-13 2025-13 2026-13','数据量相同，为什么不同来源可能训练出不同能力？'),
('20','数据转换与过滤','理解数据质量的操作含义。','HTML 转文本与正文抽取；语言识别；规则过滤；质量与有害内容分类；个人信息处理；KenLM、fastText 和 DSIR；重要性采样；模型评分；误删与漏删；过滤对领域和人群的影响；通过训练实验验证过滤效果','2024-12 2025-14 2026-14','删除低分文本，为什么可能损失有用的训练信号？'),
('21','数据去重','分清完全重复和近似重复。','哈希及碰撞；文档、段落和行级精确去重；Bloom filter 和误判率；后缀数组与重复子串；集合、n-gram 和 Jaccard；MinHash；LSH 的 band／row 参数；扩展到大数据的计算成本；去重与评测污染的关系','2024-13 2025-14 2026-14','哈希一致和语义相似，是同一件事吗？'),
('22','数据混合与训练阶段','理解训练数据怎样随目标变化。','数据配比与消融；领域重采样；数据选择和 scaling；数据混合估计；重复数据与训练轮次；长上下文数据；代码 infilling；特定领域与推理数据；合成数据与自蒸馏；midtraining；后训练数据质量及验证','2024-14 2025-13 2026-14 2026-15','合成数据增加了数量，怎样判断它也增加了有效信息？'),
('23','监督微调 SFT 与参数高效微调','理解模型怎样学会遵循指令。','预训练目标与指令目标；FLAN、Alpaca、OpenAssistant 等数据；对话格式与 loss mask；小量高质量数据；风格、事实与知识提取；幻觉与安全微调；训练设置和评估；PEFT；LoRA 的低秩更新；QLoRA、NF4 与双重量化；冻结参数与完整微调的取舍','2024-14 2024-15 2025-15 2026-15','模型回答风格变好，是否说明它学到了更多事实？'),
('24','人类偏好与 RLHF','理解奖励怎样改变模型输出。','成对偏好与奖励模型；标注者、众包伦理和人口差异；AI 反馈与 Constitutional AI；生成和评价的差距；策略、回报和 KL 约束；PPO、value、advantage 和 GAE；DPO 的目标和推导；变体与比较；奖励过优化、长度偏差、模式坍塌和校准；安全红队测试','2024-16 2025-15 2025-16 2026-15 2026-16','奖励分数不断上涨，为什么真实回答质量可能下降？'),
('25','推理强化学习 RLVR','理解可验证奖励、长推理和测试时计算。','可验证奖励与评分器；语言模型作为策略；轨迹、回报与策略梯度；baseline 与方差；GRPO 及组内归一化；序列长度和难度偏差；on-policy 与 off-policy；重要性重加权和 clipping；Dr. GRPO、GSPO、MaxRL；拒绝采样与 expert iteration；R1／R1-Zero、Kimi、Qwen 案例；蒸馏；test-time scaling；智能体环境、轨迹和 RL 基建','2025-16 2025-17 2026-16','答案可以自动判分，是否就代表奖励设计没有漏洞？'),
('26','用一个小例子走通强化学习','把抽象算法对应到一次更新。','2025 年官方排序玩具任务；生成回答；计算 token log-prob；设计排序奖励；策略梯度 loss；KL penalty；参数冻结；训练前后概率变化；采样与更新循环；不同实验设置的比较','2025-17','同一个回答获得正奖励后，哪些概率应该发生什么变化？'),
('27','多模态模型','把文字模型的认识扩展到图片与视频。','图像编码器；图文对比学习；CLIP 与 SigLIP；视觉 token 和连接器；LLaVA 与视觉指令微调；OneVision 与可变分辨率；Qwen-VL 系列；视频与时间位置；原生多模态与 Chameleon；理解与生成目标的差异','2026-17','图片进入语言模型之前，为什么也需要变成一种数值表示？'),
('28','嘉宾讲座与课程后的更新','保留公开内容入口，并明确材料缺口。','六场历年嘉宾讲座逐项登记；已取得材料后再拆知识点；更新按架构、数据、系统、评估和后训练归位；区分课程当时的说法和后续证据；不开列没有读过的讲座细节；不把模型排行榜当作长期知识','2024-18 2024-19 2025-18 2025-19 2026-18 2026-19','一个新结果需要满足什么证据条件，才值得修改已有认识？')
]

module_sources={}
for mid,title,goal,points,refs,check in modules:
    for key in refs.split(): module_sources.setdefault(key,[]).append(mid)

for year in manifest['years']:
    for l in manifest['years'][year]['lectures']:
        key=f'{year}-{l["number"]:02}'
        assert key in module_sources,key
        l['chapters']=module_sources[key]
        f=next((x for x in manifest['files'] if str(x['year'])==year and x['number']==l['number']),None)
        l['material_status']='已取得官方讲义，已核对目录线索' if f else '官网有日程，未取得独立讲义／完整字幕'
        if f: l['download']=f['url'];l['sha256']=f['sha256']
        v=video_by_lecture.get((year,l['number']))
        if v:
            l['video_url']=v['url'];l['caption_status']=v['status'];l['video_comparison_status']='未逐讲完成对照'
            if not f and v['status']=='extracted':l['material_status']='字幕全文及课堂补充已核对；无独立讲义'

def source_link(key):
    y,n=key.split('-'); l=manifest['years'][y]['lectures'][int(n)-1]
    u=l['links'][0]['url'] if int(n)<=17 and l['links'] else manifest['years'][y]['url']+'#schedule'
    return f'[{y} L{int(n):02}]({u})'

assignment_mapping={1:'02–06、09、15',2:'09–14',3:'16–17',4:'19–22',5:'18、23–26'}

lines=['# CS336 AI 陪学：完整课程目录与来源','',
'版本：0.1 · 来源核对日期：2026-10-22 · 对象：学过入门课、希望从基础重学的人。','',
'这是一份课程目录与材料索引。3.1版配套《CS336-逐点陪学教案.md/html》，原目录299点均有原创教学设计；原始课堂材料仍有缺口，不代表所有录像细节已恢复。','',
'## 1. 覆盖范围与核对结果','',
'| 范围 | 数量 | 本次状态 |','|---|---:|---|',
'| 2024、2025、2026 官网日程 | 3 年 × 19 讲 = 57 个条目 | 全部登记并映射到章节 |',
'| 每年前 17 讲的核心讲义 | 51 份 | 已下载；核对 Python 的函数／标题结构和 PDF 的逐页文本线索 |',
'| 2024 作业、2025 固定分支作业、当前作业 | 5 + 6 + 6 = 17 份 PDF | 已取得并核对章节结构；包含两份年度安全对齐补充材料 |',
'| 嘉宾讲座 | 6 个日程条目 | 逐项保留；尚未取得足够一手内容，不能声称内容完整核对 |',
'| 录像中的口头补充和问答 | 未逐段核对 | 逐讲备课时补查；不能用讲义替代录像覆盖证明 |','',
'因此，本版已完成公开日程和核心讲义层面的覆盖映射。尚不能承诺“视频里的每一句和所有嘉宾内容都已覆盖”。章节是教学重排，不是 Stanford 官方章号。','',
'官方入口：[2024](https://cs336.stanford.edu/spring2024/) · [2025](https://cs336.stanford.edu/spring2025/) · [2026](https://cs336.stanford.edu/) · [2025 官方录像](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) · [2026 官方录像](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)','',
'## 2. AI 陪学如何使用这份目录','',
'把本目录、逐点陪学教案、课堂备课资料和个人学习进度交给AI。先教当前点，提问后等待回答。每点已有教案，需要深入或更新时再查对应一手来源。','',
'1. 每轮只引入一个主要概念。先说明它解决的问题，再给具体例子。必要的数学当场补。',
'2. 讲完后只问一个能暴露理解的小问题。等待学习者回答。不要在同一轮给出答案或继续下一概念。',
'3. 根据回答解释正确之处和缺口。答对选择题不等于掌握；需要时请学习者用自己的话解释，或换一个情境。',
'4. 学习者理解后再前进。不要用机械考试拖慢已经理解的内容；学习者可以要求跳过或深入。',
'5. 不要求完成大作业。保留作业承载的知识点，用短推演、示意图或可选小实验来讲。',
'6. 不把类比当作实际机制。说明简化之处；公式先讲变量含义，再讲关系。',
'7. 对旧材料中的模型性能、硬件规格、成本和法律背景标明日期。需要讲当前情况时，查一手新来源。未证实的变化标为待核实。',
'8. 每次结束记录：当前位置、学习者已能解释的内容、仍有疑问的内容、下一点。不要把“已经讲过”写成“已经掌握”。',
'9. 本资料使用中文原创讲解并链接来源。它不是 Stanford 官方课程、官方翻译或认证课程。分享时不复制整份讲义或未授权录像。','',
'## 3. 全部知识目录','',
'学习顺序：语言模型与分词 → 张量与训练 → Transformer → 系统与推理 → Scaling → 评估与数据 → 后训练 → 多模态。第 00 章随用随补。下列问题用于 AI 备课，不是一次性布置给学习者的作业。','']

for mid,title,goal,points,refs,check in modules:
    lines.extend([f'### {mid} · {title}','',goal,''])
    for i,p in enumerate(points.split('；'),1): lines.append(f'- {mid}.{i:02} {p}')
    lines.extend(['','来源：'+' · '.join(source_link(k) for k in refs.split()),'',f'理解检查示例：{check}',''])

lines+=['## 4. 逐讲对应表：57 个日程条目','',
'“已取得”指材料可读取并完成目录核对，不指已逐句精读或已写完陪学内容。原官网的讲者个人主页不当作该讲讲义。','']
for year,d in manifest['years'].items():
    lines +=[f'### {year} 年','', '| 讲次 | 官网主题 | 官方材料 | 陪学章节 | 状态 |','|---|---|---|---|---|']
    for l in d['lectures']:
        n=l['number']; key=f'{year}-{n:02}'
        links=[]
        if n<=17:
            links=[f'[{x["label"]}]({x["url"]})' for x in l['links']]
            if l.get('download') and all('github.com' not in x['url'] for x in l['links']):
                u=l['download'].replace('https://raw.githubusercontent.com/','https://github.com/')
                parts=u.split('/');u='/'.join(parts[:5])+ '/blob/'+'/'.join(parts[5:])
                links.append(f'[讲稿源文件]({u})')
        else:
            links=[f'[官网日程]({d["url"]}#schedule)']
            if year=='2026' and n==19: links.append('[录像入口](https://www.youtube.com/watch?v=9EEm4iMAF5s)')
        if n<=17 and l.get('video_url'):links.append(f'[课堂录像]({l["video_url"]})')
        lines.append(f'| {n:02} | {l["title"]} | {" · ".join(links)} | {"、".join(l["chapters"])} | {l["material_status"]} |')
    lines.append('')

lines+=['## 5. 作业中的知识点与原始材料','',
'作业作为知识来源。学习者无需实现整份代码、参加排行榜或购买 GPU。旧版与新版的实验目标有变化，保留各年份来源。2025 使用已核实存在的 spring2025 分支，避免把当前 main 内容误标为旧版。','',
'| 年份／版本 | 材料 | 对应章节 |','|---|---|---|']
for a in sorted(assign,key=lambda x:(('2024' if 'spring2024' in x['repo'] else '2025' if '/spring2025/' in x['url'] else '2026'),x['repo'],x['name'])):
    year='2024' if 'spring2024' in a['repo'] else '2025' if '/spring2025/' in a['url'] else '2026／当前'
    n=int(re.search('assignment(\d)',a['repo']).group(1))
    lines.append(f'| {year} | [{a["name"]}]({a["html"]}) | {assignment_mapping[n]} |')
lines+=['',
'必须保留的年度差异：','',
'- 2024 A2 以 RMSNorm kernel 为核心；2025／当前 A2 转向 FlashAttention-2，当前版还含 activation checkpointing、FSDP 和并行策略分析。',
'- 2024 L15 对 LoRA、QLoRA、NF4 与双重量化有专门展开；归入第 23 章。',
'- 2024 L12–14 的 KenLM、fastText、DSIR、Bloom filter、后缀数组、长上下文、infilling 和自蒸馏不能因新版压缩讲次而漏掉。',
'- 2025 L17 的排序任务与策略梯度实验归入第 26 章；2026 L17 已换为多模态，二者都保留。',
'- 2025 A5 包含 MATH、SFT、expert iteration、策略梯度和 GRPO；当前 A5 另列 Dr. GRPO、MaxRL、off-policy 和 GSPO。安全 SFT／DPO 补充材料也保留。',
'- 2026 新版的 attention 替代路线、TPU、低精度、专家并行、智能体 RL 和多模态归入相应章节。案例只代表对应材料时点。','',
'## 6. 材料缺口、链接修正与更新边界','',
'| 项目 | 当前证据 | 处理方式 |','|---|---|---|',
'| 2024 Ce Zhang、Aakanksha Chowdhery | 官网列有姓名，未链接独立课件 | 保留第 28 章，待一手课件／录像到位后拆点 |',
'| 2025 Junyang Lin、Mike Lewis | 官网列有姓名，未链接独立课件 | 同上；不根据讲者身份猜讲题 |',
'| 2026 Daniel Selsam | 官网日程存在；本次未验证到可用独立讲义／完整录像 | 明确待补，不能据此说讲座不存在或未举行 |',
'| 2026 Dan Fu | 已定位官方频道录像入口；本次读取未取得完整字幕 | 只登记来源，未用二手笔记充当已核对的一手内容 |',
'| 2026 L02 recording version | 官网另列可执行 trace；仓库无同名 Python 源文件 | 已读取 trace 内录制版源码，并与当前主讲稿逐项比较；保留修正口径 |',
'| 2026 安全补充 PDF | 官网仍指向 main 下的 2025 文件名；仓库现有 2026 文件 | 本目录使用仓库实际存在的 2026 PDF，并保留 spring2025 历史版 |',
'| 公开视频播放列表 | 入口已核对；本次未可靠枚举列表内全部条目 | 不用日程讲数推断实际公开录像数 |',
'| 2026 春季之后的进展 | 本次没有做全领域最新论文审查 | 逐章备课时按需核实；不会把“新版课程”称为覆盖截至今日所有进展 |','',
'更新重点：长上下文与混合架构、MoE 与低精度、推理服务、数据混合、推理／智能体 RL、评测污染和多模态。只有能说明现有知识改变在哪里的更新才纳入。保留原始来源日期和验证状态。','',
'## 7. 示范小节：01.01，预测与概率','',
'目标：学习者能区分“最可能出现”和“一定出现”。先不引入神经网络结构。','',
'AI 第一轮只展示：','',
'> 看到“我把牛奶放进了＿＿”，你可能想到“冰箱”，也可能想到“杯子”。我们先把完整的续写当作候选项，暂时省去实际的分词过程。假设有 100 条相同开头的例句，其中 60 条接“冰箱”、25 条接“杯子”、15 条接其他内容。这是人为设计的例子。现在从中随机抽一条，你认为最值得猜哪个续写？这是否意味着它一定出现？请说说理由。','',
'然后等待回答。不要一次性讲完下面的反馈分支。','',
'- 若学习者区分了概率与必然：确认其理由，再换一个简单情境检验是否迁移；之后进入模型输出概率。',
'- 若认为最高概率等于必然：用剩下 40 条例句追问，先修正这个误解。',
'- 若认为其他续写出现就证明模型全错：区分一次猜测是否命中，与概率分配是否合理。后续再讲如何通过许多例子评价概率。',
'- 若质疑例句频率是否就是模型机制：肯定该质疑。说明这里只帮助理解概率，神经模型会通过参数学习规律，并非只数相同句子。',
'- 若学习者已有扎实基础：缩短该小节，不强制重复；下一点讲条件概率和上下文。','',
'结束记录模板：当前位置［填写］；已确认掌握［附真实回答证据］；待澄清［填写］；下一点［填写］。个人记录仅保存到被Git忽略的 local/learning-progress.md，不提交或自动上传。','',
'## 8. 核验记录','',
'本次以固定脚本提取全部日程条目，逐项检查 57 个条目都有章节归属，51 份核心讲义均取得有效内容，17 份作业 PDF 均可提取文本。PDF 共识别页数见随附来源索引。全部1889页版面与主图已查看，并有页范围内容说明；35份字幕文字已校读、原始音频已机器比对。未人工逐句听校或独立复现全部原实验。','',
'来源索引保存每份核心讲义和作业文件的下载 URL、字节数和 SHA-256。main 链接可变；后续制作正式教材时应固定 commit，并重新核对内容。不要把当前目录状态改写为“全教材已完成”。','']

if videos:
    success=[v for v in videos if v['status']=='extracted'];hours=sum(v['last_caption_end_seconds'] for v in success)/3600
    replacements={
        '版本：0.1':'版本：0.2',
        '| 嘉宾讲座 | 6 个日程条目 | 逐项保留；尚未取得足够一手内容，不能声称内容完整核对 |':'| 嘉宾讲座 | 6 个日程条目 | Dan Fu 已有逐讲整理、问答核对、指定画面及作者相关论文/代码；2024两场官方讲义明确不录制；2025两场与2026 Daniel Selsam完整课堂未取得 |',
        '| 录像中的口头补充和问答 | 未逐段核对 | 逐讲备课时补查；不能用讲义替代录像覆盖证明 |':f'| 官方播放列表录像字幕 | {len(success)}/{len(videos)} 个已提取，约 {hours:.1f} 小时 | 保留逐段时间戳；字幕全文已校读，原始英文音频已全程机器比对及疑难窗口复核；未人工逐句听校 |',
        '| 2026 Dan Fu | 已定位官方频道录像入口；本次读取未取得完整字幕 | 只登记来源，未用二手笔记充当已核对的一手内容 |':'| 2026 Dan Fu | 已取得官方视频的英文字幕和时间戳 | 已并入第 28 章和逐讲资料；独立本课幻灯片 PDF 仍未取得 |',
        '| 公开视频播放列表 | 入口已核对；本次未可靠枚举列表内全部条目 | 不用日程讲数推断实际公开录像数 |':'| 公开视频播放列表 | 已枚举：2025 为 17 个、2026 为 18 个 | 依据当前列表登记；2026 Dan Fu 是列表第 18 项、官网日程第 19 讲 |',
        '尚未取得独立讲义／完整字幕':'尚未取得独立讲义；字幕状态见视频清单',
        '六个嘉宾日程条目尚缺完整一手内容':'五个嘉宾日程条目尚缺足够一手内容，Dan Fu 已有课堂对照和作者相关阅读'
    }
    for old,new in replacements.items():lines=[line.replace(old,new) for line in lines]
    lines += ['## 9. 课堂录像与字幕提取清单','',
        f'当前两个官方播放列表合计 {len(videos)} 个条目。已取得 {len(success)} 个英文字幕轨道，字幕末尾时间合计约 {hours:.1f} 小时。2024 官网未列录像入口，本次没有确认到官方播放列表，不声称已取得 2024 录像。','',
        '本次取得的成功轨道都标记为 en-US、is_generated=false。这表示取得的是上传字幕轨道，不能据此保证没有转写错误。原始字幕与时间戳仅作为本地备课材料保存；本分享目录只提供来源和提取状态。','',
        '字幕文件已检查非空、段数、时间戳顺序和内容校验值。原始音频全程机器比对，疑难窗口再次转写并检查语义。L03越界幻觉尾已排除；未人工逐句听校。字幕不包含所有板书、图示、光标动作和现场代码变化。','',
        '完成课堂内容核对需要：读取对应讲义和字幕，按时间点识别额外解释、演示和实质问答；涉及公式、代码或图示的地方再查看对应画面；把验证后的新增点链接到相应章节。仅提取字幕不能标记为“视频内容已完整纳入”。','',
        '| 年份 | 列表序号 | 官方录像 | 字幕状态 | 字幕时长 | 内容对照 |','|---|---:|---|---|---|---|']
    for v in videos:
        duration=f'{v["last_caption_end_seconds"]/60:.1f} 分钟' if v['status']=='extracted' else f'{v["local_asr"]["last_end"]/60:.1f} 分钟（自动转写）' if v.get('local_asr') else '—'
        status=f'上传字幕已提取，{v["segments_count"]} 段' if v['status']=='extracted' else f'官方字幕未取得；本地自动转写已补 {v["local_asr"]["segments_count"]} 段，待听校' if v.get('local_asr') else '未取得：YouTube 请求封锁'
        lines.append(f'| {v["year"]} | {v["playlist_order"]} | [{v["title"].replace(chr(124),chr(65372))}]({v["url"]}) | {status} | {duration} | 待逐讲核对 |')
    lines += ['', '本次字幕缺口：[2026 第 8 讲 Parallelism](https://www.youtube.com/watch?v=6-cXp-aOmdg)。没有改用其他讲的字幕填补，也没有使用账号 cookies 或代理规避封锁。', '']
    if any(v.get('local_asr') for v in videos):lines += ['补缺更新：已取得第 8 讲原始英文音频，并在本地完成 1,702 段自动转写；其性质与官方上传字幕不同，技术词、公式和问答仍需听校。总计 34 份上传字幕加 1 份自动转写，约 46.4 小时。','']

if (OUT/'CS336-逐讲对照记录.json').exists():
    lines=[line.replace('版本：0.2','版本：3.1（299点陪学教案版；材料核验基线1.2）').replace('待逐讲核对','字幕全文已校读；非完整音频听校') for line in lines]
    lines += ['## 10. 配套学习资料','',
    '同目录的 CS336-AI陪学资料.md／html 已补入原创讲解、逐讲课堂条目、往年差异、基础与作业补充和 10 个 CPU 小实验。逐讲对照记录保存核对级别；课堂片段索引覆盖全部已取得的字幕段，自动候选不等于人工验收。','',
    '本版状态：公开可得材料已完成所列范围核验。35份文字轨道、35场原始音频机器比对、1889页版面/主图、初轮22帧加追加136帧、231张讲义图像、405个复合输出检查，均保留来源与纠错。完整人工听校、全帧审片、未公开课堂材料仍有边界。','']
md='\n'.join(lines)
(OUT/'CS336-AI陪学目录.md').write_text(md)

# A small dependency-free Markdown renderer for this controlled document.
def inline(s):
    s=html.escape(s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^\s)]+)\)',lambda m:'<a href="'+m.group(2)+'" target="_blank" rel="noopener noreferrer">'+m.group(1)+'</a>',s)
    return s
parts=[];in_table=False;in_list=False
def close():
    global in_table,in_list
    if in_table:parts.append('</tbody></table></div>');in_table=False
    if in_list:parts.append('</ul>');in_list=False
for line in lines:
    if line.startswith('|'):
        if re.match(r'^\|[-|: ]+\|$',line):continue
        cells=[c.strip() for c in line.strip('|').split('|')]
        if not in_table:
            close();parts.append('<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in cells)+'</tr></thead><tbody>');in_table=True
        else:parts.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in cells)+'</tr>')
    elif line.startswith('- ') or re.match(r'^\d+\. ',line):
        if in_table:close()
        if not in_list:parts.append('<ul>');in_list=True
        parts.append('<li>'+inline(re.sub(r'^\d+\. |^- ','',line))+'</li>')
    else:
        close()
        if not line:continue
        m=re.match(r'^(#{1,3}) (.*)',line)
        if m:
            level=len(m.group(1));title=m.group(2);anchor='s'+str(len(parts))
            if level==3 and re.match(r'\d\d ·',title):anchor='chapter-'+title[:2]
            parts.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
        elif line.startswith('> '):parts.append('<blockquote>'+inline(line[2:])+'</blockquote>')
        else:parts.append('<p>'+inline(line)+'</p>')
close()
nav=''.join(f'<a href="#chapter-{m[0]}">{m[0]} {html.escape(m[1])}</a>' for m in modules)
css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f3ed;color:#25332f;font:16px/1.8 system-ui,-apple-system,"PingFang SC",sans-serif}main{max-width:1120px;margin:auto;padding:48px 40px 100px}h1{font-size:36px;line-height:1.4;letter-spacing:-1px;color:#183f36}h2{font-size:25px;margin-top:60px;padding-top:15px;border-top:2px solid #c8d6cb}h3{font-size:21px;margin-top:36px;color:#205547;scroll-margin-top:24px}a{color:#216854;text-decoration:underline;text-underline-offset:3px}p{max-width:85ch}li{margin:6px 0}ul{padding-left:24px}.table-wrap{overflow:auto;margin:20px 0;background:white;border:1px solid #d8dfd7;border-radius:10px}table{border-collapse:collapse;width:100%;min-width:660px;font-size:14px;line-height:1.65}th{text-align:left;background:#e6eee7;color:#193f34}td,th{padding:12px 15px;border-bottom:1px solid #e6eae4;vertical-align:top}td:first-child{white-space:nowrap}td a{overflow-wrap:anywhere}blockquote{margin:24px 0;background:#e7eee4;border-left:4px solid #50816b;padding:18px 24px;max-width:85ch}.notice{font-size:14px;color:#5c6d64;background:#e9e6db;padding:12px 18px;border-radius:8px}.nav{display:grid;grid-template-columns:repeat(3,1fr);gap:8px 20px;font-size:14px;margin:24px 0 44px}.nav a{text-decoration:none}footer{border-top:1px solid #c8d6cb;margin-top:50px;font-size:13px;color:#617268}@media(max-width:700px){main{padding:24px 18px 60px}h1{font-size:28px}.nav{grid-template-columns:1fr 1fr}h2{font-size:22px}}@media print{body{background:white}main{padding:0}.nav{display:none}h2,h3{break-after:avoid}.table-wrap{overflow:visible}table{min-width:0;font-size:10px}a{color:inherit}}'''
doc='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CS336 · AI 陪学目录与来源</title><style>'+css+'</style></head><body><main><div class="notice">独立学习资料 · 公开可得材料核验版 · 2026-10-22 · 已配套逐讲正文、字幕全文校读补充与缺口清单</div><nav class="nav" aria-label="章节目录">'+nav+'</nav>'+''.join(parts)+'<footer>基于 Stanford CS336 官方公开材料建立索引。原创陪学设计；非 Stanford 官方出版物。</footer></main></body></html>'
(OUT/'CS336-AI陪学目录.html').write_text(doc)

for f in manifest['files']:
    f.pop('path',None)
    if '.pdf' in f['url']:
        text=(SRC/f'{f["year"]}-lecture-{f["number"]:02}.txt').read_text()
        f['pages']=len([p for p in text.split('\f') if p.strip()])
manifest['assignment_files']=assign
manifest['chapter_count']=len(modules)
manifest['knowledge_point_count']=sum(len(m[3].split('；')) for m in modules)
manifest['coverage_limits']=['讲义核对范围为结构和文本线索，不是逐句精读','未逐段复核录像和问答','六个嘉宾日程条目尚缺完整一手内容','2026 L02 录制版与当前版已完成源码差异核对；按修正后口径教学','尚未做2026春季之后全领域更新审查']
if videos:
    manifest['video_sources']=[{k:v for k,v in x.items() if k not in ('error','available_tracks')} for x in videos]
    manifest['coverage_limits']=['讲义核对范围为结构和文本线索，不是逐句精读',f'取得 {len(success)} 份上传字幕；另有一份本地自动转写；结构与指定片段有笔记，未全部逐句听校或核对画面','2026 第8讲官方字幕请求封锁，本地自动转写低置信度；2024录像来源尚未定位','五个嘉宾日程条目尚缺足够一手内容；Dan Fu 已有课堂整理与指定帧检查','2026 L02 录制版与当前版已完成源码差异核对；按修正后口径教学','尚未做2026春季之后全领域更新审查']
    for v in manifest['video_sources']:
        if v.get('local_asr'):v['local_asr']={k:x for k,x in v['local_asr'].items() if k!='path'}
(OUT/'CS336-来源索引.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print(json.dumps({'chapters':len(modules),'knowledge_points':manifest['knowledge_point_count'],'lectures':sum(len(x['lectures']) for x in manifest['years'].values()),'core_files':len(manifest['files']),'assignment_files':len(assign),'pdf_pages':sum(f.get('pages',0) for f in manifest['files']),'outputs':[p.name for p in OUT.iterdir()]},ensure_ascii=False,indent=2))
