# CS336 AI 陪学资料

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

### 2026 L01 · 课程出发点与分词

对应陪学章节：01、02。官方讲义：[2026-01](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 1: Overview, Tokenization](https://www.youtube.com/watch?v=JuoVZkPBiKk)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **模型、生成与训练**：模型根据上下文给下一 token 分配概率。生成时再用某种规则选 token，并把结果接回上下文。训练则调整参数，让真实训练文本的概率更高。分词器决定模型处理哪些单位；它与预测这些单位的神经网络是两个组件。
- **效率的含义**：同样预算下，减少显存搬运、避免训练无用数据、合理选择模型大小，都可能让最终模型更好。优化一个小模型上的热点，不能保证大模型也得到相同比例的收益。
- **BPE 的两个阶段**：训练分词器时，从字节序列统计相邻对，反复合并常见对，得到词表与合并规则。编码新文本时使用已有规则；它不会为每个新句子重新训练。常见片段可成为一个 token，罕见片段退回更小单位。

课堂展开、例子和讨论：

- [00:05:00–00:07:30](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=300s) · 课堂展开：小模型并不代表所有大模型行为。课堂用 Attention 与 MLP 所占计算比例随规模改变，说明局部优化的价值会随模型变化。这里只保留结论，不把当时示例数字作为通用比例。
- [01:11:40–01:15:50](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4300s) · 例子与代码讲解：课堂从未见过的词和 UNK 的问题转入字节级 BPE，随后逐步合并字节对。可逆性依赖完整的字节基础与正确的编码、解码实现，不能仅凭“用了 BPE”就保证无损。

二轮问答、例子与演示核对：

- [00:06:54–00:08:38](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=414s) · 讲者设问。可迁移的是机制与工程方法；规模相关的模型和数据直觉未必可迁移。
- [00:31:31–00:33:10](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1891s) · 讲者设问。模型形状、目标、优化器、初始化与学习率共同构成训练决策，不能只学结构图。

字幕全文校读补充与纠错：

- [00:08:30–00:10:35](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=510s) · Bitter Lesson 的课堂解释是重视可扩展算法，不能简化为算法不重要、只堆算力。效率和资源相乘是直觉示意，不是准确率的物理定律。
- [00:17:30–00:19:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1050s) · 讲者想打开 agent trace 时网络未通，现场没有完成这个在线展示。资料保留代码/链接入口，不声称已经看到完整 agent 演示。
- [00:21:30–00:23:10](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1290s) · 开课时讲者说不深入多模态/agents，最终日程和 L17 实际加入多模态概览。覆盖范围按最终授课材料登记，不能只凭开场承诺删掉后续内容。
- [00:24:55–00:27:20](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1495s) · 课堂对 AI 的规则旨在保留学生自己的实现学习；本资料采用用户已选的概念陪学目标。课堂学分、集群额度和当年 AI 使用政策只是背景，不要求网上学习者获得 Stanford 身份。
- [00:47:05–00:49:40](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2825s) · Scaling recipe 不只是拟合一条线，也包括超参数随预算怎样变化。讲者把可预测性与小规模最优性并列；大跑之前先预测，再看真实结果，可减少事后解释。
- [00:51:00–00:53:20](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3060s) · 课程用预先训练模型的结果缓存模拟训练 API；提交配置得到 loss 不等于现场真的启动新训练。此练习教预算下的试验设计，不要求陪学者购买算力。
- [00:54:30–00:56:40](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3270s) · 内部开发看稳定、可比较的指标；外部决策还看真实任务代表性。单一平均分可能隐藏能力差异；即使使用未上网数据，也仍需检查来源和泄漏。
- [01:07:50–01:10:10](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4070s) · 字符词表的精确 Unicode 数量与版本有关；课堂只需理解稀有字符导致词表利用不均。字节词表固定 256，但 UTF-8 多字节会使 token 序列变长。
- [01:13:10–01:17:15](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4390s) · 示例 BPE 的并列频次采用第一个候选，现实实现需固定 tie-break；预分词、特殊 token 和索引加速都还需处理。玩具算法能运行，不代表满足大作业的全部规范与性能。
- [01:17:30–01:19:17](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4650s) · 结尾提出替代 tokenizer 的抽象表示与可变计算粒度要求，属于讲者的设计判断。不能把它说成已经证明所有 byte-level 模型必须采用同一种机制。

问答处理：本次核对片段主要是讲者解释；未把课程介绍中的自问自答伪装成学生提问。

演示：教学重建 D01：对短字符串逐步合并相邻字节对，比较训练前后的 token 数。看清“训练规则”和“使用规则”的区别。

理解检查：换成一个没见过的名字，分词器为什么仍可能表示它？

缺口：板书和实际代码输出未逐帧核对；课堂历史与模型成本只保留为带年份背景。


### 2026 L02 · 张量、精度与资源核算

对应陪学章节：00、03、04、09。官方讲义：[2026-02](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 2: PyTorch (einops)](https://www.youtube.com/watch?v=kuYAsz7zspQ)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **张量先看形状**：把张量看作带形状、类型和设备的数字集合。矩阵乘法把输入维度变成输出维度；einsum 用轴名说明哪些轴相乘、哪些轴求和。形状相同不代表内存布局相同，reshape 与 transpose 也不总是复制数据。
- **精度有两个维度**：指数范围决定能表示多大或多小的值，尾数精度决定相近数能分得多细。BF16 与 FP16 都占两个字节，但范围和精度分配不同。低精度还可能使用一组数共享缩放因子，不能把每个数孤立理解。
- **算得快还是搬得慢**：FLOPs 数运算次数；FLOP/s 是速度。算术强度等于运算量除以内存搬运字节数。roofline 用带宽限制和计算峰值估计性能上界；真实程序还受调度、同步和布局影响。
- **训练显存**：参数之外，还要保存梯度、优化器状态和反向传播所需激活。梯度累积把大 batch 分成小批次；重计算则减少保存的激活。二者主要解决内存问题，不承诺降低总计算量。

课堂展开、例子和讨论：

- [00:15:30–00:17:20](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=930s) · 问答：块缩放使低位宽数值共享一个缩放范围，但同一块内数值差异过大仍然困难。课堂也区分了低比特推理与训练，不能把推理量化经验直接当作训练配方。
- [00:26:38–00:27:45](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1598s) · 问答：把二维张量铺成一维时，轴顺序决定排列。einsum／rearrange 的写法需要显式表达这个顺序。
- [00:35:12–00:37:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2112s) · 计时演示解释：GPU 调用通常异步返回。只计 Python 调用返回时间可能漏掉实际计算；计时需要合适的同步、预热和重复。

二轮问答、例子与演示核对：

- [00:11:15–00:13:06](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=675s) · 低精度与混合精度示意。参数、激活、梯度、优化器状态可使用不同精度；这段配方带课程时点，不是所有系统的硬要求。
- [00:15:14–00:16:54](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=914s) · 学生问块缩放与一比特。块内共享缩放能扩大整体范围，但不能独立解决邻近数值差距；低比特推理与直接低比特训练必须分开。讲者对一比特训练的判断属于当时所知范围。
- [00:23:30–00:24:20](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1410s) · 学生问 einops 是否更快。主要是清晰表达同一类操作，语法本身不保证加速。
- [00:26:17–00:28:03](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1577s) · 学生问二维拉平成一维的顺序。轴顺序必须在 rearrange 表达式中明确；行优先和列优先不能混用。
- [00:31:21–00:33:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1881s) · 讲者推导普通矩阵乘法 FLOPs 约 2BDK，忽略加法项中的减一。它是本课程的普通算法计数口径。
- [00:32:51–00:34:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1971s) · 学生问次立方算法。课堂聚焦实际硬件上的系统协同优化；未比较所有渐近算法。加法与乘法按各一次 FLOP 计数，不意味着所有硬件指令的延迟相同。
- [00:34:49–00:36:32](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2089s) · 计时方法。GPU 异步执行，需要正确同步和重复。字幕把 FLOP/s 说成乘时间；按单位应为 FLOPs 除以秒，资料采用除法。
- [00:38:27–00:40:16](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2307s) · 学生问 MFU 为什么不满。峰值与实测需使用同一精度和稀疏口径；利用率还受搬运与调度影响。课堂的约一半是例子。
- [00:47:49–00:49:33](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2869s) · 学生问 Transformer 的算术强度，讲者留待后文。随后用 GELU 与 ReLU 说明运算更多仍可能受内存带宽限制。
- [00:54:35–00:56:18](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3275s) · 学生问硬件为何不配置更多带宽。讲者没有给出完整芯片设计答案；资料保留为开放问题，不补造硬件设计结论。
- [00:59:38–01:01:20](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3578s) · 讲者设问反向计算成本。两层线性网络展示激活和权重两条梯度路径；不能把反向传播当作只复制一次前向。
- [01:03:14–01:04:52](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3794s) · 代码讲解。用具名轴检查梯度矩阵乘法，分别对输出轴和 batch 轴求和；对照 autograd 的结果。无需死记转置位置。
- [01:13:23–01:15:06](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4403s) · 激活重计算示意。只存一部分前向激活，反向时重算；省显存会增加计算。checkpoint API 不等同于保存训练文件。

字幕全文校读补充与纠错：

- [00:17:20–00:18:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1040s) · 讲者明确录制版在笔记本 CPU 上执行，部分 GPU 代码只讲不跑。这解释 trace 中占位峰值，不将图示代码当现场 GPU 性能验证。
- [00:29:50–00:30:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1790s) · 讲者现场发现“两周”代码实际只算一周；当前源稿已补 ×2。预算估算用数值、单位和文字三者交叉检查。
- [00:43:20–00:44:50](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2600s) · 时间约取搬运与计算两者最大值，假设能充分重叠；若无重叠、启动或同步开销大，实际时间可更长。roofline 给上界直觉，不是准确秒表。
- [00:51:15–00:53:50](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3075s) · 正方形矩阵例子用两字节元素得算术强度约 n/3；一般非方形矩阵要重新计数。大 matmul 才可能计算受限，不能据最后总结断言所有矩阵乘法都 compute-bound。
- [01:01:30–01:06:25](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3690s) · 6ND 的 2 来自乘加而非 BF16 两字节，前向两倍参数触达、反向两条梯度路径。长上下文的 Attention 额外项不在这个近似里；N/D 的命名按讲义统一定义。
- [01:09:45–01:11:15](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4185s) · 讲者现场纠正 gradient_memory 与 optimizer_state_memory 用了 parameter_memory 的单位错误；应按参数个数乘每参数字节。该口头修正与当前源稿一致。
- [01:12:30–01:13:40](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4350s) · 梯度累积主要省激活显存；口述 save compute 不作结论。正确缩放 microbatch 损失后累计一次更新，不能每个 microbatch 都更新再说完全等价。
- [01:14:50–01:16:50](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4490s) · 按层 checkpoint 与按 sqrt(L) 间隔切段不同；在简化等成本链上，后者激活峰值 O(sqrt(L))、额外重计算总量 O(L)。口述“额外计算 sqrt(L)”混了每段长度和总工作量，按当前源稿纠正。

问答处理：先让学习者解释“省一半字节”能改变什么，再区分显存容量、搬运量与数值误差。

演示：教学重建 D04：比较一次大 batch 的平均梯度，与两个小批次累积后再更新的结果。官方代码中的 GPU 计时只解释方法，不给出本机未测量的吞吐量。

理解检查：把 batch 拆成两次，为什么不能在每个小批次后都更新参数，再称它与一次大 batch 完全等价？

缺口：课堂跳过了部分训练循环；对应知识由讲稿与 A1 补足。FP4／FP8 硬件配方未作为通用建议。


### 2026 L03 · Transformer 结构与超参数

对应陪学章节：05、06、07。官方讲义：[2026-03](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 3: Architectures](https://www.youtube.com/watch?v=lVynu4bo1rY)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **注意力怎样取信息**：Q 与 K 的匹配分数经缩放和 softmax 变成权重，再加权组合 V。因果遮罩禁止当前位置使用未来 token。多个头提供多组投影；它们不等于人工指定的语法专家。
- **残差与归一化**：残差把输入直接加到变换结果，保留较直接的信息和梯度路径。pre-norm 把归一化放在子层之前。RMSNorm 不减均值，分母为 sqrt(mean(x²)+ε)，再乘可学习 gain；部分讲义公式漏了均值中的1/d，不能直接照抄。计算与搬运是否更省需看实际实现和任务。
- **位置、门控与稳定性**：RoPE 对 Q、K 做随位置变化的旋转，使点积可反映相对位置。SwiGLU 用一条分支调制另一条分支。QK norm、z-loss 与 logit soft-capping 针对不同位置的数值失稳，不能互相当成同一个技巧。
- **超参数是取舍**：深宽比、头维度、FFN 比例、词表和正则化会改变质量、成本与并行方式。论文里的常用比例是经验起点，不是数学必然。

课堂展开、例子和讨论：

- [00:15:00–00:18:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=900s) · 课堂展开：讲者用 RMSNorm 说明 FLOPs 少不等于运行快：少搬一次数据可能比少做一点算术更重要。
- [00:56:48–00:59:20](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3408s) · 问答与比较边界：讨论不同分词器的语言模型评价。per-token perplexity 受切分粒度影响；若比较原始字符串的概率，还要确保原始文本、规范化和计量单位一致。字幕中部分追问不清，未补造结论。
- [01:04:26–01:06:40](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3866s) · 问答：weight decay 除了传统的防过拟合作用，还会与学习率和优化动态相互影响。不能把预训练中所有正则化作用都解释成“数据太少”。

二轮问答、例子与演示核对：

- [00:16:35–00:18:14](https://www.youtube.com/watch?v=lVynu4bo1rY&t=995s) · 学生问 normalization 为何耗时。读写激活可能主导该算子；小模型表中的极端占比不能推广至所有模型。RMSNorm 的质量收益不是保证。
- [00:38:50–00:39:50](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2330s) · 学生问高维旋转论文。讲者没见过，不据此断言不存在；RoPE 讲的是固定二维子空间旋转的构造。
- [00:39:23–00:41:01](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2363s) · 学生问如何从报告提炼配方，以及并行 block 的精度取舍。讲者建议跨报告看规律并小规模实验；不同报告对并行 block 结果不一，没有统一受控结论。
- [00:41:13–00:42:49](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2473s) · 学生问部分 RoPE 与相对位置。只旋转部分坐标是一种取舍；字幕说 keys/values 的位置应按讲义公式理解为 Q/K，避免照抄口误。
- [00:42:39–00:44:19](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2559s) · 学生追问相对位置的因式分解约束。RoPE 是满足一组约束的方案；其他把位置偏置直接加到注意力矩阵的方案仍可能有效。
- [00:56:19–00:57:56](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3379s) · 多模态词表与跨分词器评价。多模态是否需离散词表取决于编码；原始同一字符串的概率可比较，不能直接横比不同切分的每 token 困惑度。
- [00:58:26–01:00:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3506s) · 按原始字节归一化可消除 token 数不同的计量问题，但需同一文本与预处理。后续追问未厘清，讲者提出课后再谈；资料不写成已回答。
- [01:04:12–01:05:48](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3852s) · 学生问 diffusion 结构与正则化。讲者明确未充分研究前者；weight decay 与学习率、收敛速度相互作用，不能只套防过拟合解释。
- [01:22:57–01:24:44](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4977s) · 学生问还需不需要搜索超参数及训练中改变参数。规则仍需实验；架构维度一般固定，weight decay 可随学习率变化。现役模型细节仅作当年案例。
- [01:24:42–01:26:20](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5082s) · 学生确认 MQA/GQA 也是训练结构。KV 头配置并非推理时任意更换；随后引入全局与局部 Attention 交替。

字幕全文校读补充与纠错：

- [00:12:30–00:15:00](https://www.youtube.com/watch?v=lVynu4bo1rY&t=750s) · pre/post 的命名需画残差加法位置。F(LN(x))+x、LN(F(x))+x、LN(F(x)+x) 是不同结构；“post-norm”这个词单独出现不足以判定是否规范化残差主路。
- [00:13:55–00:15:05](https://www.youtube.com/watch?v=lVynu4bo1rY&t=835s) · LayerNorm 按方差加 epsilon 的平方根归一化，RMSNorm 按均方加 epsilon 的平方根归一化。口述 divide variance 是简化；经验上质量相近不能当严格函数类包含关系。
- [00:21:40–00:25:45](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1300s) · Swish 是 x·sigmoid(x)；SwiGLU 还包含另一条线性分支的逐元素乘法。保持参数量时，三矩阵门控 FFN 的中间宽度常取原两矩阵 FFN 的 2/3；这只对相同外维度和忽略 bias 等条件成立。
- [00:32:50–00:38:55](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1970s) · 二维旋转例子 We know 与 Of course we know 使用相同相对距离。字幕将“reduce to 2D”说成 3D 属误差；对每一对坐标旋转，Q/K 点积的相对位置性质不代表上下文相关 hidden state 对位置完全不敏感。
- [00:49:40–00:54:55](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2980s) · 常用 head_count×head_dim≈model_dim 与宽深比 model_dim/layers≈100 是课程调查的经验区间；不是两者各自等于 model_dim，更不是所有模型的硬约束。
- [00:59:40–01:03:05](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3580s) · 单遍训练减少传统重复遍历的过拟合，但不能保证不记忆，语料内也可重复。训练 loss≈验证 loss 的经验不替代独立验证；weight decay 与学习率日程交互是此处重点。
- [01:06:30–01:09:50](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3990s) · z-loss 惩罚 logZ 偏离零，利用 softmax 对全体 logit 同移不变的自由度约束数值尺度。它和减最大值的稳定 softmax 是不同措施，不把 z-loss 作为防 overflow 的唯一必要条件。
- [01:10:30–01:14:30](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4230s) · QK norm 控制 Q/K 尺度，soft-capping 直接限制进入 softmax 的 logits。它们不等同，也不保证质量只升不降；有可学习增益时尤其不能简单断言所有分数绝对≤1。
- [01:16:10–01:20:30](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4570s) · KV cache 保存过去各层的 K、V，通常不保存整个 QK 注意力矩阵。口述提到 queries 和 QK submatrix 时不能照抄成缓存定义。讲义算术强度公式带整体倒数 −1，字幕只说括号内之和会把大小方向颠倒。
- [01:24:50–01:29:09](https://www.youtube.com/watch?v=lVynu4bo1rY&t=5090s) · 局部层仍可利用前面全局层汇总的信息；交替全局与 SWA 或状态空间层并不等于每层都能直接访问所有原始 token。NoPE、RoPE 和层间结构需按具体模型区分。

问答处理：不同 tokenizer 下的困惑度不可直接横比。换成按字节计的损失，也仍要控制原始文本和预处理。

演示：教学重建 D02：计算一个很小的因果 Attention 表。改变未来位置的 V，检查早先位置的输出不变。图中的颜色和屏幕参数未作为已复现证据。

理解检查：为什么 Transformer 可以在训练时并行算多个位置，生成时却通常逐 token 前进？

缺口：多模态词表及若干提问存在听不清片段；不据此给出某个现役模型的确切词表规格。


### 2026 L04 · Attention 替代路线与 MoE

对应陪学章节：07、08。官方讲义：[2026-04](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 4: Attention Alternatives](https://www.youtube.com/watch?v=cKSwj_qZ8Jg)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **改变 Attention 的成本**：滑动窗口只保留局部历史；稀疏注意力选择一部分位置；线性注意力重写计算并用紧凑状态汇总历史。这些方法对保留信息的方式不同，不能仅按复杂度排序优劣。
- **线性并不等于原算法**：去掉或改写 softmax 后，某些乘法可先算 KᵀV，再与 Q 相乘。但一般不能随意交换原 softmax Attention 的计算顺序并声称完全等价。递推状态让推理成本减少，也引入信息压缩的限制。
- **MoE 在哪里稀疏**：通常把一个 FFN 换成多个专家，由 router 为每个 token 选择少数专家。总参数决定存储等成本，激活参数更接近每 token 的计算规模。router、共享专家及加权方式仍需计算。
- **为什么负载均衡重要**：路由过度集中会让部分专家忙、部分专家闲，也可能伤害训练。辅助损失、噪声和路由偏置是不同处理办法；稀疏选择、网络通信与容量限制一起决定实际收益。

课堂展开、例子和讨论：

- [00:20:00–00:31:40](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1200s) · 课堂展开：线性、混合和稀疏方案都在解决长上下文问题，但压缩历史与保留原 token 的区别会影响任务。复杂度下降不是准确率保证。
- [00:45:33–00:47:25](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2733s) · 课堂讨论：MoE 普及受基础设施复杂、模型分布到多设备和训练不稳定等限制。课堂明确把更少 FLOPs 与更难用的系统放在一起讨论。
- [01:00:00–01:08:20](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3600s) · 推导与实验解释：top-k 路由的离散选择、专家负载和梯度目标需要一起看；去掉负载均衡损失的实验不能只看语言模型损失，还要看系统利用率。

二轮问答、例子与演示核对：

- [00:20:31–00:22:10](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1231s) · 学生问等价递推为何掉分。线性形式与它的递推可精确等价；从 softmax Attention 改到线性 Attention 这一步已经改变模型。两种等价关系不能混淆。
- [00:27:17–00:28:58](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1637s) · 学生问 indexer 是否仍是平方复杂度。轻量投影与低精度降低常数，选定 top-k 后限制后续开销；复杂度符号不足以判断实际效率。
- [00:29:02–00:30:50](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1742s) · 问答说明短上下文预训练、长上下文扩展、后训练的常见阶段顺序。关于去 softmax 的稳定性，讲者只表达个人所知，未给定理；预测未来路线的问题未形成可核实结论。
- [00:31:25–00:33:55](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1885s) · 学生问 FP4 Attention 与更复杂状态空间模型。低精度 indexer 与最终加权精度可分开；softmax 数值风险和有限状态的信息瓶颈仍需考虑，讲者未给成熟通用 FP4 Attention 配方。
- [00:33:34–00:35:11](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2014s) · 学生问有限状态与长上下文。固定小状态压缩长历史会有信息取舍；把状态扩大到与历史相当会失去原来的成本优势。
- [00:42:21–00:44:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2541s) · 学生问训练是否也稀疏、路由粒度。每 token 只激活少数专家，未选中的专家没有同样的反馈；通常 router 很简单，不能假想成懂医学的调度员。
- [00:45:15–00:46:54](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2715s) · 讲者设问 MoE 为何普及慢。基础设施、显存、跨卡分布和训练失稳共同增加复杂度；更少激活 FLOPs 不等于部署更容易。
- [00:58:00–00:59:20](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3480s) · 学生问 shared expert 怎样并行。可复制共享专家减少通信，但增加存储；它没有稀疏专家那样的选择性激活收益。
- [01:10:01–01:11:43](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4201s) · 学生问专家是否按领域分工，以及设备均衡。可出现符号和语言等路由模式，但不保证语义领域专家；过强均匀约束也会影响训练，需要兼顾每专家与每设备负载。

字幕全文校读补充与纠错：

- [00:05:25–00:09:50](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=325s) · 去 softmax 后可以结合矩阵乘法，但自回归仍需因果约束：状态只累积到当前位置。不能把 Q(KᵀV) 对整段无掩码计算，冒充因果递推。线性状态可训练时并行、推理时递推，是同一改写模型的两种执行形式。
- [00:12:10–00:17:40](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=730s) · Mamba-2 的遗忘门与 Gated DeltaNet 的写入/擦除门控制不同动作；I−βkkᵀ 只有在特定规范化与 β 条件下才是严格投影。这里讲关联记忆直觉，不把所有状态空间理论都声称已展开。
- [00:23:00–00:28:20](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1380s) · 轻量 indexer 仍可能有 O(n²) 候选打分，选 k 个历史位置后的逐 query 注意力为 O(k) 项、整段约 O(nk)（忽略头维度等）。课堂说“小段仍 quadratic”是口头概括，不能用于精确复杂度账本。
- [00:49:00–00:53:00](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2940s) · token-choice、expert-choice、全局 assignment 是不同选择约束；全局分配只优化已定义分数与容量，不证明训练质量全局最优。固定哈希路由也与每次随机路由不同。
- [00:58:50–01:06:10](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3530s) · Top-k 索引本身不提供普通连续导数；选择固定时，梯度可流过被选专家及其连续门权重。负载损失帮助避免塌缩，并没有使离散 arg-top-k 突然变为处处可导。
- [01:12:30–01:14:55](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4350s) · 把专家计算打包为 grouped/block-sparse matmul 可提高实现效率；这不意味着其任意稀疏结构都享受 GPU 规格表中的 2:4 峰值倍数。通信前低维投影以表示能力换搬运成本，要把投影成本计入。
- [01:14:55–01:16:40](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4495s) · 有容量上限和 token dropping 的 MoE 可能使同一请求受其他 batch 请求影响；dropless 实现消除这种丢弃机制，但不保证消除所有并发相关数值差异或非确定性。
- [01:17:45–01:19:50](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4665s) · MoE 小数据微调可能过拟合，可选择只调部分层/注意力或增加数据。Dense-to-MoE upcycling 是已有 dense 权重转为专家初值再训练，不是复制权重就自动得到更高质量。
- [01:24:10–01:26:05](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=5050s) · MLA 缓存低维表示，但 RoPE 的位置部分需要相容处理，不能只说存一个潜变量就等价恢复全部 K/V。MTP 是额外训练目标，也可帮助构造草稿，不保证未经验证便能并行接受全部未来 token。

问答处理：本讲部分问题由讲者提出，未一律标为学生问答。

演示：用四个 token、三个专家画出 top-1 路由表。比较全部 token 去同一专家和均匀分配时的等待；这是教学重建，不是课堂跑分。

理解检查：一个总参数更大的 MoE，为什么既可能省计算，又可能更难放进显存？

缺口：索引器图、路由图与消融曲线未逐帧核对。现役模型案例保留为课程时点案例。


### 2026 L05 · GPU、TPU 与 FlashAttention 的硬件直觉

对应陪学章节：09、10、12。官方讲义：[2026-05](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 5: GPUs, TPUs](https://www.youtube.com/watch?v=izZba4UA7iY)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **处理器为何不同**：CPU 强调较复杂控制与低延迟；GPU 用大量并行工作提升吞吐。GPU 的线程组成 warp 和 block，block 调度到 SM。寄存器、片上共享内存、缓存和 HBM 的速度与容量不同。
- **数据移动的代价**：把 GPU 想成工厂时，HBM 像仓库，片上内存像工作台。反复把中间结果运回仓库会很贵。融合、分块和重计算都在改变搬运量，但实际容量、同步和依赖仍然存在。
- **分块与尾部浪费**：tiling 让数据装入快内存后多次使用。块大小也影响可并行的块数；分不均时，最后一批工作可能只占用部分 SM，形成阶梯状性能。不是矩阵元素多一点就一定只慢一点。
- **FlashAttention**：普通实现可能把很大的注意力中间矩阵写入 HBM。FlashAttention 分块，并用稳定的在线 softmax 合并统计量，计算精确 Attention 而不保存整个中间矩阵。反向可重新计算部分量。

课堂展开、例子和讨论：

- [00:27:30–00:33:20](https://www.youtube.com/watch?v=izZba4UA7iY&t=1650s) · 图与例子解释：课堂把 roofline、内存距离和分支执行联系起来：分支两侧都存在工作时，同一 warp 不一定能真正并行执行两条路径。
- [00:47:30–00:52:30](https://www.youtube.com/watch?v=izZba4UA7iY&t=2850s) · 工厂类比与计算图：算子融合减少中间结果的反复读写；激活重计算增加一些算术，却可能降低内存需求。不要由此推断重计算在所有设备上都更快。
- [01:05:00–01:10:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=3900s) · 曲线解释：矩阵尺寸与 tile、SM 数量不对齐，会出现性能跳变。重建只说明机制，不复制视频中的具体带宽数字。

二轮问答、例子与演示核对：

- [00:12:38–00:14:17](https://www.youtube.com/watch?v=izZba4UA7iY&t=758s) · 学生问 shared memory 与 L1。缓存由硬件管理；shared memory 是程序可显式组织的工作区。物理实现和共享容量仍依具体设备。
- [00:15:05–00:16:05](https://www.youtube.com/watch?v=izZba4UA7iY&t=905s) · 学生纠正 SIMT 作用范围。共同执行指令的基本组是 warp，不能把整个 block 说成永远同一指令。
- [00:19:57–00:21:35](https://www.youtube.com/watch?v=izZba4UA7iY&t=1197s) · 学生问矩阵单元大小，讲者不知道。TPU TensorCore 与 NVIDIA Tensor Core 命名层级不同；这段 A100/132 SM 等型号数字混杂，未作为硬件规格采用。
- [00:27:25–00:28:00](https://www.youtube.com/watch?v=izZba4UA7iY&t=1645s) · 学生问 V100 前的矩阵单元。保留为 NVIDIA Tensor Core 的硬件历史，不能说没有专用单元的 GPU 就不能高效矩阵乘法。
- [00:27:58–00:29:31](https://www.youtube.com/watch?v=izZba4UA7iY&t=1678s) · 学生追问内存层级和硬件布局。物理距离、互连和资源分配影响带宽；prefill/decode 分离体现负载不同，不代表任何两种设备组合都更优。
- [00:29:45–00:30:50](https://www.youtube.com/watch?v=izZba4UA7iY&t=1785s) · 学生问每 block 有多少 warp。讲者没答出。教学按所启动的线程数及该设备 warp size 算，并检查设备上限，不把数量写死。
- [00:42:40–00:44:35](https://www.youtube.com/watch?v=izZba4UA7iY&t=2560s) · 学生问首尾层量化难点。讲者明确首层不知道；末层直连输出损失的直觉不是完整理论。MX/NV 格式的块大小与缩放类型需查对应规格，不按混杂口述抄写。
- [00:44:35–00:46:12](https://www.youtube.com/watch?v=izZba4UA7iY&t=2675s) · 学生问量化是否只支持 matmul 与 scale 是否训练。量化可降低搬运与乘法成本，但量化/反量化有开销；scale 可按块统计或历史值选取，具体是否可学习看方法，不能把课堂简化回答推广到所有量化方法。
- [00:46:14–00:47:48](https://www.youtube.com/watch?v=izZba4UA7iY&t=2774s) · 学生问结构稀疏与训练后量化。课堂描述表示能力、硬件收益和开发成本的经验取舍；QAT、PTQ 与模型规模可组合，仍需实测。
- [00:54:50–00:55:45](https://www.youtube.com/watch?v=izZba4UA7iY&t=3290s) · 学生问 DRAM 读图的 T 形布局。讲者对图行列方向有所犹豫，只保留局部连续读取与突发传输的机制，不把这张图当芯片布局实测。
- [01:01:12–01:02:49](https://www.youtube.com/watch?v=izZba4UA7iY&t=3672s) · 学生把 tiling 类比到跨 GPU，并问 PyTorch 是否自动做。常见 matmul 已在底层分块；自定义算子仍需组织局部复用。张量规模不整除 tile 会产生尾部浪费。
- [01:05:40–01:06:40](https://www.youtube.com/watch?v=izZba4UA7iY&t=3940s) · 学生问固定 tile 大小。普通模型使用者先留意矩阵维度与底层块的对齐；具体最佳 tile 要针对算子、精度和设备调优，不能统一取 32。
- [01:10:00–01:12:20](https://www.youtube.com/watch?v=izZba4UA7iY&t=4200s) · 学生问全 SRAM、64 对齐、108 SM 与晶圆级设备。成本、能耗与局部性限制全 SRAM；对齐是整除约束，不是数字越大必快。SM 数由具体芯片决定；晶圆级编译讲者未深入研究。

字幕全文校读补充与纠错：

- [00:09:50–00:12:10](https://www.youtube.com/watch?v=izZba4UA7iY&t=590s) · 口述中的 A100 SM 数混杂了不同型号，硬件数值不作背诵事实；性能解释只需实际设备的 SM 数、块数和资源限制。具体规格从设备查询或厂商文档取得。
- [00:16:00–00:17:20](https://www.youtube.com/watch?v=izZba4UA7iY&t=960s) · CUDA local memory 的 local 指线程私有作用域，物理存储可在全局设备内存，不能按口述理解成必在 SM 上的快内存。寄存器溢出也可能落到这里。依据 NVIDIA CUDA Guide 的 Local Memory 小节：https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/02-basics/writing-cuda-kernels.html 。
- [00:33:00–00:34:50](https://www.youtube.com/watch?v=izZba4UA7iY&t=1980s) · 同一 warp 中发生路径分歧可能分别执行两侧并屏蔽无关 lane；若所有 lane 走同一路径，不必执行另一侧。编译器还可能谓词化，不能把 GPU 上任何 if 都视为同样慢。
- [00:38:40–00:42:20](https://www.youtube.com/watch?v=izZba4UA7iY&t=2320s) · 按块缩放量化时，转置后分组方向改变，简单换 stride 不一定保持所需量化布局；可能需要分别量化两种方向并保留副本。速度与存储账本必须算上 scale 和副本，不能只算每元素 8 bits。
- [00:50:00–00:52:35](https://www.youtube.com/watch?v=izZba4UA7iY&t=3000s) · 连续 sigmoid 的重计算例子比较的是简化图的 HBM 读写次数，并未证明反向 FLOPs 不变或所有重算更快。多做计算换少搬数据是条件性收益。
- [00:52:50–00:57:30](https://www.youtube.com/watch?v=izZba4UA7iY&t=3170s) · Coalescing 关注同一 warp 各 lane 的地址能否合并为少量内存事务；不是所有 thread 必须读同一元素。事务大小、对齐、元素类型依设备而变，128 字节只是图中示意。
- [01:02:40–01:05:10](https://www.youtube.com/watch?v=izZba4UA7iY&t=3760s) · tile 选择同时受共享内存、并发块数、矩阵形状和内存对齐影响。autotune 会花编译/测试时间，不能把课堂的十五分钟当所有程序的固定开销。
- [01:07:30–01:09:25](https://www.youtube.com/watch?v=izZba4UA7iY&t=4050s) · wave quantization 示例从 98 个块跨过 108 个 SM 到 120 个块，多出一次不足额执行波；这是设定每块资源和 SM 驻留条件的简化例，不是每次加一维都必慢一倍。
- [01:12:30–01:17:20](https://www.youtube.com/watch?v=izZba4UA7iY&t=4350s) · Attention 核心为 QKᵀ 和归一化权重乘 V 两次矩阵乘，另有生成 Q/K/V 的投影。FlashAttention 省 HBM 中间矩阵，仍计算精确注意力；在线累积需同时重缩放分母与加权值分子，D10 已验证。

问答处理：音频补核已补GPU图层次、Tensor Core、重复加载与预取问答；学生缺词不逐字恢复，见第十节。

演示：画一个 SM 数有限、任务块数不能整除的调度表。让学习者指出最后一轮空闲位置；FlashAttention 的在线归一化用 D03 的数值稳定思路连接。

理解检查：为什么“多做一点运算”有时能让模型更容易训练，甚至运行更快？

缺口：TPU 结构图、低精度布局和 FlashAttention 图中数据流需要画面复核；未运行 GPU kernel。


### 2026 L06 · 计时、融合与 Triton kernel

对应陪学章节：10、11、12。官方讲义：[2026-06](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **先定位再优化**：端到端计时告诉你总体速度，profiler 帮助定位耗时。预热排除部分初始化与编译开销；同步避免把 GPU 未完成的工作漏计。profiler 本身也有开销，不能把它的所有耗时都当作正常执行耗时。
- **一个 kernel 的骨架**：确定程序负责的索引，生成地址，读入数据，执行运算，再写回。mask 处理超出有效长度的位置；stride 决定如何从逻辑索引走到内存地址。
- **高层代码与实际执行**：Triton 代码描述分块计算，编译器生成更低层代码，并安排线程、寄存器等资源。Python 中出现一个 load，不代表 GPU 正在逐行调用 Python 库。
- **例子递进**：GELU 适合看融合；softmax 包含按行归约与数值稳定；行求和引出分块；矩阵乘法进一步引出数据重用、共享内存和累加器。

课堂展开、例子和讨论：

- [00:20:44–00:22:35](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1244s) · 问答：同一 SM 可以容纳什么工作受资源限制。尾部浪费不能简单靠把一个 block 任意拆给其他 SM 消除，改变 block 大小也会改变其他成本。
- [00:47:54–00:50:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2874s) · 问答：讲者解释 tl.load 与 HBM 地址、编译后执行之间的关系。把编程模型和硬件动作分开，是这段问答的重点。
- [00:53:27–00:56:25](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3207s) · 问答：同一段已编译代码可由多个线程运行，线程／块 ID 使其处理不同数据。一般不必手写 PTX；这不是禁止深入低层，而是说明通常的开发层次。
- [01:04:52–01:06:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3892s) · 问答：按列做 softmax 需要改变地址与 stride 的计算。能表达同一数学操作，不保证内存访问性能也相同。

二轮问答、例子与演示核对：

- [00:20:10–00:21:55](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1210s) · 学生问多个 block 能否共用 SM。可以存在多 block 驻留，但可用寄存器、共享内存和执行资源会限制收益；尾部不均衡不能靠任意拆散 block 消除。
- [00:35:18–00:37:04](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2118s) · 学生把 compiled Triton 误读成更快。讲者明确本次结果 compiled 比 built-in 慢；性能顺序随设备与实现改变。融合机制不保证战胜库函数。
- [00:45:55–00:47:43](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2755s) · 学生问 CUDA/Triton 与 tensor unit。逐元素 CUDA 也直观；Triton 以块表达更复杂操作。具体硬件指令需看编译结果，而不是从 Python 语法推断。
- [00:47:31–00:49:16](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2851s) · 学生问 tl.load 怎样执行。它描述内存读取，经编译生成 PTX 等低层程序，不是在 GPU 上调用 Python Triton 库；局部变量的落点需看编译和资源需求。
- [00:49:00–00:50:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2940s) · 学生追问读取是否阻塞，讲者先转到 PTX；问题不能在此就标成已经回答，需接后续 warp 调度解释。
- [00:53:02–00:54:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3182s) · 学生问是否逐线程分别编译。通常同一程序由不同线程执行，用 block/thread ID 区分工作；不是为每个线程编一份不同源码。
- [00:55:00–00:56:36](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3300s) · 学生问 PTX 是否可手写及读取等待。一般由编译器生成，也有人手写；等待读取时 SM 可调度其他可运行 warp 隐藏延迟。
- [00:56:36–00:58:16](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3396s) · 学生问为什么恰有四个 warp scheduler，讲者不知道。随后区分整行可放入一块的 softmax 与需多 tile 的归约。
- [01:03:33–01:05:13](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3813s) · 学生问按列 softmax。改变 stride 可表达列访问，但访存性能可能改变；不是只看数学等价。
- [01:09:54–01:11:37](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4194s) · 学生问 accumulator 在何处。源码没有显式指定；课堂的共享内存说法属于简化，实际寄存器、共享内存与 spill 要检查编译产物，不能教成大数组自动进 shared memory。
- [01:11:24–01:13:09](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4284s) · 多 tile 归约与逐元素多 block 不同。一个程序块可循环处理同一行的多个 tile；随后 matmul+ReLU 引入复用与融合。
- [01:24:18–01:25:57](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5058s) · 学生问 Triton 的替代工具与高维张量。工具抽象与可控程度不同，需按算子需求选择；高维问题的具体答案不在这段完整出现，不补造选择建议。
- [01:26:00–01:27:15](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5160s) · 学生问高维张量应该整块读还是逐元素读。讲者明确需看计算，留待课后；资料不编造唯一方案。

字幕全文校读补充与纠错：

- [00:11:00–00:14:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=660s) · 寄存器示例按 128×160=20480 个/块，在 65536 个/SM 的简化约束下最多三块、12 个 warp，12/64=18.75%。真实驻留还受共享内存、分配粒度和其他上限影响；occupancy 更高不必然更快。
- [00:14:15–00:16:40](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=855s) · 共享内存 bank conflict 与全局内存 coalescing 是不同约束；同地址读取可广播，不可用“每 bank 永远只能一个线程”覆盖所有情况。swizzling 改变布局是减少冲突的一种方法。
- [00:24:10–00:26:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1450s) · CUDA event 用于设备时间，主机计时配同步也有用途；预热、重复和分位数取决于要测稳定吞吐还是首次延迟。不能把编译开销对任何用户场景都排除。
- [00:30:30–00:35:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1830s) · 朴素、内置、compiled GELU 应先用相同公式/近似与容差比较。多个 kernel 之间可能有缓存，HBM 搬运示意不是对每次真实 DRAM 事务的逐条实测。
- [00:40:20–00:45:40](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2420s) · Triton 的 BLOCK_SIZE 表示逻辑处理元素数，不必等于 CUDA 线程数；编译器可能让每线程处理多元素。program_id 对应逻辑 program，不能把逻辑块的每个元素当独立线程。
- [00:50:50–00:54:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3050s) · PTX 是虚拟指令层，还要经后续编译为具体 GPU 指令；讲者称“实际执行代码”用于解释底层，不表示芯片直接执行 PTX 文本。寄存器/共享内存具体落点应看编译和 profiler。
- [01:01:40–01:04:45](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3700s) · softmax 的无效 lane 填 −∞，使 exp 项为 0；整行都无效仍需单独处理。按列改 stride 在数学上可行，但访存、布局和资源占用也可能不同。
- [01:05:40–01:11:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3940s) · 讲者从长行 softmax 切换为 row sum，只示范多 tile 归约，不代表已经给出长行 softmax 的完整稳定算法。sum 的无效元素为 0，与 softmax 的 −∞ 不同。
- [01:13:30–01:22:00](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4410s) · matmul+ReLU 先沿 K 轴累积完全部乘积，再做 ReLU，才能保持原语义。不能在每个部分积后先 ReLU 再相加；较大 tile 的复用收益还需权衡并发和资源。

问答处理：保留“代码写了什么”和“硬件怎样执行”两层回答；不要用 CPU 逐行解释代替 GPU 执行模型。

演示：重建 GELU 多算子与融合版的计算图，数中间张量读写。课堂 kernel 的屏幕输出只作定位，未在本机运行性能比较。

理解检查：改变 stride 后结果正确，是否说明访存效率也保持不变？

缺口：视频标题含 XLA，但本次字幕未确认有独立 XLA 教学段；不据标题补造内容。已检查指定 profiler/PTX 帧及官方 trace；未覆盖所有界面切换与内核。


### 2026 L07 · 通信原语与分布式训练代码

对应陪学章节：13、14。官方讲义：[2026-07](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 7: Parallelism](https://www.youtube.com/watch?v=SzpOcwdIL0Y)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **rank 是谁**：rank 通常标识参与通信的进程；课堂例子采用一进程一 GPU。不要推广成“rank 在任何系统中都等于 GPU”。通信操作还需要共同的进程组、形状与调用顺序。
- **三种基本结果**：all-reduce 让每个参与者取得归约后的完整结果；reduce-scatter 让每个参与者只得一片归约结果；all-gather 再把这些片拼起来。操作组合可以等价，但具体通信算法和开销仍可不同。
- **网络与操作分开**：RDMA 描述远程内存访问能力，NVLink、交换机、InfiniBand 等涉及实现与互连。节点内外带宽、延迟和拓扑可能不同；多一张跨节点卡不一定更划算。
- **训练怎样切**：数据并行切 batch 并同步梯度；张量并行切层内矩阵；流水线并行切层并传递激活。前向通信往往还需要对应的反向通信，单靠普通 backward 不会自动创建分布式设计。

课堂展开、例子和讨论：

- [00:04:27–00:05:35](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=267s) · 演示边界：讲者明确说明：逐行 trace 采用特殊单进程展示模式，真正多进程运行的标准输出另有入口。阅读动画不能误认为看到了实际并发调度。
- [00:32:15–00:33:30](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1935s) · 问答：区分 RDMA 的操作含义与网络硬件名称。
- [00:34:44–00:36:00](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2084s) · 问答：九张 GPU 是否合理取决于第九张卡位于哪里、怎样互连，不能只按卡数推算性能。
- [01:00:52–01:02:30](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3652s) · 问答：数据并行的 local batch 需要与 world size 协调；非整除情况还要处理样本权重与填充。
- [01:08:08–01:09:40](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4088s) · 问答：all-gather 的反向与 reduce-scatter 有联系，自动微分是否处理通信取决于所用分布式算子和封装。

二轮问答、例子与演示核对：

- [00:04:02–00:05:22](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=242s) · 讲者说明可执行讲义的单步界面使用特殊单进程模式；真实多进程标准输出另有入口。不能把单步界面当成多卡现场执行。
- [00:11:37–00:13:16](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=697s) · 学生问通信 broadcast 与 NumPy broadcasting。概念上都是一对多，前者在设备之间传数据，后者描述形状与逐元素计算规则。
- [00:19:06–00:20:48](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1146s) · 学生问 gather 的目标是否固定。目标 rank 是调用参数，不必永远是 0；所有参与方需遵守同一调用协议。
- [00:20:32–00:22:22](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1232s) · 学生问 collective 只是概念还是代码。随后 torch.distributed 提供对应调用；先理解输入输出，再讨论实现拓扑。
- [00:29:02–00:30:43](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1742s) · NCCL 把通信原语映射到硬件拓扑与低层传输。模型采用何种网络的口述猜测不作为已核实事实。
- [00:31:10–00:32:15](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1870s) · 学生问 rack/tray。课堂型号和每托盘 GPU 数口述未核实，概念解释与具体硬件配置分开；机柜、计算托盘和互连域不是同一层。
- [00:31:45–00:33:33](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1905s) · 学生问 RDMA 与 NVLink/InfiniBand 的区别。前者强调远端直接内存访问能力，后者是互连技术；需避免把能力、协议和物理线路混为同一层。
- [00:33:15–00:35:03](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1995s) · 学生问 NCCL 是否为跨节点优化、九张卡如何分工。讲者未给 NCCL 内部保证；九卡能否有效取决于第九卡位于哪种网络域。
- [00:35:16–00:37:01](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2116s) · 学生问 TPU 对应关系，讲者未掌握全部细节。torch.distributed 可用 NCCL 或 CPU 的 Gloo；后续代码不强制必须有 GPU。
- [00:39:54–00:41:19](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2394s) · 演示对四个 rank 的不同张量做求和 all-reduce。打印顺序不固定；同步调用完成后各 rank 应有相同和。数值可用 D05 独立检查。
- [00:41:40–00:43:25](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2500s) · 学生问 rank 是否就是 GPU。本课使用一进程一 GPU 的常见映射；一般 rank 是进程组中的编号，不是 GPU 的固有名称。随后展示 reduce-scatter 的独立输入输出。
- [00:43:10–00:44:58](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2590s) · 学生问异步 all-reduce。返回后可做独立工作，再使用工作句柄等待和正确流同步；不能在结果未就绪时读取。
- [00:52:54–00:54:49](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3174s) · 学生问 barrier 与 CUDA synchronize 顺序。二者分别处理进程参与和设备队列，不可互相替代；讲者未给普遍顺序定论，资料不编造。
- [01:00:16–01:02:08](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3616s) · 学生问 batch 与 world size、Transformer 能否直接 DDP。等大非空分片最容易说明；梯度平均才是此处核心，字幕的参数平均应按源码纠正。实际不等分时需正确加权，补零不能改变损失统计。
- [01:07:38–01:09:21](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4058s) · 学生问张量并行的反向是否自动。前向 gather 与反向 reduce-scatter 存在对应关系；手写 primitive 不等于自动获得正确分布式反向，应选支持 autograd 的实现或写相应规则。
- [01:10:23–01:12:09](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4223s) · 流水线代码按 microbatch 接收、计算、发送局部层输出。单步接收发送的代码不自动证明获得了好的重叠调度，需进一步安排流水线。
- [01:17:58–01:19:44](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4678s) · 讲者比较显式 collective 与编译器根据分片生成通信。编译器简化操作，但仍需理解分片、拓扑和 bubble 的成本。

字幕全文校读补充与纠错：

- [00:10:00–00:11:15](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=600s) · 字幕把 scatter 的逆操作仍写为 scatter，按随后定义纠正为 gather。scatter 分片发送，gather 在指定 rank 拼接；reduce 是逐元素归约而非普通拼接。
- [00:15:00–00:17:30](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=900s) · all-reduce 可由 reduce-scatter 加 all-gather 实现，表示结果和常见通信账本等价，不保证所有尺寸、拓扑、实现的耗时严格相等。
- [00:22:30–00:29:35](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1350s) · NVLink domain 不总等于单机八卡；本讲也提 NVL72 和 RoCE。八卡以内做 TP 是课堂配置的经验，不是拓扑无关硬限制；双向链路规格与单向 HBM 带宽不能直接当同口径精确比值。
- [00:37:30–00:38:50](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2250s) · 讲者为可步进展示绕开真实 multiprocessing，笔记本用 Gloo；网页单进程轨迹与另外的多进程 stdout 要分开。master 地址端口负责初始化协调，不是所有 NCCL GPU 数据流必经 CPU 的证据。
- [00:46:40–00:53:10](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2800s) · effective bus bandwidth 常按 S×2(p−1)/(p×t) 计 all-reduce，reduce-scatter 少一个 2。这个归一化定义不能证明真实带宽与 GPU 数或拓扑无关。平均各 rank 时间也不能替代训练关键路径慢 rank 的时间。
- [00:57:30–01:02:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3450s) · DDP 参数持续一致需要相同初始化、更新规则和正确平均梯度；每卡等样本数时均值梯度直接平均成立。不同有效 token 数、丢弃样本或不同 loss 归一化时应加权。
- [01:05:00–01:15:15](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3900s) · 本讲 column TP 每层 all-gather 激活，PP 分 microbatch 并点对点传递。示例省略完整反向及通信计算重叠，不应当生产训练框架；DP 的梯度 bucket 也可在反向途中发送。
- [01:17:30–01:20:20](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4650s) · 临界 batch 是优化效率边界，不能靠无限扩大 DP 永久换取等比例收益。复制、切片、重算和跨设备存储是在计算、显存、通信之间交换资源。

问答处理：课堂有“平均参数”的口头表述；教学资料以官方训练代码为准，写成平均梯度，再由一致的优化器状态更新各副本。

演示：教学重建 D05：四个参与者输入 [0,1,2,3]+rank；all-reduce 得 [6,10,14,18]。再展示 reduce-scatter 加 all-gather 的同一结果。

理解检查：如果两个进程以不同顺序调用通信操作，为什么可能无法继续？

缺口：没有实际启动多 GPU 进程，也没有把单进程教学 trace 当作通信性能证据。


### 2026 L08 · 并行策略、分片与网络拓扑

对应陪学章节：13、14。官方讲义：[2026-08](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 8: Parallelism](https://www.youtube.com/watch?v=6-cXp-aOmdg)。字幕状态：官方字幕未取得；已有双模型完整转写、讲义与疑难窗口交叉核对，不能恢复的原问保留缺口。

讲义主线：

- **分片在分什么**：DDP 复制训练状态；ZeRO 分阶段切优化器状态、梯度与参数。FSDP 通常需要在使用参数前收集参数片段，再在适当时机释放或重新分片。省显存不等于不付通信成本。
- **流水线的气泡**：把层分给设备后，前后设备必须等待依赖。microbatch 让多个批次处在不同阶段，改善利用率，但还受激活内存、调度和批量限制。
- **组合并行**：张量、流水线、数据、序列、上下文和专家并行各切不同轴。选择方式需同时看单卡内存、拓扑、通信量和训练 batch。

课堂展开、例子和讨论：

- [00:18:22–00:20:05](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1102s) · 自动转写＋讲义对照：课堂在所用通信记账模型中比较 DDP 与 ZeRO-1：reduce-scatter 分发梯度，all-gather 收集更新参数，从而节省复制的优化器状态。近似相同的通信字节量不等于真实运行完全没有额外调度成本。
- [00:43:20–00:44:50](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2600s) · 自动转写＋讲义对照：课堂把张量并行与流水线并行的选择联系到互连拓扑。TPU mesh 与 GPU 节点内外带宽是该讲案例；不能将某代硬件的拓扑概括成所有后续设备。
- [01:03:27–01:04:50](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3807s) · 自动转写中的问答，概念经讲义核对：后段讨论 MoE 能否使用流水线、FSDP 等策略。资料只保留“可以组合，但专家并行有额外分组与通信约束”这一讲义支持的概念，不逐字恢复含糊的提问。

二轮问答、例子与演示核对：

- [00:03:44–00:05:22](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=224s) · 本地 ASR。讲者解释强调 reduce-scatter 加 all-gather 是为后续分片算法服务，并非唯一分解；随后介绍 TPU 的环面网格。学生原话转写不清，不复述具体措辞。
- [00:26:00–00:29:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1560s) · 本地 ASR，结合讲义。FSDP 每 rank 仍计算完整模型，只分时收集当前层权重；不是每设备只计算一段层。总参数通信量已包含各层之和，不能重复乘层数。更换显存容量仍需计入激活和缓冲。
- [00:36:00–00:37:15](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2160s) · 本地 ASR。学生问 PP 通信为何较少。主要传激活，示意量为 batch×sequence×hidden；比较时保持每 microbatch/每层单位一致。调度可减 bubble，但不是自动消失。
- [01:03:07–01:04:49](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3787s) · 本地 ASR。学生问 MoE 是否也可用流水线/FSDP。可组合多种并行策略；专家并行也受网络拓扑影响。推荐域大小属于经验，不作通用定律。
- [01:08:20–01:09:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4100s) · 本地 ASR。学生问循环 Transformer 对并行的影响。讲者只是探讨：重复用权重可能改变 FSDP 的聚合/释放取舍，未给已验证通用策略。

字幕全文校读补充与纠错：

- [00:00:00–00:11:15](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=0s) · 本地 ASR 的 PPU、paralyze、map mole、cancer parallel 等是识别错误；术语按官方讲义统一为 TPU、parallelize、matmul、tensor parallel。新硬件名称、功耗倍率与临时发布消息不凭这个转写确认，保留讲义来源与时点。
- [00:12:30–00:15:10](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=750s) · 16 bytes/parameter 是一组混合精度训练状态的简化账本，依参数、梯度、master weights、Adam 两矩精度而变。DDP 在固定每卡 batch 的强弱扩展设定下解释内存；固定全局 batch 时每卡激活可因本地 batch 变小而减少。
- [00:17:30–00:25:40](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1050s) · ZeRO-1/2 的“免费”指理想通信量与 DDP 同阶/相同账本，不是无同步和调度成本。ZeRO-3/FSDP 按需聚合参数，前后向各 gather、反向 reduce-scatter；预取并重叠可隐藏部分代价，不保证任意网络下零开销。
- [00:29:00–00:30:40](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1740s) · ASR 将 ZeRO-3 误作 stage two；参数切片对应第三阶段。ZeRO-1/2 仍有完整参数副本，但不能说它们完全不节省任何显存，它们分别节省优化器状态及梯度。
- [00:32:30–00:35:20](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1950s) · 流水线 bubble 随 microbatch 数增加而占比下降；microbatch 数和每个 microbatch 的样本数需分开。简化 GPipe 占比可写 (p−1)/(m+p−1)，不是任意调度都恰等于 p/m。
- [00:37:20–00:38:40](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2240s) · zero-bubble 将反向的输入梯度 B 和权重梯度 W 分离。B 优先解锁上游，W 可在空隙计算，但仍须在对应 optimizer 更新前完成；“随时”不能跨越正确更新边界。
- [00:40:00–00:43:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2400s) · 相邻 column/row 分片可避免朴素每层全激活聚合；前向 identity 与 all-reduce 的反向角色对换。高频 TP 通信与 MoE token all-to-all 不是相同 collective，不能因“所有 GPU 参与”混用名称。
- [00:45:00–00:52:20](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2700s) · 34SBH、10SBH、5AS/H 等激活公式依指定 Transformer、精度和存储策略；FlashAttention 去掉二次矩阵存储不去掉二次计算。34SBH/T 是该简化方案的账本，不是所有实现的数学下界；MLP 仍可再重算但增加成本。
- [00:49:10–01:00:50](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2950s) · sequence parallel 在此指配合 TP 切 LayerNorm/dropout 等逐 token 部分；context parallel 切长上下文注意力，两者不可仅按名字合并。EP 主要切 FFN，attention 和 MoE 可用不同 TP/EP 分组，这就是 parallel folding 的动机。
- [01:04:50–01:20:05](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3890s) · 组合策略先满足显存，再尽量 DP，快互联用于 TP/EP，慢互联多用 PP，并受有效 batch 限制。八卡是示例硬件域；本讲同时给跨节点 EP 的例外。NVIDIA 推荐配置不自动等于原模型作者真实训练配置；故障次数和具体表格数字只在一手报告确认后教学引用。

问答处理：本地 base.en 自动转写有明显术语误识别。开头一次提问不清，未据此编写问句或答案。上述条目只保留可由讲义交叉确认的概念。

演示：由官方 PDF 重建“谁保存参数、梯度、优化器状态”的分片表；通信量须声明按设备、按步骤还是全系统统计。

理解检查：ZeRO-3 比复制所有参数更省显存，为什么可能需要额外 all-gather？

缺口：官方字幕仍返回请求封锁；已用原始英文音频本地转写 1,702 段作为低置信度导航。不是官方字幕，术语、公式和完整问答仍需听校；具体画面核对范围见演示帧清单。


### 2026 L09 · Scaling laws：规律、拟合与外推

对应陪学章节：16。官方讲义：[2026-09](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 9: Scaling Laws](https://www.youtube.com/watch?v=Q15rhEWZPQ4)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **曲线描述什么**：幂律形式可以用 L(x)=L∞+a·x^(−α) 表示。x 是数据、参数或计算等某个量，L∞ 是在该模型假设下的下限。双对数直线只在相应形式和区间内成立，不能从一小段拟合推出永远成立。
- **计算预算的取舍**：更大的模型每个 token 更贵；更多数据需要更多训练步。固定预算下，要在模型规模与数据量之间分配。Chinchilla 的多种拟合方法依赖训练配方、学习率安排和计算记账方式。
- **三种方法**：可以从多次训练的最优包络读预算表现；沿相同 FLOPs 扫模型大小；也可以联合拟合模型量和数据量。比较结论前，先检查每种实验是否得到足够训练、是否公平调参。
- **边界**：重复数据、分布变化、选择质量和下游指标都会改变观察到的规律。训练损失最优，也未必是考虑服务成本后的最优部署选择。

课堂展开、例子和讨论：

- [00:10:30–00:11:50](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=630s) · 问答：讲者明确把这些规律描述为经验拟合。理论能提供候选函数形式，却没有唯一的“黄金公式”。
- [00:15:00–00:18:00](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=900s) · 例子：均值估计帮助理解误差随样本量下降的幂次。它解释一种简单来源，不等于证明语言模型的指数。
- [01:07:30–01:14:20](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4050s) · 方法比较：参数计数、嵌入参数、学习率 schedule 与拟合误差可能改变论文之间的结论；不能把某个固定 token／参数比当作永恒常数。

二轮问答、例子与演示核对：

- [00:10:03–00:11:47](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=603s) · 学生问 scaling law 的理论依据。课堂经验式来自拟合；理论可给候选函数形式，不能推出幂律对任意模型和规模都成立。
- [00:20:30–00:22:05](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1230s) · 学生问模型大于数据是什么意思。此处指在所研究的幂律区间避免过早进入模型容量限制，不是参数数必须严格大于 token 数的定义。
- [00:31:44–00:33:18](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1904s) · 讲者比较 LSTM 与 Transformer 的计算—损失曲线，说明截距和斜率都重要；小规模排序未必保持到大规模，需有外推证据。
- [00:53:35–00:55:13](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3215s) · 学生问重复运行和直接拟合下游指标。大验证集的预训练损失常低方差，但学习率或临界 batch 的噪声可能更大；单次运行是观察到的做法，不是建议免除误差分析。
- [00:55:28–00:57:09](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3328s) · 学生问训练损失和测试损失。单遍大数据时二者有时接近；科学比较仍需独立评估，数据重复、污染与分布差异会破坏该近似。

字幕全文校读补充与纠错：

- [00:15:00–00:20:20](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=900s) · 均值估计的 sigma²/n 对应均方误差，不是任意误差都此速率。非参数例子的指数依维度、光滑度和风险定义；课堂 n^(−1/d) 用于直觉，不把神经 scaling 指数直接认成已测出的唯一内在维度。
- [00:22:30–00:29:50](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1350s) · 数据混合可能改截距，也可能跨尺度改排序；小模型选配比是实验方法而非定理。重复四轮近似无损只属所引研究条件，不能推广到所有语料；图的横轴看似线性却按倍增，原课已当场说明。
- [00:37:30–00:40:50](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2250s) · MoE 的总参数和活跃参数应同时列出；未被当前 token 选中的专家仍可服务其他 token。横轴去掉 embedding/output 参数会改变小模型区间拟合，不能只说参数天然同质。
- [00:41:40–00:52:20](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=2500s) · 临界 batch 取决于目标 loss 和训练状态，是步数与样本数之间折中。梯度噪声近似不是到全局最优的保证；学习率是否按宽度缩放还取决于参数化和优化器，不能统一记为1/width。
- [01:00:00–01:02:40](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3600s) · 讲者当场把 Kaplan 的 N、D 说反后纠正：N 参数约 C^0.73，D token 约 C^0.27；Chinchilla 两者近 C^0.5。20 token/parameter 是相应预训练计算目标与配方下的经验比例。
- [01:03:30–01:14:40](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3810s) · 下包络、IsoFLOP、联合曲面是不同估计法；平滑、参数口径、warmup、batch 与非线性拟合可改变答案。外推需验证范围与不确定性，scaling law 不是数学保证的性能下界。
- [01:14:40–01:17:40](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=4480s) · 将部署成本计入后，更小模型训练更多 token 可能更合算；这里 overtrained 不等于验证误差已经恶化的 overfitting。预训练 perplexity 的优劣也不保证下游任务排序相同。

问答处理：区分“用理论理解某种趋势”和“该理论已推导出现有大模型的精确曲线”。

演示：画同一计算预算下三种模型／数据组合。先问哪些变量必须固定，再讨论如何寻找最低验证损失。

理解检查：只知道训练 loss 随算力下降，为什么仍不能直接知道聊天回答会提高多少？

缺口：原图拟合点、置信区间和所有图中读数未逐帧核对；不输出未经复核的精确缩放指数。


### 2026 L10 · 推理：从 KV cache 到动态服务

对应陪学章节：07、15。官方讲义：[2026-10](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 10: Inference](https://www.youtube.com/watch?v=EfM546A79aM)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **prefill 与 decode**：prefill 处理已有上下文，许多位置可并行；decode 反复生成新 token。KV cache 保存过去位置的 K、V，减少重复计算，但缓存随请求和上下文增长。模型权重可在 batch 内共享读取，缓存通常随请求分别保存。
- **吞吐与延迟**：大 batch 可能摊薄权重读取，提高总体吞吐，却也可能让单个请求等待更久。必须说明测量的是首 token 延迟、输出 token 间隔，还是整段完成时间。
- **有损与无损捷径**：量化、剪枝或结构改动可能影响输出质量。推测采样先用便宜模型拟候选，再由目标模型按正确规则接受或拒绝；保持目标分布需要完整接受与残差采样规则，不是只检查“看上去合理”。
- **请求是动态的**：请求起止时间和输出长度不同。连续批处理及时补入新请求；分页式 KV 管理减少连续预留带来的空间浪费。它们改善调度和存储，不会让逻辑上下文无限长。

课堂展开、例子和讨论：

- [00:59:20–01:01:30](https://www.youtube.com/watch?v=EfM546A79aM&t=3560s) · 问答：线性状态、滑动窗口和全注意力保留历史信息的方式不同，可以混合使用；不能只比较缓存大小。
- [01:10:40–01:12:10](https://www.youtube.com/watch?v=EfM546A79aM&t=4240s) · 问答：激活很大不自动代表不可删除的语义特征，还可看方差和其他统计；但也不能随手删掉恒定大值。
- [01:19:49–01:22:15](https://www.youtube.com/watch?v=EfM546A79aM&t=4789s) · 问答与类比：课堂用旧式磁盘碎片解释 KV cache 的内部和外部碎片，再引入分页。类比只说明空间管理，不表示 GPU 缓存与磁盘完全相同。

二轮问答、例子与演示核对：

- [00:18:34–00:20:12](https://www.youtube.com/watch?v=EfM546A79aM&t=1114s) · 学生纠正 GQA 图的分组计数，讲者接受。按 Q 头数、KV 头数及每 KV 头对应的 Q 头数分别定义维度，不能直接照字幕混用 K 与组数。
- [00:28:30–00:29:20](https://www.youtube.com/watch?v=EfM546A79aM&t=1710s) · 讲者说明 decode 的 batch 通常来自并发请求，到达和退出会变化；不是用户总数，也不是必能自由放大的常量。
- [00:29:47–00:31:35](https://www.youtube.com/watch?v=EfM546A79aM&t=1787s) · 讲者比较 Attention 的 prefill/decode 算术强度。推导依赖所计入的读写与精度；batch 对 MLP 和 Attention 的影响不同，不能把一个强度公式推广到整个模型。
- [00:54:40–00:55:50](https://www.youtube.com/watch?v=EfM546A79aM&t=3280s) · 学生问 MLA 比直接缩小 hidden dimension 如何。图中未做该对照，讲者承认只能猜测；不能宣称有受控实验支持 MLA 在所有预算上优胜。
- [00:59:04–01:00:42](https://www.youtube.com/watch?v=EfM546A79aM&t=3544s) · 学生问线性 Attention 与滑动窗口。前者压缩历史状态，后者保留局部精确信息；混合层可以结合不同优势，不承诺统一优胜。
- [01:00:30–01:02:12](https://www.youtube.com/watch?v=EfM546A79aM&t=3630s) · 学生继续问长上下文取舍。针落草堆任务可能暴露压缩丢信息；讲者对 Mamba/DeltaNet 表达能力的口述直觉不是任意大小状态的严格包含关系证明。
- [01:01:57–01:03:44](https://www.youtube.com/watch?v=EfM546A79aM&t=3717s) · 稀疏/压缩 Attention 图解。将 token 聚合、用轻量 indexer 选择和继续压缩是不同步骤；课堂跳过细节，资料不补出精确算法或复杂度。
- [01:09:40–01:10:45](https://www.youtube.com/watch?v=EfM546A79aM&t=4180s) · 学生问剪枝怎么识别重要单元。可用校准集的 activation 统计；随后追问高均值不必然重要，不能把简单幅值当普遍因果重要性。
- [01:10:23–01:12:00](https://www.youtube.com/watch?v=EfM546A79aM&t=4223s) · 学生问总是很大的 activation 是否代表重要。通道异常值是经验观察；高均值与高方差不同，恒定项也可能影响输出。量化保护规则不能只凭“看起来大”决定。
- [01:19:23–01:21:06](https://www.youtube.com/watch?v=EfM546A79aM&t=4763s) · 讲者设问 KV 存储。内部碎片来自预留未使用空间，外部碎片来自分散空隙；PagedAttention 用逻辑块到物理块映射组织缓存，不改变注意力数学结果。
- [01:23:28–01:25:15](https://www.youtube.com/watch?v=EfM546A79aM&t=5008s) · 总结讨论架构变化可能改善推理。state-space、线性 Attention 与 diffusion 只作方向介绍，不能视为本讲完整教授了这些路线。

字幕全文校读补充与纠错：

- [00:00:00–00:05:10](https://www.youtube.com/watch?v=EfM546A79aM&t=0s) · 开场将服务 token 数与训练 token 数比较，只能说明数量级；训练每 token 有反向，推理受上下文、batch 和架构影响，不能推出相同 token 数就等于相同算力。行业量与框架热度只作当年背景，不作当前推荐。
- [00:12:30–00:20:20](https://www.youtube.com/watch?v=EfM546A79aM&t=750s) · 本讲 N=KV groups×每组 query heads，现场承认符号需改；F=4D 是示例约定，SwiGLU 参数匹配常用其他倍数。BF16 是本账本的假设，不是所有推理必用格式。
- [00:26:00–00:35:00](https://www.youtube.com/watch?v=EfM546A79aM&t=1560s) · MLP 可跨请求复用同一权重，因此 batch 有助算术强度；decode Attention 的各请求 KV 不同，batch 项会在理想比值中相消。GQA、缓存共享、量化或不同内核可改比值，不能说所有 Transformer 都绝对无从改善。
- [00:35:00–00:45:20](https://www.youtube.com/watch?v=EfM546A79aM&t=2100s) · latency≈(参数字节+B×每序列KV字节)/带宽是忽略其他瓶颈的理想估计；TPOT、端到端延迟与 TTFT 分开。TTFT 实际还包括排队、分词和调度，不总等于纯 prefill。trace 的超容量 batch 只属公式预测。
- [00:47:30–00:53:30](https://www.youtube.com/watch?v=EfM546A79aM&t=2850s) · MQA 是 KV head=1 的端点，讲者“没人用”是过强概括；GQA 质量需看具体训练。MLA 的 RoPE 常作用 Q/K 的位置部分，不是简单说对 K/V 都旋转。
- [01:05:00–01:07:40](https://www.youtube.com/watch?v=EfM546A79aM&t=3900s) · AWQ 图里保留少量 FP16 是说明显著权重的对照，不是最终 AWQ 必须混合精度。作者方法按激活选择每通道缩放以降低低比特误差，避免低效混合精度；依据原论文：https://arxiv.org/abs/2306.00978 。
- [01:12:00–01:16:40](https://www.youtube.com/watch?v=EfM546A79aM&t=4320s) · 精确 speculative sampling 用草稿 p、目标 q，接受 min(1,q/p)，拒绝后从归一化正部 (q−p)+ 重采样。整段接受时再取额外目标 token，序列首次拒绝后不继续接受旧草稿后缀；同分布不等于同随机种子输出逐字相同。
- [01:17:30–01:25:18](https://www.youtube.com/watch?v=EfM546A79aM&t=4650s) · continuous batching 允许迭代间换请求；选择性 batching、paged KV 管理、prefix sharing 各自解决不同问题。copy-on-write 在共享块被不同续写修改时分配副本，不把文本相同自动当跨模型缓存可共用。

问答处理：把固定模型的调度优化与换模型结构分开比较，避免把质量变化隐藏在速度改进里。

演示：用三个不同长度请求画固定批处理与连续批处理时间表，分别观察空闲位置与等待时间；不提供未实测的加速倍数。

理解检查：批量变大后吞吐变高，为什么用户仍可能觉得变慢？

缺口：缓存图、性能主图及关键帧已查看；未全帧审片，带硬件型号的延迟数字是原课/论文配置，不作为本机实测。


### 2026 L11 · Scaling 的实践、μP 与优化器

对应陪学章节：04、17。官方讲义：[2026-11](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 11:  Scaling Laws](https://www.youtube.com/watch?v=vTfEyOyzV9E)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **超参数需要迁移**：扩大宽度会改变激活和更新的尺度。μP 通过初始化和分层学习率等规则，力求让合适的设置跨宽度更稳定；它不是只换一个全局学习率，也不保证跨深度、任务和所有结构都成立。
- **实验复用**：WSD 把 warmup、稳定阶段和 decay 分开，让不同预算的实验有机会共用前面的训练。复用时仍需比较 schedule 的公平性，不能把中途 loss 当最终同等质量。
- **优化器是算法加实现**：Muon 利用矩阵结构和近似正交化更新。课堂讨论用 Newton–Schulz 的矩阵乘迭代，而不是每次做昂贵的完整 SVD。是否更好需在多个计算规模和合理调参后比较。

课堂展开、例子和讨论：

- [00:29:31–00:30:55](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1771s) · 问答：把预训练与后训练联合考虑的 scaling，在本讲时点仍没有统一成熟答案；资料保留这个开放边界。
- [00:40:00–00:45:50](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2400s) · 比较方法：新优化器在小模型上的领先可能在更大预算消失。要检查横轴定义、调参投入和规模依赖，不只看单点成绩。
- [00:55:03–00:56:55](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3303s) · 问答：SVD 的成本问题不等于 Muon 实现成本；讲者强调近似正交化算法和系统实现也是方法的一部分。

二轮问答、例子与演示核对：

- [00:19:20–00:20:45](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1160s) · 学生问相同 FLOPs 为何最优学习率范围很大。还变了模型大小等变量；相同计算量不意味着训练设置相同。
- [00:29:06–00:30:44](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=1746s) · 学生问将后训练纳入 scaling。讲者明确是开放问题；预训练的覆盖与多样性可能影响后训练，尚无本课提供的统一最优解。
- [00:39:52–00:41:32](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2392s) · 学生问直接套已有 scaling law 还是重新搜索。相近规模与设置可作起点；结构、正则化和预算变化可能要求重新拟合，不能把借用经验式称为已验证外推。
- [00:53:05–00:54:48](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3185s) · Muon 的大规模采用说明能工作，缺少同规模受控消融则不能证明一定优于 AdamW。小规模研究和大规模应用的证据级别不同。
- [00:54:44–00:56:20](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3284s) · 学生问 SVD 性能和超参数。实际使用有限次 Newton–Schulz 矩阵乘迭代近似正交化，并非每步精确 SVD；分层/参数组可用不同规则，仍需调参。后续问题在本段截断，另接连续记录。
- [00:56:15–00:57:30](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3375s) · 学生问学习率、batch 与其他超参数交互。全面网格成本随维数迅速增长；先搜索敏感项再局部调整是成本取舍，不证明交互不存在。
- [01:14:00–01:15:15](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4440s) · μP 稳健性消融：某些 learned gain、特殊优化器及大 weight decay 可破坏迁移。理论前提与经验可迁移范围分别说明，不把一条层宽规则当全架构适用。

字幕全文校读补充与纠错：

- [00:05:00–00:10:00](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=300s) · muP 不只是换初始化，还可能改每层学习率、输出及残差缩放。原始宽度极限与 MiniCPM 深度缩放配方应分清，不能把一种实现的超参直接搬到任何优化器/深度。
- [00:10:00–00:15:25](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=600s) · WSD 从 stable checkpoint 分叉，继续稳定学习率后再 decay；不是把已衰减终点随意续跑就等同同一长跑。终值取零或峰值10%取决于具体日程；每条分支仍花 decay 的算力。
- [00:15:50–00:28:20](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=950s) · DeepSeek 拟合最佳超参，MiniCPM 尝试使超参跨规模稳定；二者都需要小跑验证。粗网格存在离散误差，跨 MoE/语料转移只是实验观察，不把 StepFun 的平方根 batch 规则当通用定律。
- [00:42:30–00:49:20](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=2550s) · 优化器比较应分别调学习率和 weight decay，并同时检查算力轴、token/parameter 轴。小规模 speedrun 收益不能直接外推大跑；Marin 失败例显示漂亮幂律也可能在外推后偏离/发散。
- [00:50:00–00:53:30](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3000s) · Muon 对矩阵动量更新近似做谱方向归一化；Newton–Schulz 是计算近似，不是精确 SVD。矩形矩阵不能两侧都成为方阵单位正交，零奇异值也不能无条件说被扩成1；向量/embedding 等是否用 AdamW按具体实现。
- [01:00:00–01:12:30](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3600s) · muP 推导的 O(1) 指每坐标激活/变化，向量 L2 范数可随 sqrt(width) 增长。示意只处理深线性网、单样本 SGD、无抵消等假设；operator-norm 上界不能普遍换成等式。更新展开包含 W·delta_h、delta_W·h、delta_W·delta_h 三项。
- [01:12:30–01:16:40](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=4350s) · 宽度转移成功不保证深度、batch、注意力变体等任意改变都能共用最优超参；本讲最终结论是两套工具并用、验证尺度边界，研究尚无一套保证外推成功的配方。

问答处理：不要把经验有效的参数迁移规律讲成对所有网络的数学保证。

演示：比较两个宽度下未控制的激活尺度，再说明为什么需控制初始化和更新；公式推导按需要拆开，避免一次引入全部 μP 条件。

理解检查：比较 AdamW 和 Muon 时，为什么应分别调学习率，而不是强制共用一个数？

缺口：优化器图表、具体参数配方和全部 μP 推导细节仍以官方讲义为深读来源；未复现大型训练。


### 2026 L12 · 评估：分数怎样产生，又怎样失真

对应陪学章节：18。官方讲义：[2026-12](https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 12: Evaluation](https://www.youtube.com/watch?v=JpAxdTWQJxM)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **困惑度**：平均负对数似然衡量模型给真实文本的概率，perplexity 是它的指数。切分粒度、文本预处理和评估数据都影响结果；跨 tokenizer 直接比较 per-token 数字可能没有意义。
- **基准测的是系统**：选择题的提示、答案提取和是否生成推理会改变分数。聊天偏好还混合正确性、风格与用户群体。智能体评估同时包含模型、工具、上下文组织和执行环境。
- **污染与效度**：不直接训练测试题，也可能见过题目来源或同源材料。能通过单元测试不保证满足用户目标；安全评价也必须先定义风险、使用场景和测试边界。

课堂展开、例子和讨论：

- [00:26:44–00:28:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1604s) · 问答：测试污染可能来自题目的原始来源，不能只查完全相同的测试文本。
- [00:31:19–00:32:35](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1879s) · 问答：生成字母、生成解释后提取答案等流程会改变评价；所谓同一模型得分需要说明评测协议。
- [00:36:01–00:38:10](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2161s) · 课堂讨论：Arena 类投票来自特定用户分布；胜出可能受风格影响。类似 Elo 的显示方式并不把“哪个回答更好”变成像棋局胜负一样清晰。
- [00:59:36–01:00:50](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=3576s) · 问答：图形推理任务可用图像或文本／ASCII 表示输入，表示选择本身会改变任务。

二轮问答、例子与演示核对：

- [00:26:26–00:28:06](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1586s) · 学生问测试污染。未公开训练集时不能保证无污染；题目来源、衍生版本与解答都可能构成间接重叠，不能只检测测试题逐字出现。
- [00:30:59–00:32:42](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1859s) · 学生问选择题怎样评分。可对选项似然评分，也可采样字母或推理后提取答案；本段讲的是后者，提取协议会影响可比性。
- [00:32:30–00:34:11](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1950s) · 讲者用甜菜沙拉的开放问题解释为何不能精确匹配标准答案；人类成对偏好是另一种评估信号。
- [00:34:00–00:35:47](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2040s) · 讲者解释由成对胜负拟合评分。评分模型只是观测偏好的统计模型，当前榜首与具体分数不作为持久事实。
- [00:35:35–00:37:02](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2135s) · 关键词误报：不是现场软件演示。内容讨论 Arena 参与者分布、刷票和风格/正确性混杂；不能当成代表所有真实用户的无偏采样。
- [00:58:36–01:00:18](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=3516s) · 学生问图形推理的输入格式。可用图像或文本/ASCII 表示；空间结构仍在，编码方式应纳入评估协议。讲者未确认的尺寸不采用。

字幕全文校读补充与纠错：

- [00:05:00–00:12:35](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=300s) · 交叉熵下界是目标分布的熵，perplexity 下界是其指数；两者不能直接说同一个数。达到文本分布并不自动证明解开所有真实世界任务，“perplexity all you need”在课堂本来就是供讨论的信念。
- [00:12:30–00:17:50](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=750s) · conditional perplexity 只统计回答部分，仍受 tokenizer、长度和数据分布影响。多选题按答案概率选最大值是评价规则，不等于整段 perplexity；对黑箱回传 logprob 需验证归一化和模型实现。
- [00:23:00–00:25:10](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=1380s) · GPQA Diamond 的筛选口述有误：保留两位专家均答对、且多数非专家答错的198题；不是非专家能答对才入选。65% 为原研究整体专家结果口径，不能无条件套在 Diamond 上。依据原论文：https://openreview.net/pdf?id=Ti67584b98 。
- [00:37:00–00:43:20](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2220s) · 偏好可受讨好、长度、位置与裁判自身偏差影响。稀疏比较图连通是排名识别的重要条件，有限样本/完全分离仍可能影响估计。0.98 相关只适用那批模型，不能推出现有所有更强模型也等价。
- [00:45:00–00:58:40](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2700s) · Agent 成绩评价的是模型、scaffold、工具权限与预算的组合。单元测试通过可能漏掉正确性问题；ARC 的目标是减少知识依赖，不能凭设计目标就断言已完全分离知识与推理。榜单值保留课堂日期，不当当前事实。
- [01:00:20–01:08:30](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=3620s) · 安全评价包含多种风险；拒答率只覆盖一种目标。真实职业任务不等于已经证明实际部署收益；让模型汇总敏感对话也不自动提供隐私保证，仍需实际数据流程与保护机制。
- [01:10:00–01:17:40](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=4200s) · 新时间戳和私有测试都只是减少污染的手段，复制来源、反复调试和服务端留存仍需检查。比较方法与比较最终模型是不同目标；每次测评先说明对象、用途、预算、置信区间，再抽读真实输出。

问答处理：先定义你要测的能力和使用场景，再选择评测方式。

演示：把同一回答分别按事实正确、格式符合和语气偏好打分，展示一个总分会怎样隐藏差异。此为教学设计，不是重新跑课程榜单。

理解检查：一个智能体补丁通过全部现有测试，为什么仍可能没有真正修复问题？

缺口：排行榜与新基准的当前分数未更新；图形题和安全演示画面未逐帧复核。


### 2026 L13 · 数据来源、抓取与使用边界

对应陪学章节：19。官方讲义：[2026-13](https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 13: Data (Sources, Datasets)](https://www.youtube.com/watch?v=-qm0ln33G24)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **来源决定可学到什么**：网页、书籍、代码、百科和论文有不同风格、覆盖与噪声。把多个数据集名字背下来不如理解它们如何收集、清洗、许可和采样。相同 token 数不意味着相同有效信息。
- **抓到不等于可直接训练**：抓取结果可能包含导航、重复、敏感信息、恶意内容和不适合目标任务的文本。Common Crawl 提供抓取材料，开发者仍需自己的转换、过滤、去重和来源记录。
- **许可与版权讨论如何保留**：公开可访问、robots 规则、网站条款、版权许可和可用来训练是不同问题。本资料只解释课程讨论的区别，不把课堂对诉讼、fair use 或许可变更的推测升级成当下法律结论。

课堂展开、例子和讨论：

- [00:31:33–00:33:05](https://www.youtube.com/watch?v=-qm0ln33G24&t=1893s) · 问答：有人问许可随时间变更。课堂回答带有明确的不确定性；学习资料保留问题，并要求实际使用时核对具体版本、内容、条款与适用法律，不能抽成通用授权规则。
- [00:43:39–00:45:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=2619s) · 问答：网络抓取可能碰到不应使用的来源，规模化并不自动解决来源合法性。合成数据也不能仅凭“模型生成”跳过来源和使用条件检查。
- [01:10:00–01:13:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=4200s) · 数据例子：代码数据不只有文件内容，还涉及仓库、提交和去重单位。编译或运行验证是特定数据处理手段，不等于所有代码文本都已被验证。

二轮问答、例子与演示核对：

- [00:24:14–00:25:58](https://www.youtube.com/watch?v=-qm0ln33G24&t=1454s) · 讲者设问版权与训练。仅存为法律背景讨论；不把字符重合率、课堂观点或历史判例解读当成当前使用许可。教学如涉及具体法律须另核实当时法规和裁判。
- [00:30:21–00:32:07](https://www.youtube.com/watch?v=-qm0ln33G24&t=1821s) · 学生问声音数据及许可随时间变化。前者讲者不熟悉；后者带明确非律师声明，未作已核实法律结论。许可版本和数据版本应分别留证。
- [00:43:29–00:45:02](https://www.youtube.com/watch?v=-qm0ln33G24&t=2609s) · 学生问合成数据许可及爬虫怎样避开盗版。讲者未给全面保证；公开可抓取不自动意味着可任意使用，合成输出还需检查适用条款。此处不提供法律判断。
- [01:09:14–01:11:05](https://www.youtube.com/watch?v=-qm0ln33G24&t=4154s) · 讲者提醒训练 token 数可能包含重复轮次；来源 token、独立 token 与实际训练 token 必须分开。质量过滤存在规模与筛选强度取舍。

字幕全文校读补充与纠错：

- [00:05:00–00:15:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=300s) · 互联网、公开网页、爬取快照和最终训练集是不同对象。动态服务、登录、robots、限流、使用条款与作品许可分别处理；能访问不自动表示可再分发或可训练，robots 也不能独自给出全部法律结论。
- [00:15:00–00:20:20](https://www.youtube.com/watch?v=-qm0ln33G24&t=900s) · 版权不是统一75年：美国1978年后作品通常作者终身加70年，匿名/雇佣作品另有95年首次发表或120年创作取先等规则；旧作品还需另查。依据：https://www.copyright.gov/help/faq/faq-duration.html 。
- [00:19:00–00:23:40](https://www.youtube.com/watch?v=-qm0ln33G24&t=1140s) · Creative Commons 是带具体条件的许可系列，不等于公有领域。需区分署名、相同方式共享、非商业、禁止演绎等条款，不能只看到CC就宣称任意使用；依据：https://creativecommons.org/cc-licenses/ 。
- [00:15:00–00:30:50](https://www.youtube.com/watch?v=-qm0ln33G24&t=900s) · 事实/思想与表达不同，汇编的原创选择编排可能受保护；本讲后面自己也收回“汇编都不受保护”的过强说法。fair use 按具体事实综合四因素；诉讼进展只按课堂历史叙述，不写成所有AI训练均已合法或当前案件状态。
- [00:32:10–00:45:00](https://www.youtube.com/watch?v=-qm0ln33G24&t=1930s) · Common Crawl 的 WARC 保存抓取记录，WET 是有损文本产物；页面数、总历史数和某次 crawl 数不可混算。GitHub 仓库内容、事件元数据、论文 PDF/LaTeX 是不同数据对象，各自需转换及检查来源。
- [00:45:00–00:56:30](https://www.youtube.com/watch?v=-qm0ln33G24&t=2700s) · BERT、GPT-2、CCNet、C4、GPT-3、Pile 的链条展示采集和过滤策略如何变化，不能凭缺少明确细节推断 Books1/2 的实际来源。PG-19 是以1919年前出版书籍形成的长文本语料（作者仓库：https://github.com/google-deepmind/pg19 ），不能按字幕误当2019年前书籍。
- [00:58:20–01:10:50](https://www.youtube.com/watch?v=-qm0ln33G24&t=3500s) · Llama 的 Wikipedia引用页 classifier 与 CCNet 的 Wikipedia相似度 LM 不同。RefinedWeb/FineWeb、Dolma、DCLM、Nemotron 以不同规则或模型取舍质量、规模、合成；数据集包装许可不能替每个来源作品授权。
- [01:11:00–01:20:10](https://www.youtube.com/watch?v=-qm0ln33G24&t=4260s) · 代码需处理重复、二进制、机器人事件、PII与补丁上下文；低资源代码与中间表示配对是特定数据设计，不表示所有语言都能无损编译成LLVM。Common Pile 对原始作品许可逐项处理，集合上写Apache并不足以证明底层内容可用。

问答处理：把“来源是什么”和“课程如何使用它”分开，不暗示学习资料附带这些数据的使用许可。

演示：用一个虚构网页展示原始 HTML、正文、链接和来源记录四层；不抓取私人网站，也不复制课程数据集。

理解检查：为什么一个完全公开的网页也需要记录来源和使用条件？

缺口：课堂涉及的法律和诉讼背景未做当前法律核查；不作为法律建议或可训练数据清单。


### 2026 L14 · 过滤、去重、混合与合成数据

对应陪学章节：20、21、22。官方讲义：[2026-14](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 14: Data](https://www.youtube.com/watch?v=5sxHosTLPF8)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **转换与过滤**：HTML／PDF 转文本要尽量保留有用结构。规则、语言识别、模型评分各有偏差；过滤标签的含义应与训练目标一致。最终需要通过训练实验验证，而不只是看被保留文本是否“漂亮”。
- **近似去重**：把文档表示为 n-gram 集合，Jaccard 是交集大小除以并集大小。理想 MinHash 的碰撞概率等于相似度 s。LSH 用 b 个 band、每 band r 个哈希，候选碰撞概率为 1−(1−s^r)^b；它是概率筛选，不是确定阈值判定。
- **混合不只是百分比**：若总训练 token 数为 T，来源 i 的采样权重为 w_i，来源可用 token 数为 D_i，预期经过该来源的轮数约为 T·w_i/D_i。高权重的小来源可能被无意重复很多次。
- **合成数据的验证**：合成任务、答案和智能体轨迹需要检验目标、难度、正确性、泄漏和多样性。自动生成环境不等于奖励没有漏洞。

课堂展开、例子和讨论：

- [00:20:34–00:22:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1234s) · 问答：重复训练以估计置信区间更理想，但成本会限制实验次数；即使结果看似稳定，也不应伪造统计置信度。
- [00:38:00–00:41:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2280s) · 算法演示：课堂用 12 个哈希分成 3 个 band、每 band 4 个解释如何把相似度转成更陡的候选概率。
- [00:56:37–00:58:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3397s) · 问答：出现几十个 epoch 不表示需要训练那么多次，可能只是混合配方造成的隐性重复。课堂建议检查每个来源的实际轮数。
- [01:10:52–01:12:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4252s) · 问答：极小来源在下采样后可能因取整丢失，混合算法仍要考虑有限样本。

二轮问答、例子与演示核对：

- [00:20:15–00:21:54](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1215s) · 学生问重复训练与置信区间、好数据多训是否递减。重复实验理想但昂贵；有限数据也有边际收益下降，不能从“质量好”推导无限重复都有效。
- [00:32:02–00:33:54](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1922s) · 讲者推导 MinHash 与 Jaccard 的联系。碰撞概率关系依理想 minwise 假设；MinHash 本身尚非完整高效近重复搜索流程，还需要 LSH 和候选处理。
- [00:49:59–00:51:49](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2999s) · 讲者设问怎样混合来源。混合权重是一种分布；手工设置和算法搜索都是实际方法，权重不能只看来源质量而不看数据量。
- [00:55:54–00:57:32](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3354s) · 学生问为何做 50 轮及混合怎样实现。50 轮是忽略小来源体量后的意外结果，不是推荐；重复可能浪费算力或过拟合。实现可用随机来源采样，需与后续说明连读。
- [01:10:22–01:12:07](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4222s) · 学生问缩小代理集的风险及能否在同一语料内部混合。小集合可能太少或被取整成零，代理最优并不自动可靠外推。
- [01:11:55–01:13:36](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4315s) · 回答内部混合：可先按主题与质量划格，再把每格当一个来源；后训练数据还具有任务目标，不只是预训练来源权重问题。

字幕全文校读补充与纠错：

- [00:02:30–00:07:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=150s) · PDF比HTML更高质量是讲者经验判断，不是格式自带可靠性。OCR/VLM可恢复阅读顺序，也会错公式、合并列或漏标点，需抽查；表格转纯文本或Markdown会丢布局与嵌套信息。
- [00:07:30–00:22:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=450s) · KenLM是n-gram工具，不固定只能5-gram；fastText可用词/子词及监督分类结构，不能把所有配置简化为同一种线性模型。过滤阈值依语言、目标、算力和可用数据，去掉毒性可能同时改变方言/群体覆盖；无限算力也不自动消除安全或数据分布需求。
- [00:26:50–00:37:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1610s) · 精确哈希需要处理碰撞；近似去重需定义分片单位和规范化。随机排列的MinHash碰撞概率等于Jaccard有理想随机性假设，100次估计偶得0.6不等于每次都精确；通用快速hash只是近似。
- [00:37:30–00:48:10](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2250s) · LSH概率为1−(1−s^r)^b，候选仍有误报/漏报，通常再算相似度。s≈b^(−1/r) 是转折位置近似，概率约1−e^(−1)≈0.632而非严格0.5；课堂12=3×4图与5×10数值另属两组参数。
- [00:48:10–01:00:00](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2890s) · 去重要跨合并后的来源，否则同文本在不同集合仍重复；近似删除不保证零信息损失。epoch_i=p_i×训练token总数/源token数，示例低质源0.05轮、高质源50轮；epoch cap约束应乘以源token数才能保持单位一致。
- [01:00:00–01:10:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3600s) · RegMix等先拟合小跑混合权重到loss，再优化权重；优化器可能钻代理模型外推误差。simulated epoching 按小/大token预算比例缩小每个源，保留稀缺性；只把模型缩小而保留全部小优质源会得过乐观配比。
- [01:13:40–01:23:20](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4420s) · 强模型未必最好老师，多次回答数与独立问题数分开。SWE-smith生成任务，SWE-Zero减少执行依赖，SWE-rebench收集真实PR/环境；未执行的轨迹不等于已经通过测试。字幕重复说两次SWE-Zero和“120”数量有歧义，不据此编造第二阶段数据或精确规模。

问答处理：MinHash 估相似性，LSH 找候选；不要把两者合并成“精确删除所有近重复”。

演示：教学重建 D06：计算不同 Jaccard 相似度下 3×4 LSH 的候选概率；再用来源轮数公式检查一个小数据源的重复。

理解检查：把高质量来源权重调高，为什么既可能改善模型，也可能造成重复过多？

缺口：PDF转换、混合图和合成轨迹的讲义主图与关键帧已核对；未逐帧运行现场流程。概率推导声明理想哈希假设。


### 2026 L15 · SFT、偏好数据与 RLHF

对应陪学章节：22、23、24。官方讲义：[2026-15](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 15: Mid/Post-Training](https://www.youtube.com/watch?v=2oH6PWPrYFo)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **SFT 的目标**：用输入和目标回答训练条件概率，常见对话训练只在 assistant 回答上计损失。它与预训练共享预测机制，但数据、格式、目标分布和训练阶段不同。具体 loss mask 以实际配方为准。
- **风格不等于事实**：少量示范可能很快改变回答格式和语气，却未必让模型可靠掌握新知识。缺乏知识时强制模仿详细回答，可能增加编造。安全微调也需同时看误拒绝与漏拒绝。
- **从模仿到偏好优化**：人可能擅长比较两份答案，却写不出最好答案。RLHF 使用偏好、奖励与策略更新改善输出，同时常用 KL 等约束避免过度偏离。DPO 直接使用偏好对与参考策略的相对概率构造目标，不需显式在线 PPO 循环。
- **奖励与事实之间**：标注者可能有共同偏见，模型评审也可能偏好长度和风格。奖励过优化和模式收缩说明“更高奖励”不能单独证明更好、更真实或更校准。

课堂展开、例子和讨论：

- [00:38:53–00:41:10](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2333s) · 问答：课堂讨论 decay 阶段的数据质量与中后训练混合。接近最终模型、学习率较低的阶段常被用于更精选数据，但这是经验配方，不是无条件规律。
- [00:56:48–00:58:45](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3408s) · 问答：标注者之间一致性衡量分歧，不能证明没有共同偏差。大家都依赖同一个模型时，一致也可能只是复制同一偏差。
- [00:59:20–01:01:15](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3560s) · 问答：模型辅助标注有价值，但领域专用模型是否更好要看能力和验证，不能从“专用”二字直接判断。

二轮问答、例子与演示核对：

- [00:05:42–00:07:01](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=342s) · 关键词误报：demonstration data 指人工示范回答，不是软件演示。SFT 沿用似然训练，主要变化是训练数据及损失掩码。
- [00:10:10–00:11:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=610s) · 学生问 SFT 数据必须全对吗。高质量为目标；某些实验可从残缺/奇怪数据激发指令行为，但不意味着错误答案对能力和事实性无害。
- [00:20:33–00:22:10](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1233s) · 学生问复杂能力，讲者留待后文；先区分风格偏好分数与能力。用户参与度改善也可能只来自语气与格式。
- [00:25:35–00:26:20](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1535s) · 学生问 tail knowledge 的严格定义。课堂没有统一定义，使用文章长度等代理；代理量不能直接断言模型“知道/不知道”。
- [00:26:07–00:27:42](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1567s) · 学生问 RL 为何可能降低幻觉。讲者给出“知道时引用、不知道时不引用”的直觉故事，不是已证明的内部方向或保证；SFT 对示范之外输出分配概率会受损失约束。
- [00:34:11–00:35:47](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2051s) · 学生问如何知道能力已在预训练中，以及 SFT/RL 的本质差异。讲者承认无法直接判定所有已有能力；反馈形式和由自身策略采样，比“破坏/提取特征”的二分更稳妥。
- [00:38:28–00:40:08](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2308s) · 学生问 prompt 是否遮罩、衰减阶段数据质量和代理混合。示例 midtraining 会预测 prompt；SFT 掩码配方不一。高质量数据通常留到后期，但混合比例仍需试验。
- [00:40:47–00:42:26](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2447s) · 学生追问如何据下游增益调混合。删除来源的消融可提供排序线索，不能自动得出精确权重；高质量来源体量不足以支撑整个预训练。
- [00:45:09–00:46:22](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2709s) · 关键词误报：人工示范与偏好判断的难度不同。验证有时比生成容易，这是采集比较数据的理由，不是课堂运行的演示。
- [00:52:38–00:53:55](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3158s) · 关键词误报：demographics。标注者的群体分布会影响后训练结果；此处没有软件演示。
- [00:54:19–00:55:37](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3259s) · 关键词误报：demographics。特定研究发现数据可传递隐蔽偏好；这是论文案例，不能推广成任意合成数据必然产生同一效应。
- [00:56:08–00:57:26](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3368s) · 学生问标注质量定义。可检查细化规则是否遵守；没有对任意任务自动判定好标注者的统一准则。
- [00:57:26–00:59:01](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3446s) · 学生追问专家标注。标注者一致性主要反映差异，不能排除共同偏差；专业任务可能需要专家知识，不能只用学历替代质量检验。
- [00:59:07–01:00:41](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3547s) · 学生问领域模型作标注者。需要比较实际能力与最强通用模型；LLM 标注可用于追赶已有能力，但不自动创造超出教师的新知识。
- [01:02:49–01:04:25](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3769s) · 学生问案例模型规模。规模是历史实验设置，不按今天评价“够大/太小”；Constitutional AI、自训练和蒸馏是不同的数据生成机制，不能混成一种。

字幕全文校读补充与纠错：

- [00:07:30–00:18:10](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=450s) · FLAN任务模板与后来的多轮chat/工具轨迹不同，课堂“FLAN不好”针对用户分布匹配，不能删掉其指令泛化贡献。Alpaca 原版52k指令来自 text-davinci-003，不应按口述统称为直接蒸馏ChatGPT；Vicuna和OpenAssistant来源另列。
- [00:20:00–00:27:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1200s) · 训练正确但模型未知的引用，可能同时教知识与输出引用格式；泛化时只学会格式会幻觉。这是需实验检验的机制，不是SFT绝不能学新知识或RL必能校准的定理。
- [00:27:50–00:35:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1670s) · 安全SFT需同时看违规率与误拒率，小规模例子只证明一定幅度可被少量数据改变；不证明500例足够覆盖真实使用。专家薪资、数据量与供应商调查只属所引年份和样本，不作当前行情。
- [00:35:50–00:40:50](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2150s) · mid-training可把指令数据混入衰减期，“base”名称不足以确认从未见过chat数据。用衰减阶段快速比较配比是代理试验，仍需检查能否转移回完整预训练。
- [00:42:30–01:05:40](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2550s) · SFT拟合示范分布，纯期望奖励最大化可能集中到少数答案；有KL/entropy约束时不应说必然完全塌缩。人偏好和模型偏好均受格式/长度影响，合成反馈不能独立证明真实用户收益。
- [01:07:30–01:15:30](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4050s) · PPO旧策略与KL参考策略是不同角色。DPO的指数倾斜最优策略先来自KL正则奖励目标；将隐式奖励代入Bradley–Terry偏好似然才得DPO，不能只说直接代回原RL目标就自动得到同一损失。
- [01:12:30–01:19:30](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=4350s) · DPO增加偏好对的相对对数概率裕量，不保证每个chosen样本绝对概率都升、rejected都降。数据覆盖、参考策略支持和神经模型约束仍重要；原课对RFT等的负面评价限所引比较，不删除本课下一讲的有效基线。

问答处理：SFT、RLHF 与 DPO 是训练方法，不是事实核验器。

演示：给两份答案分别标注事实、风格和长度偏好，构造一对偏好数据，再指出奖励可能学到的捷径。

理解检查：所有标注者都选同一个答案，为什么仍不能保证答案正确？

缺口：众包薪酬、平台与人口分布等时点数字未更新；不纳入长期知识结论。


### 2026 L16 · RLVR、GRPO 与推理模型

对应陪学章节：24、25。官方讲义：[2026-16](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 16: Post-Training - RLVR](https://www.youtube.com/watch?v=dIFAi87Ws4E)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **把生成视为策略**：问题是输入，生成过程是轨迹，评分器给回报。策略梯度增加高于基准的轨迹概率、降低低于基准的轨迹概率。基准旨在减少估计噪声，其设计与采样方式关系很大。
- **PPO 与 GRPO**：PPO 常使用 value 模型估计 advantage，并限制新旧策略概率比的更新。GRPO 用同一问题的多个回答形成比较组，可省去独立 value 网络；组内均值、标准差和长度归一化会改变不同样本的权重。
- **为什么长推理要小心**：输出变长可能有帮助，也可能来自归一化偏差，尤其应分开看正确和错误回答的长度。奖励可验证不等于训练信号完美，评分器、格式和环境都可能被利用。
- **训练阶段的配合**：长 CoT SFT、RLVR、通用偏好优化、蒸馏和智能体训练可按不同配方组合。学习资料保留这些组合的目的和取舍，不给所有模型套同一流水线。

课堂展开、例子和讨论：

- [00:38:41–00:40:15](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2321s) · 问答与曲线解释：课堂把正确与错误回答的长度拆开，说明总体 CoT 增长不能单独证明推理更好。
- [01:10:39–01:12:30](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4239s) · 问答：本讲所举 thinking／non-thinking 案例通过同一模型的提示标记控制，不应推断所有 API 的模式切换都采用同一种内部机制。
- [01:13:52–01:15:25](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4432s) · 问答与明确缺讲：讲者表示没有充分展开长上下文扩展，并解释它常是另一个阶段。资料明确保留这项深度缺口，不把提到过当作已完整教授。

二轮问答、例子与演示核对：

- [00:05:20–00:06:42](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=320s) · 视频引用机器人运动说明 PPO 可用于高维状态/动作；语言模型中的轨迹与优势估计仍需另定义。出现 demo 字样不意味着本堂现场训练机器人。
- [00:10:39–00:11:58](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=639s) · 课堂解释论文曲线和 PPO 工程难点，不是已在本机跑过的训练。奖励提高与 KL 变化要分清曲线符号和正则项约定。
- [00:38:08–00:39:49](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2288s) · 学生问长度归一化对正负样本的影响。它改变长短响应的权重；平均长度增长必须拆成正确与错误响应看，不能等同于推理能力提高。
- [01:10:07–01:11:50](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4207s) · 学生问 thinking 开关与 midtraining。所讨论案例用同一模型和 prompt 控制；不推广到所有 API。midtraining 如何决定 RL 泛化仍有经验细节，没有统一必要条件。
- [01:12:31–01:14:09](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4351s) · 学生问专家蒸馏与长推理所在阶段。蒸馏需要覆盖各专家能力的 prompts；长 CoT SFT 与长上下文扩展是不同数据目的，阶段命名并不固定。
- [01:14:25–01:15:45](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4465s) · 学生问多领域 RL 顺序与遗忘。讲者引用特定配方把推理领域一同训练，再做非推理 RLHF；没有给出适用于所有多任务训练的无遗忘保证。

字幕全文校读补充与纠错：

- [00:07:30–00:12:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=450s) · PPO展示中的逐样本log-ratio可为负，真实KL期望非负；截断样本估计会改目标，课堂自己说并非推荐实现。token级KL可以看作稠密reward，不意味着无法从序列级bandit角度等价组织。
- [00:15:00–00:18:30](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=900s) · on-policy初始ratio值为1仍有梯度，旧策略分母和优势需detach；把新旧概率都当同一个可微变量会使梯度消失，D07验证。G=1时样本std常为NaN，加epsilon不能自动修好，需明确std定义及组大小。
- [00:20:00–00:25:40](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1200s) · action-independent baseline保持策略梯度期望；同组均值含自身回报时有限组会有缩放偏差，leave-one-out不同。除以组内std和序列长度又改变权重。所有reward相同的组中心化后为0，不能说除零方差就必产生巨大正负信号。
- [00:27:30–00:37:30](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=1650s) · R1-Zero无专门冷启动SFT不表示没有强预训练或完全没见过指令。长CoT和aha字样不是独立推理能力证明；过程监督/MCTS未奏效是特定研究结果，不是全部任务下不可能有效。
- [00:42:10–00:43:20](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2530s) · Kimi best-of-8筛的是不写CoT直接猜答案仍能猜中的易hack题，不是把所有能解出的RL题都删除。另用SFT模型高温采样10次估难度，二者不同；依据原报告2.1：https://arxiv.org/html/2501.12599v1 。
- [00:45:00–00:55:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2700s) · Kimi相似最优点启发的平方损失是代理目标；相同最优点不自动保证所有梯度相同。缩短错误CoT不能压至毫无探索空间；数学等价检查模型和代码单元测试均可能误判，verifiable不是不可hack。
- [00:55:30–01:01:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3330s) · Qwen thinking/non-thinking融合、预算提前终止与通用RL分别作用，性能降幅是所引实验结果。Qwen3-Coder-Next名称在课堂最后纠正，不凭口头混用3.5/Next-Coder建立新模型来源。
- [01:03:20–01:14:10](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=3800s) · 专家agent分别训练再蒸馏回单模型，与MoE层内专家不是同一概念。Git未来提交泄漏说明奖励上升可能是环境漏洞；Lean案例是讲者项目经验，不据此泛称所有Lean内核均不可靠。截断rollout、权重同步和策略陈旧度需一起考虑。

问答处理：把“最终答案可判对错”与“思维链每一步都正确、忠实”分开。

演示：教学重建 D07：同一问题生成四个有不同回报的回答，计算组均值和中心化回报。再检查奖励全相等时的更新信息。

理解检查：四个回答都答错且奖励相同，组内比较还能告诉模型哪个更好吗？

缺口：完整长上下文扩展配方课程未展开；具体 GRPO 变体的实现细节由当前 A5 补充，未复现大模型 RL。


### 2026 L17 · 多模态：视觉表示接到语言模型

对应陪学章节：27。官方讲义：[2026-17](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Lecture 17: Alignment - Multimodality](https://www.youtube.com/watch?v=26FtD08ZpOU)。字幕状态：已取得，未完成逐句听校。

讲义主线：

- **图像怎样进入模型**：视觉编码器把图像变成向量序列，连接器把它映射到语言模型能使用的表示空间。视觉 token 不一定来自文字分词器的离散词表。分辨率、裁剪与帧数影响序列长度和成本。
- **CLIP 与 SigLIP**：对比学习让匹配图文表示接近、不匹配表示可区分。CLIP 的 batch 内对比与 SigLIP 的成对 sigmoid 目标不同；图文匹配能力不等同于会进行多轮视觉对话。
- **对齐与指令训练**：某些 LLaVA 阶段冻结视觉编码器和语言模型，只训练连接器；后续阶段可能解冻更多参数。先有较好的语言与视觉表示，连接器才有可对齐的内容。
- **输入多模态与输出多模态**：能读图并输出文字，不等于能生成图像或视频。原生多模态生成还涉及视觉离散编码、目标和稳定性；不能把所有模型都按“图像前加一个 adapter”概括。

课堂展开、例子和讨论：

- [00:43:56–00:45:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2636s) · 例子：课堂用表格、图表和多图推理说明不同训练技能可能组合迁移到新情境；这不是对任意组合都成功的保证。
- [00:59:54–01:01:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3594s) · 问答：讲者明确当前讨论的一组模型在输出端只生成文字，图片和视频主要在输入端。随后另讲可生成视觉内容的路线。
- [01:03:23–01:04:50](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3803s) · 问答：连接器对齐阶段所用语言模型已预训练；不把冻结策略推广到所有阶段。

二轮问答、例子与演示核对：

- [00:10:42–00:12:25](https://www.youtube.com/watch?v=26FtD08ZpOU&t=642s) · 学生问为何用图文对而非仅图像增强。文本可给出超出裁剪/旋转不变性的语义监督；不能把图像增强视为自动提供所有概念关系。
- [00:15:13–00:16:59](https://www.youtube.com/watch?v=26FtD08ZpOU&t=913s) · 学生问二维位置。分类结果中位置设计差别不大不代表空间推理也不重要；EOS 表示整个文本是该编码器的具体做法。
- [00:18:25–00:20:09](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1105s) · 学生问同一 batch 有多只狗是否混淆。可能出现伪负例和噪声；网络图文配对也很嘈杂，过滤与大量数据有作用，不能说负例标签总准确。
- [00:23:40–00:25:23](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1420s) · 学生问 SigLIP 是否需要特殊负采样。初始方案沿用成对矩阵与 sigmoid 目标；标签平衡和难负例仍是一般设计选项，不是被该目标彻底消除。
- [00:43:30–00:45:16](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2610s) · 课堂案例展示技能组合迁移：单图 OCR、跨图关系、圈选提示可迁移到多图/视频。仅对报告中的任务有证据，不承诺任意组合成功。
- [00:58:56–01:00:41](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3536s) · 学生问视频生成。此前所讲 VLM 主要输入多模态、输出文字；RL 之外多阶段仍按监督 token 训练，不能把所有训练都解释为 LLM judge。
- [01:00:31–01:02:09](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3631s) · 学生问多模态系统开销。视频加载和预处理可成瓶颈，应与计算异步重叠；增加视觉 token 还影响序列计算成本。
- [01:02:35–01:04:36](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3755s) · 学生问连接器对齐时 LM 是否预训练。该阶段冻结已有编码器与 LM 并训练 adapter；不是所有阶段永远冻结，也不是达到某个神奇阈值就自动对齐。
- [01:04:30–01:06:00](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3870s) · 学生问视觉编码器参数是否较小。讲者在画面数字中有查找和自我纠正；只保留具体模型组件分开计参数，不采纳“ViT 仅逐 patch 局部操作”的过度解释，视觉 Transformer 可跨 patch 交互。
- [01:06:09–01:07:55](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3969s) · 讲者转入 Chameleon，区分连续视觉向量作为输入与离散视觉编码参与生成；视频中的口误式模型缩写不用于命名来源。

字幕全文校读补充与纠错：

- [00:00:00–00:03:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=0s) · 本讲临时从RL改为多模态概览，正式资料应按实际内容而非原日程标题。连续视觉token指向量序列，不必对应文本词表中的某个离散词；语义单元只是设计目标，BPE也不保证每片有完整语义。
- [00:05:00–00:17:30](https://www.youtube.com/watch?v=26FtD08ZpOU&t=300s) · CLIP用归一化图/文向量和温度形成双向对比交叉熵。ViT与ResNet变体的pooling方式不能混为一谈；讲者把注意力池化说进ViT描述，教学时以原模型实现确认。32k是训练设置，不是小于某固定batch就数学上不能训练。
- [00:21:40–00:27:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1300s) · SigLIP对图文配对做二元sigmoid分类，正例在对角、负例在非对角，字幕重复“diagonal”有误。它取消跨batch softmax归一化耦合，但仍要考虑负例采样、类别不平衡与分布式传输。
- [00:30:00–00:35:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1800s) · LLaVA用已有图像caption/框的文本描述让文本教师生成对话，不能声称原始GPT-4直接看过每张图。适配器学习把视觉特征接入LM，不是已证明输出向量变成真实文本token。阶段冻结范围按具体版本。
- [00:35:50–00:52:30](https://www.youtube.com/watch?v=26FtD08ZpOU&t=2150s) · 高分辨率裁块、动态视觉token及2×2合并是不同模型的处理细节，不把所有Qwen2-VL处理等同固定224裁块。AnyRes版本归属、token数口述有歧义，精确配置需回对应论文/代码，不按字幕66等数背诵。
- [00:52:30–00:58:40](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3150s) · M-RoPE分配时间/高/宽的位置维度，interleaving改善频率分配；时间戳token又是显式文本线索。DeepStack把多层视觉特征注入LM不同深度，来源归属不按口头猜测。平方根loss归一化在讲者原话也不确定，不补写成已确认实现。
- [01:07:50–01:15:10](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4070s) · VQ-VAE以码本量化得到图像离散token，训练不仅重建损失，还有codebook/commitment与梯度估计等项。Chameleon统一token空间仍有模态统计和稳定性差异；没有adapter不等于没有图像编码/解码器。
- [01:15:00–01:16:30](https://www.youtube.com/watch?v=26FtD08ZpOU&t=4500s) · 闭源omni结构是讲者推测。音频、视频生成和diffusion没有在本讲展开，不得声称已经教完完整omni系统；视频信息密度比较也依采样和任务，不是通用物理事实。

问答处理：先问“这个模型的输入和输出分别是什么模态”，再解释结构。

演示：画出图片 → patch 向量 → 连接器 → 与文字合并 → 语言模型 → 文字回答。用同一图说明高分辨率为什么增加成本。

理解检查：一个模型会描述视频，为什么不能据此推断它会生成视频？

缺口：图片裁剪、M-RoPE与视频位置讲义主图已核对；未完整逐帧验证原示例视频。视觉生成与diffusion仍为课程概览深度。


### 2026 L18 · Dan Fu：推理系统、Megakernel 与循环模型

对应陪学章节：11、15、17、28。官方讲义：[2026-19](https://cs336.stanford.edu/#schedule)。

官方录像：[Stanford CS336 Language Modeling from Scratch | Spring 2026 | Guest Lecture: Dan Fu](https://www.youtube.com/watch?v=9EEm4iMAF5s)。字幕状态：已取得，未完成逐句听校。

编号说明：这是播放列表第 18 项，官网日程第 19 讲；不是 Daniel Selsam 那场日程条目。

讲义主线：

- **从请求生命周期看系统**：请求进入调度、prefill、decode、缓存和工具调用等环节。长代码输入、长推理输出与短聊天的负载形状不同；一种优化不必然适合所有负载。
- **缓存像需要管理的内存**：多轮和工具调用产生可复用前缀与不同等待时间。把缓存放在 GPU、CPU 或更慢存储之间调度，需要考虑容量、传输、复用概率和尾延迟。操作系统类比帮助理解，但不提供最优策略保证。
- **Megakernel 的目的**：多个单独高效 kernel 之间仍有启动间隙、尾效应和依赖等待。更大范围融合与细粒度调度可提前加载后续所需数据，但设计复杂、硬件与模型适配成本高。
- **循环模型的取舍**：重复使用一组 block，让参数量与计算量更独立地变化；循环还可能放大状态和训练失稳。少参数可改变显存、缓存容量和跨卡通信需求，但是否改善质量和速度仍需实验。

课堂展开、例子和讨论：

- [00:07:24–00:08:50](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=444s) · 明确的图示边界：讲者说明部分概览图由生成式工具制作，细小文字可能错误。资料只保留口述确认的高层流程，不抄图上的数字和标签。
- [00:27:51–00:29:50](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=1671s) · 问答：缓存卸载与经典内存调度相似；LRU 是启发式，不是知道未来。未重复未经验证的最优性倍数说法。
- [00:37:32–00:40:45](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=2252s) · 演示解释：讲者用 GPU 时间条说明更大范围融合，并解释可在 QKV／RoPE 尚未完成时准备某些后续数据。这里记录调度意图，不声称已核验原图各色条和跑分。
- [00:50:00–00:51:40](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3000s) · 数学边界：口述把谱半径近似说成 norm。教学资料使用准确区分：谱半径是特征值绝对值的最大值，一般不等于矩阵范数；负对角本身也不足以推出离散迭代收敛。具体稳定性需结合实际递推公式。
- [01:03:37–01:05:20](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3817s) · 问答：手写 Megakernel 的适配和维护成本很高。资料保留这个工程取舍，不把课堂的人年估计当作普遍成本标准。

二轮问答、例子与演示核对：

- [00:07:03–00:08:22](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=423s) · 讲者明确部分总览图由生成工具制作，细标签可能错；只采纳口述确认的流程，不复制图中小字为技术事实。
- [00:10:11–00:11:51](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=611s) · 讲者设问负载形状。代码长输入、长推理输出、摘要和多轮工具调用的输入输出分布不同；服务优化必须说明负载。
- [00:27:29–00:29:09](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=1649s) · 学生问卸载适合什么负载。GPU/CPU/SSD 的容量、等待与复用权衡类似内存调度；LRU 是启发式，口述“2 倍最优”没有核实，不纳入教材。
- [00:53:56–00:55:37](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3236s) · 讲者设问参数与数据联合扩展；此处依赖特定实验曲线，不能由趋势图直接推导所有规模的最优配比。
- [00:55:29–00:57:04](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3329s) · 循环模型的初步 scaling 比较在固定参数、数据与 FLOPs 约束下变化 recurrence；需说明各条曲线的约束，不把多循环等同于无成本提升。
- [00:56:54–00:58:40](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3414s) · 学生问三者联合 scaling。讲者描述另有较难显示的三维图；当前页面未显示该图，资料不补造图上数字或联合最优解。
- [01:00:31–01:02:10](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3631s) · 学生问能否直接循环预训练模型。讲者认为有个案并在研究原因，未给普遍保证；推理好处还取决于权重能否越过显存/片上容量阈值。
- [01:03:07–01:04:45](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3787s) · 学生问 Megakernel 取舍。手写适配与维护昂贵，可能依赖模型、设备和 batch；“不可能更快”的口述强调不当成全局最优证明。
- [01:04:35–01:06:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3875s) · 学生问架构—硬件协同。需同时算权重、KV 和临时空间；知道部署设备可影响尺寸与量化选择，但讲者推测的厂商动机不当内部事实。
- [01:05:41–01:06:59](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3941s) · 学生问循环是否计算最优。答案必须先固定预算与目标；硬件量化格式的选择也受部署设备约束。字幕具体格式支持与硬件命名需独立规格核实。
- [01:07:50–01:10:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=4070s) · 学生问 agent 与批处理的架构区别。多轮任务重视缓存复用和热缓存，单次编码可偏向双向模型；批处理也可能生成长输出，不能把所有 batch 任务等同于 BERT 编码。
- [01:09:43–01:11:24](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=4183s) · 学生问多 GPU 通信能否融合。讲者描述早期尝试及部分 MoE 算子的融合；延迟仍可能受通信约束，不保证整模型单 kernel 更快。

字幕全文校读补充与纠错：

- [00:00:00–00:05:40](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=0s) · 开场模型规模、历史类比和产业预测属于演讲背景或讲者判断。未公开的 frontier 参数量不是已知事实；陪学不需要背诵这些猜测，也不据字幕确认并购等时事。
- [00:12:30–00:20:30](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=750s) · 服务负载应包含输入/输出长度、会话间隔和首 token/后续 token 的时延要求。prefill 常偏计算、decode 常偏带宽，是工作负载相关判断；大 batch 或特殊架构会改变瓶颈，不能当绝对分类。
- [00:17:30–00:19:30](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=1050s) · 跨请求 prefix cache 复用需要相同前缀和兼容的模型/位置等条件。相同一句问候若前面系统提示不同，不自动意味着 KV 相同；命中的是 token 前缀及相应状态，而非所有文本相似请求。
- [00:22:30–00:29:20](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=1350s) · 重复 token、超长 tool loop、语言切换三个例子展示系统 bug 如何表现为模型行为。不能凭症状就认定同一根因；未初始化读等需具体代码证据。缓存可逐级放在 GPU、CPU、SSD，预取与淘汰需计搬运时间。
- [00:31:10–00:33:40](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=1870s) · cache-aware P/D disaggregation 将低命中冷请求与高命中续聊路由到不同 prefill 池；讲者报告的最高约 40% 改善受原实验负载制约，不是任意请求下保证收益。
- [00:34:50–00:40:50](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=2090s) · Megakernel 以依赖粒度调度多个操作，既减 launch gap 也允许提前加载。QKV 未完成时可开始读旧 KV，attention 未完成时可读 O 权重；这不允许使用尚未产生的新 Q 做完整 attention。72% 是该实验的带宽利用率，不是所有模型吞吐统一上限。
- [00:42:30–00:52:00](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=2550s) · looped transformer 用同一组参数增加重复计算。线性化分析将复杂非线性当扰动；谱半径不是矩阵范数的同义词。连续时间负对角 A 经适当离散化得到收缩的 A_bar；不能把任意负数反复幂都趋零，D08 已给反例。
- [00:52:30–01:00:20](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3150s) · 实验中的训练稳定、验证损失和任务分数是不同证据。固定参数比较可允许 FLOPs 增长，固定 FLOPs 的曲线则在数据和循环次数之间分配；两张图的控制变量不能混在一起。
- [01:02:00–01:07:35](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3720s) · 小参数量可释放 KV cache 或减少跨 GPU 分片，但重复计算仍有成本。把循环块直接塞进更小 Megakernel 是讲者尚未实现的愿景，不能写成已发布演示；训练大小还需考虑将来谁能部署。

问答处理：关于直接循环预训练模型是否普遍有效，讲者仍表示研究中。资料保留为开放问题。

演示：画出算子依次启动与可重叠调度的时间条；再用标量迭代 x←a·x 展示 |a| 小于、等于或大于 1 的区别。

理解检查：减少参数但增加循环次数，为什么可能减少显存，却仍增加计算和延迟？

缺口：独立本课讲义未取得；已用完整字幕、指定视频帧、作者论文和代码补充。性能图及其他画面没有全面逐帧核验，未核实的精确数值不纳入结论。

## 二、2025 年逐讲补充：保留年度差异

本节不是把 2025 课程简单删掉。17 份字幕均已建立分段导航；下列条目是与主讲义合并时保留的指定片段和教学差异。已读完全部 17 份字幕文字，并保留下面的时间段补充；听不清的发言和未显示画面仍不猜补。

### 2025 L01 · 课程与分词

来源：[2025-01](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_01.py) · [00:19:41–00:20:55](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=1181s)。

**课程完成后的能力边界**：问答明确表示这门课是理解和构建模型的第一步，不是从课程毕业即可掌握前沿模型团队的所有能力。分词与预算意识仍可沿用。

例子／演示／合并处理：课堂分词编码示例；与 2026 L01 合并，不重复要求做 A1。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:01:01–00:02:38](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=61s) · 开场介绍与课程动机，非技术问答。2025 开始公开视频，不能据此假设 2024 也有完整公开录像。
- [00:19:09–00:21:04](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=1149s) · 学生问毕业能否带队训练前沿模型。讲者明确课程只是第一步，尚缺许多环节；学习目标不承诺工业团队全部能力。
- [00:26:02–00:27:46](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=1562s) · 学生问学分，属历史教务。之后课程总览强调给定硬件、数据与时间下的效率决策。
- [00:44:10–00:45:54](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=2650s) · 学生问作业链接为何仍为 2024。讲者说明 2025 会调整；资料必须分别锁定三年作业版本，不能把旧链接当本年内容。
- [00:54:50–00:56:32](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=3290s) · 学生问作业难度。属于当年负担说明；本陪学保留知识，按用户要求不强制完成大作业。字幕中的 GRPO 全称误称 preference，采用 group relative policy optimization。
- [00:58:50–01:00:31](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=3530s) · 学生问 Slack、旁听与评分。属教务，不转成当前获取权限或课堂内容。历史 Canvas 访问说明不代表现在可访问未公开嘉宾课。
- [01:02:30–01:04:00](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=3750s) · 学生问空格为何在 token 前。具体预分词规则会把前导空格与词合在片段，不是 BPE 定义必然要求；数字切分也不必按千位。

字幕全文校读补充与纠错：

- [00:05:00–00:11:40](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=300s) · 旧版用MLP/attention算力占比和突变准确率说明小规模结果未必代表大规模；同样不能把accuracy阶跃直接当底层能力已不连续。Bitter Lesson强调适合扩展的算法；效率×资源只是直觉。
- [00:17:30–00:26:40](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=1050s) · 可执行讲义是可步进的Python及缓存展示，课堂题、集群额度和AI规则属于2025学分背景。本资料保留概念、跳过强制重作业，不假定初学者已具备课上默认先修知识。
- [00:47:30–00:50:00](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=2850s) · 随机Common Crawl文档展示确有渲染异常，不能说每个文档都已清晰读出；垃圾样本用于说明清洗必要性，不代表整个web内容比例的随机统计结论。
- [01:00:30–01:17:30](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=3630s) · byte取值0到255，256是词表大小；Unicode字符可能多字节，不能把字节/字符压缩率混用。BPE新token编号256不等于此时总词表256，已是257项；依序回放merge、特殊token和预分词边界仍需单独处理。

### 2025 L02 · 张量与资源核算

来源：[2025-02](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_02.py) · [00:13:42–00:14:55](https://www.youtube.com/watch?v=msHyYioAyNE&t=822s)。

**混合精度中的持久状态**：课堂用 FP32 的长期累积状态与低精度的暂态计算区分混合精度。它是当年示意配方；现在有不同训练方式，不能教成所有参数永远必须以 FP32 存储。

例子／演示／合并处理：保留 storage、stride、view、contiguous、数据加载、随机种子和 checkpoint 代码例子。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:13:23–00:15:07](https://www.youtube.com/watch?v=msHyYioAyNE&t=803s) · 学生问混合精度。此处 FP32 持久参数/状态、BF16 临时计算是当年示意；与 2026 配方差异保留，不能教成唯一必要做法。
- [00:26:55–00:27:45](https://www.youtube.com/watch?v=msHyYioAyNE&t=1615s) · 学生问形状注解能否强制。字符串注解本身不执行检查，可以另用运行时工具或断言；注解与检查分开。
- [00:29:01–00:30:50](https://www.youtube.com/watch?v=msHyYioAyNE&t=1741s) · 学生问 einsum/编译是否保证高效。讲者给乐观简答；资料按实际优化问题处理，不把任何表达式的编译结果当最优保证。
- [00:36:50–00:37:40](https://www.youtube.com/watch?v=msHyYioAyNE&t=2210s) · 学生问峰值表中的稀疏。这里指受硬件支持的结构稀疏（例如 2:4），不是任意零值或 MoE 激活稀疏都获得相同倍数。
- [00:37:53–00:39:36](https://www.youtube.com/watch?v=msHyYioAyNE&t=2273s) · 普通矩阵乘法约 2BDK FLOPs，是计数推导，不是现场性能结果；读写与元素操作另计。
- [00:40:20–00:42:10](https://www.youtube.com/watch?v=msHyYioAyNE&t=2420s) · 学生问 FLOPs 是否随 matmul 实现变化、加乘是否等价。课程约 2BDK 是普通算法逻辑计数；实际硬件可融合指令，计数单位不等于延迟。
- [00:48:32–00:50:23](https://www.youtube.com/watch?v=msHyYioAyNE&t=2912s) · 学生问 Tensor Core。峰值需匹配数据类型；实际是否使用专门指令受形状、类型与库实现影响，不能由用了 PyTorch 就保证峰值利用。
- [00:57:18–00:58:56](https://www.youtube.com/watch?v=msHyYioAyNE&t=3438s) · 学生问是否手写反向。作业使用 autograd；手工推导用于理解和 FLOPs 核算。6ND 是主导线性层近似，Attention 等额外计算不能忽略所有场景。
- [01:13:18–01:15:04](https://www.youtube.com/watch?v=msHyYioAyNE&t=4398s) · 学生问是否必须保存激活。反向需要对应前向信息，可保存也可重算；训练 checkpoint 文件与激活 checkpoint 要分别命名。

字幕全文校读补充与纠错：

- [00:00:00–00:03:30](https://www.youtube.com/watch?v=msHyYioAyNE&t=0s) · 144天是6ND、给定硬件峰值和0.5MFU下的预算估计。八卡总HBM除16bytes给40B只是总容量账本，隐含能切片/分配状态，朴素每卡完整复制不能训练40B；还没算激活和临时峰值。
- [00:05:00–00:13:00](https://www.youtube.com/watch?v=msHyYioAyNE&t=300s) · dtype表示动态范围与精度的不同交换，BF16保留FP32级指数范围但尾数更短。讲者“参数/优化器必须全FP32”是保守配方，不是所有现代低精度方案的不可能性证明；训练与推理量化边界不同。
- [00:17:30–00:25:30](https://www.youtube.com/watch?v=msHyYioAyNE&t=1050s) · tensor metadata还应有storage offset；view可共享存储，reshape是否复制取决于布局，非连续也并非所有view都绝不可行。逐元素操作可有原地/out版本，不能说任何逐元素操作必分新storage。
- [00:32:30–00:47:30](https://www.youtube.com/watch?v=msHyYioAyNE&t=1950s) · FLOPs数量和FLOP/s速率分开，峰值要匹配dtype/稀疏模式。原课“两周”实际一周已当场指出；MFU对模型有用工作计数，HFU可能含重算，两者都不能仅看速度快慢评价。
- [00:50:00–01:12:40](https://www.youtube.com/watch?v=msHyYioAyNE&t=3000s) · 反向包括参数梯度和输入激活梯度两条矩阵路径；原始输入不需梯度时可省一路。AdaGrad累积梯度平方而非普通均值；状态跨step保留，zero_grad只清梯度不清累计优化器状态。
- [01:15:00–01:18:20](https://www.youtube.com/watch?v=msHyYioAyNE&t=4500s) · 恢复训练需模型、优化器、step，若要重现还需随机数、数据位置、日程及混合精度状态。固定seed有助调试，不保证跨硬件/并行算法逐位一致。

### 2025 L03 · 结构与超参数

来源：[2025-03](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [00:18:22–00:19:40](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=1102s)。

**比具体模型配方更可迁移的原则**：课堂强调直接残差路径、控制激活尺度，以及结构对系统的影响。这比背某个激活函数名更适合初学者建立判断。

例子／演示／合并处理：串行与并行 block 的比较、RoPE 的旋转解释，与新版对应章节合并。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:10:15–00:11:30](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=615s) · 学生问 residual 中放 LayerNorm 为何有问题。讲者给直接恒等梯度路径的直觉，明确非证明；不能据此禁止所有 post-norm 架构。
- [00:18:05–00:19:47](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=1085s) · 学生问哪些结论更长期有效。直接残差路径、控制激活尺度、结构与系统成本联动，比背固定架构配方更可迁移。
- [00:23:47–00:25:28](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=1427s) · 学生问非单调激活的负导数是否妨碍优化。讲者给经验直觉，并非全局收敛证明；门控 FFN 的第三矩阵也要计参数，常缩小中间宽度补偿。
- [00:26:16–00:27:57](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=1576s) · 激活函数的额外算术可能不主导受带宽限制的算子；论文小模型结果需保留任务、年份与不确定性，不直接推广。
- [00:31:44–00:33:22](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=1904s) · 学生问串行是否更高效。并行 block 便于共享/融合一些计算，串行增加函数组合；实际质量与速度仍需测量。
- [00:39:30–00:40:50](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=2370s) · 学生问 RoPE 角速度、是否可训练和反向难度。所讲方案使用预先确定频率，不学习每个角度；频率日程/基数是设计选择，字幕的“不是超参数”不能扩大成不可调。
- [00:47:04–00:48:45](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=2824s) · 学生问 FFN 比例。宽矩阵利于硬件并行，但参数在宽度、深度与表示能力间的分配不同；64 倍之类个例不是通用推荐。
- [01:01:23–01:03:02](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=3683s) · 学生问 dropout 和多语言词表。讲者未给 dropout 的完整理论；更大多语言词表可减少低资源语言 token 膨胀，但仍有词表参数与训练取舍。
- [01:03:00–01:04:39](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=3780s) · 学生问 weight decay 与多模态。weight decay 可改善训练动态；接上现成 LM 的视觉模型沿用其架构，不证明所有原生多模态模型都相同。
- [01:14:36–01:16:09](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=4476s) · 学生问推理时是否保留 normalization。它是前向函数一部分，不能训练后随意删去。是否有仿射参数取决于具体归一化层，不能用“必有参数”作唯一理由。

字幕全文校读补充与纠错：

- [00:02:30–00:12:30](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=150s) · 字幕emit bias应按主线理解为omit bias；pre-norm稳定性对比也有口头把pre/post重复说反。教学以残差路径图及公式为准，区分子模块后规范化和残差相加后规范化。
- [00:12:30–00:31:40](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=750s) · RMSNorm、无bias、GLU和parallel block都是经验有效的设计，并非模型成功的必要条件。占FLOPs很少的归一化仍可因访存占不少时间；0.17%/25%是所引实验，非所有实现固定比例。
- [00:33:20–00:55:20](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=2000s) · RoPE相对位置性质对给定Q/K旋转成立，不消除上下文hidden state对位置的全部依赖。4D/8D÷3、heads×head_dim≈model_dim、宽深比约100都为默认经验，T5等例外保留，不能变成自然定律。
- [00:56:40–01:15:30](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=3400s) · 单遍训练不保证不记忆。weight decay在所引实验改善优化过程；z-loss、QK norm和softcap分别干预不同数值项，不能推出规范化必改善所有任务或无需稳定softmax。
- [01:16:10–01:26:58](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=4570s) · 推理算术强度公式需按讲义变量和整体倒数核对，字幕若漏平方/下标不能直接照写。SWA的层间感受野扩大不等于直接完整全局访问；全局NoPE仍有因果mask等位置线索，长上下文泛化并无自动保证。

### 2025 L04 · MoE

来源：[2025-04](https://github.com/stanford-cs336/spring2025-lectures/blob/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [00:06:22–00:08:10](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=382s)。

**激活参数不是全部成本**：问答指出，图上只画激活参数可能使比较显得更有利。需同时检查总参数、显存、设备间路由和真实 wall-clock。

例子／演示／合并处理：用 top-k 加权和专家负载图解释稀疏计算；保留路由前后 softmax 的不同设计。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:05:52–00:07:29](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=352s) · 学生问忽略 FLOPs 的 MoE 开销。路由与通信可能耗时；激活参数坐标轴不包含全部存储和部署成本。
- [00:07:50–00:09:00](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=470s) · 讲者确认没有学生问题，随后讲专家并行。排除问答误报，保留为架构到系统的连接说明。
- [00:17:35–00:19:05](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1055s) · 学生问 router 读 token 还是位置，以及 expert choice 的均衡。输入是已含上下文/位置影响的 hidden state；按 token 选专家与按专家选 token 在同一得分矩阵上沿不同轴取 top-k。
- [00:19:02–00:20:38](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1142s) · 学生问 router 与 k。每 token 的简单得分选择 top-k；k 是超参数，探索直觉只是解释之一，不能把 k≥2 当数学要求。
- [00:20:32–00:22:07](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1232s) · 学生问多专家输出合并。可能按路由权重求和，也有不同归一化；采用具体公式，避免字幕把 k>1 说成 k=1 的错误。
- [00:26:11–00:27:48](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1571s) · 学生问 softmax 与 top-k 为什么都需要。softmax 产生相对权重，top-k 限制激活计算；移除 top-k 会失去该稀疏计算收益。
- [00:27:40–00:28:50](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1660s) · 学生问 top-k 后权重和、router 向量与 FFN。权重和可小于 1，是否再归一化看架构；路由参数与 FFN 权重通常分别学习。
- [00:28:40–00:30:15](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1720s) · 学生问随机采样与 top-k 后重归一化。采样可能探索但有训练/推理不一致；重归一化是具体架构选择，不代表优化动态完全无差异。
- [00:30:40–00:32:19](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1840s) · 学生问为何 router 简单。更复杂路由增加成本，稀疏反馈仍限制学习信号；增加参数不保证学到最优路由。
- [00:35:52–00:37:29](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2152s) · 论文案例中细粒度专家收益较一致，共享专家收益随设置不同。不得把特定 DeepSeek 消融写成所有 MoE 的定律。
- [00:39:24–00:41:01](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2364s) · 学生问专家比例及下投影。总参数、单专家宽度、激活专家数分别算账；极窄专家可能中间维度小于模型维度。讲者未确认的具体模型配置不采用。
- [00:40:50–00:42:00](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2450s) · 学生问为何多个 shared experts。讲者未给普遍收益，某些消融甚至一个也不显著；保持同尺寸方便是设计理由，不是性能定律。
- [00:46:22–00:48:03](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2782s) · 学生问随机抖动如何解决不可导。它提供看到其他专家的机会，但不是将 top-k 变为普通可导函数；辅助负载损失是另一条训练机制。
- [00:54:30–00:56:55](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=3270s) · 学生问不考虑系统时还需负载均衡吗。专家塌缩也会减少有效容量，均衡不仅为设备利用率；device 在该图指加速器，更大通信域是另一层约束。
- [00:56:49–00:58:34](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=3409s) · 学生问哈希路由和 GPU 数量。固定哈希与每次独立随机路由不同；专家可共享设备，不能按层数×专家数直接推导所需 GPU 数。讲者对纯随机效果只是猜测。
- [01:16:10–01:17:20](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=4570s) · 学生问 DeepSeek-V3 是否仍限制目标设备数。课堂按该版报告回答保留 top-M；是模型版本事实，不推广到全部后续版本。

字幕全文校读补充与纠错：

- [00:00:00–00:16:40](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=0s) · 2025本讲专讲MoE，2026同编号加入其他架构；年度编号不能直接当相同范围。稀疏活跃参数可降低核心FLOPs，但总显存、路由与通信仍计入；GPT-4架构传闻不当官方确认。
- [00:22:30–00:32:20](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1350s) · 加权专家输出是确定的稀疏加权组合，不必理解为对随机专家的数学期望。softmax前后top-k及门权重归一化不同，正如课堂问答所指，未归一化门权重和可能小于1。
- [00:32:30–00:45:00](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1950s) · fine-grained缩小FFN中间宽度，可在相近总/活跃账本下组合更多专家；shared expert每token都会用，不能漏进active count。表中专家大小是讲者从配置反推的近似，不作为精确规格。
- [00:48:10–00:55:50](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2890s) · Switch平衡项f·P里f为离散分配比例，P为连续router概率均值，梯度主要经P。DeepSeek偏置只影响选择，实际门权重不直接加偏置；“loss-free”还保留小的序列级辅助损失。
- [01:00:00–01:10:00](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=3600s) · MegaBlocks块稀疏打包不等于规格表2:4稀疏峰值。容量丢token使输出依batch同伴变化，是非确定性的一种机制，原课未声称证实GPT-4因此不确定。upcycling复制dense初值后仍须训练。
- [01:12:30–01:21:30](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=4350s) · device-limited routing先限制设备候选再选总top-k，不是每个设备都无条件选k个。MLA通过结合投影减少缓存/重复上投影，RoPE独立部分仍需处理；MTP额外一步指比普通next-token再多预测一步，而非总共只有一个token目标。

### 2025 L05 · GPU

来源：[2025-05](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [00:10:11–00:12:05](https://www.youtube.com/watch?v=6OBtO9niT00&t=611s)。

**物理距离与内存层级**：课堂把 GPU 示意图对应到实际 SM 与内存，强调“会计算”只解释性能的一部分。临近处理单元的内存通常更快、更小，组织计算时需考虑位置。

例子／演示／合并处理：保留工厂、sin²+cos² 融合、sigmoid 重计算、tile 与在线 softmax 例子。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:09:47–00:11:30](https://www.youtube.com/watch?v=6OBtO9niT00&t=587s) · 学生对应卡通图与真实 GPU。SM、控制单元与计算单元是不同层级；内存位置同样决定性能。
- [00:18:25–00:20:04](https://www.youtube.com/watch?v=6OBtO9niT00&t=1105s) · 学生问 Tensor 的名称是否意味任意算子。可对张量索引形成矩阵块，但专用乘法单元不是任意张量函数处理器。
- [00:28:35–00:29:50](https://www.youtube.com/watch?v=6OBtO9niT00&t=1715s) · 分支图示说明同一 warp 内路径分歧的等待。不能简化成 GPU 代码绝对不能有条件判断，边界 mask 就是必要条件逻辑。
- [00:35:53–00:37:11](https://www.youtube.com/watch?v=6OBtO9niT00&t=2153s) · sin²+cos² 示例对比多个 kernel 与融合 kernel。融合减少中间 HBM 读写，不必把数据来回搬到 CPU；没有把数学恒等式简化与编译融合混为同一件事。
- [00:51:07–00:52:49](https://www.youtube.com/watch?v=6OBtO9niT00&t=3067s) · 学生指出 tile 示意中重复加载。讲者说明已在共享内存的块可复用，图上列步骤不代表必须丢弃再读。
- [00:54:31–00:56:11](https://www.youtube.com/watch?v=6OBtO9niT00&t=3271s) · 学生问预取。能否重叠读取与计算取决于缓冲和资源；共享内存占满会限制双缓冲，不能理解成硬件自动免费预取。
- [01:09:16–01:10:37](https://www.youtube.com/watch?v=6OBtO9niT00&t=4156s) · 在线 softmax 保留运行最大值、归一化和加权值统计量；FlashAttention 要的是 softmax 加权 V 的聚合，不需要输出完整概率矩阵。
- [01:10:47–01:12:25](https://www.youtube.com/watch?v=6OBtO9niT00&t=4247s) · 学生问是否必须回头再看每个 tile。最终分母在遍历后确定，但注意力输出可同时维护并重缩放累积量；不能由口述推出必须存全 N² 元素。若要逐项输出完整 softmax，需求不同。

字幕全文校读补充与纠错：

- [00:05:00–00:17:30](https://www.youtube.com/watch?v=6OBtO9niT00&t=300s) · CPU低延迟、GPU高吞吐是设计取向。SM/warp/block与TPU TensorCore不是同一层级名词；A100开头128SM口述与后面108例不一致，不作型号规格结论。local memory的逻辑私有不保证物理在SM快存储。
- [00:25:00–00:33:20](https://www.youtube.com/watch?v=6OBtO9niT00&t=1500s) · roofline按算术强度而非仅矩阵大小解释上界。低精度可同时减少字节和提高特定硬件吞吐，但指数、归约、累加与状态需独立选精度；不能保证所有输入减半bit就恰快两倍。
- [00:33:20–00:40:50](https://www.youtube.com/watch?v=6OBtO9niT00&t=2000s) · 融合减少中间写回和launch开销，重算以额外算术换访存，三sigmoid示意从8降5是简化读写账本，不是所有真实HBM事务或FLOPs总量相同。
- [00:42:30–01:04:20](https://www.youtube.com/watch?v=6OBtO9niT00&t=2550s) · coalescing看同一warp同一时刻不同线程地址，不能只看单线程沿行/沿列。tile尾部、对齐与wave是不同离散效应；引用nanoGPT词表精确数在字幕漏位，教学只保留padding到合适倍数的原则，不照错数。
- [01:05:00–01:14:00](https://www.youtube.com/watch?v=6OBtO9niT00&t=3900s) · 注意力核心是QKᵀ与softmax(QKᵀ)V两次矩阵乘，投影另计，口述“三次”不混入核心。在线softmax维护max和归一化量；FlashAttention还要同步重缩放加权V累积量，反向重算未保存的二次中间矩阵，D10补足数值验证。

### 2025 L06 · Kernel 与 Triton

来源：[2025-06](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_06.py) · [00:59:17–01:01:10](https://www.youtube.com/watch?v=E8Mju53VB00&t=3557s)。

**连续内存与转置视图**：课堂简单 kernel 假定连续布局；transpose／view 可能改变 stride。可以在外层整理布局或实现通用寻址，但这两条路有不同成本，不应把连续性断言当数学要求。

例子／演示／合并处理：保留 CUDA 手写 GELU、torch.compile、profiler 与 Triton softmax 的比较；未运行 GPU 性能测量。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:06:05–00:07:05](https://www.youtube.com/watch?v=E8Mju53VB00&t=365s) · 学生问 warp 的作用。成组执行共享控制开销，增加算术资源占比；这是 GPU/CPU 设计取舍的解释。
- [00:19:10–00:20:28](https://www.youtube.com/watch?v=E8Mju53VB00&t=1150s) · Profiler 从 sleep 到加法例子说明 CPU 与 GPU 事件表；同步等待和真正 kernel 工作要分开。
- [00:23:37–00:25:12](https://www.youtube.com/watch?v=E8Mju53VB00&t=1417s) · 学生问 CPU/CUDA 时间、matmul CPU 时间下降、profiler 开销。讲者对下降原因不知道；采样和同步有开销，微秒比较需额外谨慎。
- [00:25:00–00:26:40](https://www.youtube.com/watch?v=E8Mju53VB00&t=1500s) · 学生问 CPU 百分比。表内事件耗时占比不是整机 CPU 核心利用率；不同形状可派发到不同 matmul kernel。
- [00:30:15–00:31:56](https://www.youtube.com/watch?v=E8Mju53VB00&t=1815s) · 学生问 CPU 在 GPU 计算时做什么。讲者转到时间线解释；GELU 近似公式依实现，字幕随口说的公式不直接采用。
- [00:33:31–00:34:58](https://www.youtube.com/watch?v=E8Mju53VB00&t=2011s) · Nsight 时间线和 NVTX 标注演示，用于对应 Python 层、CPU 提交与 GPU 执行；文字过小不能据此抄精确微秒数。
- [00:37:44–00:39:10](https://www.youtube.com/watch?v=E8Mju53VB00&t=2264s) · CPU 可预先排入多个 GPU kernel，因此 CPU 上的 layer 编号可能领先实际 GPU 执行；排队和计算时间不是同一统计量。
- [00:49:42–00:51:19](https://www.youtube.com/watch?v=E8Mju53VB00&t=2982s) · CUDA 例子分 CPU wrapper 与 GPU kernel；线程按 block/thread ID 找元素，越界线程需屏蔽。调试同步选项会改变运行速度。
- [00:55:45–00:57:24](https://www.youtube.com/watch?v=E8Mju53VB00&t=3345s) · 学生问 .cu 文件。后缀区分 CUDA 源码；编译后从 Python 调用不意味着 GPU 直接执行 Python。
- [00:58:53–01:00:40](https://www.youtube.com/watch?v=E8Mju53VB00&t=3533s) · 学生问 non-contiguous 与 block size。此示例用断言限制连续布局；transpose/view 常改变 stride，需整理布局或写通用寻址。不同 block size 要实测资源占用。
- [01:00:51–01:02:32](https://www.youtube.com/watch?v=E8Mju53VB00&t=3651s) · 学生问朴素实现为何慢。主要是 GPU HBM 与 SM 之间的反复搬运及 kernel 启动，不是每次算术都往返 CPU。
- [01:13:39–01:15:20](https://www.youtube.com/watch?v=E8Mju53VB00&t=4419s) · 学生问何时手写可胜过 torch.compile。简单融合先试编译器，复杂算法或硬件专用调度可能需专门 kernel；没有“手写必快”的保证。

字幕全文校读补充与纠错：

- [00:02:30–00:07:05](https://www.youtube.com/watch?v=E8Mju53VB00&t=150s) · 32线程一组是warp，wave是多个block按驻留资源分批完成的性能概念；原课此处混说wave需纠正。block内线程可共享内存/同步，跨block特殊机制另论，不把教学简化当所有CUDA绝无跨block协作。
- [00:10:00–00:18:40](https://www.youtube.com/watch?v=E8Mju53VB00&t=600s) · 基准预热测稳态，若关心冷启动须另报首次开销；cuda.synchronize主要让主机等待设备工作，不是CPU和GPU都停止成某统一状态。口述秒/毫秒混用，以trace原数和单位为准。
- [00:20:30–00:43:20](https://www.youtube.com/watch?v=E8Mju53VB00&t=1230s) · profiler的CPU self time、GPU kernel time、重叠和端到端时间不同，不能直接加总。打印CUDA标量会同步而普通CPU字符串未必；“CPU永远不是瓶颈”过强，Python调度、tokenization/数据加载都可能成为瓶颈。
- [00:45:00–00:59:10](https://www.youtube.com/watch?v=E8Mju53VB00&t=2700s) · GELU统一tanh近似和allclose后再比速度，公式根号是sqrt(2/pi)，字幕sqrt(pi/2)错误。缓存记录里内置约1.1ms，手CUDA/Triton约1.8ms，不能说手写一定更快。
- [01:02:30–01:13:40](https://www.youtube.com/watch?v=E8Mju53VB00&t=3750s) · Triton逻辑BLOCK_SIZE不等于线程数，一个线程可管多元素；向量化加载四值和整个warp coalescing相关但不同，不是DRAM永远四字节burst。PTX仍需编译到机器指令；exp2换底用log2(e)。
- [01:15:20–01:20:05](https://www.youtube.com/watch?v=E8Mju53VB00&t=4520s) · softmax每program一行仅适合行能驻留的场景，mask无效lane应填负无穷；全mask行需边界处理。compile在此softmax结果较好、在此GELU仍慢于内置，保留具体形状/版本差异，不推广为统一排名。

### 2025 L07 · 并行一

来源：[2025-07](https://github.com/stanford-cs336/spring2025-lectures/blob/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [00:23:23–00:25:20](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1403s)。

**通信量的统计口径**：课堂追问通信量是每设备还是全系统。这段回答与后文不同公式存在口径混淆风险，教学资料不直接搬一个倍数。每次都说明发送／接收、单卡／总和、轮次与算法。

例子／演示／合并处理：保留 ZeRO 分阶段示意、激活内存与 sequence parallel 的解释，补充缺字幕的 2026 L08。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:11:30–00:12:40](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=690s) · 学生问 all-reduce 分解是否更好。数学结果可等价，带宽限制下流量可相同；价值是两阶段中间可以插入优化器更新，不保证任意 API 分解都无延迟损失。
- [00:19:32–00:21:09](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1172s) · 学生问数据并行如何能切分优化器状态。各设备已有参数与归约梯度，但仅需更新自己负责的参数块，再收集更新参数。
- [00:22:55–00:24:37](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1375s) · 学生追问通信量单位与 Adam 的逐元素状态。课堂通信口径有混淆，教材显式采用每 rank 发送量与算法步数；逐元素预条件并不表示神经网络函数的参数相互独立。
- [00:24:52–00:26:32](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1492s) · 学生问图中行与 reduce-scatter。按参数块跨数据分片归约，将结果放到负责该块的 rank；ZeRO-1 的通信收益带 bandwidth-limited 条件。
- [00:26:19–00:28:01](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1579s) · 学生问分片后是否可存更高阶状态。状态仍占资源，其他部分也会分片，不能把节约视为无上限免费内存。
- [00:37:36–00:39:17](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=2256s) · 学生问预取权重放哪里。需要额外缓冲，图未完整计入激活与峰值临时内存；分片公式不等于真实峰值显存。
- [00:40:21–00:42:02](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=2421s) · 学生问 FSDP 与模型并行区别。FSDP 临时聚合权重供局部数据计算；张量/流水线并行主要切分计算并传递激活，通信对象不同。
- [00:42:02–00:43:39](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=2522s) · 学生问 broadcast 是否足够。若整个层只由一个 rank 持有可广播；一般各 rank 持有不同分片需 all-gather，不能用特殊图示替换一般操作。
- [01:01:59–01:03:38](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=3719s) · 学生问 TP 与 PP 是否可组合。可在快网络域做 TP、域间做 PP/DP；最终选择还受模型能否装下和 batch 规模影响。
- [01:10:15–01:11:57](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=4215s) · 学生问复杂计算图。讲者未充分研究；TP 常逐层处理，PP 的依赖可能更复杂。后续 context/ring attention 只是概览，未完整展开。

字幕全文校读补充与纠错：

- [00:05:00–00:12:30](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=300s) · all-reduce和reduce-scatter+all-gather的通信量等价基于大消息等假设；256GPU、8GPU是所引拓扑设置，不是通用阈值。TPU只能直接连邻居不表示不能经路由与远端交换数据。
- [00:15:00–00:21:30](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=900s) · 16byte状态账本需列精度，120→31.4→16.6→1.9GB表示该示例每GPU状态，不是64卡总内存。Adam第二矩是梯度平方均值，不是已减均值的严格方差。
- [00:28:00–00:40:20](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1680s) · ZeRO-1/2确实省优化器/梯度显存，课堂结尾“不能scale memory”指仍有完整参数瓶颈，不等于零收益。ZeRO-3持久切片仍需临时完整层和激活，2×/3×通信是简化账本，重叠不保证任意网络下免费。
- [00:47:30–01:01:40](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=2850s) · PP bubble/有效计算≈(p−1)/m，和bubble/总时间不同；零bubble拆输入梯度和权重梯度，W仍必须更新前完成。TP前向/反向collective对偶，all-reduce不是MoE的all-to-all。
- [01:03:40–01:12:30](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=3820s) · 34SBH激活公式来自特定实现/精度，不是所有训练最低限。Megatron sequence parallel切逐token部分，ring/context parallel切长注意力序列；两者区别保留。重算可释放batch资源，但不必然缩短端到端时间。
- [01:15:00–01:23:20](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=4500s) · 先fit内存再DP的组合是经验起点，需测吞吐、优化效率和故障恢复。Llama3的148是其报告里归因faulty GPU的中断，不是所有年份硬件平均故障率；上个2026口述也不能泛化成模型总共仅失败148次。

### 2025 L08 · 并行二

来源：[2025-08](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_08.py) · [00:12:10–00:13:15](https://www.youtube.com/watch?v=LHpr5ytssLo&t=730s)。

**代码、拓扑与运行输出**：课堂从设备连接转入 collective 代码，再到数据、张量、流水线并行。逻辑上等价的切分，仍会受实际拓扑和异步调度影响。

例子／演示／合并处理：保留 reduce-scatter 后 all-gather 的数值输出、microbatch 和显式 send／recv 代码。不是 2026 L08 的替代字幕。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:10:58–00:12:45](https://www.youtube.com/watch?v=LHpr5ytssLo&t=658s) · 学生问 PCIe 是否必经 CPU、NVLink 是否连 CPU。课堂回答简化；具体点对点 DMA 可绕开 CPU 数据拷贝，必须查看实际拓扑与支持，不能教成 PCIe 必定绕 CPU 内存。
- [00:22:10–00:23:53](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1330s) · 学生问 reduce-scatter 的索引对应。输入分块与目标 rank 的约定由调用形状和 API 决定；小例子先检查维度，不猜输出落点。
- [00:30:28–00:32:07](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1828s) · 学生问通信字节是否包括输入。示例假定输入已在设备上，仅计该 collective 的流量；加载输入与网络通信分开统计。
- [00:33:56–00:35:24](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2036s) · DDP 代码先按 rank 切 batch 并构建本地 MLP；等分假设应显式说明。
- [00:36:51–00:38:15](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2211s) · 各 rank 本地 loss 可不同，归约后梯度相同；学生问各进程如何保持同一步，需接下一段同步协议。
- [00:38:08–00:39:56](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2288s) · collective 顺序必须一致，否则可能等待或失败；初始参数相同，rank 参数只用于设备放置。不能把异步进程理解成任意调用顺序。
- [00:45:29–00:47:26](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2729s) · 学生问为何没写 TP 反向。讲者说时间限制，非算法不可能；此处只演示前向，不冒充完整训练实现。
- [00:50:01–00:51:46](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3001s) · 学生问是否事件驱动。示例显式等待固定来源，与任意事件触发 handler 不同；异步进程仍受严格数据依赖约束。
- [00:52:17–00:54:06](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3137s) · 学生问重叠通信计算及多消息匹配。异步 send 返回句柄，可做独立 microbatch，使用前仍须等待；还需管理缓冲生存期与消息对应。
- [00:54:17–00:56:05](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3257s) · 学生问同源多 send 和无人 recv。顺序与后端契约有关，讲者此处不完全确定；未匹配调用可能阻塞，不能当通用消息队列使用。
- [00:55:56–00:57:35](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3356s) · 学生问最后一 rank。它得到前向输出，反向须从损失沿依赖传回；本示例未完成这个反向链。
- [01:03:05–01:04:56](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3785s) · 学生问 BatchNorm。讲者没有当场答案；LayerNorm 按样本归一化的性质不同，不能据 DDP 参数同步推导所有运行统计也自动同步。
- [01:06:05–01:07:50](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3965s) · 学生问激活重算 API 和专用硬件。可选择计算片段重算；硬件路线是当年概览，不补为设备推荐。
- [01:09:02–01:10:40](https://www.youtube.com/watch?v=LHpr5ytssLo&t=4142s) · 学生问继续训练。并行原语用于梯度步骤，本身不要求从零初始化；加载已有 checkpoint 后还需恢复相应优化器与随机状态等训练状态。
- [01:12:22–01:14:04](https://www.youtube.com/watch?v=LHpr5ytssLo&t=4342s) · 学生问图在 CPU 还是 GPU、通信由谁执行。主机控制/图记录与设备上的张量、kernel 要分开，不能把全部计算图当成一块 GPU 张量。
- [01:13:59–01:15:06](https://www.youtube.com/watch?v=LHpr5ytssLo&t=4439s) · 通信 primitive 是抽象，实际可经 CPU 或 GPU 后端执行；CPU 发起调用与 GPU/NIC 数据路径不是同一件事。

字幕全文校读补充与纠错：

- [00:00:00–00:15:30](https://www.youtube.com/watch?v=LHpr5ytssLo&t=0s) · 本讲配合独立stdout演示，多进程不能在单个可执行讲义界面完整步进。Gloo可检逻辑而不是替代NCCL性能；rank通常映射一个GPU进程，在一般分布式程序也不必等同物理设备。
- [00:17:30–00:32:20](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1050s) · empty输出的初值无含义；all-reduce确实原地改张量，reduce-scatter/all-gather另分配输出。四rank结果6/10/14/18已由stdout核对。通信归一化和900GB/s双向规格不是同口径实测；讲者自己未解释清277对70差异，不能替他归因硬件加速。
- [00:32:30–00:49:40](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1950s) · DDP需相同初始化及按样本/token正确加权，示例只多一处梯度平均。TP/PP在本讲仅forward骨架，FSDP实现明确省略，不能声称课程已现场跑了完整TP/PP训练。
- [00:49:40–01:00:50](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2980s) · 同步send/recv例子没有通信计算重叠；用异步版本还需依赖、buffer寿命和顺序。JAX/Levanter声明式分片是另一抽象层，十行展示不代表底层通信消失。
- [01:02:30–01:11:40](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3750s) · 固定静态数据流有助编排，MoE路由、形状和现实任务也可能动态。更大片上内存可少搬运，但制造、容量、功耗及灵活性有取舍，不能据此推断某专用芯片普遍优于GPU。

### 2025 L09 · Scaling 一

来源：[2025-09](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [00:10:45–00:12:15](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=645s)。

**变大也可能有反例**：课堂讨论逆 scaling 和分布外行为。预训练 loss 下降并不保证每个任务、每种提示下都单调改善；必须区分平均指标与具体行为。

例子／演示／合并处理：保留均值估计、临界 batch 和 Chinchilla 方法比较。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:10:25–00:12:07](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=625s) · 学生问是否有不 scaling 或反向 scaling。平均预训练损失与特定分布外任务行为不同，不能由前者单调推出后者。
- [00:20:06–00:21:47](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=1206s) · 学生问合成具有内在维度的数据。可构造低维函数嵌入；从 CIFAR 等真实数据估计维度困难，示例并未证明文本真实维度。
- [00:22:13–00:24:01](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=1333s) · 讲者设问重复高质量与加入新数据。有效样本量与重复轮次有关；某研究四轮后的收益变化不是所有语料的固定阈值。
- [00:24:45–00:26:00](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=1485s) · 学生问数据 scaling 图是否固定模型。该特定图固定足够大的模型以避开容量饱和；计算 scaling 的图可能联合改变参数和数据，不能用同一解释。
- [00:42:30–00:43:20](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=2550s) · 学生问临界 batch 是否适用于多模态/视觉。讲者没确认，不填入跨领域统一结论。
- [00:43:02–00:44:35](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=2582s) · 学生问 noise scale。指批采样导致的梯度噪声尺度；论文示意与本机实测分开，下游指标的 scaling 通常比交叉熵更不规则。

字幕全文校读补充与纠错：

- [00:02:30–00:20:20](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=150s) · 理论泛化上界和经验loss曲线不同；均值估计标准差是sigma/sqrt(n)，MSE是sigma²/n，log-log斜率−1。课堂口述除n及斜率1有符号/量纲简化；非参数维度指数还依光滑度、风险定义。
- [00:27:30–00:35:10](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=1650s) · 具体架构/优化器曲线平行只是实验现象，不保证任意替换只改截距。宽深比100、FFN multiplier4是不同超参，口述4到16不能误挂到宽深比。
- [00:35:10–00:41:50](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=2110s) · 临界batch在目标loss降低时可增大，讲者一处说smaller后下一句更正bigger。相同最优学习率跨宽度转移需正确muP参数化，不保证随意改深度或数据仍成立。
- [00:47:30–00:58:00](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=2850s) · C≈6ND中不能漏6，IsoFLOP扫N时D应随预算约束变化。cosine中段checkpoint可用，但不等同已经按更短日程衰减完的模型；不是“中途模型无效”的绝对判断。
- [00:58:00–01:05:05](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=3480s) · 联合曲面拟合依损失函数和优化，残差均值为零只是特定回归条件，不是所有鲁棒/非线性拟合的必然条件。Chinchilla20比1是训练最优近似，服务总成本可能偏向更小模型多token训练。

### 2025 L10 · 推理

来源：[2025-10](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py) · [00:36:57–00:38:15](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=2217s)。

**请求到达与完成时间不同**：问答把不同用户和不同输出长度引入批处理问题。缓存、调度与模型数学成本必须一起看，静态 batch 估算不是完整线上服务模型。

例子／演示／合并处理：保留状态空间与 diffusion 的路线讨论，但不把课堂概览包装成已完整教授这些架构。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:01:37–00:03:00](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=97s) · 推理支撑部署、合成数据和 RL rollout。产品生成量为历史口述数字，不当当前事实或算力预算依据。
- [00:13:00–00:14:10](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=780s) · 学生问推理为何 batch 不大于 1。可以更大，后续会展开；batch=1 是分析起点，不是推理的必然限制。
- [00:36:26–00:38:09](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=2186s) · 学生问单 GPU 假设及不同用户完成时间。静态成本模型不能直接代替动态服务，continuous batching 要处理到达与退出。
- [00:50:50–00:52:38](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3050s) · 学生问跨层 KV 共享与长 prompt 压缩。是否共享投影权重要看架构公式；摘要或 gist 压缩可能丢信息，不是无损缓存优化。
- [01:02:08–01:03:33](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3728s) · 引用 diffusion 生成过程演示先产生并行粗输出再迭代修正；展示速度不证明跨任务质量、延迟或硬件条件公平。
- [01:21:49–01:22:46](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=4909s) · 讲者总结先明确质量与资源目标，再讨论架构与执行；换模型后的质量/成本比较需控制训练与评估条件。

字幕全文校读补充与纠错：

- [00:07:30–00:13:40](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=450s) · 口述前向6ND应纠正：常用前向约2ND，训练前反合计约6ND。H100带宽3.3是TB/s不是bytes/s；计算条件是BT远小于D、F，字幕BD为误差。SymPy能核代数，不能保证人工写入的建模假设正确。
- [00:27:40–00:36:40](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=1660s) · Llama2_13B的240GB batch256只是公式结果且原课说放不下；16/64口述不一致按trace配置。latency模型是理想下界，TTFT还包括排队等；多副本吞吐线性要求请求、资源及负载均衡足够。
- [00:38:10–00:51:40](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=2290s) · GQA/MQA/MLA/CLA/SWA各减少不同维度的KV存储。讲者MLA性能表展示错并明确承认，不能把画面当其质量结论直接证据；更换架构后需要训练或适配，不能只改head数就保证质量不降。
- [00:52:40–01:04:10](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3160s) · S4/Mamba/linear attention以固定状态换有限回忆能力，合成关联回忆用来暴露特定瓶颈。有限特征映射近似softmax核与直接改模型不同；diffusion多token并行仍需多轮去噪和质量/步数折中。
- [01:05:00–01:10:50](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3900s) · LLM.int8把outlier路径单独高精度处理；AWQ主要按激活做通道缩放，不是简单把所有显著权重保留FP16，依据原论文：https://arxiv.org/abs/2306.00978 。压缩会省容量，但转换/内核开销可能令延迟不降。
- [01:12:30–01:21:40](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=4350s) · speculative精确分布需要min(1,q/p)接受及正部残差重采样，拒绝时不是直接无校正采q。原课只讲小词表推导，完整证明由陪学另展开；continuous batching和PagedAttention分别管理调度与KV布局。

### 2025 L11 · Scaling 二

来源：[2025-11](https://github.com/stanford-cs336/spring2025-lectures/blob/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [00:29:20–00:30:55](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1760s)。

**μP 不只改变初始化**：课堂明确 μP 同时涉及初始化和分层学习率规则；在某些已有初始化下，学习率差异更显著。要把实现规则和理论条件对应。

例子／演示／合并处理：保留线性网络的尺度推导及非线性、batch、初始化和正则化的消融边界。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:28:56–00:30:36](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1736s) · 学生问 μP 是否仅改变初始化。还涉及按层学习率；WSD 的 stable 阶段与衰减阶段承担不同作用，不能用同一步数曲线直接断言同最终质量。
- [00:38:27–00:40:06](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=2307s) · 学生问全局学习率和新模型是否重做 scaling。讲者明确不知所有团队做法，报告没写不证明没做实验。
- [01:06:06–01:07:41](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=3966s) · 学生问 μP 结构前提。课堂推导为深线性网络；非线性、Attention 与门控层需分别分析，不能机械套同一层宽公式。
- [01:07:35–01:09:12](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=4055s) · 学生确认 fan-in/fan-out，并问不用 μP 是否就没法训练。渐近稳定缩放与有限模型调好学习率不同，μP 是迁移调参的方法，不是训练好模型的必要条件。

字幕全文校读补充与纠错：

- [00:05:00–00:18:20](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=300s) · Cerebras与MiniCPM通过muP减少重复超参搜索，DeepSeek直接拟合最佳batch/LR；成功案例不证明从40M任意外推永远可靠。初始化1/width须分清方差与标准差，不能按一行口述把所有层设同缩放。
- [00:19:20–00:30:40](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1160s) · WSD共用stable前缀、不同终点各自decay，仍需多个模型尺寸跑。把“n² runs”改为重复前缀增加总工作量的直觉，精确复杂度取决于采样终点；192比1只是MiniCPM特定拟合。
- [00:32:30–00:47:30](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1950s) · DeepSeek二维网格有离散误差，学习率线比IsoFLOP曲线更不稳。Llama3约39、Hunyuan96依总/活跃参数及配方口径，不能排成统一自然常数；loss→accuracy的sigmoid也是经验映射。
- [00:50:00–01:05:00](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=3000s) · muP示意推导仅固定深度扩宽、单样本深线性SGD并加额外假设；随机矩阵operator范数乘输入范数是上界，独立性也不使它一般恰等号。SGD与Adam的层LR缩放不同，不能混用。
- [01:10:00–01:17:40](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=4200s) · learnable norm gain和bias是不同参数，字幕remove biases不能代替移除gain的实验。Lion是sign不是sine，特定优化器、强weight decay会破坏该研究中的LR转移；单次大跑验证也不证明任意深度/batch稳健。

### 2025 L12 · 评估

来源：[2025-12](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_12.py) · [00:13:32–00:15:00](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=812s)。

**自适应测试与可比性的矛盾**：多轮对话和红队测试需要对模型回应作出反应；固定脚本未必真实。但不同模型面对不同测试，又使横向比较更难。测试设计需说明目标和控制条件。

例子／演示／合并处理：保留静态／交互式评价、语料似然、智能体和安全评价的区别。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:13:14–00:14:55](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=794s) · 学生问是否随模型调整测试输入。多轮对话/红队需要适应回应，但各模型遇到不同测试时可比性变差；协议应说明目标。
- [00:14:49–00:16:33](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=889s) · 学生问是否只用 perplexity 就够。跨很大规模的整体相关不保证局部下游任务排序，需直接评估目标能力。
- [00:18:10–00:19:52](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1090s) · 历史模型从同分布语言建模转向广泛迁移；不同数据和参数预算的榜单回答不同问题，不能直接横比。
- [00:20:27–00:22:08](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1227s) · 学生问网页数据怎样避免测试集污染。n-gram 过滤可漏掉改写、翻译和来源衍生题，也可能误删；不能把一次去污染当作保证。
- [00:22:06–00:23:46](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1326s) · 学生问蒸馏。讲者指出该榜单主要考察给定数据集的语言建模，而非固定模型大小；蒸馏问题没有在此展开。
- [00:26:35–00:28:15](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1595s) · 学生问每个真实 token 都报概率 0.8 为何非法。只返回目标 token 分数的接口可能掩盖未归一化错误；要检查整个分布与因果遮罩，不能只看 loss 很低。
- [00:30:54–00:32:40](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1854s) · 学生问 HellaSwag 中视频的作用。数据构建用视频来源，但该评估实例是纯文本；来源数据的模态不等于模型输入模态。
- [00:36:40–00:38:17](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2200s) · 学生问 few-shot 示例及现在是否仍用。示例选择、顺序和格式都会影响成绩；少数示例可用于指定任务和输出格式，不证明从中学完学科知识。
- [00:45:38–00:47:19](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2738s) · 学生问黑箱是否偷偷联网及人类是否用 LLM。需控制工具权限并记录协议；供应商和人类测试流程存在信任边界，讲者未给完整验证办法。
- [00:47:08–00:48:50](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2828s) · 学生问专家难题能否代表普通用户。它只衡量一类高价值能力，必须另看普通问答、客服与真实使用任务。
- [00:50:28–00:52:11](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3028s) · 学生指出公开征题者偏差。难度不等于代表性；研究者出题的分布不能直接当总体用户需求。
- [00:51:59–00:53:39](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3119s) · 问句听不清；后续讨论 instruction following 与 Arena。不得根据上下文编造学生提问原意。
- [00:55:30–00:56:10](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3330s) · 学生问 Arena 的随机是什么意思。指自行到网站的参与者，非总体均匀随机抽样；这个词不能当统计采样保证。
- [00:58:05–00:59:48](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3485s) · LLM judge 有长度和自偏好问题；长度校正只是控制一种混杂，不自动得到客观真值。
- [01:03:46–01:05:31](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3826s) · 讲者设问知识与推理如何分开，引入 ARC；不把任务设计当作已经隔离所有先验知识的证明。
- [01:06:35–01:08:22](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3995s) · 安全基准举例仅保留评估目标，不重现有害操作。拒绝与正确帮助分别测量，历史模型个例不当当前安全结论。
- [01:09:27–01:11:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=4167s) · 学生问一律拒绝是否能得高分。必须把安全和有用性成对评价，不能只优化拒绝率。部署前制度属于课程时点，需另查当前政策。
- [01:14:11–01:15:54](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=4451s) · 讲者区分用户真需要答案与故意考模型；线上流量也可能含垃圾和测试。真实使用数据仍需定义目标分布。
- [01:18:20–01:19:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=4700s) · 讲者建议报告去污染检查和不确定性。没有报告不代表已确认污染，也不代表已经无污染。

字幕全文校读补充与纠错：

- [00:05:00–00:13:00](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=300s) · 先确定用户选型、科学比较或开发反馈的目标，再定义输入、调用、输出评分和解释。Pass@1/多次采样、工具权限、延迟和费用要同时说明，单一91分不足以决定部署。
- [00:16:30–00:30:00](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=990s) · 跨域零样本perplexity与同分布split不同。交叉熵下界H、perplexity下界exp(H)分开；逼近文本分布是否带来通用智能是课堂讨论，不是已证明结论，测试集还应避训练与反复调试泄漏。
- [00:32:40–00:50:00](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1960s) · HELM现场查看了题目、预测和few-shot完整prompt，能看见标准prompt与厂商榜单设置有别。GPQA非专家使用Google，字幕一处without后自改with；65%整体专家值不自动对应Diamond。
- [00:53:40–01:00:00](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3220s) · Arena协议争议是所引论文观点与当年事件，不能泛断所有分数无效。IFEval检验可验证形式约束，不保证语义质量；LLM judge相关性不是另一裁判的绝对真值。
- [01:00:00–01:20:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3600s) · agent测的是模型+scaffold+预算，CyBench可攻防双用；能力和实际倾向不同。医疗真实任务数据与隐私之间有约束，考试通过不代表临床部署有效。错误标注可能解释部分剩余错误，但原课“half”没有统一数据依据，不泛化。

### 2025 L13 · 数据一

来源：[2025-13](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_13.py) · [00:18:16–00:20:00](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1096s)。

**数据供应方不替代自己的过滤**：课堂认为 Common Crawl 包含敏感或有害材料，同时明确不确定其所有过滤细节。学习资料保留这种不确定性，不把猜测当供应方保证。

例子／演示／合并处理：保留长上下文数据、指令任务和来源介绍。版权讨论只作当年背景，不写成当下许可结论。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:00:13–00:01:53](https://www.youtube.com/watch?v=WePxmeXU1xg&t=13s) · 讲者强调数据竞争与披露不足。报告少写数据并不能证明具体秘密配方或法律动机，保留为讲者解释。
- [00:17:52–00:19:42](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1072s) · 学生问 Common Crawl 是否过滤、能否拒绝爬取。讲者不知道全部过滤细节；robots.txt 提供爬虫规则，不能等同许可或技术访问控制。
- [00:20:09–00:21:58](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1209s) · 学生问嵌入媒体和版权比例。具体抓取范围依数据版本；公开网页不等于公有领域，法律判断不由本资料提供。
- [00:33:10–00:34:46](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1990s) · 学生问平台独占数据优势。可访问、获许可、实际用于训练是不同命题；讲者推测不当作特定公司内部数据使用事实。
- [00:59:59–01:01:45](https://www.youtube.com/watch?v=WePxmeXU1xg&t=3599s) · 学生问多语言来源。Common Crawl 与派生集可多语言；本讲重点英文不能推导只需英文过滤器。版权部分按历史背景保留。

字幕全文校读补充与纠错：

- [00:00:00–00:30:00](https://www.youtube.com/watch?v=WePxmeXU1xg&t=0s) · 预训练、中训练和后训练的数据边界有重叠。现场随机打开网页/Wikipedia，随机GitHub展示未成功，不能把计划展示当成实际展示。Common Crawl只是网络快照的一部分，学习时区分来源、抓取快照与加工后的数据集。
- [00:31:17–00:35:00](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1877s) · Gutenberg口述“75年”不能当美国版权期限；常见个人作品为作者终身加70年，雇佣等作品有不同规则，依据https://www.copyright.gov/help/faq/faq-duration.html 。PG-19选出版于1919年以前的书，依据https://github.com/google-deepmind/pg19 ，不是2019年以前。
- [00:37:30–00:41:30](https://www.youtube.com/watch?v=WePxmeXU1xg&t=2250s) · Stack Exchange除问答正文还含投票等元数据；GitHub含目录、文件与提交历史，切分和去重单位须明确。字幕的“137 repositories”缺少数量级，未核准前不把它作为仓库规模教学数字。
- [00:42:30–00:59:20](https://www.youtube.com/watch?v=WePxmeXU1xg&t=2550s) · Llama1发表于2023年。Cerebras去重后的627B指tokens，不是模型参数，依据https://www.cerebras.ai/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama 。RedPajama V2的数据构建方式不同于V1；DCLM约240T指候选pool，不是3.8T-token基线数据集，项目入口https://www.datacomp.ai/dclm/ 。保留来源和单位，不能把各数据集规模混排成同口径排名。
- [01:01:48–01:10:00](https://www.youtube.com/watch?v=WePxmeXU1xg&t=3708s) · 公开访问、版权许可、网站合同是不同问题。Creative Commons不等于公有领域；具体许可可有限制。课堂登记费及诉讼进展属于历史时点，不用来推导当前可训练来源清单。
- [01:10:00–01:17:30](https://www.youtube.com/watch?v=WePxmeXU1xg&t=4200s) · 长上下文中训练介绍了书籍、数学与合成数据等方向，没有给出完整配方。Super-NaturalInstructions、Alpaca和Vicuna的数据产生方式不同；教师模型输出的使用条件需查对应条款，不能说开放模型输出都任意可用。

### 2025 L14 · 数据二

来源：[2025-14](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_14.py) · [00:10:52–00:12:15](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=652s)。

**特征哈希与近似的代价**：fastText 的 n-gram 特征空间很大，可用固定数量的桶哈希存储，接受一定碰撞。这与近重复检索中的 MinHash／LSH 不是同一种目的，不能混叫“哈希去重”。

例子／演示／合并处理：保留 KenLM、Kneser–Ney、fastText、DSIR、重要性采样和 Bloom filter；这些旧版细节不被新版压缩讲次抹去。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:10:31–00:12:12](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=631s) · fastText 用哈希桶表示 n-gram 特征，会接受碰撞；这不是 MinHash 的相似度保持哈希，二者目的不同。
- [00:17:27–00:18:50](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1047s) · DSIR 代码演示遇到零概率与随机哈希。Python hash 跨进程可能改变，不宜直接作持久跨机器数据标识；统计模型可平滑，但需明确分布支持。
- [00:21:50–00:23:10](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1310s) · 学生问 n-gram 是否只检查局部通顺。是，可能被重排等方式骗过；作为大库筛选代理的平均收益不等于逐篇质量保证。
- [00:34:11–00:36:00](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=2051s) · 学生问昂贵 LLM 为什么可用于过滤。只标注较小样本，再训练便宜分类器扫描大库；不能把标注开销与全库推断混淆。
- [00:52:02–00:54:03](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=3122s) · Bloom filter 假阳性推导依独立均匀哈希近似。它无假阴性需正确插入、查询且不错误删除；正命中不等于已确认真实重复。
- [00:57:15–00:58:40](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=3435s) · 学生问 Bloom hash 数 k 变小为何假阳性降低。k 过大使位图饱和，过小又约束不足，存在最佳范围；不是 hash 数越多越好。
- [01:13:32–01:15:16](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=4412s) · 学生问语义改写和合成文本去重。嵌入近邻可捕捉更语义的相似，但更昂贵；相似不等于应删除，阈值过宽会丢多样性。
- [01:15:18–01:17:03](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=4518s) · 学生问高质量数据是否应重复。去掉意外重复与有计划重采样不同；讲者对频次开平方/取对数是猜想，不是已验证最优规则。

字幕全文校读补充与纠错：

- [00:02:30–00:07:40](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=150s) · n-gram从计数估条件概率，Kneser–Ney涉及折扣、回退或插值，原课没有完整推导。示例里重复“da”的乱码也能有低perplexity，说明语言模型打分取决于参考分布，不能单独等同质量。
- [00:07:30–00:12:33](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=450s) · fastText把词表到类别的线性映射拆为V×H与H×K两个矩阵；在K=2、H=16时不一定省参数。这里没有隐藏层非线性，句子用词及n-gram特征聚合，哈希用于控制特征表规模。
- [00:13:20–00:20:00](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=800s) · DSIR用目标/原始分布的密度比p/q重采样，要求目标有质量的区域原始分布也有支持。课程在哈希特征上估密度，不等于完全匹配语义。概率越大与perplexity越低方向相反；trace里概率0可来自数值下溢，不能据此认定句子理论概率严格为零。
- [00:25:00–00:36:40](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1500s) · LID现场把hello判为意大利语、把代码判为俄语等，显示短文本/混合语种的模型误差。phi-1的数据筛选随机森林借助教师标签，不应与fastText或手写质量规则混成一种算法。毒性筛选还涉及标签定义和偏差。
- [00:45:00–00:58:20](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=2700s) · 标准仅插入的Bloom filter可有假阳性、没有假阴性；用它去重可能误丢独有文档。假阳性率FPR=FP/(FP+TN)，不是FP/所有判阳项。理想近似为(1−exp(−kn/m))^k，最优k≈(m/n)ln2；0.5是此时bit占用率，不是最终FPR。固定m时不能无限增加哈希数。
- [00:59:50–01:17:40](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=3590s) · Jaccard是集合重叠，不是语义相同；加一个not可能大改含义。理想MinHash碰撞概率等于Jaccard，LSH以1−(1−s^r)^b生成候选。99%不能直接换算为100词只差1词，n-gram与集合大小会改变关系；候选筛选常省工作，但全碰撞时仍可能二次复杂度。

### 2025 L15 · SFT／RLHF

来源：[2025-15](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [00:39:34–00:41:10](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2374s)。

**表面格式与复杂条件关系**：课堂用输出 emoji 的假想问题说明：SFT 可能学会输出风格，却没学会根据输入选择正确内容。能模仿样式不等于掌握目标能力。

例子／演示／合并处理：保留事实引用、安全示范和偏好数据例子；不把“少量 SFT 只能学风格”教成普遍定律。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:04:00–00:05:19](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=240s) · 关键词 demonstration 指人工参考回答。SFT、成对反馈和工程扩展是不同环节，不记为现场代码演示。
- [00:05:07–00:06:27](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=307s) · 课堂安排查看指令数据与交互练习；数据质量在小样本后训练中影响很大，具体练习必须与后续反馈一起保存。
- [00:21:05–00:22:47](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=1265s) · 学生指出工具可改善引用。讲者承认；问题在模型不能兑现示范要求时可能学到生成引用形式的捷径，不意味着所有引用生成都必然幻觉。
- [00:27:36–00:28:56](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=1656s) · SFT 可融入更大规模预训练/中期训练，阶段边界会模糊；不能按名字断言数据只能出现在最后。
- [00:32:12–00:33:51](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=1932s) · 学生问混合切换与遗忘/引用。后期高质量混合和学习率衰减共同改变训练；改变混合本身不保证模型知道引用事实。
- [00:34:40–00:35:45](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2080s) · 学生追问引用知识。可靠记忆与模板捷径可能同时存在；不能用“知识已在权重中”这个二元标签代替行为验证。
- [00:35:47–00:37:05](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2147s) · 学生问训练中自查思想 token。讲者关联 STaR/Quiet-STaR 等路线；是否事实可靠仍需外部信号，不能把自洽等同正确。
- [00:37:30–00:38:55](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2250s) · 学生追问训练时工具自查。是否对事实 token 取损失需在训练时决定；静态语料不能自动知道当前模型掌握程度，模型反应式筛选另有成本。
- [00:38:49–00:40:24](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2329s) · 学生问复杂 emoji 规则。学会在结尾输出 emoji 不等于学会输入到正确 emoji 的复杂映射。
- [00:40:16–00:41:53](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2416s) · 学生追问 SFT 能否教新知识。讲者明确足够规模与多样性可教知识；小 SFT 数据不可靠不等于 SFT 原理上只能学风格。
- [00:45:15–00:46:57](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2715s) · 作者生成与评判的标准可能不同。人更喜欢模型摘要不直接证明模型超越所有专家；验证/生成的成本差异是偏好采集理由。
- [00:51:00–00:52:44](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3060s) · 课堂投票练习：五分钟内多数人未完整核事实，较长含幻觉回答仍获偏好。讲者也不记得每一题标签，资料不补造完整答案表或投票数。
- [00:56:34–00:58:10](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3394s) · 不同标注者重视事实/格式程度不同，模型反馈也可能偏好长回答；群体相关不证明每条反馈正确。
- [00:59:29–01:01:07](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3569s) · 学生问 on/off-policy 与已有答案。相对于当前策略定义数据来源；已知可验证答案的任务与主观偏好任务提供不同监督。
- [01:02:12–01:03:38](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3732s) · 学生问公共价值来源和 judge 自偏好。提供参考资料可帮助标注但非万能，LLM 对自己输出的偏好应在评估中控制。
- [01:03:39–01:05:14](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3819s) · 学生问自改进上限和反馈相对示范的意义。讲者没有给通用迭代上限；理论可能空间不能当成实践收益保证。

字幕全文校读补充与纠错：

- [00:00:00–00:07:30](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=0s) · SFT把基础续写模型调整到指令响应分布。GPT-4相关嵌套提示代码例子在课堂中被阅读，没有据此确认现场实际执行成功。
- [00:11:40–00:15:00](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=700s) · 课堂确实用Google Form让学生写回答，约五分钟后查看结果。出现短答、表情、nah fam及无关冰激凌回答；讲者怀疑某些长答借助AI，这是推测，不能定为作弊事实。字幕无法还原表单完整题面及所有响应行，资料不编造。
- [00:15:00–00:27:30](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=900s) · 课堂比较长答、列表和短答，说明形式偏好与事实质量不同。OpenAssistant的monopsony例子兼顾内容与风格。工具/检索可辅助事实回答，但仍需训练和验证；“kill Python process”的安全误拒绝说明词面规则的局限，500条安全样例不等于覆盖全部风险。
- [00:28:50–00:41:30](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=1730s) · 中训练可混入指令数据，减少遗忘的做法不保证完全消除遗忘。SFT能学新事实，也可能在未知事实上训练出自信回答；不能简化为只能学格式、绝不学知识。
- [00:41:56–00:58:20](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2516s) · 偏好标注受时间、薪酬、题面和标注者群体影响；课堂用人类偏好AI摘要讨论评判局限。国籍/宗教与偏好研究的关联不证明因果，不能推广为整个国家或群体固定持有某种价值观。
- [00:59:06–01:13:20](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3546s) · DPO经KL正则策略解与Bradley–Terry偏好似然建立训练目标，不是直接照搬原始RL目标。PPO策略梯度使用加权log probability；奖励模型会被过优化。DPO改善chosen相对rejected的概率比，不保证chosen绝对概率一定上升或rejected一定下降。

### 2025 L16 · RL 一

来源：[2025-16](https://github.com/stanford-cs336/spring2025-lectures/blob/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [00:30:00–00:40:50](https://www.youtube.com/watch?v=46f2QTDB08Q&t=1800s)。

**归一化改变训练权重**：课堂对 GRPO 的组内中心化、标准差和序列长度展开推导。不同归一化不只是数值写法不同，可能改变问题难度和长短回答的权重。

例子／演示／合并处理：已在音频补核补入GRPO分组、GRPO/PRM区别、CoT可读性与权重同步问答；不能恢复的学生原句仍保留缺口。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:09:56–00:11:36](https://www.youtube.com/watch?v=46f2QTDB08Q&t=596s) · 学生问 DPO 比值写法和奖励过优化图。实现一般用 log 概率差避免直接概率比数值问题；拟合 reward model 与独立 oracle 的分数分开，不能用代理自评分证明真实提升。
- [00:23:50–00:25:35](https://www.youtube.com/watch?v=46f2QTDB08Q&t=1430s) · 学生问 KL 近似和参数更新。单样本 log 比可为负，期望 KL 才非负；把负值截零改变估计，不能称精确 KL。policy gradient 的样本权重按该步停止梯度，再对 log 概率反传。
- [00:31:05–00:32:10](https://www.youtube.com/watch?v=46f2QTDB08Q&t=1865s) · 学生问 GRPO 的 group。每个 prompt 生成 G 个响应，只在同一 prompt 内计算基线；不同 prompt 可同时在 batch 中，但不混成同一组。
- [00:57:04–00:58:41](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3424s) · 学生提醒 PRM 与 GRPO 不同，讲者承认遗漏。前者是逐步奖励模型，后者是策略优化方法；DeepSeekMath 与 R1 对过程奖励的结果不能混为一个实验。
- [00:58:34–01:00:11](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3514s) · 学生问翻译/总结 CoT。可读性、准确率与思维忠实性是不同目标；把输出改写得可读不证明它是内部因果解释。
- [01:11:45–01:13:24](https://www.youtube.com/watch?v=46f2QTDB08Q&t=4305s) · 学生问训练权重同步到 vLLM 和多奖励组合。2025 API 成熟度属历史观察；实际部署需当前版本接口。奖励怎么组合要显式定义尺度和权重，不能从此截断片段猜答案。
- [01:13:10–01:14:20](https://www.youtube.com/watch?v=46f2QTDB08Q&t=4390s) · 学生问多奖励怎样合并。课堂回答加权相加并经验调权；格式奖励是代理目标，权重与尺度会改变行为。

字幕全文校读补充与纠错：

- [00:00:00–00:11:40](https://www.youtube.com/watch?v=46f2QTDB08Q&t=0s) · 本讲接续DPO，介绍长度归一化与SimPO，再用代理奖励和真实胜率区分过优化。具体比较依模型、SFT和数据条件，不能把单张图的结论说成方法永久排名。
- [00:12:30–00:20:50](https://www.youtube.com/watch?v=46f2QTDB08Q&t=750s) · PPO裁剪的是概率比进入的surrogate objective，不是把环境reward截断，也不构成严格trust-region保证。课堂列举RL应用时不能把AlphaFold整体直接归为与AlphaGo相同的RL系统。
- [00:20:50–00:27:30](https://www.youtube.com/watch?v=46f2QTDB08Q&t=1250s) · 终点奖励、逐token KL惩罚及GAE是不同构件。在特定完整episode且gamma=lambda=1时，优势可化为return减value；仍须说明边界、detach与token对齐。
- [00:28:20–00:42:30](https://www.youtube.com/watch?v=46f2QTDB08Q&t=1700s) · GRPO同prompt组内均值包含样本自身，和leave-one-out不同；除标准差改变目标权重。相同reward中心化为零；G=1的无偏样本std为NaN，事后加epsilon不修复。KL形式r−log(r)−1逐样本非负，但期望等价依采样分布及支持条件，不能自动推广到离策略。
- [00:42:56–00:57:30](https://www.youtube.com/watch?v=46f2QTDB08Q&t=2576s) · R1-Zero从已有强基础模型出发，不是从随机参数直接获得推理。长思维链和aha文字不能单独证明真实反思；冷启动数据的未披露教师不猜测。PRM和MCTS在报告中的负面实验不证明所有设置下无效。
- [01:00:00–01:10:50](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3600s) · Kimi用8次无CoT猜测排除容易猜中的题，另用10次采样估难度；两种筛选不能混成排除所有可推理解决的题。依据https://arxiv.org/html/2501.12599v1 。长度惩罚和答案检查仍可能被利用；长rollout、旧策略与权重同步带来系统难题。
- [01:14:20–01:20:10](https://www.youtube.com/watch?v=46f2QTDB08Q&t=4460s) · Qwen3的3995是独立prompt数，不是所有rollout数量。thinking fusion及早停控制预算需结合训练与质量实测，增加预算不保证无限提升；数学专用RL与一般行为可能有取舍。

### 2025 L17 · RL 二：完整小模型链路

来源：[2025-17](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_17.py) · [00:32:30–00:36:00](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1950s)。

**排序任务为何适合讲策略梯度**：同一 prompt 可生成一组响应，组内平均奖励自然成为比较基准。模型参数更新后，后续响应分布也变化，所以 RL 的训练样本不是固定数据集。

例子／演示／合并处理：官方示例模型按位置独立输出，并非自回归 Transformer。两种排序奖励、中心化、归一化、冻结旧策略、重要性比、clipping 和 KL 是独立可解释步骤。

核对边界：字幕全文校读、讲义结构与所用重点对照；未逐句听音频，未遍历所有视频帧。

二轮问答、例子与演示核对：

- [00:10:23–00:12:00](https://www.youtube.com/watch?v=JdGFdViaOJk&t=623s) · 学生问 RL 数据为何变化。每轮策略变化后采样分布也变；这与重复遍历固定监督集不同，并不保证每轮数据一定更好。
- [00:12:48–00:14:29](https://www.youtube.com/watch?v=JdGFdViaOJk&t=768s) · 学生问零奖励为何不改成负值、终局奖励是否只影响最后 token。基线可产生负优势；序列 log 概率包含每个生成 token，终局奖励并非只能回传最后一 token。
- [00:14:40–00:16:27](https://www.youtube.com/watch?v=JdGFdViaOJk&t=880s) · 学生问 0/1 奖励来源。示例解析答案并对答案键检查，无需学习出的 reward model；解析规则本身也可能有漏洞。
- [00:18:10–00:19:54](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1090s) · 学生问为何选次优动作。初始策略不知道哪个动作好，采样噪声可强化次优行为；基线减少估计噪声，不等于每次样本都使模型变好。
- [00:20:23–00:22:17](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1223s) · 讲者用两状态回报展示基线降低方差。为直觉做了梯度简化假设，源码调用 torch.std，显示 5.3229 与 1.1547 实为样本标准差，字幕及变量名的 variance 应纠正；不是一般策略梯度方差公式。
- [00:22:22–00:24:06](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1342s) · 学生问冻结旧策略作基线。基线的行动独立性和停止梯度要分别满足；固定模型可以作参考，不意味着任意依赖所采样动作的基线都无偏。
- [00:28:39–00:30:27](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1719s) · 讲者统一奖励、中心化优势、标准化优势的形式；不同归一化会改权重，不能认为所有变体优化完全相同目标。末尾学生问题另查连续上下文。
- [00:30:00–00:31:20](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1800s) · 学生问 baseline 定义。字幕口述“依赖 S 和 A”与前面推导冲突，采用源码公式 b(s)：可以依状态，不能依当前采样动作，并在策略更新中停止梯度。
- [00:52:11–00:53:59](https://www.youtube.com/watch?v=JdGFdViaOJk&t=3131s) · 学生问为何不给靠近答案的 token 更大信用。早期策略选择也可能关键；终局奖励的信用分配不明，统一赋给序列是建模选择，不是证明所有 token 贡献相同。
- [01:04:40–01:06:05](https://www.youtube.com/watch?v=JdGFdViaOJk&t=3880s) · 学生问旧策略和 reference 的 KL。旧策略是采样/重要性比的短期锚，reference 是长期行为约束；两者角色不同，需接后段完整解释。
- [01:07:00–01:08:30](https://www.youtube.com/watch?v=JdGFdViaOJk&t=4020s) · 学生问第一个 inner step 比率为 1 是否白算。数值为 1 不表示导数为 0，因为旧分母固定；此时 clipping 通常不生效，但策略梯度仍可更新。

字幕全文校读补充与纠错：

- [00:00:05–00:07:30](https://www.youtube.com/watch?v=JdGFdViaOJk&t=5s) · 本讲深入上一讲的policy gradient/GRPO，用排序玩具演示。状态为prompt加已生成token，动作是下一个token；把完整响应合并为一个动作仅为终点奖励推导简化。纯文本追加的动态已知，真实agent调用工具后的外部环境并不都可这样精确模拟。
- [00:07:30–00:16:29](https://www.youtube.com/watch?v=JdGFdViaOJk&t=450s) · 0/1奖励下未减baseline的REINFORCE只从成功样本取得直接信号；极稀疏成功会使学习停滞。过程奖励若可靠可提早反馈，但不能假定每步都有可信评分。
- [00:16:29–00:29:10](https://www.youtube.com/watch?v=JdGFdViaOJk&t=989s) · 两状态回报11/9与0/2用于区分绝对奖励和相对优势。与动作无关、停止梯度的baseline可保持期望策略梯度；样本内估baseline需另析偏差。trace的5.3229与1.1547是标准差不是方差。最优标量baseline带梯度平方权重，简单状态平均奖励通常只是近似。
- [00:31:20–00:37:32](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1880s) · 排序先用位置匹配奖励，再用“输入元素出现＋相邻有序”部分奖励；后者有漏洞，讲者留作思考。本资料另给D09反例并标为新增。GRPO可自然重复同prompt采样，不能据口述断言机器人任务绝不可能有组采样结构。
- [00:37:32–00:42:50](https://www.youtube.com/watch?v=JdGFdViaOJk&t=2252s) · 玩具模型固定输入输出长度，每位置独立采样，用embedding和位置矩阵编码/解码。它不是自回归Transformer；einsum被字幕误转enzyme，vLLM也被误转，按源码术语理解。独立采样不保证各响应接近，只是分布相同。
- [00:42:50–00:50:00](https://www.youtube.com/watch?v=JdGFdViaOJk&t=2570s) · 现场一批reward碰巧全为3，讲者表示随机种子与预期不同；中心化便全零。均值0.1时原0奖励变−0.1，不是口述minus1。标准化对正比例缩放只在忽略epsilon等条件下不变；保留组内最大奖励是另提变体，不冒充原始GRPO。
- [00:50:00–00:58:20](https://www.youtube.com/watch?v=JdGFdViaOJk&t=3000s) · 旧概率须冻结；数值比为1仍可有非零梯度。实现比率应exp(logp_new−logp_old)，不是相除log probability。PPO取未裁剪与裁剪项的minimum并转负loss，不是直接clip梯度，也不硬性界定全部更新幅度。KL低方差估计的优势依分布，不能宣称任何情况下方差都更小。
- [00:58:20–01:08:37](https://www.youtube.com/watch?v=JdGFdViaOJk&t=3500s) · 分清current、rollout old、KL reference三种角色。old可缓存采样token的log probabilities，不必始终保存完整模型副本；ref更新频率取决算法。讲者自述代码比论文少一个循环及若干细节，不能称为逐行完整复现。
- [01:08:37–01:15:45](https://www.youtube.com/watch?v=JdGFdViaOJk&t=4117s) · 现场排序结果没有完全成功，中心化有改善也会卡在组内全同分，标准化展示被跳过。策略loss用的样本不断变，曲线不可当固定验证集loss直接跨时比较；奖励升高也须查实际任务成功率。课程最后明确未展开大型RL系统，CPU玩具不等于已学完其部署工程。

## 三、基础、旧版与作业中的必要补充

以下部分让陪学内容不依赖完成大作业。旧版较细的内容按主题保留，不能只看新版标题就略去。2024 没有取得官方录像，所有 2024 说明均来自讲义／作业，不写成视频问答。

### S01 · 先补数学，再读公式

对应章节：00、04。来源：[2026-02](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) · [2025-02](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_02.py)

概率分布为所有候选分配非负数，合计为 1。条件概率 P(y|x) 表示已知 x 时 y 的概率。自回归模型把整个序列的概率写成各位置条件概率的乘积；取对数后，乘积变成求和。

softmax 把 logits 变成概率；logits 不是概率。把所有 logits 同时减去最大值不会改变 softmax，却能避免指数溢出。交叉熵 −log p(真实答案) 在真实答案概率低时更大；不能把未选中的所有答案都说成错误。

向量存一组数，矩阵可表示线性变换。对 y=xW，先逐个写清 x、W、y 的形状，再解释内积。梯度给出局部变化方向；梯度下降沿负梯度移动，但学习率过大可能越过低点。链式法则让多层模型能逐层计算梯度。

演示入口：D03。

理解检查：模型给真实答案 0.2 和 0.8 的概率，哪一个损失大？为什么用负对数能惩罚过度自信的错误？

### S02 · A1：训练一个模型所需的完整链路

对应章节：02–06、15。来源：[A1-current](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf) · [A1-2025](https://github.com/stanford-cs336/assignment1-basics/blob/spring2025/cs336_spring2025_assignment1_basics.pdf) · [A1-2024](https://github.com/stanford-cs336/spring2024-assignment1-basics/blob/master/cs336_spring2024_assignment1_basics.pdf)

训练样本通常把一段 token 向左错开一位作为 labels：输入 [a,b,c] 对应目标 [b,c,d]。因果 mask 控制每个位置可看见什么；padding／response mask 控制哪些位置计损失。这是不同的遮罩。

Embedding 查表把 token ID 映成向量。线性层乘权重矩阵；Attention 混合上下文；FFN 逐位置变换；残差、归一化与位置表示组成 block。最后投影到词表 logits。参数初始化决定训练起点；模型结构本身不会自动学到知识。

SGD 的基本更新是 θ←θ−ηg。AdamW 维护梯度一阶和二阶矩估计，做偏差校正并用逐坐标尺度调节更新；weight decay 与梯度步分开应用。学习率 warmup、衰减和梯度裁剪分别控制不同阶段和异常更新，不是同一个功能。

训练循环需取 batch、前向、算损失、反向、更新和清梯度。验证集用于选择配方；最终测试集不能反复用来调参。checkpoint 可包含模型、优化器、步数和随机状态，只有保存权重未必能完全恢复训练轨迹。

生成时 temperature 调整 logits 尺度，top-k 或 top-p 限制候选并重新归一化。它们改变采样分布，不直接修改模型权重。TinyStories 等小语料可用于观察训练，但小实验结论需通过消融与尺度变化验证。

演示入口：D02、D03、D04。

理解检查：为什么随机种子相同，也不能保证换硬件后的训练结果逐位一致？

### S03 · 2024 的 LoRA 与 QLoRA

对应章节：23。来源：[2024-15](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf)

完整微调更新原权重 W；LoRA 固定 W，学习低秩增量 BA。若 W 的形状为 d_out×d_in，B 可为 d_out×r，A 为 r×d_in。r 很小时，额外训练参数更少；低秩是约束，不保证保留所有全量微调能力。

QLoRA 把冻结的基础权重以低位格式保存，同时训练小规模 adapter。计算时仍需合适的反量化与计算精度。NF4为16个非均匀码点，不是IEEE浮点。原QLoRA双重量化再将第一层缩放常数存为8bit浮点；64/256分块的常数开销约0.127bit/weight。部分旧讲义把常数量化写成4bit，需纠正。不能把它概括成“所有参数和梯度都用 4 bit 训练”。

省的是哪一部分内存要拆开算：基础权重、可训练参数、梯度、优化器状态、激活。参数高效不等于所有显存开销都按相同比例缩小。

演示入口：无 GPU 复现。

理解检查：LoRA 的参数很少，为什么长上下文和大 batch 仍可能让微调显存不足？

### S04 · 旧版数据过滤方法

对应章节：20。来源：[2024-12](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_12.py) · [2025-14](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_14.py) · [A4-current](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf)

KenLM 可为文本算 n-gram 语言模型分数，fastText 用词及 n-gram 特征做轻量分类。它们表达能力不同，也可能把少见语言、方言、代码或公式误判为低质量。

DSIR 根据目标与候选分布的差异做重要性重采样。直觉是保留更像目标分布的文本；它优化的是分布匹配，不会把任何目标语料自动变成真理。估计误差和目标选择都影响效果。

对有害内容和个人信息处理，要分清检测、遮蔽与删除。分类器分数不等于事实标签，保留比例与误删率需要检查。当前数据使用规则仍需按实际场景单独核实。

演示入口：按分类表手算，不运行大型分类器。

理解检查：筛掉所有罕见词，为什么会同时损害训练数据的某些能力？

### S05 · 旧版去重方法与概率边界

对应章节：21。来源：[2024-13](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_13.py) · [2025-14](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_14.py) · [2026-14](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py)

精确去重比较一致的表示；规范化方式决定哪些差异被忽略。普通哈希要尽量避免碰撞，MinHash 则故意让相似集合更常碰撞，二者目标不同。

标准只插入的 Bloom filter 用多个哈希位置的位判断“可能出现过”。它可能误报存在；在正常模型假设下不会漏报已插入元素。删除、并发错误或实现改变可能破坏这个性质。不能把过滤器误报率当文档重复概率。

后缀数组可帮助找重复子串，适合处理跨文档复制片段；它不等于语义检索。Jaccard／MinHash／LSH 也依赖所选 n-gram 和集合表示。近似检索通常先找候选，再决定如何验证与删除。

演示入口：D06。

理解检查：删除整份文档和删除重复段落，会怎样改变剩余训练分布？

### S06 · 任务数据、长上下文与训练阶段

对应章节：22、25。来源：[2024-14](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_14.py) · [2025-13](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_13.py) · [2026-14](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py) · [2026-16](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)

代码 infilling 让模型在前缀和后缀都给定时生成中间内容，常用特别的序列布局和标记。它与普通从左到右续写的输入格式不同，但仍可用下一 token 目标训练。

长上下文能力牵涉位置表示、有效训练长度、长数据组织、注意力成本和检索／推理任务。把最大长度配置调大，并不能保证模型可靠使用更远信息。课程介绍了若干组件，但没有完整展开可复现的上下文扩展配方。

自蒸馏／教师生成数据需区分任务生成、答案生成、筛选和训练。更强教师的输出也可能不正确。训练集表现变好时，应检查是否只是重复模板或污染了评价。

演示入口：对虚构 prefix／suffix 序列排布，不下载代码库。

理解检查：能处理长输入和能利用远处关键信息，是同一个能力吗？

### S07 · 当前 A5 的 RL 变体

对应章节：25。来源：[A5-current](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf)

原始 GRPO 常减去组均值，再除组标准差，并对每个序列的 token 损失按长度平均。Dr. GRPO 的相应消融移除标准差归一化，使用共同常数归一化，改变问题与长度的相对权重。

RFT／expert iteration 生成多份回答，只保留正确回答做监督学习。它与包含正负相对权重的策略梯度有区别；不能只因都用了模型自己生成的样本，就认定更新完全一样。

本作业的 MaxRL 变体把中心化奖励除以组均值，而不是标准差。组均值低的问题可能得到更大权重；零均值和数值稳定必须处理。作业使用固定归一化常数以控制变量，与原论文的 batch token 归一化有差异。

off-policy 样本来自旧策略，重加权试图纠正分布差异，但可能方差很大。PPO／GRPO 式 token 比率与 clipping 不是完整序列的无偏重加权。GSPO 用序列 token 概率比的几何平均并裁剪，以偏差换取稳定性；也不是无损校正。

训练与 rollout 服务有时使用不同设备和模型副本，更新后须正确同步权重并处理依赖旧权重的缓存。评分函数、答案解析、response mask、样本分组和 loss 的归一化，都是算法实际含义的一部分。

演示入口：D07；大型 RL 未复现。

理解检查：不同算法的 loss 数值更低，为什么不一定表示任务奖励更高？

### S08 · 2025 排序示例：从奖励到梯度

对应章节：26。来源：[2025-17](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_17.py)

示例输入是一组数字，输出应为其排序。官方小模型对输出位置独立采样，不是自回归 Transformer。这个简化让采样、奖励和更新可以分开观察，但不包含实际语言模型的全部依赖。

一种奖励按与正确序列相同的位置计分；另一种奖励把“输入元素出现在输出中”和“相邻输出非递减”相加。奖励定义不同，提供的中间学习信号也不同，必须检查它们是否真等价于目标。

更新使用回答的 log-prob 和回报／advantage。旧策略概率在比率中必须按算法要求固定；若分子分母都随当前参数变化，可能把该有的梯度抵消。D07 以一个标量概率展示这一点。

教材新增 D09 反例：prompt=[1,1,2]，错误输出 [1,2,2] 在包含性加顺序奖励下也得到最大 5 分。原因是 membership 不统计重数。这个反例来自对官方代码规则的分析，不声称课堂已演示它，也不表示课程的所有实验都会遇到同一种失败。

演示入口：D07、D09。

理解检查：为什么评分器完全可计算，也仍可能把错误输出评为满分？

## 四、演示文件与实际运行结果

随包的 CS336-教学小实验.py 仅用 Python 标准库，无需网络、GPU、模型或训练数据。运行 python3 CS336-教学小实验.py 会打印结果，不写文件。输出也保存在 CS336-实验结果.json。以下是重新计算的结果，不是视频原始屏幕输出。

| 实验 | 做什么 | 已运行的关键结果 | 边界 |
|---|---|---|---|
| D01 | 字节对合并 | 21 个初始字节 token → 4 个 token，可逆解码 | 极小语料、允许跨空格合并，非生产分词器 |
| D02 | 因果 Attention | 改变未来位置的 V 后，第一个位置输出仍为 [10,0] | 固定 Q、K、V，非训练出的模型 |
| D03 | softmax 与交叉熵 | [1000,1001] 的稳定 softmax 约为 [0.268941,0.731059]；真实概率 0.2 的损失大于 0.8 | 自然对数，单样本示意 |
| D04 | 梯度累积 | 大 batch 与累积后一次更新均得 w=0.6；逐小批次更新得 0.56 | 均值损失、固定参数累积、无依赖 batch 的层 |
| D05 | 通信原语语义 | 四份输入 all-reduce 得 [6,10,14,18]，reduce-scatter 加 all-gather 同结果 | 列表计算，没有真实通信或带宽测量 |
| D06 | LSH 与数据轮数 | 3×4 band／row 设置下，相似度 0.2 的候选概率约 0.00479，0.8 约 0.79420 | 理想独立哈希假设；不是确定去重阈值 |
| D07 | 组内优势与旧策略冻结 | 回报 [0,1,1,2] 中心化为 [-1,0,0,1]；固定旧分母时比率为 1 但导数约 0.119203 | 标量概率示例，非完整 PPO／GRPO 训练 |
| D08 | 递推稳定性 | a=−2 时 8 次递推从 1 变 256；负号本身不保证收敛 | 不等于 Parcae 真实递推，也不证明非线性模型稳定 |
| D09 | 奖励漏洞反例 | [1,1,2] 的错误排序 [1,2,2] 与正确排序都获包含性＋顺序奖励 5 分 | 教材新增反例，不声称讲者已经演示 |

### 已核对的具体演示帧

初轮以下22个时间点已读取并检查实际视频帧；追加136帧见第十一节。核对范围限于下述可见内容，不等于整段视频或所有屏幕变化已验证。原帧只保留在本地备课区，不随分享包分发。

| 讲次与时间 | 画面确认的内容与边界 |
|---|---|
| 2026-08 · [00:18:40](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1120s) | 画面比较 DDP 与 ZeRO-1：前者一次梯度 all-reduce，后者梯度 reduce-scatter 加更新参数 all-gather。相同通信量结论带有 bandwidth-limited 的说明；不能去掉这个条件说任何系统都免费。 |
| 2026-08 · [00:44:20](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2660s) | 画面比较张量与流水线并行，强调低延迟、高带宽互连适合张量并行；通信表达式的统计单位不同，不能把“每层”与“每 microbatch”混为一个数字。 |
| 2026-06 · [00:28:20](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1700s) | Profiler 表同时出现 CPU 项与 CUDA 时间。可以用于解释两种计时口径和采集开销；不把屏幕中的单次读数推广成正常运行占比。 |
| 2026-06 · [00:51:20](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3080s) | 画面显示 GELU correctness 检查调用、PTX 文件入口，以及 block/thread ID 和寄存器说明。看见调用不等于已经验证它在本机执行通过；PTX 全文未逐条核验。 |
| 2026-07 · [00:41:10](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2470s) | 画面显示 rank=0、world_size=4，输入为 [0,1,2,3]+rank，调用 all_reduce 的 SUM 操作。与 D05 的算术输入一致。此帧不是四个设备并发执行的实测记录。 |
| 2026-14 · [00:39:30](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2370s) | 画面明确列 n=12、b=3、r=4，并按三个 band 分组。与 D06 的 LSH 参数一致；它没有把候选碰撞变成确定阈值。 |
| 2026-18 · [00:39:30](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=2370s) | 图右侧说明蓝色为 QKV+RoPE、橙色为 Attention 的前一部分；画面明确写后者在 QKV 完成前开始加载 KV cache。验证了图的调度含义，没有验证加速倍数。 |
| 2026-18 · [00:50:20](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3020s) | 画面把线性化递推写为 h[t+1]=A_bar·h[t]+B_bar·e，并关注 rho(A_bar)。应区分这里的离散状态转移矩阵与口述中其他参数矩阵，不能简单从“负对角”推出该递推稳定。 |
| 2025-17 · [00:35:40](https://www.youtube.com/watch?v=JdGFdViaOJk&t=2140s) | 画面展示两种排序奖励，并显示 [3,1,0,2] → [0,3,1,2] 在位置匹配奖励下为 1。核对了奖励示例；D09 的重复元素反例是资料另加的分析。 |
| 2025-17 · [00:51:10](https://www.youtube.com/watch?v=JdGFdViaOJk&t=3070s) | 画面进入 freezing_parameters，显示需要固定旧策略概率，并用 sigmoid(w) 构造示例。与 D07 的冻结分母解释一致；不能据单帧声称看到了全部反向结果。 |
| 2026-01 · [01:13:30](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=4410s) | 画面从 UTF-8 字节列表开始训练 BPE，示例为 the cat in the hat，设 3 次合并。这里的 token ID 是字节/词表索引，不能把显示的整数当字符个数。 |
| 2026-02 · [00:25:10](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1510s) | rearrange 明确把 8 维 total_hidden 拆为 heads=2 与 hidden1=4；后续 einsum 再作用于指定隐藏轴。这是维度重排，不是新学习出两个注意力头。 |
| 2026-02 · [01:04:00](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3840s) | 画面显示前向 FLOPs=134217728，反向需要 h1.grad 与 w2.grad 两条乘法；检查比较手推梯度与 autograd。对应官方录制版 trace，不能把窗口显示当成本机重跑。 |
| 2026-03 · [00:36:10](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2170s) | RoPE 图明确标 Query/Key，将坐标两两成组按位置旋转；部分旋转的图是保留部分不旋转坐标。纠正口述/字幕把位置旋转对象说成 keys/values 的歧义。 |
| 2026-04 · [01:03:20](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=3800s) | Switch 辅助损失图定义 f_i 为实际路由频率、P_i 为平均路由概率，loss=alpha*N*sum(f_i*P_i)。不能把离散 f_i 当普通光滑路由概率求导；图内导数针对概率项。 |
| 2026-05 · [01:15:30](https://www.youtube.com/watch?v=izZba4UA7iY&t=4530s) | 画面分别给普通稳定 softmax 与在线 normalizer 的更新。右侧最终输出逐项 y 仍有遍历；FlashAttention 还维护加权 V 的聚合，不能把两者的输出需求混为一谈。D10 验证聚合的重缩放。 |
| 2026-07 · [00:41:40](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2500s) | 标准输出页面显示四个 rank 的 all-reduce 结果均为 [6,10,14,18]；reduce-scatter 分别得到 6、10、14、18。已与官方 stdout 文件逐项对上。 |
| 2026-08 · [00:37:50](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2270s) | Zero-bubble 图将反向拆为激活梯度 B 与参数梯度 W，再安排 W 填补空隙；蓝 F、青 B、绿 W 的调度有明确依赖，不是任意移动所有反向工作。 |
| 2026-08 · [00:59:50](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3590s) | MoE parallel folding 图将 Attention 的 TP/CP/DP/PP 与 MoE 层的 ETP/EP/EDP/PP 分开，说明两种子层可以使用不同并行组，而非全模型必须同一切分。 |
| 2026-10 · [01:14:30](https://www.youtube.com/watch?v=EfM546A79aM&t=4470s) | 画面中 target=q、draft=p，接受率为 min(1,q/p)，拒绝后从正部 (q-p)+ 归一化采样。换文献时 p/q 名称可能反过来，不能只记字母比率。 |
| 2025-14 · [00:56:20](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=3380s) | 转场中的代码文字有重影，不逐字抄写；上方数值框可辨 f≈0.0102156、k≈6.93147。已由官方 trace 和源码确认它们来自 Bloom 假阳性分析，k 是连续最优近似而非可直接使用的非整数哈希数量。 |
| 2025-17 · [01:08:40](https://www.youtube.com/watch?v=JdGFdViaOJk&t=4120s) | 画面显示 epoch=0、step=0 的排序 rollout 日志、各响应 log_probs/reward/delta，以及下一步的 loss。它证明此屏展示了训练记录，不证明所有响应都正确或训练最终已收敛。 |

## 五、必须明确保留的缺口

| 缺口 | 范围与处理 |
|---|---|
| 2026 L08 官方字幕 | 官方轨道仍返回 HTTP 429；已取得原始英文音频，并补本地自动转写。术语和问答听校仍是缺口，不用自动转写冒充官方字幕。 |
| 嘉宾一手资料 | 2024 Ce Zhang、Aakanksha Chowdhery：2024 L17第2页明确嘉宾不录制，独立讲义未取得；2025 Junyang Lin、Mike Lewis与2026 Daniel Selsam完整课堂未取得。相关官方论文不能代替本场问答。 |
| 2024 视频 | 官网未提供本次可确认的官方播放列表，不能声称该年录像已整理。 |
| 音频听校与字幕漏记 | 35 份现有字幕文字已顺序读完，补充与纠错逐讲列出；已对可取得原始音频做完整自动转写比对，疑难窗口再转写并检查语义；尚未人工逐句听校。INAUDIBLE等听不清片段、未被字幕记录的提问仍有缺口。文字读完不等于所有发言已恢复。 |
| 板书、屏幕与所有问答 | 文字能确认的操作已整理。关键演示帧、全部PDF版面主图和可执行讲义图像已检查；未显示/被裁切/细小不可读部分继续标缺，不由取样推出全部视频画面无遗漏。 |
| 源讲义逐页细节 | 已做结构与文本线索核对，并核对本文涉及的重点；PDF 共 1,889 页（核心讲义 1,439 页，作业 450 页）。每页版面主图已查看并有页范围说明；密集公式重点交叉读，未宣称全部脚注小字及引用实验已独立验证。 |
| GPU 复现 | 没有启动 GPU 训练、集群或大型 RL。10 个小实验只验证自己的教学重建，不验证课堂的加速倍数。 |
| 完整长上下文扩展配方 | 2026 L16 讲者明确未充分展开。已有概念不等于已有完整可复现教程。 |
| 后续发展与时点信息 | 2026 春季之后未做全领域更新审查。硬件、价格、榜单、法律和模型规格需要按教学需要另查一手来源。 |

这些缺口不能由 AI 根据常识猜补。若问题落在缺口内，先说明证据边界，再取得来源或换用已核实的小例子。

## 六、来源、版本与分享

课程目录与来源索引保存三年逐讲链接、作业版本、字幕提取元数据与文件校验值。逐讲对照记录另存每个课堂条目的时间范围和核对级别，便于后续补缺。自动按词匹配的候选标题只用于导航，没有被称为人工语义验收。

原始公开视频字幕和官方讲义保留在本地备课区，不打包分发全文。本资料包提供原创中文教学解释、来源链接、小实验与核对状态。分享不代表获得 Stanford 的背书或原资料再分发授权。

## 七、官方执行记录、现场展示与纠错

19 份官方 JSON trace 和 2 份多进程 stdout 已取得并核对 Git 对象哈希。JSON 内展示的数值、源码和图是作者缓存的执行过程。它们能补足单帧未显示的输出，但不能证明本机已经运行原始 GPU 程序。838个短标量/字符串已逐批检查；全部图像版面/主图、405个复合输出的检查与边界见第十二节。

| 对应讲次 | 演示与处理 |
|---|---|
| [2026 L01](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_01.json) | 字节、字符和 BPE 的编码/解码；示例 the cat in the hat 训练 3 次合并，之后编码新句子。当前词表示例与 2025 的 token 数不同，压缩率是具体 tokenizer 的结果。 |
| [2025 L01](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_01.json) | 同一 Hello/地球 emoji/你好 字符串的字节与 token 个数不同；词级例子有 UNK 风险。预分词的空格边界与 BPE 合并规则分别讲解。 |
| [2026 L02](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_02_recording.json) | 录制版实际使用 CPU，get_promised_flop_per_sec 在无 CUDA 时回传占位数 1，导致显示极大的 mfu。此数不具有 GPU 利用率含义，不能当硬件跑分。 |
| [2026 L02](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_02.json) | 当前版与录制版完整源码 diff 已读。当前版补两周预算中的 ×2、修正 ReLU 算术强度注释 1/4、按参数个数计梯度/状态字节、给 backward 标量化、改 checkpoint 复杂度 O(L)。同一网页的新版内容不是录像原文。 |
| [2026 L02](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_02.json) | 当前 trace 的 activation_checkpointing 中 memory=202114048、checkpointed_memory=202900480，示例并未显示显存下降。只能据此说明一次记录；测量包含常驻/临时开销、模型小和切分方法等，不能宣称已经验证理想节省。 |
| [2025 L02](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_02.json) | 录制例子同样为 CPU 占位峰值。16 bytes/parameter 的状态配方与 2026 的 12 bytes 配方不同；都还未计全激活等显存，不可直接用作最大可训练参数量。 |
| [2026 L06](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_06.json) | 完整列入 GELU、softmax、row sum、matmul+ReLU 的源码路径与检查步骤。trace 中朴素、内置、compiled GELU 的排序与口述一致；只保留本次排序，不给跨设备速度保证。 |
| [2025 L06](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_06.json) | 旧版还有手写 CUDA GELU、Nsight 与 torch.compile/Triton 比较。该 trace 中手写 CUDA/Triton GELU 仍慢于 built-in；softmax 的 compiled/内置/手写排序也不同。先检查正确性，再用同形状同精度计时。 |
| [2026 L07](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_07.json) | 单步 trace 与多进程 stdout 分开。四 rank 的 all-reduce 与 reduce-scatter/all-gather 数值一致；DDP 归约梯度，TP/PP 示例主要是前向，不称完整分布式反向训练。 |
| [2025 L08](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_08.json) | 旧版同样保留 communication、DDP、TP 与 PP 示例；局部 loss 不同但同步后参数一致。stdout 只是作者记录，不能证明本机有相同吞吐。 |
| [2026 L10](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_10.json) | prefill/decode 算术强度、KV 缓存、GQA/MLA、剪枝、speculative sampling、continuous batching、paged attention 都有讲稿入口；其中部分是图或公式推演，不是完整服务系统跑通。 |
| [2025 L10](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_10.json) | 公式示例 batch=256 的预估 memory=240779264000 字节，已超过所说单张 80GB H100 的容量。它是代入模型的外推点，不能当可部署配置或实测吞吐。 |
| [2026 L12／2025 L12](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_12.json) | 评估课的网页、示例题、Arena、agent、安全图是原理与案例展示。不是本资料重跑过的榜单；题目提取、工具权限、采样预算和模型版本都会改变结果。 |
| [2026 L13／2025 L13](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_13.json) | 数据集和网页举例保留来源与当年状态；看到仓库或论文链接不表示下载并审计过整个训练语料。版权和访问条款要另外核实。 |
| [2026 L14](https://github.com/stanford-cs336/lectures/blob/main/var/traces/lecture_14.json) | 变换、过滤、Jaccard/MinHash/LSH、混合和后训练数据都有对应函数。Jaccard 示意 3/5；LSH 3×4 分组图与后续 5×10 参数实验不是同一次设置。小来源 50 轮是风险例子。 |
| [2025 L14](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_14.json) | KenLM、fastText、DSIR、语言/毒性过滤、Bloom、MinHash/LSH 的示例逐项登记。DSIR 的一次概率为 0；分类器低分和数值近 1 的输出都不代表经过语义真值确认。固定随机种子/稳定哈希和平滑需另外定义。 |
| [2025 L15](https://cs336.stanford.edu/spring2025/) | 课堂成对判断练习由字幕确认：较长回答可能隐藏事实或数学错误。没有取得可完整核对的题目—投票—答案键，保留该画面细节缺口，不能补造 27 人完整投票表。 |
| [2025 L17](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_17.json) | 两状态例子实际调用 torch.std，因此 5.3229 与 1.1547 是样本标准差，非方差。变量名/口述需纠正。冻结分母示例的梯度从 0 变约 0.119203，与 D07 相符。 |
| [2025 L17](https://github.com/stanford-cs336/spring2025-lectures/blob/main/var/traces/lecture_17.json) | 排序教学模型各位置独立输出，不能当作自回归 Transformer。两种奖励、baseline、归一化、旧概率停止梯度、重要性比、clipping、reference KL 和训练日志均保留。D09 是本文新增的奖励漏洞反例。 |
| [2026 嘉宾 Dan Fu](https://www.youtube.com/watch?v=9EEm4iMAF5s) | 请求生命周期、缓存卸载、Megakernel 时间线和循环模型是这场录像的四段主线。已核对时间线和线性递推帧；Parcae 论文与 Megakernels 代码另作一手补充，不冒充缺失幻灯片 PDF。 |

### 新增 D10：在线 Attention 的分块重缩放

为避免把“在线分母”误讲为“已经得到全部注意力输出”，小实验同时维护当前最大值 m、分母 l 和加权 V 的分子 o。遇到更大分数 m_new 时，旧 l 与旧 o 都先乘 exp(m−m_new)，再加入当前块。最终输出是 o/l。

对分数 [1000,1002,999,1005,1001]，分块大小取 1、2、3、5 都与整行稳定 softmax 加权结果约 [-0.8810706584,3.7736455752] 一致，误差在浮点舍入范围内。这是固定一个 query 的 CPU 数学重建，不是 GPU 加速测试。

## 八、六场嘉宾逐项核查

检查日为 2026-10-22。未取得表示本次在所列公开入口没有拿到足够一手资料，不表示证明其从未公开。相关论文／其他课程演讲只作补充，不能替代本场问答。

| 日程 | 嘉宾 | 材料状态与处理 |
|---|---|
| [2024 L18](https://cs336.stanford.edu/spring2024/) | Ce Zhang | 2024 L17第2页明确说明嘉宾讲座不录制；本场独立讲义未取得。2024 L17第2页写明接下来周一/周三嘉宾不录制（https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2017%20-%20Evals.pdf#page=2）。个人主页说明其研究方向，但不能反推出本课讲座主题。未用其他系统演讲代替。 [相关一手入口](https://zhangce.github.io/)。 |
| [2024 L19](https://cs336.stanford.edu/spring2024/) | Aakanksha Chowdhery | 2024 L17第2页明确说明嘉宾讲座不录制；本场独立讲义未取得。2024 L17第2页是官方不录制证据。主页有 Stanford CS528 的 Gemini/PaLM-E 演讲入口，属于另一门课，未冒充本场内容。 [相关一手入口](https://www.achowdhery.com/)。 |
| [2025 L18](https://cs336.stanford.edu/spring2025/) | Junyang Lin | 本课完整录像、字幕、讲义未取得；已补官方相关阅读。Qwen 官方发布可补阶段配方：先建立一般能力，再提高知识密集内容与扩展上下文；后训练结合长推理示范、规则奖励、思考模式融合与一般行为优化。它帮助串起主线，但没有本场学生问答和现场演示，不能称为该讲重建。 [相关一手入口](https://qwenlm.github.io/blog/qwen3/)。 |
| [2025 L19](https://cs336.stanford.edu/spring2025/) | Mike Lewis | 本课完整录像、字幕、讲义未取得；已补官方相关阅读。新增视频线索 r3DC_gjFCSA 实为 Joe Spisak 的 2024 年会议演讲，不能补本场。官方报告把预训练、后训练、跨语言/代码/工具能力及多模态组合放在同一模型系列中。阅读时追踪数据、结构、计算、评估的对应关系；不把整篇论文当成本场逐字讲义。 [相关一手入口](https://ai.meta.com/research/publications/the-llama-3-herd-of-models)。 |
| [2026 L18](https://cs336.stanford.edu/) | Daniel Selsam | 本课完整录像、字幕、讲义未取得。只确认日程中的姓名，不按个人研究方向编造 Lean/形式证明讲座内容。 |
| [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s) | Dan Fu | 本课录像与字幕已纳入，补充作者论文与代码；独立本课幻灯片未取得。Parcae 原论文进一步明确循环状态由线性转移、输入注入及非线性块组成。负对角参数须经过具体离散化，不能由符号直接判断离散迭代稳定。Megakernel 代码另见 https://github.com/HazyResearch/Megakernels ，含 Llama 演示入口；未在本机编译或做 GPU 性能复现。 [相关一手入口](https://arxiv.org/html/2604.12946v1)。 |

检索路径：三年官网、两个官方播放列表、三份官方仓库完整文件树、可确认的讲者主页、姓名＋课程号的公开视频／讲义搜索。未使用第三方 AI 笔记填补嘉宾的原始课堂内容。

## 九、追加补源与逐页核验

42份PDF的1889页均已查看版面和主图，并有逐页身份记录与页范围内容说明。2026 L08的73页说明如下；其余1816页见第十三节。细小图例和原论文实验的独立验证仍有限制。完整记录见 CS336-PDF逐页核验记录.json 与 CS336-PDF内容核对.json。

补源纠错：二手文章给出的疑似 Mike Lewis 视频 r3DC_gjFCSA，经实际 YouTube 元数据核对，是 Joe Spisak 在 2024 年的会议演讲，不能纳入本课。第 8 讲官方字幕此次重查仍返回 IpBlocked，没有取得新官方字幕。

第 8 讲完整原始音频已用 small.en 二次转写，得到 1,826 段，末尾到 4,805.38 秒；与原 base.en 的 1,702 段作 161 个时间窗比对，并读取差异最大的 35 窗。相似度只帮助定位，不证明准确。两模型属于同一家族；课堂提问仍有不能恢复的部分，尚未完成逐句听校，其他34讲已另做全程自动比对与疑难窗口语义复核，见第十节。

- [00:04:00–00:04:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=240s) · 两个模型仍未清楚还原学生问句。只保留讲者明确解释：分解方式不唯一，reduce-scatter加all-gather是为后续分片算法服务。
- [00:26:00–00:27:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=1560s) · small.en使FSDP与pipeline的追问更易辨认，但仍是机器候选。讲义支持的概念是各rank仍计算完整模型，按需收集每层参数；不是每卡只计算一段层。总参数已经对层求和，不能重复乘层数。
- [00:36:00–00:36:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2160s) · 第二版更像是在追问PP相对FSDP的通信优势。只按p35、p43解释激活规模bsh与参数规模之别；具体大小仍依模型与batch，不保证任何配置都更小。
- [00:46:30–00:47:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=2790s) · base把sbh转成SVH/SPH，small更接近sbh。以p46可见公式为准：s是序列长度，b是microbatch，h是隐藏维度，a是注意力头数；不能按字幕字母抄公式。
- [00:52:30–00:53:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3150s) · 涉及MLP重算的学生问句仍不可靠。p47说明24/t来自Attention和MLP里可被TP切分的部分，不能只由口述把24全部归为MLP。可重算更多激活，但需要增加实际运算，不能把重算称为免费。
- [01:03:00–01:03:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=3780s) · 两个模型对MoE能否套用其他并行方式的学生问句仍含残缺。保留p52–53的可核概念：策略能组合，但EP分组与attention/FFN切法有额外约束。
- [01:08:00–01:09:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4080s) · SP/CP命名追问仍不够清楚；small额外识别出闭源模型循环结构传闻，讲者明确不确定。传闻不作为模型结构事实。SP与CP按p48、p54的实际切分对象区分。
- [01:15:30–01:16:30](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4530s) · Llama 405B被small转成4/5B，Gemma被两个模型转成Gamma；这说明更大ASR也会制造错误。型号按p66–68及作者报告核对。148是故障类别计数，不是所有中断；原论文百分比不一致另列PDF纠错。
- [01:17:30–01:18:00](https://www.youtube.com/watch?v=6-cXp-aOmdg&t=4650s) · Mixtral、DeepSeek V3、Nemotron等专名两版仍有误拼。p69–72的Megatron截图表示文档配置案例，不当成作者已经用该配置训练的证明。

### 2026-lecture-08.pdf · 实际页面核对

- 第 1、2、3、4、5、6 页：本讲有网络基础、并行方式及大模型配方三部分。p4–6历史计算/模型尺寸/互连图是有日期的示例；一张卡放不下特定模型不等于任意小模型都需集群。旧硬件图不能当当前平台规格。 待核：历史图中每个小字和曲线点未逐项向原报告核实。
- 第 7、8、9、10、11、12 页：图示all-reduce可以分成reduce-scatter再all-gather；带宽最优说法须有拓扑、大消息等前提。TPU图从toroidal mesh扩展到TPU8的交换网络，不能泛化TPU永远只有mesh。 待核：厂商对比表的所有规格与宣传口径尚未独立核实，不作为选型建议。
- 第 13、14、15、16、17、18 页：DP切batch并同步梯度；平均损失若改成梯度求和，要同时说明学习率/归一化约定。16B/param来自BF16权重2、梯度2、FP32主权重4与两矩各4，不含激活。ZeRO表是每GPU持久训练状态，不是全系统内存。 待核：不把此精度账本作为所有优化器与实现的固定常数。
- 第 19、20、21、22、23、24、25、26、27、28 页：ZeRO-1/2可保留近似2倍参数量通信，但仍有时延与调度开销；ZeRO-3使用两次all-gather和一次reduce-scatter的示意。p26的重叠不是任意网络都能完全遮住通信。p28改用12B/param后，8×80GB算得上限6.67B/16B/24.62B/53.33B，未预留激活与临时buffer，不能称为实际可训练尺寸。 待核：算式上限可核对，实际峰值显存需完整模型/实现和运行。
- 第 29、30、31、32、33、34、35、36 页：p30“stage1/2不让memory scale”指完整参数复制瓶颈，不能抹去优化器/梯度分片收益。PP图含F/B依赖及microbatch，bubble/useful≈(p−1)/m；若分母换总时间则为(p−1)/(m+p−1)，假设阶段平衡且忽略其他开销。 待核：图中每个实验点没有本机GPU复现。
- 第 37、38、39、40、41、42、43 页：交错PP用更多通信换利用率。Zero-bubble将输入梯度B与权重梯度W分开，W必须在相关参数更新前完成，不能“任意时候”。TP图的列切上投影/行切下投影使局部GELU后汇总部分和；p43通信公式须注明dtype、层、microbatch及前反向口径，不能只拿裸数8比较。 待核：所有零气泡时槽的小编号未逐格复算。
- 第 44、45、46、47、48、49 页：峰值内存是动态量。图式每层激活sbh(34+5as/h)、TP后的sbh(10+24/t+5as/(ht))、TP+SP+选择性重算后的34sbh/t依特定训练架构/精度。SP切norm/dropout等逐token部分；不能当FlashAttention下仍需保存整个s²矩阵的通用结论。 待核：34、24、10等系数需对应所引实现，不推广成所有Transformer的下界。
- 第 50、51、52、53、54、55、56 页：EP按专家分配并all-to-all路由，CP/ring attention则切上下文块。MoE parallel folding的attention与FFN进程组可不同，不能无条件把DP×TP×PP×CP×EP相乘求GPU总数。汇总表中EP/PP/CP的0表示未启用，不是并行度因子0；规范计算用1表示未切该轴。 待核：p56的具体交叉点是特定mesh/模型/带宽例子，不当通用阈值。
- 第 57、58、59、60、61、62 页：先让模型放下再扩DP是经验起点。p61图注明64 A100 GPUs，下面却写64 machines；教学统一为该实验64张GPU与PP8×TP8，不说64台机器。p62重算用额外算术换更大可行batch，吞吐收益随batch/并行配置变化，不能称其计算免费。 待核：引用的2021实验不是2026当前硬件推荐。
- 第 63、64、65、66 页：OLMo、DeepSeek、Yi和Llama使用不同状态精度/分片/拓扑，不能只看一个并行倍数横排优劣。Llama405B表中16384=TP8×CP1×PP16×DP128；长上下文阶段改为CP16、DP8仍为16384。 待核：各模型全文配方及后来版本尚未逐项追溯。
- 第 67 页：故障表继承原论文不一致：54天466次中断中419次非计划；表内计数合计419，但148/419≈35.32%，不是同栏30.1%。百分比栏合计94.9%。保留计数、指出冲突，不猜哪个原数该改，也不把148称为全部中断。 [原始依据](https://arxiv.org/html/2407.21783v1#S3.SS3.SSS4)
- 第 68、69、70、71、72、73 页：p71标题225B-A22B是笔误，Qwen官方名称为235B-A22B，表内截图也写235B。p69/71引用Megatron配置属于文档示例，不能当作者实际训练配置。p70 PP=0仅表示未启用；EP与CP分组有重叠，不能直接乘表格所有数字推卡数。Gemma27B例子768×8=6144只适用于该配方。 [原始依据](https://qwenlm.github.io/blog/qwen3/) 待核：表内未知值??继续保留；不从别的模型填补。

## 十、原始音频比对与问答补核

35场可取得录像均有原始英文音频证据。34场用完整base.en转写比对上传字幕，再对所有字幕INAUDIBLE及低相似度/低置信度窗口用small.en转写；L08已完整双模型处理。下列为对疑难文本的语义检查，不是人工逐句听校。两模型同属Whisper家族，可能共同错听；无法恢复的原问保留缺口。完整音频与转写留在本地，分享包仅含原创说明与源哈希。

### 2025-01 · 音频补核

- [00:00:55–00:01:42](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=55s) · TA介绍低音量人名/导师姓名仍不清，不用ASR猜名。
- [00:19:30–00:19:49](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=1170s) · 毕业后带队建frontier model是课堂预期讨论，不能写成修完课程即可独立完成。
- [00:26:15–00:26:34](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=1575s) · 学分行政问答为当年校内安排，网上陪学无需遵循。
- [00:43:55–00:44:35](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=2635s) · 学生问旧作业链接是否更新；保存年度版本差异，不能把2024链接当最终2025作业。
- [00:54:55–00:55:35](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=3295s) · 讲者称A1/A2工作量大，供课程历史范围，不将作业难度变为本陪学门槛。
- [01:05:25–01:06:05](https://www.youtube.com/watch?v=SQ3fZ1sAqXI&t=3925s) · 口述/字幕A maps97应为小写a，ASCII大写A是65；emoji127757是🌍码点，字节数与code point数分开。compression ratio1讨论后有修正，不照孤句结论。
### 2025-02 · 音频补核

- [00:36:55–00:37:35](https://www.youtube.com/watch?v=msHyYioAyNE&t=2215s) · 结构化2:4稀疏峰值依特定模式与硬件；课堂no one uses it为修辞，不能当行业全称事实。dense MFU分母不可擅用sparse宣传峰值。
### 2025-03 · 音频补核

- [00:26:23–00:26:43](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=1583s) · RMSNorm搬运元素数与LayerNorm相近，省算术不保证同幅耗时降低；small.en未恢复末技术词，不补写。
- [00:47:13–00:47:35](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=2833s) · head_dim/heads/model_dim ratio与效率问题原问仍缺词，需保持讲者复述和shape前提。
- [01:01:55–01:02:35](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=3715s) · 多语言词表对高资源英文的收益较小是课堂判断；一次epoch以内仍可能记忆，不能照口述推成无过拟合。
- [01:03:14–01:03:39](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=3794s) · weight decay可能影响训练动态，即使该图validation差别小；不推成任何设置都只优化不泛化。
- [01:26:25–01:27:03](https://www.youtube.com/watch?v=ptFiH_bHnJw&t=5185s) · 尾部RoPE局部+长程NoPE为特定混合架构例，不保证任意长上下文外推正确。base.en越过5223s源时长继续重复Okay至5246s，属无效转写尾，不纳入发言或覆盖计数。
### 2025-04 · 音频补核

- [00:05:55–00:06:35](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=355s) · 问MoE少FLOPs但wallclock是否可能仍贵，学生原问有缺字；small.en尾部幻觉重复不能采用。通信/搬运成本需另算。
- [00:20:36–00:20:59](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1236s) · 学生明确K>1问输出组合，讲者复述K=1是口误；按Top-k加权合并公式，不沿错口述教学。
- [00:26:23–00:26:42](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1583s) · softmax+Top-k的直觉追问仍缺词；不能把softmax称为已使离散选择可导。
- [00:28:25–00:30:10](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1705s) · router e为独立可学习参数，不与FFN权重绑定。采样、Top-k与加权期望区别保留；学生bias/期望问句未完全恢复。
- [00:36:01–00:36:19](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2161s) · 细粒度专家缩小每专家同时增加激活数量，可保持相近FLOPs但通信/路由不免费。
- [00:39:25–00:40:39](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2365s) · dense FFN宽度与细分专家ratio问题仍有缺失；讲者承认不知道某些exotic比例原因，不编最优性解释。
- [00:46:33–00:46:51](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=2793s) · 随机估计可改变目标梯度估计问题，但Top-k仍不连续，不能说stochastic就自动普通反传可导。
- [00:56:25–00:58:35](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=3385s) · hash routing依输入确定路由，不是每次独立乱选；讲者从未测过纯随机且猜会差，标为猜测。多expert可共置GPU，不能按层×expert一一分配GPU。
### 2025-05 · 音频补核

- [00:09:55–00:10:17](https://www.youtube.com/watch?v=6OBtO9niT00&t=595s) · 两GPU图分别卡通与芯片布局，同一抽象层次不同，不按卡通框数当物理SM数量。
- [00:18:36–00:18:55](https://www.youtube.com/watch?v=6OBtO9niT00&t=1116s) · Tensor Core可用于张量计算中的矩阵块运算，不代表硬件直接加速任意算子或任意形状而无需映射。学生末技术词仍缺。
- [00:47:55–00:48:35](https://www.youtube.com/watch?v=6OBtO9niT00&t=2875s) · coalesced按同warp线程同时访问的地址看，不仅按单线程逐次读取顺序；重复加载和不合并是两类开销。
- [00:51:22–00:51:44](https://www.youtube.com/watch?v=6OBtO9niT00&t=3082s) · 图中重复M00访问为完整画法，可在片上复用，不解释成每次都再从HBM读。
- [00:54:43–00:55:16](https://www.youtube.com/watch?v=6OBtO9niT00&t=3283s) · prefetch/读算重叠是GPU优化能力，但异步拷贝/双缓冲需具体实现，不保证全部由硬件自动完成或完全隐藏等待。
### 2025-06 · 音频补核

- [00:23:25–00:25:39](https://www.youtube.com/watch?v=E8Mju53VB00&t=1405s) · profiler CPU列的self/total比例不是CPU FLOPs利用率，也不保证CUDA时间必须小于其所列CPU算子时间。add换matmul CPU时间为何下降讲者不知；profiler有开销，不能编因果解释。
- [00:30:27–00:30:48](https://www.youtube.com/watch?v=E8Mju53VB00&t=1827s) · CPU在GPU计算时做什么问题交给后续Nsight时间线，学生原句缺词保留。
- [00:49:52–00:50:12](https://www.youtube.com/watch?v=E8Mju53VB00&t=2992s) · blockIdx/blockDim/threadIdx生成工作索引，数学坐标与物理线程调度不同。
- [00:55:51–00:56:10](https://www.youtube.com/watch?v=E8Mju53VB00&t=3351s) · 学生问CUDA文件扩展名，不能把ASR dots猜成算法概念。边界判断避免无效访问。
- [00:59:06–01:00:07](https://www.youtube.com/watch?v=E8Mju53VB00&t=3546s) · contiguous是此示例kernel断言；transpose视图可非连续，不等于底层分配发生碎片。外层contiguous拷贝与通用stride实现是不同成本。
- [01:00:57–01:01:25](https://www.youtube.com/watch?v=E8Mju53VB00&t=3657s) · 手写高层逐算子慢，不是每一步都GPU→CPU传输；中间HBM流量与多kernel launch分开。
- [01:13:48–01:14:16](https://www.youtube.com/watch?v=E8Mju53VB00&t=4428s) · 能否胜过torch.compile需实测目标shape与正确性，课堂不提供无需benchmark的保证。
### 2025-07 · 音频补核

- [00:19:42–00:20:04](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1182s) · optimizer state按参数分片，先reduce-scatter得到完整求和梯度的shard、局部更新后all-gather参数；并非只能更新本rank最初那批数据的参数。
- [00:22:55–00:24:05](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1375s) · 通信量问答把约2M口述成全系统total，需纠正记账：ring all-reduce每rank发送2(P−1)M/P元素，全系统发送2(P−1)M；P=4为每rank1.5M、全系统6M。必须注明元素/字节、发送/收发和拓扑，不能混两种分母。
- [00:26:25–00:27:12](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1585s) · 分片省每卡状态让复杂优化器可能可行，但更多矩/矩阵状态有计算通信成本，不意味应无条件增加任意高阶矩。
- [00:31:25–00:32:05](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1885s) · baby FSDP图有整层归属教学画法；实际参数通常层内分片，前向该层前all-gather，不能当仅交换激活的PP。
- [00:37:48–00:38:06](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=2268s) · prefetch需要额外buffer，显存满时不能无成本提前收集所有参数；small.en未恢复学生句，按原字幕保留。
- [00:40:32–00:41:38](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=2432s) · FSDP交换参数与TP/PP交换激活及其梯度的区别保留；学生all-gather追问部分仍缺词。
- [01:02:09–01:02:27](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=3729s) · TP与PP可组合，仍需选进程组/切分轴和调度，不是默认叠两个API就成立。
- [01:10:25–01:11:19](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=4225s) · 复杂DAG通信/流水并行追问讲者说未想过、猜测可能有更多并行，不能写成已证实无瓶颈。
### 2025-08 · 音频补核

- [00:10:55–00:11:35](https://www.youtube.com/watch?v=LHpr5ytssLo&t=655s) · PCIe是否经过CPU的问题原句有缺词；物理P2P拓扑与主机发起调度不同，不能由口述假定每次经过CPU内存。
- [00:30:25–00:32:05](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1825s) · reduce-scatter记账假设输入已在设备上，未计最初装载。它对各rank相应块归约再分片，输出一shard，不是单纯scatter或在单卡矩阵上做普通行归约。all-reduce约2倍环步骤口径另见L07修正。
- [00:37:39–00:39:35](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2259s) · rank需匹配collective调用顺序，漏一次可挂起；初始化参数应相同，rank参数只把模型放到相应GPU，不能据函数签名以为每rank初值不同。
- [00:45:54–00:47:05](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2754s) · TP backward未实现因课堂篇幅，不是已证明其数学困难；PP示例各rank可先生成同样输入用于演示，不是实际部署必须每rank持全数据。
- [00:50:15–00:51:05](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3015s) · event-driven类比中small.en重复mouse click，属幻觉不纳入。示例仍由程序控制send/recv依赖，不是后台自动推断全部调度。
- [00:52:33–00:55:48](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3153s) · 异步send返回handle并须等待；多send顺序讲者明确不确定，不能写成跨后端/stream统一FIFO保证。无匹配recv可等待/超时，不能说所有后端必永久静默挂起。
- [00:55:59–00:56:18](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3359s) · 最后rank持完整forward最终activation，不代表完成backward或同步了全部参数。
- [01:03:20–01:04:51](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3800s) · BatchNorm等数据相关buffer不能由参数同步直接推出一致。PyTorch niceties学生原问仍不可辨，不能补造完整API答复。
- [01:06:23–01:07:43](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3983s) · checkpoint API指定重算边界，字幕/ASR MAML应matmul。GPU会否被专用硬件替代是预测问答，不当确定未来事实。
- [01:08:55–01:10:40](https://www.youtube.com/watch?v=LHpr5ytssLo&t=4135s) · 这些并行机制可用于从checkpoint继续梯度训练，不会神奇免掉新数据的forward/backward计算；专用芯片物理限度追问仍有缺词。
- [01:13:25–01:15:10](https://www.youtube.com/watch?v=LHpr5ytssLo&t=4405s) · collective是抽象语义，后端可CPU/GPU，主机调度与设备执行分开。尾部改口下次课日期非知识点。
### 2025-09 · 音频补核

- [00:20:18–00:20:35](https://www.youtube.com/watch?v=6Q-ESEmDf4Q&t=1218s) · 学生问仿真如何生成给定intrinsic dimension数据，后文均匀格点例只解释直觉；不能据此称LLM真实内在维度已被测得或幂律已证明。
### 2025-10 · 音频补核

- [00:36:25–00:37:05](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=2185s) · 性能公式假设单GPU并忽略部分compute/开销；多用户不同到达/结束时间问题引到continuous batching，不能把静态batch模型当完整服务系统。
- [00:45:25–00:46:05](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=2725s) · 讲者明确所展示表不是原想说明MLA提升的表，不能以其证明提升；用后续对应原表限定结论。
- [00:50:55–00:51:43](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3055s) · CLA口述weights shared容易误解：原论文只让部分层产生K/V投影，另一些层复用之前的KV激活，不能说所有层全部权重共享，也不保证每层少一次KV读取。见[CLA原论文§2.2–2.3](https://arxiv.org/html/2405.12981v1#S2.SS2)。
- [00:51:55–00:52:35](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3115s) · gist tokens/上下文摘要可能压缩缓存但会丢信息，和无损分页管理不是同类优化；ASR just应gist。
- [00:54:25–00:55:05](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3265s) · S4在线性时不变设定有递推/卷积表示；长程合成任务强不保证语言建模强，是后续input-dependent结构的动机之一。
### 2025-11 · 音频补核

- [00:29:06–00:29:29](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1746s) · μP不只改初始化，还需分层学习率/输出等缩放，不能因学生提问只讲初始化。
- [00:30:07–00:30:36](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1807s) · 稳定LR阶段曲线接近cosine是所示实验，不保证任意预算同loss；需比较最终decay尾和复用前缀的成本。
- [00:38:38–00:39:14](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=2318s) · 所引DeepSeek实验调全局LR/batch与完整μP规则分开。small.en重复global LR问句属转写重复；讲者不确定前沿实验室是否每次重做分析，不编业内惯例。
- [01:06:15–01:06:35](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=3975s) · 架构/宽度极限假设影响μP推导，不能自动迁移任意新架构/深度。
- [01:07:25–01:08:11](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=4045s) · fanin/fanout指输入/输出维度，课中交换术语后自我澄清；矩阵存储方向与数学输入输出轴需先明确，再解释nl/nl−1。
### 2025-12 · 音频补核

- [00:14:25–00:15:35](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=865s) · perplexity与多种能力相关，不覆盖全部任务/偏好/工具能力；学生问边界而非证明统一能力尺度。
- [00:19:35–00:20:57](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1175s) · GPT-2不同规模评测表与Penn Treebank例需查污染；web训练不自动保证test未出现。
- [00:21:55–00:23:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1315s) · 蒸馏/数据预算学生问句仍不全，不能补成明确可复现实验；比较benchmark需先规定模型/工具/算力预算。
- [00:26:46–00:27:46](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1606s) · 同一上下文所有候选概率都.8会违反总和1，跨不同位置的真词各.8则可能成立。只有给定token logprob的黑盒接口难检查全词表归一，不能把低PPL当实现正确的充分条件。
- [00:30:55–00:31:35](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1855s) · HellaSwag此处模型输入全是文字，视频仅解释数据来源ActivityNet，不能因图中有播放图标称多模态评测。
- [00:36:25–00:37:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2185s) · few-shot被转为future；示例选择/相似性与格式会影响分数，测试答案泄漏不等于合法few-shot。第二问原句不全，只保留可恢复大意。
- [00:45:25–00:48:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2725s) · Google-proof不保证闭源模型无搜索/无污染，应控制工具；专家题与普通用户需求不同。small.en把模糊学生句转成cheat和重复长句，没有可靠证据不采纳。
- [00:50:25–00:51:35](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3025s) · HLE开放征题可能有参与者选择偏差，学生提出的是批评，不因此宣判基准无效。O3约20为课堂时点快照，非今天分数。
- [00:52:11–00:52:28](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=3131s) · 短问句两版均不可恢复，不为过渡到instruction benchmark编答案。
- [01:06:54–01:07:15](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=4014s) · 安全评测例把危险请求的拒绝行为当观察对象；两版对实体/细节不一致，不恢复操作细节或作为事实知识。
- [01:08:55–01:10:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=4135s) · 自动越狱suffix转移为特定历史实验，不等于所有模型都同样可绕过；拒绝所有问题可抬某些安全率却牺牲帮助性，须同时报过拒率。
- [01:20:25–01:20:48](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=4825s) · 结尾强调评测目的与游戏规则，尾段非未覆盖的新benchmark。
### 2025-13 · 音频补核

- [00:00:25–00:00:43](https://www.youtube.com/watch?v=WePxmeXU1xg&t=25s) · 数据最重要是讲者hot take，不是可验证全任务定理；Tatsu之名第二转写可恢复但不扩大为完整观点引语。
- [00:18:19–00:18:36](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1099s) · Common Crawl默认宽松是讲者判断并明确不知所有过滤细节，不能当官方无过滤保证。
- [00:20:31–00:21:35](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1231s) · 抓取响应可能含图像/其他媒体；偏文本不等于完全无非文本。copyright比例是口头概括而非逐项许可审计，不给通用授权结论。
- [00:32:55–00:33:35](https://www.youtube.com/watch?v=WePxmeXU1xg&t=1975s) · 学生提出xAI数据优势与长期公司格局假设，不能把问题中的预测作为讲者证实的事实。
- [00:46:55–00:47:35](https://www.youtube.com/watch?v=WePxmeXU1xg&t=2815s) · RefinedWeb数据现场想展示但分辨率不足后放弃，不能编看到的样例；Trafilatura专名按讲义纠正。
- [00:59:55–01:00:35](https://www.youtube.com/watch?v=WePxmeXU1xg&t=3595s) · 本讲聚焦英语不代表Common Crawl无多语；未展开多语配方部分明确保留深度边界。
### 2025-14 · 音频补核

- [00:34:25–00:35:05](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=2065s) · 昂贵LLM只标小规模约100k示例再蒸馏快速分类器，不是所有原始文档都用同一LLM打分，数字为课堂规模例。
- [00:53:55–00:54:35](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=3235s) · hit/miss口述和字幕互换，LSH按b个band每band r行推导：同band全中概率s^r，所有band未中(1−s^r)^b，至少一个候选1−(1−s^r)^b；不照混乱文字抄公式。
- [01:01:55–01:02:35](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=3715s) · MinHash随机排列例A={1,2,3,4}、B={1,2,3,5}，共享最早元素概率为Jaccard=3/5；small.en最后误重复A，实际比较A与B。
- [01:13:53–01:14:35](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=4433s) · 合成数据改写需语义去重，讲义相关SemDeDup嵌入路线不同于字节精确/MinHash词集合；不把所有改写保证检出。
- [01:18:55–01:19:11](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=4735s) · 尾部是下周alignment/RL预告，非缺教学内容；不把下一单元补编成本节演示。
### 2025-15 · 音频补核

- [00:32:20–00:33:22](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=1940s) · decay阶段换精选混合是两阶段配方动机，loss下降同时受LR与数据分布影响，不把混合独立贡献由单曲线读出。遗忘/幻觉双问题有缺词。
- [00:34:25–00:35:05](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2065s) · Schulman引用例在讨论模型不知引用时如何受示范激励；知识是否已在参数里难直接判定，不把该故事当已证明幻觉唯一原因。
- [00:35:55–00:36:35](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2155s) · 自检thought tokens与Quiet-STaR/RL相关，不保证自我一致即事实正确。学生提出方案不是课程已实现的演示。
- [00:38:55–00:39:43](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2335s) · 预训没见emoji而后训练加入的问句只能恢复大意，取决编码与组成性，不能一律断言能/不能学习新知识。
- [00:51:13–00:51:31](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3073s) · 较长回答被学生指出事实错误，课堂偏好投票不能当正确性标签；不恢复未辨认具体错误内容。
- [00:57:06–00:57:26](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3426s) · GPT-4与human win-rate相关性图限该样本与judge版本，不意味模型评判与人工完全可替代；同源judge自偏好需检查。
- [00:59:38–01:01:05](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3578s) · on/off-policy后续展开，不能仅按是否人类给标签定义。利用已有有答案数据做对齐的提问可保留，但现有答案也须核对正确性与目标形式。
- [01:01:55–01:04:06](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3715s) · 公共价值来源/投票与UI干预不能一次解决规范差异。模型自偏好是评估风险；自反馈可迭代多少讲者称经验问题，无固定信息收益/最优迭代次数。
- [01:04:40–01:05:35](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3880s) · 字幕SRT应SFT；有参考答案监督与偏好/奖励反馈形式不同。学生问题含糊，讲者继续方法slides，不能编完整额外答复。
### 2025-16 · 音频补核

- [00:09:55–00:11:35](https://www.youtube.com/watch?v=46f2QTDB08Q&t=595s) · RL代理reward上涨不保证真实偏好上涨；图中oracle/人类反馈是新鲜独立评估，不能让同一拟合RM兼任ground truth。DPO对数分式展开的问题原句缺词，不凭ASR添加新目标。
- [00:30:55–00:31:35](https://www.youtube.com/watch?v=46f2QTDB08Q&t=1855s) · GRPO需每个问题采多个回答，再按问题分组算基线；batch可含多个问题，不能用全batch均值替代各问题组。
- [00:57:14–00:57:37](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3434s) · 学生提醒GRPO是优化法、PRM是奖励/评判方式，讲者承认忘讲该区别；两者不互斥。
- [00:58:01–00:58:19](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3481s) · 语言一致性消融问答small.en重复幻觉需丢弃；原字幕/表支持特定性能与可读性取舍，不把一致语言等于faithfulness。
- [00:58:25–00:59:05](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3505s) · 翻译/摘要可改善人类可读性但不是忠实原始推理的证明；讲者半个百分点为随口举例，不写精确实验损失。
- [01:10:55–01:12:35](https://www.youtube.com/watch?v=46f2QTDB08Q&t=4255s) · Kimi长度平台不证明所有RL都无长度偏差。训练权重须同步到vLLM rollout服务；当年experimental NCCL API有未文档参数，是课程时点经验，不能当今日API使用指南。
### 2025-17 · 音频补核

- [00:21:55–00:22:35](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1315s) · 口述variance5→1与代码不一致：代码是标准差5.3229→1.1547，不能平方量混用。学生关于baseline依策略的原问仍不全；action无关baseline需在score-function项正确处理梯度，不要求它是精确最优V。
- [00:29:55–00:30:35](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1795s) · Chelsea Finn深度RL课被ASR写成Josie；baseline在此为仅依状态/问题的函数，和任意依当前动作的奖励扣分不同。
- [00:52:25–00:53:05](https://www.youtube.com/watch?v=JdGFdViaOJk&t=3145s) · 末尾100token更重要只是学生假设；outcome reward的credit assignment不能靠距离答案近就证明贡献大，折扣会改变目标。
- [01:04:55–01:07:35](https://www.youtube.com/watch?v=JdGFdViaOJk&t=3895s) · old策略约束更新幅度，reference约束偏离起点/行为目标，两者职责不同。首步ratio=1仍有梯度，旧分母须冻结；不把相同数值两边都随theta变化。clipping不保证严格KL上界。
- [01:15:55–01:16:08](https://www.youtube.com/watch?v=JdGFdViaOJk&t=4555s) · 课程末尾可提供指针，未包含额外嘉宾实录或演示；不据预告反推问答。
### 2026-01 · 音频补核

- [00:06:00–00:07:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=360s) · base.en在这一段产生大量重复句，属于转写失败而非课堂重复讲同一内容；不覆盖上传字幕。保留小规模/大规模计算分配不同的概念，课堂具体比例仍是所引配置。
- [00:38:00–00:38:30](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2280s) · 上传字幕MV link与ASR的MB link都不准，按系统讲义统一写NVLink；DGX B200的命名和8 GPU布局按官方原图与指定版本理解。
- [00:47:30–00:48:00](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=2850s) · ASR把“scaling laws are not laws of nature”误转，正文按原字幕及scaling讲义保留：规律需合适训练配方才能实现，不会自动保证外推。
### 2026-02 · 音频补核

- [00:33:00–00:33:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1980s) · 学生在问是否只考虑常规矩阵乘，以及其他如次立方算法；ASR识别失败。按上传字幕保留提问，回答是实际优化还需系统/硬件共同设计，不能据此说所有渐近快算法无价值。
- [00:55:00–00:55:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3300s) · 学生追问为何计算能力相对带宽过剩、能否重新设计加速器；ASR技术词混乱。不根据问句猜精确利用率，答案需区分峰值与端到端受限资源。
- [00:11:38–00:11:57](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=698s) · BF16仍有精度风险，mixed precision保留关键量FP32；small.en把BF16误写F16，不直接覆盖字幕。
- [00:15:25–00:16:24](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=925s) · block scale共享会限制同块数值相对动态范围，1bit问题两模型仍缺词，不能恢复具体硬件主张。
- [00:38:37–00:39:24](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2317s) · promised FLOPs为对应spec峰值，MFU50%只作课堂例，原因后续回到带宽/实现，不虚构逐器件损失分解。
- [00:48:00–00:48:17](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=2880s) · Transformer典型算术强度学生追问留待后续；需要具体prefill/decode与shape，不能给一个常数代表所有层。
### 2026-03 · 音频补核

- [00:16:44–00:17:03](https://www.youtube.com/watch?v=lVynu4bo1rY&t=1004s) · small.en补出tensor contraction，支持学生问归一化搬运为何不成比例；仍不把模糊口语恢复成逐字引语。
- [00:39:30–00:43:06](https://www.youtube.com/watch?v=lVynu4bo1rY&t=2370s) · 串并行质量差异、partial RoPE与相对位置内积追问都保留；small.en仍混技术词，结合原字幕/图理解。讲者承认缺少控制消融，不能把后续模型不用并行当严格因果证据。
- [00:56:31–00:59:21](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3391s) · 学生说multimodal，small.en误转multilingual；bits/byte跨tokenizer比较需同一字节文本和编码。后续追问讲者明确留待课后，不替他编补最终结论。
- [01:04:30–01:04:48](https://www.youtube.com/watch?v=lVynu4bo1rY&t=3870s) · weight decay优化作用问题仍有问句缺词，保留答复与不确定标记。
- [01:23:10–01:25:11](https://www.youtube.com/watch?v=lVynu4bo1rY&t=4990s) · 经验超参仍需按模型假设调参；动态超参答例为weight decay，不推成唯一可变项。MQA在训练时就定义KV头数，不能只在推理时直接删头而默认无损。
### 2026-04 · 音频补核

- [00:20:41–00:21:54](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1241s) · 线性attention并行式与因果递推在相同定义下精确等价；与一般RNN的质量比较并非该等价失效。未听清的指代继续标缺。
- [00:27:29–00:31:05](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1649s) · DSA indexer仍看全部位置，省的是常数；长上下文扩展和post-training顺序明确，不能说冻结模型随手插入即可。未来架构回答为讲者猜测，学生长问句仍无法完整恢复。
- [00:32:25–00:34:03](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=1945s) · SSM有限状态的信息瓶颈与context/state大小相关，不代表所有SSM训练更难；具体缺字不猜。
- [00:42:27–00:44:35](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=2547s) · MoE训练和推理均稀疏，路由按token；通信限制并行扩展。small.en把assignment设计尾句混乱，不作为新作业要求。
- [01:09:55–01:11:21](https://www.youtube.com/watch?v=cKSwj_qZ8Jg&t=4195s) · 专家不必形成按人类主题分工；理想每专家均衡且均匀放置可推出每设备均衡，软辅助目标未完美实现仍需设备级约束。
### 2026-05 · 音频补核

- [00:12:48–00:16:05](https://www.youtube.com/watch?v=izZba4UA7iY&t=768s) · shared memory由程序显式使用，L1 cache由缓存机制管理；block含多个warp，不是整个block每次同一条指令。
- [00:20:05–00:20:22](https://www.youtube.com/watch?v=izZba4UA7iY&t=1205s) · GPU/TPU某项典型大小的学生问句仍无法恢复，讲者明确不知道，不补出虚构数值。
- [00:28:05–00:28:22](https://www.youtube.com/watch?v=izZba4UA7iY&t=1685s) · small.en的O2/O1为L2/L1候选，按讲义解释层级/物理距离而非只凭距离解释所有延迟。
- [00:44:25–00:47:16](https://www.youtube.com/watch?v=izZba4UA7iY&t=2665s) · 量化不限于矩阵，但matmul硬件收益与scale开销决定是否划算；MoE不是所有硬件结构稀疏的同义。Apple本地模型与量化/蒸馏的具体问句仍有缺词。
- [01:00:55–01:01:52](https://www.youtube.com/watch?v=izZba4UA7iY&t=3655s) · Groq被字幕/ASR写为rock/Grok；跨设备切矩阵与单卡tile可类比，但DP/TP切分与共享存储不同，不能说整个模型总能放SRAM。
- [01:18:25–01:18:39](https://www.youtube.com/watch?v=izZba4UA7iY&t=4705s) · 尾部鼓掌而非教学遗漏；两次ASR尾端与音频长度差由结束非讲话段解释。
### 2026-06 · 音频补核

- [00:19:55–00:21:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=1195s) · 同SM可驻留多block，受寄存器/共享内存/线程等资源限制；wave例不意味着硬件绝不让block共享SM。学生具体140/148数字不恢复。
- [00:35:35–00:36:35](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2135s) · 讲者明确当前compiled Triton比built-in慢，不能照学生前提写成Triton更快。
- [00:45:55–00:57:11](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=2755s) · tl.load与Python语句的执行时间分开：编译后GPU执行，PTX并非逐thread再次编译。warp调度隐藏等待不保证全隐藏。四warp scheduler数目原因讲者不知道，继续保留未知。
- [00:58:25–01:04:58](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3505s) · softmax朴素5MN读3MN写为简化账本；单block整行与分块归约适用不同shape，按列softmax需改stride/归约轴，不能只改名字。
- [01:10:12–01:12:05](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=4212s) · “accumulator放shared”是课堂假设，Triton编译可用寄存器等，应以生成代码/资源统计核对，非tl.sum默认一律shared。
- [01:23:55–01:26:35](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=5035s) · ASR多次把matmul转为MAML，统一按演示代码纠正。末尾高维tensor整体/逐元素加载没有抽象统一答案，讲者留到课后，不编具体最佳方案。
### 2026-07 · 音频补核

- [00:11:54–00:21:48](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=714s) · NumPy broadcasting只是复制/可见性的类比；gather/reduce目标root由调用指定，不永久固定rank0。collective既是概念也是实际API。
- [00:29:28–00:35:55](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1768s) · 字幕缺词中的Llama/RDMA/NVSwitch依据上下文保留候选；RoCE是RDMA的一种，网络设备拓扑与操作语义不同。NCCL跨节点具体优化讲者不知，不能编成已证明最优。
- [00:41:54–00:43:52](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2514s) · 此课一rank对应一GPU是示范约定，不是rank定义。某how does it work学生问句两次ASR均未恢复，继续保留缺口。
- [00:52:55–00:54:38](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3175s) · CUDA异步操作计时需等待完成，barrier与device synchronize作用不同；讲者未确认的严格先后顺序不替他下结论。
- [01:00:41–01:01:53](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=3641s) · 示范要求batch至少world size且整除更方便，不是所有分布式训练的数学硬限制；DDP封装可复用于Transformer。
- [01:07:53–01:09:35](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4073s) · forward all-gather的伴随在backward需reduce-scatter，当前toy代码未实现自动反向，不能说调用backward自然全部正确。
- [01:10:41–01:18:38](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4241s) · microbatch减少流水空闲，JAX的sharding可由编译器处理；字幕starting应是sharding，不能混到优化器初始化。
### 2026-09 · 音频补核

- [00:10:14–00:10:33](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=614s) · 学生问经验scaling是否有理论根据，原字幕/第二转写仍有缺字；均值及网格例是解释，不是LLM幂律定理。
- [00:20:39–00:20:56](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1239s) · model大于data不是参数数与token数一比一边界，应看容量相对样本及训练收敛。
- [00:53:45–00:54:03](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3225s) · 重复runs/variance问答指出所见往往单次运行，不能给曲线补上不存在的seed误差条。
- [00:55:43–00:56:02](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=3343s) · 拟合train还是held-out loss需明确评测分布；反复选配方的集属于validation，不能把反复使用test说成规范无泄漏。
### 2026-10 · 音频补核

- [00:18:25–00:20:35](https://www.youtube.com/watch?v=EfM546A79aM&t=1105s) · GQA现场纠正：k是KV头/组数，g是每组query头数；讲者承认原标记有误并说课后改。学生部分问句仍不可辨，不据ASR补写。
- [00:34:25–00:35:35](https://www.youtube.com/watch?v=EfM546A79aM&t=2065s) · 字幕TFRT应为TTFT。memory-bound指此处decode与所算shape，不能扩成全部推理/prefill永远带宽受限。
- [00:58:55–01:02:24](https://www.youtube.com/watch?v=EfM546A79aM&t=3535s) · 学生比较sliding window与Mamba/DeltaNet的混合架构。固定状态压缩有长程检索损失风险；讲者对未来可改进空间的判断是研究判断，不证明任意linear attention优于窗口。末尾缺词仍未恢复。
- [01:05:55–01:06:35](https://www.youtube.com/watch?v=EfM546A79aM&t=3955s) · GPTQ按层用近似曲率补偿量化误差，非把layer by layer的ASR杂词当新算法。
- [01:11:09–01:11:29](https://www.youtube.com/watch?v=EfM546A79aM&t=4269s) · 神经元恒为100是否有意义的问答可看方差；绝对幅值本身不证明语义重要性，学生末尾原句仍不清。
### 2026-11 · 音频补核

- [00:53:17–00:53:35](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3197s) · Muon历史段的人名两次转写为Caillou但字幕缺失，不能据此确定研究者姓名，保留未辨认。
- [00:54:51–00:55:18](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3291s) · 学生原问部分不清；讲者复述为SVD在GPU上是否快，说明实现用有限步Newton–Schulz的矩阵乘，不直接运行完整SVD。
- [00:55:27–00:55:48](https://www.youtube.com/watch?v=vTfEyOyzV9E&t=3327s) · 不同优化器超参数是否共用的问句有缺词；讲者答不同，不能补成任意Muon/Adam超参可直接互换。
### 2026-12 · 音频补核

- [00:34:14–00:34:35](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2054s) · 字幕being应为beating：Bradley–Terry用分差建胜率，Elo更新与静态最大似然拟合不要混为同一步。
- [00:58:53–00:59:10](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=3533s) · human-solvable评测的能力范围与IMO/开放数学研究不同。转写more gold应为IMO gold，超人任务不能靠普通人工时限全部覆盖。
- [01:12:55–01:13:35](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=4375s) · SWE-bench Verified修订任务/测试质量，不意味着所有缺陷已消失。perplexity评测便于自动算分但仍依赖数据质量与分布。
### 2026-13 · 音频补核

- [00:30:45–00:31:35](https://www.youtube.com/watch?v=-qm0ln33G24&t=1845s) · ElevenLabs/voice data讲者明确不熟、课后讨论，不编答复。许可随时间改变的问题保留原问，使用须看具体版本许可，不能由课堂口述推成通用法律结论。
- [00:42:55–00:44:05](https://www.youtube.com/watch?v=-qm0ln33G24&t=2575s) · arXiv论文许可需逐篇核对，不能将讲者“我想大多CC”写成全部可任意使用；synthetic data probably yes为课堂判断，不能代替发布模型/数据的条款。爬虫过滤盗版网站学生问句仍不全。
- [01:09:34–01:09:53](https://www.youtube.com/watch?v=-qm0ln33G24&t=4174s) · total processed tokens与unique tokens区分：多epoch重复计数，不能把总训练token当唯一语料规模。
### 2026-14 · 音频补核

- [00:03:55–00:04:35](https://www.youtube.com/watch?v=5sxHosTLPF8&t=235s) · Resiliparse/Trafilatura/FinePDFs专名依据官方代码/屏幕，ASR音似词不收成新工具。抽取器表现依下游评测，不声明全任务赢家。
- [00:20:55–00:21:35](https://www.youtube.com/watch?v=5sxHosTLPF8&t=1255s) · 高质量数据与训练更久的交互问句转写仍不完整，不能捏造独立正交收益。
- [00:36:55–00:37:35](https://www.youtube.com/watch?v=5sxHosTLPF8&t=2215s) · MinHash签名减小表示，避免n²还需bucket/LSH检索；不能说只算签名即自动解决所有pairs比较。
- [00:55:55–00:58:35](https://www.youtube.com/watch?v=5sxHosTLPF8&t=3355s) · 为何50 epochs的追问部分仍不清；讲者强调配比会造成小语料高重复，应记每源有效epochs。通常按sequence选mixture来源、batch内混合降方差，不是每token换域或必定固定10步切换。
- [01:12:08–01:12:25](https://www.youtube.com/watch?v=5sxHosTLPF8&t=4328s) · Nemotron之后的缺名第二转写猜Omal仍不足确定，保留未辨认。Common Crawl可按域分组再选择配比这一答复可恢复。
### 2026-15 · 音频补核

- [00:20:48–00:21:06](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1248s) · 学生complex...data set原问两模型仍未恢复，讲者说稍后再讲，不能编明确问题。
- [00:26:13–00:26:36](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1573s) · small.en在低音量学生问题处幻觉重复I am not interested，应丢弃；讲者复述为RL为何或可降低幻觉，并明确只是folk story，非证明。
- [00:34:19–00:35:35](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2059s) · 讲者承认无法精确知道行为是否已在pretraining，SFT/RL边界模糊尤其expert iteration。主要区别在监督反馈形式，不能绝对写SFT破坏/RL只激活特征。
- [00:38:40–00:41:35](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2320s) · 字幕prompt误听problem需按loss masking上下文。decay通常强调更高质量是配方直觉，混合比例由小模型ablation外推也可能脆弱/试错，不编一个确定最优混合公式。
- [00:56:31–00:56:56](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3391s) · 高质量标注者无机械gold标准，需要任务正确性/一致性等证据；不能用是否同意judge定义质量。
- [01:02:55–01:03:20](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3775s) · 原字幕70有误，第二转写与成本幻灯片一致是7B小模型历史例，不是70B。
### 2026-16 · 音频补核

- [00:38:18–00:38:49](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=2298s) · 正advantage除长度会偏短，负advantage除长度可能偏长；不把length normalization一律称为鼓励更短。
- [01:10:23–01:11:12](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4223s) · 同权重思考开关可通过prompt/token控制；学生问句部分缺失，不能泛化为所有服务端仅用prompt切换。
- [01:12:42–01:13:02](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4362s) · 学生关于混合训练/蒸馏的问题small.en未恢复，保留字幕与原问缺口。
- [01:14:25–01:15:05](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=4465s) · 跨领域RL并行/顺序及遗忘问句有缺词，不编唯一最优流程。长上下文扩展的数据建议是配方例，不当完整复现实验。
### 2026-17 · 音频补核

- [00:10:55–00:11:42](https://www.youtube.com/watch?v=26FtD08ZpOU&t=655s) · 为何用图文对而非只用图像，可恢复讲者复述；学生原句仍不全。图像自监督augmentation是另一条路线，不是CLIP一定生成文字。
- [00:15:25–00:16:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=925s) · 学生问2D位置编码，讲者对CLIP消融明确用I believe并提醒局限；不写成所有视觉任务1D与2D等效。
- [00:18:25–00:19:09](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1105s) · 狗图像有多个合理caption会产生false negatives；零样本分类与生成caption分开。
- [00:20:25–00:21:05](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1225s) · bag-of-words/语言模型/对比目标的效率比较限所引实验，不是更强语言解码器永远更差。
- [00:23:55–00:24:35](https://www.youtube.com/watch?v=26FtD08ZpOU&t=1435s) · SigLIP负例采样问题原问仍缺词，讲者说初始论文沿用矩阵，不补高级hard-negative方案。
- [00:59:18–01:02:11](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3558s) · 此处VLM理解视频并输出文本，不能当视频生成系统。系统开销包含视频装载/预处理与更多视觉token；学生第二问部分只由讲者复述恢复。
- [01:02:53–01:03:32](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3773s) · 预训练语言模型如何对齐的问题学生原话未恢复，仅保留讲者复述，不能捏造adaptive threshold。
- [01:03:53–01:04:41](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3833s) · 视觉编码器参数通常比LLM小是此类架构一般观察；具体参数需对应原模型表，不写成数学必然。
### 2026-18 · 音频补核

- [00:57:08–00:57:29](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3428s) · 学生问recurrence、data、parameters联合扩展，第二转写未恢复完整问题。讲者说有3D图，所取画面未显示，不能编图。
- [01:00:25–01:01:05](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3625s) · Parcae从头训练与复用预训模型的问答保留讲者复述；small.en把后文troll blog省略成一般判断，优先已上传字幕，不把两模型差异直接合并成新事实。
- [01:03:20–01:04:05](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3800s) · Megakernel主要代价是工程劳动与硬件/模型/batch特化，几模型/一年是讲者估计，不是保证产能或跨硬件自动通用。
- [01:05:55–01:06:35](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3955s) · MXFP4/NVFP4硬件差别与是否loop compute optimal分开；课堂说法不能当AMD只能支持一种格式的排他规范。
- [01:09:55–01:11:05](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=4195s) · small.en把Megakernel/NCCL写作MagiColonel/nickel，应按代码/官方术语纠正。通信可融合仍有collective latency，讲者称初步工作，没有普遍killer use case；学生原问未完全恢复。

## 十一、补取的演示画面

追加136帧均已查看。取点来自互动/演示候选和全文校读笔记；覆盖重要变化，不声称遍历每个视频帧。有些帧只拍到讲者，不能互证其描述的屏幕。完整帧哈希及来源见逐讲对照记录。

- 2025-02 · [00:38:18](https://www.youtube.com/watch?v=msHyYioAyNE&t=2298s)、[00:38:45](https://www.youtube.com/watch?v=msHyYioAyNE&t=2325s)：补取两帧均为2BDK矩阵乘FLOPs推导，CUDA/CPU不同shape分支源码可见，未有测速结果；严格加法数可少一项但通常按2BDK估算。
- 2025-04 · [00:20:57](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1257s)、[00:21:19](https://www.youtube.com/watch?v=LPv1KfUXLCo&t=1279s)：画面为MoE结构与Top-k/hash routing幻灯片，非运行演示；替代MLP与attention heads两支分开，Top2归一化/合并依具体路由。
- 2025-05 · [01:09:41](https://www.youtube.com/watch?v=6OBtO9niT00&t=4181s)、[01:09:57](https://www.youtube.com/watch?v=6OBtO9niT00&t=4197s)、[01:11:12](https://www.youtube.com/watch?v=6OBtO9niT00&t=4272s)、[01:11:36](https://www.youtube.com/watch?v=6OBtO9niT00&t=4296s)：画面前两张为safe/online softmax两段伪代码，后两张为FlashAttention Q/K/V分块与rescale图；完整softmax输出仍要最后归一，而加权V可在线维护O，不把二者混同。
- 2025-06 · [00:10:25](https://www.youtube.com/watch?v=E8Mju53VB00&t=625s)、[00:14:20](https://www.youtube.com/watch?v=E8Mju53VB00&t=860s)、[00:20:55](https://www.youtube.com/watch?v=E8Mju53VB00&t=1255s)、[00:24:02](https://www.youtube.com/watch?v=E8Mju53VB00&t=1442s)、[00:24:25](https://www.youtube.com/watch?v=E8Mju53VB00&t=1465s)、[00:31:55](https://www.youtube.com/watch?v=E8Mju53VB00&t=1915s)、[00:33:56](https://www.youtube.com/watch?v=E8Mju53VB00&t=2036s)、[00:34:14](https://www.youtube.com/watch?v=E8Mju53VB00&t=2054s)：代码sleep(50/1000)为50ms。实际profiler表有CPU self/total与CUDA列及dim128比较；Nsight窗口可见CPU/CUDA/NVTX轨道。画面是原课结果，未本机运行；不将不同列相加为端到端，也不凭截图推导CPU耗时为何下降。
- 2025-07 · [00:25:17](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1517s)、[00:25:42](https://www.youtube.com/watch?v=l1RJcDjzK8M&t=1542s)：画面为ZeRO1步骤图，reduce-scatter按块求和各rank持一块，all-gather更新权重；不是当前机器通信实测。
- 2025-08 · [00:00:25](https://www.youtube.com/watch?v=LHpr5ytssLo&t=25s)、[00:07:45](https://www.youtube.com/watch?v=LHpr5ytssLo&t=465s)、[00:22:35](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1355s)、[00:23:01](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1381s)、[00:32:55](https://www.youtube.com/watch?v=LHpr5ytssLo&t=1975s)、[00:41:05](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2465s)、[00:45:54](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2754s)、[00:46:27](https://www.youtube.com/watch?v=LHpr5ytssLo&t=2787s)、[00:56:21](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3381s)、[00:56:45](https://www.youtube.com/watch?v=LHpr5ytssLo&t=3405s)：25s只拍讲者，不能证明屏幕内容；其余可见allreduce示意、reduce_scatter_tensor代码、TP轴切分与PP recv/send。2787s及3405s明确Backward pass: homework exercise，PP另写未处理overlap，直接证实仅forward骨架；代码画面不等于完整多卡训练。
- 2025-10 · [00:28:05](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=1685s)、[00:32:10](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=1930s)、[01:02:33](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3753s)、[01:02:50](https://www.youtube.com/watch?v=fcgPYo3OtV0&t=3770s)：1685s写可完美重叠/忽略开销的假设，1930s符号latency/throughput输出，不是真实GPU测速。3753s引用Inception演示75对14iterations，3770s为第三方speed-quality散点；迭代数不等于统一硬件时延。
- 2025-11 · [00:29:21](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1761s)、[00:29:46](https://www.youtube.com/watch?v=OSYuUqGBQxw&t=1786s)：采样画面是DeepSeek2024论文标题及讲者，未见muP/WSD具体图；口述笔记仍由字幕支持，不把这两帧当该图视觉互证。
- 2025-12 · [00:33:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=1985s)、[00:37:05](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2225s)、[00:37:29](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2249s)、[00:41:20](https://www.youtube.com/watch?v=x-R5l2HsXqM&t=2480s)：MMLU讲义与HELM实际prompt详情页可见，含计算机题few-shot Answer:D和模型标识；后切MMLU-Pro。显示了查看benchmark样本/提示，非现场训练或完整重算榜单。
- 2025-13 · [00:00:25](https://www.youtube.com/watch?v=WePxmeXU1xg&t=25s)、[00:15:00](https://www.youtube.com/watch?v=WePxmeXU1xg&t=900s)：25s课程标题/数据主线代码，900s Common Crawl概览；这两帧未捕获随机网页内容，不把“计划打开网页”视为截图证实。
- 2025-14 · [00:13:45](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=825s)、[00:16:40](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1000s)、[00:17:52](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1072s)、[00:18:09](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1089s)、[00:25:25](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1525s)、[00:30:50](https://www.youtube.com/watch?v=9Cd0THLS1t0&t=1850s)：DSIR流程图、4个hash bins代码与实际弹出prob=0可见，1072s probs=[0,.4,0,.6]说明此toy零来自未平滑零频桶，不仅可能下溢。1525s/1850s为LID和GPT3质量数据讲义，未捕获hello误分类输出，后者仍由字幕/trace支持。
- 2025-15 · [00:00:25](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=25s)、[00:03:45](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=225s)、[00:04:25](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=265s)、[00:04:39](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=279s)、[00:39:14](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2354s)、[00:39:36](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=2376s)、[01:02:37](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3757s)、[01:02:55](https://www.youtube.com/watch?v=Dfu7vC9jo4w&t=3775s)：前段为目标与SFT人工示范数据流程；2354/2376是RLHF后两阶段流程，3757/3775是CAI response/critique/revision流程，均为幻灯片，无程序运行。采样未见emoji规则/标注者表，相关问答仅依字幕，不夸大图像互证。
- 2025-16 · [00:58:59](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3539s)、[00:59:22](https://www.youtube.com/watch?v=46f2QTDB08Q&t=3562s)：画面为DeepSeek-R1 language-consistency reward的论文摘录，不是现场翻译模型运行；可读性与准确率取舍要保留，不能证明CoT忠实。
- 2025-17 · [00:00:30](https://www.youtube.com/watch?v=JdGFdViaOJk&t=30s)、[00:03:47](https://www.youtube.com/watch?v=JdGFdViaOJk&t=227s)、[00:16:54](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1014s)、[00:20:48](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1248s)、[00:21:20](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1280s)、[00:22:49](https://www.youtube.com/watch?v=JdGFdViaOJk&t=1369s)、[00:43:15](https://www.youtube.com/watch?v=JdGFdViaOJk&t=2595s)、[00:46:25](https://www.youtube.com/watch?v=JdGFdViaOJk&t=2785s)、[01:09:02](https://www.youtube.com/watch?v=JdGFdViaOJk&t=4142s)、[01:12:11](https://www.youtube.com/watch?v=JdGFdViaOJk&t=4331s)：1369s屏幕数5.3229/1.1547与torch.std源码对应，纠正variance命名。2595s明确各位置独立非autoregressive；2785s显示center/normalize代码。4142s日志与4331s训练loss/reward双图可见：mean reward改善不等于每次输出完全排序成功，不能把toy日志当大模型RL基准。
- 2026-01 · [00:17:55](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1075s)、[00:18:15](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=1095s)、[00:51:25](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3085s)、[00:52:10](https://www.youtube.com/watch?v=JuoVZkPBiKk&t=3130s)：1075/1095仅显示agent example trace链接，未展示trace内容。3085s Marin预登记forecast与已测loss图，灰色目前训练中点不当已完成结果；3130s说明API based on previous runs，非现场新训练。
- 2026-02 · [00:17:40](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1060s)、[00:17:45](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1065s)、[00:30:10](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1810s)、[00:30:15](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=1815s)、[01:03:39](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3819s)、[01:04:03](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=3843s)、[01:10:10](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4210s)、[01:10:30](https://www.youtube.com/watch?v=kuYAsz7zspQ&t=4230s)：GPU示意和cuda可用性分支只见源码未GPU结果；1810/1815的two weeks实际公式只乘7天，需×2。3819/3843 backward einsum轴与autograd allclose展示；4210/4230屏幕旧memory公式重复乘bytes，必须按当前修订源码和口头更正，Adam两moment另8B/param。
- 2026-05 · [00:43:05](https://www.youtube.com/watch?v=izZba4UA7iY&t=2585s)、[00:43:37](https://www.youtube.com/watch?v=izZba4UA7iY&t=2617s)：2585仅讲者及非本课白板残留，不收背景白板为课程公式。2617的幻灯片标题MXFP4但底部16元素/E4M3实际更接近NVFP4，需要按OCP/NVIDIA一手规格勘误，不能照抄。
- 2026-06 · [00:51:15](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3075s)、[00:52:50](https://www.youtube.com/watch?v=xnDHaNUvHBg&t=3170s)：GitHub triton_gelu-ptx.txt实见.reg/.param与多条mul.f32/add.f32，证明讲者查看生成PTX；并非SASS或芯片直接执行PTX文本，未证明寄存器实际分配/吞吐。
- 2026-07 · [00:04:27](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=267s)、[00:04:42](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=282s)、[00:20:57](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1257s)、[00:21:27](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=1287s)、[00:40:19](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2419s)、[00:40:37](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2437s)、[00:42:05](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2525s)、[00:42:33](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=2553s)、[01:10:48](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4248s)、[01:11:16](https://www.youtube.com/watch?v=SzpOcwdIL0Y&t=4276s)：267/282明确独立stdout链接与tracing关闭multiprocessing；1257是概念reduce到rank0/allgather，不能当allreduce返回仅rank0。2437实际stdout各rank allreduce[6,10,14,18]，与来源文本一致。1287/2525/2553/4248/4276只拍讲者，未证PP屏幕细节，PP由源码/字幕支持。
- 2026-09 · [00:32:09](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1929s)、[00:32:31](https://www.youtube.com/watch?v=Q15rhEWZPQ4&t=1951s)：LSTM/Transformer图横轴明确Parameters(non-embedding)，纵轴Test Loss；必须修正先前口述笔记称此特定图为计算–损失曲线。图内所测Transformer更低loss，不证明任意模型/配方。
- 2026-10 · [00:12:55](https://www.youtube.com/watch?v=EfM546A79aM&t=775s)、[00:16:25](https://www.youtube.com/watch?v=EfM546A79aM&t=985s)、[00:35:25](https://www.youtube.com/watch?v=EfM546A79aM&t=2125s)、[00:40:10](https://www.youtube.com/watch?v=EfM546A79aM&t=2410s)、[00:55:05](https://www.youtube.com/watch?v=EfM546A79aM&t=3305s)、[00:55:15](https://www.youtube.com/watch?v=EfM546A79aM&t=3315s)、[01:10:48](https://www.youtube.com/watch?v=EfM546A79aM&t=4248s)、[01:11:12](https://www.youtube.com/watch?v=EfM546A79aM&t=4272s)：775s GQA N=K*G注释混乱，以现场口头K组/G每组head更正；985是BF16算术强度BDF/(BD+BF+DF)，非测量。2410公式latency来自假定带宽，不能宣称真实服务延迟。其余MLA/activation画面另见逐帧，不能补未展示的消融。 3305/3315/4248/4272仅拍讲者，不能说已从画面验证MLA消融或activation图。
- 2026-12 · [00:36:00](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2160s)、[00:36:18](https://www.youtube.com/watch?v=JpAxdTWQJxM&t=2178s)：Arena历史榜单截图与谁在投票/偏差/刷票问题代码，非软件运行演示，也不能引用为当前排名。
- 2026-13 · [00:43:54](https://www.youtube.com/watch?v=-qm0ln33G24&t=2634s)、[00:44:16](https://www.youtube.com/watch?v=-qm0ln33G24&t=2656s)：两帧仅讲者，背景白板为其他内容，不收作本课公式；许可/爬虫问答仍依字幕与ASR，图像不增加法律证据。
- 2026-15 · [00:06:07](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=367s)、[00:06:22](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=382s)、[00:26:32](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1592s)、[00:26:55](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=1615s)、[00:45:34](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2734s)、[00:45:45](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=2745s)、[00:53:03](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3183s)、[00:53:17](https://www.youtube.com/watch?v=2oH6PWPrYFo&t=3197s)：SFT示范流程、G-V gap偏好条图和demographics表均为讲义。1592/1615幻灯片被裁切，完整图用PDF核验；不是现场finetune/程序演示。
- 2026-16 · [00:05:45](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=345s)、[00:06:01](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=361s)、[00:11:04](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=664s)、[00:11:19](https://www.youtube.com/watch?v=dIFAi87Ws4E&t=679s)：机器人动画两帧位置不同证实引用视频播放，非本课现场训练。PPO三曲线为已有实验图，左综合目标/中RM/右negativeKL不同；本机未重跑该实验。
- 2026-17 · [00:59:21](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3561s)、[00:59:48](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3588s)、[01:04:55](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3895s)、[01:05:15](https://www.youtube.com/watch?v=26FtD08ZpOU&t=3915s)：3561为VLM benchmark表，其他多拍讲者/被裁切slide；不由此断言视觉编码器具体参数量，需原模型架构表，视频理解不等同视频生成。
- 2026-18 · [00:07:28](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=448s)、[00:07:43](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=463s)、[00:10:36](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=636s)、[00:11:01](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=661s)、[00:54:21](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3261s)、[00:54:46](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3286s)、[00:55:54](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3354s)、[00:56:17](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3377s)、[00:57:19](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3439s)、[00:57:47](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3467s)、[01:02:25](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3745s)、[01:04:47](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=3887s)、[01:08:15](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=4095s)、[01:08:55](https://www.youtube.com/watch?v=9EEm4iMAF5s&t=4135s)：Dan Fu嘉宾段含Austin Silveria署名Lifetime of a Token，应保留section署名，不误当另一个课程视频。SLA图TTFT500ms/TBT30ms是示例目标且有生成工具标记；不得当行业通用指标/实测分位。3354数据参数双扩展，3377/3439 Parcae为140M/370M固定参数及isoFLOP recurrence曲线，指数约.40/.38与tokens.77/.78限该实验。3467/3745/3887只拍讲者，未显示其描述3D联合图。4095结尾Megakernel H100/B200图为给定设置；未现场执行新kernel，也未实现循环块megakernel愿景。

## 十二、可执行讲义的图与数组

312个固定仓库版本的图像来源已取得，按内容哈希去重为231张，全部查看版面和主图；两张GIF另展开全部266帧核对动态步骤。序号以《CS336-讲义图像核验.json》为准。图中小字、所有原始曲线点及所引实验不逐项独立复现。405个复合输出完成有限值/已记录shape检查，另重算BPE、softmax、逐行和、LSH及推理统计等关系；随机权重输出和GPU计时仍是作者缓存。

- 图像 0–7：课程人员/插图与五模块总览；GPT-4报告保留未披露项。FLOPs份额随模型shape变化；emergence阈值指标不能直接证明能力不连续。Stanford例token IDs依tokenizer。
- 图像 8–19：总览旧Transformer图含absolute position/dropout，不可当后续RoPE/SwiGLU精确配方。GA100完整芯片128 SM与A100产品108分开。KV cache保留历史K/V而非不再读历史。FP32/16/BF16位宽正确，FP8有E4M3/E5M2两种例。
- 图像 20–26：stride图对应连续row-major的4与1；transpose可换stride而不搬数据。batch/sequence两轴分开。22为6帧动画，展示单权重前向乘加2 FLOPs及输入梯度/权重梯度各2，共6；不含优化器、激活和非矩阵项。CTA/grid与wave尾部利用率为示意，非禁止同SM多block。融合图减少中间HBM往返，不等于取消全部读写。
- 图像 27–39：node图把跨节点线标NVSwitch是教学简图，应按具体系统区分NVLink域与NIC/RDMA跨节点网。broadcast/scatter/gather/reduce/all-gather/reduce-scatter/all-reduce的输入输出位置已逐图核对；DP切数据、TP切层内维度、PP切层序列。
- 图像 40–47：社交帖吞吐是历史自报且word/line不是token。注意力符号表K为KV头数、G=N/K为每KV头query数。GQA速度横轴组数越大越趋MHA，表中效果限该模型任务，不承诺零损失。prefill/decode瓶颈取决batch/shape。
- 图像 48–55：MLA压缩缓存，表比较KV元素数而非直接字节；CLA跨层共享KV并显示perplexity/每token缓存的取舍。滑窗/空洞/全局模式不混同；S4卷积等价依线性时不变条件，非任意input-dependent SSM。
- 图像 56–63：associative recall按key恢复value；Based/Jamba图为特定混合配方。扩散图是连续embedding模型示例，不代表全部离散扩散。速度散点为历史第三方设置。INT8无浮点指数，示意位色不应暗示有指数。对称量化示例127/5.4约23.52，反量化有舍入误差，图中完全恢复只因十进制显示精度。63首帧近空白但原资产是260帧动画；已逐帧查看，依异常值列及对应权重行分成FP16分支和按向量INT8分支，反量化后两路相加，不把全部矩阵都说成INT8。
- 图像 64–71：LLM.int8表有标准误及小幅正负差，不称量化绝无损失。AWQ根据activation选scale而非永久混精保存1%权重。剪枝成本图token单位不直接等于跨模型FLOPs。speculative拒绝后用归一化正部(q-p)，不是直接q或不归一差值；全接受额外采样1token。加速1.92–2.46倍限原配置。
- 图像 72–77：连续batch及时移除完成请求；分页逻辑序列映射非连续物理块。仍可能有最后块内部碎片。共享prefix要相同计算条件；分叉写使用copy-on-write/refcount，不能让不同续写覆盖共享KV。
- 图像 78–87：R1/Llama4/OLMo/HELM/Arena/Artificial Analysis/OpenRouter均为当年快照，指标、模型版本、推理预算不同。OpenRouter使用量不是全行业份额。Arena有CI和style control；评价危机帖是观点。GPT-2表PPL/BPB/BPC与accuracy不能直接同列比较。
- 图像 88–95：LAMBADA末词预测，HellaSwag候选续写，MMLU多选知识与MMLU-Pro更难选项分开。GPQA Diamond流程含专家复核与非专家检验；HLE难度筛选与2500公共题不等于所有数学研究。HLE低分柱不能与不同任务准确率作同一难度刻度。
- 图像 96–103：IFEval可机器检查格式/约束，非全体帮助性。AlpacaEval length control仍保留judge偏差。WildBench checklist与pairwise/individual不同。SWE-bench示例补丁仍有2失败，不能只看45通过宣布成功。Cybench是受控CTF容器评测，不推成真实网络泛化；subtask提示预算不同。MLE-bench按竞赛提交打分，不直接证明研究自主性。
- 图像 104–116：MLE-bench made/valid submission与medal分开。ARC-AGI颜色变换与成本横轴为评测题例，不推成通用智能总分。安全分类/汽车碰撞为评测覆盖类比；越狱截图只作失效现象。Clio有false positive。污染顺序检验依exchangeability假设；基准本身缺题/歧义会制造错误。nanoGPT speedrun固定loss目标，不是原模型全部任务效果复现。
- 图像 117–130：OLMo2预训3.90T与midtraining832.6B/数学10.7B分阶段；Tulu总源样本23.3M与SFT939344/DPO425145不同。Pile raw825.18GiB与effective1254.20GiB由重复epochs产生。Dolma字节/words/tokens列不能互换。DCLM百分比按原始文档数、最终1.4%是保留量，非仅过滤1.4%。fastText表现限1B-1x预算。
- 图像 131–141：DSIR按原始/目标比率重采样，不保证原始支持外有新数据。LSH候选概率S曲线不是硬阈值；固定签名总长度时band数变化也变每band行数。GRPO原论文迭代reward model步骤不代表所有规则奖励训练都更新RM。toy奖励曲线最高约3.4，不意味着已学会完美排序。
- 图像 142–155：2026课程人员更新；Delphi图十字包含forecast/estimated GPT FLOP equivalent，不能冒充已训练实测点。roofline为上界非保证性能。逐行和示意3+1+4+1+5+9+2+6+5+3=39，越界mask用加法零元；寄存器累加图修正全shared误解。2026节点图明确NVSwitch域与跨节点InfiniBand/Ethernet。
- 图像 156–177：DeepSeek-V4图为滑窗+压缩+稀疏选择，不能以图推导未披露参数。2026榜单也保留课程快照日期，不称今天最新。Terminal-bench示例1过2失败；专家/初级用时分布为特定74任务样本。ARC-AGI-1/2与交互3任务及分母不同。GDPval任务交付物与医疗任务分类只说明评测设计，不提供职业建议。
- 图像 178–191：robots/ToS图含forecast，不将趋势全部当实测。Stack-v2保留PR基底/diff/comment/review结构；公共可见不等于许可统一。CommonPile各源大小用对数轴，Comma柱图不是全项领先Qwen3。PDF绘制指令解释读取顺序缺失。过滤实验需同时看tokens与每源epochs；RegMix表最优5.34为预测，需大模型验证。配比方法是否限制重复训练依表分开。
- 图像 192–199：OpenThoughts 75k问题×16响应=1.2M，不是1.2M独立问题。SWE-smith生成bug与verified tests、SWE-rebench筛选真实任务是不同流程。SWE-Zero execution-free在推理时禁止执行，训练标签仍可来自执行；截图system prompt仅教材被研究对象。表启用execution可改变结果，不能比较不同工具预算后只归因参数量。
- 图像 200–207：CLIP双向交叉熵图/代码使用L2归一、exp(t)为逆温度尺度，非直接乘温度。SigLIP逐pair二元loss示例除n而非n²，改变归一会改梯度尺度。分块跨设备仍需交换嵌入/累积完整目标。LLaVA投影视觉特征输入语言模型；caption/box合成QA不代表文字teacher实际看原图。
- 图像 208–214：熨衣图模型描述含可见错误，不能把展示回答当ground truth。OneVision任意分辨率切块加全图；7290/8748/6272为具体裁剪/帧配置非通用固定token数。3.2M单图、1.6M混合数据分阶段；Stage1只训projector，后续full model，冻结规则不可跨模型套用。
- 图像 215–217：跨图数学示例有算术错误：r=11、角40°，面积应42.2369679而非38.01；不提前舍入总价2660.93，按题逐步两位面积42.24后乘63得2661.12，均非原答案2386.03。GUI多截图描述与足球视频示例为论文展示，未实际控制设备/视频全帧复测。
- 图像 218–227：Qwen-VL旧版3阶段冻结图与Qwen2/3-VL区分。原始5B筛到1.4B为约数，不能由分项近似数推严格总和。Qwen2 native resolution动态token与M-RoPE时/高/宽坐标分开；Qwen3图多层DeepStack视觉注入。预训S0仅merger67B，S1/S2约1T各、S3 100B和262144长度是各阶段预算，不是unique data。结果表含思考模式与不同版本，非严格控制变量排名。
- 图像 228–230：Chameleon统一混合自回归可输出图像token，再经decoder恢复图；与只输出文本VLM不同。VQ-VAE最近码与重建/commitment/codebook损失分开，量化不保证原图无损。鸟类回答是模型展示，未据此验证全部生物事实。

数组和记录的教学处理：

- 2025-lecture_01.json · tokenizer/BPE：字符与字节表示逐例核对。45个counts值是循环的中间字典快照，不能把每一步的1当最终频次；最终merge为th/the/the空格，推理按rank重放。词表前256为单字节，新增256–258拼接可逆。
- 2025-lecture_02.json · tensor/optimizer：切片别名使x首元素变100；转置后reshape顺序与原flatten不同。einsum缩并4维得全4，reduce mean全1/sum全4。随机初始化输出只作数值规模示例；两optimizer state快照约±.01更新，不证明所有后续步同幅。
- 2025-lecture_06.json · benchmark/GELU/softmax：124个times列表是多次benchmark逐步累积，非124独立完整实验。matmul1024–16384耗时非单调线性；batch/dim小shape约6ms平坦受launch/框架影响。GELU手写与融合小例同为.841192不保证所有输入逐位相同。softmax全100第一行1/3，第二行极小非零与1，不能把显示1理解为无穷精度。
- 2025-lecture_10.json · symbolic stats：配置K40→8改变KV头与参数，固定其他shape比较；B256模型内存超过80GB只是假设统计，非单卡成功运行。
- 2025-lecture_14.json · data filtering/dedup/LSH：importance权重归一化后和为1。DSIR示例probs=[0,.4,0,.6]含未平滑零桶，因此两候选权重零；不全归因underflow。set去重不保顺序且Hello!/hello保留，和A4整行重复全删的任务不同。Bloom有false positive而无本例false negative。LSH表按各(b,r)重算。
- 2025-lecture_17.json · RL arrays：reward[4,4,3,4,4]均值3.8；std采用样本修正sqrt(.8/4)=.4472136，normalization含1e-5，故约.4472037非无epsilon值。第三个prompt组全reward3→中心化全零。独立位置log_probs不能称完整自回归排序策略。
- 2026-lecture_01.json · tokenizer/BPE：2026 tokenizer不同使首例token IDs变化；UTF8与码点列表仍一致。三次merge按pair计数重算并可逆解码。不把merge函数入口空new_indices当最终空输出。
- 2026-lecture_02.json · tensor/current code：typed contents内shape已与值递归比较；einsum2D/3D和rearrange全1/全4符合缩并维。深网络y有ReLU全非负；w.grad=[1,2,3]。Adagrad g2非负并含零，不能叫Adam一二阶矩。
- 2026-lecture_02_recording.json · recording code：录制版本输入/权重随机样本与当前不同，不能逐数强行相等。ReLU输出非负、梯度[1,2,3]、einsum缩并/reshape关系一致；分支代码差异另在逐讲说明。
- 2026-lecture_06.json · Triton arrays：全1矩阵/小shape计时仍是原作者缓存。三GELU结果小例一致；两softmax数组一致且每行和约1；row sum[1..4]/[5..8]=[10,26]，不能把这一正确性检查当GPU速度重跑。
- 2026-lecture_10.json · symbolic memory/latency：每个统计条目latency=memory/3.35e12，throughput=B/latency，依完美带宽和忽略开销假设。B64/K40内存79.7GB几乎满，未加临时buffer；K8/B25665.6GB也不保证真实程序必能放入。
- 2026-lecture_14.json · dedup/LSH/mixing：typed set去重同旧版，3组LSH概率与1-(1-s^r)^b一致。混合示例high仅1e8 unique与low1e11不同，固定total若提高high比例须记录重复epochs，不能自动看作增加unique高质数据。

## 十三、其余讲义与作业逐页图表核对

以下范围覆盖其余24份核心PDF的1366页与17份作业的450页；连同第九节L08的73页，共42份1889页。每页版面/主图均已看过。作业的密集文本按所列公式重点交叉读，未声称逐字解决所有题目。若本节纠错与前文初轮课堂摘点不同，以本节明确列出的原始依据和限定为准。

### 2026-lecture-03.pdf

- 第1–9页：结构从原始encoder-decoder转到decoder-only、pre-norm、RoPE和门控FFN。模型总表用于观察课程时点样本的趋势，不把小字截图当闭源模型已证实规格，也不把“大多”改写成架构必须如此。
- 第10–13页：图中pre-norm保持未归一化的残差直通路径；non-residual postnorm与pre-norm后再加norm不等于原始post-norm。BLEU、梯度和loss曲线来自所引实验；可以解释稳定性机制，不能保证所有任务提高相同幅度或免warmup。
- 第14–19页：p14把RMSNorm分母画成sqrt(||x||²+epsilon)，缺少按维数取均值的1/d。教学用sqrt((1/d)sum(x_i²)+epsilon)，并区分可学习gain与bias。p15–17的FLOPs/运行时间说明搬运成本，数字只属于所引模型/实现，不把少0.17% FLOPs当作只能少0.17%耗时。原始依据：https://arxiv.org/html/1910.07467v1 。
- 第20–26页：ReLU、GELU、Swish与GLU的图和公式已对应；Swish=x·sigmoid(x)，SwiGLU还乘另一线性分支。相同参数预算时三矩阵门控FFN的中间宽度约为两矩阵FFN的2/3。表格支持特定实验收益，不保证某激活在所有规模第一；门控也不是模型工作所必需。
- 第27–29页：串行block应展开为h=x+Attn(Norm(x))，y=h+MLP(Norm(h))；p28引用截图的串行简式没有明确保留第二次相加前的attention残差，不当完整实现公式照抄。并行形式的MLP与Attn共用原输入，改变计算图，不只是把原串行程序同时启动。截图15%属于PaLM实验，非普遍加速。
- 第30–35页：正弦相加有内容/位置交叉项；RoPE给Q/K的成对坐标做旋转，使旋转点积只依相对位移。不是把整个语言模型变成无条件平移不变。代码向rotary_emb传value_states以取得形状/设备等，不等于这里也旋转V。二维对子旋转与三维视觉位置轴是不同概念。
- 第36–41页：4与8/3是常见FFN宽度起点，T5的64倍只是特殊配方。p38表列PaLM为4倍，已经是其理论2/3预算换算的例外，不能说所有GLU都必须8/3。50M曲线的宽平区不保证任意规模1–10倍都一样好。
- 第42–43页：头数乘head_dim不必等于model_dim。p43 PaLM行写258、ratio1.48，而作者表说明48头、head_dim256、model_dim18432，故按本页其他行的head_product/model口径应为2/3；反比才是1.5。不能把258或1.48当正确规格。依据：https://arxiv.org/html/2204.02311v5#S2.SS1 。
- 第44–47页：宽深比需看参数、并行和延迟；图中宽平区不是数学最优常数。词表表格是特定版本，不同tokenizer的per-token loss不可直接比。多语言和代码会改变切分需求，不能由“单语”就推出固定词表上限。
- 第48–51页：单遍训练也可能记忆；weight decay可影响优化动态，不能从一组train/validation散点说它绝无防过拟合作用。未报告dropout不等于已证明完全没用，闭源模型尤其不能反推。
- 第52–56页：区分输出z-loss、QK norm与soft-cap。最大化log-likelihood时减正惩罚，最小化NLL时加正惩罚；z-loss为alpha(logZ)²。softcap=c·tanh(logits/c)，范围(−c,c)而非只设上界。QK norm调Q/K不是归一化整个attention矩阵；表中perplexity变化不等于运行速度变化。
- 第57–60页：KV缓存保存历史K/V而非QK完整矩阵。p58–60算术强度公式取整个括号的倒数，并有n相对d等近似条件；prefill与逐步decode的总量不能混用，长到n不再远小于d时不能仍忽略attention FLOPs。
- 第61–63页：MQA的“one dimension”是共享一个KV头，不是每个K/V只有一个标量；每头维度仍可为k。GQA保留多组KV。图中质量/延迟折中只证明所测设置，不许改现有模型的KV头数后默认无需训练或质量无损。
- 第64–67页：图中的稀疏/滑窗遮罩保留不同连接，堆叠可传播信息但不是一层全注意力等价替代。Command A示例区分NoPE全局层和RoPE滑窗层；其他截图是带日期的混合方案，不按未公开闭源实现作确切架构结论。

原始依据：[来源1](https://arxiv.org/html/1910.07467v1) · [来源2](https://arxiv.org/html/2204.02311v5)

### 2026-lecture-04.pdf

- 第1–5页：Attention成本图需同时看n、头维度和硬件。去softmax后(QKᵀ)V=Q(KᵀV)，是结合律；原softmax不能随意这样交换。因果递推S_t只含前缀，不能直接用全序列KᵀV泄露未来；无归一化线性式与核化/带分母版本不同。
- 第6–11页：Minimax、Mamba、Gated Delta等混合图表达了状态压缩与全注意力的配合，不保证任意混合比例都更好。p9的I−βkkᵀ只有相应归一化/β条件才是正交投影；β=0停止新信息注入但γ仍可衰减旧状态，不等于整个状态原样保留。合成回忆/语言曲线分开解释。
- 第12–13页：DSA轻量indexer仍可能给所有位置对打分，不能仅凭主注意力选top-k便宣称总算法严格线性。适配需训练indexer及后续阶段，不是给任意冻结模型加选择器就默认质量无损。评测柱图保留其数据、协议及课程时点。
- 第14–24页：MoE以较少激活参数承载更多总参数，但router、路由通信、存储不免费。OLMoE速度图应分tokens/FLOPs与wall clock，不能只复述7×为通用速度。GPT4图上本就带问号，不能把传闻写成确证。多节点可用性、稀疏负载与训练稳定性共同影响收益。
- 第25–30页：token choice沿专家轴为每token选top-k；expert choice沿token轴为专家选token；global assignment是一次路由匹配优化，不是全训练过程全局最优。hash routing、REINFORCE和BASE匹配是不同路线。k值随模型版本及是否计共享专家变化，不按“DeepSeek7”这类混称写固定规则。
- 第31–35页：top-k前softmax与只在选中集合归一化不同。p31将DeepSeekV3概括成top-k后softmax不准确：V3用sigmoid affinity再对所选分数除和，并非对所选原logit做softmax。细粒度/共享专家消融并不一致，p33有收益、p34没有共享专家收益，不能只留单方。
- 第36–39页：离散top-k索引本身不光滑，不表示所选gate权重不能反传。p38 Gaussian加法噪声与p39 uniform乘法jitter不同；KeepTopK对未选项填−∞再softmax，不能填零当同义。噪声有时改善稳定性，但不是最佳方法保证。
- 第40–43页：Switch辅助损失αNΣf_iP_i中f_i视为离散统计常量，偏导对P_i为αNf_i；再传播softmax会耦合其他专家。V3的路由偏置只影响选中，输出权重用未加偏置的分数；仍有sequence-wise辅助项。去负载均衡图需同时看模型loss和负载图，不只看训练曲线。
- 第44–47页：专家并行按expert组织变长token组，MegaBlocks是块稀疏矩阵乘，不等于硬件2:4稀疏。LatentMoE下投影以降低路由字节，同时改变结构。容量约束和丢token可令输出依同batch其他请求；dropless移除这一来源，不保证所有数值/调度完全确定。
- 第48–50页：router FP32和router z-loss针对数值与训练稳定性；不能把输出层z-loss、负载均衡loss混成同一个东西。小微调集可能过拟合，只训练部分dense MLP与增加数据是不同经验方案，不能保证彻底解决。
- 第51–56页：Upcycling先复制已有权重再继续训练，并不自动产生分工。p56页题DeepSeekV3，副标题误写V2；Fine-grained(258)与p35的256冲突，作者报告为256 routed、选8、另1 shared，不能把258继续写入教材。表中专家数量须分routed/shared/总激活。
- 第57–60页：MLA用latent压缩KV，带RoPE的额外key分量也要缓存；不能只缓存latent后删掉全部位置分量。矩阵吸收取决于相对旋转与投影能否交换。MTP训练目标与推理时speculative使用是两个环节，原V3例子仅加一步预测，不是所有示意多步模块都实际运行。

原始依据：[来源1](https://arxiv.org/html/2412.19437v1#S2.SS1.SSS2)

### 2026-lecture-05.pdf

- 第1–8页：矩阵性能曲线把算术强度、tile与wave分开；计算增加常能降训练loss是经验关系，不是硬件一升级所有能力必增。CPU低延迟与GPU高吞吐是设计取向；tensor-core峰值与普通算术峰值不能跨dtype随意比较。
- 第9–14页：p9明确是full GA100设计128 SM，不能误判这张图有错；实际A100产品108 SM，p48用108正是产品例子。必须区分芯片完整设计与启用单元。CUDA local memory逻辑线程私有、物理可在设备全局内存；普通跨block交换通常经global，但现代cluster shared等扩展也使“只能global”不宜绝对化。TPU对应表是特定型号类比，不是架构单元一一同义。
- 第15–24页：warp调度隐藏延迟不等于线程没有成本。roofline吞吐上界=min(计算峰值,带宽×AI)，计算与搬运时间的max是理想重叠下界。分支若warp内分歧可串行执行路径，统一分支不必两侧都跑。低精度还要算scale、转换和数值误差。
- 第25–26页：p25把8 bytes/FLOP、4 bytes/FLOP称intensity，实际上是常用算术强度FLOP/byte的倒数。按其简化ReLU账本应为FP32 1/8、FP16 1/4 FLOP/byte；写回仅负值只适用于相应原地实现，普通独立输出仍需写全部元素。归约、累加、状态不一定随输入一起低精度。
- 第27–29页：p27–28 MXFP8每32值配E8M0 scale并需处理转置重分组。p29虽写MXFP4，却给每16值E4M3 scale，这是NVFP4式特征；OCP MXFP4规定32值/E8M0。NVFP4还含全tensor FP32第二级scale。两者E2M1元素范围相同但scale方案不同，不能混名或只记4bit。
- 第30–36页：融合工厂图减少中间张量回HBM和launch，并非所有运算被代数消掉。sin²+cos²例子教学看5算子融合，不假定编译器总直接返回1。三sigmoid重算图8→5是简化HBM读写次数，增加算术且依正确重建反向依赖，不是通用5/8内存或速度保证。
- 第37–39页：coalescing按同warp同一访存指令各lane地址观察；单线程沿行连续不保证全warp合并。图中4字节burst是玩具，真实事务/sector依架构和访问宽度；不要求所有线程同一个元素，也不能把所有coalesced访问限定为单个128-byte事务。
- 第40–44页：tiling沿共享维度分阶段加载并累积输出；N/T访问公式忽略cache等。边界mask、共享内存、寄存器、并发block与对齐共同限制tile，不能只选越大越好；不整除也可以mask/换内核，padding是办法之一。
- 第45–48页：画面确认nanoGPT示例50257→50304（64的倍数），避免旧字幕漏位。物理padding与扩充有效token集合分开；若要求原概率完全不变需处理额外logit。1792→1793配256×128 tile从98→120块，在108SM且设定驻留条件下出现新wave；不推成所有尺寸增加都翻倍。
- 第49–55页：p51把attention写为3次矩阵乘不精确：核心QKᵀ与P·V为2次，Q/K/V投影另计。FlashAttention用分块与在线softmax算精确attention；新最大值出现时分母l及加权V分子o都需重缩放。p50显示FLOPs可从66.6G升75.2G而HBM下降，不代表所有shape相同倍速。课堂未展开完整backward，作业A2保留该知识而不强迫大作业实现。

原始依据：[来源1](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) · [来源2](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) · [来源3](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html)

### 2026-lecture-09.pdf

- 第1–6页：Scaling是资源分配预测工具，不是只背模型配置。有限假设类上界需IID样本、假设数和置信度等条件；图中的上界不等于实际loss，也不证明神经模型严格达到同一指数。
- 第7–16页：历史学习曲线说明可拟合的渐近形状与多个函数族；不是所有数据范围都严格幂律。若L=L∞+aD^−α，严格线性的是log(L−L∞)对logD；熵底明显时不能仍声称logL永远直线。更多数据的期望最优风险与某次有限训练得到的loss也需区分。
- 第17–20页：均值估计在IID有限方差时MSE=σ²/n，标准差σ/√n。p19二维网格例可得n^−1/2，但所写d维n^−1/d不是一般非参数最优率：Lipschitz回归的局部均值常有bias²约h²、variance约1/(nh^d)，平衡得n^−2/(d+2)。须说明平滑性/风险定义；intrinsic-dimension图是候选解释，不是已证明LLM指数。
- 第21–24页：混合数据改变截距而不改斜率只是所引设置；DataDecide与mixing-law预测是两种选配方方法。重复数据的有效量曲线有饱和与拟合常数；“4 epoch近似新数据”是该实验观察，不是通用许可。来源权重、唯一token和重复次数分开。
- 第25–27页：p25把scaling law叫lower bound容易误解：已达到loss是未知最优可达loss的上界，经验下包络也不是不可突破的理论下界。图里正则化/ensemble能优于某标准配方，并不违背数学定理。过滤随预算变松与重复收益变化需联动看。
- 第28–36页：比较架构/优化器时检查预算、数据、总参数与非embedding参数；形状相近的loss曲线不保证下游或极大规模同排序。embedding与非embedding对计算和表达作用不同，MoE还须分总/活跃参数。1层模型对比不等于深度永远无效。
- 第37–39页：Critical batch取达到同一loss所需steps S与examples E的折中，Bcrit=Emin/Smin。图式平衡时S≈2Smin、E≈2Emin，不是只用少量steps衡量效率。目标loss降低时示例Bcrit增大；实际还看吞吐、显存和优化效率。
- 第40–42页：μP同时规定初始化、分层学习率和输出缩放；图示宽度迁移有参数化条件，不能把不同层LR或variance/SD混用。训练loss可预测不保证所有下游指标单调，p41两图明确显示差异。
- 第43–48页：固定训练预算常用C≈6ND，IsoFLOP扫N时相应D=C/(6N)；不能把6漏掉或把D也固定。Kaplan主张N∝C^.73、D∝C^.27，不能口述反转。三方法是训练曲线包络、固定FLOPs扫描及联合曲面拟合，控制训练日程后才比较。
- 第49–53页：p49“least squares”是宽泛概括；Chinchilla方法3实际对预测/观察的log loss最小化Huber，并用L-BFGS和多初始化。残差均值必为0不适用于这类任意拟合。所引复核质疑拟合和区间，课程置信数字不作为自然常数。依据：https://arxiv.org/html/2203.15556v1#S3.SS3 。
- 第54–57页：训练最优与服务总成本最优不同，重复使用多时可偏向小模型多token预训练；“overtrain”在此是相对训练计算最优多训，不等于验证loss已上升的过拟合。20tokens/param与其他模型比率均带配方/年代，不能用它们决定所有模型。

原始依据：[来源1](https://arxiv.org/html/2203.15556v1#S3.SS3)

### 2026-lecture-11.pdf

- 第1–12页：公开训练案例是带日期的配方。MiniCPM同时调整初始化、embedding、输出与深度缩放；9M至499M的迁移再到2.4B是有限外推，不能只改初始化就叫完整muP。
- 第13–18页：WSD复用稳定阶段前缀，再从不同checkpoint做衰减分支，仍有各分支成本。中间cosine checkpoint不等于为短预算专门衰减后的终点，不能说任何提前停止都无效。p18 Ultratext面板95.60与论文正文平均192统计范围不同，保留口径，不把20或192当普遍最优比例。
- 第19–30页：DeepSeek用M=每token计算量、C=MD，M不等于参数量N。bits/byte与per-token loss不能直接横比。LR/批量二维图的平坦谷底不保证唯一最优值；Qwen、Kimi、Hunyuan和MiniMax图需分总/活跃参数及架构。
- 第31–41页：StepFun图中的U形截面不能证明所有网络联合凸优化。batch与数据量的关系、优化器相对收益会随预算和配方变；AdamW和Lion应分别调LR与weight decay。拟合很好仍可在更大规模失败，held-out检验是必要证据。
- 第42–45页：Muon对带动量矩阵做近似正交化，Newton–Schulz有限步是近似；UVᵀ与矩形/秩亏矩阵条件需要说明。nanoGPT速度赛与大型Kimi训练不是同一控制实验，不能据单次成功宣布处处最优。
- 第46–50页：每坐标O(1)与向量范数O(sqrt(width))不同。谱范数乘输入范数是上界，p47的约等于不能视为任意向量恒等式。深线性、固定深度、单样本rank-1 SGD是推导假设；Adam与SGD分层LR不同，初始化方差与标准差也要分开。
- 第51–58页：muP经验迁移需明确宽度、头维度、输出缩放与gain参数化。该研究RMSNorm gain失败不等于所有RMSNorm都禁用；Lion用sign而非sine。粗网格最低点与连续最优值不同，一次10B验证支持该配方，不能保证所有深度/批量都可迁移。

原始依据：[来源1](https://arxiv.org/html/2404.06395v1)

### 2026-lecture-15.pdf

- 第1–9页：从续写到指令控制，SFT是下一token训练目标配合不同数据。三阶段RLHF图是经典路线，不能说所有当代模型都严格依此顺序；闭源数据报道不当成已公开可核验配方。
- 第10–18页：FLAN短任务、Alpaca、OpenAssistant长回答及工具调用数据目标不同。长度/列表偏好会混淆人类和模型评审，偏好胜率不等于事实或推理能力；表格含各数据源带来的提升和退化，不能只留平均分。
- 第19–27页：示范引用可能教格式也可能灌入未知事实，需核验引用本身。SFT会增加幻觉是特定知识学习实验的风险，不是所有新知识都绝对学不会。500条安全样本的图只支持所测分布；同时看违规率与误拒率，不能宣称少量数据保证全面安全。
- 第28–30页：图中梯度训练循环需正确设置labels和prompt/response mask；混入指令数据的midtraining有助减少分布突变，但不能保证完全无灾难遗忘。饼图是MiniCPM等特定混合方案，不作为所有公司已证实做法。
- 第31–44页：生成与验证能力不同，偏好数据要交代标注规则、采样模型和人群。标注分歧、风格偏见、劳动条件和专家成本影响数据；单个平台调查不能代表全球从业者，新闻工资数字不作当前行情。
- 第45–48页：模型评审在系统平均分上高相关，不等于每条回答判断准确或没有自我偏好。Constitutional AI先生成批评/修订，再收集偏好和训练；AI反馈仍依赖初始模型、原则和验证数据。长度增长可能是代理奖励偏差。
- 第49–54页：policy gradient需要对采样分布求导，奖励固定时可写score-function估计。PPO的clip是代理目标，不是把参数或概率一律截断，也不是严格KL约束保证。KL到固定参考模型与PPO到旧策略的ratio分开。
- 第55–60页：DPO从KL正则目标在非参数策略假设下得到pi*=pi_ref exp(r/beta)/Z，再用Bradley–Terry配对似然。Z(x)在同prompt比较中抵消；不是对好样本普通SFT、坏样本无限减概率。SimPO移除参考模型同时加长度归一和margin，不能说完全同一个目标。
- 第61–65页：PPO/DPO排序取决于数据、训练预算和调参。代理奖励持续升而人类偏好下降是过优化证据；不能把所有奖励源都画成同一退化曲线。p64“no longer probabilistic models”过强：仍输出概率分布，只是不自动校准，置信度/熵和多样性可能变化。

### 2026-lecture-16.pdf

- 第1–10页：可验证任务减少偏好奖励噪声，但测试覆盖不全仍可被钻空子。PPO图需区分固定参考、采样旧策略、当前策略和值模型；LM每token为动作与整回答bandit视角均可，不能混用时序奖励。
- 第11–16页：外循环采样、内循环多次更新期间旧logprob和优势保持固定。p14代码对logprob−ref_logprob取非负截断是该实现的奖励塑形，单样本log-ratio可负，不等于真实KL必为负或这样截断保证KL不发散。GAE含终止mask，gamma=lambda=1也依有效序列边界。
- 第17–24页：GRPO移除critic，以同prompt多次采样算组内均值与标准差。除std会改变问题权重，零方差组需epsilon或跳过；含自己的组均值也与动作有关，不能仅去std就无条件宣称对原目标完全无偏。长度归一使正/负优势的长短回答权重不同，Dr.GRPO改变这些项，非简单“修一个bug”。
- 第25–38页：R1-Zero长CoT和aha例子不能单独证明新能力涌现，底模也可有类似文字。R1加入冷启动SFT、语言一致性及后续通用训练；800k蒸馏是教师输出数据，非学生再次运行同规模RL。PRM/MCTS在该研究失败不等于所有任务永久无效。
- 第39–48页：Kimi K1.5基于参考答案的奖励、策略正则和长度奖励与GRPO不同。p41称只保留best-of-8全失败题过强，原文用估计pass rate排除太简单并作课程采样，不能据此断言全部题8次全错。p43错误答案的min(0,lambda)是长答案受罚、短答案不额外奖励，不是强制长度低于中点；相同长度时分母为零需实现保护。
- 第49–54页：Qwen3四阶段图含推理RL、thinking融合和general RL。3995指所述RL数据，不是全部训练只见3995条。thinking budget控制不等于最佳答案有正确性保证；stage4有部分数学指标下降，保留收益和代价。
- 第55–61页：Qwen3-Coder-Next把仓库数据、合成轨迹、专家蒸馏和自动构造环境串联。agent RL图里的git取回答案属于奖励投机，曲线跳升不能当能力提升。评测需环境隔离及防泄露；工具格式SFT与真实多步工具成功率是不同指标。

### 2025-lecture-03.pdf

- 第1–19页：逐页图示与2026架构讲义对照：pre/post norm及残差路径一致；p14 RMSNorm同样漏1/d，需按均方根修正。p16统计归一化FLOPs很少但耗时25.5%说明搬运限制，非所有设备固定比例。
- 第20–35页：激活、GLU、串行/并行与RoPE图均保留。p28串行简式未显式保留attention残差，按两次残差完整式教学。RoPE只在相应Q/K坐标旋转，p35传value_states不是旋转V；15%并行加速非普遍保证。
- 第36–52页：FFN和宽深比曲线是经验宽平区。p43 PaLM head dim258及ratio1.48延续表格错误：主文48×256/18432=2/3。词表大小历史样本不当硬规则；单遍仍可记忆，weight decay优化作用不排除防过拟合。
- 第53–58页：z-loss最大化写减惩罚，最小化NLL写加惩罚；QK norm、输出z-loss与soft-cap不同。soft-cap给上下界而非仅上界；loss尖峰与下游质量分开。
- 第59–68页：prefill与decode成本分开，算术强度取整个括号倒数。MQA共享一个KV头非一维标量，GQA为折中。滑窗层扩大可达感受野不等价任意全注意力；闭源GPT4稀疏说法不作确证。

原始依据：[来源1](https://arxiv.org/html/1910.07467v1) · [来源2](https://arxiv.org/html/2204.02311v5)

### 2025-lecture-04.pdf

- 第1–14页：MoE总参数、激活参数、wall clock分开，router和通信非零成本；7×与2×图各有自己的基线，不能混用。GPT4旁有问号，未知结构继续标未知。
- 第15–24页：token choice、expert choice与global assignment沿不同轴选择；shared expert收益两篇消融不一致。p19说V3 top-k后softmax与实际sigmoid后除和不符。p23正确列256 routed/8 active/1 shared，供后页纠错。
- 第25–31页：Gaussian加法噪声与uniform乘法jitter不同，top-k填−∞。辅助loss的f_i为离散统计，P_i可反传；V3路由偏置不进入输出权重，仍含sequence辅助项。负载图与loss一起看。
- 第32–40页：MegaBlocks图是块稀疏，不是2:4硬件稀疏。容量丢token造成batch依赖，非所有随机性都来自MoE。router FP32、z-loss、小集微调和upcycling均有条件，不自动保证稳定/专业分工。
- 第41–47页：p43标题V3却写V2(671B)和258 routed，应为V3/256；其公式给sigmoid后归一，文字Sigmoid+Softmax不准确。MLA还缓存RoPE key分量，MTP训练与推理speculative使用分开。

原始依据：[来源1](https://arxiv.org/html/2412.19437v1)

### 2025-lecture-05.pdf

- 第1–18页：GPU/CPU、层级存储、warp/block和TPU图与2026基础一致。p9 full GA100128SM与p44 A100108SM是设计/产品区别，不是矛盾。峰值tensor FLOPs需匹配dtype；内存延迟数字属所引设备。
- 第19–25页：roofline同时约束计算与带宽；p24用bytes/FLOP命名intensity是常用AI的倒数，FP32/FP16例应为1/8、1/4 FLOP/byte。warp内分歧与统一分支不同，低精度累加未必也低精度。
- 第26–40页：融合减少HBM与launch，重算增加算术减少保存。8→5是玩具访问账本。coalescing按同warp同条指令地址观察，不是单线程沿行连续；tile受寄存器/共享内存/边界mask限制，padding不是唯一方案。
- 第41–45页：nanoGPT50257→50304由图确认。1792→1793 tile98→120与108SM说明wave边界，仅在给定驻留条件下解释性能。多加一个维度不能推成通用减速/加速。
- 第46–51页：核心attention是QKᵀ、P·V两次矩阵乘，p47所谓3次混入投影；QK图画三头非三次核心乘。online softmax更新m/l/o，最大值变时分母及分子均重缩放。FlashAttention精确但浮点顺序可有差异，图示速度不作本机复现。

原始依据：[来源1](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)

### 2025-lecture-07.pdf

- 第1–10页：collective输入输出图逐一检查：all-reduce可分reduce-scatter加all-gather；p8文字Reduce应理解all-reduce而非只有root保留结果的reduce。带宽最优不代表所有小消息/拓扑最快，GPU网络表为特定系统。
- 第11–25页：DDP完整复制静态状态，ZeRO1/2/3依次切optimizer/grad/param；16B和12B是各自精度假设，包含临时聚合、激活及工作区才可判断实际能否装下。2N/3N通信式省略(P−1)/P和字节宽度，“free”只指理想带宽量，不是零开销。
- 第26–35页：data parallel批量扩展有统计边界。流水图bubble/useful=(stages−1)/microbatches，非占总时间比例；zero-bubble权重梯度可延后但有依赖/内存/优化器边界，不能任意永远拖延。
- 第36–48页：tensor并行列切/行切与f/g前后向collective对应；序列并行切逐token算子，不把attention上下文断开。激活式34、10、24是所引架构/精度账本，GQA、Flash、SwiGLU会改常数。p47合并DDP/ZeRO1为no scaling忽略ZeRO1状态收益，应仅说完整参数仍复制。
- 第49–60页：TP8和3D配方受8GPU节点拓扑限制，非永久最优。p52图注64A100 GPUs而底部说64machines，按64GPU例理解。p58 Llama故障表原论文百分比自身不一致，不继续算成精确占比；Llama3配置各轴相乘成立，其他EP/CP组未必独立。

原始依据：[来源1](https://arxiv.org/html/2407.21783v1)

### 2025-lecture-09.pdf

- 第1–18页：有限类上界、历史曲线与均值MSE图分清理论上界与实际loss。幂律带熵底时log(L−L∞)才线性；不是所有经典模型任何误差都1/n。
- 第19–25页：p19二维网格可得1/sqrt(n)，直接泛化n^−1/d不普遍，须加入平滑性和bias/variance平衡。重复数据图的4epochs是所测配方，混合数据只改截距也是经验结论。
- 第26–39页：同参数/同FLOPs是不同对比，embedding不能与非embedding混计。临界batch同时约束steps和examples。muP初始化variance与STD不同，宽度迁移不保证下游指标相同排序。
- 第40–49页：联合拟合与IsoFLOP需C≈6ND且日程匹配，不能扫模型时固定D又说同预算。p48 least squares简化实际Chinchilla log-loss Huber拟合，不套OLS残差必均值0。p45区间来自原作者，不当精确自然常数。
- 第50–53页：服务成本最优可多训较小模型，overtrain是相对训练最优，非验证集过拟合。扩散和自回归曲线各按原预算与NLL口径，不能据一组图普遍断言哪类更好。

原始依据：[来源1](https://arxiv.org/html/2203.15556v1)

### 2025-lecture-11.pdf

- 第1–9页：CerebrasGPT以40M代理调参、迁移至111M–2.7B，图支持其设置更平滑，不证明任意数据/深度同样稳定。embedding/output尺度、初始化及分层LR一起定义muP，表格是variance而非STD。
- 第10–25页：MiniCPM将宽度等配方联动，WSD共享稳定前缀再单独衰减。p17不能字面理解必须全部从头训练；p23 Ultratext95.60与p24平均192保留不同统计口径，不互相覆盖。重复衰减分支仍花计算。
- 第26–36页：DeepSeek直接拟合LR/batch，近最优0.25%条件与拟合不确定性均保留。p30文字两次10%衰减并非原图：80%tokens降至31.6%LR，90%降至10%。M为FLOPs/token非参数量，Hunyuan active ratio不与dense总参数混比。
- 第37–43页：深线性rank-1 SGD推导含最坏方向与不抵消假设，谱范数乘向量范数是上界并非普遍等式。Δloss的一阶近似需小步长，Frobenius与谱范数只在相应rank-1处一致，AdamLR与SGDLR不同。
- 第44–55页：固定depth24、headwidth128的宽度实验不等于深度迁移已证实。RMSNorm gain、Lion和weight decay的失败限于该配方；coarse LR网格与10B单次外推有实际范围，不说muP一劳永逸。

原始依据：[来源1](https://arxiv.org/html/2404.06395v1)

### 2025-lecture-15.pdf

- 第1–14页：SFT数据图、指令例子与课堂标注演示保留。p14示例回复对CS336泛泛而谈甚至列数据库/图形学，正是事实性与风格评审分开的反例，不能作课程介绍抄入资料。p13表单与课堂回复表缺少可归档原始结果时标缺口。
- 第15–29页：长度/列表偏好和实际任务质量不同，错误引用、未知事实、误拒Python process的图均保留。500条安全样本收益受原测试集限制。midtraining混合可能缓和遗忘，非绝对不会遗忘。
- 第30–44页：经典RLHF数据流由示范到比较再优化；p32成本是AlpacaFarm示例预算，不作当前市场报价。InstructGPT筛选40标注者是历史实验，Bard指南标allegedly，群体偏好/劳动问题不因自动反馈而消失。p40第二标注表单结果未归档不编造全班分布。
- 第45–54页：AI评审系统级相关不代表每例可靠，Constitutional AI的批评修订与偏好训练阶段不同。PPO ratio相对旧策略，KL参考固定SFT，两种约束不能合并成同一项。
- 第55–62页：DPO由KL正则最优策略重参数化及配对似然导出，Z同prompt抵消；SimPO还改长度和margin。p59 DPO人类胜率空白，不能把其模拟46.8%当人类实测；p60历史榜单不是今天DPO统治所有开源模型的证据。
- 第63–66页：过优化图同时显示无噪声模型反馈没有相同下降，避免挑单图概括。p65“不再是概率模型”应改为仍输出概率但可能失去校准，mode collapse与校准是相关而不同的问题。

### 2025-lecture-16.pdf

- 第1–13页：续讲DPO：p7模拟PPO/DPO46.8%相同，但DPO人类胜率空白。p13概率校准退化不等于不再输出概率；SimPO与length-normalized DPO各自目标不同。
- 第14–28页：RLVR与PPO数据流按每token动作解释，固定参考模型和旧策略分开。p26非负截断sample logratio是AlpacaFarm实现特例，不等于KL估计必如此，也不保证防发散。GAE与terminal mask、padding/EOS边界都重要。
- 第29–36页：GRPO组内z-score移除critic，但std改权重，组均值包含自身也影响无偏性。Dr.GRPO去std和长度归一不应称任何设置都严格无偏。零方差组需数值保护；组内全对/全错无相对优势。
- 第37–50页：R1-Zero、R1、蒸馏区分：长回答/aha展示不是新推理能力充分证明；800k教师数据不是蒸馏学生同样跑RL。PRM/MCTS未在该研究成功，不等于概念永久无用。
- 第51–60页：Kimi参考奖励、课程采样、长度控制和RL系统图逐项对照。p53“只保留best-of-8全失败”超过原文能支持的范围；p55错误答案长者受罚短者不奖励，不是强制中点以下。环境转移/partial rollout需保证采样策略版本。
- 第61–67页：Qwen3强弱蒸馏、thinking融合及general RL各阶段效果不同，3995是特定RL数据规模。测试时更多token有经验收益但非单题正确性保证；部分数学指标在general RL后下降也保留。

### 2024-lecture-03.pdf

- 第1–18页：2024作业架构为绝对位置embedding/GELU，与后年RoPE/SwiGLU不同，保留版本区别。p13 RMSNorm漏均值因子1/d；pre-norm残差图及FLOPs/耗时占比不能误作同一指标。
- 第19–34页：激活、门控、串并行和RoPE图逐页对照。p27串行简式漏显式attention残差，同后年修正；旋转Q/K二维对子，value_states入参不代表旋转V。
- 第35–50页：FFN比例和宽深比为经验区域，p42 PaLM258/1.48按原论文256/2/3修正。单遍数据仍可记忆，p49“不是防过拟合”仅是所引优化观察，不能泛化。
- 第51–58页：BPE自底向上合并，unigram由大候选集拟合再剪枝；SentencePiece是库不是与BPE互斥算法。p51/58“都可逆”需限定预处理后的文本：NFKC、空白合并或UNK会丢原信息。p55的上标数字归一丢格式就是反例。空白/单数字切分影响代码与算术，不是越短token总越好。
- 第59–68页：KV缓存、MQA/GQA与稀疏mask图保留。p60算术强度的倒数要覆盖整个和；MQA共享KV头非标量。滑窗多层有间接长程路径，非一层全注意力等价。

原始依据：[来源1](https://arxiv.org/html/1910.07467v1) · [来源2](https://arxiv.org/html/2204.02311v5)

### 2024-lecture-04.pdf

- 第1–16页：tokenizer可逆性必须排除有损normalization/UNK；attention AI式取整体倒数。MQA单头仍有head_dim，图示GQA质量/延迟仅在原设置成立。
- 第17–27页：MoE在训练时就稀疏，增加总参数仍增加存储、router与通信成本。Mixtral/DBRX/Grok吞吐图受并发及硬件条件影响，GPT4问号仍标未知。
- 第28–39页：三路选择图按不同轴，top-k前/后归一不同。p29 DeepSeek7的混称不能跨版本沿用，原DeepSeekMoE区分routed与shared。离散索引不可导不等于gate没有梯度；Gaussian噪声和乘法jitter不同，辅助loss需把f_i作离散统计。
- 第40–49页：expert parallel通信与MegaBlocks块稀疏图不同于硬件结构稀疏。容量丢token会让请求受同batch影响；FP32 router与z-loss缓解数值问题非保证。p48权重相关仅是Mixtral upcycling猜测，课程自己说未知，不把猜测补成已证实流程。

### 2024-lecture-05.pdf

- 第1–18页：CPU/GPU层级、warp/block、full GA100128SM图已核对，A100108SM是启用产品规格。存储延迟/价格倍数保留为历史例，不能跨设备当常数。
- 第19–31页：roofline用FLOP/byte，p23却标8/4bytes/FLOP应取倒数。混合精度不意味所有累加低精度；融合5算子与重算8→5是HBM账本，非通用速度承诺。
- 第32–44页：coalescing按同warp地址，tile需考虑mask/存储/occupancy。p43分母误写246，按题设256×128及98tiles应为256；1793变120块、108SM例有驻留条件。nanoGPT50257→50304确为64对齐。
- 第45–50页：p46“三矩阵乘”不精确，核心两次加softmax、投影另计。online softmax需更新最大值并缩放旧分母/输出分子；FlashAttention精确注意力而非稀疏近似，原图性能非本机复现。

原始依据：[来源1](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)

### 2024-lecture-07.pdf

- 第1–12页：网络图及collective结果核对，p8文字Reduce指all-reduce才与全员输出图一致；忽略启动延迟的带宽最优并非任意消息最佳。DP的SGD sum/mean与学习率约定需统一。
- 第13–24页：16B与12B参数状态账本不同，ZeRO分片仍需临时all-gather与激活。2N/3N是省略dtype与有限P系数的通信量，阶段1/2“免费”不等于无延迟/管理成本；stage3仅静态分片可线性缩，不保证实际峰值。
- 第25–36页：pipeline微批调度有bubble，(p−1)/m是idle/useful不是总时长占比。tensor列切/行切与前后向collective对应，适用低延迟网络；不能说节点外一律禁止TP。
- 第37–44页：激活峰值图不同于静态状态，34/24/10常数对应旧架构和存储假设；SP切逐token算子，attention仍需跨完整上下文。p43将DDP/ZeRO1都写无显存扩展过粗，ZeRO1优化器状态有收益。
- 第45–54页：GPT-NeoX引文TP2、PP4而页面概括TP4、PP2，虽乘积相同仍按引文区别，不能把交换轴当同配置。Falcon stage1标probably继续未知；p52图注64A100GPU，底部64machines是笔误。TP8最优来自特定集群，不泛化。

### 2024-lecture-09.pdf

- 第1–19页：理论上界、实际经验曲线与熵底区分。均值估计MSE=σ²/n；p18非参数n^−1/d仅二维例吻合，不能任意d直接推广，应说明平滑性并平衡bias²/variance。intrinsic dimension仍为解释假说。
- 第20–30页：Ye数据混合图保留seen/unseen mixture及模型/steps外推三个轴；混合只改截距、4epoch近似新数据是所引实验，不保证任意分布。架构对比同参数与同计算各有含义。
- 第31–42页：critical batch看examples/steps共同预算，muP不只初始化，variance与SD分开。训练loss与下游分数不必相同排序；联合loss(N,D)拟合需熵底且C≈6ND。
- 第43–52页：三Chinchilla方法含日程选择和固定FLOPs扫模型，p47 least squares并非精确算法名，原方法3 logloss Huber拟合。所引复核质疑不等于所有原数据全错。服务总成本最优可偏小模型多训练，overtrain不必是过拟合。

原始依据：[来源1](https://arxiv.org/html/2203.15556v1)

### 2024-lecture-10.pdf

- 第1–10页：封面CS324为沿用模板，不改成另一门课。API任务预算/重复查询规则是当年作业接口，未运行不可称现在可用。Cerebras40M代理的随机搜索与111M–2.7B迁移有范围，完整muP含分层LR/输出尺度。
- 第11–26页：MiniCPM的9M–499M sweep、2.4B外推和WSD分支图保留。p18所有配置必须从头训练太绝对；稳定前缀可复用但衰减尾不免费。Ultratext95.60面板与正文平均192需分范围，训练tokens/param不作通用常数。
- 第27–34页：DeepSeek近最优0.25%筛选后拟合，LR噪声与平谷底不能给精确最优。p31两次10%是tokens阶段长度，不是每次LR乘0.1：80%tokens至31.6%，90%至10%。M=FLOPs/token与N参数量不同。
- 第35–47页：宽度实验固定depth24/headwidth128，非完整depth/width联合证明。表格初始化variance与STD不同，RMSNorm gain/Lion/weight decay失败限于所测配方，10B一次结果不保证所有设置迁移。

原始依据：[来源1](https://arxiv.org/html/2404.06395v1)

### 2024-lecture-15.pdf

- 第1–15页：RLHF三阶段中此讲聚焦SFT；FLAN/Alpaca/OASST格式不同。p15关于CS336的泛泛回答是标注质量反例，不能当课程事实。表单现场回答未公开。
- 第16–28页：偏好分数受长度/风格影响。已背过的知识在SFT后仍可能提取变差；Gekhman/Ghosal图不能简化成SFT绝对学不到知识。安全数据500/2000例呈现有用性取舍，不能承诺消除越狱。
- 第29–38页：LIMA的1000序列约750k tokens，不能证明任何任务千例足够。URIAL为3–8个上下文示例，无参数更新，不能叫few-shot SFT。ChatGPT评分/长度偏差保留；32k数据平台不证明所有SFT数据扩展无效。
- 第39–45页：LoRA冻结基座并训练低秩增量，增量rank≤r不表示基座低秩。A随机/B零初始化时首步A梯度零而B可非零；不能两者皆零。仍需经冻结层反传输入梯度。p44的127公式是Int8例，不能说是NF4公式。
- 第46–49页：NF4用16个非均匀码点，存储4bit而计算可用BF16；非IEEE符号/指数/尾数切分。absmax分块缓解离群影响，p49所谓non-contiguous不是要求，原论文采用展平后的连续块。反量化是有损近似。
- 第50–53页：p50把量化常数量化到4bit的文字需纠正：原QLoRA第二层为8bit浮点、再存FP32尺度。原论文64/256分块给常数开销0.127 bits/weight。forward反量化加LoRA，base不更新。Guanaco的21/41GB和评分限历史配方/评测，不能代表全面等同ChatGPT。

原始依据：[来源1](https://arxiv.org/html/2305.14314)

### 2024-lecture-16.pdf

- 第1–12页：生成与评价能力有差距；成本表是当年7B例，不能当现价。偏好标注须看任务准则、招募筛选与人口分布，p10明确Allegedly来源不可改成官方确认。
- 第13–24页：课堂标注5题表单结果缺失。众包质量/劳动条件、标注者偏差和GPT judge的system-level相关不等于每例可靠。CAI有critique/revision与偏好模型两阶段；自举使用已有结构/选择信息，不是无中生有。长度与偏好混杂需控制。
- 第25–32页：policy gradient/TRPO/PPO的目标、KL参考策略和old rollout策略区分。clip不是保证每步KL受限。p32 DPO human win-rate为横杠，sim46.8不能冒充human。
- 第33–38页：AlpacaFarm代码图分rollout、固定old logp的多轮minibatch、value loss/GAE。p36 max(sample log-ratio,0)为特定实现，不等于精确KL，更不保证不发散；pad==eos末位偏移代码自带警告。γ=λ=1配方非所有RL任务默认。
- 第39–46页：best-of-n、rejection SFT、expert iteration与DPO有训练/推理成本差别。DPO闭式最优需非参数策略/参考支持等条件，Z(x)在偏好差中消掉，不应说任意有限网络自动严格最优。p45 human DPO仍缺值，历史榜单非当前排名。
- 第47–50页：proxy reward上升而gold下降是过优化例，noiseless simulated preference面板例外要保留。p49仍是概率模型，问题是校准可能变差、熵降低，不是失去概率分布定义。

### 2024-lecture-17.pdf

- 第1–6页：p2明确“No recordings for guest lectures – next mon/wed”，为2024末两嘉宾课未录制的官方课内证据，须提升缺口分类。p3–4作业曲线1e19 FLOPs/399.51M loss3.04属当年结果，稳定性与超参影响不能全归参数量。
- 第7–18页：compression/multitask/chatbot三种目标决定评测。p9“LM标准评测已死”为课堂修辞，held-out loss仍可用于训练检查，不能删除。GLUE微调与few-shot评测不同，HELM-lite/旧榜单是历史组合。
- 第19–25页：逐例辨ARC/HellaSwag/TruthfulQA/MMLU/WinoGrande/GSM8K的知识、拒绝、常识和数学目标。p20TruthfulQA不等于全为拒答题；p25约8500含train/test，test1319通常约1k，不能将8500当test。
- 第26–33页：饱和、题目错误、prompt/选项顺序/logp-vs-generation都影响排名。PC1高解释率是该模型样本的相关，不证明单一因果能力。inverse scaling的prior/priors例需保留原拼写。
- 第34–45页：真实用户偏好与研究能力评测目标不同。Arena使用分布未按用户去重，不能称全体人群代表样本。AlpacaEval需顺序随机化、长度控制、自偏好检验；LC回归缓解混杂非消除所有偏差。
- 第46–53页：安全应同时看违规拒绝及良性过拒，bias/fairness任务覆盖不同群体。直接泄漏test与相近任务分布区分；社交媒体质疑非确定污染证据。Min-k-prob为启发式membership证据，exchangeability需假设，不保证检出全部污染。

### cs336_assignment1_basics.pdf

- 第1–12页：2026版BPE：Unicode码点/UTF-8字节不同；special token隔离再预切分，不能跨边界合并。p7词频乘相邻pair出现次数，平局按字典序大者。encode重放merge rank，非最长字符串贪心；增量解码可能遇不完整UTF-8，uint16仅词表<65536适用。
- 第13–28页：两架构图为最终norm+LM head、串行pre-norm两残差。RMS公式这里含1/d，与讲义缺项勘误互证。SiLU平滑但有负值，SwiGLU三矩阵用8d/3与两矩阵4d近似等参。RoPE旋转Q/K非V、位置buffer非参数；mask=True表示可见，须核对API约定。2mnp为常用FLOPs近似，点积精确加法次数n−1。
- 第29–38页：stable cross entropy、perplexity=exp平均NLL。AdamW偏差修正与decoupled decay保留；图中把修正吸进LR时epsilon位置与标准写法数值可能不同，按版本测试。warmup/cosine边界、全梯度norm clipping、shifted targets、checkpoint optimizer状态、top-p累计重归一均需讲。
- 第39–47页：TinyStories LR/batch/norm位置/NoPE/SwiGLU消融保留，控制tokens、参数预算及重调LR，不能预写未跑的曲线。2026有CPU/MPS低资源提示，硬件时间不作本机承诺；库2.9.0限制标注历史版本。OWT迁移和榜单为可选知识背景。

### cs336_assignment2_systems.pdf

- 第1–10页：Nsight时间线区分host launch/device execution及warmup，CUDA计时需同步；mixed precision保留FP32累加与BF16存储区别。allocated/reserved峰值不同，profiler有开销。
- 第11–15页：saved_tensors hooks展示autograd残差与参数引用，不应简单重复相加推峰值；checkpoint以重算换内存，最优sqrt层数是简化模型。
- 第16–29页：block-pointer图横向advance是feature，纵向是row tile。online attention需同时更新m/l/未归一O，最后除l，并存L=m+logl；backward重算P并用D=rowSum(O*dO)。causal mask−1e6为题设近似，任意值范围下不等于数学−∞。算法双遍dQ/dK,dV避免跨block写冲突，不代表零重算。
- 第30–38页：rank图区分global/local/node，gloo/NCCL按设备；allreduce例全rank同得sum。flatten减少launch次数，异步handle必须wait且不能提前改buffer，communication overlap须看timeline。ZeRO1仅分optimizer state仍省显存。
- 第39–48页：FSDP/TP/2D公式按shape与group轴理解，reduce-scatter返回的shard为全rank求和，输入为partial full gradients；两轴通信可重叠取max是题设，争用时应加成本。critical batch非一过线必降低泛化。8B/32k双B200榜单与旧版attention单kernel目标不同，不能移植测速数字。

### cs336_assignment3_scaling.pdf

- 第1–3页：2026加入真实wall-clock预算与可变架构/优化器；合成IsoFLOPs先取每预算最低已测loss而非保证连续最优，C≈6ND不能忽略长上下文等项。
- 第4–8页：模型参数、tokens、运行时三预算不同。API校内凭证/12 B200小时拟合与48小时外推仅原作业，不尝试取得凭证。实验记录须含失败/中止，不把最终loss等同完整收敛；不能用12L d²估计任意MoE/不同FFN。

### cs336_assignment4_data.pdf

- 第1–9页：WARC原HTML/WET已有抽取/WAT元数据区分；非正文抽取会影响质量。fastText langID、规则PII、toxicity/NSFW各有误杀漏检，概率并非校准保证。Wiki正样本/CC负样本训练的是分布代理，不是真理质量标签。
- 第10–12页：exact-line题设删除所有出现>1的行，不能写成只保留首份。MinHash先候选再真Jaccard，band索引必须入bucket key；连接分量A-B/B-C不保证A-C相似。LSH增加bands在固定hash数时提升召回但可能降低候选精度。
- 第13–16页：固定GPT2模型/训练仅变数据才公平。可参看Paloma validation做筛选设计，禁止复制进训练，反复调参仍需独立最终test。网页抽样检查、保留比例/域变化和失败记录保留。

### cs336_spring2026_assignment5_alignment.pdf

- 第1–7页：2026使用OLMo-2-0425-1B/GSM8K，与2025Qwen/MATH不同；zero/few-shot/CoT提示和答案parser需独立检查格式/正确奖励。
- 第8–13页：score-function与action-independent baseline不偏前提。group mean包含自身时梯度期望乘(G−1)/G，组std再改变目标权重；torch.std默认sample与公式population需区分。sequence length平均改变token权重。
- 第14–25页：response_mask避免训练prompt/pad，old policy和vLLM权重同步，梯度累积按全局正常化而非简单每microbatch均值。图表作业要求尚无实验数据，不造30%奖励曲线。
- 第26–31页：DrGRPO去std与长度但仍有有限G均值偏差；RFT只正确样本与baseline策略不同。MaxRL本作业用constant Z，原论文按batch token数，不能称完全相同。极限G→∞二元奖励权重：Dr1，std1/sqrt(p(1−p))，mean1/p；p=0/1须epsilon/零优势处理，有限G不直接等于极限。
- 第32–39页：完整sequence重要性比需support，产品可能高方差但非每样本必指数爆炸；token ratio仅校正一步，其余前后缀仍旧策略。clipping的min依优势正负作用不同，不保证KL约束；GSPO取几何均值引入偏差并自然带1/L，CISPO梯度截断与PPO平坦区不同。

### cs336_spring2026_assignment5_supplement_safety_rlhf.pdf

- 第1–8页：保留MMLU/GSM8K/AlpacaEval/SimpleSafetyTests四类目标、parser失败率/throughput、自动judge误判与过拒。Llama3.1-8B基座与Llama3.3-70B参考不可混同。
- 第9–14页：SFT prompt-output拼接、label shift和response-only loss；有效batch与microbatch区分。沿用统一prompt/decoding才比较SFT前后；red-team例不代表全面安全。
- 第15–18页：DPO无显式RM但有隐式偏好模型，固定SFT reference，不与old rollout混用。chosen/rejected需同prompt且标注分布有偏；sum logp与长度bias、loss稳定logsigmoid要解释。

### cs336_spring2025_assignment1_basics.pdf

- 第1–13页：BPE字节合并流程、预切分边界、special token、encode merge rank与吞吐/压缩率实验均保留；2025版低资源提示不等于2026实测硬件时间。
- 第14–28页：架构/残差/SiLU图与2026同概念但页码不同；2025 d_ff建议64倍数、算量题GPT2-shaped非原GPT2架构。RMS此版有1/d，可作为讲义勘误互证。
- 第29–39页：stable CE/SGD/AdamW/cosine/gradient clipping、memmap+shift target/checkpoint、temperature/top-p均覆盖；全FP32内存账本不能直接当BF16峰值。
- 第40–50页：TinyStories与OWT、norm消融/NoPE/SwiGLU对SiLU保留。H100小时限制和排行榜为历史版本，曲线/分数属于待运行要求非原答案。

### cs336_spring2025_assignment2_systems.pdf

- 第1–9页：2025 profiler/FP32-BF16/显存截图例题与2026新增checkpoint章节区分；Nsight统计不等于PyTorch单op名字，异步必须同步后计时。
- 第10–24页：weighted-sum tile图和FlashAttention算法逐图核对，m/l/O在线更新、L保存反向重算P。旧版benchmark至16384/new版可65536不同，不借用速度结果。
- 第25–34页：global/local/node rank图、naive/flatten/异步/bucket DDP到optimizer sharding，等待和bucket-size取舍需保留。ZeRO1分optimizer后更新权重需要一致同步。
- 第35–37页：DP/ZeRO1/gradaccum算量比较保留；2025榜单只attention kernel，与2026全8B训练步不同。

### cs336_spring2025_assignment3_scaling.pdf

- 第1–3页：固定FLOPs外推，合成IsoFLOPs可拟二次曲线求低点但作业建议选已测最低；离散最优有网格误差。
- 第4–6页：当年API按FLOPs硬预算限制并非2026wall-clock接口；固定12 heads等约束和实际data/architecture recipe会限制拟合，不称已运行服务。

### cs336_spring2025_assignment4_data.pdf

- 第1–8页：WARC/WET解析、语言/PII/有害规则与质量分类代理保留，2025提供slurm并行示例不等于在线自学必须有集群。
- 第9–11页：exact-line删除重复全部出现，MinHash+LSH候选后Jaccard确认；connected component去重链条不等于每对过阈值。
- 第12–17页：并行过滤/保留域分布/人工抽样/uint16 token化/fixed-model比较保留。Paloma作筛选验证不得复制入训练；机械成功不等于质量提高。

### cs336_spring2025_assignment5_alignment.pdf

- 第1–6页：Qwen2.5-Math-1.5B/MATH基线与2026 OLMo/GSM8K区分；p3注明MATH原始数据取得限制并列替代数据。不能把拿不到原数据当作已复现实验。
- 第7–15页：SFT response mask与label shift、per-token entropy、microbatch/global normalizer；expert iteration对自身正确生成再SFT，与强模型蒸馏不同。
- 第16–24页：policy梯度/旧策略IS/GRPO clipped ratio正负优势分开；logp*adv是产生梯度的surrogate非可跨运行比较的reward指标。去std不自动去长度归一化/有限G bias。
- 第25–33页：masked_mean与constant normalizer改变长度权重，p30–31数值例用于比较；离策略增加minibatches保持old_logp冻结，prompt ablation/seed方差/格式奖励均保留。
- 第34–38页：未用KL是本作业简化，不代表KL在所有算法无用。系统优化参考vLLM/verl等仍需版本与正确性验证，排行榜为历史要求。

### cs336_spring2025_assignment5_supplement_safety_rlhf.pdf

- 第1–8页：四评测MMLU/GSM8K/AlpacaEval/SimpleSafetyTests与基座/参考模型区分，parser失败单独报，GPT judge非完美真值。
- 第9–15页：SFT拼接和mask/grad accumulation、同prompt前后对比、red teaming与过拒保留；文中H100小时为原配方。
- 第16–19页：HH chosen/rejected、固定reference及DPO logsigmoid之差，保留长度偏差/分布支持前提，不用训练loss降代替用户偏好/安全收益。

### cs336_spring2024_assignment1_basics.pdf

- 第1–10页：BPE字节/UTF8和低频词、预切分/合并顺序等保留；旧版special-token/词频toy例不当当前GPT4 tokenizer实现。
- 第11–18页：2024架构图含learned absolute position相加、GELU两矩阵、dropout，不能用2025RoPE/SwiGLU覆盖。串行两残差及最终RMSNorm保留。RMS这里已有1/d，讲义缺项可互证。
- 第19–27页：stable CE、SGD、AdamW、cosine、clip、data shift/checkpoint/temperature/top-p与新版同原理；本版dropout训练/eval行为也须讲。
- 第28–34页：旧版有parallel FFN+attention消融以及dropout超参，不应丢掉；TinyStories/OWT与训练曲线要求尚无给定结果，不编实验结论。

### cs336_spring2024_assignment2_systems.pdf

- 第1–8页：2024系统作业是RMSNorm kernel而非后来的FlashAttention。PyTorch profiler/flamegraph含调用层级和self/total，不能把CPU时间当GPU总时长。
- 第9–17页：weighted-sum逐程序一row与反向partial weight reduce、RMSNorm forward/backward公式、torch.compile对比与显存profile保留。RMSNorm rows独立但gain梯度跨rows归约，避免race。
- 第18–23页：local/global/node rank图和Slurm多node启动例保留，和单机mp.spawn不同；gloo/NCCL需设备/环境匹配。
- 第24–30页：flatten/async/bucket DDP到ZeRO1 optimizer sharding，hook触发与wait界限、bucket tradeoff需解释；未声称已复现多机数据。

### cs336_spring2024_assignment3_scaling.pdf

- 第1–3页：历史FLOPs预算/合成IsoFLOPs最低点拟合保留；离散网格及外推的不确定性不可省略。
- 第4–6页：原API固定12heads、训练数据等约束，learned absolute positions与GELU，不直接当2025/26接口。API额度/可用性未验证，无凭证不运行。

### cs336_spring2024_assignment4_data.pdf

- 第1–8页：原HTML抽取使用resiliparse，语言/PII/有害/质量过滤各看偏差。WARC/WET/WAT角色不同，不把课程示例网页当全部真实分布。
- 第9–11页：exact-line题设删除所有重复行；LSH候选后真Jaccard确认，链式cluster不要求每对相似。
- 第12–16页：旧版直接用WARC转正文再过滤，后版提供English WET起点；保留固定GPT2模型评估、域分布/保留率和人工抽样，Paloma validation不复制训练。

### cs336_spring2024_assignment5_alignment.pdf

- 第1–9页：Llama3-8B基座/70B参考的MMLU/GSM8K/AlpacaEval/SimpleSafetyTests基线，不同于后版3.1/3.3型号；parser错误与judge误判单列。
- 第10–16页：SFT response-only loss、padding/shift和gradient accumulation，固定解码设定做前后对比，red teaming与过拒均需讲。
- 第17–21页：DPO固定SFT reference、同prompt偏好对与数值稳定logsigmoid；HH的helpfulness/harmlessness偏好群体不同，不把训练loss当安全保证。
