# CS336 AI 陪学：完整课程目录与来源

版本：3.1（299点陪学教案版；材料核验基线1.2） · 来源核对日期：2026-10-22 · 对象：学过入门课、希望从基础重学的人。

这是一份课程目录与材料索引。3.1版配套《CS336-逐点陪学教案.md/html》，原目录299点均有原创教学设计；原始课堂材料仍有缺口，不代表所有录像细节已恢复。

## 1. 覆盖范围与核对结果

| 范围 | 数量 | 本次状态 |
|---|---:|---|
| 2024、2025、2026 官网日程 | 3 年 × 19 讲 = 57 个条目 | 全部登记并映射到章节 |
| 每年前 17 讲的核心讲义 | 51 份 | 已下载；核对 Python 的函数／标题结构和 PDF 的逐页文本线索 |
| 2024 作业、2025 固定分支作业、当前作业 | 5 + 6 + 6 = 17 份 PDF | 已取得并核对章节结构；包含两份年度安全对齐补充材料 |
| 嘉宾讲座 | 6 个日程条目 | Dan Fu 已有逐讲整理、问答核对、指定画面及作者相关论文/代码；2024两场官方讲义明确不录制；2025两场与2026 Daniel Selsam完整课堂未取得 |
| 官方播放列表录像字幕 | 34/35 个已提取，约 45.1 小时 | 保留逐段时间戳；字幕全文已校读，原始英文音频已全程机器比对及疑难窗口复核；未人工逐句听校 |

因此，本版已完成公开日程和核心讲义层面的覆盖映射。尚不能承诺“视频里的每一句和所有嘉宾内容都已覆盖”。章节是教学重排，不是 Stanford 官方章号。

官方入口：[2024](https://cs336.stanford.edu/spring2024/) · [2025](https://cs336.stanford.edu/spring2025/) · [2026](https://cs336.stanford.edu/) · [2025 官方录像](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_) · [2026 官方录像](https://www.youtube.com/playlist?list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

## 2. AI 陪学如何使用这份目录

把本目录、逐点陪学教案、课堂备课资料和个人学习进度交给AI。先教当前点，提问后等待回答。每点已有教案，需要深入或更新时再查对应一手来源。

1. 每轮只引入一个主要概念。先说明它解决的问题，再给具体例子。必要的数学当场补。
2. 讲完后只问一个能暴露理解的小问题。等待学习者回答。不要在同一轮给出答案或继续下一概念。
3. 根据回答解释正确之处和缺口。答对选择题不等于掌握；需要时请学习者用自己的话解释，或换一个情境。
4. 学习者理解后再前进。不要用机械考试拖慢已经理解的内容；学习者可以要求跳过或深入。
5. 不要求完成大作业。保留作业承载的知识点，用短推演、示意图或可选小实验来讲。
6. 不把类比当作实际机制。说明简化之处；公式先讲变量含义，再讲关系。
7. 对旧材料中的模型性能、硬件规格、成本和法律背景标明日期。需要讲当前情况时，查一手新来源。未证实的变化标为待核实。
8. 每次结束记录：当前位置、学习者已能解释的内容、仍有疑问的内容、下一点。不要把“已经讲过”写成“已经掌握”。
9. 本资料使用中文原创讲解并链接来源。它不是 Stanford 官方课程、官方翻译或认证课程。分享时不复制整份讲义或未授权录像。

## 3. 全部知识目录

学习顺序：语言模型与分词 → 张量与训练 → Transformer → 系统与推理 → Scaling → 评估与数据 → 后训练 → 多模态。第 00 章随用随补。下列问题用于 AI 备课，不是一次性布置给学习者的作业。

### 00 · 随用随补的基础

读懂课程中必要的数学和程序表示。

- 00.01 数字与变量
- 00.02 函数和参数
- 00.03 向量与矩阵
- 00.04 矩阵乘法
- 00.05 概率分布与条件概率
- 00.06 对数与指数
- 00.07 导数、梯度与链式法则
- 00.08 训练集、验证集和测试集

来源：[2024 L02](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py) · [2025 L02](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json) · [2026 L02](https://cs336.stanford.edu/lectures/?trace=lecture_02)

理解检查示例：为什么概率最大的结果也可能没有发生？

### 01 · 语言模型在做什么

区分预测概率、生成文字和训练。

- 01.01 下一 token 预测
- 01.02 条件概率
- 01.03 序列概率
- 01.04 训练与推理
- 01.05 自回归生成
- 01.06 预训练与后训练
- 01.07 模型历史和开放程度
- 01.08 固定数据与算力下的效率问题
- 01.09 小模型经验的适用边界

来源：[2024 L01](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_01.py) · [2025 L01](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json) · [2026 L01](https://cs336.stanford.edu/lectures/?trace=lecture_01)

理解检查示例：预测一种续写的概率，与保证它会出现，有什么区别？

### 02 · 文字如何变成 token

理解分词的作用和取舍。

- 02.01 Unicode 与 UTF-8
- 02.02 字符、字节、词和子词
- 02.03 token ID 与词表
- 02.04 BPE 的训练与编码、解码
- 02.05 特殊 token 和预分词
- 02.06 Unigram 与 SentencePiece
- 02.07 规范化、空格、数字和多语言
- 02.08 压缩率与词表大小
- 02.09 字节级和无分词器路线

来源：[2024 L01](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_01.py) · [2024 L03](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json) · [2026 L01](https://cs336.stanford.edu/lectures/?trace=lecture_01)

理解检查示例：为什么一个汉字不一定恰好对应一个 token？

### 03 · 张量与最小训练程序

能跟随模型中的数据形状。

- 03.01 形状、维度与 batch
- 03.02 dtype 和设备
- 03.03 存储、stride、view 与切片
- 03.04 广播、逐元素运算和矩阵乘法
- 03.05 einsum、einops 和形状标注
- 03.06 参数、Embedding 与 Module
- 03.07 自动求导
- 03.08 数据加载、随机种子与复现
- 03.09 保存与恢复 checkpoint

来源：[2024 L02](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py) · [2025 L02](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json) · [2026 L02](https://cs336.stanford.edu/lectures/?trace=lecture_02)

理解检查示例：两个张量形状相同，为什么仍可能有不同的内存布局？

### 04 · 模型怎样从错误中学习

走通一次参数更新。

- 04.01 logits 与 softmax
- 04.02 交叉熵及数值稳定
- 04.03 梯度下降与反向传播
- 04.04 SGD、动量、AdamW
- 04.05 初始化
- 04.06 学习率 warmup、cosine 与 WSD
- 04.07 梯度裁剪
- 04.08 batch 和梯度累积
- 04.09 训练循环
- 04.10 验证曲线、过拟合与调参
- 04.11 消融实验

来源：[2024 L02](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py) · [2025 L02](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json) · [2026 L01](https://cs336.stanford.edu/lectures/?trace=lecture_01) · [2026 L02](https://cs336.stanford.edu/lectures/?trace=lecture_02) · [2026 L03](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf) · [2026 L11](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf)

理解检查示例：怎样区分模型能力不足和训练方法没有调好？

### 05 · 从零理解 Transformer

跟随一个 token 通过整个模型。

- 05.01 Embedding 与线性层
- 05.02 Q、K、V
- 05.03 点积、缩放和 softmax
- 05.04 因果 mask
- 05.05 多头自注意力
- 05.06 位置表示
- 05.07 MLP／FFN
- 05.08 残差连接
- 05.09 归一化
- 05.10 Transformer block 的堆叠
- 05.11 输出投影与概率
- 05.12 并行训练与逐步生成

来源：[2024 L03](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)

理解检查示例：为什么训练时必须遮住未来的 token？

### 06 · 现代模型结构与超参数

理解每种设计解决的问题。

- 06.01 pre-norm 与 post-norm
- 06.02 LayerNorm 与 RMSNorm
- 06.03 double norm 和非残差 post-norm
- 06.04 ReLU、GELU、GLU 与 SwiGLU
- 06.05 移除 bias
- 06.06 串行与并行 block
- 06.07 RoPE
- 06.08 模型深度、宽度、头数和 FFN 比例
- 06.09 词表大小
- 06.10 dropout 与 weight decay
- 06.11 z-loss、QK norm 和 logit soft-capping

来源：[2024 L03](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf)

理解检查示例：一个结构修改提升了速度，是否一定也减少了 FLOPs？

### 07 · Attention 的成本与替代路线

理解长上下文带来的问题。

- 07.01 全注意力的计算和缓存成本
- 07.02 MQA 与 GQA
- 07.03 MLA 与跨层缓存共享
- 07.04 滑动窗口和稀疏注意力
- 07.05 全局与局部层交错
- 07.06 线性注意力及递推形式
- 07.07 Mamba／状态空间模型
- 07.08 Gated DeltaNet
- 07.09 混合结构
- 07.10 稀疏索引器与动态稀疏注意力
- 07.11 质量、内存和速度的取舍

来源：[2024 L03](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://github.com/stanford-cs336/spring2025-lectures/blob/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json) · [2026 L03](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf) · [2026 L04](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf) · [2026 L10](https://cs336.stanford.edu/lectures/?trace=lecture_10)

理解检查示例：线性注意力减少计算后，为什么还需要比较模型质量？

### 08 · 混合专家 MoE

区分总参数和每次激活的参数。

- 08.01 稠密与稀疏计算
- 08.02 router、top-k 与专家选择
- 08.03 token-choice 和 expert-choice
- 08.04 细粒度和共享专家
- 08.05 路由噪声与可微训练
- 08.06 负载均衡辅助损失
- 08.07 无辅助损失的偏置调整
- 08.08 capacity、通信和专家并行
- 08.09 路由稳定性
- 08.10 upcycling
- 08.11 DeepSeek 等公开案例及经验边界

来源：[2024 L04](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://github.com/stanford-cs336/spring2025-lectures/blob/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf) · [2026 L08](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)

理解检查示例：总参数翻倍，单个 token 的计算量是否一定翻倍？

### 09 · 计算量、显存与性能

能判断主要资源花在哪里。

- 09.01 参数量与 FLOPs
- 09.02 前向和反向的成本
- 09.03 训练计算量估算
- 09.04 参数、梯度、优化器状态和激活内存
- 09.05 FP32、FP16、BF16、FP8 和 FP4
- 09.06 混合精度
- 09.07 算术强度
- 09.08 带宽瓶颈与算力瓶颈
- 09.09 roofline
- 09.10 MFU
- 09.11 重计算和梯度累积的成本

来源：[2024 L02](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py) · [2025 L02](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json) · [2026 L02](https://cs336.stanford.edu/lectures/?trace=lecture_02) · [2026 L05](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)

理解检查示例：GPU 还有空闲算力，为什么程序仍然可能很慢？

### 10 · GPU、TPU 与硬件执行

建立硬件性能的直觉。

- 10.01 CPU 与 GPU
- 10.02 SM、warp、thread 和 block
- 10.03 SIMT 和分支分歧
- 10.04 寄存器、共享内存、缓存和 HBM
- 10.05 Tensor Core
- 10.06 TPU 与矩阵计算
- 10.07 低精度硬件
- 10.08 算力与带宽增长不一致
- 10.09 内存合并访问
- 10.10 tiling、对齐和 wave quantization

来源：[2024 L05](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf)

理解检查示例：为什么更少的显存读写，可能比更少的算术运算更有价值？

### 11 · 性能分析、编译与 GPU kernel

理解怎样定位并改善慢操作。

- 11.01 可靠计时、预热和同步
- 11.02 端到端基准与 profiler
- 11.03 Nsight、火焰图和内存剖析
- 11.04 算子融合
- 11.05 CUDA、PTX 与 Triton 的层次
- 11.06 torch.compile
- 11.07 GELU、softmax、行求和和矩阵乘示例
- 11.08 分块与 mask
- 11.09 自定义 autograd
- 11.10 RMSNorm 前向与反向
- 11.11 正确性和速度分开验证

来源：[2024 L06](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_06.py) · [2025 L06](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_06.json) · [2026 L06](https://cs336.stanford.edu/lectures/?trace=lecture_06)

理解检查示例：单个 kernel 更快，为什么端到端训练不一定明显变快？

### 12 · FlashAttention 与重计算

理解精确注意力怎样节省内存。

- 12.01 普通 Attention 的中间矩阵
- 12.02 HBM 与片上内存
- 12.03 分块
- 12.04 在线 softmax
- 12.05 数值稳定的增量统计
- 12.06 融合前向计算
- 12.07 保存必要统计量
- 12.08 反向重计算
- 12.09 FlashAttention-2 的工作划分
- 12.10 精确计算与近似注意力的区别

来源：[2024 L05](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf) · [2026 L06](https://cs336.stanford.edu/lectures/?trace=lecture_06)

理解检查示例：FlashAttention 为什么可以不保存整个注意力矩阵？

### 13 · 多设备通信

先理解设备怎样交换数据。

- 13.01 rank、world size 和进程组
- 13.02 broadcast、reduce、all-reduce
- 13.03 all-gather、reduce-scatter 和 all-to-all
- 13.04 环式通信
- 13.05 延迟与带宽
- 13.06 NCCL 和 torch.distributed
- 13.07 同步与异步
- 13.08 通信和计算重叠
- 13.09 节点内与节点间互连
- 13.10 拓扑与 TPU／GPU 网络差异

来源：[2024 L07](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_08.py) · [2025 L07](https://github.com/stanford-cs336/spring2025-lectures/blob/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_08.json) · [2026 L07](https://cs336.stanford.edu/lectures/?trace=lecture_07) · [2026 L08](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)

理解检查示例：all-reduce 与 reduce-scatter 加 all-gather 有什么关系？

### 14 · 分布式训练

理解每种并行方式分割什么。

- 14.01 数据并行 DDP
- 14.02 梯度平均与通信 bucket
- 14.03 ZeRO 1／2／3 与 FSDP
- 14.04 优化器、梯度和参数分片
- 14.05 张量并行的行列切分
- 14.06 流水线、microbatch 和气泡
- 14.07 1F1B 与 zero-bubble
- 14.08 序列和上下文并行
- 14.09 专家并行
- 14.10 激活重计算
- 14.11 组合并行和公开模型案例
- 14.12 通信量与显存估算

来源：[2024 L07](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_08.py) · [2025 L07](https://github.com/stanford-cs336/spring2025-lectures/blob/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_08.json) · [2026 L07](https://cs336.stanford.edu/lectures/?trace=lecture_07) · [2026 L08](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf)

理解检查示例：增加 GPU 后训练不再加速，你会先检查什么？

### 15 · 推理与服务

理解模型生成答案时的成本。

- 15.01 自回归采样
- 15.02 temperature、top-k 和 top-p
- 15.03 prefill 与 decode
- 15.04 KV cache
- 15.05 延迟与吞吐量
- 15.06 批量大小和内存带宽
- 15.07 量化及 AWQ
- 15.08 剪枝
- 15.09 缓存压缩与结构改动
- 15.10 推测采样及验证
- 15.11 连续批处理
- 15.12 PagedAttention
- 15.13 动态请求管理
- 15.14 vLLM 的用途

来源：[2025 L10](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json) · [2026 L10](https://cs336.stanford.edu/lectures/?trace=lecture_10)

理解检查示例：为什么输入处理和逐 token 输出可能需要不同优化？

### 16 · Scaling laws 的基础

会读曲线，并理解外推的限制。

- 16.01 幂律和对数坐标
- 16.02 数据、参数和计算量的关系
- 16.03 样本复杂度和均值估计
- 16.04 内在维度解释
- 16.05 分布变化
- 16.06 数据重复和有限数据
- 16.07 模型结构、深宽比及优化器的 scaling
- 16.08 临界 batch
- 16.09 Kaplan 与 Chinchilla
- 16.10 最小包络、IsoFLOPs 和联合拟合
- 16.11 拟合误差
- 16.12 训练最优与部署最优

来源：[2024 L09](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf)

理解检查示例：同一训练预算下，为什么更大的模型不一定更好？

### 17 · 把小实验扩展到大模型

理解超参数迁移与实验设计。

- 17.01 初始化、学习率和 batch 的共同变化
- 17.02 μP 的基本条件与推导
- 17.03 宽度迁移
- 17.04 标准参数化对照
- 17.05 RMSNorm、优化器和 weight decay 的适用边界
- 17.06 MiniCPM、DeepSeek、CerebrasGPT 等公开配方
- 17.07 WSD 和训练复用
- 17.08 Muon 的尺度依赖
- 17.09 多尺度实验与预算分配
- 17.10 误差和外推风险

来源：[2024 L10](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://github.com/stanford-cs336/spring2025-lectures/blob/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf)

理解检查示例：小模型上的最好设置，为什么不能直接照搬到大模型？

### 18 · 怎样评价模型

先定义能力，再选评价方法。

- 18.01 负对数似然、perplexity 和分词依赖
- 18.02 选择题、知识、数学和代码测试
- 18.03 few-shot 与提示敏感性
- 18.04 指令遵循、聊天偏好和 Arena
- 18.05 LLM-as-judge
- 18.06 长度、位置和风格偏差
- 18.07 推理和智能体评测
- 18.08 安全、偏见与公平
- 18.09 训练测试污染及检测局限
- 18.10 饱和、效度、真实任务和统计不确定性

来源：[2024 L17](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_12.json) · [2026 L12](https://cs336.stanford.edu/lectures/?trace=lecture_12)

理解检查示例：两个模型的 perplexity，什么情况下不能直接比较？

### 19 · 训练数据从哪里来

认识数据来源及其局限。

- 19.01 网页、Common Crawl 与 WARC
- 19.02 Wikipedia、代码、论文、书籍和问答
- 19.03 WebText、C4、The Pile、MassiveText、RefinedWeb、Dolma、DCLM、Nemotron-CC、Common Pile 与 The Stack
- 19.04 来源、语言、领域和时间
- 19.05 开放权重与开放数据的区别
- 19.06 版权、许可、隐私与可追溯性问题

来源：[2024 L11](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_11.py) · [2024 L13](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_13.py) · [2025 L13](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_13.json) · [2026 L13](https://cs336.stanford.edu/lectures/?trace=lecture_13)

理解检查示例：数据量相同，为什么不同来源可能训练出不同能力？

### 20 · 数据转换与过滤

理解数据质量的操作含义。

- 20.01 HTML 转文本与正文抽取
- 20.02 语言识别
- 20.03 规则过滤
- 20.04 质量与有害内容分类
- 20.05 个人信息处理
- 20.06 KenLM、fastText 和 DSIR
- 20.07 重要性采样
- 20.08 模型评分
- 20.09 误删与漏删
- 20.10 过滤对领域和人群的影响
- 20.11 通过训练实验验证过滤效果

来源：[2024 L12](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_12.py) · [2025 L14](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_14.json) · [2026 L14](https://cs336.stanford.edu/lectures/?trace=lecture_14)

理解检查示例：删除低分文本，为什么可能损失有用的训练信号？

### 21 · 数据去重

分清完全重复和近似重复。

- 21.01 哈希及碰撞
- 21.02 文档、段落和行级精确去重
- 21.03 Bloom filter 和误判率
- 21.04 后缀数组与重复子串
- 21.05 集合、n-gram 和 Jaccard
- 21.06 MinHash
- 21.07 LSH 的 band／row 参数
- 21.08 扩展到大数据的计算成本
- 21.09 去重与评测污染的关系

来源：[2024 L13](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_13.py) · [2025 L14](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_14.json) · [2026 L14](https://cs336.stanford.edu/lectures/?trace=lecture_14)

理解检查示例：哈希一致和语义相似，是同一件事吗？

### 22 · 数据混合与训练阶段

理解训练数据怎样随目标变化。

- 22.01 数据配比与消融
- 22.02 领域重采样
- 22.03 数据选择和 scaling
- 22.04 数据混合估计
- 22.05 重复数据与训练轮次
- 22.06 长上下文数据
- 22.07 代码 infilling
- 22.08 特定领域与推理数据
- 22.09 合成数据与自蒸馏
- 22.10 midtraining
- 22.11 后训练数据质量及验证

来源：[2024 L14](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_14.py) · [2025 L13](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_13.json) · [2026 L14](https://cs336.stanford.edu/lectures/?trace=lecture_14) · [2026 L15](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)

理解检查示例：合成数据增加了数量，怎样判断它也增加了有效信息？

### 23 · 监督微调 SFT 与参数高效微调

理解模型怎样学会遵循指令。

- 23.01 预训练目标与指令目标
- 23.02 FLAN、Alpaca、OpenAssistant 等数据
- 23.03 对话格式与 loss mask
- 23.04 小量高质量数据
- 23.05 风格、事实与知识提取
- 23.06 幻觉与安全微调
- 23.07 训练设置和评估
- 23.08 PEFT
- 23.09 LoRA 的低秩更新
- 23.10 QLoRA、NF4 与双重量化
- 23.11 冻结参数与完整微调的取舍

来源：[2024 L14](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_14.py) · [2024 L15](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf)

理解检查示例：模型回答风格变好，是否说明它学到了更多事实？

### 24 · 人类偏好与 RLHF

理解奖励怎样改变模型输出。

- 24.01 成对偏好与奖励模型
- 24.02 标注者、众包伦理和人口差异
- 24.03 AI 反馈与 Constitutional AI
- 24.04 生成和评价的差距
- 24.05 策略、回报和 KL 约束
- 24.06 PPO、value、advantage 和 GAE
- 24.07 DPO 的目标和推导
- 24.08 变体与比较
- 24.09 奖励过优化、长度偏差、模式坍塌和校准
- 24.10 安全红队测试

来源：[2024 L16](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://github.com/stanford-cs336/spring2025-lectures/blob/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf) · [2026 L16](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)

理解检查示例：奖励分数不断上涨，为什么真实回答质量可能下降？

### 25 · 推理强化学习 RLVR

理解可验证奖励、长推理和测试时计算。

- 25.01 可验证奖励与评分器
- 25.02 语言模型作为策略
- 25.03 轨迹、回报与策略梯度
- 25.04 baseline 与方差
- 25.05 GRPO 及组内归一化
- 25.06 序列长度和难度偏差
- 25.07 on-policy 与 off-policy
- 25.08 重要性重加权和 clipping
- 25.09 Dr. GRPO、GSPO、MaxRL
- 25.10 拒绝采样与 expert iteration
- 25.11 R1／R1-Zero、Kimi、Qwen 案例
- 25.12 蒸馏
- 25.13 test-time scaling
- 25.14 智能体环境、轨迹和 RL 基建

来源：[2025 L16](https://github.com/stanford-cs336/spring2025-lectures/blob/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_17.json) · [2026 L16](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf)

理解检查示例：答案可以自动判分，是否就代表奖励设计没有漏洞？

### 26 · 用一个小例子走通强化学习

把抽象算法对应到一次更新。

- 26.01 2025 年官方排序玩具任务
- 26.02 生成回答
- 26.03 计算 token log-prob
- 26.04 设计排序奖励
- 26.05 策略梯度 loss
- 26.06 KL penalty
- 26.07 参数冻结
- 26.08 训练前后概率变化
- 26.09 采样与更新循环
- 26.10 不同实验设置的比较

来源：[2025 L17](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_17.json)

理解检查示例：同一个回答获得正奖励后，哪些概率应该发生什么变化？

### 27 · 多模态模型

把文字模型的认识扩展到图片与视频。

- 27.01 图像编码器
- 27.02 图文对比学习
- 27.03 CLIP 与 SigLIP
- 27.04 视觉 token 和连接器
- 27.05 LLaVA 与视觉指令微调
- 27.06 OneVision 与可变分辨率
- 27.07 Qwen-VL 系列
- 27.08 视频与时间位置
- 27.09 原生多模态与 Chameleon
- 27.10 理解与生成目标的差异

来源：[2026 L17](https://cs336.stanford.edu/lectures/?trace=lecture_17)

理解检查示例：图片进入语言模型之前，为什么也需要变成一种数值表示？

### 28 · 嘉宾讲座与课程后的更新

保留公开内容入口，并明确材料缺口。

- 28.01 六场历年嘉宾讲座逐项登记
- 28.02 已取得材料后再拆知识点
- 28.03 更新按架构、数据、系统、评估和后训练归位
- 28.04 区分课程当时的说法和后续证据
- 28.05 不开列没有读过的讲座细节
- 28.06 不把模型排行榜当作长期知识

来源：[2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://cs336.stanford.edu/#schedule)

理解检查示例：一个新结果需要满足什么证据条件，才值得修改已有认识？

## 4. 逐讲对应表：57 个日程条目

“已取得”指材料可读取并完成目录核对，不指已逐句精读或已写完陪学内容。原官网的讲者个人主页不当作该讲讲义。

### 2024 年

| 讲次 | 官网主题 | 官方材料 | 陪学章节 | 状态 |
|---|---|---|---|---|
| 01 | Overview, tokenization (Percy) | [lecture_01.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_01.py) | 01、02 | 已取得官方讲义，已核对目录线索 |
| 02 | Pytorch, resource accounting (Percy) | [lecture_02.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py) | 00、03、04、09 | 已取得官方讲义，已核对目录线索 |
| 03 | Architectures, hyperparameters (Tatsu) | [lecture 3.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%203%20-%20architecture.pdf) | 02、05、06、07 | 已取得官方讲义，已核对目录线索 |
| 04 | Mixture of experts (Tatsu) | [lecture 4.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) | 02、06、07、08 | 已取得官方讲义，已核对目录线索 |
| 05 | GPUs (Tatsu) | [lecture 5.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) | 10、12 | 已取得官方讲义，已核对目录线索 |
| 06 | Kernels, Triton (Percy) | [lecture_06.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_06.py) | 11 | 已取得官方讲义，已核对目录线索 |
| 07 | Parallelism (Tatsu) | [lecture 7.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) | 13、14 | 已取得官方讲义，已核对目录线索 |
| 08 | Parallelism (Percy) | [lecture_08.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_08.py) | 13、14 | 已取得官方讲义，已核对目录线索 |
| 09 | Scaling laws (Tatsu) | [lecture 9.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) | 16 | 已取得官方讲义，已核对目录线索 |
| 10 | Scaling laws (Tatsu) | [lecture 10.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) | 17 | 已取得官方讲义，已核对目录线索 |
| 11 | Data (Percy) | [lecture_11.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_11.py) | 19 | 已取得官方讲义，已核对目录线索 |
| 12 | Data (Percy) | [lecture_12.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_12.py) | 20 | 已取得官方讲义，已核对目录线索 |
| 13 | Data (Percy) | [lecture_13.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_13.py) | 19、21 | 已取得官方讲义，已核对目录线索 |
| 14 | Data (Percy) | [lecture_14.py](https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_14.py) | 22、23 | 已取得官方讲义，已核对目录线索 |
| 15 | Alignment (Tatsu) | [lecture 15.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) | 23 | 已取得官方讲义，已核对目录线索 |
| 16 | Alignment (Tatsu) | [lecture 16.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) | 24 | 已取得官方讲义，已核对目录线索 |
| 17 | Evals (Tatsu) | [lecture 17.pdf](https://github.com/stanford-cs336/spring2024-lectures/blob/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) | 18 | 已取得官方讲义，已核对目录线索 |
| 18 | Guest lecture by Ce Zhang | [官网日程](https://cs336.stanford.edu/spring2024/#schedule) | 28 | 官网有日程，未取得独立讲义／完整字幕 |
| 19 | Guest lecture by Aakanksha Chowdhery | [官网日程](https://cs336.stanford.edu/spring2024/#schedule) | 28 | 官网有日程，未取得独立讲义／完整字幕 |

### 2025 年

| 讲次 | 官网主题 | 官方材料 | 陪学章节 | 状态 |
|---|---|---|---|---|
| 01 | Overview, tokenization (Percy) | [lecture_01.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_01.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_01.py) · [课堂录像](https://www.youtube.com/watch?v=SQ3fZ1sAqXI) | 01、02 | 已取得官方讲义，已核对目录线索 |
| 02 | PyTorch, resource accounting (Percy) | [lecture_02.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_02.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_02.py) · [课堂录像](https://www.youtube.com/watch?v=msHyYioAyNE) | 00、03、04、09 | 已取得官方讲义，已核对目录线索 |
| 03 | Architectures, hyperparameters (Tatsu) | [lecture 3.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [课堂录像](https://www.youtube.com/watch?v=ptFiH_bHnJw) | 05、06 | 已取得官方讲义，已核对目录线索 |
| 04 | Mixture of experts (Tatsu) | [lecture 4.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [课堂录像](https://www.youtube.com/watch?v=LPv1KfUXLCo) | 07、08 | 已取得官方讲义，已核对目录线索 |
| 05 | GPUs (Tatsu) | [lecture 5.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [课堂录像](https://www.youtube.com/watch?v=6OBtO9niT00) | 10、12 | 已取得官方讲义，已核对目录线索 |
| 06 | Kernels, Triton (Tatsu) | [lecture_06.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_06.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_06.py) · [课堂录像](https://www.youtube.com/watch?v=E8Mju53VB00) | 11 | 已取得官方讲义，已核对目录线索 |
| 07 | Parallelism (Tatsu) | [lecture 7.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [课堂录像](https://www.youtube.com/watch?v=l1RJcDjzK8M) | 13、14 | 已取得官方讲义，已核对目录线索 |
| 08 | Parallelism (Percy) | [lecture_08.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_08.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_08.py) · [课堂录像](https://www.youtube.com/watch?v=LHpr5ytssLo) | 13、14 | 已取得官方讲义，已核对目录线索 |
| 09 | Scaling laws (Tatsu) | [lecture 9.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [课堂录像](https://www.youtube.com/watch?v=6Q-ESEmDf4Q) | 16 | 已取得官方讲义，已核对目录线索 |
| 10 | Inference (Percy) | [lecture_10.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_10.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_10.py) · [课堂录像](https://www.youtube.com/watch?v=fcgPYo3OtV0) | 07、15 | 已取得官方讲义，已核对目录线索 |
| 11 | Scaling laws (Tatsu) | [lecture 11.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [课堂录像](https://www.youtube.com/watch?v=OSYuUqGBQxw) | 17 | 已取得官方讲义，已核对目录线索 |
| 12 | Evaluation (Percy) | [lecture_12.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_12.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_12.py) · [课堂录像](https://www.youtube.com/watch?v=x-R5l2HsXqM) | 18 | 已取得官方讲义，已核对目录线索 |
| 13 | Data (Percy) | [lecture_13.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_13.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_13.py) · [课堂录像](https://www.youtube.com/watch?v=WePxmeXU1xg) | 19、22 | 已取得官方讲义，已核对目录线索 |
| 14 | Data (Percy) | [lecture_14.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_14.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_14.py) · [课堂录像](https://www.youtube.com/watch?v=9Cd0THLS1t0) | 20、21 | 已取得官方讲义，已核对目录线索 |
| 15 | Alignment - SFT/RLHF (Tatsu) | [lecture 15.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [课堂录像](https://www.youtube.com/watch?v=Dfu7vC9jo4w) | 23、24 | 已取得官方讲义，已核对目录线索 |
| 16 | Alignment - RL (Tatsu) | [lecture 16.pdf](https://github.com/stanford-cs336/spring2025-lectures/blob/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [课堂录像](https://www.youtube.com/watch?v=46f2QTDB08Q) | 24、25 | 已取得官方讲义，已核对目录线索 |
| 17 | Alignment - RL (Percy) | [lecture_17.py](https://cs336.stanford.edu/spring2025-lectures/?trace=var/traces/lecture_17.json) · [讲稿源文件](https://github.com/stanford-cs336/spring2025-lectures/blob/main/lecture_17.py) · [课堂录像](https://www.youtube.com/watch?v=JdGFdViaOJk) | 25、26 | 已取得官方讲义，已核对目录线索 |
| 18 | Guest Lecture by Junyang Lin | [官网日程](https://cs336.stanford.edu/spring2025/#schedule) | 28 | 官网有日程，未取得独立讲义／完整字幕 |
| 19 | Guest lecture by Mike Lewis | [官网日程](https://cs336.stanford.edu/spring2025/#schedule) | 28 | 官网有日程，未取得独立讲义／完整字幕 |

### 2026 年

| 讲次 | 官网主题 | 官方材料 | 陪学章节 | 状态 |
|---|---|---|---|---|
| 01 | Overview, tokenization [Percy] | [lecture_01.py](https://cs336.stanford.edu/lectures/?trace=lecture_01) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_01.py) · [课堂录像](https://www.youtube.com/watch?v=JuoVZkPBiKk) | 01、02、04 | 已取得官方讲义，已核对目录线索 |
| 02 | PyTorch (einops), resource accounting (FLOPs, memory, arithmetic intensity) [Percy] | [lecture_02.py](https://cs336.stanford.edu/lectures/?trace=lecture_02) · [recording version](https://cs336.stanford.edu/lectures/?trace=lecture_02_recording) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_02.py) · [课堂录像](https://www.youtube.com/watch?v=kuYAsz7zspQ) | 00、03、04、09 | 已取得官方讲义，已核对目录线索 |
| 03 | Architectures, hyperparameters [Tatsu] | [lecture 3.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_03.pdf) · [课堂录像](https://www.youtube.com/watch?v=lVynu4bo1rY) | 04、05、06、07 | 已取得官方讲义，已核对目录线索 |
| 04 | Attention alternatives and mixture of experts [Tatsu] | [lecture 4.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_04.pdf) · [课堂录像](https://www.youtube.com/watch?v=cKSwj_qZ8Jg) | 07、08 | 已取得官方讲义，已核对目录线索 |
| 05 | GPUs, TPUs [Tatsu] | [lecture 5.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_05.pdf) · [课堂录像](https://www.youtube.com/watch?v=izZba4UA7iY) | 09、10、12 | 已取得官方讲义，已核对目录线索 |
| 06 | Kernels, Triton [Percy] | [lecture_06.py](https://cs336.stanford.edu/lectures/?trace=lecture_06) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_06.py) · [课堂录像](https://www.youtube.com/watch?v=xnDHaNUvHBg) | 11、12 | 已取得官方讲义，已核对目录线索 |
| 07 | Parallelism [Percy] | [lecture_07.py](https://cs336.stanford.edu/lectures/?trace=lecture_07) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_07.py) · [课堂录像](https://www.youtube.com/watch?v=SzpOcwdIL0Y) | 13、14 | 已取得官方讲义，已核对目录线索 |
| 08 | Parallelism [Tatsu] | [lecture_08.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_08.pdf) · [课堂录像](https://www.youtube.com/watch?v=6-cXp-aOmdg) | 08、13、14 | 已取得官方讲义，已核对目录线索 |
| 09 | Scaling laws [Tatsu] | [lecture_09.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_09.pdf) · [课堂录像](https://www.youtube.com/watch?v=Q15rhEWZPQ4) | 16 | 已取得官方讲义，已核对目录线索 |
| 10 | Inference [Percy] | [lecture_10.py](https://cs336.stanford.edu/lectures/?trace=lecture_10) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_10.py) · [课堂录像](https://www.youtube.com/watch?v=EfM546A79aM) | 07、15 | 已取得官方讲义，已核对目录线索 |
| 11 | Scaling laws [Tatsu] | [lecture_11.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_11.pdf) · [课堂录像](https://www.youtube.com/watch?v=vTfEyOyzV9E) | 04、17 | 已取得官方讲义，已核对目录线索 |
| 12 | Evaluation [Percy] | [lecture_12.py](https://cs336.stanford.edu/lectures/?trace=lecture_12) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_12.py) · [课堂录像](https://www.youtube.com/watch?v=JpAxdTWQJxM) | 18 | 已取得官方讲义，已核对目录线索 |
| 13 | Data (sources, datasets) [Percy] | [lecture_13.py](https://cs336.stanford.edu/lectures/?trace=lecture_13) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_13.py) · [课堂录像](https://www.youtube.com/watch?v=-qm0ln33G24) | 19 | 已取得官方讲义，已核对目录线索 |
| 14 | Data (filtering, deduplication, mixing, synthetic data) [Percy] | [lecture_14.py](https://cs336.stanford.edu/lectures/?trace=lecture_14) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_14.py) · [课堂录像](https://www.youtube.com/watch?v=5sxHosTLPF8) | 20、21、22 | 已取得官方讲义，已核对目录线索 |
| 15 | Mid/post-training (SFT/RLHF) [Tatsu] | [lecture_15.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_15.pdf) · [课堂录像](https://www.youtube.com/watch?v=2oH6PWPrYFo) | 22、23、24 | 已取得官方讲义，已核对目录线索 |
| 16 | Post-training - RLVR [Tatsu] | [lecture_16.pdf](https://github.com/stanford-cs336/lectures/blob/main/lecture_16.pdf) · [课堂录像](https://www.youtube.com/watch?v=dIFAi87Ws4E) | 24、25 | 已取得官方讲义，已核对目录线索 |
| 17 | Alignment - multimodality [Percy] | [lecture_17.py](https://cs336.stanford.edu/lectures/?trace=lecture_17) · [讲稿源文件](https://github.com/stanford-cs336/lectures/blob/main/lecture_17.py) · [课堂录像](https://www.youtube.com/watch?v=26FtD08ZpOU) | 27 | 已取得官方讲义，已核对目录线索 |
| 18 | Guest lecture: Daniel Selsam | [官网日程](https://cs336.stanford.edu/#schedule) | 28 | 官网有日程，未取得独立讲义／完整字幕 |
| 19 | Guest lecture: Dan Fu | [官网日程](https://cs336.stanford.edu/#schedule) · [录像入口](https://www.youtube.com/watch?v=9EEm4iMAF5s) | 28 | 字幕全文及课堂补充已核对；无独立讲义 |

## 5. 作业中的知识点与原始材料

作业作为知识来源。学习者无需实现整份代码、参加排行榜或购买 GPU。旧版与新版的实验目标有变化，保留各年份来源。2025 使用已核实存在的 spring2025 分支，避免把当前 main 内容误标为旧版。

| 年份／版本 | 材料 | 对应章节 |
|---|---|---|
| 2024 | [cs336_spring2024_assignment1_basics.pdf](https://github.com/stanford-cs336/spring2024-assignment1-basics/blob/master/cs336_spring2024_assignment1_basics.pdf) | 02–06、09、15 |
| 2024 | [cs336_spring2024_assignment2_systems.pdf](https://github.com/stanford-cs336/spring2024-assignment2-systems/blob/master/cs336_spring2024_assignment2_systems.pdf) | 09–14 |
| 2024 | [cs336_spring2024_assignment3_scaling.pdf](https://github.com/stanford-cs336/spring2024-assignment3-scaling/blob/master/cs336_spring2024_assignment3_scaling.pdf) | 16–17 |
| 2024 | [cs336_spring2024_assignment4_data.pdf](https://github.com/stanford-cs336/spring2024-assignment4-data/blob/master/cs336_spring2024_assignment4_data.pdf) | 19–22 |
| 2024 | [cs336_spring2024_assignment5_alignment.pdf](https://github.com/stanford-cs336/spring2024-assignment5-alignment/blob/master/cs336_spring2024_assignment5_alignment.pdf) | 18、23–26 |
| 2025 | [cs336_spring2025_assignment1_basics.pdf](https://github.com/stanford-cs336/assignment1-basics/blob/spring2025/cs336_spring2025_assignment1_basics.pdf) | 02–06、09、15 |
| 2025 | [cs336_spring2025_assignment2_systems.pdf](https://github.com/stanford-cs336/assignment2-systems/blob/spring2025/cs336_spring2025_assignment2_systems.pdf) | 09–14 |
| 2025 | [cs336_spring2025_assignment3_scaling.pdf](https://github.com/stanford-cs336/assignment3-scaling/blob/spring2025/cs336_spring2025_assignment3_scaling.pdf) | 16–17 |
| 2025 | [cs336_spring2025_assignment4_data.pdf](https://github.com/stanford-cs336/assignment4-data/blob/spring2025/cs336_spring2025_assignment4_data.pdf) | 19–22 |
| 2025 | [cs336_spring2025_assignment5_alignment.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_alignment.pdf) | 18、23–26 |
| 2025 | [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/spring2025/cs336_spring2025_assignment5_supplement_safety_rlhf.pdf) | 18、23–26 |
| 2026／当前 | [cs336_assignment1_basics.pdf](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_assignment1_basics.pdf) | 02–06、09、15 |
| 2026／当前 | [cs336_assignment2_systems.pdf](https://github.com/stanford-cs336/assignment2-systems/blob/main/cs336_assignment2_systems.pdf) | 09–14 |
| 2026／当前 | [cs336_assignment3_scaling.pdf](https://github.com/stanford-cs336/assignment3-scaling/blob/main/cs336_assignment3_scaling.pdf) | 16–17 |
| 2026／当前 | [cs336_assignment4_data.pdf](https://github.com/stanford-cs336/assignment4-data/blob/main/cs336_assignment4_data.pdf) | 19–22 |
| 2026／当前 | [cs336_spring2026_assignment5_alignment.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_alignment.pdf) | 18、23–26 |
| 2026／当前 | [cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](https://github.com/stanford-cs336/assignment5-alignment/blob/main/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf) | 18、23–26 |

必须保留的年度差异：

- 2024 A2 以 RMSNorm kernel 为核心；2025／当前 A2 转向 FlashAttention-2，当前版还含 activation checkpointing、FSDP 和并行策略分析。
- 2024 L15 对 LoRA、QLoRA、NF4 与双重量化有专门展开；归入第 23 章。
- 2024 L12–14 的 KenLM、fastText、DSIR、Bloom filter、后缀数组、长上下文、infilling 和自蒸馏不能因新版压缩讲次而漏掉。
- 2025 L17 的排序任务与策略梯度实验归入第 26 章；2026 L17 已换为多模态，二者都保留。
- 2025 A5 包含 MATH、SFT、expert iteration、策略梯度和 GRPO；当前 A5 另列 Dr. GRPO、MaxRL、off-policy 和 GSPO。安全 SFT／DPO 补充材料也保留。
- 2026 新版的 attention 替代路线、TPU、低精度、专家并行、智能体 RL 和多模态归入相应章节。案例只代表对应材料时点。

## 6. 材料缺口、链接修正与更新边界

| 项目 | 当前证据 | 处理方式 |
|---|---|---|
| 2024 Ce Zhang、Aakanksha Chowdhery | 官网列有姓名，未链接独立课件 | 保留第 28 章，待一手课件／录像到位后拆点 |
| 2025 Junyang Lin、Mike Lewis | 官网列有姓名，未链接独立课件 | 同上；不根据讲者身份猜讲题 |
| 2026 Daniel Selsam | 官网日程存在；本次未验证到可用独立讲义／完整录像 | 明确待补，不能据此说讲座不存在或未举行 |
| 2026 Dan Fu | 已取得官方视频的英文字幕和时间戳 | 已并入第 28 章和逐讲资料；独立本课幻灯片 PDF 仍未取得 |
| 2026 L02 recording version | 官网另列可执行 trace；仓库无同名 Python 源文件 | 已读取 trace 内录制版源码，并与当前主讲稿逐项比较；保留修正口径 |
| 2026 安全补充 PDF | 官网仍指向 main 下的 2025 文件名；仓库现有 2026 文件 | 本目录使用仓库实际存在的 2026 PDF，并保留 spring2025 历史版 |
| 公开视频播放列表 | 已枚举：2025 为 17 个、2026 为 18 个 | 依据当前列表登记；2026 Dan Fu 是列表第 18 项、官网日程第 19 讲 |
| 2026 春季之后的进展 | 本次没有做全领域最新论文审查 | 逐章备课时按需核实；不会把“新版课程”称为覆盖截至今日所有进展 |

更新重点：长上下文与混合架构、MoE 与低精度、推理服务、数据混合、推理／智能体 RL、评测污染和多模态。只有能说明现有知识改变在哪里的更新才纳入。保留原始来源日期和验证状态。

## 7. 示范小节：01.01，预测与概率

目标：学习者能区分“最可能出现”和“一定出现”。先不引入神经网络结构。

AI 第一轮只展示：

> 看到“我把牛奶放进了＿＿”，你可能想到“冰箱”，也可能想到“杯子”。我们先把完整的续写当作候选项，暂时省去实际的分词过程。假设有 100 条相同开头的例句，其中 60 条接“冰箱”、25 条接“杯子”、15 条接其他内容。这是人为设计的例子。现在从中随机抽一条，你认为最值得猜哪个续写？这是否意味着它一定出现？请说说理由。

然后等待回答。不要一次性讲完下面的反馈分支。

- 若学习者区分了概率与必然：确认其理由，再换一个简单情境检验是否迁移；之后进入模型输出概率。
- 若认为最高概率等于必然：用剩下 40 条例句追问，先修正这个误解。
- 若认为其他续写出现就证明模型全错：区分一次猜测是否命中，与概率分配是否合理。后续再讲如何通过许多例子评价概率。
- 若质疑例句频率是否就是模型机制：肯定该质疑。说明这里只帮助理解概率，神经模型会通过参数学习规律，并非只数相同句子。
- 若学习者已有扎实基础：缩短该小节，不强制重复；下一点讲条件概率和上下文。

结束记录模板：当前位置［填写］；已确认掌握［附真实回答证据］；待澄清［填写］；下一点［填写］。个人记录仅保存到被Git忽略的 local/learning-progress.md，不提交或自动上传。

## 8. 核验记录

本次以固定脚本提取全部日程条目，逐项检查 57 个条目都有章节归属，51 份核心讲义均取得有效内容，17 份作业 PDF 均可提取文本。PDF 共识别页数见随附来源索引。全部1889页版面与主图已查看，并有页范围内容说明；35份字幕文字已校读、原始音频已机器比对。未人工逐句听校或独立复现全部原实验。

来源索引保存每份核心讲义和作业文件的下载 URL、字节数和 SHA-256。main 链接可变；后续制作正式教材时应固定 commit，并重新核对内容。不要把当前目录状态改写为“全教材已完成”。

## 9. 课堂录像与字幕提取清单

当前两个官方播放列表合计 35 个条目。已取得 34 个英文字幕轨道，字幕末尾时间合计约 45.1 小时。2024 官网未列录像入口，本次没有确认到官方播放列表，不声称已取得 2024 录像。

本次取得的成功轨道都标记为 en-US、is_generated=false。这表示取得的是上传字幕轨道，不能据此保证没有转写错误。原始字幕与时间戳仅作为本地备课材料保存；本分享目录只提供来源和提取状态。

字幕文件已检查非空、段数、时间戳顺序和内容校验值。原始音频全程机器比对，疑难窗口再次转写并检查语义。L03越界幻觉尾已排除；未人工逐句听校。字幕不包含所有板书、图示、光标动作和现场代码变化。

完成课堂内容核对需要：读取对应讲义和字幕，按时间点识别额外解释、演示和实质问答；涉及公式、代码或图示的地方再查看对应画面；把验证后的新增点链接到相应章节。仅提取字幕不能标记为“视频内容已完整纳入”。

| 年份 | 列表序号 | 官方录像 | 字幕状态 | 字幕时长 | 内容对照 |
|---|---:|---|---|---|---|
| 2025 | 1 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 1: Overview and Tokenization](https://www.youtube.com/watch?v=SQ3fZ1sAqXI) | 上传字幕已提取，1498 段 | 78.9 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 2 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lec. 2: Pytorch, Resource Accounting](https://www.youtube.com/watch?v=msHyYioAyNE) | 上传字幕已提取，1291 段 | 79.3 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 3 | [Stanford CS336 Lang. Modeling from Scratch ｜ Spring 2025 ｜ Lec. 3: Architectures, Hyperparameters](https://www.youtube.com/watch?v=ptFiH_bHnJw) | 上传字幕已提取，1990 段 | 87.0 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 4 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 4: Mixture of experts](https://www.youtube.com/watch?v=LPv1KfUXLCo) | 上传字幕已提取，1953 段 | 82.0 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 5 | [Stanford CS336 I Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 5: GPUs](https://www.youtube.com/watch?v=6OBtO9niT00) | 上传字幕已提取，1725 段 | 74.3 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 6 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 6: Kernels, Triton](https://www.youtube.com/watch?v=E8Mju53VB00) | 上传字幕已提取，1860 段 | 80.3 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 7 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 7: Parallelism 1](https://www.youtube.com/watch?v=l1RJcDjzK8M) | 上传字幕已提取，1969 段 | 84.6 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 8 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 8: Parallelism 2](https://www.youtube.com/watch?v=LHpr5ytssLo) | 上传字幕已提取，1250 段 | 75.1 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 9 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 9: Scaling laws 1](https://www.youtube.com/watch?v=6Q-ESEmDf4Q) | 上传字幕已提取，1462 段 | 65.2 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 10 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 10: Inference](https://www.youtube.com/watch?v=fcgPYo3OtV0) | 上传字幕已提取，1464 段 | 82.8 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 11 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 11: Scaling laws 2](https://www.youtube.com/watch?v=OSYuUqGBQxw) | 上传字幕已提取，1743 段 | 78.1 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 12 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 12: Evaluation](https://www.youtube.com/watch?v=x-R5l2HsXqM) | 上传字幕已提取，1510 段 | 80.7 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 13 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 13: Data 1](https://www.youtube.com/watch?v=WePxmeXU1xg) | 上传字幕已提取，1422 段 | 79.0 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 14 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 14: Data 2](https://www.youtube.com/watch?v=9Cd0THLS1t0) | 上传字幕已提取，1394 段 | 79.1 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 15 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 15: Alignment - SFT/RLHF](https://www.youtube.com/watch?v=Dfu7vC9jo4w) | 上传字幕已提取，1696 段 | 74.8 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 16 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 16: Alignment - RL 1](https://www.youtube.com/watch?v=46f2QTDB08Q) | 上传字幕已提取，1847 段 | 80.5 分钟 | 字幕全文已校读；非完整音频听校 |
| 2025 | 17 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2025 ｜ Lecture 17: Alignment - RL 2](https://www.youtube.com/watch?v=JdGFdViaOJk) | 上传字幕已提取，1342 段 | 76.0 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 1 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 1: Overview, Tokenization](https://www.youtube.com/watch?v=JuoVZkPBiKk) | 上传字幕已提取，1509 段 | 79.3 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 2 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 2: PyTorch (einops)](https://www.youtube.com/watch?v=kuYAsz7zspQ) | 上传字幕已提取，1425 段 | 77.3 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 3 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 3: Architectures](https://www.youtube.com/watch?v=lVynu4bo1rY) | 上传字幕已提取，2020 段 | 89.2 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 4 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 4: Attention Alternatives](https://www.youtube.com/watch?v=cKSwj_qZ8Jg) | 上传字幕已提取，1938 段 | 86.3 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 5 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 5: GPUs, TPUs](https://www.youtube.com/watch?v=izZba4UA7iY) | 上传字幕已提取，1925 段 | 78.6 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 6 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 6: Kernels, Triton, XLA](https://www.youtube.com/watch?v=xnDHaNUvHBg) | 上传字幕已提取，1552 段 | 86.6 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 7 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 7: Parallelism](https://www.youtube.com/watch?v=SzpOcwdIL0Y) | 上传字幕已提取，1312 段 | 81.0 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 8 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 8: Parallelism](https://www.youtube.com/watch?v=6-cXp-aOmdg) | 官方字幕未取得；本地自动转写已补 1702 段，待听校 | 80.1 分钟（自动转写） | 字幕全文已校读；非完整音频听校 |
| 2026 | 9 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 9: Scaling Laws](https://www.youtube.com/watch?v=Q15rhEWZPQ4) | 上传字幕已提取，1812 段 | 77.9 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 10 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 10: Inference](https://www.youtube.com/watch?v=EfM546A79aM) | 上传字幕已提取，1509 段 | 85.4 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 11 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 11:  Scaling Laws](https://www.youtube.com/watch?v=vTfEyOyzV9E) | 上传字幕已提取，1808 段 | 77.0 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 12 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 12: Evaluation](https://www.youtube.com/watch?v=JpAxdTWQJxM) | 上传字幕已提取，1376 段 | 78.5 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 13 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 13: Data (Sources, Datasets)](https://www.youtube.com/watch?v=-qm0ln33G24) | 上传字幕已提取，1419 段 | 81.9 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 14 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 14: Data](https://www.youtube.com/watch?v=5sxHosTLPF8) | 上传字幕已提取，1448 段 | 84.7 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 15 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 15: Mid/Post-Training](https://www.youtube.com/watch?v=2oH6PWPrYFo) | 上传字幕已提取，1872 段 | 79.8 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 16 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 16: Post-Training - RLVR](https://www.youtube.com/watch?v=dIFAi87Ws4E) | 上传字幕已提取，1687 段 | 75.8 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 17 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Lecture 17: Alignment - Multimodality](https://www.youtube.com/watch?v=26FtD08ZpOU) | 上传字幕已提取，1264 段 | 77.6 分钟 | 字幕全文已校读；非完整音频听校 |
| 2026 | 18 | [Stanford CS336 Language Modeling from Scratch ｜ Spring 2026 ｜ Guest Lecture: Dan Fu](https://www.youtube.com/watch?v=9EEm4iMAF5s) | 上传字幕已提取，1533 段 | 71.6 分钟 | 字幕全文已校读；非完整音频听校 |

本次字幕缺口：[2026 第 8 讲 Parallelism](https://www.youtube.com/watch?v=6-cXp-aOmdg)。没有改用其他讲的字幕填补，也没有使用账号 cookies 或代理规避封锁。

补缺更新：已取得第 8 讲原始英文音频，并在本地完成 1,702 段自动转写；其性质与官方上传字幕不同，技术词、公式和问答仍需听校。总计 34 份上传字幕加 1 份自动转写，约 46.4 小时。

## 10. 配套学习资料

同目录的 CS336-AI陪学资料.md／html 已补入原创讲解、逐讲课堂条目、往年差异、基础与作业补充和 10 个 CPU 小实验。逐讲对照记录保存核对级别；课堂片段索引覆盖全部已取得的字幕段，自动候选不等于人工验收。

本版状态：公开可得材料已完成所列范围核验。35份文字轨道、35场原始音频机器比对、1889页版面/主图、初轮22帧加追加136帧、231张讲义图像、405个复合输出检查，均保留来源与纠错。完整人工听校、全帧审片、未公开课堂材料仍有边界。
