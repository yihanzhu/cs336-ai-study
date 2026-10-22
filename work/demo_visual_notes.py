# Each listed frame was visually inspected; sampled frames do not verify entire video.
DEMO_VISUAL_NOTES={}
DEMO_VISUAL_NOTES['2025-04']={'frames':[1257,1279],'finding':'画面为MoE结构与Top-k/hash routing幻灯片，非运行演示；替代MLP与attention heads两支分开，Top2归一化/合并依具体路由。'}
DEMO_VISUAL_NOTES['2025-05']={'frames':[4181,4197,4272,4296],'finding':'画面前两张为safe/online softmax两段伪代码，后两张为FlashAttention Q/K/V分块与rescale图；完整softmax输出仍要最后归一，而加权V可在线维护O，不把二者混同。'}
DEMO_VISUAL_NOTES['2025-06']={'frames':[625,860,1255,1442,1465,1915,2036,2054],'finding':'代码sleep(50/1000)为50ms。实际profiler表有CPU self/total与CUDA列及dim128比较；Nsight窗口可见CPU/CUDA/NVTX轨道。画面是原课结果，未本机运行；不将不同列相加为端到端，也不凭截图推导CPU耗时为何下降。'}
DEMO_VISUAL_NOTES['2025-07']={'frames':[1517,1542],'finding':'画面为ZeRO1步骤图，reduce-scatter按块求和各rank持一块，all-gather更新权重；不是当前机器通信实测。'}
DEMO_VISUAL_NOTES['2025-08']={'frames':[25,465,1355,1381,1975,2465,2754,2787,3381,3405],'finding':'25s只拍讲者，不能证明屏幕内容；其余可见allreduce示意、reduce_scatter_tensor代码、TP轴切分与PP recv/send。2787s及3405s明确Backward pass: homework exercise，PP另写未处理overlap，直接证实仅forward骨架；代码画面不等于完整多卡训练。'}
DEMO_VISUAL_NOTES['2025-10']={'frames':[1685,1930,3753,3770],'finding':'1685s写可完美重叠/忽略开销的假设，1930s符号latency/throughput输出，不是真实GPU测速。3753s引用Inception演示75对14iterations，3770s为第三方speed-quality散点；迭代数不等于统一硬件时延。'}
DEMO_VISUAL_NOTES['2025-11']={'frames':[1761,1786],'finding':'采样画面是DeepSeek2024论文标题及讲者，未见muP/WSD具体图；口述笔记仍由字幕支持，不把这两帧当该图视觉互证。'}
DEMO_VISUAL_NOTES['2025-12']={'frames':[1985,2225,2249,2480],'finding':'MMLU讲义与HELM实际prompt详情页可见，含计算机题few-shot Answer:D和模型标识；后切MMLU-Pro。显示了查看benchmark样本/提示，非现场训练或完整重算榜单。'}
DEMO_VISUAL_NOTES['2025-13']={'frames':[25,900],'finding':'25s课程标题/数据主线代码，900s Common Crawl概览；这两帧未捕获随机网页内容，不把“计划打开网页”视为截图证实。'}
DEMO_VISUAL_NOTES['2025-14']={'frames':[825,1000,1072,1089,1525,1850],'finding':'DSIR流程图、4个hash bins代码与实际弹出prob=0可见，1072s probs=[0,.4,0,.6]说明此toy零来自未平滑零频桶，不仅可能下溢。1525s/1850s为LID和GPT3质量数据讲义，未捕获hello误分类输出，后者仍由字幕/trace支持。'}
DEMO_VISUAL_NOTES['2025-15']={'frames':[25,225,265,279,2354,2376,3757,3775],'finding':'前段为目标与SFT人工示范数据流程；截图不含程序执行。后段学习格式/标注者选择表格按各帧另看，demonstration不得自动标成软件演示。'}
DEMO_VISUAL_NOTES['2025-15']['finding']='前段为目标与SFT人工示范数据流程；2354/2376是RLHF后两阶段流程，3757/3775是CAI response/critique/revision流程，均为幻灯片，无程序运行。采样未见emoji规则/标注者表，相关问答仅依字幕，不夸大图像互证。'
DEMO_VISUAL_NOTES['2025-16']={'frames':[3539,3562],'finding':'画面为DeepSeek-R1 language-consistency reward的论文摘录，不是现场翻译模型运行；可读性与准确率取舍要保留，不能证明CoT忠实。'}
DEMO_VISUAL_NOTES['2025-17']={'frames':[30,227,1014,1248,1280,1369,2595,2785,4142,4331],'finding':'1369s屏幕数5.3229/1.1547与torch.std源码对应，纠正variance命名。2595s明确各位置独立非autoregressive；2785s显示center/normalize代码。4142s日志与4331s训练loss/reward双图可见：mean reward改善不等于每次输出完全排序成功，不能把toy日志当大模型RL基准。'}
DEMO_VISUAL_NOTES['2026-01']={'frames':[1075,1095,3085,3130],'finding':'1075/1095仅显示agent example trace链接，未展示trace内容。3085s Marin预登记forecast与已测loss图，灰色目前训练中点不当已完成结果；3130s说明API based on previous runs，非现场新训练。'}
DEMO_VISUAL_NOTES['2026-02']={'frames':[1060,1065,1810,1815,3819,3843,4210,4230],'finding':'GPU示意和cuda可用性分支只见源码未GPU结果；1810/1815的two weeks实际公式只乘7天，需×2。3819/3843 backward einsum轴与autograd allclose展示；4210/4230屏幕旧memory公式重复乘bytes，必须按当前修订源码和口头更正，Adam两moment另8B/param。'}
DEMO_VISUAL_NOTES['2026-05']={'frames':[2585,2617],'finding':'2585仅讲者及非本课白板残留，不收背景白板为课程公式。2617的幻灯片标题MXFP4但底部16元素/E4M3实际更接近NVFP4，需要按OCP/NVIDIA一手规格勘误，不能照抄。'}
DEMO_VISUAL_NOTES['2026-06']={'frames':[3075,3170],'finding':'GitHub triton_gelu-ptx.txt实见.reg/.param与多条mul.f32/add.f32，证明讲者查看生成PTX；并非SASS或芯片直接执行PTX文本，未证明寄存器实际分配/吞吐。'}
DEMO_VISUAL_NOTES['2026-07']={'frames':[267,282,1257,1287,2419,2437,2525,2553,4248,4276],'finding':'267/282明确独立stdout链接与tracing关闭multiprocessing；1257是概念reduce到rank0/allgather，不能当allreduce返回仅rank0。2437实际stdout各rank allreduce[6,10,14,18]，与来源文本一致。1287/2525/2553/4248/4276只拍讲者，未证PP屏幕细节，PP由源码/字幕支持。'}
DEMO_VISUAL_NOTES['2026-09']={'frames':[1929,1951],'finding':'LSTM/Transformer图横轴明确Parameters(non-embedding)，纵轴Test Loss；必须修正先前口述笔记称此特定图为计算–损失曲线。图内所测Transformer更低loss，不证明任意模型/配方。'}
DEMO_VISUAL_NOTES['2026-10']={'frames':[775,985,2125,2410,3305,3315,4248,4272],'finding':'775s GQA N=K*G注释混乱，以现场口头K组/G每组head更正；985是BF16算术强度BDF/(BD+BF+DF)，非测量。2410公式latency来自假定带宽，不能宣称真实服务延迟。其余MLA/activation画面另见逐帧，不能补未展示的消融。'}
DEMO_VISUAL_NOTES['2026-10']['finding']+=' 3305/3315/4248/4272仅拍讲者，不能说已从画面验证MLA消融或activation图。'
DEMO_VISUAL_NOTES['2026-12']={'frames':[2160,2178],'finding':'Arena历史榜单截图与谁在投票/偏差/刷票问题代码，非软件运行演示，也不能引用为当前排名。'}
DEMO_VISUAL_NOTES['2026-13']={'frames':[2634,2656],'finding':'两帧仅讲者，背景白板为其他内容，不收作本课公式；许可/爬虫问答仍依字幕与ASR，图像不增加法律证据。'}
DEMO_VISUAL_NOTES['2026-15']={'frames':[367,382,1592,1615,2734,2745,3183,3197],'finding':'SFT示范流程、G-V gap偏好条图和demographics表均为讲义。1592/1615幻灯片被裁切，完整图用PDF核验；不是现场finetune/程序演示。'}
DEMO_VISUAL_NOTES['2026-16']={'frames':[345,361,664,679],'finding':'机器人动画两帧位置不同证实引用视频播放，非本课现场训练。PPO三曲线为已有实验图，左综合目标/中RM/右negativeKL不同；本机未重跑该实验。'}
DEMO_VISUAL_NOTES['2026-17']={'frames':[3561,3588,3895,3915],'finding':'3561为VLM benchmark表，其他多拍讲者/被裁切slide；不由此断言视觉编码器具体参数量，需原模型架构表，视频理解不等同视频生成。'}
DEMO_VISUAL_NOTES['2026-18']={'frames':[448,463,636,661,3261,3286,3354,3377,3439,3467,3745,3887,4095,4135],'finding':'Dan Fu嘉宾段含Austin Silveria署名Lifetime of a Token，应保留section署名，不误当另一个课程视频。SLA图TTFT500ms/TBT30ms是示例目标且有生成工具标记；不得当行业通用指标/实测分位。3354数据参数双扩展，3377/3439 Parcae为140M/370M固定参数及isoFLOP recurrence曲线，指数约.40/.38与tokens.77/.78限该实验。3467/3745/3887只拍讲者，未显示其描述3D联合图。4095结尾Megakernel H100/B200图为给定设置；未现场执行新kernel，也未实现循环块megakernel愿景。'}
DEMO_VISUAL_NOTES['2025-02']={'frames':[2298,2325],'finding':'补取两帧均为2BDK矩阵乘FLOPs推导，CUDA/CPU不同shape分支源码可见，未有测速结果；严格加法数可少一项但通常按2BDK估算。'}
