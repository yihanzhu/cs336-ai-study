"""Authored findings from actual rendered page inspection, separate from mechanical inventory."""
PDF_REVIEWS=[{
 'file':'2026-lecture-08.pdf','source_pdf_sha256_expected':'4c4d7674b333693f20f328935d84dd8a073362c9725c5e05302b6650751ad4d1','pages_viewed':list(range(1,74)),
 'scope':'All 73 page layouts and main diagrams inspected in rendered contact sheets. Selected formulas and dense tables additionally checked using full-page images, extracted text and primary references. Small labels and every cited experiment were not independently reproduced.',
 'semantic_signoff':False,
 'findings':[
 {'pages':[1,2,3,4,5,6],'finding':'本讲有网络基础、并行方式及大模型配方三部分。p4–6历史计算/模型尺寸/互连图是有日期的示例；一张卡放不下特定模型不等于任意小模型都需集群。旧硬件图不能当当前平台规格。','remaining':'历史图中每个小字和曲线点未逐项向原报告核实。'},
 {'pages':[7,8,9,10,11,12],'finding':'图示all-reduce可以分成reduce-scatter再all-gather；带宽最优说法须有拓扑、大消息等前提。TPU图从toroidal mesh扩展到TPU8的交换网络，不能泛化TPU永远只有mesh。','remaining':'厂商对比表的所有规格与宣传口径尚未独立核实，不作为选型建议。'},
 {'pages':[13,14,15,16,17,18],'finding':'DP切batch并同步梯度；平均损失若改成梯度求和，要同时说明学习率/归一化约定。16B/param来自BF16权重2、梯度2、FP32主权重4与两矩各4，不含激活。ZeRO表是每GPU持久训练状态，不是全系统内存。','remaining':'不把此精度账本作为所有优化器与实现的固定常数。'},
 {'pages':[19,20,21,22,23,24,25,26,27,28],'finding':'ZeRO-1/2可保留近似2倍参数量通信，但仍有时延与调度开销；ZeRO-3使用两次all-gather和一次reduce-scatter的示意。p26的重叠不是任意网络都能完全遮住通信。p28改用12B/param后，8×80GB算得上限6.67B/16B/24.62B/53.33B，未预留激活与临时buffer，不能称为实际可训练尺寸。','remaining':'算式上限可核对，实际峰值显存需完整模型/实现和运行。'},
 {'pages':[29,30,31,32,33,34,35,36],'finding':'p30“stage1/2不让memory scale”指完整参数复制瓶颈，不能抹去优化器/梯度分片收益。PP图含F/B依赖及microbatch，bubble/useful≈(p−1)/m；若分母换总时间则为(p−1)/(m+p−1)，假设阶段平衡且忽略其他开销。','remaining':'图中每个实验点没有本机GPU复现。'},
 {'pages':[37,38,39,40,41,42,43],'finding':'交错PP用更多通信换利用率。Zero-bubble将输入梯度B与权重梯度W分开，W必须在相关参数更新前完成，不能“任意时候”。TP图的列切上投影/行切下投影使局部GELU后汇总部分和；p43通信公式须注明dtype、层、microbatch及前反向口径，不能只拿裸数8比较。','remaining':'所有零气泡时槽的小编号未逐格复算。'},
 {'pages':[44,45,46,47,48,49],'finding':'峰值内存是动态量。图式每层激活sbh(34+5as/h)、TP后的sbh(10+24/t+5as/(ht))、TP+SP+选择性重算后的34sbh/t依特定训练架构/精度。SP切norm/dropout等逐token部分；不能当FlashAttention下仍需保存整个s²矩阵的通用结论。','remaining':'34、24、10等系数需对应所引实现，不推广成所有Transformer的下界。'},
 {'pages':[50,51,52,53,54,55,56],'finding':'EP按专家分配并all-to-all路由，CP/ring attention则切上下文块。MoE parallel folding的attention与FFN进程组可不同，不能无条件把DP×TP×PP×CP×EP相乘求GPU总数。汇总表中EP/PP/CP的0表示未启用，不是并行度因子0；规范计算用1表示未切该轴。','remaining':'p56的具体交叉点是特定mesh/模型/带宽例子，不当通用阈值。'},
 {'pages':[57,58,59,60,61,62],'finding':'先让模型放下再扩DP是经验起点。p61图注明64 A100 GPUs，下面却写64 machines；教学统一为该实验64张GPU与PP8×TP8，不说64台机器。p62重算用额外算术换更大可行batch，吞吐收益随batch/并行配置变化，不能称其计算免费。','remaining':'引用的2021实验不是2026当前硬件推荐。'},
 {'pages':[63,64,65,66],'finding':'OLMo、DeepSeek、Yi和Llama使用不同状态精度/分片/拓扑，不能只看一个并行倍数横排优劣。Llama405B表中16384=TP8×CP1×PP16×DP128；长上下文阶段改为CP16、DP8仍为16384。','remaining':'各模型全文配方及后来版本尚未逐项追溯。'},
 {'pages':[67],'finding':'故障表继承原论文不一致：54天466次中断中419次非计划；表内计数合计419，但148/419≈35.32%，不是同栏30.1%。百分比栏合计94.9%。保留计数、指出冲突，不猜哪个原数该改，也不把148称为全部中断。','sources':['https://arxiv.org/html/2407.21783v1#S3.SS3.SSS4']},
 {'pages':[68,69,70,71,72,73],'finding':'p71标题225B-A22B是笔误，Qwen官方名称为235B-A22B，表内截图也写235B。p69/71引用Megatron配置属于文档示例，不能当作者实际训练配置。p70 PP=0仅表示未启用；EP与CP分组有重叠，不能直接乘表格所有数字推卡数。Gemma27B例子768×8=6144只适用于该配方。','sources':['https://qwenlm.github.io/blog/qwen3/'],'remaining':'表内未知值??继续保留；不从别的模型填补。'}]
}]
