# Inclusive indices into trace-figures/unique.json. Actual images reviewed as contact sheets.
TRACE_FIGURE_NOTES=[
(0,7,'课程人员/插图与五模块总览；GPT-4报告保留未披露项。FLOPs份额随模型shape变化；emergence阈值指标不能直接证明能力不连续。Stanford例token IDs依tokenizer。'),
(8,19,'总览旧Transformer图含absolute position/dropout，不可当后续RoPE/SwiGLU精确配方。GA100完整芯片128 SM与A100产品108分开。KV cache保留历史K/V而非不再读历史。FP32/16/BF16位宽正确，FP8有E4M3/E5M2两种例。'),
(20,26,'stride图对应连续row-major的4与1；transpose可换stride而不搬数据。batch/sequence两轴分开。22为6帧动画，展示单权重前向乘加2 FLOPs及输入梯度/权重梯度各2，共6；不含优化器、激活和非矩阵项。CTA/grid与wave尾部利用率为示意，非禁止同SM多block。融合图减少中间HBM往返，不等于取消全部读写。'),
(27,39,'node图把跨节点线标NVSwitch是教学简图，应按具体系统区分NVLink域与NIC/RDMA跨节点网。broadcast/scatter/gather/reduce/all-gather/reduce-scatter/all-reduce的输入输出位置已逐图核对；DP切数据、TP切层内维度、PP切层序列。'),
(40,47,'社交帖吞吐是历史自报且word/line不是token。注意力符号表K为KV头数、G=N/K为每KV头query数。GQA速度横轴组数越大越趋MHA，表中效果限该模型任务，不承诺零损失。prefill/decode瓶颈取决batch/shape。'),
(48,55,'MLA压缩缓存，表比较KV元素数而非直接字节；CLA跨层共享KV并显示perplexity/每token缓存的取舍。滑窗/空洞/全局模式不混同；S4卷积等价依线性时不变条件，非任意input-dependent SSM。'),
(56,63,'associative recall按key恢复value；Based/Jamba图为特定混合配方。扩散图是连续embedding模型示例，不代表全部离散扩散。速度散点为历史第三方设置。INT8无浮点指数，示意位色不应暗示有指数。对称量化示例127/5.4约23.52，反量化有舍入误差，图中完全恢复只因十进制显示精度。63首帧近空白但原资产是260帧动画；已逐帧查看，依异常值列及对应权重行分成FP16分支和按向量INT8分支，反量化后两路相加，不把全部矩阵都说成INT8。'),
(64,71,'LLM.int8表有标准误及小幅正负差，不称量化绝无损失。AWQ根据activation选scale而非永久混精保存1%权重。剪枝成本图token单位不直接等于跨模型FLOPs。speculative拒绝后用归一化正部(q-p)，不是直接q或不归一差值；全接受额外采样1token。加速1.92–2.46倍限原配置。'),
(72,77,'连续batch及时移除完成请求；分页逻辑序列映射非连续物理块。仍可能有最后块内部碎片。共享prefix要相同计算条件；分叉写使用copy-on-write/refcount，不能让不同续写覆盖共享KV。'),
(78,87,'R1/Llama4/OLMo/HELM/Arena/Artificial Analysis/OpenRouter均为当年快照，指标、模型版本、推理预算不同。OpenRouter使用量不是全行业份额。Arena有CI和style control；评价危机帖是观点。GPT-2表PPL/BPB/BPC与accuracy不能直接同列比较。'),
(88,95,'LAMBADA末词预测，HellaSwag候选续写，MMLU多选知识与MMLU-Pro更难选项分开。GPQA Diamond流程含专家复核与非专家检验；HLE难度筛选与2500公共题不等于所有数学研究。HLE低分柱不能与不同任务准确率作同一难度刻度。'),
(96,103,'IFEval可机器检查格式/约束，非全体帮助性。AlpacaEval length control仍保留judge偏差。WildBench checklist与pairwise/individual不同。SWE-bench示例补丁仍有2失败，不能只看45通过宣布成功。Cybench是受控CTF容器评测，不推成真实网络泛化；subtask提示预算不同。MLE-bench按竞赛提交打分，不直接证明研究自主性。')]
TRACE_FIGURE_NOTES += [
(104,116,'MLE-bench made/valid submission与medal分开。ARC-AGI颜色变换与成本横轴为评测题例，不推成通用智能总分。安全分类/汽车碰撞为评测覆盖类比；越狱截图只作失效现象。Clio有false positive。污染顺序检验依exchangeability假设；基准本身缺题/歧义会制造错误。nanoGPT speedrun固定loss目标，不是原模型全部任务效果复现。'),
(117,130,'OLMo2预训3.90T与midtraining832.6B/数学10.7B分阶段；Tulu总源样本23.3M与SFT939344/DPO425145不同。Pile raw825.18GiB与effective1254.20GiB由重复epochs产生。Dolma字节/words/tokens列不能互换。DCLM百分比按原始文档数、最终1.4%是保留量，非仅过滤1.4%。fastText表现限1B-1x预算。'),
(131,141,'DSIR按原始/目标比率重采样，不保证原始支持外有新数据。LSH候选概率S曲线不是硬阈值；固定签名总长度时band数变化也变每band行数。GRPO原论文迭代reward model步骤不代表所有规则奖励训练都更新RM。toy奖励曲线最高约3.4，不意味着已学会完美排序。'),
(142,155,'2026课程人员更新；Delphi图十字包含forecast/estimated GPT FLOP equivalent，不能冒充已训练实测点。roofline为上界非保证性能。逐行和示意3+1+4+1+5+9+2+6+5+3=39，越界mask用加法零元；寄存器累加图修正全shared误解。2026节点图明确NVSwitch域与跨节点InfiniBand/Ethernet。'),
(156,177,'DeepSeek-V4图为滑窗+压缩+稀疏选择，不能以图推导未披露参数。2026榜单也保留课程快照日期，不称今天最新。Terminal-bench示例1过2失败；专家/初级用时分布为特定74任务样本。ARC-AGI-1/2与交互3任务及分母不同。GDPval任务交付物与医疗任务分类只说明评测设计，不提供职业建议。'),
(178,191,'robots/ToS图含forecast，不将趋势全部当实测。Stack-v2保留PR基底/diff/comment/review结构；公共可见不等于许可统一。CommonPile各源大小用对数轴，Comma柱图不是全项领先Qwen3。PDF绘制指令解释读取顺序缺失。过滤实验需同时看tokens与每源epochs；RegMix表最优5.34为预测，需大模型验证。配比方法是否限制重复训练依表分开。'),
(192,199,'OpenThoughts 75k问题×16响应=1.2M，不是1.2M独立问题。SWE-smith生成bug与verified tests、SWE-rebench筛选真实任务是不同流程。SWE-Zero execution-free在推理时禁止执行，训练标签仍可来自执行；截图system prompt仅教材被研究对象。表启用execution可改变结果，不能比较不同工具预算后只归因参数量。'),
(200,207,'CLIP双向交叉熵图/代码使用L2归一、exp(t)为逆温度尺度，非直接乘温度。SigLIP逐pair二元loss示例除n而非n²，改变归一会改梯度尺度。分块跨设备仍需交换嵌入/累积完整目标。LLaVA投影视觉特征输入语言模型；caption/box合成QA不代表文字teacher实际看原图。'),
(208,214,'熨衣图模型描述含可见错误，不能把展示回答当ground truth。OneVision任意分辨率切块加全图；7290/8748/6272为具体裁剪/帧配置非通用固定token数。3.2M单图、1.6M混合数据分阶段；Stage1只训projector，后续full model，冻结规则不可跨模型套用。'),
(215,217,'跨图数学示例有算术错误：r=11、角40°，面积应42.2369679而非38.01；不提前舍入总价2660.93，按题逐步两位面积42.24后乘63得2661.12，均非原答案2386.03。GUI多截图描述与足球视频示例为论文展示，未实际控制设备/视频全帧复测。'),
(218,227,'Qwen-VL旧版3阶段冻结图与Qwen2/3-VL区分。原始5B筛到1.4B为约数，不能由分项近似数推严格总和。Qwen2 native resolution动态token与M-RoPE时/高/宽坐标分开；Qwen3图多层DeepStack视觉注入。预训S0仅merger67B，S1/S2约1T各、S3 100B和262144长度是各阶段预算，不是unique data。结果表含思考模式与不同版本，非严格控制变量排名。'),
(228,230,'Chameleon统一混合自回归可输出图像token，再经decoder恢复图；与只输出文本VLM不同。VQ-VAE最近码与重建/commitment/codebook损失分开，量化不保证原图无损。鸟类回答是模型展示，未据此验证全部生物事实。')]
