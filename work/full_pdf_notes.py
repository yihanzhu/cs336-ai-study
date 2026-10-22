"""Authored page-level visual/content review notes. No full GPU reproduction claim."""
PDF_FULL_NOTES={}
PDF_FULL_NOTES['2026-lecture-03.pdf']={
'pages':67,'findings':[
(1,9,'结构从原始encoder-decoder转到decoder-only、pre-norm、RoPE和门控FFN。模型总表用于观察课程时点样本的趋势，不把小字截图当闭源模型已证实规格，也不把“大多”改写成架构必须如此。'),
(10,13,'图中pre-norm保持未归一化的残差直通路径；non-residual postnorm与pre-norm后再加norm不等于原始post-norm。BLEU、梯度和loss曲线来自所引实验；可以解释稳定性机制，不能保证所有任务提高相同幅度或免warmup。'),
(14,19,'p14把RMSNorm分母画成sqrt(||x||²+epsilon)，缺少按维数取均值的1/d。教学用sqrt((1/d)sum(x_i²)+epsilon)，并区分可学习gain与bias。p15–17的FLOPs/运行时间说明搬运成本，数字只属于所引模型/实现，不把少0.17% FLOPs当作只能少0.17%耗时。原始依据：https://arxiv.org/html/1910.07467v1 。'),
(20,26,'ReLU、GELU、Swish与GLU的图和公式已对应；Swish=x·sigmoid(x)，SwiGLU还乘另一线性分支。相同参数预算时三矩阵门控FFN的中间宽度约为两矩阵FFN的2/3。表格支持特定实验收益，不保证某激活在所有规模第一；门控也不是模型工作所必需。'),
(27,29,'串行block应展开为h=x+Attn(Norm(x))，y=h+MLP(Norm(h))；p28引用截图的串行简式没有明确保留第二次相加前的attention残差，不当完整实现公式照抄。并行形式的MLP与Attn共用原输入，改变计算图，不只是把原串行程序同时启动。截图15%属于PaLM实验，非普遍加速。'),
(30,35,'正弦相加有内容/位置交叉项；RoPE给Q/K的成对坐标做旋转，使旋转点积只依相对位移。不是把整个语言模型变成无条件平移不变。代码向rotary_emb传value_states以取得形状/设备等，不等于这里也旋转V。二维对子旋转与三维视觉位置轴是不同概念。'),
(36,41,'4与8/3是常见FFN宽度起点，T5的64倍只是特殊配方。p38表列PaLM为4倍，已经是其理论2/3预算换算的例外，不能说所有GLU都必须8/3。50M曲线的宽平区不保证任意规模1–10倍都一样好。'),
(42,43,'头数乘head_dim不必等于model_dim。p43 PaLM行写258、ratio1.48，而作者表说明48头、head_dim256、model_dim18432，故按本页其他行的head_product/model口径应为2/3；反比才是1.5。不能把258或1.48当正确规格。依据：https://arxiv.org/html/2204.02311v5#S2.SS1 。'),
(44,47,'宽深比需看参数、并行和延迟；图中宽平区不是数学最优常数。词表表格是特定版本，不同tokenizer的per-token loss不可直接比。多语言和代码会改变切分需求，不能由“单语”就推出固定词表上限。'),
(48,51,'单遍训练也可能记忆；weight decay可影响优化动态，不能从一组train/validation散点说它绝无防过拟合作用。未报告dropout不等于已证明完全没用，闭源模型尤其不能反推。'),
(52,56,'区分输出z-loss、QK norm与soft-cap。最大化log-likelihood时减正惩罚，最小化NLL时加正惩罚；z-loss为alpha(logZ)²。softcap=c·tanh(logits/c)，范围(−c,c)而非只设上界。QK norm调Q/K不是归一化整个attention矩阵；表中perplexity变化不等于运行速度变化。'),
(57,60,'KV缓存保存历史K/V而非QK完整矩阵。p58–60算术强度公式取整个括号的倒数，并有n相对d等近似条件；prefill与逐步decode的总量不能混用，长到n不再远小于d时不能仍忽略attention FLOPs。'),
(61,63,'MQA的“one dimension”是共享一个KV头，不是每个K/V只有一个标量；每头维度仍可为k。GQA保留多组KV。图中质量/延迟折中只证明所测设置，不许改现有模型的KV头数后默认无需训练或质量无损。'),
(64,67,'图中的稀疏/滑窗遮罩保留不同连接，堆叠可传播信息但不是一层全注意力等价替代。Command A示例区分NoPE全局层和RoPE滑窗层；其他截图是带日期的混合方案，不按未公开闭源实现作确切架构结论。')],
'primary_sources':['https://arxiv.org/html/1910.07467v1','https://arxiv.org/html/2204.02311v5'],
'limits':'Every page and main teaching figure inspected. Historical model catalogue tiny cells are not independently re-verified specifications; cited experiments are not rerun. Unsupported exact product claims are excluded from teaching conclusions.'}
PDF_FULL_NOTES['2026-lecture-04.pdf']={'pages':60,'findings':[
(1,5,'Attention成本图需同时看n、头维度和硬件。去softmax后(QKᵀ)V=Q(KᵀV)，是结合律；原softmax不能随意这样交换。因果递推S_t只含前缀，不能直接用全序列KᵀV泄露未来；无归一化线性式与核化/带分母版本不同。'),
(6,11,'Minimax、Mamba、Gated Delta等混合图表达了状态压缩与全注意力的配合，不保证任意混合比例都更好。p9的I−βkkᵀ只有相应归一化/β条件才是正交投影；β=0停止新信息注入但γ仍可衰减旧状态，不等于整个状态原样保留。合成回忆/语言曲线分开解释。'),
(12,13,'DSA轻量indexer仍可能给所有位置对打分，不能仅凭主注意力选top-k便宣称总算法严格线性。适配需训练indexer及后续阶段，不是给任意冻结模型加选择器就默认质量无损。评测柱图保留其数据、协议及课程时点。'),
(14,24,'MoE以较少激活参数承载更多总参数，但router、路由通信、存储不免费。OLMoE速度图应分tokens/FLOPs与wall clock，不能只复述7×为通用速度。GPT4图上本就带问号，不能把传闻写成确证。多节点可用性、稀疏负载与训练稳定性共同影响收益。'),
(25,30,'token choice沿专家轴为每token选top-k；expert choice沿token轴为专家选token；global assignment是一次路由匹配优化，不是全训练过程全局最优。hash routing、REINFORCE和BASE匹配是不同路线。k值随模型版本及是否计共享专家变化，不按“DeepSeek7”这类混称写固定规则。'),
(31,35,'top-k前softmax与只在选中集合归一化不同。p31将DeepSeekV3概括成top-k后softmax不准确：V3用sigmoid affinity再对所选分数除和，并非对所选原logit做softmax。细粒度/共享专家消融并不一致，p33有收益、p34没有共享专家收益，不能只留单方。'),
(36,39,'离散top-k索引本身不光滑，不表示所选gate权重不能反传。p38 Gaussian加法噪声与p39 uniform乘法jitter不同；KeepTopK对未选项填−∞再softmax，不能填零当同义。噪声有时改善稳定性，但不是最佳方法保证。'),
(40,43,'Switch辅助损失αNΣf_iP_i中f_i视为离散统计常量，偏导对P_i为αNf_i；再传播softmax会耦合其他专家。V3的路由偏置只影响选中，输出权重用未加偏置的分数；仍有sequence-wise辅助项。去负载均衡图需同时看模型loss和负载图，不只看训练曲线。'),
(44,47,'专家并行按expert组织变长token组，MegaBlocks是块稀疏矩阵乘，不等于硬件2:4稀疏。LatentMoE下投影以降低路由字节，同时改变结构。容量约束和丢token可令输出依同batch其他请求；dropless移除这一来源，不保证所有数值/调度完全确定。'),
(48,50,'router FP32和router z-loss针对数值与训练稳定性；不能把输出层z-loss、负载均衡loss混成同一个东西。小微调集可能过拟合，只训练部分dense MLP与增加数据是不同经验方案，不能保证彻底解决。'),
(51,56,'Upcycling先复制已有权重再继续训练，并不自动产生分工。p56页题DeepSeekV3，副标题误写V2；Fine-grained(258)与p35的256冲突，作者报告为256 routed、选8、另1 shared，不能把258继续写入教材。表中专家数量须分routed/shared/总激活。'),
(57,60,'MLA用latent压缩KV，带RoPE的额外key分量也要缓存；不能只缓存latent后删掉全部位置分量。矩阵吸收取决于相对旋转与投影能否交换。MTP训练目标与推理时speculative使用是两个环节，原V3例子仅加一步预测，不是所有示意多步模块都实际运行。')],
'primary_sources':['https://arxiv.org/html/2412.19437v1#S2.SS1.SSS2'],
'limits':'All 60 pages/main figures read. Benchmark charts are treated as original authors’ dated experimental claims, not current rankings or locally reproduced GPU speeds. Closed-model rumors are not teaching facts.'}
PDF_FULL_NOTES['2026-lecture-05.pdf']={'pages':55,'findings':[
(1,8,'矩阵性能曲线把算术强度、tile与wave分开；计算增加常能降训练loss是经验关系，不是硬件一升级所有能力必增。CPU低延迟与GPU高吞吐是设计取向；tensor-core峰值与普通算术峰值不能跨dtype随意比较。'),
(9,14,'p9明确是full GA100设计128 SM，不能误判这张图有错；实际A100产品108 SM，p48用108正是产品例子。必须区分芯片完整设计与启用单元。CUDA local memory逻辑线程私有、物理可在设备全局内存；普通跨block交换通常经global，但现代cluster shared等扩展也使“只能global”不宜绝对化。TPU对应表是特定型号类比，不是架构单元一一同义。'),
(15,24,'warp调度隐藏延迟不等于线程没有成本。roofline吞吐上界=min(计算峰值,带宽×AI)，计算与搬运时间的max是理想重叠下界。分支若warp内分歧可串行执行路径，统一分支不必两侧都跑。低精度还要算scale、转换和数值误差。'),
(25,26,'p25把8 bytes/FLOP、4 bytes/FLOP称intensity，实际上是常用算术强度FLOP/byte的倒数。按其简化ReLU账本应为FP32 1/8、FP16 1/4 FLOP/byte；写回仅负值只适用于相应原地实现，普通独立输出仍需写全部元素。归约、累加、状态不一定随输入一起低精度。'),
(27,29,'p27–28 MXFP8每32值配E8M0 scale并需处理转置重分组。p29虽写MXFP4，却给每16值E4M3 scale，这是NVFP4式特征；OCP MXFP4规定32值/E8M0。NVFP4还含全tensor FP32第二级scale。两者E2M1元素范围相同但scale方案不同，不能混名或只记4bit。'),
(30,36,'融合工厂图减少中间张量回HBM和launch，并非所有运算被代数消掉。sin²+cos²例子教学看5算子融合，不假定编译器总直接返回1。三sigmoid重算图8→5是简化HBM读写次数，增加算术且依正确重建反向依赖，不是通用5/8内存或速度保证。'),
(37,39,'coalescing按同warp同一访存指令各lane地址观察；单线程沿行连续不保证全warp合并。图中4字节burst是玩具，真实事务/sector依架构和访问宽度；不要求所有线程同一个元素，也不能把所有coalesced访问限定为单个128-byte事务。'),
(40,44,'tiling沿共享维度分阶段加载并累积输出；N/T访问公式忽略cache等。边界mask、共享内存、寄存器、并发block与对齐共同限制tile，不能只选越大越好；不整除也可以mask/换内核，padding是办法之一。'),
(45,48,'画面确认nanoGPT示例50257→50304（64的倍数），避免旧字幕漏位。物理padding与扩充有效token集合分开；若要求原概率完全不变需处理额外logit。1792→1793配256×128 tile从98→120块，在108SM且设定驻留条件下出现新wave；不推成所有尺寸增加都翻倍。'),
(49,55,'p51把attention写为3次矩阵乘不精确：核心QKᵀ与P·V为2次，Q/K/V投影另计。FlashAttention用分块与在线softmax算精确attention；新最大值出现时分母l及加权V分子o都需重缩放。p50显示FLOPs可从66.6G升75.2G而HBM下降，不代表所有shape相同倍速。课堂未展开完整backward，作业A2保留该知识而不强迫大作业实现。')],
'primary_sources':['https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/','https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf','https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html'],
'limits':'All pages/main diagrams inspected; GPU benchmarks are dated author-reported measurements, not locally reproduced. Historical latency tables are illustrative, not device-independent constants.'}
PDF_FULL_NOTES['2026-lecture-09.pdf']={'pages':57,'findings':[
(1,6,'Scaling是资源分配预测工具，不是只背模型配置。有限假设类上界需IID样本、假设数和置信度等条件；图中的上界不等于实际loss，也不证明神经模型严格达到同一指数。'),
(7,16,'历史学习曲线说明可拟合的渐近形状与多个函数族；不是所有数据范围都严格幂律。若L=L∞+aD^−α，严格线性的是log(L−L∞)对logD；熵底明显时不能仍声称logL永远直线。更多数据的期望最优风险与某次有限训练得到的loss也需区分。'),
(17,20,'均值估计在IID有限方差时MSE=σ²/n，标准差σ/√n。p19二维网格例可得n^−1/2，但所写d维n^−1/d不是一般非参数最优率：Lipschitz回归的局部均值常有bias²约h²、variance约1/(nh^d)，平衡得n^−2/(d+2)。须说明平滑性/风险定义；intrinsic-dimension图是候选解释，不是已证明LLM指数。'),
(21,24,'混合数据改变截距而不改斜率只是所引设置；DataDecide与mixing-law预测是两种选配方方法。重复数据的有效量曲线有饱和与拟合常数；“4 epoch近似新数据”是该实验观察，不是通用许可。来源权重、唯一token和重复次数分开。'),
(25,27,'p25把scaling law叫lower bound容易误解：已达到loss是未知最优可达loss的上界，经验下包络也不是不可突破的理论下界。图里正则化/ensemble能优于某标准配方，并不违背数学定理。过滤随预算变松与重复收益变化需联动看。'),
(28,36,'比较架构/优化器时检查预算、数据、总参数与非embedding参数；形状相近的loss曲线不保证下游或极大规模同排序。embedding与非embedding对计算和表达作用不同，MoE还须分总/活跃参数。1层模型对比不等于深度永远无效。'),
(37,39,'Critical batch取达到同一loss所需steps S与examples E的折中，Bcrit=Emin/Smin。图式平衡时S≈2Smin、E≈2Emin，不是只用少量steps衡量效率。目标loss降低时示例Bcrit增大；实际还看吞吐、显存和优化效率。'),
(40,42,'μP同时规定初始化、分层学习率和输出缩放；图示宽度迁移有参数化条件，不能把不同层LR或variance/SD混用。训练loss可预测不保证所有下游指标单调，p41两图明确显示差异。'),
(43,48,'固定训练预算常用C≈6ND，IsoFLOP扫N时相应D=C/(6N)；不能把6漏掉或把D也固定。Kaplan主张N∝C^.73、D∝C^.27，不能口述反转。三方法是训练曲线包络、固定FLOPs扫描及联合曲面拟合，控制训练日程后才比较。'),
(49,53,'p49“least squares”是宽泛概括；Chinchilla方法3实际对预测/观察的log loss最小化Huber，并用L-BFGS和多初始化。残差均值必为0不适用于这类任意拟合。所引复核质疑拟合和区间，课程置信数字不作为自然常数。依据：https://arxiv.org/html/2203.15556v1#S3.SS3 。'),
(54,57,'训练最优与服务总成本最优不同，重复使用多时可偏向小模型多token预训练；“overtrain”在此是相对训练计算最优多训，不等于验证loss已上升的过拟合。20tokens/param与其他模型比率均带配方/年代，不能用它们决定所有模型。')],
'primary_sources':['https://arxiv.org/html/2203.15556v1#S3.SS3'],
'limits':'All pages/main figures read; experiment-point uncertainty and original scaling fits are not re-estimated without raw runs. Exact extrapolation is conditional, not an audited promise of future model performance.'}

PDF_FULL_NOTES['2026-lecture-11.pdf']={'pages':58,'findings':[
(1,12,'公开训练案例是带日期的配方。MiniCPM同时调整初始化、embedding、输出与深度缩放；9M至499M的迁移再到2.4B是有限外推，不能只改初始化就叫完整muP。'),
(13,18,'WSD复用稳定阶段前缀，再从不同checkpoint做衰减分支，仍有各分支成本。中间cosine checkpoint不等于为短预算专门衰减后的终点，不能说任何提前停止都无效。p18 Ultratext面板95.60与论文正文平均192统计范围不同，保留口径，不把20或192当普遍最优比例。'),
(19,30,'DeepSeek用M=每token计算量、C=MD，M不等于参数量N。bits/byte与per-token loss不能直接横比。LR/批量二维图的平坦谷底不保证唯一最优值；Qwen、Kimi、Hunyuan和MiniMax图需分总/活跃参数及架构。'),
(31,41,'StepFun图中的U形截面不能证明所有网络联合凸优化。batch与数据量的关系、优化器相对收益会随预算和配方变；AdamW和Lion应分别调LR与weight decay。拟合很好仍可在更大规模失败，held-out检验是必要证据。'),
(42,45,'Muon对带动量矩阵做近似正交化，Newton–Schulz有限步是近似；UVᵀ与矩形/秩亏矩阵条件需要说明。nanoGPT速度赛与大型Kimi训练不是同一控制实验，不能据单次成功宣布处处最优。'),
(46,50,'每坐标O(1)与向量范数O(sqrt(width))不同。谱范数乘输入范数是上界，p47的约等于不能视为任意向量恒等式。深线性、固定深度、单样本rank-1 SGD是推导假设；Adam与SGD分层LR不同，初始化方差与标准差也要分开。'),
(51,58,'muP经验迁移需明确宽度、头维度、输出缩放与gain参数化。该研究RMSNorm gain失败不等于所有RMSNorm都禁用；Lion用sign而非sine。粗网格最低点与连续最优值不同，一次10B验证支持该配方，不能保证所有深度/批量都可迁移。')],
'primary_sources':['https://arxiv.org/html/2404.06395v1'],
'limits':'All 58 pages/main figures inspected. No rerun of scaling sweeps or GPU optimizer benchmarks. Dataset-specific fit ratios retain their original scope.'}
PDF_FULL_NOTES['2026-lecture-15.pdf']={'pages':65,'findings':[
(1,9,'从续写到指令控制，SFT是下一token训练目标配合不同数据。三阶段RLHF图是经典路线，不能说所有当代模型都严格依此顺序；闭源数据报道不当成已公开可核验配方。'),
(10,18,'FLAN短任务、Alpaca、OpenAssistant长回答及工具调用数据目标不同。长度/列表偏好会混淆人类和模型评审，偏好胜率不等于事实或推理能力；表格含各数据源带来的提升和退化，不能只留平均分。'),
(19,27,'示范引用可能教格式也可能灌入未知事实，需核验引用本身。SFT会增加幻觉是特定知识学习实验的风险，不是所有新知识都绝对学不会。500条安全样本的图只支持所测分布；同时看违规率与误拒率，不能宣称少量数据保证全面安全。'),
(28,30,'图中梯度训练循环需正确设置labels和prompt/response mask；混入指令数据的midtraining有助减少分布突变，但不能保证完全无灾难遗忘。饼图是MiniCPM等特定混合方案，不作为所有公司已证实做法。'),
(31,44,'生成与验证能力不同，偏好数据要交代标注规则、采样模型和人群。标注分歧、风格偏见、劳动条件和专家成本影响数据；单个平台调查不能代表全球从业者，新闻工资数字不作当前行情。'),
(45,48,'模型评审在系统平均分上高相关，不等于每条回答判断准确或没有自我偏好。Constitutional AI先生成批评/修订，再收集偏好和训练；AI反馈仍依赖初始模型、原则和验证数据。长度增长可能是代理奖励偏差。'),
(49,54,'policy gradient需要对采样分布求导，奖励固定时可写score-function估计。PPO的clip是代理目标，不是把参数或概率一律截断，也不是严格KL约束保证。KL到固定参考模型与PPO到旧策略的ratio分开。'),
(55,60,'DPO从KL正则目标在非参数策略假设下得到pi*=pi_ref exp(r/beta)/Z，再用Bradley–Terry配对似然。Z(x)在同prompt比较中抵消；不是对好样本普通SFT、坏样本无限减概率。SimPO移除参考模型同时加长度归一和margin，不能说完全同一个目标。'),
(61,65,'PPO/DPO排序取决于数据、训练预算和调参。代理奖励持续升而人类偏好下降是过优化证据；不能把所有奖励源都画成同一退化曲线。p64“no longer probabilistic models”过强：仍输出概率分布，只是不自动校准，置信度/熵和多样性可能变化。')],
'primary_sources':[],
'limits':'All 65 pages/main figures read. Original reported evaluation results are not rerun; secondary news/workforce statistics are contextual examples, not verified current facts.'}
PDF_FULL_NOTES['2026-lecture-16.pdf']={'pages':61,'findings':[
(1,10,'可验证任务减少偏好奖励噪声，但测试覆盖不全仍可被钻空子。PPO图需区分固定参考、采样旧策略、当前策略和值模型；LM每token为动作与整回答bandit视角均可，不能混用时序奖励。'),
(11,16,'外循环采样、内循环多次更新期间旧logprob和优势保持固定。p14代码对logprob−ref_logprob取非负截断是该实现的奖励塑形，单样本log-ratio可负，不等于真实KL必为负或这样截断保证KL不发散。GAE含终止mask，gamma=lambda=1也依有效序列边界。'),
(17,24,'GRPO移除critic，以同prompt多次采样算组内均值与标准差。除std会改变问题权重，零方差组需epsilon或跳过；含自己的组均值也与动作有关，不能仅去std就无条件宣称对原目标完全无偏。长度归一使正/负优势的长短回答权重不同，Dr.GRPO改变这些项，非简单“修一个bug”。'),
(25,38,'R1-Zero长CoT和aha例子不能单独证明新能力涌现，底模也可有类似文字。R1加入冷启动SFT、语言一致性及后续通用训练；800k蒸馏是教师输出数据，非学生再次运行同规模RL。PRM/MCTS在该研究失败不等于所有任务永久无效。'),
(39,48,'Kimi K1.5基于参考答案的奖励、策略正则和长度奖励与GRPO不同。p41称只保留best-of-8全失败题过强，原文用估计pass rate排除太简单并作课程采样，不能据此断言全部题8次全错。p43错误答案的min(0,lambda)是长答案受罚、短答案不额外奖励，不是强制长度低于中点；相同长度时分母为零需实现保护。'),
(49,54,'Qwen3四阶段图含推理RL、thinking融合和general RL。3995指所述RL数据，不是全部训练只见3995条。thinking budget控制不等于最佳答案有正确性保证；stage4有部分数学指标下降，保留收益和代价。'),
(55,61,'Qwen3-Coder-Next把仓库数据、合成轨迹、专家蒸馏和自动构造环境串联。agent RL图里的git取回答案属于奖励投机，曲线跳升不能当能力提升。评测需环境隔离及防泄露；工具格式SFT与真实多步工具成功率是不同指标。')],
'primary_sources':[],
'limits':'All pages/main figures inspected. Dense code screenshots read for dataflow and objective; not a GPU reproduction or an independent audit of all benchmark entries.'}
PDF_FULL_NOTES['2025-lecture-03.pdf']={'pages':68,'findings':[
(1,19,'逐页图示与2026架构讲义对照：pre/post norm及残差路径一致；p14 RMSNorm同样漏1/d，需按均方根修正。p16统计归一化FLOPs很少但耗时25.5%说明搬运限制，非所有设备固定比例。'),
(20,35,'激活、GLU、串行/并行与RoPE图均保留。p28串行简式未显式保留attention残差，按两次残差完整式教学。RoPE只在相应Q/K坐标旋转，p35传value_states不是旋转V；15%并行加速非普遍保证。'),
(36,52,'FFN和宽深比曲线是经验宽平区。p43 PaLM head dim258及ratio1.48延续表格错误：主文48×256/18432=2/3。词表大小历史样本不当硬规则；单遍仍可记忆，weight decay优化作用不排除防过拟合。'),
(53,58,'z-loss最大化写减惩罚，最小化NLL写加惩罚；QK norm、输出z-loss与soft-cap不同。soft-cap给上下界而非仅上界；loss尖峰与下游质量分开。'),
(59,68,'prefill与decode成本分开，算术强度取整个括号倒数。MQA共享一个KV头非一维标量，GQA为折中。滑窗层扩大可达感受野不等价任意全注意力；闭源GPT4稀疏说法不作确证。')],
'primary_sources':['https://arxiv.org/html/1910.07467v1','https://arxiv.org/html/2204.02311v5'], 'limits':'All pages/main diagrams inspected; historical specification and benchmark screenshots are not current independently audited catalogues.'}
PDF_FULL_NOTES['2025-lecture-04.pdf']={'pages':47,'findings':[
(1,14,'MoE总参数、激活参数、wall clock分开，router和通信非零成本；7×与2×图各有自己的基线，不能混用。GPT4旁有问号，未知结构继续标未知。'),
(15,24,'token choice、expert choice与global assignment沿不同轴选择；shared expert收益两篇消融不一致。p19说V3 top-k后softmax与实际sigmoid后除和不符。p23正确列256 routed/8 active/1 shared，供后页纠错。'),
(25,31,'Gaussian加法噪声与uniform乘法jitter不同，top-k填−∞。辅助loss的f_i为离散统计，P_i可反传；V3路由偏置不进入输出权重，仍含sequence辅助项。负载图与loss一起看。'),
(32,40,'MegaBlocks图是块稀疏，不是2:4硬件稀疏。容量丢token造成batch依赖，非所有随机性都来自MoE。router FP32、z-loss、小集微调和upcycling均有条件，不自动保证稳定/专业分工。'),
(41,47,'p43标题V3却写V2(671B)和258 routed，应为V3/256；其公式给sigmoid后归一，文字Sigmoid+Softmax不准确。MLA还缓存RoPE key分量，MTP训练与推理speculative使用分开。')],
'primary_sources':['https://arxiv.org/html/2412.19437v1'], 'limits':'Every page/main figure inspected. No rerun of sparse GPU kernels or author benchmark tables.'}
PDF_FULL_NOTES['2025-lecture-05.pdf']={'pages':51,'findings':[
(1,18,'GPU/CPU、层级存储、warp/block和TPU图与2026基础一致。p9 full GA100128SM与p44 A100108SM是设计/产品区别，不是矛盾。峰值tensor FLOPs需匹配dtype；内存延迟数字属所引设备。'),
(19,25,'roofline同时约束计算与带宽；p24用bytes/FLOP命名intensity是常用AI的倒数，FP32/FP16例应为1/8、1/4 FLOP/byte。warp内分歧与统一分支不同，低精度累加未必也低精度。'),
(26,40,'融合减少HBM与launch，重算增加算术减少保存。8→5是玩具访问账本。coalescing按同warp同条指令地址观察，不是单线程沿行连续；tile受寄存器/共享内存/边界mask限制，padding不是唯一方案。'),
(41,45,'nanoGPT50257→50304由图确认。1792→1793 tile98→120与108SM说明wave边界，仅在给定驻留条件下解释性能。多加一个维度不能推成通用减速/加速。'),
(46,51,'核心attention是QKᵀ、P·V两次矩阵乘，p47所谓3次混入投影；QK图画三头非三次核心乘。online softmax更新m/l/o，最大值变时分母及分子均重缩放。FlashAttention精确但浮点顺序可有差异，图示速度不作本机复现。')],
'primary_sources':['https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/'], 'limits':'All pages/main teaching figures inspected, no GPU benchmark reproduction.'}
PDF_FULL_NOTES['2025-lecture-07.pdf']={'pages':60,'findings':[
(1,10,'collective输入输出图逐一检查：all-reduce可分reduce-scatter加all-gather；p8文字Reduce应理解all-reduce而非只有root保留结果的reduce。带宽最优不代表所有小消息/拓扑最快，GPU网络表为特定系统。'),
(11,25,'DDP完整复制静态状态，ZeRO1/2/3依次切optimizer/grad/param；16B和12B是各自精度假设，包含临时聚合、激活及工作区才可判断实际能否装下。2N/3N通信式省略(P−1)/P和字节宽度，“free”只指理想带宽量，不是零开销。'),
(26,35,'data parallel批量扩展有统计边界。流水图bubble/useful=(stages−1)/microbatches，非占总时间比例；zero-bubble权重梯度可延后但有依赖/内存/优化器边界，不能任意永远拖延。'),
(36,48,'tensor并行列切/行切与f/g前后向collective对应；序列并行切逐token算子，不把attention上下文断开。激活式34、10、24是所引架构/精度账本，GQA、Flash、SwiGLU会改常数。p47合并DDP/ZeRO1为no scaling忽略ZeRO1状态收益，应仅说完整参数仍复制。'),
(49,60,'TP8和3D配方受8GPU节点拓扑限制，非永久最优。p52图注64A100 GPUs而底部说64machines，按64GPU例理解。p58 Llama故障表原论文百分比自身不一致，不继续算成精确占比；Llama3配置各轴相乘成立，其他EP/CP组未必独立。')],
'primary_sources':['https://arxiv.org/html/2407.21783v1'], 'limits':'All 60 pages/main schedules and tables inspected; no multi-node replay, runtime/activation constants remain source-specific.'}
PDF_FULL_NOTES['2025-lecture-09.pdf']={'pages':53,'findings':[
(1,18,'有限类上界、历史曲线与均值MSE图分清理论上界与实际loss。幂律带熵底时log(L−L∞)才线性；不是所有经典模型任何误差都1/n。'),
(19,25,'p19二维网格可得1/sqrt(n)，直接泛化n^−1/d不普遍，须加入平滑性和bias/variance平衡。重复数据图的4epochs是所测配方，混合数据只改截距也是经验结论。'),
(26,39,'同参数/同FLOPs是不同对比，embedding不能与非embedding混计。临界batch同时约束steps和examples。muP初始化variance与STD不同，宽度迁移不保证下游指标相同排序。'),
(40,49,'联合拟合与IsoFLOP需C≈6ND且日程匹配，不能扫模型时固定D又说同预算。p48 least squares简化实际Chinchilla log-loss Huber拟合，不套OLS残差必均值0。p45区间来自原作者，不当精确自然常数。'),
(50,53,'服务成本最优可多训较小模型，overtrain是相对训练最优，非验证集过拟合。扩散和自回归曲线各按原预算与NLL口径，不能据一组图普遍断言哪类更好。')],
'primary_sources':['https://arxiv.org/html/2203.15556v1'], 'limits':'All pages/main figures read; no re-estimation of raw scaling experiments or confidence intervals.'}
PDF_FULL_NOTES['2025-lecture-11.pdf']={'pages':55,'findings':[
(1,9,'CerebrasGPT以40M代理调参、迁移至111M–2.7B，图支持其设置更平滑，不证明任意数据/深度同样稳定。embedding/output尺度、初始化及分层LR一起定义muP，表格是variance而非STD。'),
(10,25,'MiniCPM将宽度等配方联动，WSD共享稳定前缀再单独衰减。p17不能字面理解必须全部从头训练；p23 Ultratext95.60与p24平均192保留不同统计口径，不互相覆盖。重复衰减分支仍花计算。'),
(26,36,'DeepSeek直接拟合LR/batch，近最优0.25%条件与拟合不确定性均保留。p30文字两次10%衰减并非原图：80%tokens降至31.6%LR，90%降至10%。M为FLOPs/token非参数量，Hunyuan active ratio不与dense总参数混比。'),
(37,43,'深线性rank-1 SGD推导含最坏方向与不抵消假设，谱范数乘向量范数是上界并非普遍等式。Δloss的一阶近似需小步长，Frobenius与谱范数只在相应rank-1处一致，AdamLR与SGDLR不同。'),
(44,55,'固定depth24、headwidth128的宽度实验不等于深度迁移已证实。RMSNorm gain、Lion和weight decay的失败限于该配方；coarse LR网格与10B单次外推有实际范围，不说muP一劳永逸。')],
'primary_sources':['https://arxiv.org/html/2404.06395v1'], 'limits':'All pages/main figures inspected. No rerun of width sweeps, closed-model configurations or scaling fits.'}
PDF_FULL_NOTES['2025-lecture-15.pdf']={'pages':66,'findings':[
(1,14,'SFT数据图、指令例子与课堂标注演示保留。p14示例回复对CS336泛泛而谈甚至列数据库/图形学，正是事实性与风格评审分开的反例，不能作课程介绍抄入资料。p13表单与课堂回复表缺少可归档原始结果时标缺口。'),
(15,29,'长度/列表偏好和实际任务质量不同，错误引用、未知事实、误拒Python process的图均保留。500条安全样本收益受原测试集限制。midtraining混合可能缓和遗忘，非绝对不会遗忘。'),
(30,44,'经典RLHF数据流由示范到比较再优化；p32成本是AlpacaFarm示例预算，不作当前市场报价。InstructGPT筛选40标注者是历史实验，Bard指南标allegedly，群体偏好/劳动问题不因自动反馈而消失。p40第二标注表单结果未归档不编造全班分布。'),
(45,54,'AI评审系统级相关不代表每例可靠，Constitutional AI的批评修订与偏好训练阶段不同。PPO ratio相对旧策略，KL参考固定SFT，两种约束不能合并成同一项。'),
(55,62,'DPO由KL正则最优策略重参数化及配对似然导出，Z同prompt抵消；SimPO还改长度和margin。p59 DPO人类胜率空白，不能把其模拟46.8%当人类实测；p60历史榜单不是今天DPO统治所有开源模型的证据。'),
(63,66,'过优化图同时显示无噪声模型反馈没有相同下降，避免挑单图概括。p65“不再是概率模型”应改为仍输出概率但可能失去校准，mode collapse与校准是相关而不同的问题。')],
'primary_sources':[], 'limits':'All pages/main figures inspected; live class form responses are not reconstructed and dated author results are not local reproduction.'}
PDF_FULL_NOTES['2025-lecture-16.pdf']={'pages':67,'findings':[
(1,13,'续讲DPO：p7模拟PPO/DPO46.8%相同，但DPO人类胜率空白。p13概率校准退化不等于不再输出概率；SimPO与length-normalized DPO各自目标不同。'),
(14,28,'RLVR与PPO数据流按每token动作解释，固定参考模型和旧策略分开。p26非负截断sample logratio是AlpacaFarm实现特例，不等于KL估计必如此，也不保证防发散。GAE与terminal mask、padding/EOS边界都重要。'),
(29,36,'GRPO组内z-score移除critic，但std改权重，组均值包含自身也影响无偏性。Dr.GRPO去std和长度归一不应称任何设置都严格无偏。零方差组需数值保护；组内全对/全错无相对优势。'),
(37,50,'R1-Zero、R1、蒸馏区分：长回答/aha展示不是新推理能力充分证明；800k教师数据不是蒸馏学生同样跑RL。PRM/MCTS未在该研究成功，不等于概念永久无用。'),
(51,60,'Kimi参考奖励、课程采样、长度控制和RL系统图逐项对照。p53“只保留best-of-8全失败”超过原文能支持的范围；p55错误答案长者受罚短者不奖励，不是强制中点以下。环境转移/partial rollout需保证采样策略版本。'),
(61,67,'Qwen3强弱蒸馏、thinking融合及general RL各阶段效果不同，3995是特定RL数据规模。测试时更多token有经验收益但非单题正确性保证；部分数学指标在general RL后下降也保留。')],
'primary_sources':[], 'limits':'All pages/main diagrams read; toy CPU objective checks are separate from unavailable full model/GPU training reproduction.'}
PDF_FULL_NOTES['2024-lecture-03.pdf']={'pages':68,'findings':[
(1,18,'2024作业架构为绝对位置embedding/GELU，与后年RoPE/SwiGLU不同，保留版本区别。p13 RMSNorm漏均值因子1/d；pre-norm残差图及FLOPs/耗时占比不能误作同一指标。'),
(19,34,'激活、门控、串并行和RoPE图逐页对照。p27串行简式漏显式attention残差，同后年修正；旋转Q/K二维对子，value_states入参不代表旋转V。'),
(35,50,'FFN比例和宽深比为经验区域，p42 PaLM258/1.48按原论文256/2/3修正。单遍数据仍可记忆，p49“不是防过拟合”仅是所引优化观察，不能泛化。'),
(51,58,'BPE自底向上合并，unigram由大候选集拟合再剪枝；SentencePiece是库不是与BPE互斥算法。p51/58“都可逆”需限定预处理后的文本：NFKC、空白合并或UNK会丢原信息。p55的上标数字归一丢格式就是反例。空白/单数字切分影响代码与算术，不是越短token总越好。'),
(59,68,'KV缓存、MQA/GQA与稀疏mask图保留。p60算术强度的倒数要覆盖整个和；MQA共享KV头非标量。滑窗多层有间接长程路径，非一层全注意力等价。')],
'primary_sources':['https://arxiv.org/html/1910.07467v1','https://arxiv.org/html/2204.02311v5'], 'limits':'All pages/main diagrams read; no 2024 lecture video available to validate undocumented oral additions.'}
PDF_FULL_NOTES['2024-lecture-04.pdf']={'pages':49,'findings':[
(1,16,'tokenizer可逆性必须排除有损normalization/UNK；attention AI式取整体倒数。MQA单头仍有head_dim，图示GQA质量/延迟仅在原设置成立。'),
(17,27,'MoE在训练时就稀疏，增加总参数仍增加存储、router与通信成本。Mixtral/DBRX/Grok吞吐图受并发及硬件条件影响，GPT4问号仍标未知。'),
(28,39,'三路选择图按不同轴，top-k前/后归一不同。p29 DeepSeek7的混称不能跨版本沿用，原DeepSeekMoE区分routed与shared。离散索引不可导不等于gate没有梯度；Gaussian噪声和乘法jitter不同，辅助loss需把f_i作离散统计。'),
(40,49,'expert parallel通信与MegaBlocks块稀疏图不同于硬件结构稀疏。容量丢token会让请求受同batch影响；FP32 router与z-loss缓解数值问题非保证。p48权重相关仅是Mixtral upcycling猜测，课程自己说未知，不把猜测补成已证实流程。')],
'primary_sources':[], 'limits':'All pages/main figures read; no exact 2024 oral Q&A or GPU reproduction is claimed.'}
PDF_FULL_NOTES['2024-lecture-05.pdf']={'pages':50,'findings':[
(1,18,'CPU/GPU层级、warp/block、full GA100128SM图已核对，A100108SM是启用产品规格。存储延迟/价格倍数保留为历史例，不能跨设备当常数。'),
(19,31,'roofline用FLOP/byte，p23却标8/4bytes/FLOP应取倒数。混合精度不意味所有累加低精度；融合5算子与重算8→5是HBM账本，非通用速度承诺。'),
(32,44,'coalescing按同warp地址，tile需考虑mask/存储/occupancy。p43分母误写246，按题设256×128及98tiles应为256；1793变120块、108SM例有驻留条件。nanoGPT50257→50304确为64对齐。'),
(45,50,'p46“三矩阵乘”不精确，核心两次加softmax、投影另计。online softmax需更新最大值并缩放旧分母/输出分子；FlashAttention精确注意力而非稀疏近似，原图性能非本机复现。')],
'primary_sources':['https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/'], 'limits':'All pages/main figures read. Full GPU benchmarks/backward kernels not reproduced.'}
PDF_FULL_NOTES['2024-lecture-07.pdf']={'pages':54,'findings':[
(1,12,'网络图及collective结果核对，p8文字Reduce指all-reduce才与全员输出图一致；忽略启动延迟的带宽最优并非任意消息最佳。DP的SGD sum/mean与学习率约定需统一。'),
(13,24,'16B与12B参数状态账本不同，ZeRO分片仍需临时all-gather与激活。2N/3N是省略dtype与有限P系数的通信量，阶段1/2“免费”不等于无延迟/管理成本；stage3仅静态分片可线性缩，不保证实际峰值。'),
(25,36,'pipeline微批调度有bubble，(p−1)/m是idle/useful不是总时长占比。tensor列切/行切与前后向collective对应，适用低延迟网络；不能说节点外一律禁止TP。'),
(37,44,'激活峰值图不同于静态状态，34/24/10常数对应旧架构和存储假设；SP切逐token算子，attention仍需跨完整上下文。p43将DDP/ZeRO1都写无显存扩展过粗，ZeRO1优化器状态有收益。'),
(45,54,'GPT-NeoX引文TP2、PP4而页面概括TP4、PP2，虽乘积相同仍按引文区别，不能把交换轴当同配置。Falcon stage1标probably继续未知；p52图注64A100GPU，底部64machines是笔误。TP8最优来自特定集群，不泛化。')],
'primary_sources':[], 'limits':'All pages/main schedules and captions inspected; exact historical training recipes not independently reproduced.'}
PDF_FULL_NOTES['2024-lecture-09.pdf']={'pages':52,'findings':[
(1,19,'理论上界、实际经验曲线与熵底区分。均值估计MSE=σ²/n；p18非参数n^−1/d仅二维例吻合，不能任意d直接推广，应说明平滑性并平衡bias²/variance。intrinsic dimension仍为解释假说。'),
(20,30,'Ye数据混合图保留seen/unseen mixture及模型/steps外推三个轴；混合只改截距、4epoch近似新数据是所引实验，不保证任意分布。架构对比同参数与同计算各有含义。'),
(31,42,'critical batch看examples/steps共同预算，muP不只初始化，variance与SD分开。训练loss与下游分数不必相同排序；联合loss(N,D)拟合需熵底且C≈6ND。'),
(43,52,'三Chinchilla方法含日程选择和固定FLOPs扫模型，p47 least squares并非精确算法名，原方法3 logloss Huber拟合。所引复核质疑不等于所有原数据全错。服务总成本最优可偏小模型多训练，overtrain不必是过拟合。')],
'primary_sources':['https://arxiv.org/html/2203.15556v1'], 'limits':'All pages/main figures read; original fit uncertainties and experiment grids not rerun.'}
PDF_FULL_NOTES['2024-lecture-10.pdf']={'pages':47,'findings':[
(1,10,'封面CS324为沿用模板，不改成另一门课。API任务预算/重复查询规则是当年作业接口，未运行不可称现在可用。Cerebras40M代理的随机搜索与111M–2.7B迁移有范围，完整muP含分层LR/输出尺度。'),
(11,26,'MiniCPM的9M–499M sweep、2.4B外推和WSD分支图保留。p18所有配置必须从头训练太绝对；稳定前缀可复用但衰减尾不免费。Ultratext95.60面板与正文平均192需分范围，训练tokens/param不作通用常数。'),
(27,34,'DeepSeek近最优0.25%筛选后拟合，LR噪声与平谷底不能给精确最优。p31两次10%是tokens阶段长度，不是每次LR乘0.1：80%tokens至31.6%，90%至10%。M=FLOPs/token与N参数量不同。'),
(35,47,'宽度实验固定depth24/headwidth128，非完整depth/width联合证明。表格初始化variance与STD不同，RMSNorm gain/Lion/weight decay失败限于所测配方，10B一次结果不保证所有设置迁移。')],
'primary_sources':['https://arxiv.org/html/2404.06395v1'], 'limits':'All pages/main figures read; unavailable historical API/GPU experiment execution explicitly excluded.'}
PDF_FULL_NOTES['2024-lecture-15.pdf']={'pages':53,'findings':[
(1,15,'RLHF三阶段中此讲聚焦SFT；FLAN/Alpaca/OASST格式不同。p15关于CS336的泛泛回答是标注质量反例，不能当课程事实。表单现场回答未公开。'),
(16,28,'偏好分数受长度/风格影响。已背过的知识在SFT后仍可能提取变差；Gekhman/Ghosal图不能简化成SFT绝对学不到知识。安全数据500/2000例呈现有用性取舍，不能承诺消除越狱。'),
(29,38,'LIMA的1000序列约750k tokens，不能证明任何任务千例足够。URIAL为3–8个上下文示例，无参数更新，不能叫few-shot SFT。ChatGPT评分/长度偏差保留；32k数据平台不证明所有SFT数据扩展无效。'),
(39,45,'LoRA冻结基座并训练低秩增量，增量rank≤r不表示基座低秩。A随机/B零初始化时首步A梯度零而B可非零；不能两者皆零。仍需经冻结层反传输入梯度。p44的127公式是Int8例，不能说是NF4公式。'),
(46,49,'NF4用16个非均匀码点，存储4bit而计算可用BF16；非IEEE符号/指数/尾数切分。absmax分块缓解离群影响，p49所谓non-contiguous不是要求，原论文采用展平后的连续块。反量化是有损近似。'),
(50,53,'p50把量化常数量化到4bit的文字需纠正：原QLoRA第二层为8bit浮点、再存FP32尺度。原论文64/256分块给常数开销0.127 bits/weight。forward反量化加LoRA，base不更新。Guanaco的21/41GB和评分限历史配方/评测，不能代表全面等同ChatGPT。')],
'primary_sources':['https://arxiv.org/html/2305.14314'], 'limits':'All 53 pages/main figures read. Original 2024 lecture audio unavailable; LoRA/QLoRA training not rerun.'}
PDF_FULL_NOTES['2024-lecture-16.pdf']={'pages':50,'findings':[
(1,12,'生成与评价能力有差距；成本表是当年7B例，不能当现价。偏好标注须看任务准则、招募筛选与人口分布，p10明确Allegedly来源不可改成官方确认。'),
(13,24,'课堂标注5题表单结果缺失。众包质量/劳动条件、标注者偏差和GPT judge的system-level相关不等于每例可靠。CAI有critique/revision与偏好模型两阶段；自举使用已有结构/选择信息，不是无中生有。长度与偏好混杂需控制。'),
(25,32,'policy gradient/TRPO/PPO的目标、KL参考策略和old rollout策略区分。clip不是保证每步KL受限。p32 DPO human win-rate为横杠，sim46.8不能冒充human。'),
(33,38,'AlpacaFarm代码图分rollout、固定old logp的多轮minibatch、value loss/GAE。p36 max(sample log-ratio,0)为特定实现，不等于精确KL，更不保证不发散；pad==eos末位偏移代码自带警告。γ=λ=1配方非所有RL任务默认。'),
(39,46,'best-of-n、rejection SFT、expert iteration与DPO有训练/推理成本差别。DPO闭式最优需非参数策略/参考支持等条件，Z(x)在偏好差中消掉，不应说任意有限网络自动严格最优。p45 human DPO仍缺值，历史榜单非当前排名。'),
(47,50,'proxy reward上升而gold下降是过优化例，noiseless simulated preference面板例外要保留。p49仍是概率模型，问题是校准可能变差、熵降低，不是失去概率分布定义。')],
'primary_sources':[], 'limits':'All pages/main figures/code screenshots read; original 2024 live execution/audio not available; GPU training not rerun.'}
PDF_FULL_NOTES['2024-lecture-17.pdf']={'pages':53,'findings':[
(1,6,'p2明确“No recordings for guest lectures – next mon/wed”，为2024末两嘉宾课未录制的官方课内证据，须提升缺口分类。p3–4作业曲线1e19 FLOPs/399.51M loss3.04属当年结果，稳定性与超参影响不能全归参数量。'),
(7,18,'compression/multitask/chatbot三种目标决定评测。p9“LM标准评测已死”为课堂修辞，held-out loss仍可用于训练检查，不能删除。GLUE微调与few-shot评测不同，HELM-lite/旧榜单是历史组合。'),
(19,25,'逐例辨ARC/HellaSwag/TruthfulQA/MMLU/WinoGrande/GSM8K的知识、拒绝、常识和数学目标。p20TruthfulQA不等于全为拒答题；p25约8500含train/test，test1319通常约1k，不能将8500当test。'),
(26,33,'饱和、题目错误、prompt/选项顺序/logp-vs-generation都影响排名。PC1高解释率是该模型样本的相关，不证明单一因果能力。inverse scaling的prior/priors例需保留原拼写。'),
(34,45,'真实用户偏好与研究能力评测目标不同。Arena使用分布未按用户去重，不能称全体人群代表样本。AlpacaEval需顺序随机化、长度控制、自偏好检验；LC回归缓解混杂非消除所有偏差。'),
(46,53,'安全应同时看违规拒绝及良性过拒，bias/fairness任务覆盖不同群体。直接泄漏test与相近任务分布区分；社交媒体质疑非确定污染证据。Min-k-prob为启发式membership证据，exchangeability需假设，不保证检出全部污染。')],
'primary_sources':[], 'limits':'All pages/main figures read. p2 is direct evidence of no 2024 guest recordings; no transcript inferred from related talks.'}
