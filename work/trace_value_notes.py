TRACE_VALUE_NOTES=[
('2025-lecture_01.json','tokenizer/BPE','字符与字节表示逐例核对。45个counts值是循环的中间字典快照，不能把每一步的1当最终频次；最终merge为th/the/the空格，推理按rank重放。词表前256为单字节，新增256–258拼接可逆。'),
('2025-lecture_02.json','tensor/optimizer','切片别名使x首元素变100；转置后reshape顺序与原flatten不同。einsum缩并4维得全4，reduce mean全1/sum全4。随机初始化输出只作数值规模示例；两optimizer state快照约±.01更新，不证明所有后续步同幅。'),
('2025-lecture_06.json','benchmark/GELU/softmax','124个times列表是多次benchmark逐步累积，非124独立完整实验。matmul1024–16384耗时非单调线性；batch/dim小shape约6ms平坦受launch/框架影响。GELU手写与融合小例同为.841192不保证所有输入逐位相同。softmax全100第一行1/3，第二行极小非零与1，不能把显示1理解为无穷精度。'),
('2025-lecture_10.json','symbolic stats','配置K40→8改变KV头与参数，固定其他shape比较；B256模型内存超过80GB只是假设统计，非单卡成功运行。'),
('2025-lecture_14.json','data filtering/dedup/LSH','importance权重归一化后和为1。DSIR示例probs=[0,.4,0,.6]含未平滑零桶，因此两候选权重零；不全归因underflow。set去重不保顺序且Hello!/hello保留，和A4整行重复全删的任务不同。Bloom有false positive而无本例false negative。LSH表按各(b,r)重算。'),
('2025-lecture_17.json','RL arrays','reward[4,4,3,4,4]均值3.8；std采用样本修正sqrt(.8/4)=.4472136，normalization含1e-5，故约.4472037非无epsilon值。第三个prompt组全reward3→中心化全零。独立位置log_probs不能称完整自回归排序策略。'),
('2026-lecture_01.json','tokenizer/BPE','2026 tokenizer不同使首例token IDs变化；UTF8与码点列表仍一致。三次merge按pair计数重算并可逆解码。不把merge函数入口空new_indices当最终空输出。'),
('2026-lecture_02.json','tensor/current code','typed contents内shape已与值递归比较；einsum2D/3D和rearrange全1/全4符合缩并维。深网络y有ReLU全非负；w.grad=[1,2,3]。Adagrad g2非负并含零，不能叫Adam一二阶矩。'),
('2026-lecture_02_recording.json','recording code','录制版本输入/权重随机样本与当前不同，不能逐数强行相等。ReLU输出非负、梯度[1,2,3]、einsum缩并/reshape关系一致；分支代码差异另在逐讲说明。'),
('2026-lecture_06.json','Triton arrays','全1矩阵/小shape计时仍是原作者缓存。三GELU结果小例一致；两softmax数组一致且每行和约1；row sum[1..4]/[5..8]=[10,26]，不能把这一正确性检查当GPU速度重跑。'),
('2026-lecture_10.json','symbolic memory/latency','每个统计条目latency=memory/3.35e12，throughput=B/latency，依完美带宽和忽略开销假设。B64/K40内存79.7GB几乎满，未加临时buffer；K8/B25665.6GB也不保证真实程序必能放入。'),
('2026-lecture_14.json','dedup/LSH/mixing','typed set去重同旧版，3组LSH概率与1-(1-s^r)^b一致。混合示例high仅1e8 unique与low1e11不同，固定total若提高high比例须记录重复epochs，不能自动看作增加unique高质数据。')]
