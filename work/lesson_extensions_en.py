"""English derivations and chapter headings, aligned to the Chinese edition."""
EXTENSIONS={
'00.07':'Chain rule step by step: u=2w, L=(u−3)². Then dL/dw=(dL/du)(du/dw)=2(u−3)×2. At w=1, u=2 and the gradient is −4. Evaluate intermediate values first, then multiply local sensitivities. With multiple paths, add their contributions.',
'04.04':'AdamW: m←β₁m+(1−β₁)g, v←β₂v+(1−β₂)g². At step t, m̂=m/(1−β₁ᵗ), v̂=v/(1−β₂ᵗ) correct zero-initialization bias. A simplified update is w←(1−ηλ)w−ηm̂/(√v̂+ε). λ controls decay and ε stabilizes the denominator. Parameter-group and schedule conventions depend on implementation. v is a second moment, not a Hessian.',
'04.06':'An illustrative cosine decay over progress s∈[0,1] is η(s)=ηmin+(ηmax−ηmin)(1+cos(πs))/2. It starts at ηmax and ends at ηmin. Warmup and WSD tails may use other functions; WSD does not prescribe one universal cosine formula.',
'05.02':'Single-head shapes: X:T×D, WQ/WK:D×dk, WV:D×dv. Thus Q/K:T×dk and V:T×dv. P=softmax(QKᵀ/√dk+mask) has shape T×T; PV has shape T×dv. Softmax normalizes over key positions. Batch and multiple heads add axes around this chain.',
'06.04':'ReLU(x)=max(0,x); GELU(x)=xΦ(x), where Φ is the standard normal CDF and implementations may approximate it; SiLU(x)=x·sigmoid(x). One SwiGLU form is [SiLU(XWg)⊙(XWu)]Wd, with elementwise ⊙. Biases and intermediate width depend on the model. The gate need not lie between 0 and 1.',
'07.08':'Use S:dv×dk, column vectors k/q:dk and v:dv. One gated-delta form is Snew=αSold+β(v−αSold k)kᵀ, read as Snew q. α forgets old state; β controls an error-directed write. Check the paper for key normalization, gates, and parallel algorithms. An arbitrary I−βkkᵀ is not automatically an orthogonal projection.',
'11.10':'For upstream aᵢ=∂L/∂yᵢ and r=√(Σxᵢ²/D+ε), ∂L/∂xⱼ=aⱼgⱼ/r−xⱼΣ(aᵢgᵢxᵢ)/(D r³), and ∂L/∂gⱼ=aⱼxⱼ/r. Shared gain gradients also sum across rows. The second input term follows from r changing with x and cannot be dropped. D is the row feature count.',
'12.05':'Merge old statistics (m,l,u) with block scores sⱼ and values vⱼ: mnew=max(m,maxⱼsⱼ), c=exp(m−mnew), lnew=c·l+Σexp(sⱼ−mnew), unew=c·u+Σexp(sⱼ−mnew)vⱼ. Final output o=u/l. Initialize from a valid first block or safely handle m=−∞. Fully masked rows need an explicit convention rather than silent 0/0.',
'14.06':'For an ideal balanced GPipe-style fill/drain schedule with p stages and m microbatches, ignoring communication and using equal forward/backward stage times, bubble fraction is approximately (p−1)/(m+p−1). For p=4,m=1 it is 3/4; with m=16 it is 3/19. This is not a universal measurement formula for 1F1B or heterogeneous stages; draw their actual schedules.',
'17.02':'Two conditions: A1 keeps each activation coordinate Θ(1); n such coordinates typically have norm Θ(√n), not Θ(1). A2 requires a nondegenerate one-step feature change. For a linear layer and single-sample SGD, ΔW=−ηg hᵀ, so ΔW h=−ηg∥h∥². If coordinates of h and g have constant scale, ∥h∥²≈n and this local update needs η of order 1/n. This illustrates why correlated outer-product updates differ from independent initialization sums; it is not a complete Transformer μP implementation table. Input, hidden, readout, and optimizer rules must still be coordinated; 2026 L11 and the original paper give broader conditions.',
'24.06':'GAE uses δt=rt+γV(st+1)−V(st) and Ât=Σl≥0(γλ)^l δt+l, with terminal bootstrapping set by the task. Small λ relies more on value estimates; larger λ incorporates longer residual chains. PPO uses ρt=πθ(at given st)/πold(at given st). Keep the min/clipping direction for A<0 too; clipping the surrogate does not directly bound token probabilities to [1−ε,1+ε].',
'24.07':'In DPO, y+ is preferred and y− is the other response to the same x. πref is frozen; β is the KL scale. Response log probabilities sum conditional token log probabilities. The closed-form optimum assumes reference support and objective conditions; finite preference data and a parameterized model still incur estimation and representation error. β is neither a learning rate nor a measure of label reliability.',
'25.09':'Under the 2026 assignment: Dr. GRPO uses R−μ and constant total normalizer Z. Its MaxRL variant uses (R−μ)/μ with constant Z; the original paper uses a batch-token denominator. GSPO sets s=exp[mean_t(logπθ−logπold)] and optimizes min(sA,clip(s)A) at sequence level. Geometric averaging and clipping trade bias for stability. Zero-mean groups need explicitly documented implementation handling; no universal rule is invented here.'
}
CHAPTERS=[
('Foundations as needed','Read the mathematical and programming notation needed for the course.'),
('What language models do','Distinguish prediction, generation, and training.'),
('From text to tokens','Understand tokenization and its tradeoffs.'),
('Tensors and a minimal training program','Follow shapes through a model.'),
('Learning from prediction errors','Walk through a parameter update.'),
('Understanding the Transformer','Follow one token through the model.'),
('Modern architectures and hyperparameters','Understand the problem each design addresses.'),
('Attention costs and alternatives','Understand long-context tradeoffs.'),
('Mixture of experts','Separate total from active parameters.'),
('Compute, memory, and performance','Identify where resources are spent.'),
('GPUs, TPUs, and hardware execution','Build performance intuition.'),
('Profiling, compilation, and GPU kernels','Locate and improve slow operations.'),
('FlashAttention and recomputation','Understand memory-efficient exact attention.'),
('Communication across devices','Understand collective communication first.'),
('Distributed training','Identify what each strategy partitions.'),
('Inference and serving','Understand generation-time costs.'),
('Scaling-law foundations','Read curves and respect extrapolation limits.'),
('Scaling small experiments to large models','Understand transfer and experimental design.'),
('Evaluating models','Define the capability before choosing a metric.'),
('Sources of training data','Understand provenance and limitations.'),
('Data conversion and filtering','Give operational meaning to data quality.'),
('Data deduplication','Separate exact and approximate duplicates.'),
('Data mixtures and training stages','Relate data choices to objectives.'),
('Supervised and parameter-efficient fine-tuning','Understand instruction learning.'),
('Human preferences and RLHF','Understand how reward changes outputs.'),
('Reasoning with verifiable rewards','Understand RLVR and test-time computation.'),
('A small reinforcement-learning walkthrough','Map an abstract update to a toy task.'),
('Multimodal models','Extend language-model concepts to images and video.'),
('Guest lectures and later evidence','Preserve public sources and explicit gaps.')]
PROTOCOL='''Read the learner’s ignored local progress first. Confirm the current point, pending question, and preferred teaching language. English is the public default; a learner’s saved choice overrides it.
Teach only one point at a time: goal, explanation, a worked example, and one understanding question. Supply prerequisite concepts as needed; do not require videos or large assignments.
Stop after asking. Do not expose tutor criteria before the learner answers. Then acknowledge what is correct and address one misunderstanding.
If wrong or unclear, use the repair guidance and ask a smaller question about the same example. If a choice is correct without reasoning, ask why. Advance only when the learner can explain the reason and relevant conditions, or explicitly requests a skip; record skips as unconfirmed.
Derivations can span several turns. A lesson is not a one-turn quota. Neither reading it nor opening its answer marks mastery.
At session end, update ignored local/learning-progress.md with position, actual answer evidence, uncertainties, pending question, and next step. Never commit or automatically push personal progress to the curriculum repository. If file tools are unavailable, return the full updated text without claiming it was saved. Cross-device transfer is the learner's explicit private choice.
Distinguish course evidence, original teaching examples, executed experiments, and unverified claims. Preserve missing-source and ASR limitations. Check primary sources at teaching time for current prices, law, rankings, or software support.
'''
