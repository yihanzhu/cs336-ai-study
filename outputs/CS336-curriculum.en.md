# CS336 curriculum and sources

Version 3.1 · 299 teaching points · shared IDs across English and Chinese.

Source links below are chapter-level navigation. File identities and hashes are retained in CS336-来源索引.json. Teaching order is an original reorganization, not official lecture numbering.

## 00 · Foundations as needed

Read the mathematical and programming notation needed for the course.

- 00.01 Numbers and variables
- 00.02 Functions and parameters
- 00.03 Vectors and matrices
- 00.04 Matrix multiplication
- 00.05 Probability distributions and conditioning
- 00.06 Logarithms and exponentials
- 00.07 Derivatives, gradients, and the chain rule
- 00.08 Training, validation, and test sets

Sources: [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

## 01 · What language models do

Distinguish prediction, generation, and training.

- 01.01 Next-token prediction
- 01.02 Conditional probability
- 01.03 Sequence probability
- 01.04 Training and inference
- 01.05 Autoregressive generation
- 01.06 Pretraining and post-training
- 01.07 Model history and openness
- 01.08 Efficiency under fixed data and compute
- 01.09 Limits of small-model experience

Sources: [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

## 02 · From text to tokens

Understand tokenization and its tradeoffs.

- 02.01 Unicode and UTF-8
- 02.02 Characters, bytes, words, and subwords
- 02.03 Token IDs and vocabulary
- 02.04 BPE training, encoding, and decoding
- 02.05 Special tokens and pretokenization
- 02.06 Unigram and SentencePiece
- 02.07 Normalization, whitespace, numbers, and languages
- 02.08 Compression and vocabulary size
- 02.09 Byte-level and tokenizer-free approaches

Sources: [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

## 03 · Tensors and a minimal training program

Follow shapes through a model.

- 03.01 Shape, dimensions, and batches
- 03.02 Dtypes and devices
- 03.03 Storage, strides, views, and slicing
- 03.04 Broadcasting, elementwise operations, and matrix products
- 03.05 Einsum, einops, and shape labels
- 03.06 Parameters, embeddings, and modules
- 03.07 Automatic differentiation
- 03.08 Loading data, seeds, and reproducibility
- 03.09 Saving and restoring checkpoints

Sources: [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

## 04 · Learning from prediction errors

Walk through a parameter update.

- 04.01 Logits and softmax
- 04.02 Cross-entropy and numerical stability
- 04.03 Gradient descent and backpropagation
- 04.04 SGD, momentum, and AdamW
- 04.05 Initialization
- 04.06 Warmup, cosine decay, and WSD
- 04.07 Gradient clipping
- 04.08 Batches and gradient accumulation
- 04.09 The training loop
- 04.10 Validation curves, overfitting, and tuning
- 04.11 Ablation studies

Sources: [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

## 05 · Understanding the Transformer

Follow one token through the model.

- 05.01 Embeddings and linear layers
- 05.02 Queries, keys, and values
- 05.03 Dot products, scaling, and softmax
- 05.04 Causal masking
- 05.05 Multi-head self-attention
- 05.06 Position representations
- 05.07 MLPs and feed-forward networks
- 05.08 Residual connections
- 05.09 Normalization
- 05.10 Stacking Transformer blocks
- 05.11 Output projection and probabilities
- 05.12 Parallel training and stepwise generation

Sources: [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

## 06 · Modern architectures and hyperparameters

Understand the problem each design addresses.

- 06.01 Pre-norm and post-norm
- 06.02 LayerNorm and RMSNorm
- 06.03 Double norm and non-residual post-norm
- 06.04 ReLU, GELU, GLU, and SwiGLU
- 06.05 Removing biases
- 06.06 Serial and parallel blocks
- 06.07 Rotary position embeddings
- 06.08 Depth, width, heads, and FFN ratios
- 06.09 Vocabulary size
- 06.10 Dropout and weight decay
- 06.11 Z-loss, QK normalization, and logit soft-capping

Sources: [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

## 07 · Attention costs and alternatives

Understand long-context tradeoffs.

- 07.01 Full-attention compute and cache costs
- 07.02 Multi-query and grouped-query attention
- 07.03 MLA and cross-layer cache sharing
- 07.04 Sliding windows and sparse attention
- 07.05 Alternating global and local layers
- 07.06 Linear attention and recurrence
- 07.07 Mamba and state-space models
- 07.08 Gated DeltaNet
- 07.09 Hybrid architectures
- 07.10 Indexers and dynamic sparse attention
- 07.11 Quality, memory, and speed tradeoffs

Sources: [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

## 08 · Mixture of experts

Separate total from active parameters.

- 08.01 Dense and sparse computation
- 08.02 Routers, top-k, and expert selection
- 08.03 Token-choice and expert-choice
- 08.04 Fine-grained and shared experts
- 08.05 Routing noise and differentiable training
- 08.06 Auxiliary load-balancing loss
- 08.07 Bias adjustment without an auxiliary loss
- 08.08 Capacity, communication, and expert parallelism
- 08.09 Routing stability
- 08.10 Upcycling
- 08.11 DeepSeek cases and limits of evidence

Sources: [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

## 09 · Compute, memory, and performance

Identify where resources are spent.

- 09.01 Parameters and FLOPs
- 09.02 Forward and backward costs
- 09.03 Estimating training compute
- 09.04 Parameters, gradients, optimizer state, and activations
- 09.05 FP32, FP16, BF16, FP8, and FP4
- 09.06 Mixed precision
- 09.07 Arithmetic intensity
- 09.08 Bandwidth and compute bottlenecks
- 09.09 The roofline model
- 09.10 Model FLOPs utilization
- 09.11 Recomputation and accumulation costs

Sources: [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

## 10 · GPUs, TPUs, and hardware execution

Build performance intuition.

- 10.01 CPUs and GPUs
- 10.02 SMs, warps, threads, and blocks
- 10.03 SIMT and branch divergence
- 10.04 Registers, shared memory, caches, and HBM
- 10.05 Tensor Cores
- 10.06 TPUs and matrix computation
- 10.07 Low-precision hardware
- 10.08 Unequal compute and bandwidth growth
- 10.09 Coalesced memory access
- 10.10 Tiling, alignment, and wave quantization

Sources: [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

## 11 · Profiling, compilation, and GPU kernels

Locate and improve slow operations.

- 11.01 Reliable timing, warmup, and synchronization
- 11.02 End-to-end benchmarks and profiling
- 11.03 Nsight, flame graphs, and memory profiling
- 11.04 Operator fusion
- 11.05 CUDA, PTX, and Triton
- 11.06 torch.compile
- 11.07 GELU, softmax, row sums, and matrix products
- 11.08 Tiles and masks
- 11.09 Custom autograd
- 11.10 RMSNorm forward and backward
- 11.11 Correctness and speed as separate checks

Sources: [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

## 12 · FlashAttention and recomputation

Understand memory-efficient exact attention.

- 12.01 Attention's intermediate matrices
- 12.02 HBM and on-chip memory
- 12.03 Tiling attention
- 12.04 Online softmax
- 12.05 Stable incremental statistics
- 12.06 Fused forward computation
- 12.07 Saving necessary statistics
- 12.08 Backward recomputation
- 12.09 FlashAttention-2 work partitioning
- 12.10 Exact versus approximate attention

Sources: [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

## 13 · Communication across devices

Understand collective communication first.

- 13.01 Ranks, world size, and process groups
- 13.02 Broadcast, reduce, and all-reduce
- 13.03 All-gather, reduce-scatter, and all-to-all
- 13.04 Ring communication
- 13.05 Latency and bandwidth
- 13.06 NCCL and torch.distributed
- 13.07 Synchronous and asynchronous operations
- 13.08 Overlapping communication and computation
- 13.09 Intra-node and inter-node links
- 13.10 Topology and TPU/GPU networks

Sources: [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

## 14 · Distributed training

Identify what each strategy partitions.

- 14.01 Distributed data parallelism
- 14.02 Gradient averaging and buckets
- 14.03 ZeRO stages and FSDP
- 14.04 Sharding optimizer state, gradients, and parameters
- 14.05 Row and column tensor parallelism
- 14.06 Pipelines, microbatches, and bubbles
- 14.07 1F1B and zero-bubble scheduling
- 14.08 Sequence and context parallelism
- 14.09 Expert parallelism
- 14.10 Activation checkpointing
- 14.11 Combining parallelism strategies
- 14.12 Estimating communication and memory

Sources: [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

## 15 · Inference and serving

Understand generation-time costs.

- 15.01 Autoregressive sampling
- 15.02 Temperature, top-k, and top-p
- 15.03 Prefill and decode
- 15.04 KV caching
- 15.05 Latency and throughput
- 15.06 Batch size and memory bandwidth
- 15.07 Quantization and AWQ
- 15.08 Pruning
- 15.09 Cache compression and structural changes
- 15.10 Speculative sampling and verification
- 15.11 Continuous batching
- 15.12 PagedAttention
- 15.13 Dynamic request management
- 15.14 vLLM's role

Sources: [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

## 16 · Scaling-law foundations

Read curves and respect extrapolation limits.

- 16.01 Power laws and logarithmic axes
- 16.02 Data, parameters, and compute
- 16.03 Sample complexity and mean estimation
- 16.04 Intrinsic-dimension explanations
- 16.05 Distribution shifts
- 16.06 Repeated and finite data
- 16.07 Architecture and optimizer scaling
- 16.08 Critical batch size
- 16.09 Kaplan and Chinchilla
- 16.10 Envelopes, IsoFLOPs, and joint fits
- 16.11 Fitting error
- 16.12 Training-optimal and deployment-optimal models

Sources: [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

## 17 · Scaling small experiments to large models

Understand transfer and experimental design.

- 17.01 Initialization, learning rate, and batch together
- 17.02 μP conditions and basic derivation
- 17.03 Width transfer
- 17.04 Standard-parameterization controls
- 17.05 RMSNorm, optimizer, and decay boundaries
- 17.06 Public scaling recipes
- 17.07 WSD and reusable training
- 17.08 Muon's scale dependence
- 17.09 Multi-scale experiments and budget allocation
- 17.10 Extrapolation error and risk

Sources: [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

## 18 · Evaluating models

Define the capability before choosing a metric.

- 18.01 NLL, perplexity, and tokenization
- 18.02 Multiple choice, knowledge, math, and code tests
- 18.03 Few-shot and prompt sensitivity
- 18.04 Instruction following, preferences, and Arena
- 18.05 LLM-as-judge
- 18.06 Length, position, and style biases
- 18.07 Reasoning and agent evaluation
- 18.08 Safety, bias, and fairness
- 18.09 Contamination and detection limits
- 18.10 Saturation, validity, real tasks, and uncertainty

Sources: [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

## 19 · Sources of training data

Understand provenance and limitations.

- 19.01 Web pages, Common Crawl, and WARC
- 19.02 Encyclopedias, code, papers, books, and Q&A
- 19.03 Named training-corpus projects
- 19.04 Source, language, domain, and time
- 19.05 Open weights versus open data
- 19.06 Copyright, licenses, privacy, and traceability

Sources: [2024 L11](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_11.py) · [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L13](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_13.py)

## 20 · Data conversion and filtering

Give operational meaning to data quality.

- 20.01 HTML-to-text and body extraction
- 20.02 Language identification
- 20.03 Rule-based filtering
- 20.04 Quality and harmful-content classifiers
- 20.05 Personal information handling
- 20.06 KenLM, fastText, and DSIR
- 20.07 Importance sampling
- 20.08 Model-based scoring
- 20.09 False removals and missed removals
- 20.10 Effects on domains and groups
- 20.11 Validate filtering through training

Sources: [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

## 21 · Data deduplication

Separate exact and approximate duplicates.

- 21.01 Hashes and collisions
- 21.02 Exact deduplication by document, paragraph, and line
- 21.03 Bloom filters and false positives
- 21.04 Suffix arrays and repeated substrings
- 21.05 Sets, n-grams, and Jaccard similarity
- 21.06 MinHash
- 21.07 LSH bands and rows
- 21.08 Scaling deduplication
- 21.09 Deduplication and evaluation contamination

Sources: [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

## 22 · Data mixtures and training stages

Relate data choices to objectives.

- 22.01 Data mixtures and ablations
- 22.02 Domain resampling
- 22.03 Data selection and scaling
- 22.04 Estimating data mixtures
- 22.05 Repetition and epochs
- 22.06 Long-context data
- 22.07 Code infilling
- 22.08 Domain and reasoning data
- 22.09 Synthetic data and self-distillation
- 22.10 Midtraining
- 22.11 Post-training data verification

Sources: [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

## 23 · Supervised and parameter-efficient fine-tuning

Understand instruction learning.

- 23.01 Pretraining and instruction objectives
- 23.02 FLAN, Alpaca, and OpenAssistant
- 23.03 Conversation formats and loss masks
- 23.04 Small high-quality datasets
- 23.05 Style, facts, and knowledge elicitation
- 23.06 Hallucinations and safety fine-tuning
- 23.07 SFT settings and evaluation
- 23.08 Parameter-efficient fine-tuning
- 23.09 Low-rank adaptation
- 23.10 QLoRA, NF4, and double quantization
- 23.11 Frozen versus full fine-tuning

Sources: [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

## 24 · Human preferences and RLHF

Understand how reward changes outputs.

- 24.01 Pairwise preferences and reward models
- 24.02 Annotators, crowdsourcing ethics, and demographics
- 24.03 AI feedback and Constitutional AI
- 24.04 The gap between generation and evaluation
- 24.05 Policies, returns, and KL constraints
- 24.06 PPO, values, advantages, and GAE
- 24.07 DPO objective and derivation
- 24.08 Preference-method variants
- 24.09 Reward overoptimization and calibration
- 24.10 Safety red teaming

Sources: [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

## 25 · Reasoning with verifiable rewards

Understand RLVR and test-time computation.

- 25.01 Verifiable rewards and scorers
- 25.02 Language models as policies
- 25.03 Trajectories, returns, and policy gradients
- 25.04 Baselines and variance
- 25.05 GRPO and group normalization
- 25.06 Length and difficulty bias
- 25.07 On-policy and off-policy learning
- 25.08 Importance weighting and clipping
- 25.09 Dr. GRPO, GSPO, and MaxRL
- 25.10 Rejection sampling and expert iteration
- 25.11 R1, R1-Zero, Kimi, and Qwen cases
- 25.12 Distillation
- 25.13 Test-time scaling
- 25.14 Agent environments, trajectories, and RL infrastructure

Sources: [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

## 26 · A small reinforcement-learning walkthrough

Map an abstract update to a toy task.

- 26.01 The official 2025 sorting toy task
- 26.02 Generating responses
- 26.03 Computing token log probabilities
- 26.04 Designing a sorting reward
- 26.05 Policy-gradient loss
- 26.06 KL penalties
- 26.07 Freezing parameters
- 26.08 Before-and-after probabilities
- 26.09 Sampling and update loops
- 26.10 Comparing toy experiments

Sources: [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

## 27 · Multimodal models

Extend language-model concepts to images and video.

- 27.01 Image encoders
- 27.02 Image-text contrastive learning
- 27.03 CLIP and SigLIP
- 27.04 Visual tokens and connectors
- 27.05 LLaVA and visual instruction tuning
- 27.06 OneVision and variable resolution
- 27.07 Qwen-VL versions
- 27.08 Video and temporal position
- 27.09 Native multimodality and Chameleon
- 27.10 Understanding versus generation objectives

Sources: [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

## 28 · Guest lectures and later evidence

Preserve public sources and explicit gaps.

- 28.01 Registering six guest lectures
- 28.02 Teach only retrieved guest material
- 28.03 Place updates by topic
- 28.04 Course-time statements and later evidence
- 28.05 Do not invent unread lecture details
- 28.06 Leaderboards and durable knowledge

Sources: [2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s)
