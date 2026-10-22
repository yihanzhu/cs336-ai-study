# CS336: AI tutoring lesson plans

Version 3.1 · 2026-10-22 · 29 chapters, 299 points in each language. English is the public default; saved learner preference controls tutoring language.

These are original teaching plans with examples, questions, answer criteria, and repair branches. They do not recover missing guest classes. Original-material evidence retains the 1.2 audit boundaries. Historical Chinese preparation notes remain available as supplementary audit context; the full 299-point teaching layer is bilingual.

## Tutor protocol

Read the learner’s ignored local progress first. Confirm the current point, pending question, and preferred teaching language. English is the public default; a learner’s saved choice overrides it.
Teach only one point at a time: goal, explanation, a worked example, and one understanding question. Supply prerequisite concepts as needed; do not require videos or large assignments.
Stop after asking. Do not expose tutor criteria before the learner answers. Then acknowledge what is correct and address one misunderstanding.
If wrong or unclear, use the repair guidance and ask a smaller question about the same example. If a choice is correct without reasoning, ask why. Advance only when the learner can explain the reason and relevant conditions, or explicitly requests a skip; record skips as unconfirmed.
Derivations can span several turns. A lesson is not a one-turn quota. Neither reading it nor opening its answer marks mastery.
At session end, update ignored local/learning-progress.md with position, actual answer evidence, uncertainties, pending question, and next step. Never commit or automatically push personal progress to the curriculum repository. If file tools are unavailable, return the full updated text without claiming it was saved. Cross-device transfer is the learner's explicit private choice.
Distinguish course evidence, original teaching examples, executed experiments, and unverified claims. Preserve missing-source and ASR limitations. Check primary sources at teaching time for current prices, law, rankings, or software support.

## Chapters

- [00 · Foundations as needed (8 points)](#c00)
- [01 · What language models do (9 points)](#c01)
- [02 · From text to tokens (9 points)](#c02)
- [03 · Tensors and a minimal training program (9 points)](#c03)
- [04 · Learning from prediction errors (11 points)](#c04)
- [05 · Understanding the Transformer (12 points)](#c05)
- [06 · Modern architectures and hyperparameters (11 points)](#c06)
- [07 · Attention costs and alternatives (11 points)](#c07)
- [08 · Mixture of experts (11 points)](#c08)
- [09 · Compute, memory, and performance (11 points)](#c09)
- [10 · GPUs, TPUs, and hardware execution (10 points)](#c10)
- [11 · Profiling, compilation, and GPU kernels (11 points)](#c11)
- [12 · FlashAttention and recomputation (10 points)](#c12)
- [13 · Communication across devices (10 points)](#c13)
- [14 · Distributed training (12 points)](#c14)
- [15 · Inference and serving (14 points)](#c15)
- [16 · Scaling-law foundations (12 points)](#c16)
- [17 · Scaling small experiments to large models (10 points)](#c17)
- [18 · Evaluating models (10 points)](#c18)
- [19 · Sources of training data (6 points)](#c19)
- [20 · Data conversion and filtering (11 points)](#c20)
- [21 · Data deduplication (9 points)](#c21)
- [22 · Data mixtures and training stages (11 points)](#c22)
- [23 · Supervised and parameter-efficient fine-tuning (11 points)](#c23)
- [24 · Human preferences and RLHF (10 points)](#c24)
- [25 · Reasoning with verifiable rewards (14 points)](#c25)
- [26 · A small reinforcement-learning walkthrough (10 points)](#c26)
- [27 · Multimodal models (10 points)](#c27)
- [28 · Guest lectures and later evidence (6 points)](#c28)

<a id="c00"></a>
## 00 · Foundations as needed

Read the mathematical and programming notation needed for the course.

<a id="p00-01"></a>
### 00.01 Numbers and variables

**Goal:** Treat a variable as a replaceable number.
**Review if needed:** No prerequisites.

**Explanation:** A variable names a value; an expression specifies a calculation. Model parameters are numbers too, but training changes them. Identify each symbol before substituting values.

**Worked example:** With b sequences of t tokens, b=2 and t=3 gives bt=6. Change only b to 4: the total is 12.

**Ask this, then wait:** With t fixed, what happens when b doubles?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The token total doubles; only b changed.

**Repair:** Substitute one variable at a time if the learner changes both.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p00-02"></a>
### 00.02 Functions and parameters

**Goal:** Separate inputs, outputs, and parameters.
**Review if needed:** No prerequisites.

**Explanation:** A function maps inputs to outputs. In f(x;w)=wx, x is the current input and w controls the rule. Training changes w; inference uses a given w. In programming, “parameter” can also mean a function argument.

**Worked example:** With w=3, inputs 2 and 4 produce 6 and 12. With x=2 fixed, changing w to 5 produces 10.

**Ask this, then wait:** Does a changed output prove that training occurred?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. An input change can also change the output; check whether w changed.

**Repair:** Write x and w separately and change only one.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p00-03"></a>
### 00.03 Vectors and matrices

**Goal:** Name the axes of an array.
**Review if needed:** No prerequisites.

**Explanation:** A vector is an ordered list of numbers; a matrix has rows and columns. Axes need meanings. The same numbers might represent features, samples, or time; shape alone does not specify semantics.

**Worked example:** Two texts with 3 features each form a 2×3 matrix. Axis 1 selects text; axis 2 selects feature.

**Ask this, then wait:** After transposing to 3×2, does the first axis still select texts?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Transposition swaps axes without adding values.

**Repair:** Label axes “text” and “feature,” rather than only counting elements.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p00-04"></a>
### 00.04 Matrix multiplication

**Goal:** Compute one matrix-product entry.
**Review if needed:** No prerequisites.

**Explanation:** For A of shape m×k and B of shape k×n, AB has shape m×n. Entry i,j sums pairwise products of row i and column j. Inner dimensions must match; multiplication generally cannot be reversed.

**Worked example:** [1,2] times the column [3,4] gives 1×3+2×4=11. [3,8] would be elementwise multiplication.

**Ask this, then wait:** What shape results from 2×3 times 3×4?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** 2×4; each entry sums 3 products.

**Repair:** If the answer is 2×3, draw one row and one column; count output locations, not product terms.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p00-05"></a>
### 00.05 Probability distributions and conditioning

**Goal:** Read a distribution and its condition.
**Review if needed:** No prerequisites.

**Explanation:** Discrete probabilities are nonnegative and sum to 1. Conditioning fixes information and changes the population under consideration. P(A given B)=P(A and B)/P(B), requiring P(B)>0. It need not equal P(B given A).

**Worked example:** Of 10 people, 4 wear hats and 3 wear blue hats. Blue-hat probability is 3/4 among hat wearers and 3/10 overall.

**Ask this, then wait:** How can both 3/4 and 3/10 be correct?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Their conditions and denominators differ.

**Repair:** Circle the population retained by the condition, then count numerator and denominator.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p00-06"></a>
### 00.06 Logarithms and exponentials

**Goal:** Use logarithms to handle products.
**Review if needed:** No prerequisites.

**Explanation:** Exponentials turn sums into products; logarithms reverse that operation. For positive a,b, ln(ab)=ln a+ln b. The log of a probability below 1 is negative. Negative log probability can measure prediction loss.

**Worked example:** Two conditional probabilities of 1/2 give joint probability 1/4. Negative log probability is 2ln2≈1.386.

**Ask this, then wait:** If target probability rises from 1/4 to 1/2, does loss rise or fall?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It falls to ln2≈0.693.

**Repair:** Compare probabilities first, then account for the minus sign.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p00-07"></a>
### 00.07 Derivatives, gradients, and the chain rule

**Goal:** Understand the direction of a gradient update.
**Review if needed:** No prerequisites.

**Explanation:** A derivative measures local sensitivity. Derivatives over parameters form a gradient. A small step against it often reduces loss; the chain rule multiplies sensitivities through successive transformations. A large step need not help.

**Worked example:** For L=(w−3)², gradient is 2(w−3). With w=1 and learning rate 0.1, gradient −4 gives w=1.4.

**Ask this, then wait:** Why does descending the gradient increase w here?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Subtracting a negative gradient moves w toward the minimum at 3.

**Repair:** Compare loss before and after; “descent” does not mean every parameter decreases.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

Chain rule step by step: u=2w, L=(u−3)². Then dL/dw=(dL/du)(du/dw)=2(u−3)×2. At w=1, u=2 and the gradient is −4. Evaluate intermediate values first, then multiply local sensitivities. With multiple paths, add their contributions.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p00-08"></a>
### 00.08 Training, validation, and test sets

**Goal:** Separate learning from evaluation.
**Review if needed:** No prerequisites.

**Explanation:** Training data updates parameters. Validation data helps select settings. Test data checks the final selected system. Repeatedly selecting settings using test scores indirectly uses the test set. Splits must also consider shared sources and time leakage.

**Worked example:** Use 80 questions for training, 10 for tuning, and 10 for final testing. Repeatedly changing settings after viewing the final 10 makes that score optimistic.

**Ask this, then wait:** If no test gradients were computed, is leakage impossible?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Selection based on test performance also leaks information.

**Repair:** Ask whether the score influenced decisions, not only whether backpropagation occurred.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** This is an original prerequisite teaching arrangement, not a separate official lecture. Examples are not fabricated classroom attributions. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c01"></a>
## 01 · What language models do

Distinguish prediction, generation, and training.

<a id="p01-01"></a>
### 01.01 Next-token prediction

**Goal:** Interpret completion as a probability prediction.
**Review if needed:** 00.05.

**Explanation:** A language model assigns probabilities to the next token given previous tokens. It learns continuation patterns, not a guaranteed unique truth. Tokens can be word fragments; whole-word examples are an initial simplification.

**Worked example:** In 100 hypothetical similar sentences, “I put the milk in the…” ends with fridge 60 times, cup 25 times, and something else 15 times.

**Ask this, then wait:** Why can the most likely continuation still be wrong?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Highest probability is not probability 1; 40 of the hypothetical cases differ.

**Repair:** Use a sentence with several reasonable endings if the learner expects one guaranteed answer.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-02"></a>
### 01.02 Conditional probability

**Goal:** Explain how context changes a distribution.
**Review if needed:** 00.05.

**Explanation:** P(next token given prefix) conditions on the preceding text. Changing that context can change probabilities without changing parameters. Model probabilities are not deterministic commitments about reality.

**Worked example:** After “I drank a cup of,” tea might be likely; adding a café context might favor coffee. These are illustrative, not measured probabilities.

**Ask this, then wait:** Can probabilities change when the prefix changes but no training occurs?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Yes. The conditioning input changed.

**Repair:** Return to f(x;w): keep w fixed and change x.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-03"></a>
### 01.03 Sequence probability

**Goal:** Combine conditional probabilities across positions.
**Review if needed:** 00.05.

**Explanation:** A sequence probability is the product of each next-token probability conditioned on its own prefix. Taking negative logs turns the product into a sum of position losses. Tokens are not assumed independent.

**Worked example:** Suppose P(I)=0.5, P(like given I)=0.4, and P(tea given I like)=0.2. The sequence probability is 0.04.

**Ask this, then wait:** If only the final conditional probability doubles, what happens?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The sequence probability doubles to 0.08.

**Repair:** Write each prefix explicitly before multiplying.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-04"></a>
### 01.04 Training and inference

**Goal:** Distinguish learning from using a model.
**Review if needed:** 00.05.

**Explanation:** Training computes prediction errors and updates parameters. Inference computes distributions or generates text using existing parameters. Training includes forward computation too. Changed context, caches, or sampled outputs usually do not imply weight updates.

**Worked example:** Different answers on two days may simply reflect random sampling. An optimizer updating weights is the training distinction used here.

**Ask this, then wait:** Does a longer chat prove the weights learned about you?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. More context and changed weights are different.

**Repair:** Track input text, KV cache, and weights separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-05"></a>
### 01.05 Autoregressive generation

**Goal:** Walk through stepwise generation.
**Review if needed:** 00.05.

**Explanation:** Choose one token, append it to the prefix, and predict again. Earlier generated choices affect later conditions; an error can alter the subsequent path. An end token or length rule stops generation.

**Worked example:** Starting with “I enjoy,” choose “drinking,” then predict from “I enjoy drinking,” perhaps choosing “tea.”

**Ask this, then wait:** Why must the second generation step know the first output?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** That output is now part of its condition.

**Repair:** Write the entire prefix at every step.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-06"></a>
### 01.06 Pretraining and post-training

**Goal:** Understand the purpose of training stages.
**Review if needed:** 00.05.

**Explanation:** Pretraining commonly learns general prediction from broad text. Post-training adjusts instruction following, preferences, reasoning, or other behavior. The same network can use different data and objectives; stage names guarantee neither capabilities nor reliability.

**Worked example:** Pretraining may already contain question-answer text. SFT adds instruction-response examples; it does not create all knowledge from nothing.

**Ask this, then wait:** Does a more polite post-trained response prove greater factual accuracy?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Style and factual correctness need separate evaluation.

**Repair:** Construct a polite but incorrect answer and score both dimensions.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-07"></a>
### 01.07 Model history and openness

**Goal:** Judge openness by components.
**Review if needed:** 00.05.

**Explanation:** Language modeling includes n-grams, neural sequence models, and Transformers. Check availability of weights, code, data, recipes, and licenses separately. Downloadable weights alone do not enable reproduction.

**Worked example:** Model A provides weights but no data; model B provides code but no weights. Different components are open.

**Ask this, then wait:** Why might open weights still be insufficient to reproduce training?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Data, processing recipes, compute, or permissions may be missing.

**Repair:** Use a component checklist instead of a single openness label.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-08"></a>
### 01.08 Efficiency under fixed data and compute

**Goal:** Compare methods within a budget.
**Review if needed:** 00.05.

**Explanation:** Design trades off data, compute, memory, and time. Compare efficiency under a stated budget. A larger model may see too few tokens to perform best. Training objectives and deployment needs also differ.

**Worked example:** With 100 hypothetical compute units, one option trains a large model briefly; another trains a smaller model on more tokens. Compare final validation quality.

**Ask this, then wait:** Why is “larger” insufficient to select the first option?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Training volume and optimization under the budget affect quality.

**Repair:** Account for every compute expense before comparing.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p01-09"></a>
### 01.09 Limits of small-model experience

**Goal:** State what a small experiment supports.
**Review if needed:** 00.05.

**Explanation:** Small models make debugging and comparisons cheaper. Scale can change communication, optimization, data needs, and capabilities. Small-scale results generate hypotheses; several scales and appropriate tuning are needed for transfer.

**Worked example:** A hypothetical 10% small-model speedup can disappear when communication dominates at larger scale.

**Ask this, then wait:** Does one small-model win prove a large-model win?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Check multiple scales, equal budgets, and uncertainty.

**Repair:** State the directly measured range before making an extrapolation.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c02"></a>
## 02 · From text to tokens

Understand tokenization and its tradeoffs.

<a id="p02-01"></a>
### 02.01 Unicode and UTF-8

**Goal:** Separate code points from encoded bytes.
**Review if needed:** 01.01.

**Explanation:** Unicode assigns code points; UTF-8 encodes them as variable-length byte sequences. Code points, visible characters, and tokens differ. A visible combined character can contain several code points.

**Worked example:** ASCII A uses 1 UTF-8 byte; 中 usually uses 3. Neither fact determines token count.

**Ask this, then wait:** Does 3 UTF-8 bytes imply 3 tokens?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. A tokenizer may merge bytes or use other units.

**Repair:** Draw visible character, code point, byte, and token as separate layers.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-02"></a>
### 02.02 Characters, bytes, words, and subwords

**Goal:** Compare tokenization units.
**Review if needed:** 01.01.

**Explanation:** Words are intuitive but complicate unseen words and multilingual text. Characters reduce vocabulary but lengthen sequences; bytes cover UTF-8 inputs but may be longer. Subwords trade vocabulary size against sequence length.

**Worked example:** “playing” could be 1 word, 7 ASCII bytes, or the illustrative subwords play and ing. Actual segmentation depends on the tokenizer.

**Ask this, then wait:** Why is there no unique token count for a text?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The unit and learned segmentation rules differ.

**Repair:** Specify a tokenizer before claiming an actual encoding.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-03"></a>
### 02.03 Token IDs and vocabulary

**Goal:** Treat IDs as indices.
**Review if needed:** 01.01.

**Explanation:** A vocabulary maps tokens to integer IDs. An embedding maps IDs to vectors. Nearby ID numbers do not imply nearby meanings. More vocabulary items change embedding and output-classification dimensions.

**Worked example:** Cat=7, dog=8, bridge=9 are arbitrary indices; 7 and 8 being adjacent does not establish semantic similarity.

**Ask this, then wait:** How does the model obtain features from an ID?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It looks up an embedding row rather than treating the ID as a continuous quantity.

**Repair:** Swap IDs together with their table rows to show numbering is arbitrary.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-04"></a>
### 02.04 BPE training, encoding, and decoding

**Goal:** Walk through a merge.
**Review if needed:** 01.01.

**Explanation:** BPE repeatedly counts adjacent symbol pairs and merges frequent pairs. Encoding applies learned merge priorities to new text; decoding maps tokens back to basic units. Ties require a fixed rule, and protected boundaries constrain merges.

**Worked example:** For abab, start a b a b. The pair ab occurs 2 times; merging gives ab ab. A later merge could combine those tokens.

**Ask this, then wait:** Must encoding a new text retrain BPE?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Apply the existing merge rules.

**Repair:** Separate learning a merge table from applying it; do not retrain on test text.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-05"></a>
### 02.05 Special tokens and pretokenization

**Goal:** Explain segmentation boundaries and control markers.
**Review if needed:** 01.01.

**Explanation:** Pretokenization limits where merges can occur. Special tokens represent termination, roles, or other structure and have designated handling. Ordinary text resembling a marker should not automatically gain control meaning; permitted behavior depends on implementation.

**Worked example:** Splitting a a at whitespace may prevent cross-space merging; an end token tells the generation loop to stop.

**Ask this, then wait:** Can changed pretokenization affect encoding with the same BPE rules?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Yes. It changes the allowed merge regions.

**Repair:** Draw protected boundaries before counting pairs.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-06"></a>
### 02.06 Unigram and SentencePiece

**Goal:** Separate algorithms from libraries.
**Review if needed:** 01.01.

**Explanation:** Unigram starts from candidate subwords and probabilities, selects likely segmentations, and prunes the vocabulary. BPE builds merge rules. SentencePiece is a toolkit supporting several models, not a synonym for Unigram.

**Worked example:** With ab, a, and b available, Unigram can compare P(ab) against P(a)P(b); BPE follows merge priorities.

**Ask this, then wait:** Does using SentencePiece prove the algorithm is Unigram?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Check configuration; BPE is also supported.

**Repair:** Write the tool name and algorithm name in separate columns.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-07"></a>
### 02.07 Normalization, whitespace, numbers, and languages

**Goal:** Recognize information lost in normalization.
**Review if needed:** 01.01.

**Explanation:** Normalization can unify case, spaces, or compatibility characters, but may destroy original distinctions. Number and multilingual segmentation affect modeling biases. Reconstructing normalized text differs from recovering original bytes.

**Worked example:** Lowercasing turns both US and us into us. Decoding alone cannot recover the original case.

**Ask this, then wait:** Does reversible decoding always recover the original text?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It may only recover the normalized form.

**Repair:** Check whether two different original inputs map to one output.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-08"></a>
### 02.08 Compression and vocabulary size

**Goal:** Balance sequence length against vocabulary cost.
**Review if needed:** 01.01.

**Explanation:** A larger vocabulary often uses fewer tokens but enlarges embeddings and output projection. State compression units, such as bytes/token, and inspect language-specific differences. Fewer tokens alone do not prove lower total training cost.

**Worked example:** 1000 bytes in 250 tokens is 4 bytes/token; in 200 tokens it is 5, but the larger vocabulary may cost more.

**Ask this, then wait:** Does improved compression alone prove cheaper computation?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Include vocabulary-layer costs and architecture.

**Repair:** Track sequence length and vocabulary size together.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p02-09"></a>
### 02.09 Byte-level and tokenizer-free approaches

**Goal:** Understand the cost of avoiding conventional segmentation.
**Review if needed:** 01.01.

**Explanation:** Byte models use a small byte alphabet. Tokenizer-free methods may still use dynamic chunks or latent representations. They avoid some vocabulary issues while retaining sequence-length, efficiency, and quality tradeoffs.

**Worked example:** A Chinese sentence may need more byte positions than subword positions, increasing attention work despite a smaller vocabulary.

**Ask this, then wait:** Why do 256 byte values not guarantee cheaper computation?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Longer sequences can increase computation and caches.

**Repair:** Count input positions as well as vocabulary parameters.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L01](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_01.py) · [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L01](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_01.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c03"></a>
## 03 · Tensors and a minimal training program

Follow shapes through a model.

<a id="p03-01"></a>
### 03.01 Shape, dimensions, and batches

**Goal:** Name tensor axes.
**Review if needed:** 00.03, 00.04.

**Explanation:** A tensor is a multidimensional array. [B,T,D] often means batch, token position, and hidden features. Axis ordering is a convention and may change between operators; check it before transforming data.

**Worked example:** [2,3,4] contains 24 values. Averaging over the feature axis leaves [2,3].

**Ask this, then wait:** Does swapping B and T change element count?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No; index meanings and layout may change.

**Repair:** Locate a value by text, position, and feature rather than shape alone.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-02"></a>
### 03.02 Dtypes and devices

**Goal:** Separate numerical format from location.
**Review if needed:** 00.03, 00.04.

**Explanation:** A dtype specifies representation and associated numerical rules; a device specifies where data lives. Cross-device operations may require transfers. Lower precision changes range or rounding as well as storage size.

**Worked example:** Moving FP32 data from CPU to GPU changes location; converting it to BF16 changes dtype.

**Ask this, then wait:** Does moving to GPU necessarily reduce bit width?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Device and dtype are independent choices.

**Repair:** Record dtype and device in separate columns.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-03"></a>
### 03.03 Storage, strides, views, and slicing

**Goal:** Separate logical shape from physical storage.
**Review if needed:** 00.03, 00.04.

**Explanation:** A stride tells how far storage advances along an axis. Transposes and slices can share storage. A view requires compatible layout; reshape can copy. Equal shapes do not guarantee contiguous or independent storage.

**Worked example:** A contiguous 2×3 array commonly has strides (3,1); its 3×2 transpose has (1,3), without moving all values.

**Ask this, then wait:** How can a cheap transpose affect later performance?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It changes addressing and access continuity; later operators may require a different layout.

**Repair:** Draw six storage cells and two coordinate systems.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-04"></a>
### 03.04 Broadcasting, elementwise operations, and matrix products

**Goal:** Apply the correct shape rule.
**Review if needed:** 00.03, 00.04.

**Explanation:** Broadcasting aligns trailing axes; dimensions must match or one must be 1. Elementwise operations align values, whereas matrix products contract an inner axis. Successful execution alone does not ensure intended semantics.

**Worked example:** [2,3]+[3] adds one vector to each row. Matrix multiplication of [2,3] and [3,4] gives [2,4].

**Ask this, then wait:** Can [2,3] and [2] broadcast directly for addition?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Trailing dimensions 3 and 2 conflict.

**Repair:** If row-wise expansion is intended, explicitly reshape [2] to [2,1].

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-05"></a>
### 03.05 Einsum, einops, and shape labels

**Goal:** Read which axes are retained or reduced.
**Review if needed:** 00.03, 00.04.

**Explanation:** Einsum labels express retained and summed axes; einops expresses rearrangement, splitting, and merging. Rearrangement does not itself imply reduction. Verify actual library syntax against its version.

**Worked example:** 'btd,dv->btv' contracts d, producing vocabulary scores for each batch and position.

**Ask this, then wait:** Which label disappears, and what does that mean?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** d disappears because hidden features are summed.

**Repair:** Write y[b,t,v]=Σd x[b,t,d]w[d,v].

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-06"></a>
### 03.06 Parameters, embeddings, and modules

**Goal:** Distinguish learnable state from lookup indices.
**Review if needed:** 00.03, 00.04.

**Explanation:** A Parameter is a learnable tensor; an Embedding selects rows by ID; a Module organizes parameters, submodules, and forward computation. Not every tensor is a parameter or should require gradients.

**Worked example:** A 5×3 embedding has 15 learnable numbers. IDs [1,4] select two rows, producing [2,3].

**Ask this, then wait:** Are input IDs optimized like embedding weights?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Usually not. IDs index the learnable table.

**Repair:** Separate index, table, and selected vectors.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-07"></a>
### 03.07 Automatic differentiation

**Goal:** Understand graph-based gradient accumulation.
**Review if needed:** 00.03, 00.04.

**Explanation:** Autograd records differentiable operations and applies the chain rule. Backward can add into existing gradient buffers, so clearing must match the training loop. Detaching or converting values can break gradient paths.

**Worked example:** For y=w² at w=3, one backward contributes 6; repeating without clearing can leave 12 in the buffer.

**Ask this, then wait:** Does 12 mean the derivative of one evaluation changed?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Contributions accumulated; each derivative is still 6.

**Repair:** Track the fresh derivative and the old buffer separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-08"></a>
### 03.08 Loading data, seeds, and reproducibility

**Goal:** Identify the state needed for reproducibility.
**Review if needed:** 00.03, 00.04.

**Explanation:** Data loading controls order and batch composition. Seeds control some random streams, but multiprocessing, kernels, and environments may remain nondeterministic. Record code, data versions, environment, and settings.

**Worked example:** The same seed with a changed data order can produce a different second batch.

**Ask this, then wait:** Does setting a seed guarantee bitwise agreement across devices?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Algorithms, environment, and execution can differ.

**Repair:** Treat the seed as one checklist item, not the whole checklist.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p03-09"></a>
### 03.09 Saving and restoring checkpoints

**Goal:** Save enough state to continue training.
**Review if needed:** 00.03, 00.04.

**Explanation:** Inference may need weights and configuration. Exact continuation also needs optimizer, scheduler, step, random state, and data position. Test restoring; a file existing does not establish recoverability.

**Worked example:** Saving only weights at step 100 loses Adam moments; restored step 101 generally differs from uninterrupted training.

**Ask this, then wait:** Why might weights alone fail to preserve the trajectory?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Optimizer, random, or data state may be missing.

**Repair:** Distinguish “can run inference” from “continues the same training run.”

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c04"></a>
## 04 · Learning from prediction errors

Walk through a parameter update.

<a id="p04-01"></a>
### 04.01 Logits and softmax

**Goal:** Convert scores into a distribution.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Logits are unnormalized real scores. Softmax uses pi=exp(zi)/Σexp(zj), producing nonnegative values summing to 1. Adding a common constant changes no probability; larger score differences concentrate the distribution.

**Worked example:** [0,ln3] exponentiates to [1,3], giving [1/4,3/4].

**Ask this, then wait:** Does adding 100 to every logit change theoretical probabilities?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. The common exponential factor cancels.

**Repair:** Distinguish a common shift from increasing one score.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-02"></a>
### 04.02 Cross-entropy and numerical stability

**Goal:** Compute target loss stably.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** For one target class, cross-entropy is −ln ptarget. Exponentiating first can overflow. Log-sum-exp with maximum subtraction computes it more stably. Batch averages also need explicit masks and denominators.

**Worked example:** Target probabilities 0.25 and 0.5 give losses about 1.386 and 0.693. Shift logits [1000,1001] to [−1,0] for stable exponentiation.

**Ask this, then wait:** Why does subtracting the maximum preserve cross-entropy?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** A common shift preserves softmax; the stable expression accounts for that shift.

**Repair:** Prove probability invariance first; do not claim floating-point error vanishes.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-03"></a>
### 04.03 Gradient descent and backpropagation

**Goal:** Connect a loss to an update.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Backpropagation computes parameter gradients; descent applies w←w−ηg. It does not directly write an answer into weights. A batch improvement need not improve validation quality.

**Worked example:** L=(w−2)² at w=0 has g=−4. With η=0.25, w becomes 1 and loss falls from 4 to 1.

**Ask this, then wait:** Must an arbitrarily large step against the gradient lower loss?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It can overshoot; step size and curvature matter.

**Repair:** Draw the minimum and try η=2.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-04"></a>
### 04.04 SGD, momentum, and AdamW

**Goal:** Compare what optimizers remember.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** SGD uses current gradients; momentum accumulates directions. Adam estimates first and second moments with bias correction. AdamW separates weight decay from adaptive gradient updates. Tune each optimizer appropriately.

**Worked example:** Same-sign gradients accumulate momentum; alternating signs can cancel in the first moment, but their squares remain positive.

**Ask this, then wait:** Do squared gradients cancel when signed gradients do?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Direction estimates and scale estimates differ.

**Repair:** Track g and g² separately; the second moment is not a second derivative.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

AdamW: m←β₁m+(1−β₁)g, v←β₂v+(1−β₂)g². At step t, m̂=m/(1−β₁ᵗ), v̂=v/(1−β₂ᵗ) correct zero-initialization bias. A simplified update is w←(1−ηλ)w−ηm̂/(√v̂+ε). λ controls decay and ε stabilizes the denominator. Parameter-group and schedule conventions depend on implementation. v is a second moment, not a Hessian.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-05"></a>
### 04.05 Initialization

**Goal:** Control initial scale.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Random initialization breaks symmetry and should control activation and gradient scales across width and depth. Identical neuron initialization can yield identical features. Suitable variance depends on fan-in, fan-out, and activation.

**Worked example:** Under independent zero-mean assumptions, summing n weighted inputs gives variance about n Var(w)Var(x). Taking Var(w) proportional to 1/n controls it.

**Ask this, then wait:** If width doubles with unchanged weight variance, what may happen?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Under those assumptions, summed variance roughly doubles.

**Repair:** State independence and zero-mean assumptions before summing variances.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-06"></a>
### 04.06 Warmup, cosine decay, and WSD

**Goal:** Distinguish learning-rate phases.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Warmup raises the learning rate gradually; cosine schedules decay using a cosine shape. WSD separates warmup, stable training, and decay, allowing shared prefixes. Ending schedules matter for fair budget comparisons.

**Worked example:** Two runs share 80 stable steps, then one decays for 20 while the other continues. Only the prefix before branching is shared.

**Ask this, then wait:** Does prefix sharing eliminate training for each ending?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Each decay branch still costs computation.

**Repair:** Draw the shared path and separate tails.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

An illustrative cosine decay over progress s∈[0,1] is η(s)=ηmin+(ηmax−ηmin)(1+cos(πs))/2. It starts at ηmax and ends at ηmin. Warmup and WSD tails may use other functions; WSD does not prescribe one universal cosine formula.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-07"></a>
### 04.07 Gradient clipping

**Goal:** Limit the norm of an update signal.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Global norm clipping computes G over gradients and, if G>c, multiplies all by c/G. It preserves direction while limiting size. Elementwise clipping is different; neither fixes every instability.

**Worked example:** [3,4] has norm 5. Clipping to norm 1 gives [0.6,0.8].

**Ask this, then wait:** Does clipping each coordinate to 1 preserve the same direction?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. [1,1] points elsewhere.

**Repair:** Compare uniform scaling with coordinate truncation.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-08"></a>
### 04.08 Batches and gradient accumulation

**Goal:** Weight microbatches correctly.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Accumulate gradients across smaller microbatches before one update. Weight by examples or valid tokens according to the objective, and clip after aggregation. Dropout or batch-dependent operators can limit equivalence.

**Worked example:** Microbatches with 2 and 6 valid tokens and mean losses 1 and 3 have combined mean (2×1+6×3)/8=2.5, not 2.

**Ask this, then wait:** Why is averaging those two means wrong?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The microbatches carry different weights.

**Repair:** Recover loss sums before dividing by the total count.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-09"></a>
### 04.09 The training loop

**Goal:** Walk through a full update.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Load a batch, run forward, compute masked loss, backpropagate, process gradients, update, and clear gradients. Validate and checkpoint on schedule. Scheduler timing must be consistent; validation usually disables gradient recording and uses evaluation mode.

**Worked example:** Backward without optimizer step does not update weights; failing to clear gradients mixes past contributions into later updates.

**Ask this, then wait:** Why is loss.backward alone not a parameter update?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It computes gradients; the optimizer applies them.

**Repair:** Inspect weight values separately from gradient buffers.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-10"></a>
### 04.10 Validation curves, overfitting, and tuning

**Goal:** Use curves to form diagnoses.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Falling training loss with rising validation loss suggests overfitting or distribution differences. Both high can reflect capacity, optimization, or code problems. First test tiny-data fitting; one curve alone does not diagnose the cause.

**Worked example:** If 10 examples cannot be fitted, inspect label shifts, masks, and updates before buying a larger training run.

**Ask this, then wait:** Why might “train longer” fail when validation worsens?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It can intensify overfitting; diagnose the gap first.

**Repair:** Compare both curves and the data split, then propose a testable explanation.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p04-11"></a>
### 04.11 Ablation studies

**Goal:** Design interpretable comparisons.
**Review if needed:** 00.06, 00.07, 03.07.

**Explanation:** Change a key factor while controlling data, budget, evaluation, and tuning effort. Interactions can make effects conditional. Repeated runs estimate randomness. Conclusions apply to the studied conditions.

**Worked example:** Changing normalization while doubling training tokens cannot isolate normalization's contribution.

**Ask this, then wait:** What should be controlled before attributing a gain to normalization?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** At least data and training budget, with the tested change explicit.

**Repair:** List every difference between the runs and remove confounders.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L01](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_01.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c05"></a>
## 05 · Understanding the Transformer

Follow one token through the model.

<a id="p05-01"></a>
### 05.01 Embeddings and linear layers

**Goal:** Connect discrete inputs to continuous features.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** An embedding selects a vector by token ID; a linear layer maps vectors through weights. Lookup and transformation differ. Tied input/output weights are a separate design choice.

**Worked example:** A 100×8 table has 800 values. One token selects an 8-vector, then an 8×4 matrix maps it to 4 features.

**Ask this, then wait:** Must lookup sum every vocabulary row?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It selects the indexed row.

**Repair:** Use an indexed shelf analogy while noting that vector contents are learned.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-02"></a>
### 05.02 Queries, keys, and values

**Goal:** Separate attention addressing from content.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** Different projections produce Q, K, and V from hidden representations. Q and K determine attention weights; those weights combine V. “Question/address/content” is an analogy, not a literal semantic assignment.

**Worked example:** Weights 0.8 and 0.2 over values [1,0] and [0,1] give [0.8,0.2].

**Ask this, then wait:** If only V changes, do these attention weights change?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Not in this layer's present computation; the combined content changes.

**Repair:** Draw score computation and content aggregation separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

Single-head shapes: X:T×D, WQ/WK:D×dk, WV:D×dv. Thus Q/K:T×dk and V:T×dv. P=softmax(QKᵀ/√dk+mask) has shape T×T; PV has shape T×dv. Softmax normalizes over key positions. Batch and multiple heads add axes around this chain.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-03"></a>
### 05.03 Dot products, scaling, and softmax

**Goal:** Explain attention score scaling.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** Attention commonly uses softmax(QKᵀ/√dk). With independent unit-variance components, dot-product variance grows with dk. Dividing by √dk controls scale and helps avoid saturation; real training need not satisfy every assumption.

**Worked example:** A dot product 8 becomes 4 when dk=4, or 2 when dk=16.

**Ask this, then wait:** Does dividing by √dk force scores into [0,1]?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Softmax later produces normalized weights.

**Repair:** Separate scaled real scores from probabilities.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-04"></a>
### 05.04 Causal masking

**Goal:** Prevent access to future targets.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** A causal mask blocks future positions. Before softmax, forbidden scores become negative infinity or an equivalent masked value, yielding zero weights. Both masking and next-token label shifts must be correct.

**Worked example:** For a b c, position b may see a,b but not c, while predicting c.

**Ask this, then wait:** Is seeing b at position b already seeing the target c?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Current input and next-position target differ.

**Repair:** Draw input and shifted-target rows before checking visibility.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-05"></a>
### 05.05 Multi-head self-attention

**Goal:** Understand separate groups of attention.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** Heads independently attend over projected features; their results are concatenated and projected. They may learn different relations without each having a fixed named meaning. More heads need not mean greater total width.

**Worked example:** Width 8 with 2 heads gives width 4 per head; concatenation returns width 8.

**Ask this, then wait:** With total width 8 and 4 heads, what is each head's width?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** 2, not 4.

**Repair:** Write total width=head count×head width before counting costs.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-06"></a>
### 05.06 Position representations

**Goal:** Explain why order needs a mechanism.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** Without position-related structure, self-attention cannot directly distinguish token order. Absolute embeddings, relative biases, and RoPE inject position differently; short-context training does not guarantee unlimited length extrapolation.

**Worked example:** “Dog bites person” and “Person bites dog” share words but differ in order and meaning.

**Ask this, then wait:** What is missing if only word vectors and no position mechanism are supplied?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Order information needed to distinguish reordered relations.

**Repair:** Swap positions with the same tokens and identify how the mechanism can detect it.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-07"></a>
### 05.07 MLPs and feed-forward networks

**Goal:** Identify position-wise nonlinear computation.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** An FFN commonly expands width, applies an activation, and projects back. It reuses weights across positions but ordinarily does not directly mix positions. Attention or another mechanism provides cross-position exchange.

**Worked example:** Expand width 4 to 12 and back to 4. Each position independently uses the same function.

**Ask this, then wait:** Can a wider position-wise FFN directly replace reading other tokens?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. More within-position computation adds no cross-position connection.

**Repair:** Contrast vertical per-position paths with attention links.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-08"></a>
### 05.08 Residual connections

**Goal:** Understand a direct information path.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** A residual y=x+f(x) supplies direct information and gradient paths. Addition needs compatible shapes. Residuals help deep optimization without guaranteeing freedom from exploding gradients or successful learning.

**Worked example:** [1,2]+[0.1,−0.2]=[1.1,1.8], not a concatenated 4-vector.

**Ask this, then wait:** If f is nearly zero, what is the output near?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** x.

**Repair:** Separate addition from concatenation and check dimensions.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-09"></a>
### 05.09 Normalization

**Goal:** Identify the normalization axis.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** Transformer normalization typically operates over each position's features, then applies learned scaling or related parameters. This differs from cross-example BatchNorm. LayerNorm and RMSNorm use different statistics.

**Worked example:** For [B,T,D], statistics are over D separately at each token, not over the whole sequence.

**Ask this, then wait:** Does changing another batch example directly change ordinary LayerNorm statistics here?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Positions are normalized independently.

**Repair:** Circle the statistic's axis before discussing normalization.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-10"></a>
### 05.10 Stacking Transformer blocks

**Goal:** Follow representations across layers.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** Blocks combine attention, FFN, residuals, and normalization. Later layers use earlier contextual representations. Repeated block structure usually has distinct weights in each layer, unless sharing is explicit.

**Worked example:** One layer can form contextual relations used by the next; this is intuition, not a measured fixed responsibility for each layer.

**Ask this, then wait:** Does repeating one structure 12 times imply only one set of weights?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Check whether weights are explicitly shared.

**Repair:** Distinguish a structural template from parameter instances.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-11"></a>
### 05.11 Output projection and probabilities

**Goal:** Map hidden features to vocabulary scores.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** A final hidden vector projects to one logit per vocabulary item, then softmax yields a distribution. Hidden width D differs from vocabulary V. Tying weights is optional; sampling still selects a token.

**Worked example:** D=4 and V=10 use a 4×10 projection to produce 10 scores.

**Ask this, then wait:** Do 10 probabilities already constitute a sentence?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Select a next token and iterate.

**Repair:** Follow vector→scores→probabilities→token selection.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p05-12"></a>
### 05.12 Parallel training and stepwise generation

**Goal:** Separate execution order from information access.
**Review if needed:** 02.03, 03.01, 04.01.

**Explanation:** Training knows the target sequence and can compute positions in parallel under a causal mask. Generation normally needs a previous sampled token before the next step. Parallel execution does not authorize future-token access.

**Worked example:** Given a b c, training predicts b after a and c after b together. With only a during generation, b must first be produced.

**Ask this, then wait:** Why retain a causal mask during parallel training?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Parallelism concerns execution; masking controls allowed information.

**Repair:** Ask separately when computation runs and which positions it may read.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c06"></a>
## 06 · Modern architectures and hyperparameters

Understand the problem each design addresses.

<a id="p06-01"></a>
### 06.01 Pre-norm and post-norm

**Goal:** Compare normalization paths.
**Review if needed:** 05.10.

**Explanation:** Pre-norm commonly writes y=x+f(N(x)); post-norm writes y=N(x+f(x)). The location changes residual and gradient paths. Pre-norm often helps optimization, but outcomes depend on depth, initialization, and training.

**Worked example:** If f=0, pre-norm outputs x; post-norm still outputs N(x).

**Ask this, then wait:** Must the two forms agree when f=0?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. One still normalizes the residual output.

**Repair:** Trace whether the direct path must pass through N.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-02"></a>
### 06.02 LayerNorm and RMSNorm

**Goal:** Separate centering from scale normalization.
**Review if needed:** 05.10.

**Explanation:** LayerNorm centers and divides by √(variance+ε), then scales/shifts. RMSNorm divides x by √(mean x²+ε), usually scaling without centering. Variance and mean square are different.

**Worked example:** For [1,3], mean=2, variance=1, mean square=5. Ignoring ε and learned parameters, LN gives [−1,1]; RMSNorm gives [1,3]/√5.

**Ask this, then wait:** Must RMSNorm produce zero-mean features?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It does not subtract the mean.

**Repair:** Use positive inputs as a counterexample and retain 1/D in mean square.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-03"></a>
### 06.03 Double norm and non-residual post-norm

**Goal:** Locate each normalization in a graph.
**Review if needed:** 05.10.

**Explanation:** Double normalization can normalize both sublayer inputs and outputs. A branch-only form is y=x+N₂(f(N₁(x))), unlike normalizing the entire residual sum. Names vary; the graph defines the operation.

**Worked example:** Place N₂ after the branch but before addition; the main x path remains direct.

**Ask this, then wait:** Is branch-output normalization equivalent to normalizing the entire sum?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Generally not. Placement changes the main path.

**Repair:** Draw the addition node and mark every norm before or after it.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-04"></a>
### 06.04 ReLU, GELU, GLU, and SwiGLU

**Goal:** Understand activation and gating.
**Review if needed:** 05.10.

**Explanation:** ReLU removes negative values; GELU smoothly modulates them. GLU multiplies a content branch by a gate; SwiGLU uses a SiLU gate. Gated FFNs usually add projections, so compare with adjusted intermediate widths.

**Worked example:** Content value 2 multiplied by gate 0.3 gives 0.6. The gate modulates features, not a choice between tokens.

**Ask this, then wait:** At equal intermediate width, must gated and ordinary FFNs have equal cost?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Gating generally adds an input projection.

**Repair:** Count matrices before comparing parameter or compute budgets.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

ReLU(x)=max(0,x); GELU(x)=xΦ(x), where Φ is the standard normal CDF and implementations may approximate it; SiLU(x)=x·sigmoid(x). One SwiGLU form is [SiLU(XWg)⊙(XWu)]Wd, with elementwise ⊙. Biases and intermediate width depend on the model. The gate need not lie between 0 and 1.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-05"></a>
### 06.05 Removing biases

**Goal:** Recognize an input-independent offset.
**Review if needed:** 05.10.

**Explanation:** In y=xW+b, b is an input-independent shift. Removing some biases simplifies parameters and computation but does not remove every addition or guarantee better quality.

**Worked example:** With x=0, a biased linear layer gives b; an unbiased one gives 0. Other network components can still output nonzero values.

**Ask this, then wait:** Does removing bias also remove residual connections?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Adding an input is different from adding a constant.

**Repair:** Label x and b separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-06"></a>
### 06.06 Serial and parallel blocks

**Goal:** Compare dependency chains.
**Review if needed:** 05.10.

**Explanation:** In serial blocks, the FFN reads attention-updated features. Parallel branches can read the same input and merge later. This shortens dependencies but changes the function and needs suitable training.

**Worked example:** Serial: x+a(x)+m(x+a(x)). Simplified parallel: x+a(x)+m(x).

**Ask this, then wait:** Is this only a reordered execution of the same function?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Generally no; the FFN input differs.

**Repair:** Circle the argument passed to m in each expression.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-07"></a>
### 06.07 Rotary position embeddings

**Goal:** Understand relative position through rotations.
**Review if needed:** 05.10.

**Explanation:** RoPE rotates paired Q/K dimensions by position. Rotation composition makes dot products depend on relative position. It does not merely append an integer; frequencies and long-context extensions require validation.

**Worked example:** (R(p)q)ᵀR(s)k=qᵀR(s−p)k exposes the position difference.

**Ask this, then wait:** If both positions shift equally, does their difference change?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No, with the same rotation-frequency rule.

**Repair:** Use a two-dimensional rotation without claiming infinite-length extrapolation.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-08"></a>
### 06.08 Depth, width, heads, and FFN ratios

**Goal:** Account for architecture choices.
**Review if needed:** 05.10.

**Explanation:** Depth counts layers; width counts hidden features; heads partition attention; FFN ratio sets intermediate width. Dense matrix parameters often grow quadratically with width. Changing heads at fixed width differs from increasing width.

**Worked example:** A D×D matrix has D² parameters. Doubling D makes 4 times as many; doubling layers roughly doubles this component.

**Ask this, then wait:** Why can doubling width cost more parameters than doubling depth?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Both matrix dimensions increase.

**Repair:** Draw matrix area; embeddings and other components can scale differently.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-09"></a>
### 06.09 Vocabulary size

**Goal:** Include vocabulary-layer costs.
**Review if needed:** 05.10.

**Explanation:** A vocabulary V and hidden width D require about VD embedding parameters. Output projection is similar; tied weights can share storage. Weigh shorter sequences against parameters, softmax cost, and language coverage.

**Worked example:** V=1000,D=64 gives 64000 values. Doubling V doubles this table rather than quadrupling it.

**Ask this, then wait:** Does weight tying eliminate the output computation?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It shares storage, not the need to calculate scores.

**Repair:** Separate shared parameters from eliminated operations.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-10"></a>
### 06.10 Dropout and weight decay

**Goal:** Distinguish masking from shrinking weights.
**Review if needed:** 05.10.

**Explanation:** Dropout randomly masks activations during training with suitable scaling and is ordinarily disabled for evaluation. Weight decay shrinks weights during optimization. Both can regularize, but their mechanisms and settings differ.

**Worked example:** An AdamW weight can shrink when its gradient is 0; dropout instead changes activation paths in a forward pass.

**Ask this, then wait:** Does disabling dropout automatically disable weight decay?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Locate one in forward computation and the other in the optimizer.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p06-11"></a>
### 06.11 Z-loss, QK normalization, and logit soft-capping

**Goal:** Separate stabilization methods.
**Review if needed:** 05.10.

**Explanation:** Z-loss can penalize squared log-sum-exp; QK normalization changes query/key scales; soft-capping uses a smooth bound such as c·tanh(z/c). Formulas, coefficients, and locations depend on implementation.

**Worked example:** For c=2, a large positive score approaches 2 after capping. QK normalization acts on vectors before scoring.

**Ask this, then wait:** Can all three be described as gradient clipping?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. They act on forward scores or vectors, unlike gradient clipping.

**Repair:** Place each method on its own computational-graph location.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L03](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e9cb2488fdb53ea37f0e38924ec3a1701925cef3/nonexecutable/2025%20Lecture%203%20-%20architecture.pdf) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c07"></a>
## 07 · Attention costs and alternatives

Understand long-context tradeoffs.

<a id="p07-01"></a>
### 07.01 Full-attention compute and cache costs

**Goal:** Separate training and decoding length costs.
**Review if needed:** 05.05, 09.01.

**Explanation:** Training scores have shape T×T per head, with QKᵀ work roughly T²d. During generation, KV storage generally grows linearly with history; computing one new token also reads increasing history. Do not confuse full-sequence training with one decode step.

**Worked example:** Increasing T from 100 to 200 grows score entries from 10000 to 40000, but doubles cached positions.

**Ask this, then wait:** Must doubling context quadruple the KV cache?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Usually not; cache size roughly doubles. Pairwise score matrices have the quadratic term.

**Repair:** Draw a square score table separately from a history list.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-02"></a>
### 07.02 Multi-query and grouped-query attention

**Goal:** Explain shared keys and values.
**Review if needed:** 05.05, 09.01.

**Explanation:** MHA has separate KV heads for query heads. MQA shares one KV group; GQA shares among groups. This chiefly reduces KV storage and reads while retaining multiple queries. Quality and actual speed still require measurement.

**Worked example:** With 8 query heads and 2 KV heads, groups of 4 queries share KV. The cache is smaller than with 8 KV heads.

**Ask this, then wait:** Do 8 query heads require 8 KV groups?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No; inspect the number of KV heads.

**Repair:** Draw 8 readers sharing 2 records while keeping independent queries.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-03"></a>
### 07.03 MLA and cross-layer cache sharing

**Goal:** Distinguish compression from layer reuse.
**Review if needed:** 05.05, 09.01.

**Explanation:** MLA compresses part of KV information into smaller latent representations with associated projections. Cross-layer sharing reuses caches across layers. Compressed width and independent layer count are different savings; positional components may need separate storage.

**Worked example:** Reducing 100 stored dimensions to 30 is width compression. Letting two layers share 100 dimensions is layer sharing.

**Ask this, then wait:** Does a low-dimensional latent make cache size independent of length?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Historical positions generally still need stored latents or related information.

**Repair:** Write cache size as positions×per-position storage×independent layers.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-04"></a>
### 07.04 Sliding windows and sparse attention

**Goal:** Recognize the cost of missing connections.
**Review if needed:** 05.05, 09.01.

**Explanation:** A window of W positions can reduce attention work from roughly T² to TW. Sparse attention retains selected connections. Removed long-range information may require deeper propagation or another route.

**Worked example:** With T=1000,W=10, a position reads roughly 10 positions; the last token cannot directly read the first in one local layer.

**Ask this, then wait:** Does a cheaper local window prove distant information is unimportant?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It imposes a restriction that can hurt tasks.

**Repair:** Test a reference to a name far outside the window.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-05"></a>
### 07.05 Alternating global and local layers

**Goal:** Account for both layer types.
**Review if needed:** 05.05, 09.01.

**Explanation:** Local layers save work; global layers restore direct distant exchange. Total cost depends on the mix. Having local layers does not make every length-dependent term linear.

**Worked example:** In 4 layers with 3 local and 1 global, costs include local TW terms and a global T² term.

**Ask this, then wait:** Which term may dominate at sufficiently long context?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The global quadratic term, even with fewer such layers.

**Repair:** Sum the layers before reasoning from the majority type.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-06"></a>
### 07.06 Linear attention and recurrence

**Goal:** Express history as sufficient statistics.
**Review if needed:** 05.05, 09.01.

**Explanation:** One kernelized form keeps S=Σφ(k)vᵀ and z=Σφ(k), then reads with φ(q)ᵀS and normalizes. Statistics can update recursively; this is not an identity for ordinary softmax attention.

**Worked example:** A new position adds φ(k)vᵀ to S rather than retaining every old value separately. State size depends on feature dimensions.

**Ask this, then wait:** Does a fixed-size summary necessarily preserve every historical detail?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Retrieval and expressive capacity can differ.

**Repair:** Identify the changed operator rather than calling it exact softmax rearrangement.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-07"></a>
### 07.07 Mamba and state-space models

**Goal:** Understand recurrent state.
**Review if needed:** 05.05, 09.01.

**Explanation:** A state-space model can update ht=At ht−1+Bt xt and generate outputs from state. Mamba adds input-dependent selection and related design; training can exploit parallel algorithms. Fixed state trades memory efficiency against representation limits.

**Worked example:** With ht=0.5ht−1+xt, h0=0 and inputs 2 then 0 produce states 2 then 1.

**Ask this, then wait:** Why is state nonzero after the input becomes zero?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It retains a decaying influence from the earlier input.

**Repair:** Calculate two steps before distinguishing forgetting from immediate clearing.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-08"></a>
### 07.08 Gated DeltaNet

**Goal:** Separate forgetting from targeted rewriting.
**Review if needed:** 05.05, 09.01.

**Explanation:** Gated DeltaNet combines a forgetting gate with delta updates. Forgetting scales prior state; delta writes according to target value minus current prediction rather than blindly accumulating values. Strict projection interpretations need normalization and other conditions.

**Worked example:** A simplified stored prediction 3, target 5, and write rate 0.5 give error 2 and an update toward 4; forgetting may first weaken old state.

**Ask this, then wait:** Why compare the old prediction with the new target before writing?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The error directs a correction rather than uncontrolled accumulation.

**Repair:** Start with one scalar correction, then return to matrix state without inventing implementation constants.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

Use S:dv×dk, column vectors k/q:dk and v:dv. One gated-delta form is Snew=αSold+β(v−αSold k)kᵀ, read as Snew q. α forgets old state; β controls an error-directed write. Check the paper for key normalization, gates, and parallel algorithms. An arbitrary I−βkkᵀ is not automatically an orthogonal projection.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py) · [Gated Delta Networks paper](https://arxiv.org/abs/2412.06464)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-09"></a>
### 07.09 Hybrid architectures

**Goal:** Understand combined layer types.
**Review if needed:** 05.05, 09.01.

**Explanation:** Hybrids mix attention, state-space, or other recurrent layers to trade off retrieval, local computation, and long-context cost. Benefits do not automatically combine without loss; layer ratios and training need validation.

**Worked example:** Adding attention periodically offers direct history access but also introduces KV caching for those layers.

**Ask this, then wait:** Can every hybrid be called cache-free?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Its attention components may need KV caches.

**Repair:** Account for state layer by layer.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-10"></a>
### 07.10 Indexers and dynamic sparse attention

**Goal:** Count candidate-selection work.
**Review if needed:** 05.05, 09.01.

**Explanation:** Dynamic sparse attention first uses a cheaper indexer to select positions, then applies more detailed attention there. Indexing itself costs compute and memory. Missed relevant positions can hurt quality; distinguish specific algorithms.

**Worked example:** Choose 20 candidates from 1000 positions. Fine attention cannot directly repair information omitted from the candidate set.

**Ask this, then wait:** Does counting only 20 fine comparisons capture total cost?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Include indexing and selection.

**Repair:** Budget candidate finding separately from detailed attention.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p07-11"></a>
### 07.11 Quality, memory, and speed tradeoffs

**Goal:** Make structural comparisons testable.
**Review if needed:** 05.05, 09.01.

**Explanation:** Control quality, context length, hardware, batch, precision, and latency/throughput goals. Smaller caches do not guarantee faster end-to-end execution; fewer FLOPs can lose to irregular accesses.

**Worked example:** A halves cache but loses quality; B preserves quality and is faster only for long inputs. The target conditions determine the choice.

**Ask this, then wait:** How would you make “A is faster” a testable claim?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Specify hardware, lengths, batch, precision, quality constraints, and timing.

**Repair:** Complete the experimental conditions and mark unmeasured ranges.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L03](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%203%20-%20architecture.pdf) · [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L03](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_03.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c08"></a>
## 08 · Mixture of experts

Separate total from active parameters.

<a id="p08-01"></a>
### 08.01 Dense and sparse computation

**Goal:** Separate total from active parameters.
**Review if needed:** 05.07, 04.03.

**Explanation:** A dense FFN uses the same parameters for every token. MoE routes tokens through selected experts. Total parameters affect storage; active parameters better reflect some per-token work, but routing, shared layers, and communication also cost resources.

**Worked example:** 8 experts hold 800 parameter units; activating 2 accesses 200 expert units per token, plus shared components.

**Ask this, then wait:** If expert count doubles but top-2 remains fixed, must expert compute double?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Active count is unchanged; storage and communication can change.

**Repair:** Maintain separate total and active budgets.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-02"></a>
### 08.02 Routers, top-k, and expert selection

**Goal:** Walk through a routed output.
**Review if needed:** 05.07, 04.03.

**Explanation:** The router scores experts, selects top-k, and combines selected outputs with weights. Score-to-weight transformations differ by model; not all routing uses the same softmax.

**Worked example:** Scores [0.7,0.2,0.1] select the first two. If selected weights are normalized, they become 7/9 and 2/9.

**Ask this, then wait:** Does the third expert contribute in this simplified top-2 case?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Perform selection first, then compute selected weights.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-03"></a>
### 08.03 Token-choice and expert-choice

**Goal:** Compare who selects whom.
**Review if needed:** 05.07, 04.03.

**Explanation:** Token-choice gives each token expert choices and can create imbalance. Expert-choice selects tokens up to each expert's capacity; token coverage may vary. Do not assume both balanced experts and fixed k per token without extra constraints.

**Worked example:** Two tokens both favor expert A. Token-choice can overload it; expert-choice with 1 slot per expert changes allocations.

**Ask this, then wait:** Does expert-choice alone ensure exactly two experts per token?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No; extra constraints would be needed.

**Repair:** Count selections across token rows and expert columns separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-04"></a>
### 08.04 Fine-grained and shared experts

**Goal:** Separate shared work from routed work.
**Review if needed:** 05.07, 04.03.

**Explanation:** Fine-grained designs split experts into smaller units with different active combinations. Shared experts run for all tokens; routed experts handle differences. Shared-branch work must always be included.

**Worked example:** One shared expert plus 2 selected routed experts executes 3 expert branches, not only top-2.

**Ask this, then wait:** Can the shared branch be skipped because the router did not select it?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Not under this shared-expert design.

**Repair:** Draw the shared branch outside the selection box.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-05"></a>
### 08.05 Routing noise and differentiable training

**Goal:** Explain gradients around discrete selection.
**Review if needed:** 05.07, 04.03.

**Explanation:** Top-k selection is not differentiable everywhere, but selected continuous weights and outputs still carry gradients. Noise can encourage exploration and reduce concentration. Approximations and auxiliary objectives vary; a discrete ranking has no ordinary derivative everywhere.

**Worked example:** Small noise can switch nearly tied experts and expose more experts to training; this does not guarantee balance.

**Ask this, then wait:** Does adding noise remove the need for all other load control?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Measure loads and stability.

**Repair:** Distinguish exploration from a strict capacity guarantee.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-06"></a>
### 08.06 Auxiliary load-balancing loss

**Goal:** Understand a second training objective.
**Review if needed:** 05.07, 04.03.

**Explanation:** Task loss can concentrate tokens on few experts. An auxiliary term uses assignment/probability statistics to encourage balance, reducing congestion and idleness. Overweighting it may hurt task quality; batch balance is not uniform use by every token.

**Worked example:** Loads [90,5,3,2] over 4 experts motivate balancing, not activating all 4 for every token.

**Ask this, then wait:** Does low balancing loss prove best language quality?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Check the main loss and efficiency too.

**Repair:** Separate task and system objectives in the plots.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-07"></a>
### 08.07 Bias adjustment without an auxiliary loss

**Goal:** Recognize feedback outside the task loss.
**Review if needed:** 05.07, 04.03.

**Explanation:** Some methods lower selection biases for overloaded experts and raise them for underused experts. This changes choices without adding the same auxiliary loss. “Auxiliary-loss-free” does not mean “no balancing mechanism.”

**Worked example:** Persistently overloaded expert A receives a negative score adjustment, reducing selection when other conditions match.

**Ask this, then wait:** Does omitting balancing loss imply uncontrolled loads?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Bias feedback can control them.

**Repair:** Separate learned gradients from load-driven control updates.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-08"></a>
### 08.08 Capacity, communication, and expert parallelism

**Goal:** Include dispatch constraints.
**Review if needed:** 05.07, 04.03.

**Explanation:** Capacity limits tokens per expert. Overflow can be dropped, rerouted, or handled by dropless approaches. Distributed experts require dispatch and return traffic, often all-to-all; sparse FFNs do not guarantee little communication.

**Worked example:** If 100 tokens target an expert with capacity 60, the remaining 40 need an explicit treatment.

**Ask this, then wait:** Can adding experts still leave the system slow on a weak network?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Yes. Dispatch and imbalance can dominate.

**Repair:** Map each token to a device before counting compute savings.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-09"></a>
### 08.09 Routing stability

**Goal:** Separate isolated switches from systemic instability.
**Review if needed:** 05.07, 04.03.

**Explanation:** Routing can change across training or precision choices. Switches affect gradients, loads, and recomputed probabilities in RL. Inspect distributions, overflow, and numerical sensitivity; stability does not mean permanent assignments.

**Worked example:** Scores 0.5001 and 0.5000 may tie or reorder after rounding. This illustrates sensitivity, not measured frequency.

**Ask this, then wait:** Does one expert switch prove failure?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Examine systematic changes and quality effects.

**Repair:** Separate a single event from aggregate evidence.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-10"></a>
### 08.10 Upcycling

**Goal:** Understand dense-to-MoE initialization.
**Review if needed:** 05.07, 04.03.

**Explanation:** Upcycling copies or transforms pretrained dense FFNs into experts, then continues training to develop specialization. Reuse saves an initial training stage, but copied experts can begin identical; routing and adaptation are still needed.

**Worked example:** Copying one FFN into 4 identical experts initially need not change output when routing changes.

**Ask this, then wait:** Does fourfold expert storage immediately add fourfold new knowledge?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Initial contents can be redundant.

**Repair:** Distinguish four identical books from four different books.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p08-11"></a>
### 08.11 DeepSeek cases and limits of evidence

**Goal:** Read public recipes conditionally.
**Review if needed:** 05.07, 04.03.

**Explanation:** Reports combine fine-grained/shared experts, balancing, and systems choices. DeepSeek-V3 selected routing weights involve sigmoid scores and normalization, not a universal MoE softmax rule. Whole-system gains do not isolate each component.

**Worked example:** A successful combination does not show that copying only its shared-expert choice yields the same gain.

**Ask this, then wait:** How would you identify one component's contribution?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Use ablations or direct evidence with comparable data and budgets.

**Repair:** Retain unpublished-data and unreplicated-training limits.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L04](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%204%20-%20details%20%2B%20MoEs.pdf) · [2025 L04](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/98455ec198c9a88ec1ab2b1c4058662431b54ce3/nonexecutable/2025%20Lecture%204%20-%20MoEs.pdf) · [2026 L04](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_04.pdf) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c09"></a>
## 09 · Compute, memory, and performance

Identify where resources are spent.

<a id="p09-01"></a>
### 09.01 Parameters and FLOPs

**Goal:** Separate capacity from operation count.
**Review if needed:** 00.04, 03.02.

**Explanation:** Parameters count independent learnable values; FLOPs count floating-point operations. A parameter can be reused at many tokens. Counting multiply-add as 2 gives approximately 2mkn for m×k times k×n. Sharing storage does not remove repeated use.

**Worked example:** 2×3 times 3×4 has 8 outputs, each summing 3 products; conventionally estimate 2×2×3×4=48 FLOPs.

**Ask this, then wait:** With fixed parameters, does doubling input tokens leave computation unchanged?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Usually not; weights are used at more positions.

**Repair:** Count stored entries and executions separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-02"></a>
### 09.02 Forward and backward costs

**Goal:** Include the extra gradient products.
**Review if needed:** 00.04, 03.02.

**Explanation:** For Y=XW, backward computes gradients for X and W, often requiring two similar matrix products. A rough training estimate is 1 forward plus 2 backward units, not a universal operator identity.

**Worked example:** Forward cost 100 and two gradient costs near 100 each total about 300, before optimizer and recomputation costs.

**Ask this, then wait:** Does “backward costs twice forward” mean total training costs twice forward?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Include forward for roughly threefold total.

**Repair:** Draw the forward path and both gradient paths before summing.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-03"></a>
### 09.03 Estimating training compute

**Goal:** Use 6ND with assumptions.
**Review if needed:** 00.04, 03.02.

**Explanation:** C≈6ND is a common dense-Transformer training estimate, with major parameter count N and training tokens D. It assumes dominant dense products; long-context attention, MoE activity, recomputation, and system overhead need separate treatment.

**Worked example:** N=10⁶,D=10⁸ gives about 6×10¹⁴ FLOPs, not seconds.

**Ask this, then wait:** Does dividing by peak FLOPs/s guarantee wall-clock training time?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Utilization, communication, and IO add limitations.

**Repair:** Label idealized estimates, then use measured throughput.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-04"></a>
### 09.04 Parameters, gradients, optimizer state, and activations

**Goal:** Budget training memory by component.
**Review if needed:** 00.04, 03.02.

**Explanation:** Training memory includes weights, gradients, optimizer states, activations, and temporary workspace. Precision and sharding alter each. Inference has different state and KV caches. Weight-file size alone cannot establish training feasibility.

**Worked example:** 1000000 FP32 weights use about 4MB; gradients add 4MB and two Adam moments add 8MB: 16MB before activations.

**Ask this, then wait:** Do 4MB weights imply training fits in 4MB?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Other state is required.

**Repair:** Sum components and state the approximate MB convention.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-05"></a>
### 09.05 FP32, FP16, BF16, FP8, and FP4

**Goal:** Separate bit width from numerical behavior.
**Review if needed:** 00.04, 03.02.

**Explanation:** Formats allocate exponent and significand bits differently. BF16 and FP16 both use 16 bits but differ in range and precision. FP8/FP4 have multiple formats and scaling schemes; a name alone does not specify exact range.

**Worked example:** FP16 and BF16 use 2 bytes each; BF16 generally has wider exponent range and less significand precision.

**Ask this, then wait:** Does equal bit width imply equal largest representable value?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Bit allocation differs.

**Repair:** Distinguish how many bits exist from how they are assigned.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-06"></a>
### 09.06 Mixed precision

**Goal:** Assign different numerical roles.
**Review if needed:** 00.04, 03.02.

**Explanation:** Large products can use low precision while accumulation or states use higher precision. FP16 often uses loss scaling against underflow. Unscale before clipping and handle nonfinite gradients. BF16 is not a universal stability guarantee.

**Worked example:** Multiplying loss by S multiplies gradients by S; divide before updating to recover the intended scale.

**Ask this, then wait:** Is loss scaling intended to change the optimum?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Correct unscaling preserves the target gradient.

**Repair:** Track where S enters and is removed.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-07"></a>
### 09.07 Arithmetic intensity

**Goal:** Compute work per transferred byte.
**Review if needed:** 00.04, 03.02.

**Explanation:** I=FLOPs/bytes transferred at a specified memory level. Reuse can raise I. State which traffic is counted; the same mathematical algorithm can have different intensities in different implementations.

**Worked example:** 1000 FLOPs over 100 bytes gives I=10; reducing traffic to 50 bytes raises it to 20.

**Ask this, then wait:** What happens to I if work is fixed and traffic falls?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It increases.

**Repair:** Do not confuse FLOPs/second with FLOPs/byte.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-08"></a>
### 09.08 Bandwidth and compute bottlenecks

**Goal:** Estimate the limiting resource.
**Review if needed:** 00.04, 03.02.

**Explanation:** Compute limits arithmetic throughput; bandwidth limits data supply. Launch, synchronization, and latency can also dominate. Do not force every slowdown into only the first two categories.

**Worked example:** 100 FLOPs and 100 bytes on 1000 FLOPs/s and 10 bytes/s require at least 0.1s arithmetic versus 10s transfer.

**Ask this, then wait:** Would doubling compute nearly double speed here?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. The roughly 10s bandwidth bound remains.

**Repair:** Estimate arithmetic and transfer times separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-09"></a>
### 09.09 The roofline model

**Goal:** Read an ideal performance bound.
**Review if needed:** 00.04, 03.02.

**Explanation:** Performance≤min(peak compute,bandwidth×intensity). Low-intensity performance is bandwidth-bounded; high intensity reaches the compute ceiling. Actual performance may lie below both due to other costs.

**Worked example:** Peak 1000 FLOPs/s and 10 bytes/s give a bound of 200 at I=20, or 1000 at I=200.

**Ask this, then wait:** Does being below the roofline prove a bug?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It is an upper bound, not a guarantee.

**Repair:** Identify omitted launch, access, or synchronization limits.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-10"></a>
### 09.10 Model FLOPs utilization

**Goal:** Define MFU's numerator and denominator.
**Review if needed:** 00.04, 03.02.

**Explanation:** MFU usually divides useful model FLOPs/s by aggregate hardware peak. State FLOP estimates, precision, and card count. It differs from device-busy percentage; recomputation may not count as useful model work.

**Worked example:** 4 cards at peak 100 each and useful throughput 120 give 120/400=30% MFU.

**Ask this, then wait:** Does high device utilization guarantee high MFU?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Time may go to transfers, recomputation, or inefficient work.

**Repair:** Write the units and formula rather than reusing a dashboard percentage.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p09-11"></a>
### 09.11 Recomputation and accumulation costs

**Goal:** Evaluate memory-saving tradeoffs.
**Review if needed:** 00.04, 03.02.

**Explanation:** Recomputation stores fewer activations and repeats forward work during backward. Accumulation processes several smaller batches before updating. Both can lower peak memory but add compute or reduce parallel efficiency.

**Worked example:** Two sequential microbatches can emulate a larger effective batch without holding both at once; recomputation instead trades more FLOPs for storage.

**Ask this, then wait:** Does halving saved activations guarantee twice the training speed?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Recomputing can slow execution.

**Repair:** Compare peak memory, step time, and quality separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L02](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_02.py) · [2025 L02](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_02.py) · [2026 L02](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_02.py) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c10"></a>
## 10 · GPUs, TPUs, and hardware execution

Build performance intuition.

<a id="p10-01"></a>
### 10.01 CPUs and GPUs

**Goal:** Understand different execution strengths.
**Review if needed:** 09.07, 09.08.

**Explanation:** CPUs handle complex control and low-latency work well; GPUs exploit large regular parallel workloads. Transfers and launches cost time, so small scattered tasks may not benefit. CPUs can still perform matrix multiplication.

**Worked example:** Adding two numbers may cost much more in GPU launch overhead than arithmetic; large products better amortize overhead.

**Ask this, then wait:** Why can a higher-peak GPU lose on a tiny task?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Fixed costs and insufficient parallel work dominate.

**Repair:** Count available parallelism before quoting peak throughput.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-02"></a>
### 10.02 SMs, warps, threads, and blocks

**Goal:** Distinguish execution levels.
**Review if needed:** 09.07, 09.08.

**Explanation:** CUDA threads form blocks scheduled on SMs; a warp commonly contains 32 threads. Blocks cooperate through shared memory, with different rules across blocks. Resource counts depend on architecture.

**Worked example:** A 128-thread block commonly contains 4 warps. Several blocks may reside on one SM subject to resource limits.

**Ask this, then wait:** Does a 128-thread block imply 128 SMs?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. These are different levels.

**Repair:** Map GPU, SM, scheduled blocks, warps, and threads.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-03"></a>
### 10.03 SIMT and branch divergence

**Goal:** Explain lost parallel efficiency.
**Review if needed:** 09.07, 09.08.

**Explanation:** SIMT groups execute a shared instruction stream. Different branches within a warp can execute in stages with inactive lanes masked. They do not necessarily become fully independent CPU-like paths.

**Worked example:** Half the lanes take long branch A and half B; both paths may run with part of the warp inactive.

**Ask this, then wait:** Can every divergent path be assumed fully simultaneous?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Divergence can serialize work.

**Repair:** Illustrate masked groups without claiming every compiler handles branches identically.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-04"></a>
### 10.04 Registers, shared memory, caches, and HBM

**Goal:** Understand the memory hierarchy.
**Review if needed:** 09.07, 09.08.

**Explanation:** Registers are close to threads; shared memory supports block cooperation; caches reuse some data automatically; HBM is larger but costly to access. Capacity, sharing, latency, and bandwidth differ. Excess registers can reduce residency or spill.

**Worked example:** Reading a reused value from HBM 10 times transfers more than loading once and reusing it on-chip.

**Ask this, then wait:** Why not put every value in registers?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Capacity and residency limits; spills can use slower storage.

**Repair:** List speed, capacity, and occupancy constraints together.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-05"></a>
### 10.05 Tensor Cores

**Goal:** Identify specialized matrix hardware.
**Review if needed:** 09.07, 09.08.

**Explanation:** Tensor Cores accelerate supported matrix multiply-accumulate shapes and precisions. Layout, tile, and dtype conditions matter. Arbitrary GPU arithmetic does not automatically achieve Tensor Core peak.

**Worked example:** A large regular product can exploit matrix units; scattered scalar additions do not correspond to the same peak.

**Ask this, then wait:** Can matrix-unit peak predict every kernel's speed?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Work and format must match the hardware.

**Repair:** Identify the unit actually executing the kernel.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-06"></a>
### 10.06 TPUs and matrix computation

**Goal:** Build a dataflow intuition.
**Review if needed:** 09.07, 09.08.

**Explanation:** TPUs target tensor computation with matrix units and compiler-managed execution. Systolic arrays pass data through units to reuse multiply-add inputs. Programming, networking, and precision support differ from GPUs and between generations.

**Worked example:** A matrix block flows through an array; tiny or irregular blocks can leave units underfilled.

**Ask this, then wait:** Does specialized hardware remove data-movement limits?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Supply, layout, and bandwidth still matter.

**Repair:** Draw input, array, and output flows.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-07"></a>
### 10.07 Low-precision hardware

**Goal:** Separate storage from arithmetic support.
**Review if needed:** 09.07, 09.08.

**Explanation:** Benefits depend on storage, transfers, multiplication, accumulation, scales, and dequantization. A storable format is not necessarily efficient native arithmetic. Numerical and task errors need evaluation.

**Worked example:** 4-bit weights expanded to 16-bit before multiplication save storage without necessarily using 4-bit arithmetic peak.

**Ask this, then wait:** Does a smaller file establish a fourfold inference speedup?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Inspect the actual kernel and bottleneck.

**Repair:** Trace disk, device-storage, and compute formats independently.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-08"></a>
### 10.08 Unequal compute and bandwidth growth

**Goal:** Recognize changing bottlenecks.
**Review if needed:** 09.07, 09.08.

**Explanation:** Compute, bandwidth, and memory capacity can grow at different rates. A task can become supply-bound as arithmetic improves. Cross-generation comparisons also need software, cost, and power; current specifications require vendor checks.

**Worked example:** If compute grows 4-fold and bandwidth only 2-fold, a bandwidth-bound workload may ideally gain only about 2-fold.

**Ask this, then wait:** Why not infer end-to-end speed from new-card FLOPs alone?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Bandwidth, communication, or other costs may dominate.

**Repair:** Compare rooflines at the workload's intensity.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-09"></a>
### 10.09 Coalesced memory access

**Goal:** Map thread indices to addresses.
**Review if needed:** 09.07, 09.08.

**Explanation:** Neighboring threads reading neighboring addresses can combine traffic into fewer memory transactions. Strided or scattered reads can require more transactions despite identical logical element counts.

**Worked example:** 32 threads reading 32 adjacent values generally coalesce better than each reading a distant value; exact transactions depend on architecture and alignment.

**Ask this, then wait:** Must two kernels reading 32 values have equal bandwidth efficiency?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Addresses and alignment matter.

**Repair:** Draw thread-to-address mappings first.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p10-10"></a>
### 10.10 Tiling, alignment, and wave quantization

**Goal:** Separate edge waste from scheduling waste.
**Review if needed:** 09.07, 09.08.

**Explanation:** Tiling improves reuse, with masks or padding at edges. Alignment affects hardware utilization. Wave quantization occurs when block count does not fill all concurrent slots in the final wave.

**Worked example:** With 4 equal-work slots, 5 blocks need two waves with only 1 active slot in the last; 8 blocks also need two full waves.

**Ask this, then wait:** Why need time not grow in proportion from 5 to 8 blocks?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The wave count can stay fixed while idle slots fill, under the stated assumptions.

**Repair:** Separate tile-edge inefficiency from incomplete scheduling waves.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c11"></a>
## 11 · Profiling, compilation, and GPU kernels

Locate and improve slow operations.

<a id="p11-01"></a>
### 11.01 Reliable timing, warmup, and synchronization

**Goal:** Measure actual execution.
**Review if needed:** 10.04, 09.09.

**Explanation:** GPU submission is often asynchronous. Time completed work using synchronization or device events; warm up and repeat measurements. Separate compilation/cold-start costs from steady state without hiding first-request latency.

**Worked example:** A hypothetical 0.1ms Python return may only submit GPU work that takes 10ms.

**Ask this, then wait:** Why confirm completion before ending a timer?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Otherwise only submission overhead may be measured.

**Repair:** Draw separate CPU and GPU timelines.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-02"></a>
### 11.02 End-to-end benchmarks and profiling

**Goal:** Optimize an actual hotspot.
**Review if needed:** 10.04, 09.09.

**Explanation:** Measure representative end-to-end work, then locate compute, communication, waits, and memory costs. Amdahl's law limits the effect of accelerating a small component.

**Worked example:** If a kernel occupies 10% of 100 time units, eliminating it entirely leaves 90: about 1.11-fold speedup.

**Ask this, then wait:** Why might a tenfold kernel gain barely change training time?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Its time share may be small; other costs remain.

**Repair:** Ask for time share before local speedup.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-03"></a>
### 11.03 Nsight, flame graphs, and memory profiling

**Goal:** Read each profiling view correctly.
**Review if needed:** 10.04, 09.09.

**Explanation:** Timelines show CPU/GPU activity and overlap; kernel analysis examines execution and memory; flame graphs represent sampled stacks; memory views show allocation and peaks. A wide bar need not be pure computation.

**Worked example:** An idle GPU while CPU prepares inputs suggests supply delay. Overlapping temporary tensors may explain a memory peak.

**Ask this, then wait:** Does a GPU gap prove its arithmetic kernel is too slow?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It may await CPU work, data, or synchronization.

**Repair:** Locate the wait cause before choosing a tool.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-04"></a>
### 11.04 Operator fusion

**Goal:** Explain saved launches and traffic.
**Review if needed:** 10.04, 09.09.

**Explanation:** Fusion combines operations within a kernel, reducing launches and intermediate HBM traffic. It can increase registers or restrict scheduling. FLOPs need not fall; floating-point order may change.

**Worked example:** Bias addition followed by GELU can keep the intermediate on-chip instead of writing it back.

**Ask this, then wait:** Does faster fusion prove fewer arithmetic operations?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Reduced movement or launches may explain it.

**Repair:** Count FLOPs, bytes, and launches separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-05"></a>
### 11.05 CUDA, PTX, and Triton

**Goal:** Separate programming and execution layers.
**Review if needed:** 10.04, 09.09.

**Explanation:** CUDA names a platform and language ecosystem; PTX is a virtual instruction layer; Triton expresses GPU work through higher-level block operations. Compilation still leads to machine instructions and hardware constraints.

**Worked example:** Row reduction can use CUDA threads or Triton block expressions; both must handle layout and resources.

**Ask this, then wait:** Does Triton remove the need to reason about memory access?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Draw source→compiled representations→execution rather than equating tool names with algorithms.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-06"></a>
### 11.06 torch.compile

**Goal:** Understand conditional compilation benefits.
**Review if needed:** 10.04, 09.09.

**Explanation:** Compilation can capture and optimize computation, reducing Python work or fusing operations. Graph breaks, dynamic shapes, and compilation cost affect benefits. Version-specific support must be checked; no universal speedup is promised.

**Worked example:** The first run compiles; repeated shapes may reuse work, while changing shapes can trigger recompilation.

**Ask this, then wait:** Does timing only a second run represent a cold online request?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Report cold/compilation and steady-state costs separately.

**Repair:** Choose the timing window to match the usage scenario.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-07"></a>
### 11.07 GELU, softmax, row sums, and matrix products

**Goal:** Recognize different computational patterns.
**Review if needed:** 10.04, 09.09.

**Explanation:** GELU is largely elementwise; softmax and row sums reduce across values; matrix multiplication combines reduction with reuse. One elementwise optimization template does not solve all patterns.

**Worked example:** Softmax takes a row maximum, exponentiates, sums, and normalizes; a GELU output can be computed independently.

**Ask this, then wait:** Why does one softmax output depend on other row entries?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Its denominator is the row's exponential sum.

**Repair:** Draw elementwise arrows separately from reductions.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-08"></a>
### 11.08 Tiles and masks

**Goal:** Handle incomplete blocks correctly.
**Review if needed:** 10.04, 09.09.

**Explanation:** The last tile may be partially filled. Masks protect out-of-range reads/writes; reductions need appropriate fill values. Sum often uses 0; maximum often uses negative infinity, not always 0.

**Worked example:** Length 5 with tiles of 4 leaves one valid entry in the last tile. Zero fill can incorrectly win a maximum over negative inputs.

**Ask this, then wait:** Why is zero unsafe as a universal maximum fill?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** All valid entries might be negative.

**Repair:** Identify the operation's neutral or safe fill value first.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-09"></a>
### 11.09 Custom autograd

**Goal:** Validate backward as well as forward.
**Review if needed:** 10.04, 09.09.

**Explanation:** A custom operation needs correct gradients and saved intermediates or recomputation. Correct outputs do not prove correct derivatives. Compare small gradients with a reference or finite differences, respecting precision and nondifferentiable points.

**Worked example:** A correct forward x² with incorrect backward x rather than 2x trains incorrectly.

**Ask this, then wait:** Can output-only comparisons validate a training kernel?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Check derivatives and boundary inputs.

**Repair:** Differentiate a scalar example before generalizing.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-10"></a>
### 11.10 RMSNorm forward and backward

**Goal:** Account for the shared denominator.
**Review if needed:** 10.04, 09.09.

**Explanation:** Let r=√(mean x²+ε), yᵢ=gᵢxᵢ/r. Every output depends on the whole row through r. Backward includes both direct and denominator-change terms; treating r as constant is wrong. A full derivative is included in the extension.

**Worked example:** Increasing one coordinate increases r and can decrease other normalized coordinates.

**Ask this, then wait:** Why are feature gradients coupled despite coordinate-wise scaling?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Their shared r depends on every input coordinate.

**Repair:** Trace r in two dimensions before using the full derivative, retaining 1/D.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

For upstream aᵢ=∂L/∂yᵢ and r=√(Σxᵢ²/D+ε), ∂L/∂xⱼ=aⱼgⱼ/r−xⱼΣ(aᵢgᵢxᵢ)/(D r³), and ∂L/∂gⱼ=aⱼxⱼ/r. Shared gain gradients also sum across rows. The second input term follows from r changing with x and cannot be dropped. D is the row feature count.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p11-11"></a>
### 11.11 Correctness and speed as separate checks

**Goal:** Require valid computation before benchmarking.
**Review if needed:** 10.04, 09.09.

**Explanation:** Compare reference outputs and gradients across tolerances, edge shapes, and extreme inputs. Then time equivalent hardware, precision, layout, and inputs. A fast benchmark cannot establish correctness; a tiny passing case is not universal proof.

**Worked example:** Skipping tail values looks faster but fails lengths not divisible by tile size.

**Ask this, then wait:** Why is “fastest” insufficient for adoption?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It may omit work or violate numerical/task requirements.

**Repair:** Set acceptance conditions before timing; retain failing cases.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L06](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_06.py) · [2025 L06](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_06.py) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c12"></a>
## 12 · FlashAttention and recomputation

Understand memory-efficient exact attention.

<a id="p12-01"></a>
### 12.01 Attention's intermediate matrices

**Goal:** Locate the large stored tensors.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** A basic implementation builds S=QKᵀ, normalizes to P, then computes PV. Both S and P can be T×T while output is T×d, creating large intermediate storage.

**Worked example:** T=1000 gives 1000000 score values per head; with d=64, output has 64000.

**Ask this, then wait:** Must the largest intermediate match output size?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Pairwise position relationships can be much larger.

**Repair:** Draw the shape flow through Q,K,scores,V,output.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-02"></a>
### 12.02 HBM and on-chip memory

**Goal:** Separate IO optimization from changed mathematics.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** HBM is large but costly to move data through; on-chip memory is fast and small. FlashAttention organizes computation to avoid repeatedly storing complete score/probability tables in HBM.

**Worked example:** Read QKV in blocks and accumulate output rather than materializing the entire score table externally.

**Ask this, then wait:** Why not simply keep the whole T×T table on-chip?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Capacity is limited; tiling and mergeable statistics are needed.

**Repair:** Label both capacity and bandwidth.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-03"></a>
### 12.03 Tiling attention

**Goal:** Preserve all required interactions.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** Each query block processes all required KV blocks. Exact full attention still covers every valid pair; savings concern IO and intermediate storage, not silently deleting distant blocks.

**Worked example:** Splitting 4 positions into two blocks still requires both KV blocks for each query block, subject to causal masking.

**Ask this, then wait:** Is tiling the same as a sparse sliding window?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. One schedules work; the other removes connections.

**Repair:** Enumerate block pairs and check all allowed interactions remain.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-04"></a>
### 12.04 Online softmax

**Goal:** Keep one normalization across blocks.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** Each row's softmax denominator spans all blocks. Independently normalized blocks cannot simply be added. Track maximum, exponential sum, and weighted value sum; rescale when merging, then normalize globally.

**Worked example:** Separate blocks with scores 0 and ln3 each have local softmax 1, but full-row weights are 1/4 and 3/4.

**Ask this, then wait:** Why not concatenate independent block softmaxes?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** They sum to 2 and lose cross-block relative magnitudes.

**Repair:** Use the two-value counterexample before introducing statistics.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-05"></a>
### 12.05 Stable incremental statistics

**Goal:** Rescale accumulated values consistently.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** Keep old maximum m and exponential sum l. On a new block choose m'=max(m,block maximum), scale old sums by exp(m−m'), and add new terms relative to m'. Rescale weighted sums the same way.

**Worked example:** With old score 0 and new ln3, old l becomes 1/3, new term is 1, total 4/3; weights remain 1/4 and 3/4.

**Ask this, then wait:** Why rescale previous sums when the maximum increases?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Otherwise old and new terms use different exponential reference scales.

**Repair:** Rewrite each old exp(s−m) using the new maximum.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

Merge old statistics (m,l,u) with block scores sⱼ and values vⱼ: mnew=max(m,maxⱼsⱼ), c=exp(m−mnew), lnew=c·l+Σexp(sⱼ−mnew), unew=c·u+Σexp(sⱼ−mnew)vⱼ. Final output o=u/l. Initialize from a valid first block or safely handle m=−∞. Fully masked rows need an explicit convention rather than silent 0/0.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-06"></a>
### 12.06 Fused forward computation

**Goal:** Complete attention within a tiled flow.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** Fusion joins block scores, stable normalization statistics, and value accumulation without writing the full P to HBM. Normalize accumulated output at the end. Masks and precision still matter.

**Worked example:** Each KV block updates m,l,u; after all blocks output u/l, without keeping complete P resident.

**Ask this, then wait:** Does not saving P mean attention weights never affected computation?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Their effects are computed locally and accumulated.

**Repair:** Distinguish not retaining a tensor from not computing its effect.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-07"></a>
### 12.07 Saving necessary statistics

**Goal:** Retain what backward can use.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** Backward can keep output and row log-sum-exp statistics, then reconstruct local probabilities from QKV. Exact saved sets vary. A summary alone does not replace every training input.

**Worked example:** A row normalization statistic plus recomputed scores can recover local probabilities with much less saved storage.

**Ask this, then wait:** Can a normalization statistic alone reconstruct P?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It also needs scores, typically recomputed from QK.

**Repair:** List every required recovery input.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-08"></a>
### 12.08 Backward recomputation

**Goal:** Trade arithmetic for activation memory.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** Backward recomputes local scores/softmax using saved statistics and QKV, then gradients. Extra arithmetic can reduce expensive IO and activation storage. Reordered floating-point operations can differ slightly.

**Worked example:** Recompute and discard each local P instead of retaining the full matrix, enabling longer training sequences.

**Ask this, then wait:** How can more FLOPs still run faster?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** They can eliminate more expensive transfers in an IO-bound workload.

**Repair:** Return to the roofline rather than assuming recomputation must slow everything.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-09"></a>
### 12.09 FlashAttention-2 work partitioning

**Goal:** Identify implementation improvements.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** FlashAttention-2 changes block/warp work allocation and reduces some non-matrix arithmetic overhead. It retains exact-attention principles; gains depend on lengths, head dimensions, and hardware.

**Worked example:** More independent blocks can improve occupancy when previous partitioning exposes too little work, but not every shape benefits equally.

**Ask this, then wait:** Does FA2's speed require a new approximate attention objective?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. The main changes concern execution organization.

**Repair:** Separate mathematical operator, numerical precision, and scheduling.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p12-10"></a>
### 12.10 Exact versus approximate attention

**Goal:** Define what exact means here.
**Review if needed:** 05.03, 09.07, 10.04.

**Explanation:** Exact means the same softmax-attention expression, rather than sparse connections or low-rank replacement. Floating-point rounding and reduction order need not be bitwise identical. Linear or sparse approximations change the operator.

**Worked example:** 1.000000 versus 0.999999 may satisfy tolerance; deleting half the connections changes the definition.

**Ask this, then wait:** Must exact attention match every floating-point bit?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Usually no; validate tolerances and gradients.

**Repair:** Compare formulas before interpreting numerical differences.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L05](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%205%20-%20GPUs.pdf) · [2025 L05](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/nonexecutable/2025%20Lecture%205%20-%20GPUs.pdf) · [2026 L05](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_05.pdf) · [2026 L06](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_06.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c13"></a>
## 13 · Communication across devices

Understand collective communication first.

<a id="p13-01"></a>
### 13.01 Ranks, world size, and process groups

**Goal:** Identify communication participants.
**Review if needed:** 03.01.

**Explanation:** Rank numbers a process within a group; world size counts group members. A rank is not a hardware model or necessarily a machine number. One process may belong to several parallel groups.

**Worked example:** Two machines with 4 processes each form a group of world size 8. Local rank differs from global rank.

**Ask this, then wait:** Does rank=3 imply the fourth machine?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. The process-to-machine/device mapping determines location.

**Repair:** List machines, devices, and processes separately before numbering.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-02"></a>
### 13.02 Broadcast, reduce, and all-reduce

**Goal:** Compare replication and aggregation.
**Review if needed:** 03.01.

**Explanation:** Broadcast sends a source value to all. Reduce combines values at a chosen destination; all-reduce returns the combined result to everyone. Whether a sum is divided by world size depends on the stated operation or training semantics.

**Worked example:** Ranks holding 1,2,3 receive 6 after sum all-reduce; a mean additionally divides by 3.

**Ask this, then wait:** Is sum all-reduce automatically an average?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Separate communication from scaling to avoid extra or missing gradient division.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-03"></a>
### 13.03 All-gather, reduce-scatter, and all-to-all

**Goal:** Track the final data layout.
**Review if needed:** 03.01.

**Explanation:** All-gather collects all shards everywhere. Reduce-scatter combines values then distributes result shards. All-to-all sends different pieces to different destinations. Their layouts and uses differ.

**Worked example:** With [a,b] and [c,d] on two ranks, sum reduce-scatter can leave a+c on rank 0 and b+d on rank 1.

**Ask this, then wait:** Does each rank then hold the whole [a+c,b+d]?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. All-gather is needed to assemble it everywhere.

**Repair:** Move paper shards physically and inventory each recipient.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-04"></a>
### 13.04 Ring communication

**Goal:** Understand segmented exchange.
**Review if needed:** 03.01.

**Explanation:** A ring all-reduce can combine reduce-scatter and all-gather. Ranks forward chunks and aggregate along the ring. Large messages can use bandwidth well, but multiple rounds incur startup latency.

**Worked example:** For p ranks, an ideal ring uses about 2(p−1) rounds with chunk size near message size/p.

**Ask this, then wait:** Must a larger ring be faster for tiny messages?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. More rounds can increase latency costs.

**Repair:** Compare both round count and bytes per round.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-05"></a>
### 13.05 Latency and bandwidth

**Goal:** Separate fixed and size-dependent cost.
**Review if needed:** 03.01.

**Explanation:** A simplified transfer time is α+bytes/bandwidth. Small messages are often latency-bound; larger ones emphasize bandwidth. Multiple rounds can pay α repeatedly, and topology affects effective bandwidth.

**Worked example:** With α=1ms and 100MB/s, 1KB is mostly startup cost; 100MB takes about 1 second plus α.

**Ask this, then wait:** Does doubling bandwidth nearly halve a 1KB transfer here?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Usually not.

**Repair:** Split time into fixed and size-dependent terms.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-06"></a>
### 13.06 NCCL and torch.distributed

**Goal:** Locate library responsibilities.
**Review if needed:** 03.01.

**Explanation:** torch.distributed exposes groups and collectives; NCCL commonly provides GPU communication underneath. Participants must match operation order, shapes, dtypes, and groups. Check installed-version documentation for APIs.

**Worked example:** Rank 0 calling all-reduce while rank 1 calls all-gather can mismatch and hang.

**Ask this, then wait:** Does healthy networking guarantee no collective hangs?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Program ordering can be wrong.

**Repair:** Inspect per-rank operation logs before blaming hardware.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-07"></a>
### 13.07 Synchronous and asynchronous operations

**Goal:** Distinguish submission from completion.
**Review if needed:** 03.01.

**Explanation:** An asynchronous handle means work was submitted. Reading results or reusing buffers must respect completion and dependencies; wait or synchronization establishes the necessary order. Asynchronous does not mean immediate readiness.

**Worked example:** Independent work can follow an async reduction, but overwriting its gradient buffer too early can corrupt it.

**Ask this, then wait:** May the source tensor be freely overwritten immediately after submission?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Generally not; obey the API's completion requirements.

**Repair:** Mark buffer ownership and safe reuse times.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-08"></a>
### 13.08 Overlapping communication and computation

**Goal:** Hide work only when dependencies permit.
**Review if needed:** 03.01.

**Explanation:** Completed gradient buckets can communicate while earlier layers continue backward. Tiny buckets pay launch overhead; huge buckets delay communication. Real overlap depends on dependencies and resource contention.

**Worked example:** 10ms computation plus 4ms communication can ideally take about 10ms if fully hidden, versus 14ms serially.

**Ask this, then wait:** Does an async API prove the 4ms was hidden?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Check the timeline.

**Repair:** Distinguish concurrent submission from overlapping execution.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-09"></a>
### 13.09 Intra-node and inter-node links

**Goal:** Place communication on actual links.
**Review if needed:** 03.01.

**Explanation:** Within-node GPU links and between-node networks often differ in bandwidth and latency. Frequent large communications should account for placement. Inspect actual topology rather than assuming uniform full connectivity.

**Worked example:** Tensor parallelism crossing a slow network may lose to groups kept within a node.

**Ask this, then wait:** Can identical card counts run at different speeds due to placement?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Yes. Paths and link contention differ.

**Repair:** Draw the device network and label links.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p13-10"></a>
### 13.10 Topology and TPU/GPU networks

**Goal:** Relate communication algorithms to connectivity.
**Review if needed:** 03.01.

**Explanation:** GPU clusters may use switches and GPU links; TPUs have generation-specific interconnects. Rings, trees, and meshes alter hops and congestion. No single product generation represents every device.

**Worked example:** Four nodes connected only in a ring require multi-hop distant traffic; assuming full connectivity underestimates paths.

**Ask this, then wait:** Does equal link bandwidth guarantee equal all-reduce time across topologies?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Algorithms, paths, and contention matter.

**Repair:** Measure end-to-end effective communication, not one link's specification.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c14"></a>
## 14 · Distributed training

Identify what each strategy partitions.

<a id="p14-01"></a>
### 14.01 Distributed data parallelism

**Goal:** Identify what DDP replicates.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Classic DDP keeps a full model per rank, processes different examples, combines gradients, then updates replicas. Consistent initial weights, optimizer settings, and update order maintain agreement. Avoid unintended duplicated data.

**Worked example:** Two cards each process 4 examples, giving an effective batch of 8, while each still holds the full unsharded model state.

**Ask this, then wait:** Does DDP automatically halve parameters stored per card?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It splits data, not necessarily model state.

**Repair:** Draw data partitioning separately from parameter sharding.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-02"></a>
### 14.02 Gradient averaging and buckets

**Goal:** Match distributed gradients to the objective.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Gradients need weighting by actual examples or valid tokens. Buckets trade startup overhead against overlap. Averaging local means is wrong for a global token mean when local valid-token counts differ.

**Worked example:** Ranks with 2 and 6 valid tokens and mean gradients 1 and 3 have global mean 2.5, not 2.

**Ask this, then wait:** Does dividing by world size fix unequal token weighting automatically?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Loss scaling or a global denominator must be correct.

**Repair:** Define the global objective before deriving rank contributions.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-03"></a>
### 14.03 ZeRO stages and FSDP

**Goal:** Track progressively sharded state.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** ZeRO-1 shards optimizer state; stage 2 also shards gradients; stage 3 also shards parameters. FSDP is a full-sharding approach with implementation-specific prefetch and APIs. Computation can temporarily gather parameters.

**Worked example:** Four-way parameter sharding does not imply peak memory is always one quarter of total parameters.

**Ask this, then wait:** Can static stage-3 shard size determine peak memory alone?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Include temporary gathers, activations, and communication buffers.

**Repair:** Separate persistent, temporary, and peak allocations.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-04"></a>
### 14.04 Sharding optimizer state, gradients, and parameters

**Goal:** Follow a shard's lifecycle.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Optimizer state belongs to parameter shards; reduce-scatter can deliver gradient shards to their owners. Updated parameters are gathered or synchronized as needed. Storage savings add communication and scheduling constraints.

**Worked example:** Rank 0 updates the first half using its moments; rank 1 updates the second half. Layers then obtain needed weights.

**Ask this, then wait:** Can a rank lacking complete moments independently update the entire model?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Shard owners cooperate.

**Repair:** For every shard, record owner, computation, and exchange time.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-05"></a>
### 14.05 Row and column tensor parallelism

**Goal:** Explain summation versus concatenation.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** For Y=XW, splitting W's output columns produces separate output features. Splitting the contracted input axis with X produces partial sums requiring reduction. Efficient layer combinations can avoid some intermediate communication.

**Worked example:** Column shards give left/right Y pieces to concatenate. Inner-axis shards give Y₁+Y₂, not concatenation.

**Ask this, then wait:** Why do the two splits combine differently?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** One splits a retained output axis; the other splits a summed axis.

**Repair:** Mark shard ownership in the matrix-product summation.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-06"></a>
### 14.06 Pipelines, microbatches, and bubbles

**Goal:** Identify idle pipeline stages.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Pipeline parallelism assigns layers to devices and streams microbatches. Filling and draining create bubbles. More microbatches can improve occupancy but affect activation memory, batch size, and scheduling.

**Worked example:** In stages A,B, the first slot runs only A; later A handles batch 2 while B handles batch 1. The final slot leaves A idle.

**Ask this, then wait:** Why can more stages fail to accelerate training?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Bubbles, transfer, imbalance, or too few microbatches can dominate.

**Repair:** Draw a time grid rather than simply dividing time by devices.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

For an ideal balanced GPipe-style fill/drain schedule with p stages and m microbatches, ignoring communication and using equal forward/backward stage times, bubble fraction is approximately (p−1)/(m+p−1). For p=4,m=1 it is 3/4; with m=16 it is 3/19. This is not a universal measurement formula for 1F1B or heterogeneous stages; draw their actual schedules.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-07"></a>
### 14.07 1F1B and zero-bubble scheduling

**Goal:** Understand scheduling conditions.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** 1F1B alternates forward and backward after warmup to control saved activations. Zero-bubble approaches can split input-gradient and weight-gradient work to fill gaps. Near-zero bubbles require memory, balance, and scheduling conditions.

**Worked example:** A stage waiting for inputs may execute nonblocking weight-gradient work; this is a scheduling intuition.

**Ask this, then wait:** Does “zero-bubble” guarantee no idle time for any model or device count?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** List stage timings and memory constraints before claiming a gap can be filled.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-08"></a>
### 14.08 Sequence and context parallelism

**Goal:** Check dependencies across token shards.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Sequence parallelism often shards some token-wise activations; context parallelism distributes one long sequence and exchanges attention information. Names vary. Splitting a sequence does not remove cross-shard dependencies.

**Worked example:** If early tokens are on A and later tokens on B, B's full causal attention can still need A's KV.

**Ask this, then wait:** Can full attention usually run independently after splitting the token axis?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Visibility crosses the boundary.

**Repair:** Overlay the causal triangle on the partition.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-09"></a>
### 14.09 Expert parallelism

**Goal:** Place routed experts across devices.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Expert parallelism distributes experts, dispatches token representations according to routing, and returns combined results. Loads, all-to-all, and shared-expert placement affect efficiency. This differs from full-model data replication.

**Worked example:** A token originating on A may need an expert on B, requiring outward and return traffic.

**Ask this, then wait:** Does an idle expert necessarily mean no training data exists?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Routing and load allocation can leave it unused.

**Repair:** Inspect received-token counts and wait times by expert.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-10"></a>
### 14.10 Activation checkpointing

**Goal:** Choose recomputation boundaries.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Save boundary activations and recompute internal forwards during backward. Fewer boundaries generally save more memory but add work. Random operations need appropriate state restoration to preserve training semantics.

**Worked example:** Saving every 4-layer boundary allows recomputation within each group, not discarding all weights and gradients.

**Ask this, then wait:** Is saved byte count enough to select a scheme?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Include extra computation, throughput, and random-state handling.

**Repair:** Compare peak memory with time to the same quality.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-11"></a>
### 14.11 Combining parallelism strategies

**Goal:** Map logical groups to physical devices.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Large training can combine data, tensor, pipeline, context, and expert parallelism. Groups communicate differently. Public layouts are cases for specific hardware and scales, not universal optima.

**Worked example:** 8 cards might form 2-way tensor parallelism×4-way data parallelism, with different membership in each group.

**Ask this, then wait:** Does “2×4” fully describe the layout?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Map groups to devices and specify their meaning.

**Repair:** List each rank's groups and estimate link loads.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p14-12"></a>
### 14.12 Estimating communication and memory

**Goal:** State accounting conventions.
**Review if needed:** 13.02, 13.03, 09.04.

**Explanation:** Communication volume depends on the scheme, steps, and whether bytes sent or received are counted. Ring all-reduce sends about 2(p−1)S/p per rank. Sharding still needs temporary gathers, activations, and buffers.

**Worked example:** For p=4 and S=100MB, ideal sent traffic is about 150MB per rank, excluding protocol/topology effects.

**Ask this, then wait:** Does a 100MB gradient imply only 100MB transferred by each rank?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Not necessarily. Multiple rounds determine traffic.

**Repair:** Name the collective and byte-counting convention first.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L07](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%207%20-%20Parallelism%20basics.pdf) · [2024 L08](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_08.py) · [2025 L07](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/4eff81bee0a853217209e163936b264f03572b66/nonexecutable/2025%20Lecture%207%20-%20Parallelism%20basics.pdf) · [2025 L08](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_08.py) · [2026 L07](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_07.py) · [2026 L08](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_08.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official captions for 2026 L08 were not obtained; some student questions remain unrecoverable in local ASR. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c15"></a>
## 15 · Inference and serving

Understand generation-time costs.

<a id="p15-01"></a>
### 15.01 Autoregressive sampling

**Goal:** Separate a distribution from a generation path.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Each step samples conditionally and appends the result. Fixed weights can yield different outputs. Greedy chooses a locally most likely token, not necessarily the globally most probable complete sequence.

**Worked example:** First-step A=0.6,B=0.4; best following probabilities are 0.1 after A and 0.9 after B. B's path gives 0.36 versus 0.06.

**Ask this, then wait:** Does stepwise greedy guarantee the highest sequence probability?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Multiply probabilities on a two-level tree; do not call randomness a weight update.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-02"></a>
### 15.02 Temperature, top-k, and top-p

**Goal:** Understand sampling transformations.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Temperature uses z/T for T>0. Top-k retains k highest-scoring choices. Top-p retains a smallest sorted prefix reaching a probability threshold, then renormalizes. These alter sampling, not guaranteed factual accuracy.

**Worked example:** For [0.5,0.3,0.2], top-p=0.7 normally retains the first two, totaling 0.8; renormalization gives 0.625,0.375.

**Ask this, then wait:** Does top-p=0.7 mean always retaining 7 tokens?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Count depends on the distribution.

**Repair:** Sort and accumulate before counting retained choices.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-03"></a>
### 15.03 Prefill and decode

**Goal:** Separate input processing from output steps.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Prefill processes input positions and builds caches. Decode repeatedly adds few new tokens and reads history. Shapes and bottlenecks often differ, so measure time to first token and inter-token time separately.

**Worked example:** A 1000-token input and 20-token output require prefill over 1000 positions and about 20 generation steps.

**Ask this, then wait:** Does a slow first token imply every output token is equally slow?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. The stages differ.

**Repair:** Split the service timeline into prefill and decoding.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-04"></a>
### 15.04 KV caching

**Goal:** Explain what the cache stores.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Historical keys/values avoid recomputing the entire prefix each step. New queries still interact with historical information. This is not an answer database and does not eliminate long-history read costs.

**Worked example:** At token 11, reuse KV for the previous 10 positions and add current information; cache grows with history.

**Ask this, then wait:** Does caching remove all reads of history?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Attention still reads KV or compressed history.

**Repair:** Separate not recomputing KV from not reading it.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-05"></a>
### 15.05 Latency and throughput

**Goal:** Choose metrics for the user goal.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Latency measures one request's waiting; throughput measures work per time. First-token, inter-token, and completion latency differ. Batching can improve throughput while worsening interaction; inspect percentiles too.

**Worked example:** Waiting to batch 10 requests may improve tokens/s but make the first request wait longer.

**Ask this, then wait:** Does doubled throughput imply halved waiting for each user?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Queueing and batching change latency.

**Repair:** Record arrival, execution start, first token, and completion separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-06"></a>
### 15.06 Batch size and memory bandwidth

**Goal:** Understand weight reuse across requests.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Small-batch decode can be weight-bandwidth-bound. Larger batches reuse weights for more tokens and raise arithmetic intensity, but add per-request KV memory. Unlimited batch growth does not guarantee speed.

**Worked example:** Reading weights for 8 requests amortizes traffic compared with 1, but requires 8 histories.

**Ask this, then wait:** Why can memory limit batch size before compute does?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Caches, activations, and workspaces grow with concurrency.

**Repair:** Separate shared weights from request-specific state.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-07"></a>
### 15.07 Quantization and AWQ

**Goal:** Understand calibrated low-bit approximation.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Quantization approximates values using low-bit integers or codebooks and scales. AWQ uses calibration activations to identify important weights and protect them through scaling, not merely retaining largest weights. Kernel performance and calibration coverage matter.

**Worked example:** At scale 0.1, 0.26 may round to 0.3, giving error 0.04.

**Ask this, then wait:** Do close weights guarantee unchanged quality on every task?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Evaluate end-to-end behavior and calibration coverage.

**Repair:** Distinguish numerical error from task error; do not call it universally lossless.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-08"></a>
### 15.08 Pruning

**Goal:** Separate zeros from skipped work.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Pruning removes weights, channels, heads, or layers. Unstructured zeros may still execute in dense kernels. Structured pruning changes shapes more directly but can lose quality; sparse hardware may require particular patterns.

**Worked example:** Zeroing half the entries but calling the same dense matrix multiplication may still compute all positions.

**Ask this, then wait:** Does 50% zeros imply twice the speed?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Representation, kernels, and overhead determine gains.

**Repair:** Ask whether execution actually skips zeros.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-09"></a>
### 15.09 Cache compression and structural changes

**Goal:** Include information and compatibility costs.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Caches can be quantized, evicted, or compressed; structures such as GQA/MLA reduce storage differently. Eviction loses information; quantization adds error. Structural changes generally need suitable training, not arbitrary replacement in old weights.

**Worked example:** Deleting the oldest half of KV saves memory but can hurt questions about the first sentence.

**Ask this, then wait:** Is compression ratio sufficient to choose a method?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Test retrieval, latency, and compatibility.

**Repair:** Use a task that depends on information the method removed.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-10"></a>
### 15.10 Speculative sampling and verification

**Goal:** Understand drafting plus distribution correction.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** A cheap draft proposes several tokens; a target verifies them in parallel. Exact sampling uses accept/reject and residual-distribution correction. Keeping only target argmax is not a general random-sampling correction.

**Worked example:** For draft q(a)=0.6 and target p(a)=0.3, acceptance of proposed a is min(1,p/q)=0.5; rejection still needs a corrected draw.

**Ask this, then wait:** Can arbitrary replacement after rejection preserve the target distribution?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Explain the correction before discussing acceptance-rate speedups.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-11"></a>
### 15.11 Continuous batching

**Goal:** Let batch membership change.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** At generation boundaries, completed requests leave and new requests join. Unlike static batching, capacity need not wait for every original request. Scheduling still needs fairness and memory management.

**Worked example:** Requests of output lengths 2,20,20 allow a new request to replace the short one instead of leaving its slot idle.

**Ask this, then wait:** Must all continuously batched requests start and finish together?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Write the membership list at each iteration.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-12"></a>
### 15.12 PagedAttention

**Goal:** Separate logical order from physical storage.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** KV is mapped through fixed-size physical blocks, allowing a logical sequence to occupy noncontiguous storage. This reduces fragmentation and supports some sharing. It is not automatic disk paging or unlimited memory.

**Worked example:** Three logical blocks can reside in physical blocks 2,7,9 with a block table preserving order.

**Ask this, then wait:** Must adjacent logical tokens occupy adjacent physical blocks?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. The mapping and kernel handle the layout.

**Repair:** Use page numbers and shelf locations, retaining block-tail waste.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-13"></a>
### 15.13 Dynamic request management

**Goal:** Treat scheduling as constrained optimization.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** Service chooses admission, batches, chunked prefill, preemption, and resumption under throughput, latency, fairness, and memory constraints. Long requests can dominate resources; validate policies with realistic arrival patterns.

**Worked example:** A long exclusive prefill stalls existing decoders; chunking may help responsiveness while adding scheduling cost.

**Ask this, then wait:** Why can optimizing mean throughput harm some users?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Tail latency, starvation, and long waits can be hidden by averages.

**Repair:** Inspect p95 and groups by request length.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p15-14"></a>
### 15.14 vLLM's role

**Goal:** Separate model weights from an inference engine.
**Review if needed:** 01.05, 05.12, 09.08.

**Explanation:** vLLM integrates serving, scheduling, KV management, and kernels; it is not a new training objective. Supported models, quantization, parallelism, and APIs change by version and need official documentation.

**Worked example:** The same weights can run in different engines; precision and generation settings can still alter output.

**Ask this, then wait:** Does changing the inference engine mean retraining?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Usually not.

**Repair:** Trace weights→runtime→interface and compare the changed layer.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L10](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_10.py) · [2026 L10](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_10.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c16"></a>
## 16 · Scaling-law foundations

Read curves and respect extrapolation limits.

<a id="p16-01"></a>
### 16.01 Power laws and logarithmic axes

**Goal:** Read a fitted scaling curve.
**Review if needed:** 00.06, 09.03.

**Explanation:** For L=E+A C^(−α), distinguish irreducible E from reducible loss. log(L−E)=log A−αlog C is linear where defined; raw L versus C is not. α relates to the fitted slope.

**Worked example:** If α=0.5, multiplying compute by 4 halves the reducible term, not necessarily total loss.

**Ask this, then wait:** Why not claim loss approaches 0 when E>0?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The asymptote is E.

**Repair:** Draw the floor before interpreting improvement.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-02"></a>
### 16.02 Data, parameters, and compute

**Goal:** Balance model and data within a budget.
**Review if needed:** 00.06, 09.03.

**Explanation:** A simplified fit is L(N,D)=E+A/N^α+B/D^β with C≈6ND. Fixed compute makes larger N reduce available D. Fits provide conditional empirical evidence, not guarantees far outside measured ranges.

**Worked example:** With ND=100, N=10 permits D=10; N=20 permits D=5. Which wins depends on loss terms.

**Ask this, then wait:** Why can N and D not both grow arbitrarily at fixed compute?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Their product is constrained.

**Repair:** Write the budget equation and solve for the other variable.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-03"></a>
### 16.03 Sample complexity and mean estimation

**Goal:** Understand the independent-sample assumption.
**Review if needed:** 00.06, 09.03.

**Explanation:** For independent identically distributed samples of finite variance, mean standard error is σ/√n. Greater precision costs samples. Correlation, repetition, and distribution shifts complicate effective sample counts in text.

**Worked example:** Halving standard error requires about 4 times as many independent samples, not 2.

**Ask this, then wait:** Does copying each document 4 times halve uncertainty under this formula?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Copies are not independent new evidence.

**Repair:** Check independence before applying 1/√n.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-04"></a>
### 16.04 Intrinsic-dimension explanations

**Goal:** Separate a mechanism hypothesis from a fit.
**Review if needed:** 00.06, 09.03.

**Explanation:** High-dimensional data may lie near lower-dimensional structure; coverage and neighbor error can motivate power laws. This is one explanatory route, not a unique proven cause of every language-model exponent.

**Worked example:** A curve in a 2-dimensional plane can be described locally by one position coordinate.

**Ask this, then wait:** Does an observed power law uniquely reveal language's true dimension?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Several mechanisms and noise can yield similar curves.

**Repair:** Separate fitting a phenomenon from identifying its cause.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-05"></a>
### 16.05 Distribution shifts

**Goal:** Know when old scaling fits stop applying.
**Review if needed:** 00.06, 09.03.

**Explanation:** Training/evaluation languages, domains, time, and difficulty affect loss and optimal allocation. A fit from one distribution needs validation on another, even at the same token count.

**Worked example:** A news-based curve may fail to predict code performance because syntax and evaluation changed.

**Ask this, then wait:** Does poor prediction on a new domain necessarily prove an implementation bug?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Distribution shift may explain it.

**Repair:** Compare data and metrics with the original fitting conditions.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-06"></a>
### 16.06 Repeated and finite data

**Goal:** Count exposure separately from information.
**Review if needed:** 00.06, 09.03.

**Explanation:** Repeated training can improve optimization with diminishing benefits or overfitting. Record unique tokens and total exposure separately. Epoch count alone does not determine useful data quantity.

**Worked example:** 100 independent examples read 10 times give 1000 exposures, not 1000 independent examples.

**Ask this, then wait:** Why can equal exposure totals produce different generalization?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Coverage, repetition, and noise differ.

**Repair:** Record unique amount, exposure, and epochs together.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-07"></a>
### 16.07 Architecture and optimizer scaling

**Goal:** Compare whole curves rather than one point.
**Review if needed:** 00.06, 09.03.

**Explanation:** Architecture, aspect ratio, and optimizers can change curve slopes, offsets, and stable regimes. A small-budget winner may lose later. Use suitable tuning and comparable budget definitions.

**Worked example:** An illustrative A starts lower while B improves faster; curves can cross at larger compute.

**Ask this, then wait:** What evidence supports A at a large target budget?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Measurements near that budget or validated multi-scale extrapolation with uncertainty.

**Repair:** Draw crossing curves to counter universal rankings from one point.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-08"></a>
### 16.08 Critical batch size

**Goal:** Assess diminishing batch benefits.
**Review if needed:** 00.06, 09.03.

**Explanation:** Larger batches reduce gradient noise and can reduce update count. Beyond a regime, additional examples yield little step reduction. The critical batch varies with model, phase, and optimizer.

**Worked example:** Going from 32 to 64 may help, while 4096 to 8192 may barely reduce steps yet consume more tokens.

**Ask this, then wait:** Must a larger batch save total FLOPs?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Measure examples needed to reach the same objective.

**Repair:** Multiply step count by examples per step.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-09"></a>
### 16.09 Kaplan and Chinchilla

**Goal:** Read historical allocation results conditionally.
**Review if needed:** 00.06, 09.03.

**Explanation:** The studies differ in data, recipes, fitting, and recommended allocations. Chinchilla-style results emphasize more balanced growth of parameters and tokens under their conditions, not an eternal ratio for all training and deployment.

**Worked example:** An oversized undertrained model can improve through reallocating budget toward data, not merely changing parameter count.

**Ask this, then wait:** Does longer training alone establish compute optimality?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Optimality requires a budget-constrained comparison.

**Repair:** State which cost is optimized before interpreting historical results.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-10"></a>
### 16.10 Envelopes, IsoFLOPs, and joint fits

**Goal:** Distinguish experimental designs.
**Review if needed:** 00.06, 09.03.

**Explanation:** A minimum envelope selects best losses by compute. IsoFLOPs sweeps N,D at fixed compute. Joint fits estimate a loss surface across configurations. Sampling and errors differ; these are not identical raw observations.

**Worked example:** At one budget, compare small/medium/large models; minima across several budgets form an envelope.

**Ask this, then wait:** Can one model size alone establish the optimum envelope?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Other allocations at the same budget are missing.

**Repair:** Inventory actually measured configurations first.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-11"></a>
### 16.11 Fitting error

**Goal:** Quantify extrapolation uncertainty.
**Review if needed:** 00.06, 09.03.

**Explanation:** Noise, sample selection, model form, and incomplete tuning can all affect a fit. Hold out scales, inspect residuals, and test sensitivity. High R² does not guarantee far extrapolation.

**Worked example:** Three nearly collinear small-scale points can fail to predict a 100-fold larger model with new bottlenecks.

**Ask this, then wait:** Why hold out scales despite a close training fit?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** To test transfer rather than only interpolation.

**Repair:** Separate measured points from unobserved prediction ranges.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p16-12"></a>
### 16.12 Training-optimal and deployment-optimal models

**Goal:** Include lifetime serving cost.
**Review if needed:** 00.06, 09.03.

**Explanation:** Training optimality concerns quality within training budget. Deployment adds request volume, latency, hardware, and operations. A smaller model trained longer may lower total serving cost if quality constraints hold.

**Worked example:** A trains cheaply but costs more per request; B trains expensively but serves cheaply. At high volume B can cost less overall.

**Ask this, then wait:** Does training-compute optimality imply cheapest deployment?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Include lifetime inference costs and constraints.

**Repair:** Use total cost=training+requests×per-request cost as an explicit simplification.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L09](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2025 L09](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/fb79eb018fa047bf99c4c785dcbbd62fff361e54/nonexecutable/2025%20Lecture%209%20-%20Scaling%20laws%20basics.pdf) · [2026 L09](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_09.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c17"></a>
## 17 · Scaling small experiments to large models

Understand transfer and experimental design.

<a id="p17-01"></a>
### 17.01 Initialization, learning rate, and batch together

**Goal:** Control scale and updates jointly.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** Width affects activations, gradients, and update scales. Initialization, learning rate, and batch cannot be copied independently. Fix comparable objectives and parameterization, then test scales; one adjustment cannot compensate for every structural change.

**Worked example:** Changing n summed terms to 2n may require variance scaling, while feature-update size still needs a separate check.

**Ask this, then wait:** Does stable initial variance prove stable update size?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Initialization and updates are separate constraints.

**Repair:** Measure step-0 activations and step-1 changes separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-02"></a>
### 17.02 μP conditions and basic derivation

**Goal:** Identify nondegenerate activation and update scales.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** μP aims to maintain activation and feature-update scales as width grows. Independent unit-variance inputs suggest Var(w)≈1/n for initialization, but gradients correlate with inputs, so update scaling needs another argument and layer/optimizer-specific rules.

**Worked example:** n independent terms with weight standard deviation 1/√n give roughly constant variance. This alone does not set readout or input-layer updates.

**Ask this, then wait:** Why not reuse 1/√n as every layer's learning rate?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Correlations and layer roles differ; initialization and update constraints are distinct.

**Repair:** Derive variance first, then identify where that independence argument fails for updates.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

Two conditions: A1 keeps each activation coordinate Θ(1); n such coordinates typically have norm Θ(√n), not Θ(1). A2 requires a nondegenerate one-step feature change. For a linear layer and single-sample SGD, ΔW=−ηg hᵀ, so ΔW h=−ηg∥h∥². If coordinates of h and g have constant scale, ∥h∥²≈n and this local update needs η of order 1/n. This illustrates why correlated outer-product updates differ from independent initialization sums; it is not a complete Transformer μP implementation table. Input, hidden, readout, and optimizer rules must still be coordinated; 2026 L11 and the original paper give broader conditions.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf) · [μTransfer paper](https://arxiv.org/abs/2203.03466)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-03"></a>
### 17.03 Width transfer

**Goal:** Validate consistent parameterization.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** μTransfer tunes base hyperparameters in narrow models and transfers using consistent μP rules. It is not copying all raw numbers, nor a guarantee across depth, data, or training phases.

**Worked example:** A fixed base learning rate can still imply width-dependent effective layer rates. Compare optimal base rates across widths.

**Ask this, then wait:** Does an unchanged base rate imply identical updates for every tensor?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Per-layer scaling can change.

**Repair:** Separate base hyperparameters from effective layer settings.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-04"></a>
### 17.04 Standard-parameterization controls

**Goal:** Audit the full comparison.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** Standard parameterization may preserve initialization without preserving feature updates across width. Compare initialization, readout scaling, optimizer rules, and tuning effort; a changed label or one shared rate is insufficient.

**Worked example:** Two implementations both specify 0.01, but different output scaling yields different effective updates.

**Ask this, then wait:** Do equal learning-rate numbers prove fairness?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Tabulate initialization and update rules by parameter class.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-05"></a>
### 17.05 RMSNorm, optimizer, and decay boundaries

**Goal:** Check modern components.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** RMSNorm gains, decay, optimizer epsilon, and attention scaling can affect transfer. The course treats such components as conditions to check, not automatically covered guarantees. Inspect activations, coordinate updates, and losses after adapting a system.

**Worked example:** Hidden weights can follow scaling rules while an unchanged norm-gain update creates another inconsistent path.

**Ask this, then wait:** Does adopting a μP library remove the need to inspect custom layers?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Trace activations and updates layer by layer.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-06"></a>
### 17.06 Public scaling recipes

**Goal:** Read configurations as conditional evidence.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** MiniCPM emphasizes transfer and WSD; DeepSeek combines architecture, data, and systems; CerebrasGPT reports multiple scales. Compare budgets, parameterization, data, and evaluation. Combining unrelated recipe fragments creates no validated reproduction.

**Worked example:** Combining A's learning rate with B's data mix does not reproduce either project.

**Ask this, then wait:** Why is a public hyperparameter table insufficient for guaranteed reproduction?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Code, data, budget, architecture, and unpublished details may matter.

**Repair:** Retain the source and configuration for each number.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-07"></a>
### 17.07 WSD and reusable training

**Goal:** Count all branches of an experiment.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** WSD permits shared stable prefixes followed by different decay tails. Total tuning cost includes every tail, trial, and failed run. Intermediate loss is not a substitute for a fully decayed comparison.

**Worked example:** An 80-step prefix plus two 20-step tails costs 120 steps, not 100 or 200.

**Ask this, then wait:** Why not report only the longest 100-step path as total tuning cost?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The second tail consumed 20 additional steps.

**Repair:** Draw a branching tree and count each executed edge once.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-08"></a>
### 17.08 Muon's scale dependence

**Goal:** Separate update geometry from implementation.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** Muon approximately normalizes spectral directions of matrix momentum, often via Newton–Schulz rather than complete SVD each step. Benefits depend on width, shape, budget, and implementation. Other parameters may use another optimizer.

**Worked example:** A rectangular matrix cannot have identity orthogonality on both sides merely by adjusting singular directions.

**Ask this, then wait:** Does a small-budget Muon win prove universal large-budget superiority?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Compare multiple scales with separate tuning and system costs.

**Repair:** Check whether the horizontal axis is tokens, FLOPs, or time.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-09"></a>
### 17.09 Multi-scale experiments and budget allocation

**Goal:** Buy evidence rather than repeated guesses.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** Use cheap runs for debugging, several scales for trends, and reserved budget near the target to validate predictions. Record failed runs. Spending everything tuning one small scale gives little extrapolation evidence.

**Worked example:** A hypothetical 10-unit budget could spend 2 debugging, 3 across scales, and 5 validating; this is not a prescribed course ratio.

**Ask this, then wait:** Why reserve a larger held-out model?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** To test whether transfer predictions hold.

**Repair:** State which hypothesis each experiment can rule out.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p17-10"></a>
### 17.10 Extrapolation error and risk

**Goal:** Define failure signals in advance.
**Review if needed:** 04.04, 04.05, 16.02.

**Explanation:** Transfer can fail with new architectures, data, bottlenecks, or optimization phases. Predictions need conditions, uncertainty, and stopping rules. Do not dismiss larger-scale failure as bad luck.

**Worked example:** Stable small-model rates followed by large-model loss spikes require numerical, update-scale, and data investigation.

**Ask this, then wait:** What evidence would make you pause an extrapolated recipe?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Systematic target-scale errors, divergence, or failed cross-scale validation.

**Repair:** Specify observable failure signals and separate bugs from failed assumptions.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L10](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2010%20-%20Scaling%20details.pdf) · [2025 L11](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/00191bba00d6d64621dc46ccaed9122681413a24/nonexecutable/2025%20Lecture%2011%20-%20Scaling%20details.pdf) · [2026 L11](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_11.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c18"></a>
## 18 · Evaluating models

Define the capability before choosing a metric.

<a id="p18-01"></a>
### 18.01 NLL, perplexity, and tokenization

**Goal:** Compare likelihood metrics on compatible units.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Perplexity is exp(L) for natural-log mean NLL L. Different tokenizers, preprocessing, or masks change the per-token unit. Align units or use a comparable byte-based measure before comparing.

**Worked example:** L=ln4 gives perplexity 4. A tokenizer with fewer tokens changes the unit of prediction.

**Ask this, then wait:** Does perplexity 5 versus 6 unconditionally establish the better model?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Data, units, and task goals must align.

**Repair:** Match denominator and preprocessing first.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-02"></a>
### 18.02 Multiple choice, knowledge, math, and code tests

**Goal:** Match metrics to target abilities.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Multiple choice measures option discrimination; math checks answers; code tests execute cases. Coverage and scoring limit all of them. Passing tests does not prove correctness for all inputs, nor does accuracy describe every ability.

**Worked example:** Code passes 5 cases but fails a sixth edge case; preserve test and environment details.

**Ask this, then wait:** Does a correct numerical answer prove every reasoning step was valid?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Define whether the target is final answers, process, or another ability.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-03"></a>
### 18.03 Few-shot and prompt sensitivity

**Goal:** Control evaluation prompts.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Few-shot examples orient a task, but selection, order, templates, and option arrangement affect scores. Fix or systematically vary these conditions; do not hide prompt-search cost by reporting only the best.

**Worked example:** Reordering A/B options changes an answer, exposing position or prompt sensitivity.

**Ask this, then wait:** What is missing from a best-prompt-only score?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Selection procedure, variability, and a fixed protocol.

**Repair:** Check several fixed templates and separate development from test.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-04"></a>
### 18.04 Instruction following, preferences, and Arena

**Goal:** Separate correctness from preference.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Instruction metrics test requirements; preference measures what evaluators favor. Arena-style comparisons depend on opponents, users, and sampling. A strong preference rank does not guarantee factual, mathematical, or safety performance.

**Worked example:** A friendly long answer wins votes despite an incorrect number.

**Ask this, then wait:** Why not replace fact checking with a preference ranking?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Different criteria and distributions are measured; style influences choices.

**Repair:** Score format, facts, and appeal separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-05"></a>
### 18.05 LLM-as-judge

**Goal:** Treat a model judge as a measurement tool.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Model judges apply rubrics cheaply but have bias, randomness, and adversarial weaknesses. Calibrate against human samples, check consistency, and save judge versions/prompts. A score is not ground truth.

**Worked example:** Repeated judgments of the same pair can disagree; swap order and repeat.

**Ask this, then wait:** Does a judge's score of 9 eliminate all human checking?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Inspect disagreements between human and model judgments.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-06"></a>
### 18.06 Length, position, and style biases

**Goal:** Detect confounding in judgments.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Judges may favor earlier, longer, more confident, or familiar-style outputs. Order randomization, length controls, and identity blinding reveal some bias without guaranteeing its removal.

**Worked example:** If the winner follows position after swapping two answers, order is affecting evaluation.

**Ask this, then wait:** What does a score increase from irrelevant extra text suggest?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Possible length confounding.

**Repair:** Change style while holding facts fixed and inspect the rubric.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-07"></a>
### 18.07 Reasoning and agent evaluation

**Goal:** Count complete test-time budgets.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Report reasoning tokens, samples, and selection for reasoning; agents additionally need environment/tool versions, step limits, retries, and success rules. pass@k differs from single-attempt success.

**Worked example:** One system samples 100 answers and another 1; any gain cannot be assigned only to model weights.

**Ask this, then wait:** Why report tool calls and retries?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** They change chances, cost, and task conditions.

**Repair:** Evaluate an entire trajectory rather than only final text.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-08"></a>
### 18.08 Safety, bias, and fairness

**Goal:** Define specific evaluation goals.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Safety tests particular risky behavior; fairness requires specified populations, contexts, and metrics. Excess refusal can reduce legitimate help. Aggregate scores can hide subgroup gaps. This lesson makes no current legal-compliance determination.

**Worked example:** Overall accuracy 90% can combine group results 99% and 81%, motivating investigation.

**Ask this, then wait:** Does equal refusal rate prove equally useful help?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Request distributions, errors, and helpful completion differ.

**Repair:** Break results down by group and error type.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-09"></a>
### 18.09 Contamination and detection limits

**Goal:** Understand what deduplication cannot prove.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Training overlap with tests or variants inflates scores. Exact/approximate matching, time splits, and fresh tests reduce risk. No detected match does not rule out paraphrases or indirect answers.

**Worked example:** Changing names and numbers can hide the same template from exact string matching.

**Ask this, then wait:** Does no matching hash prove a clean benchmark?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It only excludes a specified kind of exact overlap.

**Repair:** State the similarity types and datasets actually checked.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p18-10"></a>
### 18.10 Saturation, validity, real tasks, and uncertainty

**Goal:** Report what a score establishes.
**Review if needed:** 00.05, 00.08, 04.02.

**Explanation:** Validity asks whether a metric measures the intended ability. Saturated tests distinguish poorly. Finite samples, correlated questions, and selection create uncertainty. Examine realistic tasks, subgroups, and intervals.

**Worked example:** A 1-question difference over 100 may be unstable; shared templates also weaken independence.

**Ask this, then wait:** Why not declare a certain ability gap from a 1-point difference?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Sampling variation, dependence, and protocol differences may explain it.

**Repair:** Inspect paired outcomes and suitable statistics, not only totals.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L17](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2017%20-%20Evals.pdf) · [2025 L12](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_12.py) · [2026 L12](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_12.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c19"></a>
## 19 · Sources of training data

Understand provenance and limitations.

<a id="p19-01"></a>
### 19.01 Web pages, Common Crawl, and WARC

**Goal:** Separate archived responses from training text.
**Review if needed:** 00.08.

**Explanation:** Web archives contain crawl records; WARC can preserve HTTP responses and metadata. Raw responses still include HTML, repetition, junk, and personal information. Downloading is not equivalent to obtaining suitable training data.

**Worked example:** A response contains navigation, adverts, and body text. WARC preserves the response; body extraction is a later step.

**Ask this, then wait:** Does a WARC file already contain a clean training corpus?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Parsing, extraction, filtering, and source/permission checks remain.

**Repair:** Draw crawl→archive→body text→training samples.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L11](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_11.py) · [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L13](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_13.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p19-02"></a>
### 19.02 Encyclopedias, code, papers, books, and Q&A

**Goal:** Understand source-specific coverage.
**Review if needed:** 00.08.

**Explanation:** Sources offer different structures and knowledge, with language, date, authorship, and licensing differences. Variation within each source remains; its name alone is no quality guarantee.

**Worked example:** Papers emphasize technical writing; Q&A offers questions and explanations. Mixture proportions affect practiced tasks and style.

**Ask this, then wait:** Are all paper sentences correct and freely usable for training?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Truth, permissions, and suitability are separate.

**Repair:** Record source versions and intended uses rather than only “high quality.”

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L11](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_11.py) · [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L13](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_13.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p19-03"></a>
### 19.03 Named training-corpus projects

**Goal:** Identify each project's role.
**Review if needed:** 00.08.

**Explanation:** WebText selects linked pages; C4 filters web text; The Pile mixes sources; MassiveText is a project corpus, not a fully open reproduction package. RefinedWeb, Dolma, DCLM, and Nemotron-CC emphasize web processing/recipes. Common Pile emphasizes openly licensed sources; The Stack focuses on code. Names do not imply identical licenses or completeness.

**Worked example:** Compare provenance, extraction/filtering, deduplication, mixtures, and versions before TB. This historical course list is not a current ranking.

**Ask this, then wait:** Does a downloaded derivative necessarily contain the complete original training corpus?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Subsets, versions, and released scope differ.

**Repair:** Tabulate source, processing, and availability; mark unknowns explicitly.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L11](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_11.py) · [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L13](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_13.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p19-04"></a>
### 19.04 Source, language, domain, and time

**Goal:** Keep provenance metadata.
**Review if needed:** 00.08.

**Explanation:** Metadata supports coverage, bias, and contamination analysis. Distinguish publication, crawl, and processing dates; a recent crawl does not make old content current. Break counts down by language and domain.

**Worked example:** An article published in 2018 and crawled in 2026 is not automatically a statement of 2026 facts.

**Ask this, then wait:** Why is crawl date not the date knowledge arose?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Crawling and publication are different events.

**Repair:** Keep publication, retrieval, and version fields separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L11](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_11.py) · [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L13](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_13.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p19-05"></a>
### 19.05 Open weights versus open data

**Goal:** Separate access to parameters from input transparency.
**Review if needed:** 00.08.

**Explanation:** Weights permit uses subject to their license; released data allows some inspection of training inputs. Reproduction also needs processing, order, recipe, and compute. Even all components released do not promise bitwise reproduction.

**Worked example:** A weight file without the original web collection cannot reveal the complete training distribution.

**Ask this, then wait:** Can weights alone audit the origin of every training item?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Parameters are not a per-document provenance list.

**Repair:** Track weights, code, data, licenses, and recipe separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L11](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_11.py) · [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L13](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_13.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p19-06"></a>
### 19.06 Copyright, licenses, privacy, and traceability

**Goal:** Ask distinct governance questions.
**Review if needed:** 00.08.

**Explanation:** Copyright, authorization terms, privacy/context, and provenance are separate concerns. Public access does not itself authorize training or redistribution. Legal conclusions vary by jurisdiction/date; this lesson is not legal review.

**Worked example:** Reading an online article does not imply permission to redistribute its full text. Keep source and license records.

**Ask this, then wait:** Does “publicly online” alone establish unrestricted redistribution rights?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Check source terms and intended use; mark unresolved permissions without offering a legal guarantee.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L11](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_11.py) · [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L13](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_13.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c20"></a>
## 20 · Data conversion and filtering

Give operational meaning to data quality.

<a id="p20-01"></a>
### 20.01 HTML-to-text and body extraction

**Goal:** Preserve useful structure.
**Review if needed:** 19.01, 00.05.

**Explanation:** Remove navigation, boilerplate, and scripts while retaining paragraphs, tables, code, and formulas. Simply stripping tags can join words or destroy structure. Sample different site templates.

**Worked example:** Joining two paragraphs without separation loses a boundary; removing code indentation can damage an example.

**Ask this, then wait:** Does shorter extracted text prove better extraction?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Useful content may have been deleted.

**Repair:** Compare the original page with extraction for structure and information.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-02"></a>
### 20.02 Language identification

**Goal:** Treat labels as imperfect predictions.
**Review if needed:** 19.01, 00.05.

**Explanation:** Language classifiers can struggle with short, mixed-language, code-heavy, and low-resource text. Thresholds alter coverage; low confidence is not synonymous with uselessness.

**Worked example:** A Chinese tutorial with Python identifiers may be labeled unknown and wrongly removed.

**Ask this, then wait:** Does low language confidence prove a document is junk?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Inspect errors by language and length, including mixed content.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-03"></a>
### 20.03 Rule-based filtering

**Goal:** Understand interpretable but fallible rules.
**Review if needed:** 19.01, 00.05.

**Explanation:** Length, character ratios, repetition, and keyword rules are cheap and auditable. Threshold combinations can delete useful material. Passing a rule set is not a general definition of quality.

**Worked example:** A high digit ratio can indicate corruption or useful mathematical derivation/table content.

**Ask this, then wait:** What can be lost by deleting all number-heavy text?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Mathematics, tables, and other useful domains.

**Repair:** Construct a false-positive example and adjust rules or domains.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-04"></a>
### 20.04 Quality and harmful-content classifiers

**Goal:** Separate scoring dimensions.
**Review if needed:** 19.01, 00.05.

**Explanation:** Quality scores reflect labels or reference data; harmfulness scores target risk categories. They differ: discussing risky words is not necessarily giving harmful assistance. Choose thresholds according to goals and error costs.

**Worked example:** A history text may describe violence for education; keywords alone miss context.

**Ask this, then wait:** Does a high quality score guarantee safety and truth?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. These are different dimensions, with imperfect classifiers.

**Repair:** Score quality, factuality, and risk separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-05"></a>
### 20.05 Personal information handling

**Goal:** Recognize de-identification limits.
**Review if needed:** 19.01, 00.05.

**Explanation:** Detection can combine patterns, entities, and context. Removing names does not guarantee anonymity; rare attribute combinations can re-identify. Choices depend on purpose and policy. Use fictional placeholders, not real sensitive data.

**Worked example:** Removing a name while keeping a unique job and event may still identify someone.

**Ask this, then wait:** Does masking names alone ensure anonymity?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Context and quasi-identifier combinations remain.

**Repair:** Inspect combinations without trying to identify real people.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-06"></a>
### 20.06 KenLM, fastText, and DSIR

**Goal:** Distinguish scoring mechanisms.
**Review if needed:** 19.01, 00.05.

**Explanation:** KenLM models n-gram probabilities and can support perplexity filtering; fastText can train lightweight classifiers; DSIR estimates importance weights from target/candidate feature distributions. Familiarity, class scores, and target similarity are not identical notions of quality.

**Worked example:** A repeated familiar sentence can score well under KenLM without adding diversity. Target similarity does not guarantee truth.

**Ask this, then wait:** Does high DSIR weight mean absolute high quality?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It is relative to a target distribution.

**Repair:** Ask what reference each score uses.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-07"></a>
### 20.07 Importance sampling

**Goal:** Check support and variance.
**Review if needed:** 19.01, 00.05.

**Explanation:** To approximate target p using candidates q, ideal weights are p/q. q must cover p's support; extreme weights increase variance. Estimated features and clipping add approximation. Reweighting creates no absent data.

**Worked example:** A domain occupying 10% of q and 20% of p gets an illustrative weight 2. If q=0 where p>0, weighting cannot supply it.

**Ask this, then wait:** Can weighting create code samples when none exist?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Check coverage before adjusting ratios or variance.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-08"></a>
### 20.08 Model-based scoring

**Goal:** Account for judge bias and cost.
**Review if needed:** 19.01, 00.05.

**Explanation:** Models can score perplexity, educational value, or rubrics, but results depend on model, prompts, and training bias. Large-scale selection needs cost estimates and audits; high automatic scores do not verify facts.

**Worked example:** Preference for standard English can undervalue dialect or mixed-language text and narrow the corpus.

**Ask this, then wait:** Does a larger scoring model eliminate auditing?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Inspect threshold cases and stratify by domain/group.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-09"></a>
### 20.09 False removals and missed removals

**Goal:** Measure both error types.
**Review if needed:** 19.01, 00.05.

**Explanation:** False removal discards desired text; missed removal retains unwanted text. Thresholds trade these off. Evaluate against defined labels rather than only reporting the fraction deleted.

**Worked example:** Among 100 good texts, 10 wrongly removed measure one error; among 100 bad texts, 20 retained measure another.

**Ask this, then wait:** Does deleting 90% establish a better filter?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It may discard useful data.

**Repair:** Draw desired retain/delete against actual retain/delete in a 2×2 table.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-10"></a>
### 20.10 Effects on domains and groups

**Goal:** Inspect uneven coverage.
**Review if needed:** 19.01, 00.05.

**Explanation:** Filters can disproportionately remove languages, styles, or domains. An improved aggregate can hide harmed tasks. Report group retention, errors, and downstream behavior.

**Worked example:** 50% overall retention with only 5% for one language is a signal to investigate, not automatic proof of its cause.

**Ask this, then wait:** Do differing retention rates alone establish discrimination's cause?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. They motivate investigation.

**Repair:** Inspect source composition, labels, and rule triggers separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p20-11"></a>
### 20.11 Validate filtering through training

**Goal:** Test the proxy against downstream goals.
**Review if needed:** 19.01, 00.05.

**Explanation:** Offline scores are proxies. Compare filtered/unfiltered training with controlled model, budget, and evaluation, including subgroups. State whether token budgets match when comparing quality against quantity.

**Worked example:** Lower loss after training on half as many tokens is interesting, but repetition, tokenization, and leakage differences may confound it.

**Ask this, then wait:** Does a gain on one benchmark establish improvement for every use?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Predefine target evaluations and retain failures and side effects.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L12](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_12.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c21"></a>
## 21 · Data deduplication

Separate exact and approximate duplicates.

<a id="p21-01"></a>
### 21.01 Hashes and collisions

**Goal:** State what a fingerprint establishes.
**Review if needed:** 20.01, 00.05.

**Explanation:** Hashes map content to fixed-length fingerprints. Identical normalized bytes give identical hashes; distinct inputs can collide. Cryptographic collision risk may be tiny without logical impossibility. Hashes do not recognize semantic paraphrases.

**Worked example:** “hello” and “hello ” usually hash differently unless whitespace is normalized first.

**Ask this, then wait:** Do different hashes prove different meanings?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** State normalization rules before exact matching.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-02"></a>
### 21.02 Exact deduplication by document, paragraph, and line

**Goal:** Choose a meaningful granularity.
**Review if needed:** 20.01, 00.05.

**Explanation:** Document deduplication removes complete duplicates. Paragraph/line matching can find shared fragments but damage context or valid templates. Decide what copy to retain and preserve provenance.

**Worked example:** Two articles with a common footer differ as documents but share lines.

**Ask this, then wait:** Can deleting every repeated line harm code or quotations?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Yes.

**Repair:** Compare sample integrity at different granularities.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-03"></a>
### 21.03 Bloom filters and false positives

**Goal:** Understand one-sided membership error.
**Review if needed:** 20.01, 00.05.

**Explanation:** A Bloom filter uses bits and multiple hashes to answer “definitely absent” or “possibly present.” A correctly implemented ordinary filter without deletion has no false negatives, but can have false positives. It is not an exact dictionary.

**Worked example:** A new document may appear present because other entries set the same bits to 1; verify or accept a stated removal risk.

**Ask this, then wait:** Does “possibly present” prove duplication?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Illustrate bit overlap between different entries.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-04"></a>
### 21.04 Suffix arrays and repeated substrings

**Goal:** Find literal contiguous duplication.
**Review if needed:** 20.01, 00.05.

**Explanation:** Sorting suffixes exposes shared prefixes between neighbors and thus repeated substrings. Document boundaries and minimum lengths matter. Literal substring methods do not automatically recognize equivalent paraphrases.

**Worked example:** Suffixes of banana reveal repeated ana; concatenated corpora need boundaries to avoid false cross-document fragments.

**Ask this, then wait:** Does a suffix array directly prove semantic equivalence?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Mark character positions and separate substrings from unordered word collections.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-05"></a>
### 21.05 Sets, n-grams, and Jaccard similarity

**Goal:** Compute set overlap.
**Review if needed:** 20.01, 00.05.

**Explanation:** Jaccard equals intersection size divided by union size. Turning text into n-gram sets discards counts and some order. The n choice matters; empty sets need an explicit convention.

**Worked example:** A={a,b,c},B={b,c,d}: intersection 2, union 4, similarity 1/2.

**Ask this, then wait:** Does repeating a 100 times create 100 copies in a set?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Distinguish sets from multisets before counting.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-06"></a>
### 21.06 MinHash

**Goal:** Interpret a probabilistic similarity sketch.
**Review if needed:** 20.01, 00.05.

**Explanation:** Using a shared random permutation or suitable hashing, the probability that minimum elements match equals Jaccard similarity. Multiple signatures estimate it with sampling error; matches are not an exact duplication proof.

**Worked example:** 60 matches among 100 signature positions estimate about 0.6, not necessarily exactly 0.6.

**Ask this, then wait:** Do fully matching sketches logically prove identical original text?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Explain what information the compressed signature omits and optionally verify originals.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-07"></a>
### 21.07 LSH bands and rows

**Goal:** Derive candidate probability.
**Review if needed:** 20.01, 00.05.

**Explanation:** Partition br MinHash values into b bands of r rows. One fully matching band creates a candidate. Under ideal independence, probability is 1−(1−s^r)^b. Candidate verification remains separate.

**Worked example:** For s=0.5,r=2,b=2, probability is 1−(1−0.25)²=0.4375.

**Ask this, then wait:** At fixed r, does increasing b make candidacy easier?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Yes. There are more opportunities for one band to match.

**Repair:** Compute one minus the probability that every band fails.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-08"></a>
### 21.08 Scaling deduplication

**Goal:** Count indexing and candidate explosion.
**Review if needed:** 20.01, 00.05.

**Explanation:** Costs include signatures, buckets, sorting, transfer, pair verification, and grouping. Common templates can create huge buckets with quadratic candidate pairs. Hashing alone does not represent total cost.

**Worked example:** A 1000-document bucket has about 499500 pairs to verify.

**Ask this, then wait:** Why can a popular footer slow approximate deduplication?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It groups many unrelated documents and explodes candidate work.

**Repair:** Inspect bucket-size distributions before estimating verification costs.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p21-09"></a>
### 21.09 Deduplication and evaluation contamination

**Goal:** Separate internal from cross-split checks.
**Review if needed:** 20.01, 00.05.

**Explanation:** Internal training deduplication reduces repeated exposure. Training/test matching addresses direct contamination. Both can miss paraphrases or indirect answers; removing literal test text alone does not prove unseen solutions.

**Worked example:** Removing 10 training copies of A leaves 1; if A is a test item, cross-split removal or isolation is still needed.

**Ask this, then wait:** Does no internal duplication prove a clean test set?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Draw internal edges separately from train–test edges.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L13](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_13.py) · [2025 L14](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_14.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c22"></a>
## 22 · Data mixtures and training stages

Relate data choices to objectives.

<a id="p22-01"></a>
### 22.01 Data mixtures and ablations

**Goal:** Specify the sampling unit.
**Review if needed:** 19.04, 16.02.

**Explanation:** Mixture proportions can be by token or example; state which. Choose proportions using target evaluations and controlled experiments. Equal training tokens can still differ in coverage.

**Worked example:** 70% web and 30% code by tokens is not 70 versus 30 documents when lengths differ.

**Ask this, then wait:** Can document proportions determine token proportions alone?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Lengths and sampling rules matter.

**Repair:** Fix the denominator before comparison.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-02"></a>
### 22.02 Domain resampling

**Goal:** Separate exposure from new information.
**Review if needed:** 19.04, 16.02.

**Explanation:** Oversampling a domain may improve target behavior but reduces other exposure and can increase repetition. It creates no independent new examples; monitor forgetting and overfitting.

**Worked example:** 10 code and 90 web documents can yield half code exposures through weighting, while still containing only 10 unique code documents.

**Ask this, then wait:** Does doubled code exposure double unique code content?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Track unique content and total exposure together.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-03"></a>
### 22.03 Data selection and scaling

**Goal:** Validate mixture gains at multiple scales.
**Review if needed:** 19.04, 16.02.

**Explanation:** A selection method or mix may work only at certain model/data sizes. Gains can change as data is exhausted. One small-scale optimum is not automatically a large-scale optimum.

**Worked example:** A small model may favor dense tutorials while a larger model benefits from broader web data; this is a hypothesis to test.

**Ask this, then wait:** What should be added when scaling a mix tuned at one size?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Multi-scale or target-near controls.

**Repair:** Treat mixture choice as another transfer problem.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-04"></a>
### 22.04 Estimating data mixtures

**Goal:** Recognize proxy limitations.
**Review if needed:** 19.04, 16.02.

**Explanation:** Small models, loss matrices, or proxy optimization can estimate mixtures. Objectives, reference models, and dynamics matter; proxy optimality does not guarantee target-model optimality.

**Worked example:** High mathematics loss may suggest extra data or merely greater difficulty, not the greatest marginal benefit.

**Ask this, then wait:** Should the highest-loss domain get unlimited sampling weight?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Distinguish current difficulty from value of additional budget.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-05"></a>
### 22.05 Repetition and epochs

**Goal:** Use validation to stop repeated exposure.
**Review if needed:** 19.04, 16.02.

**Explanation:** Repeated epochs can reduce training loss while validation benefits diminish or reverse. Learning rates, deduplication, and quality affect the best stopping point; neither one epoch nor unlimited epochs is universally right.

**Worked example:** Validation improves at epoch 2 but deteriorates by epoch 5 despite falling training loss.

**Ask this, then wait:** Why stop while training loss still falls?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Generalization may no longer improve; continuation can overfit or waste compute.

**Repair:** Plot training and validation together.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-06"></a>
### 22.06 Long-context data

**Goal:** Separate long inputs from a complete extension recipe.
**Review if needed:** 19.04, 16.02.

**Explanation:** Useful long training needs long-range dependencies plus position handling, length scheduling, budgets, and evaluation. Concatenation alone does not guarantee utilization. Official 2026 L16 did not fully develop a complete extension recipe; this remains a documented gap.

**Worked example:** Joining 100 unrelated short documents creates length without necessarily teaching cross-document dependencies.

**Ask this, then wait:** Does accepting 100000 tokens prove reliable use of all of them?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Separate capacity from effective retrieval/reasoning and retain the missing-recipe boundary.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Official 2026 L16 did not fully develop a complete long-context extension recipe; no verified full recipe is invented here. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-07"></a>
### 22.07 Code infilling

**Goal:** Explain reordered prefix, suffix, and middle.
**Review if needed:** 19.04, 16.02.

**Explanation:** Infilling supplies left and right context through a specified token format and predicts the missing middle. It is not merely renamed ordinary next-token training; sequence layout defines the task.

**Worked example:** For “x=__; print(x),” a known suffix can condition the missing value. Actual marker formats depend on the recipe.

**Ask this, then wait:** Does knowing the suffix violate the infilling task?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It is an allowed condition; labels and visibility must match the format.

**Repair:** Draw the original three pieces and their training order.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-08"></a>
### 22.08 Domain and reasoning data

**Goal:** Validate final answers and processes separately.
**Review if needed:** 19.04, 16.02.

**Explanation:** Domain data adds terminology and tasks; reasoning data needs answer, process, and difficulty checks. Longer explanations are not necessarily correct. Independent domain evaluation and leakage controls remain necessary.

**Worked example:** A solution with the right answer but wrong derivation can teach invalid reasoning.

**Ask this, then wait:** Does checking only the final number verify the whole demonstration?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Record final-answer and step-verification statuses separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-09"></a>
### 22.09 Synthetic data and self-distillation

**Goal:** Assess information and error propagation.
**Review if needed:** 19.04, 16.02.

**Explanation:** Generated data can expand tasks or recombine skills; self-distillation trains on a model's own outputs. Neither guarantees new facts. Control errors, repetition, and distribution narrowing through external checks and real-data anchors.

**Worked example:** 100 rewordings of one answer increase item count without necessarily expanding knowledge.

**Ask this, then wait:** How would you test whether synthetic data adds useful information?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Use independent tasks, verification, diversity, and downstream controls.

**Repair:** Reserve evaluation data outside generation and selection.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-10"></a>
### 22.10 Midtraining

**Goal:** Identify the actual stage objective.
**Review if needed:** 19.04, 16.02.

**Explanation:** Midtraining often means continued training between broad pretraining and behavioral post-training, possibly emphasizing domains, code, or context. Terminology varies; inspect data, objectives, and budget rather than infer a standard recipe.

**Worked example:** General text→focused code training→instruction SFT is one possible stage sequence.

**Ask this, then wait:** Does the word midtraining imply RL?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It may still use next-token loss.

**Repair:** Write inputs and loss for every stage.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p22-11"></a>
### 22.11 Post-training data verification

**Goal:** Audit the whole example.
**Review if needed:** 19.04, 16.02.

**Explanation:** Check answerability, correctness, roles, format, provenance/permissions, duplicates, and test separation. Validate facts and preference labels separately. Small datasets can amplify individual mistakes; human approval is not infallibility.

**Worked example:** An instruction requesting two sentences paired with ten sentences and a wrong number teaches both format and fact errors.

**Ask this, then wait:** Is a fluent response sufficient quality control?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Audit question, answer, format, and supporting checks separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2025 L13](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_13.py) · [2026 L14](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_14.py) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c23"></a>
## 23 · Supervised and parameter-efficient fine-tuning

Understand instruction learning.

<a id="p23-01"></a>
### 23.01 Pretraining and instruction objectives

**Goal:** Understand what SFT changes.
**Review if needed:** 01.06, 04.02.

**Explanation:** SFT commonly retains token cross-entropy but uses instructions and appropriate responses. It changes the behavioral training distribution without requiring a new network. Pretraining may already contain Q&A; this is not a knowledge/no-knowledge boundary.

**Worked example:** A translation instruction paired with a translation raises that response's conditional probability.

**Ask this, then wait:** Does SFT require reinforcement-learning rewards?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Demonstration tokens provide direct supervision.

**Repair:** Separate target-token labels from reward scores.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-02"></a>
### 23.02 FLAN, Alpaca, and OpenAssistant

**Goal:** Compare instruction-data construction.
**Review if needed:** 01.06, 04.02.

**Explanation:** FLAN reformats tasks as instructions; Alpaca uses synthetic instruction generation; OpenAssistant includes crowdsourced dialogue. Provenance, quality, scale, and licensing differ. Historical project names are not current download recommendations.

**Worked example:** A sentiment instruction and a multiround conversation teach different interaction structures.

**Ask this, then wait:** Can mixing datasets ignore roles and templates?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Inspect actual sample fields before comparing projects.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-03"></a>
### 23.03 Conversation formats and loss masks

**Goal:** Separate loss inclusion from visibility.
**Review if needed:** 01.06, 04.02.

**Explanation:** Templates encode system/user/assistant roles. Loss masks choose scored targets; attention masks control visible inputs. Common SFT scores assistant replies while retaining user text as context.

**Worked example:** 3 user tokens plus 2 response tokens give a loss denominator of 2 valid targets, not all 5 tokens, when only replies are trained.

**Ask this, then wait:** Does excluding user-token loss make user input invisible?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Draw visibility and loss-weight masks separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-04"></a>
### 23.04 Small high-quality datasets

**Goal:** State the prerequisite capability.
**Review if needed:** 01.06, 04.02.

**Explanation:** A few curated examples can substantially steer a capable base model. Effects depend on existing skills, coverage, and evaluation; no fixed tiny size suffices for every task.

**Worked example:** A base model that translates may learn “output only the translation” from few examples. A model lacking the language needs more than formatting examples.

**Ask this, then wait:** What often underlies small-data success?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Relevant base capability and suitable task coverage.

**Repair:** Separate eliciting an existing skill from learning new knowledge.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-05"></a>
### 23.05 Style, facts, and knowledge elicitation

**Goal:** Evaluate factual changes independently.
**Review if needed:** 01.06, 04.02.

**Explanation:** SFT changes formatting, tone, and elicitation, and can also learn information. Style improvement alone does not establish better facts. Independent task/factual evaluation is needed.

**Worked example:** Changing “perhaps 2” to “definitely 2” does not fix a question whose answer is 3.

**Ask this, then wait:** Why evaluate both preference and factuality?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Style can mask factual errors; they measure different goals.

**Repair:** Score content separately from expression on the same response.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-06"></a>
### 23.06 Hallucinations and safety fine-tuning

**Goal:** Consider both unsafe answers and over-refusal.
**Review if needed:** 01.06, 04.02.

**Explanation:** Safety demonstrations teach refusals and appropriate assistance while trying to preserve useful help. They do not eliminate every hallucination, jailbreak, or risk. Evaluate harmful and benign scenarios.

**Worked example:** Refusing all medical education may raise refusal rate while reducing legitimate help.

**Ask this, then wait:** Are more refusals always better?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Distinguish appropriate refusal from mistaken refusal.

**Repair:** Use similar-worded requests with different intents without supplying dangerous procedures.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-07"></a>
### 23.07 SFT settings and evaluation

**Goal:** Record settings and regression effects.
**Review if needed:** 01.06, 04.02.

**Explanation:** Record templates, masks, learning rate, batch, epochs, and base version. Measure target and general tasks; falling loss need not improve instruction following. Excess training may forget prior capabilities.

**Worked example:** Customer-service quality improves while mathematics declines; report the tradeoff, not only the gain.

**Ask this, then wait:** Why evaluate tasks outside the tuning objective?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** To detect forgetting and broader side effects.

**Repair:** Maintain target and regression metric sets.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-08"></a>
### 23.08 Parameter-efficient fine-tuning

**Goal:** Separate trainable state from total work.
**Review if needed:** 01.06, 04.02.

**Explanation:** PEFT trains selected or added parameters, including adapters, prompt parameters, or LoRA. It saves trainable state and storage, but forward work, activations, and some backward propagation remain.

**Worked example:** With 1% trainable parameters, frozen layers still compute features and may pass gradients to earlier adapters.

**Ask this, then wait:** Does 1% trainable mean 1% training time?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Separate state updates, gradient flow, and forward computation.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-09"></a>
### 23.09 Low-rank adaptation

**Goal:** Count the update parameters.
**Review if needed:** 01.06, 04.02.

**Explanation:** LoRA uses W'=W+(α/r)BA, B:dout×r and A:r×din, freezing W and training A/B. It adds r(din+dout) parameters. Low rank constrains the update, not the entire resulting matrix; initialization often starts with zero update.

**Worked example:** din=dout=100,r=4 gives 800 added parameters versus 10000 base entries.

**Ask this, then wait:** Must W' have rank at most r?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. BA is low rank; W can be full rank.

**Repair:** Keep the original matrix separate from its increment.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-10"></a>
### 23.10 QLoRA, NF4, and double quantization

**Goal:** Separate storage, compute, and adapter precision.
**Review if needed:** 01.06, 04.02.

**Explanation:** QLoRA freezes a 4-bit base and trains LoRA through its computation. NF4 uses a nonuniform codebook suited to near-normal weights; double quantization compresses quantization constants, involving 8-bit constants in the original method, not universally 4-bit operations.

**Worked example:** 4-bit base weights are dequantized using scales for computation while adapters update at their training precision.

**Ask this, then wait:** Must backward update frozen 4-bit integers?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Gradients can pass through while updating only adapters.

**Repair:** Trace storage→dequantized computation→adapter update separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [QLoRA paper](https://arxiv.org/abs/2305.14314)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p23-11"></a>
### 23.11 Frozen versus full fine-tuning

**Goal:** Choose by constraints and quality.
**Review if needed:** 01.06, 04.02.

**Explanation:** Frozen bases save optimizer state and permit small task adapters. Full tuning offers a larger update space but costs memory and forgetting management. Quality depends on tasks and budgets; neither is universally equivalent or superior.

**Worked example:** 10 adapters can take less storage than 10 complete models, but serving switches, merging, and quantization still need design.

**Ask this, then wait:** Does a smaller adapter prove unchanged task quality?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Use comparable data, budgets, and independent evaluation.

**Repair:** Compare quality, memory, deployment, and update flexibility.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L14](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/lecture_14.py) · [2024 L15](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2015%20-%20Alignment%20by%20SFT.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c24"></a>
## 24 · Human preferences and RLHF

Understand how reward changes outputs.

<a id="p24-01"></a>
### 24.01 Pairwise preferences and reward models

**Goal:** Learn a preference score.
**Review if needed:** 23.01, 00.05.

**Explanation:** For two responses to one prompt, a reward model outputs scalar scores. Bradley–Terry commonly sets P(A preferred)=sigmoid(rA−rB) and fits pairwise likelihood. This is preference, not probability of factual truth.

**Worked example:** If rA−rB=ln3, preference probability is 3/4; adding 100 to both rewards changes nothing.

**Ask this, then wait:** Does a large absolute reward prove truth?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Pairwise data chiefly constrains differences and preferences.

**Repair:** Evaluate factual correctness separately from liking.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-02"></a>
### 24.02 Annotators, crowdsourcing ethics, and demographics

**Goal:** Identify whose preferences are represented.
**Review if needed:** 23.01, 00.05.

**Explanation:** Annotator groups, guidance, compensation, task design, and culture influence labels. Disagreement can reflect legitimate different needs rather than pure noise. Preserve provenance; one group's preference is not universal consensus.

**Worked example:** Some want concise answers; others need detail. Labels from only the first group can favor over-compression.

**Ask this, then wait:** Does majority vote establish every user's preferred style?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Compare legitimate needs without making claims about identifiable people.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-03"></a>
### 24.03 AI feedback and Constitutional AI

**Goal:** Separate stated principles from behavior.
**Review if needed:** 23.01, 00.05.

**Explanation:** AI feedback evaluates or revises outputs. Constitutional AI uses explicit principles for critique, revision, and preference training. Principles and judges remain fallible; lower cost does not remove governance or checks.

**Worked example:** An AI critiques an unsupported citation and revises it. The new citation still needs verification.

**Ask this, then wait:** Do good written principles guarantee compliance every time?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Separate rules, model behavior, and external validation.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-04"></a>
### 24.04 The gap between generation and evaluation

**Goal:** Validate each capability.
**Review if needed:** 23.01, 00.05.

**Explanation:** Comparing candidates can be easier than generating from scratch, but not always. Judges can exploit surface cues or miss hidden errors; verifiers also have limited coverage. Assess both generation and evaluation.

**Worked example:** A calculator can compare short equations, while a subtle long-proof error may remain hard to detect.

**Ask this, then wait:** Does a larger judge guarantee detection of every error?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Calibrate on a small independently verifiable task.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-05"></a>
### 24.05 Policies, returns, and KL constraints

**Goal:** Interpret reward minus deviation cost.
**Review if needed:** 23.01, 00.05.

**Explanation:** A policy π distributes responses. A common objective is Eπ[r]−βKL(π to πref), trading reward against reference deviation. β is not an absolute safety guarantee; specify sequence or token estimation.

**Worked example:** A high-reward answer that requires much more deviation can have lower net value; β=0 removes this penalty.

**Ask this, then wait:** Does KL prohibit changing every probability?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It permits changes at a cost.

**Repair:** Interpret a soft cost rather than frozen weights.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-06"></a>
### 24.06 PPO, values, advantages, and GAE

**Goal:** Separate roles in a policy update.
**Review if needed:** 23.01, 00.05.

**Explanation:** PPO samples from an old policy. A value predicts returns; advantage compares an action with a baseline. GAE sums TD residuals δt=rt+γVt+1−Vt with powers of γλ. PPO's min(ratio·A,clip(ratio)·A) limits its surrogate, with implementation-specific additional losses.

**Worked example:** Return 5 minus baseline 3 gives simplified A=2. Increasing relative probability helps the surrogate until clipping limits that incentive.

**Ask this, then wait:** Are value estimates and observed rewards the same source?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. One predicts; the other comes from a scorer/environment.

**Repair:** Start with single-step return minus baseline, then build GAE and clipping.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

GAE uses δt=rt+γV(st+1)−V(st) and Ât=Σl≥0(γλ)^l δt+l, with terminal bootstrapping set by the task. Small λ relies more on value estimates; larger λ incorporates longer residual chains. PPO uses ρt=πθ(at given st)/πold(at given st). Keep the min/clipping direction for A<0 too; clipping the surrogate does not directly bound token probabilities to [1−ε,1+ε].

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-07"></a>
### 24.07 DPO objective and derivation

**Goal:** Connect a KL optimum to pairwise learning.
**Review if needed:** 23.01, 00.05.

**Explanation:** Under its conditions, a reward/KL optimum satisfies π*∝πref exp(r/β), hence r=βlog(π*/πref)+a prompt constant. Substituting in preference likelihood yields −logσ(β[log(πθ(y+)/πref(y+))−log(πθ(y−)/πref(y−))]). The constant cancels for the same prompt.

**Worked example:** Increasing both relative log-probabilities equally leaves their difference unchanged. DPO concerns relative preference, not only winner probability.

**Ask this, then wait:** Does standard offline DPO require a separately trained explicit reward model and online sampling during tuning?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. It uses preference data and a reference under stated assumptions.

**Repair:** Explain differences and constant cancellation before hiding the finite-data limitations.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

In DPO, y+ is preferred and y− is the other response to the same x. πref is frozen; β is the KL scale. Response log probabilities sum conditional token log probabilities. The closed-form optimum assumes reference support and objective conditions; finite preference data and a parameterized model still incur estimation and representation error. β is neither a learning rate nor a measure of label reliability.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf) · [DPO paper](https://arxiv.org/abs/2305.18290)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-08"></a>
### 24.08 Preference-method variants

**Goal:** Compare the changed component.
**Review if needed:** 23.01, 00.05.

**Explanation:** Variants alter losses, references, length normalization, labels, or data collection. State what changed, whether sampling is online, and all costs. Without matched conditions no ranking establishes universal superiority.

**Worked example:** Changing loss on the same data differs from regenerating data with a stronger model; do not attribute both to the loss.

**Ask this, then wait:** If B changes data and improves, is its loss proven better?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Tabulate objective, data, sampling, and budget differences.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-09"></a>
### 24.09 Reward overoptimization and calibration

**Goal:** Recognize proxy exploitation.
**Review if needed:** 23.01, 00.05.

**Explanation:** Reward optimization can exploit judge bias. Length bias can reward verbosity, collapse reduces diversity, and confidence can become poorly calibrated. Track independent quality alongside reward and distributions.

**Worked example:** A judge likes lists, so the model always produces 20 items regardless of usefulness.

**Ask this, then wait:** When should rising reward trigger concern?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** When independent quality falls, length becomes abnormal, or outputs converge.

**Repair:** Use held-out evaluators and inspect high-scoring failures.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p24-10"></a>
### 24.10 Safety red teaming

**Goal:** Bound the evaluation claim.
**Review if needed:** 23.01, 00.05.

**Explanation:** Authorized red teaming seeks failures such as jailbreaks, privacy leakage, bias, and over-refusal. Record categories, criteria, and retests in a controlled environment. No found failure is not proof of universal safety.

**Worked example:** Use fictional sensitive placeholders rather than real secrets and perform no real attack.

**Ask this, then wait:** Does one fully passing red-team round prove absolute safety?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Coverage and attack budgets are finite.

**Repair:** State tested and untested categories without giving harmful execution procedures.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L16](https://raw.githubusercontent.com/stanford-cs336/spring2024-lectures/main/nonexecutable/Lecture%2016%20-%20Alignment%20by%20RLHF.pdf) · [2025 L15](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/61eddac004df975466cff0329b615f2d24230069/nonexecutable/2025%20Lecture%2015%20-%20RLHF%20Alignment.pdf) · [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2026 L15](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_15.pdf) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c25"></a>
## 25 · Reasoning with verifiable rewards

Understand RLVR and test-time computation.

<a id="p25-01"></a>
### 25.01 Verifiable rewards and scorers

**Goal:** Define what the verifier checks.
**Review if needed:** 24.05, 00.07.

**Explanation:** RLVR uses automatically checkable signals such as answer matching or code tests. A scorer verifies its covered rules only. Format exploits, incomplete tests, and parsing errors remain reward-design problems.

**Worked example:** A program printing one fixed sample can pass a one-input test without implementing the intended function.

**Ask this, then wait:** Does passing tests prove correctness for every input?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Add varied inputs and boundaries and state the verifier's coverage.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-02"></a>
### 25.02 Language models as policies

**Goal:** Identify states and actions.
**Review if needed:** 24.05, 00.07.

**Explanation:** The state includes prompt and generated prefix; an action is the next token, distributed by πθ. Completion forms a trajectory that may receive only a terminal reward. The objective raises expected sampled return rather than fitting demonstration tokens alone.

**Worked example:** At prefix “2+2=”, choose “4,” append it, then score after termination.

**Ask this, then wait:** Is the selected token a parameter or an action?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** An action; θ contains model parameters.

**Repair:** Label state, action, parameter, and reward separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-03"></a>
### 25.03 Trajectories, returns, and policy gradients

**Goal:** Derive the update direction.
**Review if needed:** 24.05, 00.07.

**Explanation:** For J=Eπ[R], the log-derivative gives ∇J=E[R∇logπ(y)]. Sequence log probability sums generated-token conditional logs. For minimization use a negative sign and treat reward as non-differentiable here.

**Worked example:** For loss=−R logp, R=1 encourages increasing p; R=−1 reverses the effect.

**Ask this, then wait:** Why does a positive-reward minimization loss have a minus sign?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Reducing loss should raise log probability.

**Repair:** Check the sign on a single adjustable probability before shared parameters.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-04"></a>
### 25.04 Baselines and variance

**Goal:** State the unbiasedness condition.
**Review if needed:** 24.05, 00.07.

**Explanation:** An action-independent baseline can reduce variance without changing the expected policy gradient. A finite group mean including the current sample has correlation subtleties; leave-one-out removes that self-dependence. Not every group baseline is automatically unbiased.

**Worked example:** Rewards 8 and 10 centered by 9 become −1 and 1, illustrating relative quality.

**Ask this, then wait:** Can an action-dependent baseline use the same unbiasedness argument directly?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Show E[b∇logπ]=b∇Σπ=0 and identify its condition.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-05"></a>
### 25.05 GRPO and group normalization

**Goal:** Compute relative advantages.
**Review if needed:** 24.05, 00.07.

**Explanation:** For G answers to one prompt, a common advantage is (Ri−mean)/(std+ε), with probability ratios and clipping. This avoids an explicit value model in this setup, but all-equal rewards provide no relative signal.

**Worked example:** Rewards [0,1], population std 0.5 and mean 0.5 yield approximately [−1,1], ignoring ε.

**Ask this, then wait:** If all group rewards are 0, can relative reward rank them?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. All centered advantages are 0.

**Repair:** Compute mean/std first and state population versus sample convention.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-06"></a>
### 25.06 Length and difficulty bias

**Goal:** Find implicit weighting.
**Review if needed:** 24.05, 00.07.

**Explanation:** Dividing each sequence by its own length changes trajectory weighting; reward-standard-deviation normalization reweights prompts by difficulty. Positive and negative advantages must be analyzed separately, not reduced to “longer always earns more.”

**Worked example:** With A=1, lengths 2 and 10 give per-token factors 1/2 and 1/10; with A=−1, retain the sign.

**Ask this, then wait:** Is averaging each sequence's loss always objective-neutral?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Length-dependent denominators change weights.

**Repair:** Write actual coefficients for both advantage signs.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-07"></a>
### 25.07 On-policy and off-policy learning

**Goal:** Track the behavior-policy version.
**Review if needed:** 24.05, 00.07.

**Explanation:** On-policy data comes from the current policy; off-policy data comes from another or older policy. Reusing a batch across updates creates divergence from its sampling policy. Save old probabilities and versions.

**Worked example:** Data sampled at update 0 and reused at update 5 came from a different policy even if recently collected.

**Ask this, then wait:** Is fresh data necessarily strictly on-policy?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Compare sampling and gradient-evaluation policies.

**Repair:** Record the generator version and reuse count.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-08"></a>
### 25.08 Importance weighting and clipping

**Goal:** Separate exact correction from surrogates.
**Review if needed:** 24.05, 00.07.

**Explanation:** Importance ratios πθ/πold correct distributions under support conditions. Sequence ratios multiply token ratios and can have high variance. Token-level surrogates and clipping often add bias for stability; they are not exact full corrections.

**Worked example:** Token ratios 2 and 3 give sequence ratio 6; using only 3 ignores prefix change.

**Ask this, then wait:** Can clipping guarantee zero bias while arbitrarily reducing variance?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Compute the full product before identifying the surrogate's change.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-09"></a>
### 25.09 Dr. GRPO, GSPO, and MaxRL

**Goal:** Compare denominators and ratio levels.
**Review if needed:** 24.05, 00.07.

**Explanation:** Dr. GRPO removes std normalization and uses a constant instead of per-sequence lengths. GSPO uses a geometric-mean sequence ratio and sequence clipping. MaxRL normalizes advantage by group mean. The 2026 assignment uses constant total normalization, unlike the paper's batch-token denominator.

**Worked example:** For [0,1], mean and std are both 0.5; for [0,0,0,1] they differ. Zero means require stated implementation handling.

**Ask this, then wait:** Can renaming unchanged code implement these three methods?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Check advantages, length denominators, and ratio level.

**Repair:** Compare formulas; do not invent a universal zero-mean rule.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

<details><summary>Derivation: unfold across turns if needed</summary>

Under the 2026 assignment: Dr. GRPO uses R−μ and constant total normalizer Z. Its MaxRL variant uses (R−μ)/μ with constant Z; the original paper uses a batch-token denominator. GSPO sets s=exp[mean_t(logπθ−logπold)] and optimizes min(sA,clip(s)A) at sequence level. Geometric averaging and clipping trade bias for stability. Zero-mean groups need explicitly documented implementation handling; no universal rule is invented here.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf) · [GSPO paper](https://arxiv.org/abs/2507.18071) · [Dr. GRPO paper](https://arxiv.org/abs/2503.20783) · [2026 A5: sections 5.1/5.3/6.3](https://raw.githubusercontent.com/stanford-cs336/assignment5-alignment/main/cs336_spring2026_assignment5_alignment.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-10"></a>
### 25.10 Rejection sampling and expert iteration

**Goal:** Understand selection followed by learning.
**Review if needed:** 24.05, 00.07.

**Explanation:** Rejection sampling retains acceptable generated responses for supervised training. Expert iteration repeats generate→verify/select→update. This differs from directly applying negative policy gradients to every failed trajectory.

**Worked example:** Generate 4 answers, keep 1 verified success for SFT, then sample again from the updated model.

**Ask this, then wait:** Does an accepted final answer guarantee valid intermediate reasoning?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. The verifier may check only the result.

**Repair:** State what selection verifies and what remains unchecked.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-11"></a>
### 25.11 R1, R1-Zero, Kimi, and Qwen cases

**Goal:** Read recipes by version and stage.
**Review if needed:** 24.05, 00.07.

**Explanation:** R1-Zero shows an RL route without preliminary cold-start SFT; R1 uses a multi-stage pipeline. Kimi and Qwen recipes differ. No case proves every base model gains reasoning from RL alone; do not mix versions.

**Worked example:** Compare base model, cold start, rewards, sampling budget, and evaluation rather than model name alone.

**Ask this, then wait:** Are R1 and R1-Zero the same training pipeline?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Draw stages and retain unpublished-detail limits.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-12"></a>
### 25.12 Distillation

**Goal:** Identify what the student learns.
**Review if needed:** 24.05, 00.07.

**Explanation:** A teacher supplies outputs, distributions, or reasoning trajectories. Its mistakes and selection bias can transfer. Students need not share architecture and need not acquire all teacher capabilities.

**Worked example:** A small model learns verified large-model solutions through SFT; that differs from its own RL exploration.

**Ask this, then wait:** Does copying long reasoning format prove equal reasoning ability?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Evaluate formatting, correctness, and novel-task generalization separately.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-13"></a>
### 25.13 Test-time scaling

**Goal:** Count candidate generation and selection.
**Review if needed:** 24.05, 00.07.

**Explanation:** Long reasoning, samples, search, and verifiers can improve success depending on task and budget. For independent attempts with success p, at least one success among k is 1−(1−p)^k. Choosing that success still needs a selector.

**Worked example:** p=0.2,k=2 gives 0.36 probability of a correct candidate; final success can be lower if selection fails.

**Ask this, then wait:** Does better pass@k guarantee the same gain for the final answer?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Separate a set containing success from selecting it.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p25-14"></a>
### 25.14 Agent environments, trajectories, and RL infrastructure

**Goal:** Evaluate actions in their environment.
**Review if needed:** 24.05, 00.07.

**Explanation:** Agents call tools and observe results; trajectories include failures, retries, and termination. Infrastructure needs consistent versions, asynchronous sampling, rewards, and logs. Use authorized isolated environments; offline replay does not prove live reliability.

**Worked example:** A fictional file-search sandbox checks both the submitted answer and permitted actions, not just a claim of completion.

**Ask this, then wait:** Why can final text miss an agent failure?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Intermediate actions may be invalid or outcomes misreported.

**Repair:** Compare each action with environment results, without unauthorized operations.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L16](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/e94e33f433985e57036b25215dff2a4292e67a4f/nonexecutable/2025%20Lecture%2016%20-%20RLVR.pdf) · [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py) · [2026 L16](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_16.pdf)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c26"></a>
## 26 · A small reinforcement-learning walkthrough

Map an abstract update to a toy task.

<a id="p26-01"></a>
### 26.01 The official 2025 sorting toy task

**Goal:** Start from the actual task definition.
**Review if needed:** 25.03, 03.07.

**Explanation:** 2025 L17 demonstrates RL with sorting generation. Read official input, format, and reward definitions first. This package's simplified CPU reconstruction is not a reproduction of the original GPU training.

**Worked example:** An original teaching example maps [3,1,2] to [1,2,3], separating order from preservation of elements; it is not claimed as a recorded classroom output.

**Ask this, then wait:** Besides order, what must sorting preserve?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Input elements and multiplicities.

**Repair:** Check task definition before reward code; do not guess implementation from intuition.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-02"></a>
### 26.02 Generating responses

**Goal:** Retain the sampled trajectory.
**Review if needed:** 25.03, 03.07.

**Explanation:** Responses may contain wrong numbers, duplicates, or explanations. Save full text/tokens, valid length, termination, and policy version. Random seeds can produce different trajectories.

**Worked example:** For [3,1,2], candidates “1 2 3” and “1 1 1” differ: the latter is nondecreasing but not a valid permutation.

**Ask this, then wait:** What is lost if only parsed numbers are stored?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Formatting, extra tokens, termination, and parse failures.

**Repair:** Save raw generation before parsing so cleaning does not hide errors.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-03"></a>
### 26.03 Computing token log probabilities

**Goal:** Align generated targets correctly.
**Review if needed:** 25.03, 03.07.

**Explanation:** Recomputed logits must align with shifted next-token targets and score only valid generated positions. Sequence log probability sums conditional log probabilities, not raw logits.

**Worked example:** Token probabilities 0.5 and 0.25 give probability 0.125 and log probability −ln2−ln4=−ln8.

**Ask this, then wait:** Why not add 0.5 and 0.25 for sequence probability?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Conditional probabilities multiply; their logarithms add.

**Repair:** Draw prompt/generation boundaries and shifted labels.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-04"></a>
### 26.04 Designing a sorting reward

**Goal:** Construct a reward exploit.
**Review if needed:** 25.03, 03.07.

**Explanation:** Checking only nondecreasing order rewards invalid solutions such as repeated numbers. Check length, input multiset, and order as required. Strict rewards may be sparse; partial rewards need explicit meanings.

**Worked example:** Input [3,1,2], output [0,0,0] is ordered but wrong. This is an original CPU counterexample, not claimed as a classroom question.

**Ask this, then wait:** Why is checking adjacent order insufficient?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It does not establish that output is a permutation.

**Repair:** Ask for another ordered but invalid output.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-05"></a>
### 26.05 Policy-gradient loss

**Goal:** Connect advantage and minimization.
**Review if needed:** 25.03, 03.07.

**Explanation:** A simplified loss is −stopgrad(A)Σlogπθ(generated token given prefix), excluding masked positions. Do not differentiate arbitrary reward paths; normalization also changes weighting.

**Worked example:** For A=2,total logp=−3, loss=6. Raising logp to −2.5 lowers loss to 5, encouraging the trajectory.

**Ask this, then wait:** With positive advantage, should raising log probability raise or lower loss?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Lower.

**Repair:** Substitute numbers to check signs rather than memorizing “reward times logp.”

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-06"></a>
### 26.06 KL penalties

**Goal:** Separate sampled terms from expectations.
**Review if needed:** 25.03, 03.07.

**Explanation:** The reference is usually frozen. Full-vocabulary KL and sampled estimates are different computations. One sampled logπ−logπref can be negative even though its expectation under π is nonnegative KL.

**Worked example:** Current probability 0.2 and reference 0.4 give log ratio ln0.5<0.

**Ask this, then wait:** Does a negative sampled term contradict KL nonnegativity?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. A sample term differs from the weighted expectation.

**Repair:** Write the sum with π weights before discussing nonnegativity.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-07"></a>
### 26.07 Freezing parameters

**Goal:** Audit model roles and updates.
**Review if needed:** 25.03, 03.07.

**Explanation:** The policy updates; reference, old-policy snapshots, and scorers are usually frozen according to the algorithm. Frozen parameters do not necessarily stop gradients through all computations. Inspect optimizer membership and snapshot refresh rules.

**Worked example:** If reference weights follow every policy step, the KL baseline moves and changes the intended constraint.

**Ask this, then wait:** Why should the reference not silently receive optimizer steps?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It provides a fixed or explicitly scheduled comparison baseline.

**Repair:** List model roles, gradient flags, and optimizer membership.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-08"></a>
### 26.08 Before-and-after probabilities

**Goal:** Avoid overclaiming one update's effect.
**Review if needed:** 25.03, 03.07.

**Explanation:** Positive advantage tends to encourage a sampled trajectory, but shared parameters, other samples, KL, clipping, and step size affect actual changes. Not every token or good trajectory must rise monotonically after a combined update.

**Worked example:** Two batch responses can share tokens but have conflicting reward signals; the total gradient determines the result.

**Ask this, then wait:** Must every token in a positive-reward answer gain probability?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Explain the isolated intuition, then add shared-parameter interference and measure changes.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-09"></a>
### 26.09 Sampling and update loops

**Goal:** Track the full learning cycle.
**Review if needed:** 25.03, 03.07.

**Explanation:** Sample prompts with a specified policy, save old probabilities, score, build loss, update, and evaluate. Record reuse count and policy version; training reward alone cannot replace independent testing.

**Worked example:** A second update on v1-generated data must acknowledge that the current policy may already differ.

**Ask this, then wait:** Why bind every trajectory to its generating policy version?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Correct ratios, reproducibility, and off-policy diagnosis.

**Repair:** Draw each loop's inputs and outputs and inspect stale state reuse.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p26-10"></a>
### 26.10 Comparing toy experiments

**Goal:** Use common success measures.
**Review if needed:** 25.03, 03.07.

**Explanation:** Control prompt distribution, budget, samples, and independent checks when varying reward, KL, freezing, or optimizers. A toy success validates a mechanism, not general reasoning or human alignment.

**Worked example:** Reward A checks only order; B checks valid sorting. A's larger training reward may merely reflect a weaker metric.

**Ask this, then wait:** Why not compare raw averages from different reward definitions?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Their scales and meanings differ.

**Repair:** Report common success, exploit rate, and actual execution scope.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2025 L17](https://raw.githubusercontent.com/stanford-cs336/spring2025-lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c27"></a>
## 27 · Multimodal models

Extend language-model concepts to images and video.

<a id="p27-01"></a>
### 27.01 Image encoders

**Goal:** Convert pixels into representation sequences.
**Review if needed:** 05.01, 23.03.

**Explanation:** Encoders such as ViTs map pixels/patches to features. A patch is not necessarily an object. Resolution and patching control token count and details; outputs still need a language-model interface.

**Worked example:** A 224×224 image with 16×16 patches gives 14×14=196 patches when divisible, excluding extra special tokens.

**Ask this, then wait:** Must one patch correspond to one whole object?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Draw the grid and locate objects crossing several patches.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-02"></a>
### 27.02 Image-text contrastive learning

**Goal:** Understand paired similarity objectives.
**Review if needed:** 05.01, 23.03.

**Explanation:** Contrastive training raises matched image/text similarity relative to other pairs, often using normalization and temperature. Some negatives may still be semantically compatible; matching labels are imperfect.

**Worked example:** Two images and two captions form a 2×2 similarity table with preferred diagonal pairs, although a caption might suit both images.

**Ask this, then wait:** Why might a negative pair not be semantically unrelated?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** It lacks that labeled pairing but can still share content.

**Repair:** Separate dataset pairing labels from semantic truth.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-03"></a>
### 27.03 CLIP and SigLIP

**Goal:** Compare representation-learning losses.
**Review if needed:** 05.01, 23.03.

**Explanation:** CLIP typically uses bidirectional batch softmax contrastive loss. SigLIP uses sigmoid-based pairwise binary objectives without that same batch-softmax denominator. Both learn representations rather than directly generating explanations.

**Worked example:** Softmax makes entries compete within a row; sigmoid scores labeled pairs, still requiring negative-pair design.

**Ask this, then wait:** Does a sigmoid objective make SigLIP a chat model?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Separate encoder objectives from a connected generator.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-04"></a>
### 27.04 Visual tokens and connectors

**Goal:** Bridge feature spaces.
**Review if needed:** 05.01, 23.03.

**Explanation:** Linear, MLP, or other connectors map visual features to the language model's width as conditioning representations. Matching width solves an interface issue, not semantic alignment; visual-token count also affects context cost.

**Worked example:** 196 features of width 768 map to 196 vectors of width 1024 for a corresponding language model.

**Ask this, then wait:** Does compatible shape prove image understanding?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Training must teach use of the features.

**Repair:** Distinguish interface compatibility from task capability.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-05"></a>
### 27.05 LLaVA and visual instruction tuning

**Goal:** Understand multimodal instruction stages.
**Review if needed:** 05.01, 23.03.

**Explanation:** LLaVA connects a visual encoder, connector, and language model and uses alignment/visual-instruction training stages. Freezing and recipes differ by version; do not generalize one setup to the entire family.

**Worked example:** An image plus “Where is the cup?” pairs with “on the left,” requiring both visual and text conditions.

**Ask this, then wait:** Can text-only Q&A demonstrate that visual ability was learned?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Change the image while keeping the question to test visual dependence.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-06"></a>
### 27.06 OneVision and variable resolution

**Goal:** Trade detail for visual-token cost.
**Review if needed:** 05.01, 23.03.

**Explanation:** OneVision-style systems use high-resolution crops and variable inputs for image/video, with version-specific rules. More tokens can preserve OCR detail but cost compute, caches, and language context.

**Worked example:** Small text can vanish in a downscaled image; more crops may retain it while adding tokens.

**Ask this, then wait:** Why is doubled visual-token count more than just a larger file?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The model processes more sequence positions.

**Repair:** Track detail coverage, token count, and inference cost together.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-07"></a>
### 27.07 Qwen-VL versions

**Goal:** Compare reported systems carefully.
**Review if needed:** 05.01, 23.03.

**Explanation:** Qwen-VL generations develop resolution, position, video, and other capabilities differently. Course visual demonstrations also contain arithmetic mistakes. Fluent output and model names do not establish correct chart reading.

**Worked example:** Verify sector proportions and prices independently rather than trusting demonstrated chart arithmetic; documented course corrections remain available.

**Ask this, then wait:** Does one successful chart demonstration prove all charts are read accurately?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Keep model version and original figure, then independently check numbers.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-08"></a>
### 27.08 Video and temporal position

**Goal:** Preserve temporal relationships.
**Review if needed:** 05.01, 23.03.

**Explanation:** Video can be sampled into frame features with temporal order/timestamps alongside spatial information. Frame rate and compression affect coverage. A bag of images does not automatically retain action direction.

**Worked example:** Sampling 1 frame per second can miss rapid events; reversing two frames can change “pick up” to “put down.”

**Ask this, then wait:** Will adding frames without representing order necessarily resolve action direction?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Reverse a two-frame example to test temporal sensitivity.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-09"></a>
### 27.09 Native multimodality and Chameleon

**Goal:** Distinguish joint-sequence designs.
**Review if needed:** 05.01, 23.03.

**Explanation:** Chameleon-style models place discrete image and text tokens in one sequence. “Native multimodal” has multiple definitions and does not mean no image encoder/discretizer. Check representations and objectives.

**Worked example:** Text followed by image tokens and more text can use one autoregressive sequence, unlike continuous features only conditioning a text model.

**Ask this, then wait:** Does a unified sequence mean all modalities start in the same raw format?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Pixels and text still require encodings.

**Repair:** Draw modality-specific entrances into the shared sequence.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p27-10"></a>
### 27.10 Understanding versus generation objectives

**Goal:** Evaluate the requested output modality.
**Review if needed:** 05.01, 23.03.

**Explanation:** Image understanding produces textual responses; image generation produces visual content, possibly using different losses and decoders. Description skill does not guarantee correct generation; attractive images need not match facts.

**Worked example:** Correctly saying “three balls” does not guarantee drawing three. Counting and visual appeal need separate checks.

**Ask this, then wait:** Can a visual-chat score represent image-generation quality?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No.

**Repair:** Specify input/output modalities before choosing metrics.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2026 L17](https://raw.githubusercontent.com/stanford-cs336/lectures/main/lecture_17.py)

**Source gaps and limits:** Public sources support teaching this point. Global audiovisual-review limits still apply; not all student questions have been recovered. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.


<a id="c28"></a>
## 28 · Guest lectures and later evidence

Preserve public sources and explicit gaps.

<a id="p28-01"></a>
### 28.01 Registering six guest lectures

**Goal:** State availability precisely.
**Review if needed:** 18.10.

**Explanation:** 2024 Ce Zhang and Aakanksha Chowdhery were officially not recorded; standalone class slides were not obtained. Full 2025 Junyang Lin/Mike Lewis and 2026 Daniel Selsam class materials were not obtained. Dan Fu 2026 has video/captions and related papers/code, but no standalone class slides. Availability is not a judgment of a speaker.

**Worked example:** Qwen3 reports are related reading, not recovered Junyang Lin classroom Q&A. Do not infer Selsam's class topic from his research.

**Ask this, then wait:** May another talk by a speaker be substituted as the same class?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Label different events and preserve the gap.

**Repair:** Check the guest ledger's evidence types without inventing transcripts.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s)

**Source gaps and limits:** See 28.01 and guest_checks in the lecture-comparison ledger for six individual statuses. Unavailable classes are not reconstructed as Q&A or demos. Dan Fu has video/captions, but standalone slides for that class were not obtained. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p28-02"></a>
### 28.02 Teach only retrieved guest material

**Goal:** Turn available evidence into bounded lessons.
**Review if needed:** 18.10.

**Explanation:** Retrieved Dan Fu material supports recurrent state, Parcae's linear transitions/input injection/nonlinear blocks, and Megakernel execution ideas. Negative continuous coefficients alone do not guarantee discrete stability. GPU performance was not reproduced.

**Worked example:** In an original Euler illustration, hnext=(1+Δa)h with a=−1,Δ=3 has coefficient −2 and growing magnitude. This is not claimed as Parcae's full equation.

**Ask this, then wait:** Why does a negative continuous coefficient not guarantee stability at any discrete step size?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** The discretization and step alter the iteration factor; inspect the actual stability conditions.

**Repair:** Validate the scalar counterexample, then return to the paper without inventing missing classes.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s) · [Parcae paper](https://arxiv.org/html/2604.12946v1) · [Megakernels author code](https://github.com/HazyResearch/Megakernels)

**Source gaps and limits:** See 28.01 and guest_checks in the lecture-comparison ledger for six individual statuses. Unavailable classes are not reconstructed as Q&A or demos. Dan Fu has video/captions, but standalone slides for that class were not obtained. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p28-03"></a>
### 28.03 Place updates by topic

**Goal:** Integrate evidence rather than names.
**Review if needed:** 18.10.

**Explanation:** Assign new work to architecture, data, systems, evaluation, or post-training. Record claims, conditions, evidence, and relation to prior understanding. Unverified news stays pending rather than changing confirmed facts.

**Worked example:** A KV-compression result belongs with caching/architecture and its quality/hardware conditions, not a “latest best” label.

**Ask this, then wait:** What must accompany a claimed twofold speedup?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Hardware, lengths, batch, precision, baseline, and quality constraints.

**Repair:** Use one evidence table to establish comparability.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s)

**Source gaps and limits:** See 28.01 and guest_checks in the lecture-comparison ledger for six individual statuses. Unavailable classes are not reconstructed as Q&A or demos. Dan Fu has video/captions, but standalone slides for that class were not obtained. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p28-04"></a>
### 28.04 Course-time statements and later evidence

**Goal:** Preserve dates and evidence levels.
**Review if needed:** 18.10.

**Explanation:** Course statements, author reports, independent replications, and local measurements differ. Updates should say what was supported, limited, or contradicted without rewriting historical statements.

**Worked example:** A 2025 report and 2026 replication may differ because settings changed, not necessarily because one refutes the other.

**Ask this, then wait:** Why not automatically trust the newer of conflicting papers?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Tasks, budgets, and definitions can differ; date does not replace evidence.

**Repair:** Align conditions before declaring a contradiction.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s)

**Source gaps and limits:** See 28.01 and guest_checks in the lecture-comparison ledger for six individual statuses. Unavailable classes are not reconstructed as Q&A or demos. Dan Fu has video/captions, but standalone slides for that class were not obtained. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p28-05"></a>
### 28.05 Do not invent unread lecture details

**Goal:** Stop at evidence gaps.
**Review if needed:** 18.10.

**Explanation:** If the class is unavailable, say so. Related papers can be taught as supplements, not as that speaker's classroom examples, student questions, or demos. Unrecoverable ASR must not become invented quoted speech.

**Worked example:** A scheduled speaker name does not establish that a particular system was demonstrated; class-specific evidence is required.

**Ask this, then wait:** How should an unrecoverable student question be handled?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** Mark it unavailable; a reliable lecturer restatement may be taught as a restatement, not a verbatim question.

**Repair:** Require a source for classroom attribution; otherwise relabel as original teaching or omit it.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s)

**Source gaps and limits:** See 28.01 and guest_checks in the lecture-comparison ledger for six individual statuses. Unavailable classes are not reconstructed as Q&A or demos. Dan Fu has video/captions, but standalone slides for that class were not obtained. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.

<a id="p28-06"></a>
### 28.06 Leaderboards and durable knowledge

**Goal:** Treat rankings as dated observations.
**Review if needed:** 18.10.

**Explanation:** Scores depend on tasks, prompts, judges, budgets, versions, and contamination. They can motivate questions, while mechanisms and measurement are durable learning goals. Recheck current prices/specifications/rankings from primary sources when teaching.

**Worked example:** A leader allowed 100 samples is not directly comparable to a model answering once.

**Ask this, then wait:** Must a new leader overturn existing architecture knowledge immediately?

<details><summary>Tutor reference: open after the learner answers</summary>

**Answer criteria:** No. Seek comparable conditions and reproducible mechanisms.

**Repair:** Rewrite a rank as a dated, versioned, protocol-bound claim.

Advance only when the learner explains the reason and relevant conditions. A correct choice without a reason calls for a follow-up on the same example. Use repair guidance after an incorrect answer and stay on this point. Record explicitly requested skips as unconfirmed.

</details>

**Sources:** [2024 L18](https://cs336.stanford.edu/spring2024/#schedule) · [2024 L19](https://cs336.stanford.edu/spring2024/#schedule) · [2025 L18](https://cs336.stanford.edu/spring2025/#schedule) · [2025 L19](https://cs336.stanford.edu/spring2025/#schedule) · [2026 L18](https://cs336.stanford.edu/#schedule) · [2026 L19](https://www.youtube.com/watch?v=9EEm4iMAF5s)

**Source gaps and limits:** See 28.01 and guest_checks in the lecture-comparison ledger for six individual statuses. Unavailable classes are not reconstructed as Q&A or demos. Dan Fu has video/captions, but standalone slides for that class were not obtained. Original teaching reconstruction, not a verbatim classroom transcript. Evidence baseline 1.2: no full manual audio listening or all-frame video inspection; no GPU training rerun. Examples are teaching constructions unless an executed experiment is explicitly linked.
