# **Research Proposal: Adaptive Attacks on Reasoning Distillation Defenses**

Author(s): **[Mustafa Ozdayi](mailto:ozdayi00@gmail.com)**

Created: **Apr 19, 2026**

Last Update: **Apr 19, 2026**

---

## **1\. Problem Statement**

When a large language model (LLM) solves a complex problem — whether in mathematics, coding, science, or multi-step analysis — it often shows its work in a chain-of-thought (CoT) reasoning trace that walks through the solution step by step. These traces turn out to be extremely valuable for training smaller models. Through a process called reasoning distillation, a practitioner can collect these traces and use supervised fine-tuning (SFT) to train a compact "student" model that learns to reason like the teacher. Li et al. (2025) showed that fine-tuning on just 17,000 traces from DeepSeek-R1 (Guo et al., 2025\) produces a student competitive with OpenAI's o1-preview.

This makes distillation an economic threat to frontier labs. Training a frontier reasoning model costs billions of dollars in compute; distilling from its outputs costs a tiny fraction of that. In early 2026, OpenAI, Google, and Anthropic each reported large-scale unauthorized distillation campaigns targeting their APIs. Anthropic's disclosure ("Detecting and Preventing Distillation Attacks," February 2026\) named DeepSeek, Moonshot, and MiniMax as running industrial-scale campaigns that generated over 16 million exchanges across 24,000 fraudulent accounts. Google detected a single campaign using over 100,000 prompts to replicate Gemini's multilingual reasoning.

In response, researchers have proposed output-level defenses — methods that modify or corrupt the teacher's reasoning traces so they still look correct to human users but become harmful for training a student. Published defenses include Antidistillation Sampling (ADS; Savani et al., NeurIPS 2025), Defensive Output Generation (DOGe; Li et al., 2025, submitted to ICLR 2026), and Information-Preserving Antidistillation Reformulation of Reasoning Traces (PART; Chen et al., 2025). Whether any frontier lab actually deploys these academic defenses is unknown. The only confirmed real-world defense, revealed through the leaked Claude Code source in March 2026, is injection of fake tool definitions into API responses — conceptually similar to ADS.

A critical gap remains: **no published work tests any of these defenses against an attacker who knows the defense exists and adapts.** Every defense paper evaluates only against an attacker who naively runs SFT on the defended traces. History suggests this is insufficient. In the image classification setting, Nasty Teacher (Ma et al., ICLR 2021 Spotlight) was considered a strong defense against model stealing — the teacher maintained its own accuracy but produced outputs specifically designed to mislead any student trained on them. Jandial et al. (ECCV 2022\) broke it using simple high-temperature calibration, recovering up to 68% of lost student performance. ADS and DOGe use the same principle (adversarially modify outputs to mislead students while preserving accuracy), and neither has been tested against an informed attacker.

We propose developing adaptive attacks against reasoning distillation defenses, grounded in recent theoretical work on what information the student actually learns during distillation.

## **2\. What Makes Reasoning Distillation Work**

### **2.1 Structure Over Content**

Li et al. ("LLMs Can Easily Learn to Reason from Demonstrations," 2025\) ran a series of ablation experiments that reveal what matters in reasoning traces. They deliberately corrupted different aspects of the training traces and measured the impact on student performance:

* **Wrong answers** in traces: only \-3.2% accuracy. The student still learns to reason, even when the final answers are incorrect.  
* **Randomized digits** within steps (e.g., replacing "3 × 4 \= 12" with "7 × 2 \= 19"): only \-4.3%.  
* **Removed reasoning keywords** ("wait," "let me think again"): only \-3.3%.  
* **Shuffled reasoning steps** (67% of steps randomly reordered): \-12.8%.  
* **Deleted reasoning steps** (67% removed entirely): \-12.8%.

The takeaway: the student learns a *structural template* — the pattern of "think step by step, check your work, backtrack if wrong" — not the specific mathematical content. Corrupting content barely matters. Disrupting structure is devastating.

### **2.2 Epistemic Verbalization**

Kim et al. ("Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty," March 2026\) go deeper. They argue that reasoning traces contain two independent types of information:

* **Procedural information:** The actual computation. "x \= 4 because 3 × 4 \= 12."  
* **Epistemic verbalization:** Moments where the model expresses uncertainty about its own reasoning. "Wait, that doesn't seem right — let me check." "Hmm, maybe I should try a different approach."

Their key theoretical result: when a model goes down a wrong path during reasoning, purely procedural continuation cannot recover — the model keeps computing confidently but incorrectly. Epistemic verbalization is what enables self-correction: the model explicitly doubts its current trajectory, which triggers it to backtrack and try something else.

To test this, Kim et al. ran an experiment they call hindsight distillation. They took correct reasoning traces — ones with heavy epistemic verbalization — and asked the model to rewrite them as clean, confident derivations, using the prompt: "Re-derive the result from scratch. Do not express any uncertainty — never say 'I think,' 'probably,' or 'it seems.' State everything with full confidence."

To illustrate, a trace with epistemic verbalization (from the LIMO dataset) might read:

> *"So we need to minimize a² \+ b² \+ c². Hmm, let me try assuming a \= b. Then c \= 23/a². Substituting... Wait, wait a second\! I think I made a critical mistake here. The problem asks for the smallest sphere that can contain EACH box in B — so we need the MAXIMUM space diagonal, not the minimum..."*

The hindsight version of the same problem:

> *"The smallest sphere containing all boxes must accommodate the maximum space diagonal. We set a \= b and solve for c \= 23/a². Computing a² \+ b² \+ c² \= 657/16. Therefore r² \= 657/64."*

Both reach the correct answer. But the first trace shows the model making a mistake, catching it, and correcting — exactly the behavior a student needs to learn. The second just states the clean solution. Training students on hindsight traces (no epistemic verbalization) produces catastrophic results:

| Model | Base | LIMO (epistemic) | Hindsight (no epistemic) |
| ----- | ----- | ----- | ----- |
| Qwen3-14B-Base | 16.7% | 60.0% | 3.3% |
| DeepSeek-R1-Distill-32B | 80.0% | 73.3% | 23.3% |
| Qwen2.5-7B | 13.3% | 26.7% | 3.3% |

*(AIME24 pass@1 accuracy. LIMO — "Less is More for Reasoning" (Ye et al., COLM 2025\) — is a curated dataset of 800 reasoning traces from DeepSeek-R1 and QwQ-32B. The traces were selected for problems the teacher found difficult, so they are rich with self-doubt and self-correction. "Wait" appears \~77 times per response on average.)*

Importantly, Kim et al. show that the specific tokens ("Wait," "Hmm") are not what matters — it's the *behavior* of expressing uncertainty. When they block these specific tokens at inference time, the model finds alternative expressions like "Is that correct? Let me check." Performance drops only 19-25%, not catastrophically. The behavior persists even when individual tokens are suppressed.

### **2.3 High-Entropy Forking Tokens**

Wang et al. ("Beyond the 80/20 Rule," NeurIPS 2025\) provide complementary evidence from a different angle. They measured the entropy (uncertainty) of each token during CoT generation and found that only about 20% of tokens have high entropy — moments where the model is deciding between multiple possible continuations. They call these "forking tokens." For example, after "So the answer is," the next token is low-entropy (the model knows what comes next). But after "Hmm, maybe I should," the next token is high-entropy (the model could go in many directions).

In their experiments with Reinforcement Learning with Verifiable Rewards (RLVR) — the RL training paradigm used to train reasoning models like DeepSeek-R1 — restricting gradient updates to only these 20% of high-entropy tokens matches or exceeds the performance of updating on all tokens. Training only on the remaining 80% causes severe degradation. These forking tokens overlap heavily with Kim et al.'s epistemic tokens — both point to the same small set of informationally critical moments in reasoning traces.

### **2.4 Distributional Alignment**

Kim et al. uncover one more piece of the puzzle: epistemic tokens in the training traces must be tokens the student model could plausibly generate. They measured how likely each student model considers the tokens in LIMO traces *before any training*, and found a stark divide:

* **Models where LIMO works** (e.g., Qwen3-14B-Base): tokens like "Wait" are unusual but not impossible — the model assigns them low but nonzero probability. SFT can upweight them.  
* **Models where LIMO fails** (e.g., Qwen2.5-Math-7B): tokens like "Wait" have near-zero probability. This math-specialized model was trained to be procedurally confident and never express doubt. The result: accuracy *drops* from 16.7% to 0.0% — LIMO makes the model *worse* than doing nothing.

To understand why, consider what happens during SFT. The training loss penalizes the model for failing to predict each token in the trace. When the trace says "Wait" and the student assigns it near-zero probability, the gradient from that single token is enormous — the loss is essentially \-log(≈0), which dominates the gradient for the entire sequence. With 800 training samples and "Wait" appearing \~77 times per sample, the model is overwhelmed by gradient signal pushing it to produce a token it has no natural capacity for. Instead of learning the useful behavioral pattern from the surrounding context, the training destabilizes.

This explains a puzzling asymmetry in Kim et al.'s results. On AIME24 (hard competition math), Qwen2.5-Math-7B drops from 16.7% to 0.0% after LIMO training. But on MATH500 (easier problems), the same model actually *improves* from 52.4% to 59.0%. The model can absorb procedural content from LIMO on easy problems where epistemic behavior isn't needed. But on hard problems — where self-correction is essential and the model would need to use the epistemic tokens it can't produce — it collapses.

The implications cut both ways:

* **For defenders:** Distributional misalignment is a *natural* defense. Some student models simply can't learn from epistemic traces without modification. A defender who knows their attacker is using a LLaMA-family student might specifically choose to express epistemic behavior using tokens that are out-of-distribution for LLaMA.  
* **For attackers:** The misalignment is an obstacle, but potentially a solvable one — through loss masking, preference optimization, or paraphrasing through the student's own model family (discussed in Section 4.4).

## **3\. Defense Taxonomy and Vulnerabilities**

### **3.1 Token-Level Defenses**

**How they work:** These defenses subtly change *which specific tokens* the teacher outputs, choosing tokens that are natural to humans but adversarial for student training.

* **Antidistillation Sampling (ADS; Savani et al., NeurIPS 2025):** Uses a proxy student model to identify which tokens would be most harmful for the real student to train on, then biases the teacher's sampling toward those tokens. For example, where the teacher might naturally say "calculate," ADS might choose "compute" if that specific token would push the student's gradients in a harmful direction.  
* **Defensive Output Generation (DOGe; Li et al., 2025):** Fine-tunes only the teacher's final output layer to maximize divergence from what a proxy student would expect, while preserving the teacher's own accuracy.  
* **Fake tool injection (Anthropic, leaked March 2026):** Inserts fake tool definitions into API responses. If someone is recording the API traffic to train a competitor, the fake tools pollute their training data.

**Why we hypothesize they're vulnerable:** These defenses operate on the information axis Li et al. showed matters least — the specific tokens used to express reasoning. If the student learns *behavioral structure* (when to doubt, when to backtrack) rather than specific token choices, then swapping individual tokens should have limited impact on what the student actually learns. Moreover, ADS was evaluated on GSM8K (grade school math), where the teacher solves problems effortlessly. On these easy problems, the teacher has plenty of headroom to swap tokens without hurting its own accuracy. We suspect that on harder problems — where the teacher itself needs to backtrack and self-correct — the perturbation budget would shrink, because corrupting the high-entropy forking tokens that matter would hurt the teacher's own performance. However, this has not been tested.

### **3.2 Structural Defenses**

**How they work:** PART (Chen et al., 2025\) modifies the *organization* of reasoning traces. It removes self-talk sentences (e.g., deleting "Hmm, let me think about this differently") and reorders the trace so that sub-conclusions appear before their derivations (e.g., stating "x \= 4" before showing the steps that led there). This targets both epistemic behavior (removing doubt expressions) and logical structure (disrupting step ordering).

**Why it's partially effective:** PART targets axes that matter more than token-level perturbation. But its reported effects are moderate. One possible explanation is that removing individual self-talk sentences doesn't fully eliminate the epistemic behavioral pattern — surrounding context may still imply self-correction even without the explicit doubt sentences. This has not been directly tested.

### **3.3 Behavioral Defenses**

**How they work:** These defenses eliminate epistemic verbalization entirely. Hindsight distillation (Kim et al., March 2026, used as an analytical tool) regenerates confident traces from correct solutions. Anthropic's connector-text summarization (leaked March 2026\) serves compressed summaries of reasoning rather than full traces — effectively real-time hindsight distillation.

**Why they're the most effective — but potentially still vulnerable:** Kim et al.'s Table 2 shows these produce by far the largest student degradation. However, we hypothesize they can be circumvented. Kim et al.'s own framework establishes that procedural information and epistemic verbalization are orthogonal axes. If that's true, an attacker could take the defender's confident procedural traces (which still contain useful domain knowledge) and mix in epistemic traces from a separate source — for example, from open-source reasoning models like DeepSeek-R1 or QwQ-32B, or from curated datasets like LIMO. The student would learn *what to compute* from the defended traces and *when to doubt itself* from the supplement. Whether this actually recovers student performance is an open empirical question and a central aim of this proposal.

### **3.4 DistillGuard's Meta-Finding**

DistillGuard (Jiang et al., March 2026\) tested several generic output-level defenses (paraphrasing, random poisoning, truncation) and found most of them ineffective. Their conclusion: "output-level defenses are generally insufficient to prevent knowledge distillation. The most effective defense — chain-of-thought removal — achieves its impact by withholding reasoning traces rather than corrupting the output."

## **4\. Proposed Attacks**

### **4.1 Quality-Guided Trace Selection (Unified Framework)**

Our key insight: the adaptive attack is fundamentally a **data curation problem.** The attacker doesn't need a clever training algorithm — they need to select good training data. We propose a scoring function that rates each available trace based on everything we know about what makes distillation work:

**Score(trace, student) \= correctness × epistemic\_density × distributional\_alignment × difficulty\_match**

Each component comes directly from the findings above:

* **Correctness:** Does the trace reach the right answer? (Hard filter — wrong traces are discarded.)  
* **Epistemic density:** Does the trace contain doubt-verify-proceed patterns? (Kim et al.'s key finding: this is what drives learning.)  
* **Distributional alignment:** How likely is the student to generate the epistemic tokens in this trace? (Kim et al.'s Figure 9: out-of-distribution tokens destabilize training.)  
* **Difficulty match:** Is this problem at the student's capability boundary? (LIMO's curation principle: problems the teacher found hard produce the best training traces.)

The attacker scores all available traces — from the defended teacher, from open-source models, from synthetic generation — and trains on the highest-scoring subset. Our hypothesis is that different defenses corrupt different components, but a principled scoring function can handle all of them. This framework subsumes several specific attack strategies:

### **4.2 Against Token-Level Defenses (ADS, DOGe)**

**Gold-guided gradient reweighting.** The attacker collects a small "gold set" of \~200 clean reasoning traces from any open-source reasoning model and uses them as a reference during training. Every few training steps, the attacker compares the gradient direction from each defended trace against the gradient direction from the gold set. Traces whose gradients point in the opposite direction from the gold set get downweighted — they are likely corrupted by the defense. Traces whose gradients align with the gold set get upweighted — they contain useful signal despite the defense.

This adapts a well-studied technique from the noisy-label literature (Ren et al., ICML 2018; Zhang et al., October 2025), which has been scaled to LLMs up to 34B parameters (ScaleBiO; Pan et al., 2024\) but never applied against distillation defenses.

### **4.3 Against Behavioral Defenses (Hindsight Distillation)**

**Epistemic supplementation.** The defender strips epistemic verbalization from their traces; the attacker adds it back from other sources. Three variants:

* **Naive mixing:** Simply concatenate the defender's confident traces with epistemic traces from open-source models (e.g., LIMO). The hypothesis is that the student would learn domain knowledge from the defended traces and learn *when to doubt itself* from the epistemic supplement, since Kim et al.'s framework treats these as independent information axes.  
* **Preference optimization:** Construct training pairs where the epistemic version is "preferred" over the confident version. Using algorithms like ORPO or DPO, the student learns to favor the doubt-verify-proceed reasoning style without being forced to reproduce any specific tokens — potentially sidestepping distributional alignment issues entirely.  
* **Synthetic augmentation:** Use an LLM to rewrite the defender's own traces, injecting fabricated self-doubt: take "Step 3: x \= 4\. Step 4: Therefore y \= 12" and rewrite as "Step 3: x \= 4\. Wait, let me verify — 3 × 4 \= 12, yes that checks out. Step 4: Therefore y \= 12." The doubt is fake — the model never actually doubted — but the student may still learn the self-checking behavior. If this works, the attacker needs no external data source at all.

### **4.4 Against Distributional Misalignment**

When the student model can't naturally produce epistemic tokens (as with Qwen2.5-Math-7B on LIMO), two approaches:

* **Loss masking:** During SFT, skip the loss computation on known epistemic tokens ("Wait," "Hmm," "Perhaps," etc.). The student still *sees* these tokens in context and can potentially learn the surrounding behavioral structure, but isn't penalized for failing to predict them. This should prevent the noisy gradients that we believe destabilize training. It's a one-line code change — making it the cheapest experiment to test.  
* **Target-family paraphrasing:** Have the student model (or a model from the same family) rewrite the epistemic traces in its own words. The doubt-verify-proceed behavior survives the paraphrase, but the surface tokens shift to whatever the student naturally uses to express uncertainty.

## **5\. Expected Contributions**

1. **First adaptive attacks** on LLM reasoning distillation defenses. Every existing defense has been evaluated only against naive SFT. We aim to test what happens when the attacker understands the defense and adapts.

2. **A unified attack framework** based on trace quality scoring. Rather than designing separate attacks for each defense, we hypothesize that principled data curation — guided by what we know about epistemic verbalization, distributional alignment, and problem difficulty — can defeat defenses across the board.

3. **Methods for overcoming distributional misalignment** in reasoning distillation (loss masking, preference optimization). If successful, these would have independent value as distillation methodology contributions, useful even outside the adversarial setting.

4. **Empirical evidence regarding a fundamental limitation of output-level defenses.** We aim to test whether the defender's dilemma holds in practice: any reasoning trace useful to a human reader may inherently preserve the epistemic behavioral structure that enables distillation.

## **Key References**

* Anthropic (February 2026). Detecting and Preventing Distillation Attacks. Blog post.  
* Chen et al. (2025). PART: Information-Preserving Antidistillation Reformulation of Reasoning Traces.  
* Guo et al. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.  
* Jandial et al. (ECCV 2022). Distilling Model Failures as Directions in Latent Space.  
* Jiang et al. (March 2026). DistillGuard: Evaluating Defenses Against LLM Knowledge Distillation.  
* Kim et al. (March 2026). Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty.  
* Li, Cao et al. (2025). LLMs Can Easily Learn to Reason from Demonstrations.  
* Li, Tan et al. (2025). DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation.  
* Ma et al. (ICLR 2021). Undistillable: Making a Nasty Teacher That CANNOT Teach Students.  
* Pan et al. (2024). ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting.  
* Qian et al. (NeurIPS 2025). Demystifying Reasoning Dynamics with Mutual Information.  
* Ren et al. (ICML 2018). Learning to Reweight Examples for Robust Deep Learning.  
* Savani et al. (NeurIPS 2025). Antidistillation Sampling.  
* Wang et al. (NeurIPS 2025). Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning.  
* Ye et al. (COLM 2025). LIMO: Less is More for Reasoning.  
* Zhang et al. (October 2025). Revisiting Meta-Learning with Noisy Labels.

