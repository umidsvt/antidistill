# Literature pass: answers-only training against teacher distillation (2026-11-19)

The question: what exists for training a student from a teacher's final answers only (plus the student itself), and
where is room for a new method? This draws on searches of 2025–2026 work. Most entries were read at abstract level, and
claims marked (abstract) need checking in the paper before we cite numbers.

## 1. The attack side: does releasing only answers protect?

| Work | What it says | Relevance |
| --- | --- | --- |
| DistillGuard (arXiv 2603.07835) | Of 27 benchmark-defense pairs, only removing the chain of thought (releasing answers only) protected substantially, on MATH-500 (abstract). It notes an attacker could train the student to write its own reasoning from the answers. | **The claim our paper can test directly:** is answers-only release a defense against a strong answers-only attacker? |
| The Distillation Game (arXiv 2605.22737) | Answer-only outputs are less informative than traces but allow some recovery. | Our setup 2 comes from its code. |
| Stealing Reasoning Traces (arXiv 2608.09867) | Traces beat answer-only distillation, 76.0% vs 68.4% for one model (abstract). | A weak answers-only baseline: plain SFT on answers. |
| Ma et al. (arXiv 2602.15143) | Answer-only SFT baseline at 5.2% on GSM8K. | We showed it understates the answer channel about sevenfold (STaR 35.2%). |
| Javaheri et al. (arXiv 2609.35699) | Defenses break after RL. | RL as the attacker's tool; closest to our level A. |
| What Does It Mean to Break a Distillation Defense? (arXiv 2606.25059) | Evaluation criteria for distillation defenses. | Framing. |
| Stepwise-Querying (arXiv 2610.41769) | Queries the teacher for step explanations. | Needs extra teacher queries, so outside our threat model and outside what we build. |

**Gap in this literature:** no work measures how close the strongest answers-only training gets to trace
distillation, for a realistic (instruction-tuned) attacker model, against the trace-distilled student as the target.

## 2. Methods that learn from answers (all assume answers are ground truth unless noted)

| Family | Key works | How the answer is used | Status for us |
| --- | --- | --- | --- |
| Rejection-sampling SFT / self-training | STaR (Zelikman et al. 2022), RFT (Yuan et al. 2023), ReST-EM (Singh et al. 2023) | Filter the student's own samples by answer; STaR's rationalization uses the answer as a hint | Baseline. Implemented (`dg_star.py`) |
| RL with verifiable rewards | GRPO (Shao et al. 2024), Dr. GRPO (Liu et al. 2025), DAPO | Reward 1 if the answer matches | Baseline. Implemented (`dg_rl_base.py`) |
| Answer-likelihood rewards | VeriFree (arXiv 2505.21493), JEPO, RLPR, LaTRO; comparison by Kwiatkowski et al. (arXiv 2602.03979): the log-probability of the reference answer is the only reward that works in every setup they test (abstract) | Reward = the student's probability of the reference answer after its own reasoning; dense, no checker | Baseline to implement |
| On-policy self-distillation with privileged information (OPSD) | Self-Distilled Reasoner (Zhao et al., arXiv 2601.18734, ICML 2026): the same model, shown the ground-truth answer or a reference solution, gives token-level targets on the student's own rollouts | The answer as privileged context | **The closest prior work to our "answer-anchored self-distillation" idea.** Baseline to implement |
| OPSD follow-ups | PS-OPSD (arXiv 2608.01589): more privileged information is not better. Rethinking OPSD for thinking models (ICML 2026): up to −17% for thinking models. Overcoming Scaling Limits (arXiv 2609.37915): a solution to a different problem also helps. What Does Privileged Information Add? (arXiv 2609.20612): reference-free distillation explains much of the gain | Ablations of what the privileged context adds | **OPSD gains may be partly generic regularization:** a wrong-answer control is essential |
| Hint-guided RL for problems with no correct sample | Scaf-GRPO (arXiv 2510.19807), Off-Context GRPO (arXiv 2607.19313, importance correction toward the unhinted policy), RGPO (arXiv 2610.07342), ADHint (arXiv 2512.13095), SEELE (arXiv 2509.06923), LTE (arXiv 2510.26109, own failed tries as hints) | Hints (partial solutions; sometimes only the answer) for zero-reward problems | **Our "hinted rollouts" idea overlaps with these.** Off-Context GRPO is the one to beat |
| Rationale filtering (spurious rationales) | CREST (arXiv 2411.06387, follow-up questions), REPS (EMNLP 2024, pairwise self-evaluation), certified reasoning (arXiv 2306.04031), semi-supervised pseudo-labelling (arXiv 2606.16811) | Keep only rationales that pass extra checks | Components for consistency weighting |

## 3. Noisy answers: the teacher is wrong on 12% (GSM8K) and 38% (MATH) of our training problems

| Work | What it says |
| --- | --- |
| Online Label Refinement, OLR (arXiv 2604.03993) | Separates inactive noise (wasted rollouts) from active noise (reinforced wrong answers). Replaces a noisy label with the policy's majority answer once that answer stabilizes. |
| When Can LLMs Learn to Reason with Weak Supervision? (arXiv 2604.18574) | With labels replaced by the model's most frequent wrong answer, Llama on MATH keeps nearly the same reward curves: it memorizes. |
| Noisy Data is Destructive to RLVR (arXiv 2603.16140) | Earlier "noise is fine" results were contaminated (more than 16% correct). Truly noisy labels cost about 9 points on MATH-500. |
| Label-free RL: TTRL (arXiv 2504.16084), RESTRAIN (arXiv 2510.02172) | The student's own majority vote as the label; RESTRAIN weights answers by vote counts. Our "no teacher" control. |

**Our "consistency-weighted answers" idea overlaps with OLR.** What differs in our regime:
- The noise is the teacher's own errors: plausible wrong answers, concentrated on hard problems. It is not random or
  adversarial.
- The target is U, which learns the same errors from the traces.
- So beating U on test accuracy through answer trust is possible in principle. No work tests this.

OLR is the baseline for this angle.

## 4. Theory hooks (for the framing; from the 2026-11-08 notes)

- Outcome supervision is not statistically harder than process supervision under some conditions (Jia, Rakhlin and
  Xie, arXiv 2502.10581).
- Learning to reason with and without chain-of-thought supervision (Joshi et al., COLT 2025; arXiv 2604.24737).
- Altabaa et al. (arXiv 2505.15927).

## 5. Where a new method can be novel

1. **The regime, not a single trick.**
   - An imperfect teacher's answers as the only signal.
   - An instruction-tuned attacker.
   - U as the target, measured with the share of the teacher's gain recovered.
   - Every component above was built for ground-truth labels and benchmark gains, not for this comparison.
2. **Answer trust for an imperfect teacher.**
   - Weight each released answer by evidence from the student: agreement across its samples, across its
     answer-conditioned rationales, and under checks.
   - Train on the trusted ones with dense credit.
   - Baselines: OLR, plus plain filtering.
   - The puzzle testbed already shows the attacker's own model filters most wrong answers (m8 §22).
3. **A combination justified by a measured gap decomposition** (coverage, spurious rationales, teacher errors,
   behaviors), with the corresponding baselines from each family. This is publishable only if the combination
   clearly beats the strongest single baseline.

Not novel on its own: answer-conditioned OPSD, answer-hinted GRPO, or answer-likelihood rewards. These are
baselines.
