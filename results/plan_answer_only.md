# Plan: an answers-only fine-tuning method that approaches teacher distillation (draft 2026-11-19, updated 2026-11-20)

Decided at the 2026-11-18 meeting:
- The main aim is a new level-A fine-tuning method (RL, SFT, a combination, consistency, …) on the math setups. The
  attacker owns only the student and the teacher's released answers. The goal is to get as close as possible to the
  student fine-tuned on the teacher's solutions (U).
- The student becomes an instruction-tuned model.
- The puzzle testbed stays as a second contribution.
- Levels B–D and the audit move to the appendix.

**Decisions of 2026-11-19:**
- Students: Llama-3.2-3B-Instruct (main) and Qwen2.5-3B-Instruct (second).
- A bigger teacher for the main setup. The field's setup (R1-Distill-Qwen-7B, 1,024 tokens) moves to the appendix as
  the narrow-gap case.
- ADS is closed (calibration only).
- The literature pass is done (`results/lit_answer_only.md`).

**Phase 0 result and decisions of 2026-11-20** (m8 §24–25; write-up `results/phase0_results.md`):
- With Llama-3.2-3B-Instruct, training on the teacher's solutions does not beat the untrained student on GSM8K or
  MATH with either teacher (gaps −3.6, −3.1, −7.8, +0.5 points). On the puzzles it takes the student from 0.8% to
  53.9% (+53.1, 95% CI 50.3 to 55.9). No math setting passes the gap rule (≥ 10 points, CI > 0).
- **The puzzle testbed is the main setting.** GSM8K/MATH are reported as "no gap for an instruct attacker" (main
  text briefly, appendix in full). The plan's risk 1 below happened.
- One recipe check on MATH (full fine-tuning, seed 123; m8 §25.1) is running.
- Phase 1 runs on the puzzles (m8 §25.2): B1 answers only reached 14.9% on seed 123 (26% of the gap, with no
  reasoning at all); B2 self-training with hints 2.6% (1,617 of 1,712 kept texts from hint rounds); RL B3/B4 running (1 epoch,
  about 13 h each with scoring); the controls follow.
- Paper: new framing draft `icml2027_answers/` in the paper repo.
- GPUs until 2026-11-22: only 0, 1 and 3 (4–7 for jafar, 2 for the user's other project).

This file is a plan, not a rule. Each phase gets its m8 entry (what is run, what each outcome means) before it runs,
and nothing starts without a go-ahead.

## 0. What we already know (base Llama-3.2-3B student; m8 §9, §12–15, §19)

| | GSM8K-Platinum (1,209) | MATH (5,000) |
| --- | --- | --- |
| Teacher R1-Distill-Qwen-7B, 1,024 tokens (setup's own budget) | 87.2% | 63.1% |
| Untrained base student | 0.6% | 0.9% |
| U: base student on the teacher's solutions | 59.8% | 15.5% |
| Best answers-only (STaR; GRPO alone 32%, STaR then GRPO 30%) | 35.2% | 9.6% |
| Llama-3.2-3B-Instruct as a solver (setup's generation) | 81.7% | 48.0% |
| Llama-3.2-3B-Instruct on the training problems | 89.6% | 71.5% |

Three consequences for the new plan:
1. **With the setup's teacher, an instruct student has little to gain on GSM8K.** It starts within 5.5 points of
   the teacher. U for an instruct student can hardly exceed the teacher, so the gap a method can close there is at
   most a few points, about the size of seed noise. MATH leaves up to 15 points.
2. **The instruct model has probably seen the GSM8K and MATH training sets.** It solves 8 and 24 points more
   training than test problems. Answers-only training on those problems may mostly rehearse memorized solutions.
   Test scores stay valid, but the gap and contamination must be reported.
3. **Most components of a method already exist in 2026 work:**
   - Answer- or solution-conditioned on-policy self-distillation (OPSD and its follow-ups).
   - Answer-likelihood rewards (VeriFree, JEPO, RLPR; Kwiatkowski et al. 2026).
   - Hint-guided GRPO for problems with no correct sample (Scaf-GRPO, Off-Context GRPO, ADHint).

   Novelty has to come from what our regime adds. The guide is an imperfect teacher whose answers are wrong on 12%
   (GSM8K) and 38% (MATH) of training problems; the target is U, not ground truth; and the gap gets measured and
   decomposed.

## 1. Phase 0: the setup with an instruct student (gate; about 1 day on 4 GPUs)

- **Student:** Llama-3.2-3B-Instruct, the instruct version of our current student and the level-B model. It gets the
  same LoRA recipe as U (r 128, 3 epochs, lr 5e-4), its own chat template, and the setup's evaluation (temperature
  0.6, answer forcing).
- **Measured, 3 seeds, GSM8K and MATH:**
  - I0, the untrained instruct student.
  - U_inst, the instruct student on the teacher's stored solutions (regenerated 2026-10-31; no new teacher run).
- **Bigger teacher (S2, the main candidate):** DeepSeek-R1-Distill-Qwen-32B with up to 8,192 new tokens, on both
  datasets. The field's setup (S1) is kept for the appendix. The rule is m8 §24.
- **Gate (to fix in m8):** a setup is used for method development only if U_inst − I0 is clearly above seed noise,
  for example at least 5 points with a 95% interval excluding 0. If GSM8K fails, it stays as a comparability check
  and MATH (or the full-budget variant) becomes the main setup.

## 2. Phase 1: strong answers-only baselines and the gap decomposition (about 3 days)

The bar to beat, all on the instruct student with the teacher's released answers, same data and same compute
budget. Development on seed 123; model selection on held-out training problems scored against the teacher's answers
(what an attacker can do), never on the test set.
- SFT on the answers alone (Ma et al.'s 5.2% baseline, for the base student).
- RFT/STaR: k samples, keep those matching the teacher's answer, with the answer-hint round (our §13 recipe).
- Iterated ReST-EM (2–3 rounds).
- GRPO (Dr. GRPO loss) rewarded by matching the teacher's answer; STaR then GRPO.
- One answer-likelihood reward (VeriFree-style) and answer-conditioned OPSD, as the strongest published members of
  their families.
- **Controls:**
  - The same best method with the student's own majority answers instead of the teacher's (TTRL-style). This shows
    what the teacher's answers add.
  - The same with ground-truth answers. This is an upper reference for any answers-only method.

**Gap decomposition** (diagnostic; uses ground truth, which only we have). Where does U_inst beat the best baseline?
- **Coverage:** problems where the student never reaches the teacher's answer.
- **Spurious rationales:** right answer, wrong reasoning.
- **Teacher errors:** wrong teacher answers learned or filtered.
- **Behaviors:** length, checking, re-solving that the teacher's traces teach and answers cannot.

Each candidate method below targets one of these, so the decomposition decides which to build.

## 3. Phase 2: method development (about 1–2 weeks)

Candidates, each tied to a gap term, built on the strongest Phase-1 baseline:
1. **Answer-anchored self-distillation (coverage, credit).** The student conditioned on the teacher's answer acts as
   a token-level teacher for itself on its own rollouts, with the verifiable reward kept. Our addition is a
   correction for answer leakage: the conditioned model prefers reasoning that jumps to the answer it was given.
   - Control: a wrong answer as the privileged information. 2026 work reports that unrelated privileged content also
     helps.
2. **Hinted rollouts with an off-context correction (coverage).** For problems with no correct sample, rollouts with
   the answer as a hint, reweighted toward the unhinted policy, with hints annealed.
3. **Consistency-weighted answers (spurious rationales, teacher errors).** Weight each teacher answer and each kept
   rationale by how consistently the student reaches it: across samples, across re-solving from the rationale, and
   by checking backward from the answer. This down-weights the teacher's wrong answers that U learns from.
   - The puzzle testbed already shows the effect: the attacker's own model reached only 7.5% of the corrupted answers
     (m8 §22).
   - This angle is specific to an imperfect teacher, and in principle it could even beat U on test accuracy.
4. Combinations of 1–3 with the best RL baseline.

**Fairness:** every candidate and baseline gets the same number of training problems, released answers and samples,
and comparable GPU hours. The released answers are a fixed set (one per training problem); no method chooses which
problems to send to the teacher.

## 4. Phase 3: confirmation (about 4–5 days on 6–8 GPUs)

The rule is written in m8 before any run. The best 1–2 methods and the best baseline run with 3 seeds on the setups
that pass Phase 0.
- **Reported:**
  - The test score.
  - The share of the teacher's gain recovered, (M − I0) / (U_inst − I0).
  - The paired difference from the best baseline, with its 95% interval (t, 2 df).
  - Both controls.
- **Generality checks:**
  - A second instruct student (Qwen2.5-3B-Instruct; RL on Qwen models is known to gain even from spurious rewards,
    so the controls matter most there).
  - The puzzle testbed at level A, where answers alone give 1.7% against U's 48.8%: a hard stress test.

## 5. Phase 4: the paper (in parallel from Phase 1)

- **Main story:** how close can an answers-only attacker get to distillation?
  - The method and its share of the teacher's gain.
  - The floor framing: an answer-preserving defense can protect at most U − F_method.
  - The puzzle testbed as the setting where the teacher is needed.
- **Appendix:** levels B–D, the necessity audit, rewriting, the corrupted-answer floor and the ADS calibration.

## 6. Risks

- **The gap may be small for an instruct student** (Phase 0 gate). If so, the main setup needs the full-budget or a
  stronger teacher, which costs about a day.
- **Contamination of GSM8K/MATH training sets** for Llama-3.2-3B-Instruct. Report it; consider a less-seen training
  pool if it distorts results.
- **Novelty:** each component exists. The claim rests on the regime (imperfect teacher, U as target), the measured
  decomposition and the consistency angle. A full literature pass comes before Phase 2.
- **Compute:** a GRPO-style run is about 10–20 GPU-hours per seed. About 9 weeks remain before an ICML deadline in
  late January 2027.

## 7. Next steps (each needs a go-ahead)

1. ~~Close ADS~~ (done 2026-11-19).
2. ~~Literature pass~~ (done: `results/lit_answer_only.md`).
3. ~~Phase 0~~ (done 2026-11-20: m8 §24, `results/phase0_results.md`). Puzzle U_inst seed 123 rerun done (53.1%).
4. MATH recipe check (m8 §25.1): running; read-out into m8, `results/phase0_results.md` and the paper.
5. Phase 1 on the puzzles (m8 §25.2, approved 2026-11-20): B1 and B2 running; then RL baselines B3/B4 (adapt
   `dg_rl_base.py` to the puzzles, smoke test, fix settings in m8), the two controls, and the gap decomposition.
6. Phase 2: the method, chosen by the decomposition.
7. Second student (Qwen2.5-3B-Instruct; download needed) on the puzzles.
8. Paper: fill `icml2027_answers/` as results land; the authors add the missing references listed in its README.
