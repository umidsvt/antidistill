# External review of the audit draft (2026-10-31)

An independent AI agent's ICML-style review of `antidistillation-overleaf/icml2027_audit/main.pdf` as compiled on
2026-10-31 (branch `audit-sync-m8-2026-10-30`). Recorded verbatim; only the agent's opening and closing chat lines
are removed. Point-by-point check: `results/old_plan_review_2026_10_31.md` (§1, §7, §8 still apply); current plan: `results/plan_2026_11_protection_map.md`.

---

I scored the draft as if it had been submitted to ICML in its current state, with the pending cells and placeholders still in. I used the ICML 2026 main-track form, which rates Overall Recommendation from 1 to 6. At the end I say what score I'd expect once the pending work is done.

| Field | Score |
|---|---|
| Soundness | 2 (fair) |
| Presentation | 2 (fair) |
| Significance | 3 (good) |
| Originality | 3 (good) |
| **Overall Recommendation** | **2: Reject** (as submitted) |
| Confidence | 4 |

---

## Summary

The paper argues that anti-distillation defenses are evaluated with the wrong reference point. The standard metric is U − D: a student trained on undefended teacher outputs minus a student trained on defended ones. That metric assumes the teacher is the attacker's only route to a good student. The authors propose:
- six validity properties, with necessity as the central one;
- an attacker ladder (A: owns only the student; B: owns the student's instruct sibling; C: owns an open model of the teacher's size; D: owns an open reasoning model);
- a protection score π_ℓ = (U_ℓ − A_ℓ)/(U_ℓ − R_ℓ), which measures a defense against the best student the attacker can build without the teacher.

They audit the two setups used by most published defenses. In the GSM8K setup, a student trained on solutions from Llama-3.2-3B-Instruct, a weaker solver than the teacher, matches U (59.4 vs. 59.0). In the LIMO setup, a student trained on R1-Distill-Qwen-7B solutions beats U (66.2 vs. 56.5). Only an attacker at level A needs the teacher. Adaptive attacks against hindsight rewriting and PART, plus several analyses of failure modes (seeds, small test sets, finishing rate, a pipeline bug), round out the paper. The constructive part, a setup that passes the checks with π computed for real defenses, is still pending.

## Strengths

- **S1. The paper asks the right question.** Requiring a no-teacher baseline is simple, clearly motivated and consequential. It plays the same role for distillation defenses that "blind baselines" play for membership inference. If it holds up, it changes how this subfield should report results.
- **S2. The GSM8K evidence is hard to dismiss.** The authors reproduce the published numbers closely (59.0 vs. 57.2; 0.7 vs. 0.76; MATH 15.0 vs. 15.2). The level-B result then stands on its own: the student's own instruct sibling trains an equally good student while solving fewer test problems than the teacher (81.3 vs. 87.2).
- **S3. The diagnostic work is useful in its own right:**
  - The 1,024-token limit makes the reasoning teacher the weaker solver on MATH (63% vs. 73% for Qwen2.5-7B-Instruct).
  - The paper reports accuracy among finished answers to separate the finishing confound.
  - The published hindsight pipeline leaks the scratchpad: the defended student goes from +6.7 over the untrained model to −0.3 once fixed.
- **S4. The statistics are carefully done where they are done.** I re-derived the n=3 t-intervals (+9.7 [2.7, 16.8]; +6.7 [−11.9, 25.3]), the necessity margins and the recovery percentages, and they are correct. Writing each decision rule down before the result is good practice.
- **S5. The limitations section is honest.**

## Weaknesses

These are ordered by how much they drive the score.

**W1. The submission is incomplete, and as submitted this decides the score.**
- Table 8 is empty.
- π_ℓ is never computed for any defense at a level where it is defined.
- Rows of Tables 3 and 4 are pending (PoE-defended; STaR on MATH; level-C Mix).
- Appendices C and D are placeholders.
- There are "[cite …]", "[Insert the full prompt]" and "[Add the passing-setup result]" markers.

The net effect: the paper proposes a metric it never applies, and Contribution 6 rests on no evidence.

**W2. The ladder fixes the wrong student, which is the main conceptual issue.**
- At level B on GSM8K, the attacker owns Llama-3.2-3B-Instruct. That model scores 81.3% by itself, 22 points above U and the same size as the student.
- On MATH, the gap is 48.0 vs. 15.0.
- At LIMO level D, the attacker owns R1-Distill-Qwen-7B, again the student's size and far above U.

The paper defines R_ℓ as the best student buildable from the attacker's resources. Under that definition R_B is at least 81.3, not 59.4. Necessity therefore fails more strongly than reported, but the framework is also incoherent: no attacker would distill into a base checkpoint that is worse than a model they already own.

The cleaner formulation sets the student's initialization to the strongest model the attacker owns at the target size. U_ℓ is then that model distilled on the teacher's outputs, and necessity asks whether the teacher improves the best model the attacker could otherwise field. This is probably also where a passing setup lies. For example: Llama-3.2-3B-Instruct (48% on MATH) as the student, the teacher at full token budget (79%), and a longer scoring budget.

**W3. The level-D LIMO result is partly circular, and the claims go beyond it.**
- LIMO's reasoning chains were built by generating solutions with three strong reasoners: DeepSeek-R1, DeepSeek-R1-Distill-Qwen-32B, and QwQ-32B. The level-D attacker owns R1-Distill-Qwen 7B or 32B, which are distilled from R1. In effect, the attacker already holds a distilled copy of the teacher's family, so the attack has already succeeded upstream.
- R1-Distill-Qwen-7B is built on Qwen2.5-Math-7B, so its solutions are close to the student's own distribution. That may explain why solutions that are only 37% correct beat LIMO's curated ones. The finish rate for the Solo students is never reported, even though P6 asks for it.
- For LIMO, only level D was run. Contribution 3 ("fail necessity at every level where the attacker owns an open model") and the Figure 1 caption ("above level A") therefore overclaim. The GSM8K results at levels B and C are the clean evidence, and the paper should lead with them.

**W4. The central thesis is asserted, not demonstrated.**

A defense author will answer: "The open teacher is a cheap stand-in for a frontier teacher. U − D measures how much the transformation destroys the distillation signal, and that transfers." The paper has the material to rebut this but never puts it together. Students learn from 37%-correct Solo data; deleting nine doubt words costs 7 points; the defenses mostly remove doubt markers. Together these suggest that in a setup failing necessity, U − D measures the destruction of generic style rather than of teacher-specific capability.

The experiment that would show this: compare U − D in a setup that fails necessity against π in one that passes, and show the verdict or ranking changes for at least one defense. Without it, the contribution shrinks to "the toy setups are unrealistic." That is still useful, but it is partly anticipated by Javaheri et al.'s observation that RL from the untrained student often beats distillation.

**W5. The paper breaks its own stability property (P3).**
- U in the LIMO setup spans 62.8/58.5/48.2 across seeds (standard deviation ≈ 7.5); one seed is below the untrained model.
- All of Table 6 is a single seed, yet its conclusions rest on gaps of 3–7 points: re-voice vs. Style-on-PART (6.3 points); deleting doubt words (−7.0); Style vs. Search at 7B. By the paper's own formula Φ(SNR/√2), a 6.3-point gap is ordered correctly by one run only about 72% of the time.
- Table 5's U − D values (13.3, 12.3) and every recovery percentage use the best seed (62.8). Against the 3-seed mean, hindsight's drop is 7.0 points.
- Figures 1a and 4 show different values of U for the same setup.
- Unlike the GSM8K setup, the LIMO setup is never compared against Kim et al.'s published numbers. Given the seed instability, the training recipe needs that anchor.

**W6. The necessity margin is inflated by a format artifact.**
- B = 0.7% on GSM8K is a formatting failure under the chat prompt. With 8-shot prompting, the same model solves 60% of the training problems unaided.
- The margin, a quarter of U_ℓ − B, is therefore about 15 points. The level-B verdict of "fails" (+3.0 on seed 123) depends entirely on that number, while seed noise on GSM8K is about 1 point.
- Fixes: measure B under the best prompting; report statistical necessity (the CI of U_ℓ − R_ℓ excludes 0) separately from practical necessity (an effect-size threshold).
- Separately, the MATH pilot's margin of 3.8 doesn't match (18.1 − 0.8)/4 = 4.3. Please state which B was used.

**W7. The protection score is under-specified.**
- U_ℓ, A_ℓ and R_ℓ are each lower bounds, so the bias in π has no fixed sign. A weak estimate of A_ℓ inflates π, and a weak estimate of R_ℓ deflates it. The claim that π cannot credit protection the attacker never needed holds only if R_ℓ is well estimated.
- π ≥ 0 is not guaranteed; the paper's own hindsight attacks exceed U. Requiring U_ℓ's recipe list to include every adaptive recipe applied to undefended data fixes this, because an attacker holding undefended outputs can simulate the defense. That gives 0 ≤ π ≤ 1 by construction.
- π is a ratio of noisy differences and needs bootstrap CIs, especially when U_ℓ ≈ R_ℓ.

**W8. The audit is thinner than the text says.**
- Table 2 is read from the papers' text; only two setups are actually run.
- PART's real setup uses R1 solutions on Bespoke-17k and OpenThoughts-114k. What the paper tests is a re-implementation on LIMO-800 with a different rewriter (Qwen2.5-32B-Instruct instead of GPT-4o).
- DOGe and TARS only loosely match "the GSM8K setup."
- The PART audit sentence relies on AIME24 (30 problems), which §7.4 itself calls unresolvable. It also compares a model's own accuracy with a student's accuracy, which is not a test of necessity.
- No sampling defense is run by the authors. Table 5 mixes Allouah et al.'s numbers, measured against their 57.2, with the authors' own runs.

**W9. The adaptive-attack results are framed as recovery when the defended data adds nothing.**
- Against hindsight, Solo, which uses no defended data, is the best "attack" at both sizes (32B: 67.5 vs. 64.2/64.3; 7B: 69.5 vs. 68.3/60.8). So A_ℓ ≈ R_ℓ: the recovery comes from the attacker's own model, not from the defended data. The abstract and Contribution 4 should say so.
- Against PART, no Solo result with Qwen2.5-32B-Instruct is reported, so the 68% recovery cannot be attributed to the defended data either.

**W10. Contamination is an unaddressed alternative explanation.**

The LIMO evaluation uses a Qwen2.5 student on MATH500, AMC and AIME24. Wu et al. report strong signs of data contamination in the Qwen2.5 series models when evaluated on commonly used benchmarks, such as MATH-500, AMC, and AIME2024. That is a competing explanation for "any fine-tuning data helps, correct or not." The LIMO conclusions need a post-cutoff or synthetic test set that is large enough to resolve the effects.

**W11. Smaller points:**
- P4 is run on hindsight v1, the buggy pipeline the paper itself diagnoses. It should be rerun on v2.
- P5 with two defenses means Kendall's τ can only be ±1, which makes for a weak test.
- A "validity property" whose violations are expected (P4) is really a diagnostic.
- Notation clashes: B is both the untrained accuracy and a ladder level, and D is both the defended accuracy and a ladder level ("R_D = 66.2"). Rename the levels, e.g. L0–L3.
- The draft is not in the ICML template (it uses numbered citations), and the main text runs onto page 9.
- The self-citation "Anonymous [2]" is to a published NeurIPS 2026 paper. Cite it in the third person; the anonymous citation plus "we follow the same template" is more revealing, not less.

**Missing references:**
- Gudibande et al. (2023), *The False Promise of Imitating Proprietary LLMs*: imitation copies style more than capability, which is the likely mechanism behind the necessity failures.
- Das, Zhang & Tramèr (2024), on blind baselines for membership inference: the closest methodological analogue to necessity.
- Wu et al. (2025), on contamination.
- The STaR, Wei et al. and DeepMath placeholders still need filling.

## Key questions for authors

1. **What happens if the student is the strongest model the attacker owns at the target size?** For example, Llama-3.2-3B-Instruct as the student with a full-budget teacher on MATH. If necessity passes there and π can be computed, my main conceptual objection (W2) goes away and much of W1 with it.
2. **In a setup that passes necessity, does π change the verdict or ranking of at least one defense relative to its U − D in the standard setups?** A yes would make the thesis a demonstrated evaluation failure, and this is the single biggest lever on my score.
3. **Do the LIMO conclusions hold under P3 and W10?** Specifically: three seeds for the Table 6 rows used in claims; finish rates for the Solo students; agreement with Kim et al.'s published U; a clean benchmark. If the 7B Solo > U result survives on a clean set, W3 and W10 weaken considerably.
4. **With B measured under few-shot prompting and a purely statistical criterion, does GSM8K at level B still fail necessity?** If the +3.0 is statistically real, the paper's claim "fails even with the combination student" needs softening.
5. **What does Solo with Qwen2.5-32B-Instruct score?** Without it, the PART attack results cannot be interpreted.

## Limitations

Partly adequate. The section misses three things: contamination, the degenerate threat model at levels B and D (W2), and the fact that the LIMO-level-D evidence uses models distilled from the teacher's own family.

## Ethical concerns

None. The impact statement is reasonable.

---

## Where the score goes from here

- **As submitted: 2 (Reject).** Empty result tables and placeholders, and the proposed score is never applied.
- **Pending cells filled as planned, nothing else changed: 3 (Weak reject).** The GSM8K finding is solid, but W2–W5 would still each draw a reviewer comment, and "the toy setup is unrealistic" alone is a borderline ICML contribution.
- **W2 reframed, plus a passing setup with π for at least three defenses over three seeds that disagrees with U − D somewhere (Q1–Q2), plus the stability problems fixed: 4–5 (Weak accept to Accept).** At that point it is a methodology paper that should change practice in this subfield.

If you only have time for one thing before the deadline, do Q1 and Q2 together. Making the student the strongest model the attacker owns at that size is both the fix for the incoherent threat model and probably your fastest route to a setup that passes.


Sources:
- [ICML 2026 Reviewer Instructions](https://icml.cc/Conferences/2026/ReviewerInstructions)
- [LIMO: Less is More for Reasoning (summary of data pipeline)](https://summary-of-some-paper-in-cuda.readthedocs.io/en/latest/training-methods/LIMO-LessisMoreforReasoning/)
- [Wu et al., Reasoning or Memorization? Unreliable Results of RL Due to Data Contamination](https://arxiv.org/html/2507.10532v3)