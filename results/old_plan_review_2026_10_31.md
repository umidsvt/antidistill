# OLD PLAN (superseded 2026-10-31 by `results/plan_2026_11_protection_map.md`): answering the external review of 2026-10-31

Review: `results/external_review_2026_10_31.md` (ICML 2026 form: overall 2 = Reject as submitted, confidence 4;
soundness 2, presentation 2, significance 3, originality 3). Every factual claim in it was checked against
`main.tex`, m8 and the records in `results/` on 2026-10-31. Nothing below has started; each experiment needs its m8
rule and a go-ahead from the authors (CLAUDE.md rules 1–2).

## Bottom line

- The review is careful and mostly right. It re-derives our intervals and margins correctly, and its central
  criticism is one we cannot answer by rewording.
- **Two points change the paper, not just its text.** W2: the ladder keeps the base student even when the attacker
  owns a better model of the same size. W4/Q2: we never show that π and U − D disagree about a real defense. The
  reviewer's own path from 3 to 4–5 runs through these two. Everything else is writing or small runs.
- **Three claims are wrong or partly wrong** (§7): the "60% unaided" in W6 is pass@8, not accuracy; the MATH pilot
  margin of 3.8 is correct (B there is 2.9%, re-measured at 8,192 tokens); Solo finish rates exist in m8.
- **The review missed two things that matter for its own proposal** (§8): Llama-3.2-3B-Instruct has probably seen
  the MATH training problems, and level D fails by construction whenever the teacher is itself an open model.

## 1. Point-by-point check

| Point | Holds? | What I checked | Fix |
| --- | --- | --- | --- |
| W1 incomplete | yes | Table 8 empty; π never computed; 21 pending cells in Tables 3–4; Appendices C–D placeholders; 11 `\todo` markers | E1–E2, writing |
| W2 wrong student | yes | §3 defines R_ℓ as the best student from the attacker's resources; the owned model is such a student. Llama-3.2-3B-Instruct: 81.3% GSM8K, 48.0% MATH, against U 59.0% and 15.0% | D1, E1 |
| W3 LIMO overclaims | yes | LIMO ran only level D, but Contribution 3 says "every level" and Fig. 1 says "above level A". The provenance of LIMO's chains (R1, R1-Distill-32B, QwQ) is not checked yet: verify in Ye et al. | writing |
| W4 thesis not shown | yes | No defense has π and U − D side by side | E2, E3 |
| W5 P3 broken | yes, mostly | LIMO U seeds 62.8 / 58.5 / 48.2. Table 6 is seed 42 only, and Table 5's U − D (13.3, 12.3) and the recovery percentages use that seed's 62.8. The 3-seed hindsight drop is 7.0. Fig. 1a uses U = 56.5, Fig. 4 uses 62.8. The Kim et al. anchor exists but is not in the paper: our grader reproduces their released generations (4/30, 8/30), and hindsight v2 matches their AIME24 cell exactly (1/30). The LIMO cell gives 20.0 vs 26.7, two problems apart | E5, writing |
| W6 margin inflated | partly | B = 0.7% on GSM8K is a chat-format failure (true). The "60% unaided" is at least one of eight attempts at temperature 0.8 (App. B): pass@8, not accuracy, but our wording invited the misreading. Single-attempt few-shot B has not been measured. The MATH pilot margin is right (§7) | D2, E4, writing |
| W7 π under-specified | yes | No sign guarantee when A_ℓ > U_ℓ, no CI, bias of the three lower bounds not discussed | writing, bootstrap code |
| W8 audit thin | yes | Table 2 is from text; PART is re-implemented on LIMO-800 with Qwen2.5-32B-Instruct; Table 5 uses Allouah et al.'s D against their 57.2 | E2, writing |
| W9 recovery framing | yes | Solo (no defended data) is the best "attack" at both sizes: 67.5 vs 64.2/64.3 (32B), 69.5 vs 68.3/60.8 (7B). No Qwen2.5-32B-Instruct Solo for PART | E5, writing |
| W10 contamination | yes, as a risk | AIME25 (30 problems) is the only part of the 600 after Qwen2.5's release; size of the effect unknown | E5c |
| W11 smaller points | yes | P4 runs on hindsight v1 (Fig. 3); P5 uses two defenses, so τ = ±1. B and D also name ladder levels. Article class with numbered natbib; the conclusion is on page 9. `Anonymous [2]` in the first person | writing, E5b |

## 2. Decisions for the authors (before any new run)

**D1. Start the student from the strongest model the attacker owns at the target size (W2, Q1). Recommendation:
yes.** It makes §3 consistent with its own definition of R_ℓ, and it is what a real attacker would do.
- Level A is unchanged (S₀ = the base model), so the GSM8K pass (34.3% vs 59.0%) stands.
- At levels B and C, S₀ = Llama-3.2-3B-Instruct. Then R_B is at least the Instruct model as it is (81.3% GSM8K, 48.0%
  MATH), and U_B is that model fine-tuned on the teacher's outputs. U_B has never been measured, so "fails more
  strongly" needs E1 before the paper can say it.
- Level D: when the teacher is an open model, the attacker can own the teacher itself, so U_D = R_D. Every academic
  setup with an open teacher fails at D by construction; the paper should say so rather than present it as a finding.
- Keep the published setup's student (base, fixed) as a reported row: it is what the defense papers measured.

**D2. Margin and the untrained score (W6, Q4). Recommendation: keep the pre-registered margin as the primary rule
and add two reported quantities.** Changing the rule after seeing results breaks rule 1.
- Statistical necessity, reported separately: the paired 95% interval of U_ℓ − R_ℓ over three seeds excludes 0.
- A sensitivity row with B measured under the few-shot prompt (one greedy attempt).
- Under D1, B matters mainly at level A. A higher few-shot B shrinks the margin there, so the level-A pass holds
  either way.

**D3. Place of the LIMO setup (W3, W5). Recommendation: lead with the GSM8K/MATH setup.** Seeds there cost about
3 GPU-hours (GSM8K) or 7.5 (MATH), against about 11 hours on four GPUs per LIMO student.
- Keep in the main text only LIMO claims with three seeds: necessity at level D (already three seeds), hindsight v2
  and PART as D, Style on PART vs re-voice, and Solo from Qwen2.5-32B-Instruct.
- Move the other single-seed rows to an appendix marked exploratory, with the chance that one run orders each pair
  correctly, Φ(SNR/√2): Mix, Style/Search, the nine deleted doubt words, and the 1.5B/3B students.

**D4. Defenses for π (Q2). Recommendation: PoE and ADS, plus one rewriting defense.** PoE and ADS are already in the
Distillation Game code we run. For the rewriting defense, use PART pass 1 with Qwen2.5-32B-Instruct
(`scripts/22_part_run.sh`, adapted to setup 2). Three defenses make a ranking test meaningful.

**D5. Training problems for the passing setup.** Llama-3.2-3B-Instruct solves 71.5% of MATH's training problems but
48.0% of its test problems (§8). Choose between:
- **MATH train** (recommended for the main run): keeps the setup recognizable, and the paper reports the
  contamination.
- **A pool the student has not seen**: the follow-up, if U_B gains little on MATH train.

## 3. Experiments, in order of value

GPU estimates come from measured durations: GSM8K run about 3 h on one GPU; MATH run about 7.5 h (2026-10-31);
LIMO 7B about 8 h training plus 3 h evaluation on four GPUs. Long-teacher MATH runs are estimated at about one GPU-day
per student and seed.

| # | Experiment | Answers | Size | Estimate |
| --- | --- | --- | --- | --- |
| E0 | Setup-2 rerun (running), then combination students on all seeds (approved in principle in §11, go-ahead pending) | W1, Q4 | 24 runs, then the combinations | rerun finishes about 1 Nov; combinations about 1 day on 4 GPUs |
| E1 | Instruct student in the existing setups, GSM8K and MATH at 1,024 tokens: U_B, R_B (as is; self-trained on released answers), Solo C into Instruct, Mix. Config change only (student model), to check | W2, Q1 | about 10 runs per dataset × 3 seeds | about 1.5 days on 4 GPUs |
| E1b | **Passing-setup candidate:** MATH, Instruct student, teacher at 4,096 or 8,192 tokens, scored at 8,192. Solo C from both Qwen2.5-7B-Instruct and Qwen2.5-Math-7B-Instruct (the strongest open 7B for math, §8). Pilot on seed 123, then three seeds | Q1 | 6 students × 3 seeds + teacher generation | about 5 days on 4 GPUs |
| E2 | **π for three defenses,** three seeds, with bootstrap intervals, in two places: at level A on GSM8K (base student) and in E1b at its passing level. Per defense: passive D plus one adaptive attack (Allouah's adaptive student, or Mix with own data; Style for PART). Pre-registered test: π ranks the defenses differently from U − D in the published setup, or a defense with a large U − D gets a π whose interval contains 0 | W4, Q2 | GSM8K about 54 runs; MATH about 18 | GSM8K about 1 day on 4 GPUs; MATH about 5 days |
| E3 | **Defense applied to a non-teacher writer:** PoE/ADS sampling on Qwen2.5-7B-Instruct (Qwen2.5-3B proxy, same tokenizer), GSM8K, three seeds. If Solo − Solo_defended ≈ U − D, then U − D measures damage to any training data, not protection of the teacher's capability | W4 | 6–9 runs | under 1 day on 4 GPUs |
| E4 | Few-shot untrained score (greedy, one attempt) on GSM8K-Platinum and MATH; statistical necessity from E0's three seeds | W6, Q4 | 2 inference runs | under 3 GPU-hours |
| E5a | LIMO, to three seeds: hindsight v2 (data in git), PART and Style on PART vs re-voice (data regenerated), Solo from Qwen2.5-32B-Instruct (new) | W5, W9, Q3, Q5 | about 14 students | about 6.5 days on one group |
| E5b | P4 on hindsight v2: 10 / 25 / 50% LIMO mixed in, one seed (a stress test) | W11 | 3 students | about 1.5 days on one group |
| E5c | Contamination: Wu et al.'s partial-prompt test on Qwen2.5-7B (MATH500, AMC23, AIME24 against AIME25), and every E5a checkpoint scored on a post-release set (AIME 2025/2026, HMMT Feb 2025/2026; about 120–200 problems; resolves gaps of about 10 points or more only) | W10, Q3 | inference | under 1 day |
| E6 | R1-Distill-Qwen-7B's own pooled-600 accuracy (the PART audit sentence) or drop the sentence | W8 | 1 evaluation | about 3 h on 4 GPUs |

E1, E1b and E2 are the ones that move the score; E3 is cheap and gives the mechanism behind the thesis. E5 is the
largest block and could shrink to E5a's three-seed core if D3 moves the rest to the appendix.

## 4. Writing changes that need no GPU (can start now)

- **Claims (W3, W9).**
  - Scope Contribution 3, the abstract and Fig. 1 to the levels actually run: LIMO at D only, GSM8K/MATH at A–C.
  - Lead with the GSM8K results.
  - Reframe the adaptive-attack results as *bypass* (Solo replaces the defended data) against *inversion* (the
    defended data has value of its own), the vocabulary of the `icml2027/` draft. State that against hindsight
    A_ℓ ≈ R_ℓ.
- **Stability (W5).**
  - Mark every single-seed number and state the Φ(SNR/√2) chance next to claims that rest on 3–7-point gaps.
  - Recompute Table 5 against the 3-seed U where D has three seeds, or say "paired, seed 42".
  - Make Fig. 4 use the same U as Fig. 1a.
  - Add the Kim et al. anchor from `results/REPORT.md` and `results/m3b_hindsight_v2.md`.
  - Add the Solo finish rates from m8 §9.5: 562 / 563 / 560 of 600 answers boxed, vs U 571 / 566 / 571.
- **Margins (W6).**
  - Say that the 60% is pass@8.
  - State the B used for each margin (the MATH pilot: 2.9% at 8,192 tokens).
  - Report statistical and practical necessity separately.
- **π (W7).**
  - Define U_ℓ's recipe list to include every recipe the attacker could apply to undefended outputs, including the
    adaptive ones, and report raw differences when A_ℓ > U_ℓ.
  - Discuss the direction of bias from each lower bound.
  - Bootstrap intervals over problems and seeds.
- **Audit (W8).**
  - Label Table 2 "read from the papers".
  - List our PART re-implementation's differences.
  - Mark Allouah et al.'s numbers in Table 5 as theirs, until E2 replaces them.
- **Notation and form (W11).**
  - Rename ladder levels L0–L3, or rename B to U₀ or "Base".
  - Move to the ICML 2026 style as a layout proxy (`icml2027/icml2026.sty`, as the other draft does) and fit the
    main text into 8 pages.
  - Cite our NeurIPS 2026 paper in the third person.
- **Placeholders (W1).**
  - Fill Appendix A (the full hindsight prompt and PART pass 2), Appendix C (data checks, from the m4/m6 records)
    and Appendix D (the rules and deviations table, from m8).
  - Fill in the compute totals.
- **Llama-3.2-3B-Instruct (§8).**
  - Report its training-problem accuracy next to its test accuracy at main.tex:359 and :407, once the GSM8K rerun
    gives the GSM8K figure.
  - Recorded in m8 first (agreed with the user 2026-10-31).
- **References for the authors to add** (agents do not edit `references.bib`):
  - Gudibande et al. 2023 (*The False Promise of Imitating Proprietary LLMs*).
  - Das, Zhang and Tramèr 2024 (blind baselines for membership inference).
  - Wu et al. 2025 (arXiv 2507.10532).
  - STaR (Zelikman et al. 2022), Wei et al. 2022, DeepMath-103K (He et al. 2025).

## 5. Schedule

The ICML 2027 dates are not announced; this assumes a late-January deadline like ICML 2026's. Group 0–3 runs the
setup-2 work and group 4–7 the LIMO work (more than 4 GPUs at once: agreed with the user 2026-10-30).

| Weeks | GPUs 0–3 | GPUs 4–7 | Writing |
| --- | --- | --- | --- |
| 2–8 Nov | E0 ends; E4; E1 | E6; E5c partial-prompt test | D1–D5 decided; m8 rules for E1–E5; §4 pass 1 |
| 9–22 Nov | E1b pilot; E2 at level A on GSM8K | E5a (PART data, seeds) | |
| 23 Nov – 13 Dec | E1b three seeds; E3 | E5a continued | |
| 14 Dec – 3 Jan | E2 in the passing setup | E5b; E5c scoring | Table 8, Table 5 |
| 4–17 Jan | reruns, buffer | buffer | restructure around D1; figures; 8 pages |
| 18 Jan – deadline | | | internal review, checklist, release |

## 6. Expected score, by the review's own mapping

| State | Score (1–6) |
| --- | --- |
| As submitted | 2, Reject |
| Pending cells filled, nothing else | 3, Weak reject |
| D1 adopted, a passing setup with π for at least three defenses over three seeds that disagrees with U − D somewhere, stability fixed | 4–5, Weak accept to Accept |

The risk sits in E1b. If the Instruct student gains little from the teacher on MATH, no setup passes above level A.
The paper then still has D1, E2 at level A and E3, which show the thesis on GSM8K at level A only. That is a narrower
threat model, but it is a real demonstration, which the reviewer said is the single biggest lever.

## 7. Where the review is wrong or overstated

- **W6, "solves 60% of the training problems unaided".** Appendix B: eight attempts per problem at temperature 0.8;
  3,123–3,168 of 5,238 problems have at least one attempt that reaches the answer. That is pass@8 on training
  problems, not the untrained model's accuracy. The single-attempt few-shot score is unmeasured (E4).
- **W6, the MATH pilot margin.** m8 §10.1 re-measured the untrained student at 8,192 tokens: 2.9%, not 0.8%.
  (18.1 − 2.9)/4 = 3.8, as reported. The paper does not say which B it used, which is the real error.
- **W3, "the finish rate for the Solo students is never reported".** It is in m8 §9.5 but not in the paper:
  Solo boxes 562 / 563 / 560 answers of 600 against U's 571 / 566 / 571. Solo wins while finishing slightly less often,
  so its lead is not a finishing effect.
- **W5, "never compared against Kim et al.'s published numbers".** We match their hindsight AIME24 cell exactly and
  re-grade their released generations exactly. Their own numbers are AIME24 only (30 problems), so the anchor is weak
  by our P3, and the paper should say both.
- **W2 calls the framework "incoherent".** That overstates it. Auditing the published setups with their own fixed
  student is legitimate, and every negative verdict survives the reviewer's reading. What is inconsistent is our §3
  definition of R_ℓ, which D1 fixes.
- **W8 and W2 pull against each other.** W8 rejects comparing a model's own accuracy with a student's, but under
  W2's formulation the owned model's own accuracy *is* a lower bound on R_ℓ. The PART sentence then fails only
  because it rests on 30 AIME24 problems and on a different decoding.

## 8. What the review missed

- **Llama-3.2-3B-Instruct has probably trained on MATH's training problems.** Rerun 2026-10-31, per-problem files:
  it solves 72.2% and 70.9% of the 5,000 training problems on seeds 123–124, against 48.0% of the test problems.
  The other writers show no such gap: R1-Distill-Qwen-7B 62.0% vs 63.1%; Qwen2.5-7B-Instruct 76.0% vs 73.1%.
  - This weakens our sentence that this model trains an equally good student "although it solves 15 points fewer
    test problems than the teacher": on the problems the student trains on, it is about 9 points *ahead*.
  - It also affects the reviewer's proposed passing setup, whose student would train on problems it has seen (D5).
  - GSM8K is checked when the rerun reaches it.
- **Level D fails by construction with an open teacher**, since the attacker can own the teacher (D1).
- **Level C's Solo uses Qwen2.5-7B-Instruct, not the strongest open 7B for math** (Qwen2.5-Math-7B-Instruct). This is
  fine for the failure claims, which need only a lower bound on R_C. It is required for any claim that a setup
  passes at level C (E1b).
