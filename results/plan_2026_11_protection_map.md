# Plan: a protection map for anti-distillation defenses (proposed 2026-10-31, for the meeting of 1 November)

**Status: proposal.** Nothing in this plan has started. Each experiment needs a go-ahead from the authors and its
decision rule written in m8 before its first result (CLAUDE.md rules 1–2). The draft rules in §7 are starting
points for those m8 entries, not fixed rules yet. Older plan: `results/old_plan_review_2026_10_31.md` (superseded;
its point-by-point check of the review, §1, §7 and §8, still holds).

---

## 0. Summary in five sentences

1. An outside review scored the current draft 2 of 6 (reject) and said it could reach 4–5 if we (a) fix the attacker
   ladder and (b) show, for real defenses, that the protection score and the usual drop U − D disagree.
2. Finishing the audit as planned would give a paper at roughly ICML's base rate (about 25–35%; acceptance is about
   27%), because the defense authors can answer "our open teacher is only a stand-in" and we have no setting where a
   defense can show protection.
3. The missing piece is a **setting where the teacher is actually needed**. Without one, a defense cannot show
   protection and an attack cannot show it broke anything, and none of the papers we audit has one.
4. Proposal: keep the audit as Part 1, then (Part 2) map where protection is measurable, as a function of the gap
   between the teacher and the attacker's own best model. Then (Part 3) give the first valid protection scores for
   four to five defenses there, under the strongest known adaptive attacks.
5. The first two weeks decide the story cheaply, and every outcome is publishable:
   - defenses fail even where protection is measurable, or
   - some defense protects, which would be the first valid evidence of protection.

## 1. Why change course

### 1.1 What the review says

The review used the ICML 2026 form: overall 2 = Reject, confidence 4; soundness 2, presentation 2, significance 3,
originality 3. It gave a path:
- **As submitted:** 2.
- **Pending cells filled:** 3 (weak reject).
- **4–5 (weak accept to accept):** the ladder fixed (W2), plus a passing setup with π for three or more defenses over
  three seeds that disagrees with U − D somewhere (W4/Q2), plus the stability problems fixed.

Our check of each point: `results/old_plan_review_2026_10_31.md` §1.

### 1.2 Why the audit alone is capped (my assessment)

1. **The defense authors' reply is ready-made.** "The open 7B teacher stands in for a frontier teacher; U − D
   measures how much the transformation damages the distillation signal, and that transfers." Our necessity failures
   show the stand-in is poor. They do not show that U − D is wrong about real protection.
2. **The positive part may not exist above level A.** With an open teacher, levels C and D fail by construction:
   the attacker can own an equally good open model, or the teacher itself. The instruct-student MATH setup from the
   old plan has perhaps a 1-in-3 chance of passing. Its student has also probably seen the MATH training problems:
   Llama-3.2-3B-Instruct solves 71.5% of them (seeds 123–124) against 48.0% of the test problems.
3. **Significance and novelty.** About ten papers in the subfield; the same template as our NeurIPS 2026 paper.

### 1.3 What the closest prior work already covers (checked in the PDFs, 2026-10-31)

- **Javaheri et al. (arXiv 2609.35699, 28 Sep 2026).**
  - RL after distillation breaks ADS (Qwen2.5-0.5B student).
  - A weak off-the-shelf model can expand API trace *summaries* plus final answers into full traces. After RL,
    these train as well as the full traces.
  - Their conclusion: "any distillation defense that leaks sufficient information to reconstruct approximate
    reasoning traces is likely ineffective".
  - **This takes the novelty of a "one general attack" paper,** the direction I recommended earlier today before
    reading their Section 5. The attack below is therefore a tool, not our headline.
- **Our hook against Javaheri et al.** Their own numbers show RL from the untrained student beating distillation in
  4 of their 5 setups: Qwen2.5-0.5B 40.3% vs 35.8%; 1.5B 64.6% vs 61.1%; 3B 72.6% vs 71.6%; Llama-3.2-3B 24.3%
  vs 19.8%; the exception is Qwen2.5-Math-7B, 79.3% vs 84.7%. So their setups also fail necessity, at level A.
  A defense "broken" in a setup where the attacker never needed the teacher was never shown to protect anything.
  The necessity problem affects attack papers as much as defense papers.
- **Allouah et al. (the Distillation Game).** Argues for adaptive students (reweighting the training examples) and
  reports them recovering most of the drop on GSM8K. Undefended 57.2%; ADS passive 34.3% → adaptive 51.5% (75% of
  the gap); PoE passive 39.3% → adaptive 49.5% (57%).
- **Still open, and ours:**
  - Necessity as a precondition, applied across the literature.
  - The map of where protection is measurable.
  - Rewriting defenses (hindsight, PART) evaluated in a valid setting.
  - Valid protection scores.

### 1.4 The four directions considered

| Direction | What it is | Odds (judgment) | Why not alone |
| --- | --- | --- | --- |
| 1. Flaw + one general attack | Necessity, then one attack that removes protection everywhere | 30–40% | Javaheri et al. §5 has the attack's core idea |
| 2. Protection vs. capability gap | Where protection is measurable; scores there | 35–45% | Needs adaptive attacks to be convincing (from 1) |
| 3. Testbed by design | A teacher with a private skill (TOFU-style), so necessity holds at every level | 30–45% | Slowest (4–6 weeks for the teacher); realism objection; fits NeurIPS Datasets & Benchmarks better |
| 4. Detection, not prevention | Detect that a student was trained on our traces; Solo students as hard negatives | 25–35% | Prior work on radioactive watermarks and fingerprinting to check first |
| New prevention defense | — | 10–20% | Our own audit predicts it cannot be shown to work in current setups |

**Recommendation: 2 as the core, with the adaptive attacks of 1 inside it. Odds about 35–50%.** That is above the
base rate but not "very likely". In the NeurIPS 2014 and 2021 consistency experiments, about half of the accepted
papers were rejected by a second committee. Directions 3 and 4 are follow-ups (NeurIPS 2027).

## 2. The paper we would submit

**Working title:** *Is There Anything to Protect? Measuring Anti-Distillation Protection Where It Can Exist.*

**One-sentence claim:** in the setups the literature uses, the usual drop U − D cannot show protection, because the
attacker does not need the teacher. Protection is measurable only when the teacher is clearly better than everything
the attacker owns. Where it is measurable, we report the first valid protection scores for four to five defenses
under the strongest known adaptive attacks.

**Contributions.**
- **C1. Necessity and the protection score,** with the definitions fixed per the review:
  - the student starts from the strongest model the attacker owns at the target size;
  - the ladder is indexed by the attacker's best model;
  - π comes with intervals.
- **C2. The audit.** The setups of seven defense papers, one evaluation study and one attack paper (Javaheri et al.)
  fail necessity. Mostly existing evidence, plus Javaheri et al.'s own numbers.
- **C3. The protection map.** Necessity as a function of the gap between the teacher and the attacker's best writer
  model (MATH), plus level A (GSM8K, MATH).
- **C4. Valid protection scores** for ADS, PoE, hindsight, PART (and DOGe or TARS if their code runs), each under
  passive and adaptive attackers. Compared with the U − D each defense gets in the published setup.
- **C5. A reporting checklist and code** for every check.

**Page budget** (8 pages, ICML style):

| Section | Pages | Content |
| --- | --- | --- |
| 1 Introduction | 1.0 | the problem, the map figure, contributions |
| 2 Necessity, ladder, protection score | 1.0 | definitions (revised), properties P1–P6 |
| 3 Published setups fail necessity | 1.25 | GSM8K, MATH, LIMO (three-seed claims only), Javaheri et al.'s setups, survey table |
| 4 Where protection is measurable | 1.25 | level A; the gap curve |
| 5 Protection where it is measurable | 1.5 | π for each defense and attack; U − D vs π |
| 6 Failure modes | 0.75 | pipeline bug, AIME-only, seeds, finishing (trimmed) |
| 7 Checklist, limitations, conclusion | 0.75 | |
| Appendix | — | LIMO adaptive attacks (single seed, exploratory), prompts, data checks, rules and deviations, compute |

**What moves or goes.**
- **To the appendix:** the LIMO attack matrix (single seed), the doubt-word analysis, the 1.5B/3B students, the
  leakage curve.
- **Dropped:** the PART AIME24 sentence (30 problems, different decoding).
- **Optional:** the instruct-student MATH setup (old plan E1/E1b). It is replaced by the gap curve, which asks the same
  question more generally.

## 3. Definitions (revised; to be written into the paper and m8)

- **Teacher T**, student start S₀, training problems, test set: as now.
- **Attacker's resources at point w:** the attacker owns writer model w (an open model) and the student's base model.
  Level A means the attacker owns only S₀.
- **R_w,** the best no-teacher student, is the best of:
  - Solo from w (the student trained on w's own solutions);
  - w itself, when w has the student's size (the review's W2);
  - level-A self-training.
- **U_w:** the best student with the teacher's undefended outputs added: max(U, Mix(Solo_w + teacher)).
- **A_w:** the best student with the defended outputs added: max(D, adaptive attacks run with w, Mix(Solo_w +
  defended)).
- **Gap_w:** the teacher's accuracy minus w's accuracy, both **on the training problems.** Training problems, not
  test, because that is what the student learns from, and test accuracy can mislead (Llama-3.2-3B-Instruct: 71.5% on
  MATH training problems vs 48.0% on test).
- **Necessity at w (P2):** U_w − R_w exceeds a quarter of U_w − B (practical) and its paired 95% interval excludes 0
  (statistical). Both reported separately.
- **Protection π_w** = (U_w − A_w) / (U_w − R_w), defined only where P2 passes, reported with a 95% bootstrap interval
  over problems and seeds. When A_w > U_w, U_w is an underestimate: report the raw differences and do not clip
  silently.
- **Recovery** = (A − D) / (U − D): the share of the drop an attack wins back.
- **Inversion vs bypass:**
  - An attack *inverts* the defense if A_w − R_w exceeds its interval: the defended data adds something beyond the
    attacker's own.
  - Otherwise it *bypasses* it: the attacker's own data does the work, as Solo does against hindsight on LIMO.

## 4. The settings

### 4.1 Level A on GSM8K (exists; passes)

- **What exists:** the Distillation Game setup. Teacher R1-Distill-Qwen-7B at 1,024 tokens, student Llama-3.2-3B with
  LoRA r128, 5,238 training problems, GSM8K-Platinum test (1,209).
- **Result so far:**
  - U = 59.0%; level-A self-training (few-shot STaR on the teacher's answers) = 34.3%.
  - U − R_A = 24.2–25.2 points on every seed, against a margin of 14.6, so necessity passes.
- **Rough level-A protection already implied:** Allouah et al.'s defended and adaptive students against our R_A give
  π_A ≈ 0.25 (ADS) and 0.34 (PoE). This mixes their U = 57.2 with our R_A, so it is a sketch, not a result.

### 4.2 Level A on MATH (to run)

- **Expected:** a large gap. The untrained student scores 0.8%; U = 15.0%.
- **Prerequisite:** self-training on MATH, using the same recipe as GSM8K with a MATH few-shot prompt.

### 4.3 The gap curve on MATH (to run)

- **Teacher:** R1-Distill-Qwen-7B, the teacher of the five papers, with its token limit raised so it is the strongest
  solver. 4,096 new tokens; scoring at 8,192.
  - At 1,024 tokens it solves only 62.0% of the training problems, below most writers. That alone is why the
    published MATH setup fails.
  - At 8,192 tokens it scores 79.0% on test, but its student finishes only 41% of answers (the pilot of 27 October).
  - 4,096 tokens is the compromise; the pilot in N5a checks it.
- **Student:** Llama-3.2-3B, as published.
- **Writer ladder w** (published MATH scores, to be re-measured on our training problems):
  - Llama-3.2-1B-Instruct (about 30%)
  - Qwen2.5-1.5B-Instruct
  - Llama-3.2-3B-Instruct (48.0% test, 71.5% training, measured)
  - Qwen2.5-3B-Instruct
  - Qwen2.5-7B-Instruct (73.1% test, 76.0% training, measured)
  - Qwen2.5-Math-7B-Instruct (about 84%; the strongest open 7B for math)
- **Expected shape:** necessity passes for weak writers and fails once a writer comes close to the teacher on the
  training problems.
  - **Where it passes,** we measure protection (§5, N6).
  - **The crossover point** says how large the gap must be before any defense can be evaluated at all. That is the
    practical rule for defense authors.

## 5. The attacks (tools, not the headline)

Every defense in a valid setting gets the passive attacker and the strongest adaptive attackers allowed at that level.

| Attack | How it works | Allowed at | Code |
| --- | --- | --- | --- |
| Passive (D) | train on the defended outputs | all | exists |
| **Rationalize** | the attacker's own model solves each problem with the defended solution shown as a reference; keep attempts that reach the released answer; retry unsolved ones with the answer as a hint; train on its own text only, with the hint removed from the prompt | A (generator = the student, few-shot); w (generator = writer w) | small change to `scripts/dg_star.py` (hint = defended solution instead of answer) |
| Adaptive student (Allouah et al.) | reweight the training examples by holdout gradients | all | in the Distillation Game code |
| RL after distillation (Javaheri et al.) | GRPO starting from the passive D student, reward = the released answer | all | our GRPO v3 recipe; starting from a student that already formats answers |
| Mix with own data | defended + the attacker's Solo solutions | w | exists |

- **Why rationalize should work against sampling defenses.** ADS and PoE change which tokens the teacher picks, so
  the text is a poor training target. Regenerating the text with the attacker's model removes those token choices and
  keeps the solution path.
- **Why it should work against rewriting defenses.** Hindsight and PART remove behaviours: doubt, checking,
  self-talk. Regeneration restores the generator's own way of reasoning, as our LIMO Search attack did (68.3% vs
  undefended 62.8%, one seed).
- **Controls, which every report includes:**
  - rationalize with the *undefended* teacher solution as the reference: how much rationalizing costs, against
    imitating directly;
  - answer-only self-training (R_A): to beat it, the attack must use the defended text (inversion), not just the
    answers.

## 6. Experiments, in order

**Duration basis (measured):** a GSM8K run takes about 3 h on one GPU (paper, Appendix E). A MATH run at 1,024
tokens takes 7.5 h on one GPU (2026-10-31, 00:12 → 07:43). MATH runs at 4,096 tokens are estimated at about 15 h.

| # | Experiment | Size | GPU-hours (estimate) | Gate |
| --- | --- | --- | --- | --- |
| N0 | Finish the setup-2 rerun (running) and record it under m8 §12 | 20 jobs left | (running) | the reproduction rule |
| N0b | Combination students (Mix) on the regenerated files, seeds 123–125, GSM8K and MATH at levels B and C: Part 1's pending cells in Tables 3–4 (old m8 §11) | 12 runs | about 60 | N0 within tolerance |
| N1 | Reproduce ADS and PoE in the published GSM8K setup (Allouah et al.'s settings): passive D, 3 seeds → U − D | 6 runs | about 40 | our drop within 3 points of theirs (22.9 ADS, 18.0 PoE) |
| N2 | **Level-A attack pilot, GSM8K, seed 123:** rationalize and the adaptive student against ADS and PoE; control rationalize-U | 5 runs | about 20 | **G1** |
| N3 | Rewriting defenses in the GSM8K setup: hindsight (R1-Distill-Qwen-32B rewriter, v2 pipeline) and PART (Qwen2.5-32B-Instruct) on the teacher's solutions; data checks; then D, rationalize, adaptive, 3 seeds | 2 datasets + 18 runs | about 80 | data checks pass |
| N4 | Seeds 124–125 of N2; RL after distillation on the strongest-surviving defense; π_A for every defense with intervals | about 12 runs | about 70 | — |
| N5a | **MATH pilot, seed 123:** teacher at 4,096 tokens; U; Solo from the weakest and the strongest writer | 4 runs | about 60 | **G2** |
| N5 | The gap curve: teacher and U, the six writers' Solo, self-training at level A; 3 seeds; Mix where plain U − Solo is within the margin | about 25 runs | about 380 | **G3** |
| N6 | Protection on the curve: at the strongest writer where necessity passes, and at level A on MATH: 4 defenses × (D, rationalize, adaptive), 3 seeds | about 48 runs | about 720 (halve by running one point only) | — |
| N7 | U − D vs π: rankings and the pre-registered comparison (§7); figures | analysis | 0 | — |
| N8 | Optional and cheap: few-shot untrained score (old E4); a defense applied to a non-teacher writer (old E3) | 2 + 6 runs | about 25 | — |

**Total:** about 1,450 GPU-hours, roughly 7–8 days of all eight GPUs, or about 3 weeks of one 4-GPU group. Spread over
7 weeks this leaves room for labmates' jobs and reruns. DOGe or TARS is added in N3 only if public code runs in our
setup within 3 days of trying.

### Decision gates (outcomes written in advance)

- **G1, after N2 (about 14 November).** The share of the drop the best attack recovers, seed 123:
  - **≥ 90% against both defenses:** Part 3's story is "defenses fail even where protection is measurable". Continue
    N3–N4.
  - **Between 50% and 90%:** run one stronger variant, fixed in advance: two rounds of rationalize, the round-1
    student as the generator. Then read again.
  - **Below 50% for a defense:** that defense protects at level A, the first valid protection result. Continue N3–N4,
    and N6 tests whether it holds at larger attacker resources.
- **G2, after N5a (about 14 November).**
  - **Continue the curve** only if the teacher at 4,096 tokens solves more training problems than every writer except
    Qwen2.5-Math-7B-Instruct, and its U student finishes at least 80% of its answers at 8,192 tokens.
  - **Otherwise run the curve at 1,024 tokens:** the published setting, with a shorter ladder.
- **G3, after N5 (about 6 December).**
  - **At least one writer passes necessity:** run N6 there and at level A.
  - **None passes:** run N6 at level A only. The curve then shows that necessity fails at every writer from the weakest
    one up, which is a result in itself.

## 7. Draft rules (to be fixed in m8 before each result)

- **Seeds** 123–125 for everything reported; pilots on seed 123 only, never read as results.
- **Defense reproduction (N1):** our U − D for ADS and PoE within 3 points of Allouah et al.'s, over 3 seeds.
  Otherwise use their settings unchanged, report the difference, and do not tune.
- **Rewriting data checks (N3), as for LIMO:** answers preserved, numbers preserved, fallbacks counted. No problem is
  dropped; failures are salvaged or reported (CLAUDE.md rule 3).
- **Necessity:** practical (a quarter of the gain) and statistical (paired interval excludes 0), reported separately.
  The margin rule stays as pre-registered.
- **π:** with a 95% bootstrap interval (resample problems within seeds, then seeds); undefined where P2 fails.
- **Comparison of U − D and π (N7), primary.** Is there a defense with U − D ≥ 10 points in the published setup whose
  π interval in a valid setting lies entirely below 0.5? That is: a defense the usual metric calls strong keeps less
  than half its protection when measured correctly.
- **Comparison, secondary:** Kendall's τ between the defenses' order by U − D and by π (four or more defenses).

## 8. Schedule (assuming a late-January ICML 2027 deadline, not yet announced)

| Week | GPUs 0–3 | GPUs 4–7 | Writing |
| --- | --- | --- | --- |
| 2–8 Nov | N0 ends, recorded; N0b; N1 | N5a (pilot) | m8 rules for N1, N2, N5a; old-plan writing fixes, pass 1 |
| 9–15 Nov | N2 → **G1** | N5a → **G2**; N3 data (32B rewriters) | definitions section rewritten |
| 16–29 Nov | N3 runs; N4 | N5 (curve) | audit section rewritten (Javaheri et al. added) |
| 30 Nov – 20 Dec | N4 ends; N8 | N5 ends → **G3**; N6 | map section drafted |
| 21 Dec – 3 Jan | N6 (with 4–7) | N6 | buffer (holidays) |
| 4–17 Jan | reruns | reruns | N7; protection section; ICML style; 8 pages |
| 18 Jan – deadline | | | internal review; checklist; code release |

## 9. Risks and fallbacks

| Risk | Effect | Fallback |
| --- | --- | --- |
| Overlap with Javaheri et al. | "Defenses break" is not new | Lead with the map and valid π. Their setups fail necessity by their own numbers, and they do not cover rewriting defenses or PoE |
| The gap curve looks "true by definition" | Weak significance | The new part is what happens to defenses where protection exists, and how far U − D is from π |
| Long solutions not finished on MATH (P6) | The curve is uninformative | G2: the 4,096-token teacher, or fall back to 1,024 |
| ADS or PoE not reproduced | Part 3 compares the wrong defense | Their settings unchanged; report; never tune toward a result |
| Every defense recovered at level A, curve fails everywhere | Narrow positive part | Still a full paper: the audit plus "no valid setting shows protection" plus the map. Odds lower (about 30%) |
| Compute: GPUs shared | Delays | Priority order N1 > N2 > N5a > N0b > N3 > N5 > N4 > N6 > N8; N6 halves to one point |
| Deadline earlier than assumed | Less time | N6 at level A only; the curve without Mix |

## 10. What carries over from the old plan

- **Still to do** (old plan §4, writing, no GPU):
  - claims scoped to the levels run;
  - single-seed numbers marked;
  - Fig. 1a and Fig. 4 made consistent;
  - the Kim et al. anchor and the Solo finish rates added;
  - "60%" corrected to pass@8;
  - the B for each margin stated;
  - Table 2 labelled;
  - notation (rename levels or B);
  - ICML 2026 style as a layout proxy, 8 pages;
  - third-person self-citation;
  - Appendices A, C, D filled.
- **The Llama-3.2-3B-Instruct contamination:** recorded in m8 once the GSM8K rerun gives its training-problem
  accuracy (agreed 2026-10-31). The two sentences at main.tex:359 and :407 are then rewritten. In the new plan this
  finding also justifies measuring the gap on training problems.
- **References for the authors to add:** as in the old plan, plus Zhang et al. 2026 (a trained summary expander),
  cited by Javaheri et al.
- **Dropped or optional:** old E1/E1b (instruct student) becomes optional; old E5 (LIMO to three seeds) becomes
  appendix-only unless GPUs are free; old E6 is dropped with its sentence.

## 11. Decisions for the meeting

1. **Direction:** adopt the protection-map paper (§2) instead of finishing the audit as is?
2. **Definitions:** the student starts from the strongest model the attacker owns at the target size; the ladder is
   indexed by the attacker's best writer; gaps are measured on training problems?
3. **Defenses:** ADS, PoE, hindsight and PART, plus DOGe or TARS only if their code runs quickly?
4. **First two weeks:** go-ahead for N0b, N1 and N2 on GPUs 0–3 and N5a on GPUs 4–7, once N0 ends (about
   1 November) and their m8 rules are written?
5. **LIMO:** keep only the three-seed claims in the main text and move the rest to the appendix?
6. **Writing fixes:** start the old plan's §4 fixes now, on a branch of the paper repository?
