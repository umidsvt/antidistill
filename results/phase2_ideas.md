# Phase 2: ideas for an answers-only method that teaches reasoning (living document, started 2026-11-24)

Each idea: what it targets, why it might work, how it differs from published work, the cheapest test, status.
Experiments are recorded in `results/m8_audit_design.md` (§29 on) before they run; this file summarizes.
Setting: puzzle testbed (Knights and Knaves, 7–8 people), student Llama-3.2-3B-Instruct, the teacher's released
answers for 1,712 of 2,000 training puzzles (277 of them wrong). Target U_inst = 53.9%; best so far B5 = 25.5%
(answers-only SFT + RL, no reasoning); untrained 0.8%.

## What any method has to overcome (measured, m8 §26–28)

1. **No reward without help.** Solving from scratch, the untrained student reaches the teacher's answer on 117 of
   1,712 puzzles in 8 tries (6.8%); half its replies are not even a readable answer.
2. **Hints are copied.** Given the answer as a hint, it restates it; the teacher judges 68% of such texts invalid.
3. **It cannot verify.** Shown an assignment, its "consistent" verdict is the same for right and wrong ones (61–65%).
4. **Answers teach a shortcut.** SFT on answers learns surface statistics (B1 16.9%; a 31-feature logistic regression
   matches its per-person accuracy); RL on top raises it to 25.5% (B5), still without reasoning.
5. **Repair is easier than solving, but untargeted.** Given the teacher's answer with one person flipped (temperature
   0.6), it reaches the teacher's answer on 27–29% of puzzles in 8 replies (4× solving), but changes the wrong people
   (worse than flipping at random).

The common thread: the student needs training problems whose answers it can reach by valid reasoning, built from the
teacher's answers alone, and on which copying earns nothing.

## The ideas

### I1. Answer-repair curriculum (corrupted answers)
Give the puzzle with the teacher's answer, k people flipped; reward agreement with the teacher's answer; raise k.
- Targets 1 (signal at small k), 2 (the given answer is wrong by construction: copying earns 0), 3 (finding the
  flipped people is verification, learned through the reward rather than assumed).
- Prior work: hint annealing (HATCH, question augmentation) gives *correct* hints, which can be copied; GASP repairs
  corrupted *reasoning* with ground-truth rewards; reverse curriculum needs solutions. Corrupted *answers* as a
  curriculum toward solving, with an imperfect teacher's answers as the target, is not in what we found.
- Test: RL at k ∈ {1, 2} (test 2); evaluate repair skill, solving from scratch, and guess-then-repair (I3).

### I2. Answer-infilling curriculum (masked answers)
Give the puzzle with the teacher's labels for all but m people; the student gives the full answer; reward agreement.
- The masked people cannot be copied; with one person masked the deduction is often one statement long.
- Closer to partial-solution hints than I1 (the revealed part is correct). Its role: a contrast with I1. Do
  corrupted answers (which must be detected) teach more than masked ones (where it is known what is missing)?
- Test: RL at m ∈ {1, 2}, same evaluations as I1.

### I3. Shortcut-then-repair
The answers-only student (B1/B5) is a cheap proposer: its wrong answers still label 61% of people right (near
misses). Train a repairer whose starting answer is the shortcut student's own guess; at test time, guess then repair.
- Turns finding 4 from a dead end into a starting point: reasoning only has to fix the few people the shortcut got
  wrong.
- Prior work: proposer–verifier and self-refinement pipelines repair a model's own attempts with ground-truth rewards
  or critics; here the proposer itself is learned from the teacher's answers, and the repairer is trained only
  against them.
- Test: RL with B1's guesses on the training puzzles as starting answers; evaluate the pipeline B5 guess → repair.
- Baseline: the untrained student asked to repair B5's guesses (does unguided "repair" help or hurt?).

### I4. Repair self-training (repair-STaR)
I1 with self-training instead of RL: sample repairs at k = 1, keep those reaching the teacher's answer, fine-tune.
- Cheap (no RL). Unlike B2, the kept texts cannot come from copying (the given answer is wrong), so they must contain
  at least one change that the reasoning had to find.
- Test: one round at k = 1 (temperature 0.6, 8 tries), SFT, same evaluations.

### I5. Mixed curriculum
RL on a mixture: half repair (k ∈ {1, 2}), half solving from scratch. Repair supplies the signal; solving prompts
let the skill transfer to the test format within one model.

### Not pursued
- Program-based checking or solving: excluded by the threat model (the testbed stands in for domains without a
  solver).
- Test-time search with many samples: compared with U's single pass it would not be like for like.

## Evaluations (every trained student, seed 123, test puzzles)
- **Solve**: the standard test (`kk_eval.py`), comparable with every earlier student.
- **Repair skill** (diagnostic): the true solution with one person flipped, open repair prompt.
- **Guess → repair**: B5 s123's test answers as the starting answer; the attacker's two-step system.
- Reference points: I0 0.8%, B5 25.5% (s123: 22.7%), U_inst 53.9% (s123: 53.1%).

### I2b. Wider infill curriculum
I2 with m ∈ {1, 2, 3, 4} hidden people: can the infill signal be pushed toward solving (7–8 people)? Added
2026-11-24 15:30 on the idle GPU 5, because I2 was the only run with a dense reward early on.

## Status
- 2026-11-24 14:31: wave 1 (I1–I5, I1 at lr 2e-5, baselines) launched on 7 GPUs, m8 §29.
- 2026-11-24 15:30, early read-outs (m8 §29):
  - Untrained student: solve 1.3%, repair 4.2%, **guess → repair 3.9%** (it ruins B5's 22.7%: unguided repair hurts),
    infill of one person 31.0%.
  - I4 yield: own repairs reach the teacher's answer on 458 of 1,712 puzzles (26.8%); SFT running.
  - RL, first 12 steps: I1, I3, I5 flat at 0–7% exact agreement; **I2 22–43%**. I2b added.
- 2026-11-25, **wave 1 results** (m8 §29): every idea learns its own task (infill 31 → 60%; repair 4 → 7–9%), but
  **none transfers to solving** (0.2–2.1% vs 1.3% untrained) and none improves on B5's guesses (all lower B5's 22.7% to
  4–8%). Likely reasons: small hidden sets reward guessing (1/2, 1/4), the curriculum never reached the full puzzle,
  and repair stays too sparse to learn in one epoch.

## Wave 2 (2026-11-25)
- **W2a. Scaffolded exploration** (no training first): the root problem is that the untrained student reaches the
  teacher's answer on only 6.8% of puzzles. Does a worked example (Xie et al.'s two-person demonstration) or a
  case-analysis instruction ("assume a person is a knight, derive consequences, a contradiction makes them a knave,
  check the final assignment") raise that? If coverage rises substantially, answer-filtered self-training and RL on
  the scaffolded prompt become viable, followed by removing the scaffold.
- **W2b. Annealed infill to the full puzzle**: infill RL in stages (m ∈ {1–3} → {3–5} → {5–N} → solve), each stage
  starting from the previous adapter; guessing stops paying as m grows. About 21 h on one GPU.

### Wave 2a result (m8 §30)
No scaffold helps: the teacher's answer is reached on 5.9% (worked example), 6.6% (case-analysis instruction) and 4.6%
(both) of puzzles, against 6.8% unscaffolded, and readability drops. 6.8% is near what guessing gives (about 2–3% by
luck). W2b in its first form is replaced by I7 below.

### What wave 1's infill students actually learned (m8 §30, diagnostic)
Scored per person group at one hidden person: I0 copies the given labels right on 62% of readable replies and gets the
hidden person right on 60%; I2 92% and **67%** (a coin flip gives 50%). Infill RL taught reading and copying. The one
deduction step (a person's label from their statement and the given labels) barely moved. **The bottleneck is a single
deduction step, not puzzle length**, which explains why no curriculum transferred.

## Wave 3 (launched 2026-11-25 17:14, m8 §31)

### I6. Answer views for the answers-only student
The only route that has worked (B1 → B5, 47% of the gap) learns from answers without reasoning. Give it more of the
puzzle's structure per answer: besides puzzle → answer, train on infill views (the teacher's labels of a random subset
given, the full answer as target). Each answer then supervises many conditionals (a person's label given the
others'), not just the joint answer. SFT on plain + 3 views per puzzle, then B5's RL.
- Prior work: masked or any-order training of structured outputs is standard in other areas (masked language models,
  any-order autoregressive models); in answers-only distillation the multi-part structure of the teacher's answers has
  not been used as far as we found.
- **I6c control**: plain + 3 renamed copies (same number of examples and steps, no partial answers); also tests
  renaming as augmentation.

### I7. Infill curriculum with a hidden-person reward
I2's reward paid for copying the given labels; the new reward scores only the hidden people. Stages m ∈ {1,2} →
{2,3,4} → {4,5,6} → {6,7,8} (the last stage is half plain puzzles), each from the previous adapter, lr 2e-5. The
chain stops after stage 1 unless the hidden person is right on at least 75% of readable replies at one hidden person.
- Prior work: reverse curricula (R³, Xi et al.) start from late states of a *demonstration*; here the partial states
  come from the answer alone, which works because a puzzle's answer decomposes into partial assignments.

### I7b. The deduction step alone
Two epochs of RL on one-hidden-person items only. Can answers-only RL sharpen the single deduction step? If not, the
reasoning route is closed for this student without demonstrations, and the paper's story is that answers teach a
shortcut because the student cannot bootstrap the deduction.

### I8. A second student: Qwen2.5-3B-Instruct, untrained
Is near-chance coverage specific to Llama-3.2-3B (which is known to improve little under RL on reasoning tasks
without priming; Gandhi et al. 2025)? Solve, one-hidden-person deduction, and coverage over 8 tries. If Qwen
deduces much better, the reasoning route (RL with the teacher's answers from the untrained student) may work for it.

### Wave 3 results so far (2026-11-25 21:15, seed 123; m8 §31–32)
| Student | Solve after SFT | After RL | Recovered (s123, after RL) |
| --- | --- | --- | --- |
| B1 → B5 | 14.9 | 22.7 | 41% |
| I6, plain + 3 infill views | 29.9 | 32.2 | 60% |
| **I6c, plain + 3 renamed copies** | **33.2** | **36.8** | **69%** |

- **Renamed-copy augmentation roughly doubles the answers-only student** (one seed so far). The infill views help less
  than plain renaming, so the gain is not from partial-answer structure. Still no reasoning (67-token answers).
- Open question (I6d, running): is it more SFT steps or more varied names? Seeds 124–125 of I6c and a 7-copy run are
  running.
- **I7 stage 1 fails its gate**: with the copying reward removed, the hidden person is right on 61.8% (gate 75%). RL
  does not teach this student the deduction step; I7b (two epochs on the single step) still running.
- **I8 Qwen2.5-3B-Instruct**: hidden person right 76.3% (Llama 59.8%), coverage 12.2% (Llama 6.8%), solve 1.6%.
- Without reasoning, the augmented answers-only students get the hidden person right on 89–90%: intuition learned
  from answers beats the student's own reasoning at the single step.

**Direction this suggests:** the method that works is answers-only learning made data-efficient by answer-preserving
augmentation, then RL. Next questions: does it scale with copies; which augmentations help (statement order, names);
does augmentation also help the RL stage; does it hold for Qwen; does it close more of the gap at more seeds.

## Focus after the user's direction (2026-11-25 22:00)
Renamed copies are a baseline improvement, not the contribution: one GPU (the I6d control). Focus: a novel
answers-only technique that works, and possibly a second task (feasibility reported to the user; nothing started).

### I9. Answer-space iterative refinement (m8 §33) — current main candidate
The answers-only student trained on infill views labels one person from the others' labels right on 91.6% of people,
but solves only 31% of whole puzzles. Use the local step as a search: start from the student's own answer, then for
each person in turn ask "given everyone else's current labels, who is this?" and update; repeat. Learned from the
teacher's answers alone; no reasoning text, no solver.
- Related: discrete diffusion / masked iterative decoding solves Sudoku from input-output pairs (Ye et al., "Beyond
  Autoregression", 2024); Gibbs-style decoding of structured outputs. New here: as an answers-only distillation
  method, with an off-the-shelf autoregressive student, compared at matched tokens with distillation from solutions.
- **Test 1 (no training):** 31.5% → **35.9%** after 3 sweeps (I6 + RL), 1,588 generated tokens per puzzle (U_inst:
  4,561). Ceiling with this step: the reference answer is a fixed point on 54% of puzzles. 614 puzzles end at a wrong
  fixed point; 189 of them one person off.
- **Test 2 (running):** block sweeps (2–3 people updated jointly) to leave wrong fixed points; I10 RL on the views to
  sharpen the one-person step; I11 a dense view set (every person's view, fresh names) for SFT, then RL.
- Next if it works: train the student to write the refinement chain in one generation (single pass, like U), and
  restarts with selection of answers the student itself finds stable.
