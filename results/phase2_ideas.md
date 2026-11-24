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
