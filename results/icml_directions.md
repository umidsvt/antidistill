# Publishing at ICML: a reviewer's view and four directions

> **Superseded 2026-10-20.** After the meeting with Mustafa and the advisor, the paper direction is the
> audit of anti-distillation setups: `results/m8_audit_design.md`. Directions 1 and 2 below are folded into it.

Written 2026-10-16, from `results/REPORT.md` as of that date; directions rewritten the same day to be
clearer. Read as an ICML reviewer and area chair would read the current results. Check every citation
before it goes into a paper.

---

## 1. Verdict

**As it stands, this is a strong workshop paper, but borderline to weak reject for the ICML main track.**
The experiments are careful. What is missing is breadth (more models, more defenses, repeated runs), and
one clear message that all the experiments support.

## 2. What a reviewer would credit

- **Solo.** An attacker that owns a reasoning model beats the defense without using the defended data at
  all: Solo scores 69.5%, above even the undefended student (62.8%).
- **Re-voice against Style.** Putting back only the first-person voice recovers nothing (52.5% against
  PART's 50.5%); adding doubt and checks recovers 68% of what PART removed (58.8%). So doubt and checking
  are what matter, not the voice.
- **Cause, not just correlation.** Deleting nine doubt words from Solo's data costs 7 points; the PART
  ablation shows that reordering adds nothing.
- **A bug in Kim et al.'s published defense code.** It saved two solutions per problem; fixed, the
  defended student goes from +6.7 to −0.3 points against the untrained model.
- **Method.** Decision rules written before results, 600 test problems instead of AIME's 30, and
  unfinished answers counted.

## 3. What a reviewer would attack

1. **One student model, one training set, one run each.** Every number comes from a single training run.
   We have never measured how much a result changes if the same training is repeated with a different
   random seed. Differences of 2 to 7 points could be within that variation. This would be the first
   question.
2. **Our attacker's model already contains what the defense hides.** R1-Distill was itself trained on
   DeepSeek-R1's solutions. A reviewer will say: of course the attacker does not need the teacher's
   solutions; it already learned from a strong teacher. We never tested a teacher that is far stronger
   than anything the attacker owns.
3. **800 training solutions teach a habit, not new knowledge.** This is the LIMO setting. Companies worry
   about attackers who collect 100,000 or more solutions, and our results may not carry over.
4. **Similar findings exist.** Li et al. 2025 ("structure, not content, is what matters") and Gandhi et
   al. 2025 ("cognitive behaviors") report that the reasoning habits matter more than correct answers. Our
   new part has to be what this means for defenses.
5. **We never check that the defended solutions are still useful to honest users.** Any defense can stop
   distillation by making its output useless; reviewers will ask for both sides.
6. **We test only defenses that rewrite the solution.** Not covered: defenses that change how the teacher
   samples (Antidistillation Sampling, Savani et al. 2025; the co-author's PoE, Allouah et al.), and simply
   hiding the solution or showing a summary, which is what the big API providers do.

## 4. Four directions

| | in one sentence | what we already have | cost |
| --- | --- | --- | --- |
| 1 | Show that every defense can be got around, and give a checklist for testing defenses | most of the attacks | lowest |
| 2 | Find out how much weaker the attacker must be before the teacher's solutions (and so a defense) matter | the starting point: our current setup is one point on the map | most new runs |
| 3 | Show by controlled experiments that the student learns the habit of checking, not the answers | several pieces of evidence | low |
| 4 | Work out the most any defense can protect, then design one | the measure | highest risk |

### Direction 1 — Show that every defense can be got around

**The idea.** Each published defense was tested against simple attacks. We test each one against an
attacker who knows how the defense works, as security papers do (Athalye, Carlini and Wagner's ICML 2018
best paper did this for defenses against adversarial examples). So far both defenses we tried fall:
hindsight to Solo (69.5%, above the undefended 62.8%), PART to Style (58.8%, two-thirds of the way back
from 50.5% to 62.8%).

**What we would do.**
1. Add two defenses of a different kind: one that changes how the teacher samples its text (PoE or
   Antidistillation Sampling), and hiding the solution completely.
2. Test every defense against the same fixed list of attackers. An attacker is described by what it has
   (only the problems; the problems and the teacher's final answers; the problems and the defended
   solutions) and by what model it owns (none, an ordinary instruction model, a reasoning model).
3. Repeat the main results with a second student model and with three training seeds.
4. Write a checklist for testing future defenses: always include Solo, use at least 600 test problems,
   count unfinished answers, check the defense's code.

**What it would show.** Whether any defense protects more than an attacker can get without the defended
data. **Cost:** lowest, since most attacks exist. **Risk:** reviewers may call it incremental; the breadth
and the checklist are the answer.

### Direction 2 — How much weaker must the attacker be before the teacher's solutions matter?

**The idea.** Our attacker's own model already solves 37% (7B) to 49% (32B) of the 800 training problems.
So it can write useful solutions of its own, and the teacher's solutions add little. That is why Solo
does so well. If the training problems were so hard that the attacker's model could solve almost none of
them, its own solutions would be useless, and the student would need the teacher's. Only then can a
defense protect something. We do not know where that point is. Direction 2 measures it.

**What we would do.** Keep the student (Qwen2.5-7B), the attacker's model (R1-Distill-7B) and the test
problems fixed. Change only the training problems, in three steps.

*Step 1 — build four training sets.* Each has 800 problems that a strong teacher solves correctly. They
differ in how many the attacker's model solves by itself:

| training set | problems the attacker's model solves by itself |
| --- | --- |
| easy | about 640 of 800 |
| current (our LIMO set) | about 300 of 800 (37%, measured: Solo's solutions reach LIMO's answer on 37%) |
| hard | about 80 of 800 |
| very hard | almost none |

The teacher's solutions for the new sets could come from public collections of DeepSeek-R1 solutions to
hard problems (for example OpenR1-Math or DeepMath-103K; to check).

*Step 2 — on each training set, train six students:*

- **A** on the teacher's solutions;
- **B** on the teacher's solutions after the PART defense;
- **B′** on our Style attack on B;
- **C** on the attacker's own solutions (Solo);
- **D** on the attacker's own solutions, kept only if they reach the teacher's final answer (more attempts
  are sampled per problem, so all 800 problems stay);
- **E** on no solutions at all: RL, where the student is rewarded when its answer matches the teacher's.
  E starts from the untrained student, which settles the co-author's question (their "Passive + RL" starts
  from a student already trained on the teacher's solutions).

C, D and E never use the teacher's written solutions; D and E use only its final answers.

*Step 3 — test every student* on the same test problems as now (the 600 problems and OlympiadBench).

**The result is a table of test scores.** Each cell is one trained student; the number is the share of
the 600 test problems it answers correctly. Four cells exist already, from our current set; "to run"
means a student not trained yet (20 new training runs).

| student trained on | easy set | current set | hard set | very hard set |
| --- | --- | --- | --- | --- |
| A · teacher's solutions | to run | **62.8%** (LIMO) | to run | to run |
| B · after the PART defense | to run | **50.5%** (PART) | to run | to run |
| B′ · Style attack on B | to run | **58.8%** | to run | to run |
| C · attacker's own solutions | to run | **69.5%** (7B Solo) | to run | to run |
| D · own, matching the teacher's answer | to run | to run | to run | to run |
| E · RL on the teacher's answers | to run | to run | to run | to run |

**If our explanation is right:** C, D and E score about as well as A on the easy and current sets, as
now, and far below A on the very hard set. Only there can a defense protect anything, and rows B and B′
show whether PART does. The result is a map of when the teacher's solutions are needed, and so when
defending them makes sense.

**How to get the harder training sets.** We do not write new problems. Two ways, cheapest first:

1. *Make the attacker weaker instead of the problems harder.* Keep our 800 LIMO problems and replace the
   attacker's model with weaker ones: R1-Distill-1.5B, and a small model with no reasoning training. Rows
   A, B and B′ do not depend on the attacker, so they stay as they are; only C and D need new students
   (about 4 training runs). Limits: row E does not change with this knob (it uses no attacker model), and a
   weaker model's solutions differ in style as well as in correctness.
2. *Pick hard problems from public collections that already contain a strong teacher's solutions.*
   OpenR1-Math-220k and DeepMath-103K (to check) contain math problems with DeepSeek-R1's full solutions
   and a checked final answer; DeepMath also rates difficulty.
   1. Keep only problems where DeepSeek-R1's solution reaches the correct answer (the teacher solves it).
   2. Remove anything that overlaps with our test problems.
   3. Take the problems rated hardest, a few thousand candidates.
   4. Run the attacker's model (R1-Distill-7B) four times on each candidate.
   5. Very hard set: 800 problems the attacker's model fails all four times. Hard and easy sets: 800
      problems each at the other solve rates.

   Before committing, a pilot: run the attacker's model on 500 of the hardest candidates (a few hours) and
   count how many it fails every time. If about one in five, the very hard set needs about 4,000
   candidates, roughly a day of GPUs 2–7 (estimate). Risks: too few very hard problems with a correct R1
   solution (then accept "fails 3 of 4"); topics that differ from LIMO's (match them where possible); R1's
   solutions are not curated like LIMO's.

**Why ICML would care.** It turns a puzzling observation ("no teacher solutions works best") into a rule
with one measurable variable, and it answers reviewer point 2. **Cost:** the most new runs, about 20
students. **Risk:** finding enough very hard problems with correct teacher solutions.

### Direction 3 — What exactly does the student learn from a solution?

**The idea.** Our results suggest the student learns the habit of doubting and checking, not the answers:
solutions that are wrong most of the time (Solo's) still train the best students; deleting nine doubt
words costs 7 points; first-person voice alone does nothing (re-voice) while added doubt recovers 68%
(Style). Direction 3 turns these hints into a controlled experiment.

**What we would do.**
1. Take one set of 800 solutions and make versions with 0, 1, 2 and 4 checking passages per solution
   ("Wait, let me check..."), all of the same length.
2. Train a student on each version and plot accuracy against the number of checks.
3. Place the checks at the steps where the student usually goes wrong, against at random steps.
4. Repeat with a second student model.

**What it would show.** Whether checking causes the gain, how much of it is needed, and whether its place
matters. **Cost:** low. **Risk:** the overlap with Li et al. and Gandhi et al.; it works best as the
analysis section of direction 1 or 2.

### Direction 4 — Can any defense protect the teacher at all?

**The idea.** If the teacher shows its correct final answers, an attacker can generate its own attempts
and keep the ones that reach those answers, or train with RL. Then no change to the written solution can
protect the problems the attacker can solve by itself. A defense can only protect the rest.

**What we would do.**
1. Define protection: the student trained on the teacher's full solutions, minus the best student the
   attacker can get.
2. Show, with an argument and with experiments, that protection is limited by how many problems the
   attacker cannot solve by itself.
3. Design a defense aimed at that remainder, and check that the teacher's output stays useful to honest
   users.

**Cost and risk:** highest risk and highest payoff; it needs a theory argument. Better as a second paper.

## 5. Recommendation

Submit **directions 1 and 2 together as one paper**, with direction 3 as its analysis section. The
message: changing the teacher's written solutions does not stop an attacker who can solve the problems
itself (direction 1), and here is how hard the problems must be before it does (direction 2).

## 6. First steps, whatever we choose (cheapest first)

1. **Three training seeds** for LIMO, hindsight v2, PART, 7B Solo and Style-on-PART. We have one run of
   each, so this is 10 new training runs of about 8 hours each, plus about 3 hours of evaluation each, on
   4 GPUs: roughly 5 days of GPUs 4–7.
2. **A second student model,** for example Llama-3.1-8B, for the main results.
3. **Check that defended solutions stay useful** to honest users.
4. **Write down the threat model** (what the attacker has and owns), and the compute each attack costs.

The ICML deadline is usually late January, about three and a half months away. That is tight but feasible
for directions 1 and 2 if the seed runs start now.

## 7. References to check before citing

- Athalye, Carlini, Wagner. *Obfuscated Gradients Give a False Sense of Security.* ICML 2018.
- Li et al. *LLMs Can Easily Learn to Reason from Demonstrations: Structure, not content, is what
  matters!* 2025.
- Gandhi et al. *Cognitive Behaviors that Enable Self-Improving Reasoners.* 2025.
- Savani et al. *Antidistillation Sampling.* 2025.
- Allouah et al. (the co-author's PoE defense).
- Ding et al. 2025, arXiv:2510.11545 (PART); Kim et al., arXiv:2603.15500 (hindsight).
- OpenR1-Math and DeepMath-103K (public collections of DeepSeek-R1 solutions; check contents and licences).
