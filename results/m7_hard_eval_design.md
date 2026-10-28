# M7 — a harder evaluation: do the defender's solutions matter on hard problems?

**Status: complete (2026-10-16): no difference between LIMO and Solo; all eight students evaluated.** Results go to
`results/REPORT.md` Part 10.

## 1. The question

Solo gives the best student in this study, above LIMO, although it uses neither the defender's solutions
nor its answers. A co-author sees the same pattern with RL on answers. Is that because the evaluation is
easy?

What we already know (REPORT Part 10, no new runs): split by MATH500's difficulty level, Solo's whole lead
over LIMO is on levels 1–4, where the untrained model already solves 63%. On level 5 (134 problems) and
on AMC23 + AIME24 + AIME25 (100 problems) Solo and LIMO are tied, and invented doubt (Style) falls 11–14
points below LIMO on level 5. A tie on 134 problems is ±4 points, and the split was chosen after seeing
the totals. This run tests the lead on a hard set chosen in advance.

## 2. What is run

**OlympiadBench**, 675 olympiad problems with a final answer, as shipped with Kim et al.'s evaluation code
(`third_party/kim_eval/data/olympiadbench/test.jsonl`; same prompt file as MATH500). It has not been run
in this project before. No training: the students are the ones we already have.

| order | student | why |
| --- | --- | --- |
| 1 | no training | where the benchmark sits for this model |
| 2 | LIMO | trained on the teacher's solutions |
| 3 | 7B Solo | best student; no solutions or answers from the defender |
| 4 | 32B Solo | the same with the stronger attacker |
| 5 | 32B Style | invented doubt; fell behind on MATH500 level 5 |
| 6 | hindsight v2 | defense |
| 7 | PART | defense |
| 8 | 7B Search | attack that does use the defended solutions |

Settings are those of every pooled-600 number: greedy, 32,768 new tokens, `tensor_parallel_size` 4, the
standard grader. Script: `scripts/24_olympiad_eval.sh`, GPUs 4–7, one student at a time.

## 3. Rule, fixed before any result

Paired on the same 675 problems, exact two-sided test on the problems only one of the two students solves.

| outcome | reading |
| --- | --- |
| LIMO above **both** Solo students, each with p < 0.05 | on hard problems the teacher's solutions are needed; the "solutions are not needed" result is a product of an easy evaluation |
| either Solo student above LIMO with p < 0.05 | Solo's lead holds on hard problems too; the easy-evaluation explanation is wrong |
| anything else | no difference shown at this size (675 problems resolve about ±4 points) |

Prediction written now: a tie between LIMO and Solo, and Style below LIMO.

Reported with it: the Style-against-LIMO comparison (same test), accuracy among finished answers and the
number of finished answers for every student, because students trained on long solutions stop less often
on hard problems and that alone can move the score.

**Validity conditions.** (1) The untrained model must score above 10%; below that the set is too hard to
separate students. (2) Before reading any comparison, 30 graded answers of the untrained model are checked
by hand; olympiad answers are often expressions or lists, and if the grader is wrong on more than 2 of 30
the set is re-graded offline from the saved responses with the fix applied to every student.

**Both conditions met (2026-10-13 08:20, before any trained student finished).** The untrained model
scores **201/675 = 29.8%** (598 finished; 33.6% among finished). Of 30 graded answers drawn at random
(seed 0) and checked by hand, 29 are graded as a person would grade them, including pairs, lists and
expressions. One is lenient: problem 395 asks for a value (−1/2) and the model returned an unevaluated sum
of cosines that equals −1/2, which the grader accepts. The grader is the same for every student, so it is
kept as is. Table and verdict: `scripts/38_olympiad_audit.py`, written before any trained student finished.

## 3a. Result

| student | correct of 675 | accuracy | finished | accuracy among finished | against LIMO: only this student / only LIMO |
| --- | --- | --- | --- | --- | --- |
| no training | 201 | 29.8% | 598 | 33.6% | +69 / −125, p < 0.001 |
| LIMO | 257 | **38.1%** | 565 | 45.5% | — |
| 7B Solo | 261 | **38.7%** | 582 | 44.8% | +69 / −65, p = 0.80 |
| 32B Solo | 261 | **38.7%** | 574 | 45.5% | +81 / −77, p = 0.81 |
| 32B Style | 251 | 37.2% | 597 | 42.0% | +80 / −86, p = 0.70 |
| hindsight v2 | 164 | 24.3% | 668 | 24.6% | +53 / −146, p < 0.001 |
| PART | 204 | 30.2% | 536 | 38.1% | +59 / −112, p < 0.001 |
| 7B Search | 275 | **40.7%** | 617 | 44.6% | +78 / −60, p = 0.15 |

**Verdict by the rule fixed in advance: no difference shown at this size.** This is the outcome predicted
before the run. Solo's lead over LIMO, 6.7 points on the pooled 600, is 0.6 points on the hard set; LIMO is
not ahead either, in accuracy or among finished answers. Training on either adds 8 to 9 points over no
training. So the easy evaluation explains Solo's *lead*, but a harder evaluation does not show that the
teacher's solutions are *needed*: a student trained on the attacker's own solutions, which are wrong on
51–63% of the training problems, matches the student trained on the teacher's correct ones at every
difficulty we can measure. What remains untested is a harder *training* setting (§4).

Secondary (2026-10-15): 32B Style ties LIMO (p = 0.70), so the prediction "Style below LIMO" made in §3
is **not** borne out. Hindsight v2 is below the untrained model (24.3% against 29.8%). Added 2026-10-16:
PART is level with the untrained model (30.2%); 7B Search is highest at 40.7% (against LIMO p = 0.15). Two
runs hung at shutdown after writing complete results (17 min and 27 h of idle GPUs);
`scripts/26_eval_exit_watchdog.sh` now handles this.

## 4. What this does not test

The training problems are unchanged (the same 800). This is a harder **evaluation**, not a harder
training setting. Whether an attacker without a model that can attempt the training problems still
manages is a separate experiment and needs new teacher solutions.
