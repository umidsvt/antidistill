# Phase 0: how much does an instruction-tuned student gain from the teacher's solutions?

Status 2026-11-20 (all seeds in). Design record: `results/m8_audit_design.md` §24. Numbers are recomputed from the stored
per-problem files by `scripts/phase0_table.py` (output `outputs/phase0/table.json`).

## Question

An anti-distillation defense keeps the teacher's final answers correct. So the most it can protect is the gap between
two students:
- **U_inst:** the student fine-tuned on the teacher's full solutions;
- **I0:** the same student, untrained.

This gap is also the target for the answers-only fine-tuning method we plan to develop: the method sees only the
teacher's answers and should get as close to U_inst as possible. Phase 0 measures the gap. A setting counts as having
a gap if U_inst − I0 ≥ 10 points and its 95% interval excludes 0 (rule agreed 2026-11-19).

## Result

| Setting | Teacher | Teacher's accuracy | Student, untrained (I0) | Student, trained on teacher's solutions (U_inst) | Gap (95% CI) |
| --- | --- | --- | --- | --- | --- |
| GSM8K | R1-Distill-Qwen-7B, 1,024 tokens (field setup) | 87.2 | 82.2 | 78.7 | −3.6 (−4.4 to −2.7) |
| GSM8K | R1-Distill-Qwen-32B, 8,192 tokens | 96.6 | 82.3 | 79.2 | −3.1 (−3.8 to −2.4) |
| MATH | R1-Distill-Qwen-7B, 1,024 tokens (field setup) | 63.1 | 48.6 | 40.8 | −7.8 (−10.4 to −5.2) |
| MATH | R1-Distill-Qwen-32B, 8,192 tokens | 93.3 | 48.6 | 49.1 | +0.5 (−3.7 to +4.7) |
| Puzzles (Knights and Knaves, 7–8 people) | R1-Distill-Qwen-32B, 8,192 tokens | 71.8 (training puzzles) | 0.8 | 53.9 | +53.1 (50.3 to 55.9) |

Accuracy in % on the test set, mean over seeds 123–125. The gap is U_inst − I0, paired by seed: the mean of the
per-seed differences ± t(0.975, n − 1) × SD / √n.

- **GSM8K and MATH: no gap.** Training the student on the teacher's solutions lowers its accuracy on GSM8K with both
  teachers, and on MATH with the field's teacher. On MATH with the 32B teacher it changes nothing. No math setting
  meets the gap rule.
- **Puzzles: a gap of about 53 points.** The untrained student solves almost no puzzles; trained on the teacher's
  solutions it solves 54%.
- **Consequence:** with this student and these data sizes, the GSM8K/MATH setups leave nothing for an answers-only
  method to recover. The puzzle testbed leaves about 53 points. The field's own setup (7B teacher, 1,024 tokens) does
  not just have a narrow gap for an instruction-tuned attacker: the gap is negative.

## Models

| Role | Model | Notes |
| --- | --- | --- |
| Student, every row | Llama-3.2-3B-Instruct | the ungated copy `unsloth/Llama-3.2-3B-Instruct`; the attacker's model |
| Teacher, field setup | DeepSeek-R1-Distill-Qwen-7B | solutions cut at 1,024 tokens; the setup's own stored solutions are reused |
| Teacher, bigger | DeepSeek-R1-Distill-Qwen-32B | up to 8,192 tokens; generated for Phase 0; also the puzzle teacher |

## Setup: GSM8K and MATH

The Distillation Game setup (Allouah et al., arXiv 2605.22737), the setup the field uses for anti-distillation, run
from its code (`third_party/ext/distillation-game`). Our changes are listed at the end of this section.

**Data.** Each seed shuffles the training pool and takes its own training split.

| | Training problems per seed | Distinct over 3 seeds | Test problems |
| --- | --- | --- | --- |
| GSM8K | 5,238 | 7,252 | 1,209 (GSM8K-Platinum) |
| MATH | 5,000 | 7,221 | 5,000 (MATH test set) |

The test set is the same for every seed.

**Prompt.** The setup's system prompt ("You are a math teacher. You will be given a math problem and you will solve
it step by step. You will output your final solution like \boxed{ANSWER}. ..."), then the problem as the user turn, in
each model's chat template. The prompt is cut at 512 tokens.

**Teacher solutions.** One solution per training problem: temperature 0.6, top-p 0.95, up to 1,024 tokens (7B) or
8,192 tokens (32B).
- **Answer forcing (the setup's):** after the teacher stops, the code adds `</think>` if the thinking was not closed,
  then the cue `**Final Answer**` `\[\boxed{`, and lets the model write 32 more tokens greedily. What it writes in
  the box is the forced answer. The cue exists because many solutions are cut off before an answer.
- **How the teachers' solutions end:**

  | Teacher, training solutions | Thinking closed | Finished within the limit | Median length |
  | --- | --- | --- | --- |
  | 7B, GSM8K (seed 123) | 89.5% | — | — |
  | 7B, MATH (seed 123) | 25.8% | — | — |
  | 32B, GSM8K (7,252 distinct) | 100% | 99.99% | 423 tokens |
  | 32B, MATH (7,221 distinct) | 95.0% | 94.7% | 1,392 tokens (90th percentile 5,247) |

**Training text.** The student's chat template with the setup's prompt; the assistant turn is the teacher's solution
followed by a final line `**Final Answer**` `\[\boxed{...}\]`.
- **7B teacher:** the setup's forced final line, unchanged.
- **32B teacher (our change, decided 2026-11-19):** when the teacher finished, the final line carries its own boxed
  answer; when it was cut off, the forced line stays. Reason: the 32B's forced answer contradicted its own correct
  answer in 8.1% of MATH training solutions (single letters such as "D", the literal "ANSWER" from the prompt,
  dropped signs). The student copies such lines: at test time, the field-teacher students wrote "\boxed{B}" after
  their own correct answer. No reference answers are used. The original files are kept.

**Training recipe (the setup's).**

| | |
| --- | --- |
| Method | LoRA, rank 128, alpha 128, no dropout |
| Optimizer | AdamW, learning rate 5e-4, cosine schedule, 3% warm-up, weight decay 0.1, gradient clipping 1.0 |
| Epochs | 3 |
| Effective batch | 24 (7B solutions: 6 × 4 accumulation; 32B solutions: 1 × 24) |
| Maximum length | 2,048 tokens (7B solutions); 9,216 tokens (32B solutions, so none is cut) |
| Loss | on the assistant turn only |

**Scoring.** One reply per test problem with vLLM: temperature 0.6, top-p 0.95, up to the teacher's limit (1,024 or
8,192 tokens); then the same answer forcing as for the teacher. The setup's checker (math_verify) counts a reply as
correct if its own answer or its forced answer matches the reference. I0 is scored at both limits.

**Our changes to the setup.**
- Instruction-tuned student instead of the setup's base Llama-3.2-3B.
- The 32B teacher at 8,192 tokens, with the final-line rule above.
- vLLM scorer (`scripts/dg_score_vllm.py`) instead of the setup's HF loop. Checked against it on I0: GSM8K 82.13% in
  both, MATH 49.04% vs 48.2%.

## Setup: puzzles (Knights and Knaves)

**Puzzles.** Generated with Xie et al.'s (2024) generator (`third_party/ext/mem-kk-logic`) with fresh seeds. Each
person makes one statement built from "X is a knight/knave" with and/or/not/implies/iff, and every puzzle has exactly
one solution.
- Training: 2,000 puzzles, 1,000 with 7 people and 1,000 with 8.
- Test: 1,000 puzzles, 500 with 7 people and 500 with 8.
- 7–8 people is where the teacher solves most puzzles and open models of the attacker's size solve few (pilot, m8
  §17: the teacher solves 75.5% at 7 people and 71.5% at 8).

**Prompt.** Xie et al.'s instruction: reason, then end with "CONCLUSION:" and one numbered line per person.

**Teacher solutions.** One per training puzzle from R1-Distill-Qwen-32B: temperature 0.6, top-p 0.95, up to 8,192
tokens. 1,435 of 2,000 are correct (71.8%). The student is trained on all 2,000, correct or not.

**Training recipe.** As for GSM8K/MATH: LoRA rank 128, alpha 128, learning rate 5e-4, cosine, 3 epochs, effective
batch 24, weight decay 0.1, loss on the solution only. Maximum length 8,704 tokens.

**Scoring.** One reply per test puzzle: temperature 0.6, top-p 0.95, up to 8,192 tokens. A reply is correct when its
last CONCLUSION section labels every person exactly once and every label matches the solution. Scoring the first
answer instead gives the same numbers within 0.3 points.

## Per-seed results

**GSM8K and MATH.** Accuracy %; in brackets the share of replies that finished within the limit and the median reply
length in tokens.

| Setting | Seed | I0 | U_inst | Gap |
| --- | --- | --- | --- | --- |
| GSM8K, 7B teacher, 1,024 | 123 | 82.13 (99.8%, 181) | 78.16 (73.4%, 926) | −3.97 |
| | 124 | 81.89 (99.8%, 173) | 78.58 (78.9%, 805) | −3.31 |
| | 125 | 82.71 (99.8%, 180) | 79.24 (75.9%, 895) | −3.47 |
| GSM8K, 32B teacher, 8,192 | 123 | 82.13 (99.8%, 181) | 78.74 (99.3%, 439) | −3.39 |
| | 124 | 81.89 (99.9%, 173) | 79.07 (99.5%, 435) | −2.81 |
| | 125 | 82.80 (99.9%, 180) | 79.65 (99.7%, 435) | −3.14 |
| MATH, 7B teacher, 1,024 | 123 | 49.04 (87.4%, 344) | 41.18 (31.8%, 1,024) | −7.86 |
| | 124 | 47.78 (87.9%, 353) | 41.08 (25.9%, 1,024) | −6.70 |
| | 125 | 48.96 (89.1%, 357) | 40.14 (28.6%, 1,024) | −8.82 |
| MATH, 32B teacher, 8,192 | 123 | 49.10 (94.4%, 348) | 48.34 (62.4%, 2,086) | −0.76 |
| | 124 | 47.58 (95.4%, 347) | 49.98 (71.5%, 1,708) | +2.40 |
| | 125 | 49.14 (95.5%, 356) | 49.00 (63.9%, 2,071) | −0.14 |

**Puzzles.** Solve rate %; in brackets the share of replies that finished and the mean reply length in tokens.

| Seed | I0 | U_inst | Gap |
| --- | --- | --- | --- |
| 123 | 1.3 (98.2%, 730) | 53.1 (75.1%, 4,561) | +51.8 |
| 124 | 0.5 (96.6%, 861) | 54.1 (77.1%, 4,592) | +53.6 |
| 125 | 0.6 (97.1%, 914) | 54.5 (77.2%, 4,556) | +53.9 |

For comparison, the base Llama-3.2-3B of the earlier puzzle runs (m8 §18): untrained 0.3%, trained on the same 2,000
solutions 48.8%.

## Why the math students do not gain

**MATH, 32B teacher: the student gets stuck in repetition loops.** 28–38% of its replies run to the 8,192-token limit
(the teacher's own test replies finish 96% of the time), and 85–94% of those end in a loop: their last 300 characters
occur earlier in the reply. Split by whether the student's reply finished, against I0 on the same problems:

| Seed | Finished replies | U_inst on them | I0 on them | Cut-off replies | U_inst on them | I0 on them |
| --- | --- | --- | --- | --- | --- | --- |
| 123 | 3,121 | 67.0 | 64.2 | 1,879 | 17.4 | 24.0 |
| 124 | 3,573 | 63.0 | 58.7 | 1,427 | 17.3 | 19.7 |
| 125 | 3,194 | 67.6 | 63.8 | 1,806 | 16.2 | 23.2 |

Where the student finishes, it beats I0 by 3–4 points. Where it loops, it loses. The training recipe is the setup's,
built for solutions of at most 1,024 tokens. A recipe suited to long solutions may reduce the loops. But even the
finished replies gain only 3–4 points, far from the 10-point rule.

**GSM8K, both teachers: the student is worse even where it finishes.** With the 32B teacher, 99% of replies finish,
and on those problems U_inst scores 79.2–79.8% against I0's 81.9–82.9%. The instruction-tuned student already solves
GSM8K well, and 5,238 teacher solutions only disturb it.

**MATH, 7B teacher: the student inherits the cut-off.** The field teacher's MATH solutions mostly stop at 1,024 tokens
before an answer (thinking closed in 25.8%). The student learns to run to the limit: 68–74% of its replies are cut
off, and it is worse than I0 both where it finishes and where it is cut.

## Caveats

- **Puzzle seed 123 (U_inst)** was run a day after the others. Its first two training attempts on 2026-11-19 ran out
  of memory when another process took most of GPU 3, and a launcher bug logged the job as done; the bug was fixed and
  the same job rerun on 2026-11-20 (53.1%).
- **Scorer fix (2026-11-19).** Our vLLM scorer first joined reply and forced answer with one newline too few. For
  replies containing `</think>` the checker then read only the forced answer. It was fixed before the 32B-teacher
  students were scored. The 7B-teacher MATH students of seeds 123 and 124 were rescored. Seed 123's replies were
  identical in the rescore, so its change (39.66% → 41.18%) is the fix alone. Seed 124's replies came out
  different despite the same seed (39.98% → 41.08%); every number in this file is from one run with the fixed scorer.
- **One sample per problem** at temperature 0.6, for teachers and students alike, as in the setup.
- **Recipe untuned.** The setup's LoRA recipe was used for every student; no hyperparameter was tuned for the
  instruction-tuned student or for long solutions.

## Files

| What | Where |
| --- | --- |
| 32B teacher solutions | `outputs/phase0/teacher32b_{gsm8k,math}/` (`train_s<seed>.json`; originals with forced final lines `train_s<seed>_forced.json`; `test.json`, `summary.json`) |
| 7B teacher solutions | `outputs/setup2/{gsm8k,math}_U_s<seed>_*/teacher/` |
| I0 scores | `outputs/phase0/I0/<dataset>_s<seed>_<limit>.json` |
| U_inst, 7B teacher | `outputs/phase0/S1/<dataset>_s<seed>/` (`test_1024.json`, `train_info.json`) |
| U_inst, 32B teacher | `outputs/phase0/S2/<dataset>_s<seed>/` (`test_8192.json`, `train_info.json`) |
| Puzzle students | `outputs/kk_step2/students/{untrained_inst,Uinst}_s<seed>/test/` |
| Puzzle data | `data/kk/step2_{train,test}.jsonl`; teacher solutions `outputs/kk_step2/train_sets/U.jsonl` |
| Examples (teacher solutions and student attempts) | `results/phase0_examples.md` |
| Scripts | `scripts/56_phase0.sh`, `57_kk_instruct.sh`, `dg_teacher_vllm.py`, `dg_own_final.py`, `dg_train_from_traces.py`, `dg_score_vllm.py`, `kk_sft.py`, `kk_eval.py`, `phase0_table.py` |

`outputs/` is `/data/umid/antidistill/outputs` on dspl1.
