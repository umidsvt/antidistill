# M8 — Auditing anti-distillation setups: design

**Status: framing revised 2026-10-20 after the user's guidance (§8, which supersedes the plan in §6); decisions fixed (§7); E1 running.** This replaces the recommendation in
`results/icml_directions.md` (directions 1 and 2 are folded in here).

---

## 1. The idea (from the meeting)

A defense against distillation only makes sense if the attacker actually **needs** the teacher's
solutions. If the attacker can get an equally good student without them — by training on solutions
written by an open model it already has (Solo), or by RL on its own student with only the final answers —
there is nothing for the defense to protect, and any "the defense works" result in that setup means
nothing.

Our LIMO setup is such a case: the student trained on the attacker's own 7B model's solutions scores
69.5%, above the student trained on the teacher's undefended solutions (LIMO, 62.8%).

**Contribution:** a short list of checks that any anti-distillation setup must pass; an audit of the
setups used by published defense papers; setups that do pass; and how defenses and our attacks do there.

## 2. The checks

For one setup (teacher data, student model, test set), train these students and test them on the same
test problems:

| name | the student is trained on |
| --- | --- |
| **base** | nothing (the untrained student) |
| **U** | the teacher's undefended solutions |
| **Solo** | solutions written by an open model the attacker could have, for the same problems |
| **RL** | nothing written: RL on the untrained student, rewarded when its answer matches the teacher's answer |

Let **S** be the best of base, Solo and RL: the best student the attacker gets **without** the teacher's
written solutions.

| check | passes if | what failing means |
| --- | --- | --- |
| **0 · distillation helps** | U is clearly above base | distillation gives nothing; a defense has nothing to remove |
| **1 · the teacher is needed** | U is clearly above Solo | an open model already gives the same; the attacker would not query the teacher |
| **2 · solutions are needed, not only answers** | U is clearly above RL (only if the defense still releases final answers) | hiding or changing the solutions cannot matter |
| **3 · it is measured properly** | 3 training seeds, a confidence interval, at least 600 test problems, unfinished answers counted | differences may be noise |

"Clearly above" = above by more than the 95% confidence interval across seeds. The size of margin to
require is to be fixed by the team before the first new result (§7).

**How a defense is then reported.** Protection = U minus the best student the attacker can get, counting
both attacks on the defended data and S. A defense can protect at most U − S.

**Our LIMO setup:** check 0 passes (62.8% vs 49.8%); check 1 fails (Solo 69.5% vs 62.8%; on OlympiadBench
38.7% vs 38.1%). U − S is negative: nothing to protect.

## 3. Setups used by published defense papers (read 2026-10-20)

Facts below are from each paper's own text (sources at the end). "—" = not stated in the parts read.

| paper | type of defense | teacher → student | training data | test sets | base student reported? | student without the teacher reported? |
| --- | --- | --- | --- | --- | --- | --- |
| Kim et al. 2026 (hindsight) | rewrite: remove doubt | LIMO solutions → Qwen2.5-7B, Qwen3-8B/14B-Base, R1-Distill-7B/32B | LIMO-v2, 800 | AIME24 (+AMC23, MATH500) | yes | no |
| PART (Ding et al. 2025) | rewrite: remove self-talk, reorder | DeepSeek-R1 solutions → Qwen2.5-Math-7B, Qwen2.5-7B-Instruct, 1.5B | Bespoke-Stratos-17k; OpenThoughts-114k | MATH500, AIME24 | — | no |
| Antidistillation Sampling (Savani et al., NeurIPS 2025) | change sampling | R1-Distill-Qwen-7B (proxy Qwen2.5-3B) → Llama-3.2-3B | GSM8K, MATH train | GSM8K, MATH | — | no |
| DOGe (Li et al., NeurIPS 2025) | fine-tune teacher's last layer | R1-Distill-Qwen-7B, Qwen3-8B → Qwen2.5-0.5B, Qwen3-0.6B, Gemma-3-1B, Llama-3.2-1B, Llama-3.1-8B | GSM8K or Tulu prompts | GSM8K, MATH, ARC-C, CSQA | — | no |
| Distillation Game / PoE (Allouah et al. 2026) | change sampling | R1-Distill-Qwen-7B (proxy Qwen2.5-3B) → Llama-3.2-3B | GSM8K 5,238; MATH 5,000 | GSM8K, MATH | yes: 0.76% / 1.01% | answers only: 3.43% / 11.25%; no RL, no Solo |
| Trace rewriting (Ma et al. 2026) | rewrite with an LLM | R1-Distill-Qwen-7B → Llama-3.2-1B/3B, Qwen2.5-1.5B | GSM8K, MATH train | GSM8K-Platinum, MATH | no | answers only |
| TARS (Hartman et al. 2026) | remove key sentences | R1-Distill-Qwen-7B, Gemma-3-12B → Llama-3.2-1B/3B, Gemma-3-1B, Qwen3-1.7B | — | MMLU-Pro, MATH, DeepMath | — | — |
| DistillGuard (Jiang 2026), evaluation study | paraphrase, poisoning, solution removal | Qwen3-14B (no thinking) → Qwen2.5-7B-Instruct | 10,000 prompts (3,000 MATH) | MATH-500, HumanEval+, MT-Bench | yes | no |

**What this already shows, before any experiment:**

1. **No defense paper reports Solo or RL from the untrained student.** Two report an answers-only student.
2. **DistillGuard fails check 0 by its own numbers.** On MATH-500 the untrained student scores 68.4% and
   the undefended distilled student 67.8%: distillation adds nothing, so no defense can protect anything
   there. Its strongest "defense" (removing the solution) drops the student to 31.4%, below the untrained
   student.
3. **Five papers share almost the same small setup:** R1-Distill-Qwen-7B as the teacher, 1B–3B Llama or
   Qwen students, GSM8K and MATH (ADS, PoE, Trace rewriting, DOGe, TARS). One audit of that setup speaks to
   all five. Here the "teacher" is itself an open model, and GSM8K is solved by many open models, so
   check 1 is likely to fail; to be measured.
4. **PART's setup may fail check 1 without any training.** The open R1-Distill-Qwen-7B scores 50.0% on
   AIME24 (greedy, Kim et al. Table 5), above PART's undefended Qwen2.5-Math-7B student (32.7% with
   Bespoke-Stratos-17k, 46.7% with OpenThoughts-114k; different decoding, so to be re-measured).

## 4. Closest prior work, and what is left for us

- **Javaheri et al. 2026, "Distillation Defenses Easily Break After Reinforcement Learning"
  (arXiv 2609.35699, 28 Sep 2026).** Closest. They add RL after distillation to the threat model and show
  that ADS and trace summarization stop working. In their own setups RL from the untrained student already
  beats distillation from undefended solutions (Qwen2.5-0.5B: 40.3% vs 35.8%; 1.5B: 64.6% vs 61.1%; 3B:
  72.6% vs 71.6%; Llama-3.2-3B: 24.3% vs 19.8%), except Qwen2.5-Math-7B (79.3% vs 84.7%). They also
  expand API summaries into full solutions with a 3B model, close to our Style attack. Limits they state:
  students of at most 3B, short training contexts, the SimpleRLZoo problem sets.
- **Allouah et al. 2026, "The Distillation Game."** Argues defenses must be judged against adaptive
  students (reweighting the training examples); no RL or Solo baseline.
- **DistillGuard (Jiang 2026).** Finds most output-level defenses ineffective; does not notice that its
  undefended distillation gives no gain.

**What is new in our version:**

1. **The checks as a precondition, applied to other people's setups,** not a new attack. Javaheri et al.
   show RL often wins in their setups; we ask, for each published defense, whether its own setup could show
   anything at all.
2. **Solo** — training on an open model's solutions — as a check that needs no RL and works for long
   solutions (LIMO, s1K, Bespoke-Stratos: 10,000+ token solutions), where Javaheri et al.'s short-context
   RL does not reach.
3. **Check 0,** which DistillGuard fails.
4. **The positive half:** setups that pass, and how defenses (hindsight, PART, PoE) and our attacks (Style,
   Search, Mix, Style on PART) do in them.
5. **Rewriting defenses** (hindsight, PART), which Javaheri et al. do not cover.

**Risk:** Javaheri et al. take much of the novelty of "RL beats distillation". The paper must lead with the
audit and the passing setups, cite them prominently, and not claim the RL observation as ours.

## 5. Students: smaller models (agreed 2026-10-20)

Use **Qwen2.5-1.5B and Qwen2.5-3B** (same family as our 7B) and **Llama-3.2-3B** (the student of the shared
GSM8K/MATH setup), possibly for the main results as well; the 7B only where a result needs it.

Open question to settle first (§6, E1): do 1.5B and 3B students gain anything from 800 long LIMO
solutions? If not (check 0 fails), long-solution setups need the 7B, and small students are used for the
GSM8K/MATH setups only.

## 6. Plan

| weeks | step | GPUs |
| --- | --- | --- |
| 1 | Fix the checks' margins (§7). Read the remaining papers fully (TraceGuard, LADS, Fang et al., Distillation Traps and Guards); add them to §3. | none |
| 1–2 | **E1 · small students on LIMO.** Qwen2.5-1.5B and 3B: base, U (LIMO), Solo (the 7B Solo solutions we already have). 4 training runs. | 2–4 |
| 1–4 | **E2 · the shared GSM8K/MATH setup** (ADS, PoE, Trace rewriting, DOGe, TARS). U = the co-author's passive Llama-3.2-3B student. Solo = solutions from open non-teacher models (e.g. Qwen2.5-7B-Instruct). RL from the untrained Llama-3.2-3B, by the co-author. | co-author + 2–4 |
| 2–5 | **E3 · more long-solution sets.** s1K-1.1 (1,000 R1 solutions) and a Bespoke-Stratos or OpenThoughts subset: base, U, Solo, with the student chosen in E1. | 4 |
| 3–8 | **E4 · find setups that pass.** The very hard training set (pilot first, `results/icml_directions.md` direction 2) and weaker attacker models. | 4–6 |
| 6–11 | **E5 · defenses and attacks in the passing setups.** Hindsight, PART, PoE (co-author); Style, Search, Mix, Style on PART. Report protection against S. | 4–6 |
| 10–14 | Seeds for the headline cells; writing; release the checks as a script. | some |

Every new LIMO-style training pool keeps all its problems (800 for LIMO), as in the rest of the project.

## 7. Decisions (fixed 2026-10-20, before any new result)

Agreed by the user on 2026-10-20:

1. **"Clearly above"** = above by more than the 95% interval across 3 training seeds, **and** by at least
   a quarter of U's gain over the untrained student.
2. **The attacker's models** = the strongest open model no larger than the teacher.
3. **RL reward** = agreement with the teacher's released final answers, not ground truth.
4. **RL is run by the co-author** (their GRPO code), starting from the untrained Llama-3.2-3B in their
   GSM8K and MATH setups.

### E1 rule (fixed before E1 starts)

E1 asks only whether small students are usable for long-solution setups; it is one seed, so it is not a
check verdict. **A student size is usable for long-solution setups if training on LIMO raises its pooled
600-problem score by at least 5.0 points over the untrained student** (the 7B gained 13.0). Solo against
LIMO is reported as well, marked as single-seed.

**Tie-break (fixed 2026-10-20 09:00, before any E1 trained result):** if both sizes are usable, setup 1 uses the
3B, the closer to our 7B results; if neither is, setup 1 waits for a decision (the 7B costs about 3x more).
Setup 1's seed queue starts on its own when E1 ends: `scripts/30_setup1_queue.sh`.

E1 runs on GPUs 6–7 (0–5 were busy with the user's other project on 2026-10-20): training on both GPUs at
global batch 8 (2 GPUs × batch 1 × accumulation 4), evaluation at TP=1, one student per GPU, the same for
all six E1 students. Configs: `configs/train/qwen2.5-{1.5b,3b}_{limo,recon7b_solo}.yaml`; runner:
`scripts/28_e1_small_students.sh`.

## 8. Revised framing and plan (2026-10-20, supersedes §6)

**Guidance from the user:** do not attack every published setup. Take **two** setups and show that the
checks are valid, find or **create one setup that passes**, and raise novelty by stating **what is
expected from a setup**, with more tests. Template: our NeurIPS 2026 paper *Evaluating an Evaluation:
Membership Inference Attacks as Machine Unlearning Diagnostics* (`../papers/`), whose recipe is
"formalize validity properties, build a normalized diagnostic score, and stress-test with graded variants
of the gold-standard reference", ending in a reporting checklist.

### 8.1 The object of study

The standard evaluation of an anti-distillation defense: train a student on the teacher's undefended
outputs and on its defended outputs, and report the drop. We ask when this evaluation can be trusted.

### 8.2 Validity properties of an anti-distillation setup

Students in one setup: **base** (untrained), **U** (teacher's undefended outputs), **D** (defended outputs,
plain fine-tuning), **A** (best attack on the defended outputs), and the **no-teacher reference R** (best of
base, Solo, RL on the teacher's answers, answers only). R plays the role the retrained model plays in
unlearning: the gold standard for "nothing was taken from the teacher".

| property | requirement | example of failure we already have |
| --- | --- | --- |
| **P1 Headroom** | U is reliably above base | DistillGuard: 67.8% against base 68.4% |
| **P2 Necessity** (new; no unlearning analogue) | U is reliably above R | LIMO: Solo 69.5% against U 62.8%; Javaheri et al.: RL beats distillation |
| **P3 Stability** | across seeds, the protection estimate has a bounded coefficient of variation; report the signal-to-noise ratio and the chance that one run orders two defenses correctly | our AIME24-only result: LIMO +0.0 pp vs +13.0 pp pooled on the same checkpoints |
| **P4 Monotonicity under graded leakage** | as a growing share of the defended outputs is replaced by undefended ones (0, 25, 50, 75, 100%), the student's score does not fall | hindsight v1 mixes: 56.5% → **52.2%** (10%) → 60.0% → 63.8% → 62.8% |
| **P5 Consistency across test sets** | rankings of defenses and attacks agree across benchmarks (Kendall's τ > 0) | GSM8K vs the pooled 600 (REPORT Part 6) |
| **P6 Robustness to confounders** | differences are not explained by how often answers finish or by solution length | the P4 failure disappears in accuracy among finished answers (64.3 → 64.4 → 68.5 → 73.6 → 78.2%) |

### 8.3 A normalized protection score

Defined only when P2 holds (U > R):

    protection = (U − max(A, R)) / (U − R)

It runs from 0 (no protection) to 1 (the attacker is pushed down to what it could get without the
teacher). It cannot exceed 1, because the attacker can always fall back on R. Published papers report
U − D, which credits a defense even when R ≥ U and ignores adaptive attacks.

### 8.4 Graded stress tests

1. **Graded leakage** (P4): 0/25/50/75/100% undefended outputs mixed into the defended set.
2. **Graded attacker strength:** no model, 1.5B, 7B, 32B reasoning model for Solo.
3. **Graded problem difficulty:** easy to very hard training sets (`results/icml_directions.md`,
   direction 2), to find where P2 starts to hold.

### 8.5 Setups

| setup | what | role |
| --- | --- | --- |
| **1 · long solutions** | LIMO (s1K-1.1 as a second dataset of the same kind), Qwen2.5 students | expected to fail P2 (LIMO already does) |
| **2 · short solutions** | the shared GSM8K/MATH setup of five papers: R1-Distill-Qwen-7B → Llama-3.2-3B; U, D (PoE, ADS) and RL by the co-author, Solo by us | P2 to be measured |
| **3 · constructed to pass** | very hard training problems or a weaker attacker | P1–P6 should hold; run hindsight, PART, PoE and our attacks here |

"Showing the checks are valid" means showing that each property catches a real problem in setups 1–2,
and that in setup 3 the properties hold and the protection score behaves (stable, monotone, consistent).

### 8.6 Order of work

1. **E1** (running): are 1.5B/3B students usable for long solutions? Also measures how long a small run
   takes, which sets the seed budget.
2. **Setup 2 with the co-author:** RL from the untrained Llama-3.2-3B (P2), plus Solo.
3. **Setup 1 with seeds:** base, U, Solo, hindsight, PART, Style on PART; 3 seeds each; graded leakage.
4. **Setup 3:** pilot for the very hard set, then the same grid.
5. **Writing:** properties, failure modes with their statistics, protection score, checklist.

## 9. Setup 2 protocol (built 2026-10-20; the user asked to run it ourselves rather than via the co-author)

**Code.** The Distillation Game's public code (`third_party/ext/distillation-game`, commit 56ecdf7), so U and
the untrained student are measured exactly as in that paper. Two patches, both marked `ANTIDISTILL PATCH`:
its data package imports two modules missing from the public repo; and the proxy model, used only by ADS, is
skipped when no ADS teacher is requested. Separate environment `.venv-dg` (torch 2.6, transformers 4.53.2,
TRL 0.19.1, vLLM 0.8.5) so the main pipeline is untouched.

**Setup (from its `configs/gsm8k.yaml`).** Teacher DeepSeek-R1-Distill-Qwen-7B, sampled at 0.6 with 1,024 new
tokens and answer forcing; GSM8K train split 5,238 problems; test = GSM8K-Platinum (1,209 problems); student
Llama-3.2-3B (the ungated `unsloth` copy, same weights) with LoRA r=128, 3 epochs, lr 5e-4; evaluation with
the same sampling and answer forcing. Seeds 123, 124, 125 (each also reshuffles the train split, as in the
paper). Holdout reduced to 64 problems: it feeds only ADS and the adaptive student, and the train split is
unchanged.

**Students per seed.** base (untrained; `scripts/dg_eval_base.py`, the paper's own evaluation function);
**U** (the teacher's undefended traces); **Solo** (traces from Qwen2.5-7B-Instruct, an open model of the
teacher's size, decision 2; same pipeline with the teacher swapped); **RL** (`scripts/dg_rl_base.py`: GRPO
from the untrained student with the same LoRA, reward 1 when its boxed answer matches the teacher's released
answer for that problem under the paper's answer checker; 8 samples per prompt, 8 prompts per step, one pass
over the 5,238 train prompts, lr 2e-5, temperature 1.0, 1,024 new tokens; scored with the paper's evaluation).

**Smoke tests (2026-10-20, 48 problems):** the pipeline runs end to end (teacher 79% on the test sample;
untrained student 0%, the paper reports 0.76%); RL runs 3 steps and evaluates; the reward fires on 1% of
samples at the start, as expected for a student that starts near 0%.

**Cost (estimate from the smoke tests):** U or Solo about 3 h each on one GPU; RL about 8–11 h; base 10 min.
About 45 GPU-hours for the 3 seeds, about 12 h on four GPUs. Runner: `scripts/29_setup2_run.sh`.

### 9.1 Setup 2 results (2026-10-20, 3 seeds; GSM8K-Platinum, 1,209 problems)

| student | seed 123 | seed 124 | seed 125 | mean |
| --- | --- | --- | --- | --- |
| untrained | 0.58% | 0.91% | 0.74% | **0.74%** (paper: 0.76%) |
| **U**: distilled from the teacher, R1-Distill-Qwen-7B | 59.64% | 57.82% | 59.55% | **59.0%** (paper: 57.24%) |
| **Solo**: distilled from Qwen2.5-7B-Instruct | 59.14% | 60.96% | 59.39% | **59.8%** |
| **RL** from the untrained student, teacher's answers | 0.91% | 5.79% | 1.32% | **2.7%** |

Teacher accuracy on the test set: 86.4 / 87.7 / 87.4% (paper: 87.22%); Qwen2.5-7B-Instruct: 93.6 / 94.4 / 94.6%.

**By the rule fixed in §7:**
- **Check 0 passes:** U is 58 points above the untrained student.
- **Check 1 fails:** Solo is at or above U in two of three seeds (mean 59.8% vs 59.0%). The rule needed U above
  Solo by more than the interval across seeds and by at least a quarter of U's gain (14.6 points).
- **Check 2 is not settled.** Our RL did not learn: the reward stayed at 1–2% over the whole pass, because
  the untrained student almost never ends with a correct boxed answer, so there is nothing to reinforce. This
  is a weak attacker, not evidence that the teacher's solutions are needed; Javaheri et al. report 24% for
  the same student with a standard RL recipe. A stronger recipe (a prompt the untrained model can follow, a
  format reward, or full fine-tuning) is needed before reading check 2.

**So the setup shared by ADS, PoE, trace rewriting, DOGe and TARS fails the necessity check:** an attacker
with an open 7B instruction model gets as good a student without the teacher, so a defense in this setup
protects nothing (protection score undefined; U − R ≤ 0).

### 9.2 E1 and the first setup-1 runs (2026-10-21)

Pooled 600 problems (MATH500, AMC23, AIME24, AIME25), greedy, TP=1; one seed unless marked.

| student | untrained | U (LIMO) | Solo (7B Solo data) | hindsight v2 |
| --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | 1.3% (54/500 MATH answers finished) | 10.5%; seed 43: 10.7% | 18.3%; seed 43: 19.8% | **27.8%** |
| Qwen2.5-3B | 41.2% | 42.5% | 44.7% | — |
| Qwen2.5-7B (Parts 4–5) | 49.8% | 62.8% | 69.5% | 49.5% |

- **3B fails the E1 rule** (LIMO adds 1.3 points; 5.0 required), i.e. check 0 fails for the 3B on LIMO.
- **1.5B passes it mechanically** (+9.2 points), but only because the untrained 1.5B almost never produces an
  answer in this prompt format; its LIMO student still finishes only 95 of 500 MATH500 answers. The rule chose
  1.5B for setup 1, as fixed in advance.
- **Check 1 fails at every size:** Solo is above U for 1.5B, 3B and 7B.
- **At 1.5B the defended student beats the undefended one** (hindsight 27.8% vs LIMO 10.5%): hindsight's short
  solutions teach the small student to finish. This is a clean P6 failure (finishing confounds the score).
- **10 of 13 setup-1 jobs failed to start** (2026-10-21 10:46–10:51): another user's job took 40 GB of GPU 3 at
  10:14, every training on GPUs 2–3 ran out of memory within 35 seconds, and the queue moved on without
  retrying. Done: U s43, Solo s43, hindsight v2 s42. Not done: PART ×3, Style on PART ×3, U s44, Solo s44,
  hindsight v2 s43–s44. The queue needs a free-memory check and retries before it is used again.

### 9.3 The attacker ladder (decided 2026-10-21, before any of its results)

Solo depends on which models the attacker has, so the necessity check is reported per level of attacker
resources. A setup *passes at a level* if U is clearly above (§7 margin) the best student the attacker gets
at that level; a defense paper must say which level it defends against.

| level | the attacker has | student built by | setup 2 |
| --- | --- | --- | --- |
| A | only the student | RL from the student, rewarded by the teacher's answers | v1 failed to learn (reward 1–2%); v2 below |
| B | the student's own instruction-tuned version | Solo from Llama-3.2-3B-Instruct | **59.8 / 60.0 / 58.6%, mean 59.4% vs U 59.0%: fails** |
| C | an open model of the teacher's size | Solo from Qwen2.5-7B-Instruct | done: 59.8% vs U 59.0% |
| D | an open reasoning model | Solo from an R1-Distill model | not planned yet |

**Level A, RL recipe v2.** v1 never learned because the untrained student almost never ends with a correct
boxed answer, so almost every group of samples had the same reward and nothing to learn from. v2 keeps the
same prompt, LoRA and evaluation (so the score stays comparable with U and Solo) and changes four things:
a small reward (0.2) for writing a boxed answer at all, on top of 1.0 for a correct one; LoRA learning rate
1e-4 (v1: 2e-5); sampling temperature 0.8 (v1: 1.0); and a second variant with 16 samples per prompt
instead of 8 (4 prompts per step, so the same 64 samples per step). Same budget as v1: 655 steps.
Implementation note (before the pilots): samples are processed 8 at a time in every variant (16 at once ran
out of memory), so variant B accumulates over 8 micro-batches; v1 was already 8 at a time. Smoke test, 2 steps:
the boxed-answer reward fires on 14% of samples and no group of samples has identical rewards (v1: 92% did).

**Pilot rule, fixed now.** Both variants run 60 steps on seed 123. A variant passes if its correctness reward
(without the format bonus) averages at least 5% over steps 41–60 (v1 started at 1–2% and stayed there). The
better passing variant is run for 3 seeds; if neither passes, level A is reported as "RL from the student
does not work in this setup with standard recipes", which is itself a check-2 pass for this setup.

**Pilot result (2026-10-22 00:40): neither variant passes, and the reason is a flaw in v2, not the setup.**
Correctness reward over steps 41–60: 0.000 (A) and 0.000 (B); test accuracy 0/300 for both. The boxed-answer
reward was hacked: it went from 0.26 at step 5 to 1.00 by step 15, every sample ran to the 1,024-token limit
(100% cut off), correctness fell to 0, and with identical rewards in every group the loss went to 0. The
format reward paid for the string `\boxed{` anywhere, so writing it and never stopping was the easiest way to
collect it; the higher learning rate made the collapse fast.

The fallback written in the rule above ("level A reported as RL does not work here, a check-2 pass") would
turn a recipe bug into a finding, and is not applied without a decision (2026-10-22): two failed recipes do
not show that RL from this student cannot work (Javaheri et al. report 24% for the same student).

**RL recipe v3 (fixed 2026-10-22 00:50, before its pilots).** Same as v2 except: the format bonus (0.2) is paid
only when the text contains exactly one boxed answer and ends with it, so it cannot be collected by writing
`\boxed{` and running to the limit; and lower learning rates, 5e-5 (v3a) and 2e-5 (v3b), 8 samples per prompt,
temperature 0.8. Same pilot rule (correctness reward averaged over steps 41–60 at least 5%); the better passing
variant runs 655 steps for seeds 123–125 (`scripts/32_setup2_rl_v3.sh`). If neither passes, level A stays open
and the next candidate is rejection-sampling fine-tuning (sample the student many times, keep the attempts that
reach the teacher's answer, fine-tune on them).

**v3a pilot (2026-10-22 01:06): passes the rule on paper, but by a degenerate policy.** Correctness reward over
steps 41–60: 0.072 (≥ 0.05); test accuracy 21/300 = 7.0%. But by step 40 the answers had shrunk to about 7 tokens
(mean length 600 → 7) and the format reward reached 1.0: the student learned to write only `\boxed{number}`,
with no reasoning, and guesses right about 7% of the time. The rule did not anticipate a policy that stops
reasoning. The automatic 3-seed continuation was to be stopped (01:13), but the wrong process was stopped; it
started at 01:30 after v3b and was stopped 10 seconds later, before any training step. v3b (lr 2e-5): correctness
over steps 41–60 0.013, format reward 0.006, test 3/300 = 1.0% (does not pass). No v3 full run exists. Decision
pending: self-training (rejection-sampling fine-tuning) or Javaheri et al.'s published RL recipe.

**Level B result (2026-10-22 02:07).** Solo from Llama-3.2-3B-Instruct: 59.80 / 59.97 / 58.56% (seeds 123–125),
mean 59.4% against U 59.0% (paired differences +0.2, +2.2, −1.0). That model scores 81.6 / 81.4 / 80.9% on the test
set, below the teacher (86.4 / 87.7 / 87.4%), yet its solutions train an equally good student. **The GSM8K setup
fails necessity at level B:** an attacker who owns only the student's released instruction-tuned version needs
nothing from the teacher.

**Level A by self-training (decided 2026-10-22 23:40; fixed before any result).** One round of STaR from the untrained
Llama-3.2-3B, using only the teacher's released final answers (`scripts/dg_star.py`): (1) 8 attempts per training
problem in the setup's own prompt format (temperature 0.8, 1,024 tokens, answer forcing as in its evaluation); keep
the first that reaches the teacher's answer; (2) for problems never solved, the teacher's answer is added to the
prompt as a hint and 8 more attempts are made; the training text keeps the prompt without the hint
("rationalization", as in STaR); (3) still unsolved: 32 hinted attempts. Every problem is kept; any problem unsolved
after (3) is reported. The student is then fine-tuned exactly as the distilled students (LoRA r128, 3 epochs,
lr 5e-4) and scored with the setup's evaluation. Seeds 123–125, each on the same split and teacher answers as its
U run. This is the level-A number; RL stays reported as four recipes that did not learn.

**Crash and fix (2026-10-23).** Every generation run (s123 three times, s124 once; 2.5 hours each) crashed at the
answer-forcing step: the cue was appended to the decoded text, and re-encoding that text gave up to 2,476 tokens
against vLLM's 1,664 limit (the untrained student writes byte sequences that decode to more tokens than were
generated). The cue is now appended to the generated token ids, and each round is saved so a restarted run resumes
after the last finished round. Nothing about the procedure above changes.

**Smoke test of the fixed script (2026-10-23 13:05, first 200 training problems of seed 124, k_salvage 8): the
recipe runs, but its data is junk.** 19 problems solved by the student's own attempts, 176 with the hint, 2 in the
salvage pass, 3 unsolved. 18 of the 19 "own" solutions reached the answer only through answer forcing, after text
unrelated to the problem (other invented problems, code fragments, repeated words): the untrained base model does not
follow this setup's chat prompt (it scores 0.7% on the test set). 82 of the 178 hinted solutions repeat the hint
sentence, and the ones read by hand are unrelated text ending in the hinted number. Fine-tuning on this would measure
the prompt, not what an attacker with only the student can do. Seed 123 of the recipe as fixed keeps running on GPU 2,
as the record of the pre-registered procedure; seeds 124–125 are held. **Proposed change (not applied without a
decision):** few-shot generation, as in STaR itself (Zelikman et al. 2022 prompt the model with a few hand-written worked
examples): the base model is prompted with the 8 standard hand-written GSM8K chain-of-thought examples (Wei et al. 2022) instead of
the chat prompt; the kept solutions are then trained under the setup's own prompt exactly as before. These 8 examples
are public text written by people, not by the teacher, so the attacker still owns only the student.

**Result of the recipe as fixed, seed 123 (2026-10-23 20:32): 4.9%** (59 of 1,209; U on the same seed 59.6%).
Generation: 617 of 5,238 problems solved by the student's own attempts (11.8%), 4,504 with the hint, 108 in the
salvage pass, 9 unsolved and so not in the training set (5,229 solutions). As the smoke test predicted, a student
trained mostly on hinted text learns little; this is a recipe result, not a level-A result. Level A stays open
until the few-shot change above is decided.

**Few-shot self-training (approved 2026-10-27; fixed before any result).** Identical to the recipe above except
the generation prompt. (1) The untrained Llama-3.2-3B gets the 8 hand-written GSM8K chain-of-thought examples of
Wei et al. (2022) as raw text ("Q: … A: … The answer is N."), then "Q: <problem> A:"; generation stops at the next
"Q:". 8 attempts per problem, temperature 0.8, top-p 0.95, up to 1,024 tokens. An attempt counts only if it states
"The answer is N" with N equal to the teacher's released answer; there is no answer forcing (in the smoke test it was
the source of the lucky guesses). (2) Problems never solved: the hint "(The answer is N.)" is placed after the question,
in the 8 examples as well, and 8 more attempts are made; (3) still unsolved: 32 hinted attempts; anything left is
reported. Kept solutions end in "The answer is \boxed{N}." and are trained under the setup's own chat prompt, without
the examples or the hint, exactly as U; the student is scored exactly as U (setup's prompt, no examples). Seeds
123–125, each on the same split and teacher answers as its U run, `scripts/dg_star.py --fewshot`. Outputs
`outputs/setup2/star_fewshot_gsm8k_s<seed>/`. A 200-problem smoke test checks the format first; it has no threshold.
**Reading:** this is the level-A number for GSM8K. The setup passes necessity at level A if U (59.0%) is above it by
the §7 margin (more than the interval across seeds and at least 14.6 points); it still fails at levels B and C.

**Few-shot smoke test (2026-10-27 06:46, first 200 problems of seed 124, k_salvage 8).** 124 solved by the student's
own attempts (62%, against 19 = 9.5% with the setup's prompt), 74 with the hint, 1 in the salvage pass, 1 unsolved;
generation took 1.5 minutes. The student's own solutions read as real worked solutions (median 216 characters). Some
hinted solutions reason to a different number and then state the hinted one (seen in 1 of the 2 read); this is how
STaR's rationalization works when only the final answer is checked, and it is kept as fixed. The process then hung at
exit inside vLLM for 28 minutes after writing everything; the gen phase now exits as soon as its files are written
(re-tested on 50 problems: exits in 2 minutes).

**Few-shot self-training result (2026-10-27 09:46; level A on GSM8K).**

| seed | own solutions | with hint | salvage | unsolved (not trained on) | test accuracy | U, same seed | U − A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | 3,123 | 2,031 | 52 | 32 | **34.41%** | 59.64% | +25.2 |
| 124 | 3,141 | 2,005 | 60 | 32 | **33.58%** | 57.82% | +24.2 |
| 125 | 3,168 | 1,978 | 67 | 25 | **34.99%** | 59.55% | +24.6 |
| mean | 60% of 5,238 | | | | **34.3%** | 59.0% | **+24.7** |

**By the rule, the GSM8K setup passes necessity at level A**: U is above the self-trained student by 24.2–25.2 points on
every seed, far beyond the interval across seeds and the 14.6-point margin. Under §11 the comparison is U_A − R_A with
U_A ≥ U, so the verdict can only get stronger. The attacker who owns only the student needs the teacher; one who also
owns the student's instruction-tuned sibling does not (R_B = max(Solo B 59.4%, 34.3%) = 59.4%; combination students
pending). One-round STaR is a lower bound on level A (more rounds or RL could do better); Javaheri et al. report 24% for
RL from the same student in their setup.

### 9.4 MATH variant of setup 2 (started 2026-10-22 23:40)

The same five papers also use MATH, where the teacher is far ahead of the student (published: teacher 61.8%,
passive student 15.2%), so this is where necessity is most likely to hold. Configs `configs/setup2/math_*`, the
published sizes (5,000 training, 5,000 test problems); untrained, U, Solo C (Qwen2.5-7B-Instruct) and Solo B
(Llama-3.2-3B-Instruct), seeds 123–125; job pool `scripts/33_setup2_pool.sh` on GPUs 3–7.

**Third patch to the authors' code (2026-10-22 23:53).** Answer forcing detects the chat format from the first
problem only; when that problem is longer than the 512-token prompt limit, its prompt is truncated, the chat marker
is cut off and the run crashes (seed 124, twice). The marker is now taken from the first problem that has one;
runs that worked are unchanged. Long problems are still truncated at 512 prompt tokens, as upstream, for
comparability with the published numbers.

**Results (complete 2026-10-23 17:20).** MATH test, 5,000 problems.

| student | seed 123 | seed 124 | seed 125 | mean |
| --- | --- | --- | --- | --- |
| untrained | 0.72% | 0.86% | 0.72% | **0.77%** |
| **U**: distilled from R1-Distill-Qwen-7B | 14.58% | 15.36% | 15.04% | **15.0%** (paper: 15.2%) |
| **Solo C**: from Qwen2.5-7B-Instruct | 17.18% | 17.04% | 17.80% | **17.3%** |
| **Solo B**: from Llama-3.2-3B-Instruct | 14.90% | 15.28% | 14.44% | **14.9%** |

Accuracy of the model whose solutions are used, on the test set: R1-Distill-Qwen-7B 63.1 / 63.4 / 62.8% (paper:
61.8%); Qwen2.5-7B-Instruct 73.1 / 73.0 / 73.2%; Llama-3.2-3B-Instruct 48.2 / 48.5 / 47.3%.

- **By the §7 rule the MATH variant fails necessity at levels B and C.** Paired differences against U: Solo C +2.6,
  +1.7, +2.8 (above U on every seed); Solo B +0.3, −0.1, −0.6 (mean −0.1). The rule needed U above each by more than
  the interval across seeds and by at least a quarter of U's gain (14.2 / 4 = 3.6 points).
- Llama-3.2-3B-Instruct solves 15 points fewer test problems than the teacher (48% vs 63%) and still trains an
  equally good student, as on GSM8K.
- **The setup's token limit makes the teacher weaker than an ordinary instruction model.** With 1,024 new tokens
  R1-Distill-Qwen-7B scores 63% on MATH (about 92% on MATH500 with its full budget), 10 points below
  Qwen2.5-7B-Instruct, so the "stronger teacher" of this setup is in fact the weaker solver. GSM8K shows the same
  order (teacher 87%, Qwen2.5-7B-Instruct 94%).

### 9.5 Setup 1 at 7B: seeds 43–44 of U and Solo (complete 2026-10-25 11:45)

Same recipe as the seed-42 runs (LIMO's, 15 epochs, global batch 8), on 4 GPUs with gradient accumulation 2 (seed 42
used 8 GPUs × 1); evaluation greedy, pooled MATH500 + AMC23 + AIME24 + AIME25 (600 problems). Solo = the 7B Solo data
(R1-Distill-Qwen-7B's own solutions). All four trainings ran the full 1,500 steps with normal losses.

| student | seed 42 | seed 43 | seed 44 | mean |
| --- | --- | --- | --- | --- |
| untrained Qwen2.5-7B | 49.8% | | | 49.8% |
| **U**: LIMO's solutions | 62.8% | 58.5% | 48.2% | **56.5%** |
| **Solo D**: R1-Distill-Qwen-7B's solutions | 69.5% | 68.7% | 60.5% | **66.2%** |

MATH500 alone: U 69.0 / 64.0 / 51.6%, Solo 77.0 / 76.6 / 66.4%. Answers containing a boxed answer: U 571 / 566 / 571
of 600, Solo 562 / 563 / 560, so seed 44's drop is not a finishing effect (accuracy among those: U 66.0 / 62.0 / 50.6%).

- **Necessity fails at level D on every seed.** Paired Solo − U: +6.7, +10.2, +12.3 points (mean +9.7, 95% interval
  +2.7 to +16.8 with 3 seeds).
- **Headroom (P1) is not shown with 3 seeds.** U's gain over the untrained model is +13.0, +8.7 and −1.6 points (mean
  +6.7, 95% interval −11.9 to +25.3); seed 44's U is below the untrained model. The seed-to-seed spread of U (standard
  deviation 7.5 points on 600 problems) is larger than the gain it is meant to measure. This is a P3 finding as much as
  a P1 one: a single seed of this setup can show a 13-point gain or none.
- Seed 42 was the best seed for both students; the single-seed numbers used elsewhere (62.8 vs 69.5) are therefore not
  biased toward Solo.

## 10. Setup 3 (built to pass): pilot (fixed 2026-10-22 23:50, before any result)

**Goal.** A training set on which the teacher (DeepSeek-R1, public solutions) is far ahead of the attacker's own
model, so that P2 can hold. Source: DeepMath-103K (MIT licence): 103,022 problems with a verifiable final answer,
a difficulty rating and three DeepSeek-R1 solutions each; 17,085 are rated 8 or higher.

**Pilot.** From the problems rated ≥ 8, keep those whose first R1 solution reaches the stated answer, fits a
16,384-token training context (Qwen tokenizer), and does not appear in any of our test sets (exact or 60-character
prefix match on the question). Draw 300 at random (seed 0). The attacker's model, DeepSeek-R1-Distill-Qwen-7B (the
level-D attacker of the LIMO setup), answers each 4 times (temperature 0.6, top-p 0.95, up to 16,384 tokens).

**Readout and rule.** The share of candidates the attacker solves 0 of 4 times. If at least 20%, an 800-problem
very hard set needs at most about 4,000 candidates: build it. If under 5%, DeepMath's hardest problems are not hard
enough for a 7B reasoning attacker, and setup 3 instead varies the attacker (a weaker model on the same problems).
In between: report and decide. Script: `scripts/34_setup3_pilot.py`.

**Pilot result (2026-10-23 01:25).** 300 candidates (306 screened: 0 with a wrong R1 solution, 5 too long, 1 in a test
set). R1-Distill-Qwen-7B solves them 0/1/2/3/4 of 4 times: 31 / 20 / 33 / 61 / 155; per-sample accuracy 74%; 94% of
samples finish. **Share solved 0 of 4 times: 10.3% → the "report and decide" band of the rule.** By rating: 19/205
at 8.0, 7/66 at 8.5, 5/28 at 9.0. A very hard set of 800 would need about 7,800 candidates screened four times
with up to 16k-token answers, which is out of reach. DeepMath's hardest problems are not hard for an open 7B
reasoning model; the gap to the teacher has to come from the attacker side (the ladder), not from harder problems.

### 10.1 Option 4: the MATH setup with the teacher at full budget, pilot (approved 2026-10-27; fixed before any result)

**Why.** Option 1 (LIMO with weaker attackers) cannot pass: with three seeds the LIMO setup fails P1 (§9.5). The MATH
setup has strong P1 and P3 (U 15.0% vs 0.8%, seed SD 0.4) and fails P2 because the teacher is capped at 1,024 new
tokens and solves 63% of MATH, below Qwen2.5-7B-Instruct (73%) (§9.4). Lifting the cap is the most direct way to make
the teacher the strongest model in the setup.

**Pilot, seed 123 only.**
1. *Teacher.* R1-Distill-Qwen-7B on the seed-123 split (5,000 training and 5,000 test problems) with the setup's
   prompt (cut at 512 tokens, as upstream), sampling (temperature 0.6, top-p 0.95) and answer forcing, but up to 8,192
   new tokens instead of 1,024. Generated with vLLM instead of the authors' HF loop (same sampling distribution).
   Reported: test accuracy and the share of solutions that finish within the limit.
2. *Students.* Llama-3.2-3B with the setup's recipe (LoRA r128, 3 epochs, lr 5e-4, effective batch 24), one per data
   source on the same 5,000 problems: **U_long** (the 8,192-token teacher solutions, new); **U_1024**, **Solo C**
   (Qwen2.5-7B-Instruct) and **Solo B** (Llama-3.2-3B-Instruct), retrained from their existing seed-123 training files
   because the original runs did not save student weights. All four use the same two changes so that long solutions
   are not cut and fit in memory: training cut-off 9,216 tokens instead of 2,048, and micro-batch 1 × accumulation 24
   instead of 6 × 4 (same effective batch). Weights are saved.
3. *Scoring.* The untrained model and all four students on the 5,000 test problems with the setup's prompt, sampling
   and answer forcing, up to 8,192 new tokens, with vLLM. **Scorer check first:** the retrained U_1024 and Solo C
   scored with the vLLM scorer at 1,024 tokens must land within 1.5 points of their original seed-123 scores (14.6%,
   17.2%); otherwise the scorer is fixed before any comparison is read.

**Rule.** Margin m = max(3.6, a quarter of U_long's gain over the untrained model), all at 8,192 tokens.
- U_long − Solo C ≥ m: level C passes on the pilot seed; seeds 124–125 of every student follow, and the §7 test
  (interval across seeds and margin) decides.
- Otherwise, U_long − Solo B ≥ m: level B passes on the pilot seed; seeds 124–125 of U_long and Solo B follow.
- Otherwise the MATH setup fails necessity even with a full-budget teacher; reported as such, no further seeds.
U_1024 at 8,192 tokens is reported alongside, to separate the effect of the longer teacher from that of the longer
scoring limit. Scripts: `scripts/38_math_long_teacher.py`, `scripts/dg_train_from_traces.py`,
`scripts/dg_score_vllm.py`; outputs `outputs/setup3_math/`.

**Teacher (2026-10-27 09:41).** R1-Distill-Qwen-7B with 8,192 new tokens: **79.0%** on the 5,000 test problems (63.1% at
1,024), 93% of solutions finish; training split 78.5%, 92% finish; median 1,724 tokens, 90th percentile 7,039. Check of
the vLLM teacher at 1,024 tokens on 300 training problems: 65.7% vs 62.3% for the setup's own run (same prompts and
forced answers). The teacher is now 6 points above Qwen2.5-7B-Instruct (73.1%), not the ~90% hoped for.

**Pilot result (2026-10-28 00:40), seed 123, MATH test (5,000).** Scorer check passes: retrained U_1024 at 1,024 tokens
15.3% (original 14.6%), retrained Solo C 18.4% (original 17.2%), both within 1.5 points.

| student (seed 123) | 1,024 tokens | 8,192 tokens | finished at 8,192 | accuracy among finished |
| --- | --- | --- | --- | --- |
| untrained | 2.7% | 2.9% | 771 | 3.5% |
| U_1024 (teacher capped at 1,024) | 15.3% | 14.7% | 4,966 | 14.8% |
| Solo B (Llama-3.2-3B-Instruct) | 16.6% | 15.9% | 4,829 | 16.2% |
| Solo C (Qwen2.5-7B-Instruct) | 18.4% | 18.5% | 4,821 | 19.0% |
| **U_long** (teacher at 8,192) | 17.7% | **18.1%** | **2,059** | **31.4%** |

- **By the rule, U_long alone fails at both levels**: U_long − Solo C = −0.4, U_long − Solo B = +2.2, margin
  m = max(3.6, (18.1 − 2.9)/4 = 3.8). By §11 the long-teacher combination students run before the pilot is read as a
  failure (§11, below).
- **The failure is a finishing effect (P6), not a lack of value in the long solutions.** The U_long student imitates the
  teacher's long reasoning and finishes only 41% of its answers within 8,192 tokens; on those it is right 31.4% of the
  time, 12 points above Solo C on its finished answers. Unfinished answers are right 8.8% of the time (forced guesses).
  A 3B student does not learn when to stop from 5,000 long solutions in 3 LoRA epochs, as the 1.5B student on LIMO did
  not (§9.2). Options that would address it (not decided): a longer scoring limit, or the teacher at an intermediate
  budget (e.g. 4,096), so that its solutions are shorter than the student's limit.

## 11. Same attacker in U, A and R; necessity against U_ℓ (agreed 2026-10-27, before any result below)

**The point (raised by the co-author, agreed by the user).** R_ℓ lets the attacker use everything it owns at level ℓ,
but U and A did not: U was plain distillation and A only used the defended outputs. An attacker can combine, e.g. first
train on its own model's solutions (or self-train) and then on the teacher's outputs, or mix both. Such a student can
beat both U and R_ℓ, and with unequal resources π can go negative for reasons unrelated to the defense.

**Definitions (replace §8.3).** At level ℓ:
- R_ℓ: best student from the level-ℓ resources alone (unchanged).
- A_ℓ: best student from the level-ℓ resources plus the **defended** outputs (includes D and R_ℓ).
- U_ℓ: best student from the level-ℓ resources plus the **undefended** outputs (includes plain U).
- π_ℓ = (U_ℓ − A_ℓ) / (U_ℓ − R_ℓ), undefined when U_ℓ ≤ R_ℓ.
"Best" is the best of a stated list of recipes, a lower bound on the true best, and is reported as such.

**Necessity (replaces P2's U).** U_ℓ − R_ℓ must exceed the interval across seeds and a quarter of U_ℓ's gain over the
untrained student. Earlier results compared plain U with R_ℓ; they stay valid as "plain distillation is no better than
the attacker's own data", but the necessity verdicts are provisional until U_ℓ is measured.

**Combination recipes for U_ℓ (setup 2: GSM8K and MATH, levels B and C).** Llama-3.2-3B, the setup's recipe and
settings (LoRA r128, 3 epochs, lr 5e-4, micro-batch 6 × 4, cut-off 2,048), scored with the setup's own evaluation
(HF loop, 1,024 tokens), each on the same seed split as U:
- *Mix*: one training set holding both the attacker's Solo solutions and the teacher's solutions for every problem
  (2 × 5,238 or 2 × 5,000 examples, 3 epochs).
- *Sequential*: train on the Solo solutions, merge, then train a new LoRA on the teacher's solutions (3 + 3 epochs).
Level B uses Solo from Llama-3.2-3B-Instruct, level C from Qwen2.5-7B-Instruct.
Seed 123: both recipes, both levels, both datasets (8 runs). Seeds 124–125: per dataset and level, only the recipe
that scored higher on seed 123 (8 runs). Choosing on seed 123 can only raise U_ℓ, i.e. it favours passing.
U_ℓ per seed = max(plain U, the combination(s) run on that seed); R_B = max(Solo B, level-A self-training);
R_C = max(R_B, Solo C).

**Reading.** If U_ℓ − R_ℓ passes the margin, the setup passes necessity at level ℓ after all (the teacher's outputs
add value on top of the attacker's own data) and protection is measured against U_ℓ. If not, the earlier verdict
holds in the stronger sense: the teacher's outputs add nothing the attacker cannot already get.

**LIMO (setup 1) and the MATH pilot.** LIMO: one *mix* student (7B Solo solutions + LIMO's, 1,600 examples, seed 42)
after the pilot frees four GPUs; more seeds only if it beats Solo by the margin. MATH pilot (§10.1): if U_long alone does
not pass, the same two recipes with the long-budget teacher are run on seed 123 before the pilot is read as a failure.
Scripts: `scripts/dg_train_from_traces.py` (several solution files, `--init`, `--eval`), `scripts/42_combo_pool.sh`.

**Seed-123 results so far (2026-10-28 07:00; GSM8K-Platinum or MATH test, setup's own evaluation).** GSM8K B sequential
and MATH C sequential have not run (their GPU was held by an orphaned vLLM process); MATH C mix is running.

| setup, level | plain U | mix | sequential | U_ℓ so far | R_ℓ | U_ℓ − R_ℓ | margin (¼ of U_ℓ − B) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GSM8K, B | 59.64 | **62.78** | – | 62.78 | 59.80 (Solo B) | +3.0 | 15.6 |
| GSM8K, C | 59.64 | **61.21** | 59.88 | 61.21 | 59.80 (Solo B) | +1.4 | 15.2 |
| MATH, B | 14.58 | **17.02** | 16.04 | 17.02 | 14.90 (Solo B) | +2.1 | 4.1 |
| MATH, C | 14.58 | running | – | | 17.18 (Solo C) | | |

- Mixing the attacker's own solutions with the teacher's gives the best student in every completed cell, 2–3 points
  above both plain U and Solo. So the teacher's outputs do add something on top of the attacker's data, but far less
  than the margin: on seed 123, necessity fails at levels B and C for GSM8K and at level B for MATH even against U_ℓ.
- Seeds 124–125 run the recipe that won on seed 123 where both have run (GSM8K C: mix; MATH B: mix).

## Sources

- Kim et al., arXiv 2603.15500 (local PDF, Tables 5 and 7, Appendix E).
- Ding et al. (PART), arXiv 2510.11545 — `results/m4_part_design.md` §0–1.
- Savani et al., Antidistillation Sampling, arXiv 2504.13146.
- Li et al., DOGe, arXiv 2505.19504.
- Allouah et al., The Distillation Game, arXiv 2605.22737 (Table 1, Appendix C.2).
- Ma et al., Trace rewriting, arXiv 2602.15143.
- Hartman et al., TARS, arXiv 2604.23238; TraceGuard, ICML 2026 workshop (OpenReview vW0SgjtZlU).
- Jiang, DistillGuard, arXiv 2603.07835 (Appendix B).
- Javaheri et al., arXiv 2609.35699 (Table 1).
- Diao et al., LADS, arXiv 2605.18829; Fang et al., arXiv 2602.03396 (not yet read in full).
- Our NeurIPS 2026 paper, *Evaluating an Evaluation* (`../papers/NeurIPS_2026_Unlearning_Camera_Ready.pdf`), Definition 1, Proposition 1, Section 6.8.
