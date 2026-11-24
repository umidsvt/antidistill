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

**Stopped by the machine rebuild (2026-10-29).** MATH C mix was running and GSM8K B sequential and MATH C sequential
were waiting; all setup-2 outputs, including the solution files these students train on, were deleted. The seed-123
results above stay as recorded.

## 12. Regenerating setup 2 after the machine rebuild (fixed 2026-10-31, before any result; go-ahead from the user)

**Why.** The rebuild deleted every setup-2 output, including the teacher and Solo solution files that the combination
students of §11 train on. Re-running the grid regenerates them and checks the rebuilt environment against the recorded
numbers.

**What is run.** The unchanged configs `configs/setup2/{gsm8k,math}_{U,solo_l3i,solo_q7i}_s{123,124,125}.yaml` (18
distillation runs) and the untrained student for each dataset and seed (6 runs), with the same code, the same three
patches and `.venv-dg` installed from `scripts/env_dg.lock.txt`. One GPU per job on GPUs 0–3, with
`scripts/33_setup2_pool.sh` and the job list `outputs/setup2/regen_jobs.txt`. A smoke run comes first
(`configs/setup2/smoke.yaml`, 48 problems): it only checks that the pipeline runs end to end and has no threshold.
Differences from the first runs: the host (driver 615.71 and CUDA 13.4 instead of 580.95 and 13.0) and the CPU and
memory pinning of each job (`scripts/gpu_env.sh`). Nothing in the recipe changes. Sampling at temperature 0.6 need not
reproduce the lost solutions token for token.

**Rule (reproduction).** For each dataset, and for each of U, Solo B, Solo C and the untrained student, the new 3-seed
mean must lie within 1.5 points (GSM8K-Platinum, 1,209 problems) or 1.0 point (MATH, 5,000 problems) of the recorded
mean (§9.1, §9.3, §9.4). Each solution writer's own test accuracy must lie within 1.0 point of its recorded mean. The
tolerances are about twice the standard error of a 3-seed mean on each test set.
- **All within:** the environment reproduces the setup. The recorded numbers stay in the paper unchanged, and the
  regenerated runs are reported as a reproduction, with the §7 necessity verdicts recomputed on them. The regenerated
  solution files become the inputs of the combination students. Those students are compared only with the U and Solo
  students trained on the same regenerated files, never with the recorded ones. They need their own go-ahead.
- **Any mean outside:** stop and report it. Nothing is built on the regenerated files until the cause is found and the
  authors decide. The recorded numbers are not replaced.

**Result (complete 2026-11-01 01:24; smoke run passed 2026-10-31 00:09).** All 24 jobs ran on the first attempt.
Numbers recomputed from the per-problem files (`outputs/setup2/<run>/student/teacher_standard/test_naive_beta_1.0.json`,
`.../teacher/{test,train}_standard.json`, `outputs/setup2/base_<dataset>_s<seed>.json`), not from logs.

| dataset | student | rerun, seeds 123 / 124 / 125 | rerun mean | recorded mean | difference | within |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K-Platinum | untrained | 0.41 / 0.66 / 0.66% | 0.58% | 0.74% | −0.17 | yes (1.5) |
| | U | 59.31 / 60.22 / 59.80% | 59.77% | 59.00% | +0.77 | yes |
| | Solo B (Llama-3.2-3B-Instruct) | 61.62 / 57.82 / 59.80% | 59.75% | 59.44% | +0.31 | yes |
| | Solo C (Qwen2.5-7B-Instruct) | 60.05 / 58.81 / 57.82% | 58.89% | 59.83% | −0.94 | yes |
| MATH | untrained | 0.78 / 1.04 / 0.72% | 0.85% | 0.77% | +0.08 | yes (1.0) |
| | U | 15.30 / 15.42 / 15.70% | 15.47% | 14.99% | +0.48 | yes |
| | Solo B | 15.44 / 15.34 / 15.56% | 15.45% | 14.87% | +0.57 | yes |
| | Solo C | 17.46 / 17.18 / 18.14% | 17.59% | 17.34% | +0.25 | yes |

Each writer's own test accuracy, rerun mean vs recorded mean (tolerance 1.0): GSM8K teacher 87.15 vs 87.17, Qwen2.5-7B-
Instruct 94.18 vs 94.20, Llama-3.2-3B-Instruct 81.69 vs 81.30; MATH teacher 63.07 vs 63.10, Qwen2.5-7B-Instruct 73.11
vs 73.10, Llama-3.2-3B-Instruct 47.95 vs 48.00. All within.

- **By the rule, all within: the rebuilt environment reproduces setup 2.** The recorded numbers stay in the paper; the
  rerun is reported as a reproduction. The regenerated solution files are now the inputs for the combination students
  (their go-ahead is still pending).
- **Necessity verdicts recomputed on the rerun (§7 rule): unchanged.** GSM8K, margin (59.77 − 0.58)/4 = 14.8: U − Solo
  B = −2.3, +2.4, 0.0 (mean +0.0, 95% interval −5.8 to +5.9); U − Solo C = −0.7, +1.4, +2.0 (mean +0.9, −2.7 to +4.5).
  MATH, margin (15.47 − 0.85)/4 = 3.7: U − Solo B = −0.1, +0.1, +0.1 (mean +0.0); U − Solo C = −2.2, −1.8, −2.4 (mean
  −2.1, −3.0 to −1.3; Solo C above U on every seed). Necessity fails at levels B and C on both datasets.
- **New observation: Llama-3.2-3B-Instruct solves far more training problems than test problems.** The pipeline also
  scores each writer on the training problems it writes solutions for (`teacher/train_standard.json`):

  | writer | GSM8K train (5,238) | GSM8K-Platinum test | MATH train (5,000) | MATH test |
  | --- | --- | --- | --- | --- |
  | R1-Distill-Qwen-7B (teacher) | 87.9% | 87.2% | 62.0% | 63.1% |
  | Qwen2.5-7B-Instruct | 95.1% | 94.2% | 76.0% | 73.1% |
  | Llama-3.2-3B-Instruct | **89.6%** | 81.7% | **71.5%** | 48.0% |

  Means over the three seeds. The other two writers score within 3 points on both splits; Llama-3.2-3B-Instruct is
  8 points higher on GSM8K's training problems and 23.5 points higher on MATH's. It has most likely been trained on
  both training sets. Consequence for the paper: on the problems the student is trained on, this writer is *not*
  weaker than the teacher (GSM8K 89.6% vs 87.9%; MATH 71.5% vs 62.0%), so the sentences that Solo B matches U although
  its writer "is less accurate than the teacher" (main.tex, GSM8K) and "solves 15 points fewer test problems" (MATH)
  must report training-problem accuracy too. The verdicts do not change; Solo B is still a level-B attacker's student.
  Any later setup should measure writer–teacher gaps on the training problems (`results/plan_2026_11_protection_map.md`
  §3).

## 13. Level A with RL after self-training (fixed 2026-11-01, before any result; go-ahead from the user)

**Why.** Our four RL recipes from the untrained student did not learn (§9.3): it almost never ends with a correct boxed
answer, so GRPO has nothing to reinforce. A labmate reports 62.9% on GSM8K-Platinum (one run, seed 456, about ±2 points
between runs) with SFT then GRPO: SFT on solutions written by Qwen2.5-3B-Instruct and filtered by the PoE-defended
teacher's final answers (59.3%), then GRPO rewarded by those answers. **That copier owns an open model
(Qwen2.5-3B-Instruct), so it is not level A** (§9.3: level A owns only the student). The level-A version keeps their
SFT-then-GRPO structure and replaces their step 1 by the student's own few-shot self-training (§9.3), which uses only the
teacher's released answers.

**What is run.** Seeds 123–125, one GPU each (group 0–3), `scripts/46_levelA_star_grpo.sh`:
1. Few-shot STaR exactly as §9.3 (`scripts/dg_star.py --fewshot`, teacher answers from the regenerated U runs of §12),
   now also saving its LoRA adapter. This regenerates the files lost in the rebuild.
2. GRPO continuing that same LoRA (`scripts/dg_rl_base.py --init-adapter`): reward 1 when the boxed answer matches the
   undefended teacher's released answer, else 0; 8 problems × 8 samples per step, temperature 1.0, top-p 1.0, prompts up
   to 512 tokens, completions up to 512 tokens, 300 steps, learning rate 2e-5 (cosine, 3% warm-up), no KL penalty.
   Scored with the setup's own evaluation, as U. Differences from the labmate's recipe besides step 1: TRL 0.19.1 with
   vLLM 0.8.5 (theirs 0.23), its default loss (`bnpo`), the cosine schedule, and our setup's evaluation.
3. A smoke test first (a few problems and steps), which only checks that the pipeline runs and has no threshold.

**Rule.**
- **STaR reproduction (gate):** the regenerated STaR 3-seed mean must lie within 1.5 points of the recorded 34.3%
  (§9.3, tolerance of §12). If not: stop and report; the GRPO result is not read.
- **A failed recipe is not a result:** if GRPO collapses (mean completion under 20 tokens over the last 50 steps, or
  correctness reward falling below its value at step 0), it is reported as a failed recipe, as in §9.3.
- **Necessity at level A** (§7, compared with the regenerated U of the same seed, §12; margin a quarter of U's gain,
  (59.77 − 0.58)/4 = 14.8): R_A per seed = max(STaR, STaR + GRPO).
  - **Passes** if the paired U − R_A has a 3-seed mean above 14.8 and its 95% interval excludes 0: level A still needs
    the teacher, now against a stronger copier, and R_A replaces 34.3% in the paper.
  - **Fails** if the mean is 14.8 or less: the GSM8K setup then cannot show protection at any level of the ladder; the
    paper's level-A pass is withdrawn and reported as such.
- The labmate's own recipe (with Qwen2.5-3B-Instruct) is a level-B copier and is not part of this rule.

**Result (complete 2026-11-01 16:36).** Smoke test passed after one fix (vLLM needs `--vllm-mem 0.5` next to the
training model, as in `scripts/32_setup2_rl_v3.sh`; the first attempt stopped at engine start, before any training).
GSM8K-Platinum, 1,209 problems, from the per-problem files (`test_star.json`, `test_rl.json`):

| seed | STaR (regenerated) | STaR + GRPO | GRPO − STaR | U (regenerated, §12) | U − R_A |
| --- | --- | --- | --- | --- | --- |
| 123 | 35.81% | 32.42% | −3.4 | 59.31% | +23.5 |
| 124 | 34.41% | 28.87% | −5.5 | 60.22% | +25.8 |
| 125 | 35.48% | 29.28% | −6.2 | 59.80% | +24.3 |
| mean | **35.24%** | **30.19%** | **−5.1** (95% interval −8.7 to −1.4) | 59.77% | **+24.5** (21.6 to 27.5) |

STaR kept 3,187 / 3,148 / 3,203 of its own solutions (recorded 3,123 / 3,141 / 3,168) and left 34 / 35 / 25 problems
unsolved. GRPO training: correctness reward (8 samples at temperature 1.0 on 8 new training problems per step, so
noisy) first logged 0.28 / 0.35 / 0.33, mean over the last 50 steps 0.48 / 0.44 / 0.49; mean completion length over
the last 50 steps 118 / 89 / 124 tokens.

- **STaR reproduction gate: passes** (35.2% vs 34.3%, +0.9, tolerance 1.5).
- **No collapse** by the rule (lengths far above 20 tokens; reward not below its first value), so GRPO counts as a run
  recipe, and **it lowers the test score on every seed** (−5.1 points on average) although its reward on training
  problems rises a little. The cause is not known. Candidates, none tested: our TRL 0.19.1 (`bnpo` loss) against the
  labmate's 0.23; the gap between training (temperature 1.0, 512 tokens) and evaluation (the setup's sampling with
  answer forcing); and the weaker starting student (35% vs their 59%). The GRPO adapters were not saved, so the
  policies cannot be re-scored.
- **Necessity at level A: passes.** R_A = max(STaR, STaR + GRPO) = STaR on every seed; U − R_A = +24.5 (interval 21.6 to
  27.5) against the margin of 14.8. The GSM8K setup still needs the teacher against a copier who owns only the student,
  and the labmate's GRPO step does not change that here.
- **Paper (decision for the authors):** the rule says R_A replaces 34.3%, but R_A turned out to be the regenerated run
  of the same self-training. Proposed: keep 34.3% next to the recorded U (59.0%), as §12 does for the other students,
  and report the rerun (35.2%) and the GRPO attempt (30.2%) in Appendix B.

## 14. Level A by GRPO alone from the base student (fixed 2026-11-01, before any result; go-ahead from the user)

**Why.** A labmate asks for the simplest level-A copier: GRPO only, from base Llama-3.2-3B, no SFT, the student
reasoning freely, rewarded when its final answer matches the teacher's released answer, plus a format reward. Our v1–v3
(§9.3) were this recipe and did not learn: under the setup's chat prompt the base model almost never ends with a correct
answer (617 of 5,238 problems solved in 8 tries, §9.3), and both format rewards were gamed (v2: `\boxed{` then rambling;
v3a: a bare `\boxed{n}`). Javaheri et al. (2026, Appendix A) report 40.7% on GSM8K for the same recipe family (GRPO only
from base Llama-3.2-3B, SimpleRL-Zoo, a plain-text prompt, batch 1,024, rewarded by correct answers). A level-A copier
near 45% on our test would remove the setup's only pass (§13: U − R_A = 24.5 against a margin of 14.8).

**What is run.** `scripts/47_levelA_grpo_only.sh`, GPUs 4–7, two prompt arms that differ only in the prompt:
- **Arm C:** the setup's chat prompt (as v1–v3).
- **Arm P:** a plain-text prompt, no chat template: `Question: {question}\nPlease reason step by step, and put your
  final answer within \boxed{}.\nAnswer: Let's think step by step.`
- **Both:** a new LoRA on base Llama-3.2-3B (rank 128, the capacity of every student here); reward 1.0 when the answer
  matches the teacher's released answer (the setup's checker), plus 0.2 when the completion has exactly one boxed answer,
  ends with it, and has at least 20 words of reasoning before it (closes both earlier hacks); GRPO with 8 problems × 8
  samples, temperature 1.0, top-p 1.0, completions up to 512 tokens, learning rate 2e-5 (cosine, 3% warm-up), no KL,
  TRL 0.19.1 with vLLM 0.8.5. Adapters are saved.
- **Prompt check first** (descriptive, no threshold): the base model's per-sample and 8-try success against the teacher's
  answers on 512 training problems under each prompt.
- **Pilot:** seed 123, 60 steps per arm, scored on 300 test problems. An arm passes if (i) its correctness reward over
  steps 41–60 averages at least 0.05, (ii) its mean completion length over steps 41–60 is at least 50 tokens, and (iii)
  it scores at least 5% on the 300 problems. Passing arms run 600 steps (one pass over 4,800 problems) for seeds
  123–125. If no arm passes: reported as failed recipes; no fallback without the authors.
- **Scoring:** 1,209 test problems, the setup's sampling and answer forcing, each student with the prompt it was trained
  with (C: the setup's evaluation; P: the same with the plain prompt). Arm P is also scored with the chat prompt
  (reported only).

**Rule (necessity at level A, as §13).** R_A per seed = the best of STaR (§13) and the passing GRPO-only arms; compared
with the regenerated U of the same seed, margin 14.8.
- **Passes** if the 3-seed mean of U − R_A is above 14.8 and its 95% interval excludes 0: level A still needs the teacher.
- **Fails** otherwise: the GSM8K setup cannot show protection at any level of the ladder, and the paper's level-A pass is
  withdrawn.

**Prompt check (2026-11-01 20:43; 512 training problems × 8 samples, temperature 1.0, 512 tokens).** Under both prompts
the untrained student never ends an answer (100% of samples reach the 512-token limit).

| prompt | samples matching the teacher's answer | problems with a matching sample | format reward fires |
| --- | --- | --- | --- |
| C, chat | 1.0% | 6.3% | 0.0% |
| P, plain | 2.3% | 17.8% | 0.02% |

**Pilot (seed 123, 60 steps; complete 2026-11-01 22:35).**

| arm | correctness reward, steps 41–60 | format reward | mean length | test, 300 problems | pilot |
| --- | --- | --- | --- | --- | --- |
| C | 0.012 | 0.000 | 323 tokens | 4.0% | **fails** (i), (iii) |
| P | 0.055 (0.019 at step 5) | 0.282 (0.031 at step 5) | 130 tokens (228 at step 5) | **27.3%** (chat prompt: 0.0%) | **passes** |

Two incidents, neither a change to the recipe: arm C's first attempt ran out of memory and its retry ran normally; arm
P's three attempts each finished training and then crashed when scoring (the plain-prompt tokenizer wrapper lacked
`__call__`, which answer forcing uses). After the fix, the adapter saved by the third attempt was scored with
`--eval-only`; its training log is the one above. By the rule, arm P runs 600 steps for seeds 123–125 (started
2026-11-01 22:40, GPUs 4–6); arm C stops here.

**Full runs, arm P (complete 2026-11-02 10:56).** Test, 1,209 problems, plain prompt: **2.2 / 2.0 / 2.2%** (seeds
123–125; chat prompt 0.1 / 0.8 / 0.0%). Every seed **collapsed**: training improved until about step 55–105 (correctness
reward 0.11 / 0.13 / 0.20, format reward up to 0.88 / 0.59 / 0.90, about 100-token answers), then the format reward fell
to 0, answers ran to the 512-token limit (seed 125: every sample from step 155 on; seed 123 also passed through
20-token answers) and correctness fell to 0.00–0.04, below its first logged value (0.016 / 0.031 / 0.034); 95–100% of
groups then had identical rewards, so GRPO could not recover. The learning-rate schedule explains why the pilot did
not show this: with 60 steps the cosine schedule had decayed to 3.5e-6 by step 45 and about 0 by step 60, so the pilot
stopped at its best point; the 600-step runs stayed at 1.7–2.0e-5 from step 30 to 155, where all three collapsed.
Several attempts also ran out of memory or lost their Hugging Face connection after 3–4 hours and restarted from
scratch (the retry logic in `scripts/47_levelA_grpo_only.sh`); the logged runs are the attempts that finished.

- **By the §13 collapse criterion, the three full runs are failed recipes, not results:** they do not show what GRPO
  alone can reach. The best GRPO-only student observed is the pilot's, 27.3% on 300 test problems (seed 123, 60 steps).
- **Necessity at level A (rule above): passes.** R_A = STaR on every seed (35.81 / 34.41 / 35.48%); U − R_A = +24.5
  (21.6 to 27.5) against a margin of 14.8, as in §13. The pass holds against every level-A copier built so far (STaR,
  STaR + GRPO, GRPO alone), but a stable GRPO-only run has not been obtained, so it is not yet tested against the
  strongest one. A fix (shorter or lower learning-rate schedule, a KL penalty, or checkpoint selection on held-out
  training problems scored by the teacher's answers) is a recipe change for the authors to decide (rule 1).

## 15. Level A by GRPO alone, with Javaheri et al.'s stabilizers (fixed 2026-11-03, before any result; option B chosen by the user)

**Why.** §14's GRPO-only runs collapsed after about 100 steps. Javaheri et al.'s code (`third_party/ext/treason`,
commit 127de4c: `simpleRL/train_llama3B_rl.sh`, `train_grpo_math_tune_ray.sh`, SimpleRL-Zoo on verl) trains GRPO alone from
base Llama-3.2-3B with what our runs lacked: 1,024 problems × 8 samples per step, a KL penalty to the starting model (1e-4,
`low_var_kl`), an entropy bonus (0.001), learning rate 5e-7 on all parameters, answers up to 2,048 tokens, and the reward
+1 correct / −0.5 wrong / −1 without a boxed answer. Option A (their code unchanged) is deferred; option B ports their
settings into our TRL script.

**What is run.** `scripts/48_levelA_grpo_stable.sh`, seeds 123–125, one GPU each (GPUs 4–6). Level A: only the base student
and the teacher's released answers; GSM8K only.
- **Data:** the setup's 5,238 training problems with the teacher's answers. 256 of them (the last 256 of the seeded split)
  are held out for checkpoint selection; GRPO trains on the other 4,982. No MATH data, no test data.
- **Prompt:** §14's plain prompt (their "simple" prompt text is not used: it sits in their data files, which are not read).
- **Reward (theirs):** +1.0 if the completion has a boxed answer that matches the teacher's answer (the setup's checker),
  −0.5 if it has a boxed answer that does not, −1.0 if it has none. No other reward.
- **GRPO:** a new LoRA (rank 128); 32 problems × 8 samples per step (256 samples; 32 times ours, a quarter of their
  1,024 problems); learning rate 5e-6, constant after a 3% warm-up (their 5e-7 on all parameters, times 10 for LoRA);
  KL penalty 1e-4 (TRL's `beta`); no entropy bonus (TRL 0.19 has none); temperature 1.0, top-p 1.0; answers up to 2,048
  tokens; 300 steps (about 1.9 passes over the 4,982 problems).
- **Checkpoint selection (what a copier can do):** the adapter is saved every 25 steps; each saved adapter is scored on
  the 256 held-out training problems against the teacher's answers (the setup's sampling and answer forcing); the best one
  is kept (ties: the earlier step). No test score is used to choose.
- **Scoring:** the kept adapter and the last one, on the 1,209 test problems, with the plain prompt, the setup's sampling
  and answer forcing; the kept adapter also greedily (temperature 0, as Javaheri et al. score; reported only).
- **Smoke test first** (a few steps; no threshold).

**Rule.**
- **Not a failed recipe** unless the kept checkpoint is the one at step 25 and the correctness reward over the last 50
  steps is below its first logged value (that is, training only ever made it worse).
- **Necessity at level A (as §13).** R_A per seed = the best of STaR (§13: 35.81 / 34.41 / 35.48%) and this run's kept
  adapter (setup's evaluation); compared with the regenerated U of the same seed (§12), margin 14.8.
  - **Passes** if the 3-seed mean of U − R_A is above 14.8 and its 95% interval excludes 0.
  - **Fails** otherwise: the GSM8K setup cannot show protection at any level of the ladder; the paper's level-A pass is
    withdrawn.
- If this run is a failed recipe, option A (their code unchanged, adapted for level A) is the next step, with its own rule.

**Result (complete 2026-11-04 18:56; recorded 2026-11-08).** Smoke test passed after one fix: TRL 0.19.1 computes the
reference model's log-probabilities over the whole generation batch at once when `beta > 0` (31 GiB, out of memory);
`ChunkedGRPOTrainer` in `scripts/dg_rl_base.py` passes the micro-batch size, as TRL does for every other call. Full runs
on GPUs 4, 6, 7 (GPU 5 was taken by another lab member), about 22 hours each. GSM8K-Platinum, 1,209 problems, from the
per-problem files:

| seed | kept step (held-out score) | kept adapter | last adapter | kept, greedy (reported only) | STaR (§13) | U (§12) | U − R_A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | 250 (32%) | 29.11% | 30.11% | 31.68% | 35.81% | 59.31% | +23.5 |
| 124 | 275 (30%) | 28.29% | 25.89% | 29.69% | 34.41% | 60.22% | +25.8 |
| 125 | 300 (29%) | 31.35% | 31.35% | 33.75% | 35.48% | 59.80% | +24.3 |
| mean | | **29.58%** | 29.11% | 31.71% | **35.24%** | 59.77% | **+24.5** (21.6 to 27.5) |

Training: no collapse. The correctness reward stays near its start (0.03–0.04) for about 150 steps while answers first
grow (to about 600 tokens) and then shorten, and rises from about step 180 to 0.12 / 0.15 / 0.12 over the last 50 steps;
the held-out scores fall from 17–22% at step 25 to 9–16% around steps 75–125 and climb to their best at steps 250–300, so
every run was still improving when it stopped.

- **Not a failed recipe** by the rule (kept steps 250 / 275 / 300; reward well above its first value).
- **Necessity at level A: passes.** R_A = STaR on every seed; U − R_A = +24.5 (21.6 to 27.5) against a margin of 14.8.
  GRPO alone with the stabilizers reaches 29.6% (31.7% greedy), below STaR's 35.2%; the copier would need about 45% to
  remove the pass.
- **Caveat:** the runs were still improving at step 300 (about 1.9 passes over the training problems; Javaheri et al.
  train for many more updates on four times as many problems per step), so this is not a ceiling for GRPO alone.

**Continuation (fixed 2026-11-08, before any result; option 1 chosen by the user).** Each seed continues for 300 more
steps from its step-300 adapter (`outputs/setup2/grpo_stable/s<seed>/ckpt_0300`), with every §15 setting unchanged
(`CONT=1 scripts/48_levelA_grpo_stable.sh`, outputs in `s<seed>_cont/`). Differences, all mechanical: the optimizer
state is not carried over (a fresh optimizer with the same 3% warm-up), and the problem order restarts from the seeded
shuffle, so steps 301–600 replay the order of steps 1–300. The KL reference stays the base model. Adapters are saved
every 25 steps (steps 325–600) and scored on the same 256 held-out training problems.
- **Kept adapter:** the best held-out score over all 24 adapters of the seed (steps 25–600; ties: the earlier step).
- **Rule:** as §15. R_A per seed = the best of STaR and the kept adapter's test score; necessity at level A passes if
  the 3-seed mean of U − R_A is above 14.8 and its 95% interval excludes 0, and fails otherwise.

**Continuation result (complete 2026-11-09 16:28; recorded 2026-11-11).** GPUs 4, 6, 1; about 20–27 hours of training
per seed. The continuation script chose only among its own 12 adapters. The rule's choice among all 24 adapters was
made afterwards from the stored `select_ckpt_*.json` files of both runs (256 held-out training problems, teacher's
answers). Test scores are from the per-problem files (GSM8K-Platinum, 1,209 problems):

| seed | kept step (held-out) | kept adapter | greedy (reported only) | step 600 | STaR (§13) | R_A | U (§12) | U − R_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | 475 (35.5%) | 35.32% | 35.24% | 34.82% | 35.81% | STaR | 59.31% | +23.5 |
| 124 | 275 (30.1%), from the first 300 steps | 28.29% | 29.69% | 21.17% | 34.41% | STaR | 60.22% | +25.8 |
| 125 | 350 (32.4%) | 32.59% | 34.08% | 8.60% | 35.48% | STaR | 59.80% | +24.3 |
| mean | | **32.07%** | 33.00% | 21.53% | 35.24% | | 59.77% | **+24.5** (21.6 to 27.5) |

- **Necessity at level A: passes.** R_A = STaR on every seed, so the verdict and the numbers are those of §15.
- **More GRPO helped two seeds and then became unstable.** The best adapters rose from 29.6% (§15) to 32.1%. Seed 123
  reached 35.3%, within 0.5 points of its STaR. But seeds 124 and 125 collapsed late in the continuation: their held-out
  scores fell to 4% around steps 450–500, and their step-600 adapters score 21.2% and 8.6%.
- With these settings, GRPO alone from the base student does not overtake one round of STaR, and stays more than 24
  points below the teacher-trained student.

## 16. Answer distributions of the attacker's own models (fixed 2026-11-08, before any result; go-ahead from the user)

**Why.** At the 8 November meeting the professor proposed framing the paper around answer-only attacks: the attacker
may own models at levels A–D but takes from the teacher only its released final answers (the plan document is not
written yet). The starting point: an attack that keeps the attacker's own solutions only when
they reach the teacher's answer aims at the attacker's model conditioned on that answer. This covers STaR, rejection
sampling, several rounds of either, and RL rewarded by the answer with a KL penalty toward the starting model. That
target is exactly −log2 p bits (KL) from the model, where p is the probability that the model's answer matches the
teacher's. This step measures p for every training problem and every owned model. Nothing is trained.

**What is run.** `scripts/49_answer_dist.sh`, which calls `scripts/dg_answer_dist.py` (phases `gen` and `grade`).
- **Pools:** seed 123's training pools, GSM8K 5,238 problems and MATH 5,000 problems, with the teacher's answers from
  the regenerated U runs (§12).
- **Models:**
  - A: base Llama-3.2-3B with a few-shot raw-text prompt. GSM8K uses Wei et al.'s 8 examples, as in §9.3. MATH uses the
    4 examples of Lewkowycz et al. (2022), which is new: the MATH level-A cell has not been run before. Those examples
    are MATH reference solutions, written by people.
  - B: Llama-3.2-3B-Instruct, with the setup's chat prompt and answer forcing.
  - C: Qwen2.5-7B-Instruct, the same as B.
  - No level D: in setup 2 the level-D model is the teacher.
- **Sampling:** 16 tries per problem at temperature 0.6 and top-p 0.95, the setup's own sampling (STaR used 0.8), up to
  1,024 tokens, vLLM seed 123.
- **Storage:** every try, with its text, its final answer, whether it matches the teacher's answer and whether it
  matches the true solution. Both checks use the setup's checker, the first as in STaR.
- **GPUs** 2, 3 and 7, one model per GPU at a time.
- **Pilot first:** the first 200 problems of each pool, to check prompts and answer extraction and to time the runs.
  The pilot has no threshold.

**Quantities.** `scripts/answer_bits.py` computes these from the stored `tries.jsonl`, never from logs. At level ℓ the
attacker's tries are the pooled tries of the models it owns (A: A; B: A and B; C: A, B and C), 16 per model, equal
weight.
- p_i: the fraction of those tries whose answer matches the teacher's answer for problem i.
- Surprisal s_i = −log2 p_i, defined when p_i > 0; such a problem is *reachable*.
- Per level: the mean surprisal over reachable problems, the number of unreachable problems, the histogram of p_i, and
  the share of problems whose majority answer (normalized answer string) is the teacher's. The same quantities against
  the true answers are reported as a diagnostic.
- B alone and C alone are also reported.
- Resolution: the largest measurable surprisal is 4 bits at A (16 tries), 5 at B (32) and 5.6 at C (48).

**What the outcomes mean.** This step gives no necessity verdict. It decides what is trained next.
- **GSM8K.** The prediction from §12–15: the teacher's answers carry little that an owned open model does not already
  give. Concretely, mean surprisal at B and at C is at most half of A's, and the share of problems where the attacker's
  majority answer is the teacher's is at least 20 points higher than at A. If this holds, the measure is consistent
  with the answers being worth 34.6 points at A (STaR 35.2% against 0.6% untrained) and with Solo at B and C needing no
  answers at all (−0.4 and −0.8 points against U). If it fails, the surprisal does not explain the pattern, and this
  is recorded.
- **MATH.** If mean surprisal at both B and C is higher than on GSM8K, answer-filtered students at B and C on MATH come
  next, under their own rule. If not, they are deprioritized.
- **The tries are kept** as the solution pools for later answer-filtered students, so those need no new generation.
  No problem is removed. Unreachable problems are counted and reported.

**Pilot (2026-11-08 19:07 – 20:11; first 200 problems of each pool, no threshold).** All six runs generated and were
graded. Answer extraction was checked by hand on samples of every prompt format.
- Times: GSM8K 1.0 / 2.0 / 3.6 minutes for A / B / C; MATH 3.5 / 5.2 / 8.1 minutes.
- Llama-3.2-3B-Instruct almost never writes a boxed answer under the setup's prompt; on GSM8K, 3,148 of its 3,200 tries
  get their answer from answer forcing, as in the setup's evaluation and the Solo runs.
- Two runs (MATH A, GSM8K C) wrote their output and then hung at exit for about 50 minutes; they were stopped by PID.
  `scripts/49_answer_dist.sh` now has a watchdog that stops a run 2 minutes after its last file is written.

**Deviation (2026-11-09, found in the pilot, before the full run): the teacher's reference answer.** The stored
`extracted_answer` is normalized by the setup, which deletes commas: `(0, 0, 3)` becomes `(0 0 3)` and `-3, 5` becomes
`-3 5`. So multi-part MATH answers (tuples, lists of solutions, intervals) never matched the teacher, even when right.
On the pilot this hit 13 of 200 MATH problems.
- **Fix:** the reference is now the teacher's own boxed text (`teacher_ref` in `scripts/dg_answer_dist.py`), taken
  from the same source the setup used: the raw solution if it had an answer, else the answer-forced ending.
- **Check on the full seed-123 pools:** the rebuilt reference reproduces the setup's verdict on the teacher's
  correctness for 5,222 of 5,238 GSM8K and 4,909 of 4,968 MATH problems. The 59 MATH exceptions come from the setup's
  checker, which tests the teacher's whole solution text and so credits a partial boxed answer when the full true
  answer appears elsewhere (`train_23`: boxed −3, true answer "−3 or 5").
- **Pilot regraded:** on tries where the teacher is right, the teacher check and the truth check now disagree on 0 /
  45 / 64 of 2,080 MATH tries (A / B / C), down from 14 / 159 / 207. On GSM8K they disagree on 0 / 4 / 11 of 2,816.
- **Scope:** STaR (§9.3, §13) and GRPO (§14–15) are GSM8K-only. GSM8K answers are single numbers, so the comma
  deletion only removes thousands separators there, which is correct; those results are unaffected.

**Full run (2026-11-09 00:19 – 03:46; GPUs 2, 3, 7).** Five of the six runs completed: GSM8K A, B and C; MATH A and B.
The watchdog stopped MATH B once, after its output was written.
- **MATH C failed twice at startup.** One MATH prompt is 1,717 tokens under Qwen's tokenizer, longer than the
  1,664-token model length. Nothing was generated. The setup itself truncates prompts at 512 tokens (Solo and U); our
  runs do not truncate. Under the Llama and Qwen tokenizers, 58 and 56 of the 5,000 MATH prompts exceed 512 tokens; the
  longest are 1,639 and 1,717. So MATH B ran with every prompt complete. Proposed fix, not yet run: a model length that
  fits the longest prompt (3,072 tokens) for MATH C, keeping prompts complete as for A and B.

**Result, GSM8K (recorded 2026-11-11, from `tries.jsonl` via `scripts/answer_bits.py`,
`outputs/setup2_answers/bits_gsm8k_s123.json`).** 5,238 problems; the teacher's answer is wrong on 618.

| level (tries pooled) | mean surprisal, reachable (bits) | unreachable | majority answer = teacher's | majority = true answer |
| --- | --- | --- | --- | --- |
| A (16) | 1.926 | 1,533 | 39.9% | 40.5% |
| B: A + B (32) | 1.078 | 265 | 84.5% | 91.5% |
| C: A + B + C (48) | 0.751 | 239 | 88.6% | 96.8% |
| B alone (16) | 0.386 | 376 | 86.1% | 93.8% |
| C alone (16) | 0.097 | 479 | 88.5% | 96.3% |

- **Verdict on the GSM8K prediction: holds at C, fails at B.**
  - Required: mean surprisal at most half of A's (0.963 bits), and the majority-equals-teacher share at least 20
    points higher than at A (59.9%).
  - C: 0.751 bits and 88.6%. Holds.
  - B: 1.078 bits, above the 0.963 bar, although its 84.5% passes the second condition. Fails.
- **Why B fails.** The rule pools the levels' models with equal weight, so at B half the tries come from the base
  student, which matches the teacher only 26% of the time. B alone carries 0.386 bits, a fifth of A's. Equal-weight
  pooling is a poor model of an attacker, who would draw most samples from its best model. Any change to the primary
  measure is a new rule for later steps, not a revision of this verdict.
- **Exploratory, not part of the rule: the teacher's errors.** Most unreachable problems at B and C are ones where the
  teacher's answer is wrong: 456 of C alone's 479. There the open models agree with the true answer instead. Restricted
  to the 4,620 problems where the teacher is right, mean surprisal is 1.863 bits at A, 0.263 for B alone and 0.042 for
  C alone. The surprisal against the teacher counts the teacher's mistakes as information. A later measure should
  report the two groups separately.

**Partial result, MATH (`bits_math_s123.json`).** 5,000 problems; MATH C is missing, so the MATH rule (B *and* C higher
than on GSM8K) cannot be judged yet.

| level (tries pooled) | mean surprisal, reachable (bits) | unreachable | majority answer = teacher's | majority = true answer |
| --- | --- | --- | --- | --- |
| A (16) | 2.789 | 3,001 | 15.8% | 15.3% |
| B: A + B (32) | 1.676 | 1,196 | 62.0% | 83.1% |
| B alone (16) | 0.725 | 1,380 | 62.6% | 85.6% |

B is higher than on GSM8K both pooled (1.676 against 1.078 bits) and alone (0.725 against 0.386).

## 17. Puzzle testbed, step 1: solve-rate pilot on Knights and Knaves (fixed 2026-11-13, before any result; step 0 approved by the user, the pilot run not yet)

**Why.** In both setups examined so far (GSM8K, MATH), open models train a student as good as the teacher's at levels
B–D, so a defense has nothing to protect there; only level A shows a gap. Both datasets are also probably in the open
models' training data. We need a task where the teacher clearly beats every model the attacker owns, which no model
has seen, and whose answers a program can check. Knights and Knaves puzzles meet the last two conditions by
construction. This pilot tests the first: is there a difficulty at which the teacher solves most puzzles and the
attacker's open models solve few?

**Puzzles.** Generated with Xie et al.'s (2024) generator (`third_party/ext/mem-kk-logic`, commit 35385cf, MIT;
`data_prep/lib_kk.py`). The generator, the puzzle text and the answer format are theirs, not ours.
- N people, N = 3 to 8. Each person makes one statement: a random tree of depth ≤ 2 and width ≤ 2 over "X is a
  knight/knave" with and / or / not / implies / iff, as in their main experiments.
- Only puzzles with exactly one solution are kept (their brute-force check).
- Fresh puzzles: sampler seed 20261113 + N, not their seed 1234. 100 puzzles per N, 600 in all
  (`scripts/kk_make.py` → `data/kk/pilot_test.jsonl`).
- Known limitation: an attacker who can program could write a solver (their code includes one) and would need no
  teacher. The testbed stands in for domains where no program solver exists.

**Models.** All in the lab cache; nothing is trained.

| role | model | max new tokens |
| --- | --- | --- |
| teacher (stands in for a closed frontier model) | DeepSeek-R1-Distill-Qwen-32B, 4 GPUs (tensor parallel) | 8,192 |
| B | Llama-3.2-3B-Instruct | 2,048 |
| C | Qwen2.5-7B-Instruct | 2,048 |
| D | DeepSeek-R1-Distill-Qwen-7B (the open reasoning model of setup 2's ladder) | 8,192 |
| D, newer (reported separately) | Qwen3-8B, thinking mode | 8,192 |

Level A (base Llama-3.2-3B) is left to step 2: it needs a few-shot prompt, and the go/no-go below depends only on
the best owned model.

**Generation** (`scripts/kk_pilot.py`, vLLM 0.8.5 in `.venv-dg`).
- Prompt: Xie et al.'s instruction (`dataset/prompt.py`, `system_instruction`: reason, then end with "CONCLUSION:" and
  one numbered line per person), followed by the puzzle, as the user message of each model's chat template.
- 4 tries per puzzle, temperature 0.6, top-p 0.95; vLLM seed 123.

**Scoring.** A try is correct when the text after a reasoning model's thinking (after its last `</think>`; a reasoning
model's try with no `</think>` never finished and counts as wrong) has a "CONCLUSION:" section that labels every person
exactly once, and every label matches the solution. Markdown emphasis and capitalization are ignored. Xie et al.'s
own scorer (`parse_cot_eval`, a copy in the script) is also recorded, to show agreement.

**Quantities, per model and N:** solve rate (the fraction of all tries that are correct); the fraction of puzzles
solved at least once in 4 tries; the fraction of tries that finished; the mean length of all tries and of correct
tries.

**What the outcomes mean.**
- **Go to step 2** if at least one N has a teacher solve rate of at least 70% and a solve rate of at most 30% for the
  best of B, C and D (R1-Distill-Qwen-7B). Step 2 (students trained on the teacher's solutions, on open models'
  solutions and on answers only, at the chosen N) gets its own rule.
- **Qwen3-8B** is reported separately. If it exceeds 30% at every N that passes, the result is recorded as "a gap
  exists only against the older open reasoning model", and the authors decide whether step 2 goes ahead.
- **No N passes:**
  - If the teacher stays below 70% wherever the open models are below 30%, the 8,192-token budget or the task is too
    hard for the teacher.
  - If the open models stay above 30% wherever the teacher reaches 70%, open models close the gap on Knights and
    Knaves too.

  Either way the result is recorded, and the authors decide between Zebra puzzles and stopping.
- **Length flag for step 2:** the mean length of the teacher's correct solutions at the chosen N is reported. If it
  exceeds 4,096 tokens, step 2 needs longer-context training than the setup's 2,048 tokens. That failure stopped the
  full-budget MATH attempt (§10.1).

**Puzzles made (2026-11-13 01:15).** `data/kk/pilot_test.jsonl`: 600 distinct puzzles, 100 per N, 55.3% knights.
Running `scripts/kk_make.py` imports the cloned repository's `lib_kk.py`; the user approved this after it was read
(standard library and numpy only; no file, network or process calls).

**Smoke test (2026-11-13 01:23 – 01:29; the first 12 puzzles, all N = 3; 2 tries; no threshold).** All five models
ran, the teacher on GPUs 0–3 with tensor parallel 4.

| model | solved (fixed checker) | Xie et al.'s criterion | finished | mean tokens |
| --- | --- | --- | --- | --- |
| teacher, R1-Distill-Qwen-32B | 24/24 | 24/24 | 100% | 1,439 |
| B, Llama-3.2-3B-Instruct | 6/24 | 6/24 | 100% | 445 |
| C, Qwen2.5-7B-Instruct | 8/24 | 9/24 | 100% | 636 |
| D, R1-Distill-Qwen-7B | 21/24 | 20/24 | 96% | 2,643 |
| Qwen3-8B | 24/24 | 21/24 | 100% | 1,982 |

**Checker fix (made after the smoke test, before any pilot result).** Qwen3-8B adds an explanation after its
numbered list that quotes the statements (e.g. "Jackson is a knave or Jackson is a knight"), and the first checker
read those quotes as a second label: it scored 19 of Qwen3-8B's 24 tries correct. The checker now reads only the
numbered lines (1) to N of the last CONCLUSION section (or, without numbers, the text up to the first blank line), and
it passes 11 hand-written cases. Rescored, Qwen3-8B has 24/24; the other models are unchanged.

The 5 remaining disagreements with Xie et al.'s criterion were read one by one. In each, the fixed checker is right:
- C wrote two contradictory full answers. Their criterion accepts this; ours rejects it.
- D's correct final answer followed an earlier CONCLUSION. Their criterion reads the first one.
- Three correct Qwen3-8B answers had explanations containing "if and only if". Their criterion rejects any answer that
  contains "if".

**Result (2026-11-13 01:29 – 03:17; recorded 2026-11-13).** 600 puzzles × 4 tries per model, from
`outputs/kk_pilot/<model>/tries.jsonl` and `summary.json`. Each cell: solve rate (fraction of tries correct) / fraction
of puzzles solved at least once / fraction of tries that finished / mean tokens of correct tries.

| model | N = 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- |
| teacher, R1-Distill-Qwen-32B | .96/1.00/.99/1,739 | .93/1.00/.98/2,105 | .90/.99/.97/2,343 | .83/.99/.96/2,539 | **.76**/1.00/.92/2,784 | **.71**/.95/.87/3,164 |
| B, Llama-3.2-3B-Instruct | .20/.55/1.00/373 | .10/.38/.99/467 | .04/.17/.98/554 | .02/.09/.96/563 | .01/.05/.99/480 | .00/.01/.96/586 |
| C, Qwen2.5-7B-Instruct | .41/.80/1.00/598 | .25/.61/1.00/707 | .15/.41/1.00/762 | .08/.31/1.00/945 | .05/.15/1.00/869 | .04/.16/.99/966 |
| D, R1-Distill-Qwen-7B | .80/.97/.95/1,900 | .63/.93/.89/2,528 | .52/.85/.86/2,489 | .39/.82/.80/2,733 | **.28**/.63/.74/3,048 | **.20**/.50/.61/3,241 |
| Qwen3-8B (reported separately) | 1.00/1.00/1.00/2,106 | .99/1.00/.99/2,664 | .99/1.00/.99/2,951 | .97/1.00/.98/3,196 | .95/1.00/.97/3,433 | .92/1.00/.95/3,735 |

Run times: teacher 90 minutes (4 GPUs); B 5, C 10, D 99, Qwen3-8B 106 minutes (1 GPU each).

- **Go condition: met at N = 7 and N = 8.** The teacher solves 76% and 71%; the best of B, C and D (always D) solves 28%
  and 20%. At N ≤ 6, D is above 30%.
- **Qwen3-8B exceeds 30% at both passing N (95% and 92%) and beats the teacher at every N.** By the rule: *a gap exists
  only against the older open reasoning model*, and the authors decide whether step 2 goes ahead.
- **Length:** the teacher's correct solutions average 2,784 (N = 7) and 3,164 (N = 8) tokens, below the 4,096 flag. But
  they exceed the setup's 2,048-token training length, so step 2 needs a longer training context in any case. At N = 8,
  13% of the teacher's tries hit the 8,192-token limit.
- **Agreement with Xie et al.'s criterion** is recorded per try in `tries.jsonl`; the smoke test showed where the two
  differ.
- **Authors' decision (2026-11-13):** keep the design. The teacher stays R1-Distill-Qwen-32B and level D stays
  R1-Distill-Qwen-7B; step 2 goes ahead at N = 7 and 8. Qwen3-8B beating the teacher is reported as a limitation of the
  testbed. The alternatives considered were Qwen3-8B as the teacher, and a stronger teacher with harder puzzles (the
  cached Qwen3.8-27B is not supported by either installed vLLM).

## 18. Puzzle testbed, step 2: is there something to protect at N = 7–8? (fixed 2026-11-13, before any result; not yet approved to run)

**Why.** §17 found the first setting where the teacher clearly beats the attacker's owned models of levels B–D on
solving. Solving is not training: what matters is whether a student trained on the teacher's solutions beats students
trained on what the attacker owns. This step applies the necessity check (§11) at levels B, C and D, and measures the
floor of §16's framing at level D: a student trained only with the help of the teacher's answers.

**Puzzles** (`scripts/kk_make.py`, Xie et al.'s generator, depth ≤ 2, width ≤ 2; disjoint from each other and from
the pilot by statement, which is checked):
- Training pool: 1,000 puzzles at N = 7 and 1,000 at N = 8 (seed 20261313 + N).
- Test set: 500 at N = 7 and 500 at N = 8 (seed 20261213 + N).

**Solution writers**, all with §17's prompt and sampling (temperature 0.6, top-p 0.95, 8,192 new tokens for reasoning
models, 2,048 for the others):
- Teacher: R1-Distill-Qwen-32B, 1 try per training puzzle.
- B, C: 1 try per training puzzle.
- D: R1-Distill-Qwen-7B, 4 tries per training puzzle.

**Students** (student: base Llama-3.2-3B), each trained on one solution per training puzzle:

| student | trained on | uses from the teacher |
| --- | --- | --- |
| U | the teacher's solution, correct or not (as U in setup 2) | solutions |
| Solo_B, Solo_C | B's or C's solution, correct or not | nothing |
| Solo_D | D's first try, correct or not | nothing |
| F_D (the floor at D) | D's first try that reaches the teacher's answer. For puzzles with none in 4 tries, D gets the teacher's answer as a hint (STaR's rationalization, §9.3) and 4 more tries; the hint is removed from the training text. Puzzles still unsolved are reported, not trained on. | answers only |
| untrained | — | — |

Level A (the base student alone) is deferred: it needs a few-shot prompt for these puzzles.

**Training** (a new script, `scripts/kk_sft.py`; the setup-2 code's prompts are math-specific): setup 2's recipe where
it applies, namely LoRA rank 128 (alpha 128), learning rate 5e-4 with cosine decay and 3% warm-up, 3 epochs, effective
batch 24, weight decay 0.1, gradient clipping 1.0. Loss on the solution only. Maximum length 8,704 tokens (setup 2:
2,048, too short for these solutions, §17). The student sees the puzzle as the user turn of the Llama-3.2 chat
template, with §17's instruction. Seeds 123, 124, 125.

**Scoring** (`scripts/kk_pilot.py`'s checker): each student gets 1 try per test puzzle (temperature 0.6, top-p 0.95, up
to 8,192 tokens, vLLM with the LoRA). The answer part is the text after the last `</think>` if there is one, otherwise
the whole text. Solve rate on the 1,000 test puzzles (primary), and per N.

**Rule.** Margin = a quarter of what training on the teacher's solutions adds: (U − untrained) / 4, mean of 3 seeds.
- **Necessity at level ℓ (B, C, D):** passes if the 3-seed mean of U − Solo_ℓ is above the margin and its 95% interval
  excludes 0.
- **The floor at D:** the protectable part at D is U − F_D. It *exists* if its 3-seed mean is above the margin and its
  95% interval excludes 0.

**What the outcomes mean.**
- **Necessity passes at D:** the first setting where the teacher's solutions are needed against an attacker who owns an
  open reasoning model. Step 3 (defenses, measured against the floor) gets its own rule.
  - If the protectable part also exists, a defense has something to protect, and step 3 measures how much of it each
    defense removes.
  - If it does not exist, the teacher's answers alone recover what its solutions give at D, so no answer-keeping
    defense can protect anything here. That is the central claim of the floor framing, in a setting where the teacher
    is needed.
- **Necessity fails at D:** even when the teacher is a far better solver, an attacker with R1-Distill-Qwen-7B does not
  need its solutions. This is recorded, and the authors decide between stopping and a teacher that is further ahead.
- **The limitation of §17 stands either way:** Qwen3-8B solves these puzzles better than the teacher.

**Smoke test first** (20 training puzzles, 10 test puzzles, 1 seed, a few steps; no threshold).

**Puzzles made (2026-11-13 14:10).** `data/kk/step2_train.jsonl` (2,000) and `data/kk/step2_test.jsonl` (1,000). No
statement is shared between them or with the pilot.

**Smoke test (2026-11-13 14:18 – 15:30; `outputs/kk_step2/smoke/`).** Every stage ran:
- **Writers on 20 training puzzles:** the teacher 13/20 correct, D 18–25% of tries, B 0/20, C 1/20.
- **Hint round and build:** 2 puzzles had no readable teacher answer, and 8 went to the hint round. F_D got 14
  examples: 10 from the first round and 4 from the hint round; 4 stayed unsolved.
- **Training texts** are formatted as intended: the R1 writers' text starts with `<think>`, and every text ends with the
  numbered CONCLUSION list.
- **Students:** all six were trained for 3 steps and scored on 10 test puzzles. All solve 0, as expected after 3 steps.

Two mechanical changes, both before any step-2 result:
- **GPU layout.** Labmates held GPUs 3 and 5, so neither group had 4 free GPUs. The teacher runs as tensor-parallel-2
  copies instead of one tensor-parallel-4 copy: two copies, one per N, in the smoke test. The full data phase runs one
  copy on GPUs 0 and 3, because GPUs 1, 6 and 7 now carry the user's own jobs. This changes speed, not the sampled
  solutions' distribution.
- **Free-memory wait.** The launcher waited for 42 GB free before every job. On GPU 1, a 3.5 GB job kept the F_D
  training job waiting for 34 minutes, so it was run by hand on GPU 2. Training now waits for 30 GB and scoring for
  40 GB.

**Data phase (2026-11-13 15:35 – 2026-11-14 00:46).** GPUs 0+3 (teacher, tensor parallel 2), 2 and 4.
- **Writers:** correct at N = 7 / 8 were the teacher 73.9% / 69.6% (424 minutes), D 27.5% / 17.4%, C 5.8% / 2.1%,
  B 1.5% / 0.2%.
- **No readable teacher answer:** 288 of 2,000 puzzles, mostly where the teacher ran out of its 8,192 tokens. U trains
  on these solutions; F_D cannot use them.
- **Hint round:** 814 puzzles.
- **Training sets** (`build_report.json`):
  - U: 2,000 examples, 1,435 correct.
  - Solo_B: 2,000 (17 correct). Solo_C: 2,000 (79). Solo_D: 2,000 (430).
  - F_D: 1,304 examples (1,189 correct): 898 from the first round, 406 from the hint round. 408 stayed unsolved and 288
    had no teacher answer; both groups are reported and not trained on.

**Training phase (2026-11-14 00:46 – 10:18; GPUs 0–4, 6; 18 runs).**

**Scoring bug, found 2026-11-15, before this result was recorded.** The students are trained in the Llama-3.2 chat
template, whose turns end with `<|eot_id|>`. `kk_eval.py` ran the base model's stop setting, which stops only at
`<|end_of_text|>`. So generation ran past the answer into an invented next turn (`…Isabella is a knave.user <think>
Alright, so…`), up to the 8,192-token cap. This is why only 3–10% of the reasoning students' tries "finished". The
checker reads the last CONCLUSION, so some verdicts come from that extra text instead of the student's answer.
- **Fix:** `kk_eval.py` now stops at `<|eot_id|>`. The adapters are kept, so a clean rescore needs no retraining. It is
  not yet run: it waits for the authors' go-ahead.
- **Not affected:** the pilot (§17) and the solution writers. They used instruct and reasoning models whose stop
  settings include their end-of-turn tokens.

**Provisional result, two scorings of the stored outputs** (1,000 test puzzles; seeds 123 / 124 / 125). "As scored"
reads the last answer in the text; "first answer" reads the first CONCLUSION section, as a diagnostic.

| student | as scored | first answer |
| --- | --- | --- |
| U (teacher's solutions) | 39.8 / 42.0 / 41.0 → **40.9%** | 49.8 / 49.1 / 47.4 → **48.8%** |
| F_D (answers only, floor at D) | 16.7 / 21.1 / 18.9 → **18.9%** | 20.5 / 23.7 / 20.5 → **21.6%** |
| Solo_D (R1-Distill-Qwen-7B's solutions) | 8.6 / 14.0 / 11.5 → 11.4% | 8.8 / 15.1 / 13.0 → 12.3% |
| Solo_C (Qwen2.5-7B-Instruct's) | 4.6 / 4.9 / 4.9 → 4.8% | 4.7% |
| Solo_B (Llama-3.2-3B-Instruct's) | 0.5 / 0.6 / 0.2 → 0.4% | 0.4% |
| untrained | 0.1 / 0.0 / 0.7 → 0.3% | 0.3% |

Rule check, the same under both scorings (margin 10.2 as scored, 12.1 on first answers):
- **Necessity at D:** U − Solo_D = +29.6 (25.6 to 33.5) as scored, +36.5 (26.7 to 46.2) on first answers. Passes.
- **Necessity at C:** +36.1 and +44.0. Passes.
- **Necessity at B:** +40.5 and +48.3. Passes.
- **Protectable part at D:** U − F_D = +22.0 (19.3 to 24.8) as scored, +27.2 (22.3 to 32.1) on first answers. It
  exists.
- **The answers alone add** F_D − Solo_D = +7.5 points as scored, +9.3 on first answers.

**Provisional verdict:** for the first time in this project, the teacher's solutions are needed against an attacker
who owns an open reasoning model, and something is left to protect. The final numbers come from the rescore.

**Rescore (2026-11-15 20:07 – 2026-11-16 01:01; `test_eot/`; 18 runs; GPUs 0–4, 6, 7).** Stopping at `<|eot_id|>`
did not change the outputs. The reasoning students (U, F_D, Solo_D) still finish only 3–7% of their tries, averaging
about 8,000 tokens, and half of U's tries still show an invented next turn right after an answer line.

**Root cause, found 2026-11-16.** The students are LoRA adapters on the attention and MLP projections of *base*
Llama-3.2-3B. The base model's chat special tokens were never trained, and the LoRA does not touch the embedding or
output rows. So a student cannot produce `<|eot_id|>`: after its answer it emits a garbage token
(`…is a knight.átkuüyorduuser`) and runs on to the cap. Setup 2's recipe for this base model has the same property.
This is a property of how the students were trained, not of the scoring.

| student (3 seeds) | rescore, last answer (rule as written) | first answer (first CONCLUSION after the first `</think>`) |
| --- | --- | --- |
| U | 38.8 / 45.6 / 40.9 → **41.8%** | 48.2 / 50.6 / 47.7 → **48.8%** |
| F_D | 20.0 / 20.3 / 20.4 → **20.2%** | 24.7 / 21.8 / 20.7 → **22.4%** |
| Solo_D | 8.3 / 14.7 / 11.7 → 11.6% | 8.6 / 15.7 / 14.3 → 12.9% |
| Solo_C | 4.5 / 4.5 / 5.0 → 4.7% | 4.5% |
| Solo_B | 0.7 / 0.4 / 0.3 → 0.5% | 0.5% |
| untrained | 0.1 / 0.0 / 0.7 → 0.3% | 0.3% |

Rule check (rescore, last answer; margin 10.4):

| test | difference | 95% CI | verdict |
| --- | --- | --- | --- |
| necessity at B: U − Solo_B | +41.3 | 32.4 to 50.2 | passes |
| necessity at C: U − Solo_C | +37.1 | 28.3 to 45.9 | passes |
| necessity at D: U − Solo_D | +30.2 | 28.0 to 32.4 | passes |
| protectable part at D: U − F_D | +21.5 | 13.2 to 29.9 | exists |

On first answers (margin 12.1): +48.4, +44.4, +36.0 and +26.4 (19.7 to 33.1). The same verdicts.

**Verdict, the same under all three scorings** (first scoring, rescore, first answer): the teacher is needed at B, C
and D, and a protectable part exists at D. The answers add F_D − Solo_D = +8.7 points (1.1 to 16.2) on the rescore and
+9.5 (−4.6 to 23.7) on first answers; this is secondary and not part of the rule.

**Open, for the authors:** which numbers the paper reports. There are two options:
- **(a)** score the student's first complete answer;
- **(b)** retrain the students with the special-token rows trainable (PEFT `modules_to_save` or `trainable_token_indices`
  for `<|eot_id|>`) so they stop cleanly. That is 15 training runs plus scoring, about a day of GPU time.

Until decided, the paper uses the first-answer numbers, marked provisional.

**Decision (2026-11-16).** The draft reports (a): each student's **first complete answer**, the first CONCLUSION
section after its first `</think>`, or in the whole text if there is none. This is the natural reading of one try,
because the student cannot stop after it. (b), retraining with a trainable end-of-turn token, stays open for before
submission. The first-answer numbers are final under (a): U 48.8%, F_D 22.4%, Solo_D 12.9%, Solo_C 4.5%, Solo_B 0.5%,
untrained 0.3%. Margin 12.1. Necessity passes at B, C and D (U − Solo_D = +36.0, 27.9 to 44.0). The protectable part
at D exists (U − F_D = +26.4, 19.7 to 33.1). All later puzzle students (§20 and after) are scored the same way.


## 19. Level A on MATH: self-training on the teacher's answers (fixed 2026-11-16, before any result; go-ahead from the user)

**Why.** The MATH level-A cell is empty (Table 2 and Figure 1b of the draft). It is the floor F_A of an attacker who
owns only the student.

**What is run.** `scripts/52_levelA_fill.sh`, job `math <seed>`, calling `scripts/dg_star.py --fewshot` on the MATH
configs.
- **Seeds and data:** 123–125. Each seed's own pool of 5,000 MATH training problems, with that seed's teacher answers
  from the regenerated U run (§12).
- **Prompt:** the same procedure as GSM8K STaR (§9.3, §13), except that the prompt is the 4 MATH examples of
  Lewkowycz et al. (`minerva_prompt` in `dg_answer_dist.py`, as used for §16).
- **Generation:** 8 tries per problem at temperature 0.8, up to 1,024 tokens.
- **Keeping a try:** the first try whose answer matches the teacher's answer, using the teacher's own boxed text as the
  reference (§16 deviation).
- **Hints:** unsolved problems get the teacher's answer as a hint, on every example, for 8 and then 32 more tries. The
  kept text has no hint. Problems still unsolved are reported and not trained on.
- **Training and scoring:** U's recipe (LoRA r128, 3 epochs), scored on the 5,000 MATH test problems with the setup's
  evaluation, exactly as U, Solo and the untrained student were.
- **Expectation, from §16:** the base student matches the teacher on only 8% of tries, and 3,001 of 5,000 problems
  never matched in 16 tries at temperature 0.6. So most kept solutions will come from the hint rounds.

**Rule.** F_A = the STaR student's test accuracy per seed. Margin = (U − untrained) / 4 = (15.47 − 0.85) / 4 = 3.7.
- **The protectable part at A exists** if the 3-seed mean of U − F_A is above 3.7 and its 95% interval excludes 0.
- **If it does not exist:** on MATH too, the answers alone carry what the teacher's solutions give the weakest
  attacker.
- R_A = the untrained student (0.85%), so necessity at A passes (U − R_A = +14.6).

**Result (2026-11-16 16:33 – 20:53; GPUs 2–4).** From `outputs/setup2/star_fewshot_math_s<seed>/test_star.json`, 5,000
MATH test problems:

| seed | kept: own / hint / salvage | unsolved (reported) | F_A | U (§12) | U − F_A |
| --- | --- | --- | --- | --- | --- |
| 123 | 1,444 / 3,516 / 8 | 32 | 10.00% | 15.30% | +5.30 |
| 124 | 1,456 / 3,501 / 11 | 32 | 9.56% | 15.42% | +5.86 |
| 125 | 1,442 / 3,518 / 7 | 33 | 9.14% | 15.70% | +6.56 |
| mean | | | **9.57%** | 15.47% | **+5.91** (4.34 to 7.48) |

- **The protectable part at A exists on MATH:** +5.9 points against a margin of 3.7.
- **The answers carry most of what the teacher adds:** 8.7 of 14.6 points (60%; F_A − untrained = 9.57 − 0.85). The
  same share as on GSM8K (58%).
- About 29% of the kept solutions are the student's own; the rest come from the hint rounds.

## 20. Level A on the puzzles: self-training on the teacher's answers (fixed 2026-11-16, before any result; go-ahead from the user)

**Why.** The puzzle level-A cell is empty (Figure 1c).

**What is run.** `scripts/52_levelA_fill.sh`, job `kk <seed>`.
- **Generation** (`scripts/kk_star.py`):
  - The pool is §18's 2,000 training puzzles. The teacher's answer is its first try's CONCLUSION (§18 writers); the 288
    puzzles without a readable answer are reported and not trained on.
  - The untrained base student is prompted with Xie et al.'s instruction and their two-person worked demonstration
    (`demonstration_2char`, verbatim; it comes from their repository, not from a model we own), then the puzzle.
  - 8 tries at temperature 0.8, up to 2,048 tokens. The first try whose CONCLUSION equals the teacher's answer is
    kept.
  - Unsolved puzzles get the teacher's answer as a hint, on the demonstration too, for 8 and then 32 more tries. The
    kept text has no hint. Puzzles still unsolved are reported and not trained on.
- **Training:** `scripts/kk_sft.py` with §18's recipe.
- **Scoring:** `scripts/kk_eval.py` on the 1,000 test puzzles, scored on first answers (§18 decision).
- **Seeds:** 123–125. The seed sets both generation and training; the pool is the same for every seed.

**Rule.** F_A = the STaR student's first-answer accuracy per seed. U, untrained and the margin (12.1) are §18's,
scored on first answers.
- **The protectable part at A exists** if the 3-seed mean of U − F_A is above 12.1 and its 95% interval excludes 0.
- **F_A against F_D** (22.4%) is reported: how much the open reasoner adds to an answers-only attacker.
- R_A = the untrained student (0.3%), so necessity at A passes.

**Smoke test first,** for both §19 and §20: 16 problems, 2 tries per round, 2 training steps, 10 test puzzles; no
threshold.

**Smoke test (2026-11-16 16:12 – 16:20).** Both pipelines ran end to end.
- **MATH (16 problems):** 1 solved alone, 9 with the hint, 5 more in the salvage round, 1 unsolved.
- **Puzzles (16):** 2 had no teacher answer, and only 1 of 14 was kept.

A probe of the hint prompt found no bug. The base student copies the hinted answer on 19 of 64 tries (8 tries on 8
puzzles); the other tries mostly fall into a repetition loop ("Knight is a knight and Knight is a knight …") and hit
the 2,048-token cap. With the full 8 + 8 + 32 tries per puzzle, most puzzles should yield a kept solution, almost all
of them from the hint rounds. No setting was changed.

**Result (2026-11-17; `outputs/kk_levelA/s<seed>/`, test files `test_eot/results.jsonl`, scored on first answers).**

| Seed | Solved alone | With hint | Salvaged | Kept | F_A (%) | U (%) | U − F_A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 123 | 9 | 1,655 | 48 | 1,712 | 2.1 | 48.2 | 46.1 |
| 124 | 11 | 1,656 | 45 | 1,712 | 0.8 | 50.6 | 49.8 |
| 125 | 10 | 1,652 | 50 | 1,712 | 2.3 | 47.7 | 45.4 |
| Mean | | | | | **1.7** | 48.8 | **+47.1** |

- Every puzzle with a teacher answer yielded a kept solution (none unsolved); 288 had no teacher answer, as fixed.
  1,435 of the 1,712 kept answers are the true solution (the teacher's answer is wrong on the other 277), the same
  for every seed.
- Almost every kept solution comes from the hint rounds (about 99%): the base student copies the given answer after a
  text it cannot use.
- **Verdict: the protectable part at A exists.** U − F_A = +47.1 points, 95% interval 41.2 to 53.0, above the 12.1
  margin.
- **F_A against F_D:** F_D − F_A = +20.7 points (15.4 to 25.9). The open 7B reasoner is what makes the teacher's
  answers useful here.
- **What the answers carry at A:** (F_A − untrained) / (U − untrained) = (1.7 − 0.3) / (48.8 − 0.3) = 3%, against
  58% on GSM8K (§13) and 60% on MATH (§19).
- Generation took 147–155 minutes per seed on one GPU; training and scoring about 40 minutes more.

## 21. Phase 1: defenses on the puzzle testbed, first defense: trace rewriting (fixed 2026-11-16, before any result; rule and rewriter approved by the user)

**Why.** §18 found a protectable part at level D of the puzzle testbed: U − F_D = +26.4 points (first answers). Phase 1
measures how much of it published defenses remove. Rewriting comes first because it is cheap and has no strength
setting. ADS and PoE follow, each with its own rule; their strength will be the strongest that lowers the teacher's own
accuracy by at most 5 points.

**Defense.** Ma et al. 2026's optimized instruction-based rewriting (OPT, their best), applied to the teacher's 2,000
step-2 training solutions, the ones U was trained on.
- **Rewriter:** the teacher itself, R1-Distill-Qwen-32B; the user chose this over a separate model. The provider
  rewrites its own outputs, so no other model's knowledge enters the defended text. (Ma et al. used gpt-oss-120b to
  rewrite a 7B teacher.)
- **Instruction**, verbatim from their Appendix A.2: "Your task is to process a provided chain of logical deductions and
  reformulate it utilizing an esoteric, formal, and densely technical lexicon, thereby obfuscating its clarity for a
  non-specialist audience. Refrain from any extraneous discourse."
- **Input:** as in their method, the instruction followed by the teacher's solution (its reasoning, then its answer
  part), with no separate question.
- **Generation:** an empty thinking block is prefilled, so the rewriter writes the rewrite directly. Temperature 0.6,
  top-p 0.95, up to 8,192 new tokens, vLLM seed 123, tensor parallel 2 (two copies, half the puzzles each).
- **Answers preserved by construction:**
  - Any CONCLUSION section the rewriter writes is cut off.
  - The teacher's own CONCLUSION section is appended verbatim, so every defended solution ends with exactly the
    teacher's answer.
  - For 15 solutions the answer is readable but not laid out as a numbered CONCLUSION list. There the same answer is
    appended in the standard layout. (Checked offline: on the other 1,697, the extracted section parses to exactly the
    teacher's answer.)
  - Where the teacher's solution has no readable answer (288 puzzles), the rewrite is used as it is, as U uses the
    teacher's text as it is.
- **Format:** the defended training text is `<think>` + rewrite + `</think>` + the teacher's CONCLUSION, the same
  shape as U's targets. Rewrites that hit the token cap are kept and counted.

**Student.** D_rewrite, trained on the defended solutions with §18's recipe, seeds 123–125, scored on the 1,000 test
puzzles on first answers (§18 decision). U = 48.8%, F_D = 22.4%, untrained 0.3% (§18).

**Quantities.**
- **Protection:** the attacker's best student under the defense is max(D_rewrite, F_D), because it can ignore the
  defended solutions and use the answers. Protection = U − max(D_rewrite, F_D).
- **Share removed** = protection / (U − F_D), reported with the 95% interval of U − D_rewrite over the 3 seeds.
- **Cost to users:** the final answers are unchanged by construction. Reported: the length of the rewrites against the
  originals, and the share that hit the cap.

**What the outcomes mean.**
- **D_rewrite not clearly below U** (U − D_rewrite under the margin of 12.1, or its interval includes 0): rewriting
  protects nothing, even where protection is possible.
- **D_rewrite between F_D and U:** the share removed is the first measured protection share of a published defense.
- **D_rewrite at or below F_D:** rewriting removes all that any answer-preserving defense can remove. Below F_D, the
  defended solutions are worse than none, but the attacker still keeps F_D.

**Smoke test first:** 16 rewrites. Check the format, that the appended answer is the teacher's, and the length. No
threshold.

**Smoke test (2026-11-16 21:51 – 21:55, GPUs 2–3).** Deviation: 8 rewrites were run, not 16 (`--limit 8` in
`scripts/53_kk_rewrite.sh`). Output `outputs/kk_step2/rewrite_smoke/part1.jsonl`.
- **Format and answers:** all 8 have the `<think>` rewrite `</think>` CONCLUSION shape, and every appended answer
  parses to exactly the teacher's labels. None hit the cap.
- **Length:** the rewrites are 61–1,166 tokens against 1,722–8,011 for the originals; mean ratio 0.12.
- **Content:** none is written in the esoteric, technical style the instruction asks for.
  - 5 of 8 are plain, readable step-by-step summaries of the teacher's deduction, with the answer list first.
  - 1 is the answer list with a one-sentence justification.
  - 2 are only the answer list, one of them in French.

So with the empty thinking block, the rewriter summarizes its solution instead of obfuscating it. This is a property
of the rewriter as configured, not a bug in the pipeline. The full run is on hold until the authors decide whether to
keep the rule as it is or change the rewriter setting.

**Amendment (2026-11-17, before any result of the new setting; the user chose option 1: "the teacher thinks first").**
This replaces the "Generation" item above; everything else in §21 is unchanged.
- **Generation:** no prefilled empty thinking block. The rewriter (still the teacher, R1-Distill-Qwen-32B) thinks
  first, as it was trained to, then writes the rewrite. The kept rewrite is the text after its last `</think>`; its
  thinking is discarded. Temperature 0.6, top-p 0.95, vLLM seed 123, tensor parallel 2. The budget rises to 16,384
  new tokens (thinking and rewrite together).
- **Thinking that never closes:** if a try has no `</think>` within the budget, that solution is tried again with seed
  124, then 125. If all three tries fail, the last try's text is kept as it is and counted, as capped rewrites are.
  If that happens to more than 2% of the 2,000 solutions, the authors decide before any training.
- **Reported in addition:** the rewriter's thinking length, the rewrite's length without the thinking (against the
  original solution), retries, and how many never closed.
- **Smoke test: 16 rewrites** (the first 16 puzzles of the pool). Check format, appended answer and length, as before.
  **Go or no-go for the full run:** go if at least 12 of the 16 rewrites present the deduction in a formal, technical
  register, as the instruction asks, rather than a plain summary or a bare answer list. This is judged by reading;
  examples are quoted in the result. Otherwise the authors decide.

**Smoke test with thinking (2026-11-17 00:16 – 00:27, GPUs 2–3; `outputs/kk_step2/rewrite_smoke_think/part1.jsonl`).**
**Verdict: no-go (0 of 16 in a formal, technical register; 12 needed).** Back to the authors.
- **Mechanics:** 16 of 16 rewritten, none capped, no thinking left unclosed, no retries. Every appended answer parses
  to the teacher's labels; 2 of the 16 had no readable teacher answer and keep the rewrite as it is.
- **Length:** thinking averages 1,954 tokens. The kept rewrites average 294 tokens (0 to 1,004), 0.08 of the
  originals.
- **Content:**
  - 0 of 16 thinking traces mention the rewriting task (no "rewrite", "formal", "technical", "esoteric",
    "obfuscate"). Each starts "Alright, so I'm trying to solve this knights and knaves puzzle" and solves the puzzle
    again from the statements quoted in the teacher's solution.
  - The text after the thinking is a plain summary of that new solution: an answer list with one sentence
    (#0, #1, #4, #8, #10), or a short plain step list (the rest).
  - In #7 the rewriter wrote only a CONCLUSION, which is cut, so the defended text is the teacher's answer alone.
- **Reading:** the teacher, with or without thinking, does not carry out Ma et al.'s instruction; it re-solves the
  puzzle. What it produces would measure "a short summary of the teacher's own solution", not the published defense.

**Second amendment (2026-11-17, before any result of the new rewriter; the user approved gpt-oss-120b).** Replaces
the rewriter and generation items above; the instruction, input, answer handling, format, student and quantities of
§21 are unchanged.
- **Rewriter:** gpt-oss-120b (`openai/gpt-oss-120b`, lab cache), the rewriter Ma et al. used and the one their
  instruction was optimized for. It runs in `.venv-infer` (vLLM 0.11), tensor parallel 2.
- **Generation:** the model's own chat template with its default system message and default reasoning effort (Ma et
  al. do not state theirs). The instruction and the teacher's solution go in one user message, as before. Temperature
  0.6 (Ma et al., Appendix D.2), top-p 1.0 (vLLM default), up to 16,384 new tokens, vLLM seed 123. The kept rewrite is
  the text of the model's final channel; its analysis channel is discarded. A try without a final channel is retried
  with seeds 124 and 125; after three failures the last try's text is kept and counted; above 2% of the 2,000, the
  authors decide.
- **Lengths** are counted with the teacher's tokenizer, so the ratio to the original is in the same units.
- **Departure from the teacher, reported:** where the rewriter writes its own CONCLUSION (cut, as before), whether it
  matches the teacher's answer; separately on puzzles where the teacher's answer is wrong. A stronger rewriter could
  correct the teacher in the defended text; this measures how often its own conclusion says so.
- **Smoke test:** the first 16 puzzles, with the same checks and the same go or no-go (at least 12 of 16 in a formal,
  technical register).
- **Cost** (reported): a second, 120B model's inference for every released solution. The final answers users see
  are unchanged by construction.

**Smoke test with gpt-oss-120b (2026-11-17).**
- **First try: an inference bug.** 14 of 16 outputs were "!!!…" to the 16,384-token cap; 1 good, 1 good but capped.
  Output kept in `outputs/kk_step2/rewrite_smoke_gptoss_nan/`.
  - A probe (first 16 prompts, 48 tokens) found the cause: vLLM's prefix caching.
    - Default settings: 15 of 16 broken; every prompt after the first reuses the cached shared prefix (system
      message and instruction) and decodes NaN logits.
    - Prefix caching off: 0 of 16 broken. Eager mode: 0 of 16. Chunked prefill off: 15 of 16.
  - **Deviation:** prefix caching is off for gpt-oss (`scripts/kk_rewrite.py`). It changes no sampling setting.
  - Also: the launcher puts the venv's `bin` on PATH; vLLM 0.11 builds a kernel with `ninja` at start-up.
- **Second try (11:52 – 11:54, GPUs 2–3; `outputs/kk_step2/rewrite_smoke_gptoss/part1.jsonl`).**
  **Verdict: go (16 of 16 in a formal, technical register; 12 needed).**
  - **Mechanics:** 16 of 16 rewritten; none capped, none without a final channel, no retries, no "!!!" output. Every
    appended answer parses to the teacher's labels (2 of 16 have no teacher answer and keep the rewrite as it is).
  - **Length:** reasoning 436 tokens on average. Rewrites average 1,294 teacher tokens, 0.39 of the originals
    (620 to 1,913 against 1,722 to 8,192).
  - **Content:** every rewrite recasts the deduction as propositional logic in LaTeX ("Let 𝔻 = {…} denote the seven
    denizens… unary predicate K(x)…"), with terms such as "denotators", "ontic statuses", "Kripke-style model". Each
    ends with its own statement of the assignment.
  - **Departure from the teacher.** The pre-written measure found nothing to measure: gpt-oss never writes a
    CONCLUSION list, so no rewrite was parsed. By reading the 6 puzzles where it matters:
    - Teacher wrong (4): 3 rewrites restate the teacher's wrong answer. In 1 (#2) the rewrite states the true answer,
      so the defended text contradicts the teacher's CONCLUSION appended after it.
    - No teacher answer (2): the rewriter supplies its own answer, once right (#15) and once wrong (#13).
    - For the full run this is reported by reading a random sample of 50 rewrites (seed 123) with a wrong or missing
      teacher answer. It is descriptive and changes no verdict.

**Result (2026-11-17).** Rewriting 11:55 – 12:47 (two copies, GPUs 2–3 and 6–7); training and test 12:47 – 15:48.
Files: `outputs/kk_step2/rewrite/part{1,2}.jsonl` (+ `.summary.json`), `outputs/kk_step2/train_sets/D_rewrite.jsonl`,
`outputs/kk_step2/students/D_rewrite_s<seed>/test_eot/results.jsonl`, scored on first answers.
- **The rewrites:**
  - All 2,000 rewritten. The model's reasoning ended in every case (no retries). 55 rewrites (2.8%) hit the
    16,384-token cap and are kept as they are.
  - Mean rewrite 1,390–1,466 teacher tokens, 0.46 of the original. Reasoning 586–676 tokens.
  - 270 have no teacher CONCLUSION to append: the 288 without a readable answer, minus 18 with a numbered
    CONCLUSION that does not parse, which is appended verbatim.

| Seed | D_rewrite (%) | U (%) | U − D_rewrite | F_D (%) |
| --- | --- | --- | --- | --- |
| 123 | 4.0 | 48.2 | 44.2 | 24.7 |
| 124 | 10.2 | 50.6 | 40.4 | 21.8 |
| 125 | 4.9 | 47.7 | 42.8 | 20.7 |
| Mean | **6.4** | 48.8 | **+42.5** (37.7 to 47.2) | 22.4 |

- **Verdict: D_rewrite is below F_D,** so rewriting removes all that any answer-preserving defense can remove here.
  - Protection = U − max(D_rewrite, F_D) = 48.8 − 22.4 = 26.4 points. **Share removed: 100%.**
  - F_D − D_rewrite = +16.0 (4.7 to 27.3): the defended solutions are worse than none. The attacker's best response
    is to ignore them and use the answers with its own reasoner.
  - Reading of the rule: rewriting reaches the floor. Nothing more can be protected against this attacker.
- **Robustness of the verdict to unfinished answers:**
  - The defended students often write no CONCLUSION in the 8,192-token budget: 484, 994 and 739 of 1,000.
  - Among outputs that do conclude, they are right 8.3, 10.3 and 6.6% of the time, against 30–36% for F_D and 63–69%
    for U. So D_rewrite would stay below F_D even if every output concluded.
- **Departure from the teacher** (50 rewrites, seed 123, from the 565 puzzles with a wrong or missing teacher answer;
  read by hand):
  - **Teacher wrong (25 of the sample):** 23 restate the teacher's wrong answer; 1 states the true one; 1 is capped
    before any answer.
  - **No teacher answer (25):** 16 state the true answer; 4 a wrong one; 2 give two models (one of them right in 1
    case); 3 none (2 capped, 1 "unsatisfiable").
  - Extrapolated: gpt-oss adds a correct solution to roughly 180 puzzles where the teacher's text has none. The
    defended set therefore carries some knowledge the teacher's does not, and still trains a weaker student. This
    makes the verdict conservative.
- **Cost:** a second, 120B model's inference per released solution: 2,000 rewrites took 50 minutes on 2 × 2 L40S.
  Final answers are unchanged by construction.
- **Scope:** this attacker trains on the defended solutions as released. An adaptive attacker that first maps the
  formal rewrites back to plain reasoning with its own models is not measured; if that exceeded F_D, the protection
  would shrink. The authors decide whether to run it.

## 22. Phase 2: the floor under corrupted answers (fixed 2026-11-17, before any result; go-ahead from the user)

**Why.** A provider can lower the answer floor only by making some released answers wrong, which users pay for. This
measures how far the floor falls per point of that cost. The attacker trusts the answers it is given: the testbed
stands in for domains where no program can check an answer (§17, known limitation).

**Corruption.**
- The pool is the 1,712 step-2 training puzzles with a readable teacher answer. They are put in a random order
  (seed 20261117).
- Level c ∈ {12, 30, 50}% takes the first round(c × 1,712) puzzles of that order (205, 514, 856), so the levels are
  nested.
- Each chosen puzzle's released answer becomes an assignment drawn uniformly from those different from the teacher's
  answer (same seed; the same wrong answer at every level). The teacher's own wrong answers (277) stay as they are
  unless chosen.
- Level 0 is F_D itself (§18, 22.4%).

**The floor student F_D^c**, built exactly as F_D with the released answers:
- D's first try (of its 4 step-2 tries) whose answer equals the released answer.
- Otherwise a hint round of 4 D tries with the released answer as the hint, the hint removed from the training text.
  - Puzzles whose answer is unchanged reuse §18's hint round.
  - Corrupted puzzles get a new hint round, same settings (`gen/D_chint*`).
- Puzzles still unmatched, and the 288 without a teacher answer, are reported and not trained on.
- Training and scoring: §18's recipe, seeds 123–125, 1,000 test puzzles, first answers.

**Reported.**
- F_D^c per seed and its mean.
- The drop F_D − F_D^c with its 95% interval.
- The cost to users: the share of the 2,000 released answers that are wrong or missing at each level.
- How many corrupted answers D reaches in its own tries and in the hint round. A corrupted answer D never reaches
  drops out of training, so the attacker's own model filters part of the corruption.

**What the outcomes mean.**
- **A drop exists at level c** if its mean is above the margin (12.1) and its interval excludes 0.
- The curve (cost to users, F_D^c) shows the price of lowering the floor. Lowering it also lowers the most any
  answer-preserving defense could protect against an attacker that does not check answers.

**Check before any GPU run:** the corruption and the build are run offline on the stored tries (counts only).

**Offline check and build (2026-11-17 21:54 to 2026-11-18 00:20).** `scripts/kk_corrupt.py`, `scripts/54_kk_corrupt.sh`.
- Levels: 205, 514 and 856 of the 1,712 readable answers.
- D reaches only 1 of the 856 corrupted answers (the top level) in its 4 step-2 tries, so 855 puzzles went to the new
  hint round (D, 4 tries, two single-GPU copies on GPUs 6 and 7, 2 h 26 min).
- Build (`train_sets/build_report_corrupt.json`):

| Level | Corrupted | Corrupted answers kept (first round + hint round) | Training examples | Released answers wrong or missing (of 2,000) |
| --- | --- | --- | --- | --- |
| 0 (F_D) | 0 | – | – | 28.3% (277 wrong + 288 missing) |
| 12% | 205 | 16 (0 + 16) | 1,162 | 36.6% |
| 30% | 514 | 39 (1 + 38) | 942 | 50.0% |
| 50% | 856 | 64 (1 + 63) | 706 | 64.5% |

- D reaches 7.5–7.8% of the corrupted answers at every level. The other 92% leave the training set rather than enter it
  with a wrong answer: the attacker's own model filters most of the corruption, so F_D^c mostly measures a smaller
  training set with a few wrong answers.
- Students: 9 jobs on GPUs 4, 6 and 7 from 00:20; F_D_c50_s123 trained in 89 min.

**Result (2026-11-18 11:44; first answers; `outputs/kk_step2/corrupt/result.json`).** The scorer reproduces §18's
first-answer numbers for U and F_D exactly. Drops are seed-paired, 95% interval with t(2 df) = 4.303, as in §18.

| Level | Released answers wrong or missing | Training examples (wrong among them) | F_D^c, seeds 123 / 124 / 125 | Mean | Drop F_D − F_D^c | 95% CI | Drop exists? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 (F_D) | 28.3% | 1,304 (0 corrupted) | 24.7 / 21.8 / 20.7 | 22.4 | – | – | – |
| 12% | 36.6% | 1,162 (16) | 18.2 / 21.1 / 23.0 | 20.8 | +1.6 | −9.5 to 12.7 | no |
| 30% | 50.0% | 942 (39) | 17.0 / 17.6 / 21.4 | 18.7 | +3.7 | −6.7 to 14.2 | no |
| 50% | 64.5% | 706 (64) | 15.4 / 14.3 / 12.3 | 14.0 | +8.4 | 6.2 to 10.6 | no (below the 12.1 margin) |

- **Verdict: no drop at any level.**
  - At 12% and 30% the interval includes 0.
  - At 50% the interval excludes 0, but the mean, 8.4, is below the margin.
- **The price of lowering the floor is high.** Making half of the readable answers wrong raises the wrong-or-missing
  share of released answers from 28.3% to 64.5%, and lowers the floor by 8.4 points (22.4 to 14.0). The most an
  answer-preserving defense could then protect against this attacker grows from 26.4 to 34.8 points (U = 48.8).
- **Why so little.** D reaches only 64 of the 856 corrupted answers. The rest drop out of training, so F_D^c50 trains
  on 706 examples instead of 1,304, with 64 wrong. The fall measures mostly a smaller training set. An attacker that
  checks answers against its own tries throws most corruption away for free.

## 23. Phase 1: ADS and PoE on the puzzle testbed (fixed 2026-11-17, before any result; go-ahead from the user)

**Defenses**, as implemented in the Distillation Game code (`clean_sweep/generation/methods.py`, `core.py`).
- **ADS:** `AntidistillationLogitsProcessor`, identity penalty, eps 0.01. The proxy student is Qwen2.5-3B (base), as
  in their configs.
  - The proxy gradient is their `compute_student_holdout_grad` on teacher solutions of held-out puzzles, in the
    proxy's chat format, completion only.
  - The held-out puzzles are the teacher's first tries on the pilot's N = 7 and 8 puzzles (64 of each; disjoint
    from step-2 training and test).
  - Maximum length 8,704, as the students.
- **PoE:** `ProductOfExpertsLogitsProcessor`, the same proxy: (1 − γ) × teacher logits + γ × proxy logits.
- **Teacher generation:** Hugging Face `generate` with their logits processors (their code path; vLLM cannot run
  them). §18's prompt, temperature 0.6, top-p 0.95, up to 8,192 new tokens. The teacher is split across the GPUs
  of one group.

**Timing pilot first** (16 calibration puzzles; strength 0, ADS λ = 0.1, PoE γ = 0.4). Reported: tokens per second,
minutes per 100 puzzles, memory. No accuracy is read. If the full plan below would take more than about 3 days of
one GPU group, the authors decide on a smaller one.

**Strength (the rule fixed in §21):** the strongest setting that lowers the teacher's accuracy by at most 5 points.
- **Calibration set:** 100 fresh puzzles (50 at N = 7, 50 at N = 8; `scripts/kk_make.py`, seed 20261413+N), one try
  each, strict checker.
- **Baseline:** strength 0 through the same Hugging Face pipeline.
- **Grids:** ADS λ ∈ {0.05, 0.1, 0.2, 0.4}; PoE γ ∈ {0.2, 0.4, 0.6} (0.6 is the Distillation Game's GSM8K setting).
- **Choice:** the largest value whose accuracy is at least the baseline's minus 5 points. If none qualifies, the
  defense is reported at its weakest grid value with its cost.
- With 100 puzzles one point of accuracy is one puzzle; the intervals are reported.

**Defended students.** At the chosen strength the teacher writes one solution per training puzzle (2,000). D_ADS and
D_PoE are trained with §18's recipe, seeds 123–125, first answers. Quantities and outcomes as in §21:
- protection = U − max(D_φ, F_D);
- share removed = protection / (U − F_D);
- cost: the teacher's calibration accuracy at that strength, and its finished rate.

**Proxy gradient (2026-11-17 22:01).** `scripts/kk_defend.py grad`: 128 held-out solutions, gradient norm 2.49,
2.6 min (`outputs/kk_defense/proxy_grad.pt`). The first try ran out of memory: the proxy's full-vocabulary logits
for a batch of 2 × 8,704 tokens in fp32 take 9.6 GiB. The loss is now computed with the vocabulary projection in
checkpointed chunks of 1,024 positions. Checked against the model's own loss on one batch: loss 0.6557208 both ways,
gradient relative difference 2.2e-5. The method is unchanged.

**Timing pilot result (2026-11-18 01:52).** GPUs 0–2 for the teacher, GPU 3 for the proxies; 16 puzzles, one batch
of 16; no accuracy read (`outputs/kk_defense/timing/*/summary.json`).

| Setting | Minutes | Sec per step | Tokens per sec | Minutes per 100 puzzles | Solutions at the 8,192 cap | Peak memory, GiB (GPUs 0–3) |
| --- | --- | --- | --- | --- | --- | --- |
| Standard | 42.1 | 0.308 | 20.1 | 263 | 2 of 16 | 30.8 / 33.6 / 32.2 / 0 |
| ADS λ 0.1 | 58.8 | 0.431 | 15.3 | 367 | 4 of 16 | 30.8 / 33.6 / 32.2 / 21.0 |
| PoE γ 0.4 | 51.3 | 0.376 | 32.0 | 320 | 11 of 16 | 30.8 / 33.6 / 32.2 / 10.6 |

- Every batch ran the full 8,192 steps, because one capped solution holds the whole batch. With the teacher's 12.5%
  cap rate in step 2, almost every batch of 16 has one. Sorting puzzles by the length of an earlier try saves about
  7%: on the pilot's 200 puzzles at N = 7–8, try 1 and try 2 lengths correlate at 0.42, and sorting by try 1 costs
  0.93 of the all-capped time for try 2.
- **Full plan at this speed:** calibration (baseline 4.4 h, ADS 4 × 6.1 h, PoE 3 × 5.3 h) 45 h; 2,000 solutions
  with ADS 122 h and with PoE 107 h. Total 274 h, about 11.4 days of one GPU group, plus 6 students.
- **Verdict:** over the 3-day budget. The authors decide on a smaller plan.

**Authors' decision (2026-11-18): ADS only, after a speed check.** Fixed before any calibration result.
- **PoE is not run.** It is reported as not run, with the reason: about 4.5 days of one GPU group for its 2,000
  solutions. ADS is the published defense; PoE uses the same proxy's logits in a simpler way.
- **Speed check, part 1: static cache with compilation.** Hugging Face `generate` with `cache_implementation="static"`
  (which compiles the forward pass) was probed on Qwen2.5-3B split over 2 GPUs, 8 prompts, greedy, 256 steps
  (`outputs/kk_defense/speed/compile_probe*`).
  - With flash attention it was slower: 92 and 204 ms per step against 58 ms eager.
  - With sdpa, compilation failed.
  - Dropped.
- **Speed check, part 2: batch size.** Batch 25 instead of 16, so that the 100 calibration puzzles fill 4 batches.
  - The estimated peak is 41 GiB on the busiest teacher GPU: 22 GiB of weights plus the cache, which grows with the
    batch from 12 GiB at 16.
  - The first calibration run checks it. If it runs out of memory, every run uses batch 20 instead, recorded here.
  - The first run also gives the speed at batch 25.
  - Batch size changes only which puzzles share a batch, not the method.
- **Calibration:** unchanged from the rule above. Baseline, then ADS λ ∈ {0.05, 0.1, 0.2, 0.4} on the 100
  calibration puzzles, one try each, strict checker.
  - The choice is computed by `scripts/kk_defend.py choose` from the stored tries: the largest λ whose number of
    correct puzzles is at least the baseline's minus 5.
  - Wilson 95% intervals are reported for each accuracy.
- **Defended solutions:** at the chosen λ, one try per training puzzle (all 2,000), batch 25. Puzzles are sorted by
  the teacher's step-2 length (speed only, about 7%).
  - The run resumes from its stored tries after a crash. A resumed run's sampling seeds differ from an uninterrupted
    run's, which is recorded if it happens.
- **D_ADS:** built like U (§18): every one of the 2,000 solutions, as written, wrong or unfinished included. Students
  D_ADS_s123–125 use §18's recipe, scored on first answers (`test_eot`).
- **Outcomes:** as in §21. Protection = U − max(D_ADS, F_D); share removed = protection / (U − F_D). Cost: the
  teacher's calibration accuracy at the chosen λ, its finished rate, and its accuracy on the 2,000 training puzzles.
- **GPUs:** group A (0–2 teacher, 3 proxies). Each run waits until all four GPUs are free of other processes and
  have 42 GB free.
  - Group A stayed busy (other jobs on GPUs 0–2) from 07:28. When GPU 5 came free, the chain moved to group B
    before generating anything: GPUs 4–6 for the teacher, 7 for the proxies. It started at 13:50 on 2026-11-18. The
    group-A wait is in `logs/kk_ads_groupA_waited.log`.
- **Estimated time:** calibration about 19 h; 2,000 solutions about 76 h; 3 students about 5 h in parallel. About
  4 days in all.

**Calibration result (2026-11-18 13:50 to 2026-11-19; `outputs/kk_defense/calib/*/summary.json`, `tries.jsonl`).**
Batch 25 fit: peak 40.7 GiB on the busiest teacher GPU and 26.4 GiB for the proxies.

| Setting | Correct of 100 (strict checker, the rule) | Wilson 95% | Finished | Unreadable (strict) | Sec per step | Minutes |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline (λ 0) | 75 | 65.7 to 82.5 | 85 | 15 | 0.438 | 239 |
| ADS λ 0.05 | 53 | 43.3 to 62.5 | 86 | 27 | 0.568 | 310 |
| ADS λ 0.1 | 1 | 0.2 to 5.4 | 85 | 97 | 0.560 | 306 |
| ADS λ 0.2 | 0 | 0.0 to 3.7 | 76 | 100 | 0.560 | 306 |
| ADS λ 0.4 (stopped at 50 puzzles) | 0 of 50 | 0.0 to 7.1 | 25 of 50 | 50 of 50 | – | – |

- **Under the rule, no λ qualifies.** Qualifying needs at least 70 correct, and the weakest value, 0.05, has 53.
- **ADS changes the answer format.** Where the teacher writes "Olivia is a knight", ADS writes "Olivia - Knight" or
  "Knight - Logan". The strict checker (§17) cannot read those lines.
- **Diagnostic, not the rule:** a lenient reader that accepts each numbered line naming one person and one of
  knight/knave, in any order. It reads 75 / 60 / 38 / 2 correct for λ 0, 0.05, 0.1, 0.2, and 0 of 50 at 0.4. So λ 0.05 also fails under
  lenient reading (−15). Formatting explains part of the strict drop (λ 0.1: 1 strict, 38 lenient), and the rest is
  wrong answers.
- **Speed at batch 25:** 0.0175 s per puzzle-step for the baseline (0.0193 at batch 16) and 0.0224 for ADS (0.0269).
  That is a 1.1–1.2× gain, not the 1.4× estimated. 2,000 ADS solutions would take about 95 h.
- **Stopped before the full run (2026-11-19 10:58).** The chain would have gone on to the pre-written fallback
  (2,000 solutions at λ 0.05). It was stopped by hand under CLAUDE.md rule 1, and the λ 0.4 run was left to finish.
  - λ 0.05 costs 22 points (strict), which no provider would accept.
  - Its training text would teach the student an answer format the strict scorer cannot read. That would lower
    D_ADS for a reason that has nothing to do with what the solutions teach.
  - The authors decide how to continue.

**ADS closed (authors' decision, 2026-11-19).** The paper's main aim moved to answers-only fine-tuning (§24). ADS is
reported from the calibration alone: on this testbed no strength in the grid keeps the teacher within 5 points, and
the weakest, λ 0.05, costs 22 points. No defended solutions or D_ADS students are made. The λ 0.4 run is left to
finish so the grid is complete; its row is added when it ends.

## 24. Phase 0 of the answers-only method: the setup with an instruction-tuned student (2026-11-19; teacher and gap threshold confirmed by the user; not yet approved to run)

From this section on, results may be interpreted after they are seen and the approach changed after analyzing them
(user, 2026-11-19; CLAUDE.md rule 1). Each entry records what is run, what came out and why the next step changed.

**Why.** From 2026-11-18 the main aim is an answers-only fine-tuning method that gets as close as possible to the
student fine-tuned on the teacher's solutions (`results/plan_answer_only.md`, `results/lit_answer_only.md`). The
student becomes an instruction-tuned model.
- With the field's teacher, Llama-3.2-3B-Instruct already solves 81.7% of GSM8K-Platinum against the teacher's
  87.2%, and 48.0% of MATH against 63.1% (§12). So little is left to close.
- Phase 0 measures that gap with the field's teacher, and with a bigger teacher, before any method work. The
  field's case goes to the appendix.

**Students.**
- **Llama-3.2-3B-Instruct** (`unsloth/Llama-3.2-3B-Instruct`), the main student, with its own chat template.
- Qwen2.5-3B-Instruct, the second student, is measured in Phase 3, not here.

**Setups** (the same training problems and seeds 123–125 as §12; test sets GSM8K-Platinum, 1,209 problems, and MATH,
5,000):
- **S1, the field's setup (appendix case).** Teacher R1-Distill-Qwen-7B with 1,024 new tokens and answer forcing.
  The stored teacher solutions of §12 are reused; no new teacher run.
- **S2, a bigger teacher (main candidate).** DeepSeek-R1-Distill-Qwen-32B, the same family as the field's teacher.
  - Generation: temperature 0.6, top-p 0.95, up to 8,192 new tokens, the setup's answer forcing, one solution per
    training problem.
  - It is also scored on both test sets with the same settings.
  - The 1,024-token limit is what makes the field's 7B teacher weaker than an ordinary instruct model (§9.4); S2
    lifts both the model size and the budget.

**What is measured, per setup, dataset and seed.**
- **I0:** the untrained instruct student.
- **U_inst:** the instruct student fine-tuned on that setup's teacher solutions, with the setup's recipe (LoRA r 128,
  3 epochs, lr 5e-4).
  - For S2 the maximum training length is raised from 2,048 to 8,704 tokens (a deviation, recorded), and truncation
    is reported.
- **Evaluation:** the setup's evaluation (temperature 0.6, answer forcing), with 1,024 new tokens for S1 and 8,192
  for S2. I0 is scored under both budgets, so each comparison uses the same budget.

**Rule.**
- **The gap** is G = U_inst − I0: seed-paired, 3-seed mean, 95% interval with t(2 df).
- **A (setup, dataset) is a development setting** if G ≥ 10 points and its interval excludes 0.
  - Why 10: methods must differ by a few points and still be readable as shares of the gap. Seed noise here is about
    1–2 points.
- **If S2-MATH passes and S2-GSM8K does not:** MATH is the main dataset and GSM8K is reported as a narrow-gap case.
- **If no S2 setting passes:** stop and report to the authors.
- **S1 is reported whatever its G** (the appendix case: the literature's setup leaves a narrow gap).
- **Also reported:** the teacher's test accuracy, the share of training problems where the teacher's answer is
  wrong, and the train–test gap of I0 (contamination).

**Cost (estimate).**
- S2 teacher solutions and teacher test scores: about 12 h on 2 GPUs.
- U_inst: S1 about 3 h per run; S2 about 3 h (GSM8K) and 9 h (MATH) per run.
- Evaluations: 1–3 h each.
- Total about 85 GPU-hours, 1.5–2 days on 4–6 GPUs.

**Check before the full runs:** a smoke test of each new piece (the S2 teacher on 20 problems; U_inst training for a
few steps; evaluation on 50 problems). It only checks that the pipeline runs, and has no threshold.

**Go-ahead 2026-11-19 11:40. Implementation** (`scripts/56_phase0.sh`, `scripts/dg_teacher_vllm.py`; outputs
`outputs/phase0/`):
- **One S2 solution per distinct problem.** The setup's example ids are positions after a per-seed shuffle, so
  problems are matched by text. The three seeds' training splits hold 7,252 distinct GSM8K and 7,221 distinct MATH
  problems. The 32B teacher writes one solution for each (sampling seed 123), and each seed's file is cut from them in
  that seed's order. The test split is the same set for every seed and is generated once.
  - The field's runs regenerated the teacher per seed. One release per problem is the realistic case for an attacker,
    and it costs a third less.
- **S2 teacher:** one vLLM instance with tensor parallelism over GPUs 4–7. On a GPU pair the 32B weights leave about
  8 GB per GPU for the cache.
- **S2 training cut-off:** 9,216 tokens with micro-batch 1 × accumulation 24, the setting §10.1 used for 8,192-token
  solutions (not 8,704 as first written).
- **Scoring:** `scripts/dg_score_vllm.py` (the §10.1 scorer, checked against the setup's HF loop on MATH), now with
  the setup's GSM8K prompt for GSM8K configs. Scorer check on GSM8K: I0 at 1,024 tokens, seed 123, must land within
  1.5 points of Llama-3.2-3B-Instruct's own test score in the setup's HF loop (82.13%, the Solo B run's writer score;
  MATH 48.2%).
- **Smoke tests (2026-11-19, GPU 0):**
  - The scorer on 50 GSM8K problems: I0 80% (the instruct model writes "ANSWER: $18"; answer forcing and the checker
    read it).
  - U_inst training on 48 of the field teacher's solutions, then scoring the merged student on 50 problems: 72%. The
    student writes in the teacher's format, ends with the boxed answer and stops (92% finished).
  - The 32B smoke test (16 MATH problems) runs first in the launcher.
- **Scorer check on GSM8K passes:** I0 at 1,024 tokens, seed 123, 82.13% (993 of 1,209), the same as the setup's HF
  loop (82.13%).
- **Scorer check on MATH passes:** I0 at 1,024 tokens, seed 123, 49.04% against 48.2% in the setup's HF loop.
- **S2 teacher on GSM8K (2026-11-19 12:05–12:50, GPUs 4–7):**
  - 96.5% on the test set (field teacher 87.2%), 94.5% on the 7,252 distinct training problems.
  - Per-seed training files: 94.8 / 94.5 / 94.5%.
  - Every solution finishes; median 423 tokens (90th percentile 606).
- **I0 on GSM8K:** 82.13 / 81.89 / 82.71% at 1,024 tokens and 82.13 / 81.89 / 82.80% at 8,192 (median 180 tokens).
  The teacher is 14 points above the untrained student, so U_inst − I0 can be at most about that.
- **I0 on MATH:** 49.04 / 47.78 / 48.96% at 1,024 tokens (87–89% finished) and 49.10 / 47.58 / 49.14% at 8,192
  (94–96% finished; median about 350 tokens).
- **First S1 student (2026-11-19 17:27): U_inst on MATH, seed 123, 39.66% at 1,024 tokens, 9.4 points below I0.**
  - It finishes 31.8% of its answers (I0: 87.4%), median 1,024 tokens: it imitates the field teacher's long
    reasoning, much of which was itself cut at 1,024 tokens, and runs to the limit.
  - Training: 5,000 solutions, 86 min, 4 longer than the 2,048-token cut-off.
  - On the field's setup the teacher's solutions make the instruct student worse on MATH.
- **S2 teacher on MATH:** running since 12:50; 10,173 of 12,221 items by 18:10 (about 1,120 tokens/s, slower than
  estimated). Expected about 19:50.

**Addendum (2026-11-19 evening, the user's request): Phase 0 on the puzzle testbed** (`scripts/57_kk_instruct.sh`).
- **Runs, seeds 123–125, the 1,000 step-2 test puzzles:** the instruction-tuned student untrained (I0), and trained
  on the teacher's 2,000 step-2 solutions (U_inst, `train_sets/U.jsonl`, the §18 recipe).
  - `kk_sft.py` and `kk_eval.py` take `--student`; the base-student default is unchanged.
- **Scoring:** as §18, reported on first answers and on the last answer. The instruct student can end its turn, so
  the last answer is meaningful here.
- **Compared with:** the base student (U 48.8%, F_D 22.4%, F_A 1.7%, untrained 0.3%).
- **For the examples file only:** the teacher on 6 test puzzles (3 at N = 7, 3 at N = 8), one try each.
- **GPUs:** only those Phase 0's queue has released, plus GPUs 1–3 when free of other processes. Per-GPU locks keep
  the two launchers apart.
- **ADS λ 0.4 stopped after 2 of its 4 batches** (50 puzzles) to free GPUs 4–7 (§23 closed).

**Scorer fix (2026-11-19, 19:40).** A bug in our own scorer (`scripts/dg_score_vllm.py`) made it stricter than the
setup's checker for replies that already contain `</think>`.
- **How the setup checks** (`clean_sweep.generation.core._answer_force`, `eval/correctness.py`): it appends the cue
  `\n\n**Final Answer**\n\[\boxed{` to the reply, lets the model write the forced answer, and splits the joined text
  at the cue. It accepts if the full text, the reply alone, or the forced answer alone matches the reference.
- **The bug:** our scorer stripped the reply and the cue and joined them with a single newline. When the reply had
  `</think>`, the cue then lost one of its two leading newlines, the checker did not find it, and read only the full
  text, which took the forced answer. A correct reply followed by a malformed forced answer was scored wrong.
  Replies without `</think>` get `\n</think>` before the cue, so the cue survived; their scores are unaffected.
- **Affected stored files** (offline count: rows whose reply alone matches the reference but are scored wrong):

  | File | Replies with `</think>` | Scored wrong, reply right | Stored | At least |
  | --- | --- | --- | --- | --- |
  | I0, all 12 files | 0 | 0 | unchanged | unchanged |
  | S1 MATH s123, 1,024 | 1,054 | 89 | 39.66% | 41.44% |
  | S1 MATH s124, 1,024 | 960 | 71 | 39.98% | 41.40% |

  "At least" adds only those rows. The forced answers were not stored, so the third part of the rule (the forced
  answer alone) cannot be rechecked offline; an exact figure needs a rescore.
- **Fix:** the scorer now joins reply, cue and forced answer exactly as the setup does, and stores the forced part
  (`forced`). Checked on the two 32B smoke cases (`\dfrac` written as `dfrac`; `(0, 0, 3)` as `0, 0, 3`): rejected
  before, accepted after. Python files are read when a job starts, so every scoring job from S1 MATH s125 on uses the
  fix. S1 MATH s123 and s124 are rescored with the fixed scorer (go-ahead from the user, 19:50): the old files are
  kept as `test_1024_old_scorer.json`, and two jobs `s1 math 123|124` are put at the front of the queue's remaining
  lines. They find `train_info.json`, skip training and only score the stored model (about 27 minutes each).
- **The 32B teacher's `correct` field** (`scripts/dg_teacher_vllm.py`, same join) is affected in the same way. It is
  not used for training (students train on every solution). Offline, with the setup's rule: GSM8K training 94.75% →
  94.77% (1 problem), test 96.53% → 96.61% (1), MATH smoke 14 → 16 of 16. The script is fixed after the MATH merge,
  which runs from it.

**32B teacher on MATH (merged 2026-11-19 20:39; `outputs/phase0/teacher32b_math/`).** Offline analysis with the setup's
checker rule (`forced_full.py`, scratch):

| Split | n | Finished | Own answer right | Stored `correct` (bug) | Setup's rule | Own right, forced wrong | Cut off; right after forcing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Distinct training | 7,221 | 94.7% | 91.2% | 87.0% | 92.3% | 532 (8.1% of own-right) | 383; 134 |
| Training, seed 123 | 5,000 | 94.8% | 91.2% | 87.2% | 92.3% | 369 (8.1%) | 261; 97 |
| Test | 5,000 | 96.1% | 92.4% | 88.0% | 93.3% | 322 (7.0%) | 197; 73 |

- Median length 1,392 tokens, 90th percentile 5,247 (training). So 5.3% of training solutions are cut at 8,192; the
  smoke sample (16 of 16 finished) was too small to show this.
- **The forced answer is often not a copy of the teacher's own answer.** Seed 123's training file: 195 forced lines
  are a single letter ("B", "D"), 92 are the literal "ANSWER" (from the setup's system prompt, "\boxed{ANSWER}"),
  114 run on past the box (for example "Wait, hold on! I just realized I made a mistake"). Others drop a sign
  (−48 → 48) or part of a fraction (1/108 → 108). Every finished solution has its own boxed answer after `</think>`.
- **Consequence for U_inst:** the student is trained on solution + forced line (setup's format). In 7.4% of seed
  123's MATH targets the last line contradicts the teacher's own correct answer. The checker reads the last boxed
  answer, so a student that learns to end this way loses those points. GSM8K: 6 of 4,963 (0.1%).
- **Decision (the user, 2026-11-19 20:55): S2 targets end in the teacher's own answer.** When the teacher finished,
  its training target's final line is built from its own answer; when it was cut off, the forced line stays (its
  only answer). Format and every other part unchanged; no reference answers used; S1 stays exactly as the setup. The
  same rule defines the 32B teacher's released answer for Phase 1.
- **Implementation** (`scripts/dg_own_final.py`; final files written 20:57:52, before the first S2 training started at 21:05:54). "Own answer" =
  every box on the line after `</think>` where the last box starts. Two refinements after looking at the output:
  - the first version took only the last box; 44 MATH cases with several answers ("the solutions are \boxed{81} and
    \boxed{256}") lost all but one, so every box on that line is kept, joined as the forced line joins them;
  - a line-by-line search missed a box spanning lines (a matrix, 1 row), so boxes are found over the whole text.

  The originals are kept as `train_s<seed>_forced.json`, and each row keeps its forced line (`af_final_forced`) and
  says which rule applied (`final_from`); counts in `own_final.json`.

  | Seed 123 | Final line from own answer | Kept forced (cut off) | Line changed | Checker accepts the final line, forced → own |
  | --- | --- | --- | --- | --- |
  | MATH | 4,739 | 261 | 1,337 | 4,162 → 4,493 |
  | GSM8K | 5,237 | 1 | 1,032 | 4,957 → 4,958 |

  Most changed lines are the same answer in the teacher's own writing: GSM8K "\$135,000" or "13 \text{ years}"
  where the forced line has "135000" or "13" (the setup's prompt asks for units in the box), MATH spacing and
  `\dfrac`. 26–41 forced lines per file ran on past the box ("\boxed{60}\] feet"). One MATH case goes the other way
  (forced right, own wrong). Seeds 124 and 125 are alike (`own_final.json`).
- **The teacher script's checker join** is fixed the same way as the scorer's (the stored `correct` and
  `summary.json` of both teacher runs keep the old join; the table above has the setup's rule).

**S1 MATH rescored with the fixed scorer (2026-11-19 21:06; GPUs 4 and 5).**

| Seed | Old scorer | Fixed scorer | Finished, old → new | Replies identical | Wrong → right | Right → wrong |
| --- | --- | --- | --- | --- | --- | --- |
| 123 | 39.66% | 41.18% | 31.8% → 31.8% | 5,000 of 5,000 | 88 | 12 |
| 124 | 39.98% | 41.08% | 27.0% → 25.9% | 149 of 5,000 | 489 | 434 |

- **Seed 123** generated the same replies, so its change is the scorer alone. The 12 that became wrong are replies
  where the student itself wrote a forced-style line with a wrong answer after its own correct one ("... solutions
  is \boxed{-6}. **Final Answer** \[\boxed{B}\]"), learned from the field teacher's forced lines. The old join
  split at the student's own line and read the answer before it; the setup's join splits at the scorer's cue and
  reads the last box, B. The fixed scorer does what the setup does, so these count as wrong. This is the failure the
  S2 decision above removes from the 32B teacher's targets.
- **Seed 124** generated different replies in the rescore (149 identical), although the seed and the sampling
  settings are the same; vLLM's seeded sampling is not reproducible across these two runs. Its change mixes the
  scorer fix with resampling (the offline lower bound on the old replies was 41.40%). Every later score comes from
  one run with the fixed scorer.
- Old files kept as `test_1024_old_scorer.json`.

**Phase 0 result (queue done 2026-11-20 06:12; `scripts/phase0_table.py` → `outputs/phase0/table.json`).** G = U_inst −
I0, seed-paired, 95% interval with t (2 df). Accuracy in %, the setup's checker with the fixed scorer.

| Setting | I0 (s123 / s124 / s125) | U_inst (s123 / s124 / s125) | U_inst finished | G | 95% CI | Gap rule (≥ 10, CI > 0) |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K, S1 (7B teacher, 1,024) | 82.13 / 81.89 / 82.71 | 78.16 / 78.58 / 79.24 | 73–79% | −3.58 | −4.44 to −2.73 | no |
| GSM8K, S2 (32B teacher, 8,192) | 82.13 / 81.89 / 82.80 | 78.74 / 79.07 / 79.65 | 99.3–99.7% | −3.12 | −3.84 to −2.39 | no |
| MATH, S1 (7B teacher, 1,024) | 49.04 / 47.78 / 48.96 | 41.18 / 41.08 / 40.14 | 26–32% | −7.79 | −10.43 to −5.16 | no |
| MATH, S2 (32B teacher, 8,192) | 49.10 / 47.58 / 49.14 | 48.34 / 49.98 / 49.00 | 62–72% | +0.50 | −3.66 to +4.66 | no |

- **No GSM8K or MATH setting has a gap.** Training the instruction-tuned student on the teacher's solutions lowers
  it on GSM8K (both teachers) and on MATH with the field's teacher, and leaves it unchanged on MATH with the 32B.
- **Where the S2 MATH student loses** (per-problem split, seeds 123 / 124 / 125): replies that finish are better
  than I0 on the same problems (67.0 vs 64.2, 63.0 vs 58.7, 67.6 vs 63.8); replies cut at 8,192 tokens (28–38% of
  all) are worse (17.4 vs 24.0, 17.3 vs 19.7, 16.2 vs 23.2), and 85–94% of them end in a repetition loop (their last
  300 characters occur earlier in the reply). The teacher finished 96% of the test problems. So the recipe (the
  setup's LoRA r128, 3 epochs, lr 5e-4, built for 1,024-token solutions) leaves the student looping on long
  solutions; on finished replies the gain is +3 to +4 points.
- **GSM8K S2:** 99% of replies finish, and they are worse than I0 on the same problems (79.2 vs 82.2, 79.2 vs 81.9,
  79.8 vs 82.9).
- **MATH S1:** the student runs to 1,024 tokens on 68–74% of problems (the field teacher's solutions were cut there
  too), and is worse than I0 on both its finished and its cut replies.
- **Puzzles** (1,000 test puzzles, last answer / first answer): I0 1.3 / 1.3, 0.5 / 0.5, 0.6 / 0.6 (s123–125);
  U_inst s124 54.1 / 53.8, s125 54.5 / 54.5 (finished 77%). Seed 123's U_inst never ran: both training attempts
  ran out of memory (another process, PID 4005144, took 28.7 GB of GPU 3 at 19:29), and the launcher logged the job
  "done" (bug: the retry loop's status was that of its last `sleep`; fixed). Rerun of that approved job queued
  2026-11-20 14:00 (`u 123`, the same command; the failed log kept as `train_oom_2026-11-19.log`).
- **Interpretation (after the numbers, as §24 allows):** with an instruction-tuned 3B student and 5,000 problems, the
  GSM8K/MATH setups leave nothing for an answers-only method to close; the puzzle testbed leaves about 53 points.
  The next step is the authors' decision (asked 2026-11-20).

**Puzzle seed 123 (U_inst rerun, scored 2026-11-20 16:43).** 53.1% last answer / 53.0% first (finished 75.1%). Over
3 seeds: I0 0.8%, U_inst 53.9%, **G = +53.1 points (95% CI 50.3 to 55.9)**; per seed +51.8, +53.6, +53.9.

## 25. Phase 1: answers-only baselines on the puzzle testbed, and a recipe check on MATH (2026-11-20; go-ahead from the user)

**Decisions of the user (2026-11-20, ~15:50).**
- **The puzzle testbed is the main setting** for the answers-only method. GSM8K/MATH are reported as "no gap for an
  instruction-tuned attacker" (§24), briefly in the main text and in full in the appendix.
- **One recipe check on MATH** (below).
- **Phase 1 on the puzzles** goes ahead (below).
- **GPUs:** GPUs 4–7 are not used until 2026-11-22 ~15:40 (jafar's runs); GPU 2 is kept free for the user's other
  project. Our jobs use GPUs 0, 1 and 3.

### 25.1 Recipe check: the 32B-teacher MATH student with full fine-tuning

**Why.** The S2 MATH student loops on 28–38% of test problems (§24). The setup's recipe is LoRA (rank 128, learning
rate 5e-4), built for 1,024-token solutions. Reasoning distillation is usually done by full fine-tuning (DeepSeek's
R1-Distill models, s1, LIMO). If full fine-tuning also leaves the gap far below 10 points, the result of §24 does not
depend on the LoRA recipe.

**Run.** Seed 123, the same 5,000 training texts as S2 MATH s123 (`teacher32b_math/train_s123.json`, own-answer
final lines), the same chat template, prompt, completion-only loss and data order as the setup's trainer.
- Changes: every weight trained (no LoRA); learning rate 1e-5 (LoRA's 5e-4 is far too high for full fine-tuning).
- Kept: 3 epochs, cosine schedule, 3% warm-up, weight decay 0.1, clipping 1.0, effective batch 24 (1 × 24), cut-off
  9,216 tokens, bf16 compute.
- To fit one 46 GB GPU: fp32 weights with 8-bit AdamW (bitsandbytes) and Liger's fused cross-entropy (both
  installed into `.venv-dg` without changing other packages), gradient checkpointing.
- Code: `scripts/dg_full_ft.py`. A smoke test on the 24 longest texts checks memory first.
- Scored like every Phase-0 student: `dg_score_vllm.py`, 8,192 tokens, the fixed scorer. One GPU, about 4 h
  training and 4–5 h scoring.

**Read-out.** Accuracy, gap against I0 s123 (49.10%), finished share and the loop share, next to the LoRA student
(48.34%). One seed, so the read-out is descriptive: whether the gap moves toward 10 points.

**Smoke test (2026-11-20 15:57, GPU 0).** One optimizer step on the 24 longest texts (209,789 tokens): peak memory
35.6 GiB, 54 s. 3 of the 24 come out at 9,217 tokens after the setup's decode/re-encode and lose their last token at
the 9,216 cut-off, exactly as in the setup's LoRA trainer (same code path). Launched with
`scripts/58_recipe_check.sh` (G=0): training, then scoring.

**First launch ran out of memory (16:00, at the second optimizer step).** The smoke test covered only the first
step, before the optimizer states exist; from the second step on, fp32 weights, fp32 gradients and the 8-bit states
(32-bit for the tied embedding) do not fit together. Fix, a memory setting only: bitsandbytes' paged 8-bit AdamW
(states move to CPU memory under pressure) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; the smoke test
now runs two steps (48 longest texts: 51 s per step, no slowdown from paging). Relaunched 16:03. Log of the failed
run: `logs/phase1/fullft_math_s123_oom1.log`.

**Result (2026-11-21 00:24; `outputs/phase0/S2full/math_s123/`).** Training 216 min, average loss 0.294 (LoRA 0.303),
final loss about 0.19. Test accuracy **52.94%** against I0 49.10% and the LoRA student 48.34%: **G = +3.8 points**
(one seed), +4.6 over LoRA (749 problems right only with full fine-tuning, 519 only with LoRA). Finished 63.7% (LoRA
62.4%): on finished replies 72.2% against I0's 64.7% on the same problems; on the 1,813 cut replies 19.1% against
21.7%, and 93% of them end in a repetition loop. Full fine-tuning learns more from the solutions, but the student
still loops on a third of the problems, and the gap stays far below the 10-point rule. The §24 conclusion (no usable
gap on MATH for the instruct student) does not depend on the LoRA recipe.

### 25.2 Phase 1 on the puzzles: answers-only baselines with the instruction-tuned student

**Attacker.** Llama-3.2-3B-Instruct with the teacher's released answers: for each of the 2,000 step-2 training
puzzles, the CONCLUSION of the teacher's solution (`outputs/kk_step2/gen`, the teacher's first try, as §20). 1,712
puzzles have a readable answer, of which 1,435 are right; 288 have none. An answers-only attacker has nothing for
those 288: they are reported and not trained on, as in §20.

**Fixed for every baseline.** Seed 123 for development. The §18 training recipe (`kk_sft.py`: LoRA rank 128, lr
5e-4, 3 epochs, effective batch 24) unless a method says otherwise. Scored like U_inst (§24): 1,000 test puzzles, one
try, temperature 0.6, top-p 0.95, 8,192 tokens, the strict last-answer checker, first answer also reported. No
choice is made on the test puzzles; a method with a free setting fixes it before its run here.

**Baselines, in order.**
1. **Answers only (B1).** SFT on the puzzle → "CONCLUSION:" plus the teacher's labels, no reasoning (the
   answers-only SFT baseline of Ma et al. 2026).
2. **Self-training with hints (B2), §20's recipe for the instruct student.** The untrained instruct student tries
   each puzzle 8 times in its chat format (the U_inst prompt); the first try whose CONCLUSION equals the teacher's
   answer is kept. Puzzles never solved get the teacher's answer as a hint for 8, then 32 more tries; the kept text has
   no hint. SFT on the kept texts. Same sampling as §20 (temperature 0.8, 2,048 tokens), so that B2 and §20 differ only
   in the student and its prompt format.
3. **RL rewarded by the teacher's answer (B3), and a partial-credit variant (B4).** GRPO from the untrained instruct
   student; reward 1 if the CONCLUSION equals the teacher's answer (B3), or the share of people labeled as the teacher
   labeled them (B4), 0 for an unreadable answer. The untrained student solves about 1% of puzzles, so B3's reward
   is rarely non-zero; B4 tests whether partial credit fixes that. Settings (group size, rollout length, steps) are
   fixed here before the first run, after a smoke test of the code.
4. **Controls** for the best of B1–B4: the student's own majority answers instead of the teacher's (what the
   teacher's answers add), and the true answers (an upper reference).

**Implementation and smoke tests (2026-11-20 16:05).** Launched 16:07.
- B1: `scripts/kk_phase1_sets.py answers` → `outputs/kk_phase1/train_sets/B1_answers.jsonl` (1,712 examples, 1,435
  with the right answer, 288 puzzles without a teacher answer).
- B2: `scripts/kk_star.py --student unsloth/Llama-3.2-3B-Instruct`: the instruct student tries in the U_inst prompt
  (`kk_eval`'s chat template string, stop at `<|eot_id|>`), no demonstration; the hint is appended to the puzzle as
  in the base version. Smoke test (20 puzzles, 2 tries per round): 18 with a teacher answer, 1 solved alone, 16 with
  the hint, 1 unsolved. The kept texts end in "In conclusion: 1. Henry is a knight. ..."; the checker reads that
  format (its fallback: the lines after "conclusion:" up to the first blank line), at training and at test alike.
- Launcher `scripts/59_phase1_kk.sh`: one job per GPU among 3 and 1, each waiting until its GPU is free of other
  processes; B1 then B2, seed 123.

**Read-out per baseline.** Test solve rate; the share of U_inst's gain recovered, (M − I0) / (U_inst − I0); and the
gap decomposition on the training puzzles where possible: puzzles the method never solves (coverage), kept texts
whose reasoning is wrong while the answer matches (spurious rationales, from §20 the main failure of the base
student), and wrong teacher answers learned (277 of the 1,712).

**B3/B4 settings (fixed 2026-11-20 17:39, after the smoke test; `scripts/kk_grpo.py`, launcher
`scripts/60_phase1_rl.sh`).**
- GRPO (TRL 0.19.1, vLLM colocated) from the untrained instruct student; LoRA rank 128, alpha 128, all projections;
  Dr. GRPO loss (no length or std normalization); learning rate 5e-6 constant after 3% warm-up; KL 1e-4 toward the
  untrained student (the settings of the stable level-A GRPO in §15); 16 puzzles × 8 replies per step; temperature
  1.0; replies up to 4,096 tokens; **1 epoch over the 1,712 puzzles with a teacher answer (107 steps)**; adapters
  every 25 steps; the final adapter is scored with `kk_eval.py` like every student. No checkpoint selection.
- B4 rewards the share of people labeled as the teacher labeled them, B3 exact agreement; both rewards and agreement
  with the reference solution are logged, only the chosen one is trained on.
- Smoke test (2 steps, B4, GPU 1): 13.7 min, peak 29.9 GiB; replies 1,280 tokens on average, 9–14% cut at 4,096;
  exact-agreement reward 0 of 256 replies, partial 0.11–0.12 (unreadable answers score 0). A first try with 4
  replies per forward pass ran out of memory in the log-probabilities (7.8 GiB); 2 per pass fits.
- Why one epoch: two would take about 21 h per run on one GPU. B3 is expected to learn little, since its reward is
  almost always 0 at the start (about 1 group in 12 has a success at a 1% solve rate); it is run with the same budget
  so that B3 and B4 differ only in the reward.
- Order: B4 on GPU 1 now; B3 on GPU 3 after B2 has finished there (the launcher waits for "GPU 3: queue empty" in
  `logs/phase1/kk.log`), else on GPU 0 after the recipe check.

**B1 result (seed 123, 2026-11-20 17:11; `outputs/kk_phase1/students/B1_s123/`).** The student trained on the
teacher's answers alone writes only a CONCLUSION (67 tokens on average; every answer readable) and solves **14.9%** of
the test puzzles (7 people 14.4%, 8 people 15.4%), against I0 1.3% and U_inst 53.1% for seed 123: a recovered share of
(14.9 − 1.3) / (53.1 − 1.3) = 26%. Guessing would solve under 1% (1 in 128 or 256 assignments); on average it labels
66% of the people right. So the answers alone teach a reasoning-free shortcut that is far above chance, and above the
base student's level-A floor (F_A 1.7%, §20). Seeds 124–125 come later with the other baselines.

**B2 result (seed 123, 2026-11-20 17:43; `outputs/kk_phase1/star_s123/`, `students/B2_s123/`).** Tries: 95 of the
1,712 puzzles solved in the student's own 8 tries (5.5%), the other 1,617 in the first hint round, none left
unsolved; 1,435 kept texts end in the right answer (the teacher's errors pass through). The student trained on them
solves **2.6%** of the test puzzles (finished 98.4%, 714 tokens), against I0 1.3%: a recovered share of 2.5%, far
below B1's 26%. As with the base student (§20, F_A 1.7%), almost every kept text comes from a hint round, where the
student states the given answer; training on them teaches little. Self-training with hints is the weakest baseline
so far, and the answers-only shortcut of B1 the strongest.

**B3 and B4 results (seed 123, 2026-11-21 05:17 and 05:47; `outputs/kk_phase1/students/B{3,4}_s123/`).** Both ran
107 steps (B4 689 min, B3 715 min). The rewards hardly moved: B4's partial-credit reward 0.084 → 0.096 → 0.120 (steps
1–10, 46–55, 98–107), exact agreement 0.002 throughout; B3's exact reward 0.002 → 0.003 → 0.005. Replies stayed about
1,200–1,300 tokens, 9–13% cut at 4,096. Test: **B4 0.9%, B3 1.0%** (I0 1.3%): no gain. Most replies never reach a
readable CONCLUSION (an unreadable answer scores 0), so even partial credit gives almost no signal; with learning rate
5e-6 on LoRA, one epoch did not change that. RL from the untrained student, as configured, recovers nothing.

### 25.3 Phase 1, continued: B1 on more seeds, its two controls, and RL from B1 (2026-11-21; go-ahead from the user)

**Why.** B1 (SFT on the teacher's answers alone) is the strongest baseline (14.9%, seed 123); self-training with hints
(B2, 2.6%) and RL from the untrained student (B3 1.0%, B4 0.9%) recover almost nothing. RL failed because the untrained
student rarely writes a readable answer, so its reward is almost always 0. Starting RL from B1, whose answers are all
readable and 15% right, gives a reward that fires from the first step.

**Runs** (launcher `scripts/61_phase1_more.sh`, GPUs 0, 1, 3):
1. **B1, seeds 124 and 125**: the B1 recipe unchanged.
2. **C1, the student's own majority answers (seed 123)**: the untrained instruct student tries every training puzzle
   8 times (U_inst prompt, temperature 0.8, 2,048 tokens, as B2's own round; `scripts/kk_sample.py`); each of the
   1,712 puzzles B1 trains on gets the most common readable answer among the 8 (ties: the first seen); puzzles with no
   readable try are reported and not trained on. Then B1's recipe on these answers. It shows what the teacher's
   answers add over the student's own.
3. **C2, the reference answers (seed 123)**: B1's recipe on the true answers of the same 1,712 puzzles. It shows what
   the teacher's 277 wrong answers cost; no attacker has these answers.
4. **B5, RL from B1 (seed 123)**: B3's settings (§25.2: GRPO, exact-agreement reward with the teacher's answer, Dr.
   GRPO loss, lr 5e-6, 16 × 8 replies per step, 4,096 tokens, one epoch, 107 steps), starting from B1 s123's adapter
   instead of a new one. The KL term's reference is the untrained student (TRL's reference for a PEFT model is the
   base with the adapter off); at 1e-4 its weight is small. Exact agreement, not partial credit, because B1's answers
   are always readable and the test counts only complete answers.

Each student is scored with `kk_eval.py` as before. Seed 123 only for C1, C2 and B5 at this stage.

**Smoke tests and launch (2026-11-21, launched 09:55).** `kk_sample.py` (20 puzzles × 2 tries): 40% of tries readable, 10%
right. B5 from B1's adapter (2 steps): replies 67 tokens, exact-agreement reward 0.07 and 0.12, partial 0.62–0.65,
so the reward fires from the start, unlike B3. Launched with `scripts/61_phase1_more.sh` on GPUs 1, 3, 0.

**Results (2026-11-21 10:22–11:10; `outputs/kk_phase1/`).**

| Student | Seed(s) | Solve rate (%) | Recovered share |
| --- | --- | --- | --- |
| B1, teacher's answers alone | 123 / 124 / 125 | 14.9 / 18.3 / 17.5, mean **16.9** | 30% (95% CI 21 to 39) |
| B5, RL from B1 (exact agreement) | 123 | **22.7** | 41% |
| C2, B1's recipe on the reference answers | 123 | 8.5 | 14% |
| C1, B1's recipe on the student's own majority answers | 123 | 1.6 | 1% |

- **B1, 3 seeds:** B1 − I0 = +16.1 points (10.6 to 21.6); U_inst − B1 = 37.0 (34.0 to 40.0). Every student answers in
  about 67 tokens with no reasoning; all answers are readable.
- **B5:** 107 steps in 45 min. The exact-agreement reward rose 0.106 → 0.204 → 0.266 (steps 1–10, 49–58, 98–107),
  agreement with the reference 0.091 → 0.253, replies stayed at 67 tokens. Not converged after one epoch. RL from
  B1 adds 7.8 points over B1 on seed 123.
- **C1:** the untrained student's tries are readable 42.8% of the time and right 0.89% (2,000 puzzles × 8,
  `own_tries_s123/summary.json`); its majority answer exists for 1,625 of the 1,712 puzzles (87 have no readable try,
  not trained on) and is right on 36. Trained on these, the student stays at I0 (1.6%). So the teacher's answers
  carry B1's gain: +13.3 points over the student's own answers on seed 123.
- **C2, unexpected:** the reference answers train a weaker student (8.5%) than the teacher's (14.9%, same seed),
  although the two sets differ only on the 277 puzzles where the teacher is wrong. One seed; B1's seeds span 14.9 to
  18.3, so 8.5 is outside that range, but C2 needs seeds 124–125 before it is interpreted. A possible reason, not
  tested: the teacher's wrong answers may follow surface patterns of the puzzles that a reasoning-free student can
  pick up, while the true answers of those puzzles do not.

### 25.4 Phase 1: seeds 124–125 for B5 and the controls; B5 for three epochs (2026-11-21; go-ahead from the user)

**Runs** (launcher `scripts/62_phase1_seeds.sh`, GPUs 0, 1, 3):
1. **B5, seeds 124 and 125**: §25.3's settings, each from the B1 adapter of the same seed.
2. **C2, seeds 124 and 125**: B1's recipe on `C2_reference.jsonl`, training seed 124 or 125.
3. **C1, seeds 124 and 125**: B1's recipe on `C1_majority_s123.jsonl` (the majority answers from the seed-123 tries),
   training seed 124 or 125. As for B1 and C2, the training set is fixed and only the training seed changes.
4. **B5 for three epochs (seed 123, `B5e3_s123`)**: §25.3's settings with `--epochs 3` (321 steps, constant learning
   rate), because the reward was still rising at the end of one epoch. Adapters every 25 steps. A development run on
   seed 123: if it helps, the longer schedule is then run on seeds 124–125 before any claim.

Read-out: 3-seed means with 95% intervals for B5, C1 and C2; for C2, whether the reference answers train a weaker
student than the teacher's on every seed (§25.3).

**Results (2026-11-21 12:11–13:40; `outputs/kk_phase1/students/`).** 3-seed means, 95% intervals with t (2 df).

| Student | s123 / s124 / s125 (%) | Mean | Minus I0 | Recovered share | Minus B1 |
| --- | --- | --- | --- | --- | --- |
| B1, teacher's answers alone | 14.9 / 18.3 / 17.5 | 16.9 | +16.1 (10.6 to 21.6) | 30% (21 to 39) | — |
| **B5, RL from B1** | 22.7 / 30.0 / 23.8 | **25.5** | +24.7 (14.1 to 35.3) | **47% (28 to 65)** | **+8.6 (1.7 to 15.5)** |
| C2, reference answers | 8.5 / 19.2 / 3.8 | 10.5 | +9.7 (−10.3 to 29.7) | 18% (−19 to 55) | −6.4 (−24.5 to 11.7) |
| C1, own majority answers | 1.6 / 0.8 / 2.2 | 1.5 | +0.7 (−1.1 to 2.6) | 1% | −15.4 (−20.6 to −10.1) |

- **B5 is the strongest baseline**: RL from B1 adds 8.6 points over B1, with an interval above 0. Its students answer
  in 67–100 tokens without reasoning.
- **The teacher's answers carry the gain**: with the student's own majority answers the recipe stays at I0 on every
  seed (C1 − B1 = −15.4, interval below 0).
- **C2 does not confirm §25.3's unexpected result**: the reference-answer students vary from 3.8% to 19.2% across
  seeds; C2 − B1 = −6.4 with an interval from −24.5 to +11.7. The one-seed difference was within the seed noise of
  this recipe; the "surface patterns" explanation in §25.3 is withdrawn. Answers-only SFT on the reference answers
  is much less stable across seeds than on the teacher's answers (SD 7.7 against 1.8 points); not investigated.
- **Three epochs of RL (B5e3, seed 123): 23.6%** against 22.7% after one. The exact-agreement reward plateaued at
  0.28–0.31 after about 100 steps (0.094 → 0.277 → 0.310 → 0.284, steps 1–10, 101–110, 201–210, 312–321). The
  one-epoch schedule is kept.
- **Phase 1 conclusion:** the best answers-only student recovers 47% of the teacher's gain (25.5% against U_inst
  53.9%) without writing any reasoning; the remaining 28 points are what the teacher's reasoning teaches beyond its
  answers so far. Self-training with hints (2.6%) and RL from the untrained student (about 1%) fail because the
  student almost never reaches the answer by reasoning (95 of 1,712 puzzles in 8 tries).

## 26. Phase 1: where the gap lies (gap breakdown, 2026-11-21; go-ahead from the user)

**Why.** Phase 1's best answers-only student (B5, 25.5%) recovers 47% of the teacher's gain without writing any
reasoning; methods that try to reason (B2, B3, B4) recover almost nothing. Before choosing the Phase 2 method, measure
what the remaining gap is made of. Diagnostic only: reference answers are used here, never for training. Seed 123.

**Measurements.**
- **D1, reaching the answer (coverage and format).** From the untrained student's 8 tries per training puzzle
  (`own_tries_s123`) and its test replies (I0, 3 seeds): the share of readable answers; why the others are unreadable
  (no CONCLUSION, a person labeled twice or not at all, cut at the token limit); the share of people labeled
  correctly in readable answers (chance 50%); puzzles reached at least once.
- **D2, spurious rationales.** The teacher (R1-Distill-Qwen-32B, the stand-in for a strong grader; diagnostic only)
  judges whether a kept B2 text derives its answer from the statements or assumes it: all 95 texts the student found
  alone, 200 random hint-round texts, and as calibration 100 random correct and 100 random wrong teacher solutions
  (`scripts/kk_judge.py`, 2 GPUs, one judgment each).
- **D3, the teacher's errors.** B1, B5 and U_inst (seed 123) answer every training puzzle once (`kk_eval.py` on the
  training pool). On the 277 puzzles with a wrong teacher answer: how often each reproduces the teacher's answer, the
  true answer, or neither; on the 1,435 others, how often it is right (fit to the training answers).
- **D4, where U_inst beats B5 on the test puzzles.** Per-puzzle cross-table (seed 123), by puzzle size, by U_inst's
  reply length and finishing.
- **D5, what the shortcut uses.** A logistic-regression predictor of each person's label from surface counts of the
  puzzle (how often the person is called knight or knave by others, by themselves, under negation, and the operators
  of their own statement), fit on the training puzzles with the teacher's answers; its test solve rate (all people
  right) next to B1's. If comparable, B1's shortcut is largely such surface statistics.

Code: `scripts/kk_decomp.py` (D1, D4, D5 and the D3 tables), `scripts/kk_judge.py` (D2); results in
`outputs/kk_phase1/decomp/` and here.

**Results D1, D4, D5 (2026-11-21 15:50; `outputs/kk_phase1/decomp/breakdown.json`).**
- **D1, the untrained student mostly cannot produce an answer, and its answers are guesses.** Test replies: readable
  46.7 / 45.3 / 40.7% (s123–125); of the rest, 497–554 per 1,000 write a CONCLUSION that does not label every person
  exactly once, 18–34 are cut at the limit, 6–18 have no CONCLUSION. Readable answers label 51.9–53.5% of the people
  right (chance 50%). On the training puzzles (8 tries each): 42.8% readable, 0.9% right, 52.2% of people right; the
  teacher's answer is reached at least once on 117 of the 1,712 puzzles (6.8%). So the reward of B2–B4 is zero for
  about 57% of replies by format alone, and the rest are near chance.
- **D4, U_inst against B5 (test, seed 123):** both right 147, U_inst only 384, B5 only 80, neither 389 (by size, 7
  people: 78 / 219 / 43 / 160; 8 people: 69 / 165 / 37 / 229). U_inst is right on 70.6% of the 751 replies it finishes
  and on 0.4% of the 249 cut at 8,192 tokens. B5's wrong answers label 60.8% of the people right.
- **D5, B1's shortcut is mostly surface statistics.** A logistic regression on 31 surface counts per person (how
  often others call the person a knight or knave, under which operator and negation, the person's own statement type,
  N) labels 67.7% of the test people right and solves 7.9% of the puzzles when fit on the teacher's answers (67.6% and
  7.8% on the reference answers). B1 s123 labels 66.1% of people right and solves 14.9%: per person about the same,
  with more fully right puzzles.

**Results D2, D3 (2026-11-21 16:41–16:50; `outputs/kk_phase1/decomp/judge_s123/`, `train_preds/`).**
- **D2, the teacher as judge** (one judgment per text; VALID / INVALID / no verdict): B2 texts found without the hint
  54 / 40 / 1 of 95 (43% invalid); B2 hint-round texts 63 / 135 / 2 of 200 (**68% invalid**); calibration: correct
  teacher solutions 89 / 11 / 0 of 100 (11% invalid), wrong teacher solutions 46 / 52 / 2 of 100 (52%). The judge is
  noisy (it passes 46% of solutions that end in a wrong answer), but it rejects the hinted texts more often than even
  the teacher's wrong solutions: most of what B2 trains on is reasoning that does not support its answer.
- **D3, the teacher's errors (B1 and B5 on their own training puzzles, one try each):** B1 reproduces the teacher's
  answer on 306 of the 1,435 puzzles where it is right (21%; its test rate is 16.9%), and on the 277 where it is wrong
  gives the teacher's wrong answer 35 times, the truth 20, neither 222. B5: 465 of 1,435 (32%); on the 277, teacher
  42, truth 33, neither 202. The answers-only students hardly fit their training answers, so the teacher's wrong
  answers are not what limits them. **U_inst** (trained on all 2,000 teacher solutions, right or wrong): right on 887
  of the 1,435 puzzles where the teacher is right (62%); on the 277 where it is wrong, the teacher's wrong answer 36,
  the truth 77 (28%), neither 67, unreadable 97; right on 100 of the 288 the teacher left unanswered (35%). Trained on
  the teacher's reasoning, the student reaches the true answer more often than it copies the teacher's error, also
  where that reasoning went wrong (2026-11-21 17:59).
- **Paper:** D1, D2 and D5 carry the argument of Section 4 (the student cannot reach the answers; hinted reasoning is
  spurious; answers teach surface statistics) and go into the paper; D3 and D4 stay here.

## 27. Can the untrained student verify an answer? (feasibility check for a verify-then-solve method, 2026-11-21; go-ahead from the user)

**Why.** The candidate Phase 2 method ("verify the teacher's answer first, then learn to solve") rests on one premise:
checking a given assignment is within the untrained student's reach even though finding one is not (§26 D1: it
reaches the teacher's answer on 117 of 1,712 puzzles). If the student cannot tell right from wrong assignments
either, the method fails at its first step.

**Check** (`scripts/kk_verify_probe.py`, GPU 0, seed 123). Training puzzles with the teacher's answer. Items:
- 300 random puzzles where the teacher is right: (i) the teacher's answer; (ii) the same answer with one random
  person flipped (always wrong, since every puzzle has exactly one solution);
- all 277 puzzles where the teacher is wrong: the teacher's answer.

The untrained instruct student gets the puzzle, the assignment and an instruction to check every person's statement
against it (a knight's statement must be true, a knave's false) and to end with "VERDICT: CONSISTENT" or "VERDICT:
INCONSISTENT"; 4 replies per item, temperature 0.6, 2,048 tokens, its chat template with the fixed date line. Ground
truth: the assignment is consistent iff it equals the solution (checked by evaluating the statements).

**Read-out.** Per item type: share of replies with the right verdict, share of items whose 4-reply majority is right,
share with no verdict. Interpretation after the numbers: if the student accepts right answers and rejects wrong ones
well above chance (both types of wrong), verification is a usable first step; if it is near chance, the method needs a
different first step.

**Result (2026-11-21 16:59; `outputs/kk_phase1/verify_probe_s123/`).** Replies with a verdict: 74–76% in every item
type. Among them, the student calls the assignment CONSISTENT for 63.5% of the teacher's right answers (580 of 913),
60.6% of the one-flip answers (550 of 908) and 64.7% of the teacher's wrong answers (529 of 818). The verdict does not
depend on whether the assignment is right: **the untrained student cannot verify an assignment**; it leans toward
"consistent" and is otherwise at chance (4-reply majorities right: 57.3%, 29.7%, 19.9%). Checking an answer is not
within its reach any more than finding one is. A verify-then-solve method would have to teach verification first,
and a first step that only asks the student to judge answers gives no usable signal without training. Not in the
paper (a development check).

## 28. Repairing corrupted answers: does the untrained student reach the teacher's answer? (test 1, 2026-11-24; go-ahead from the user)

**Why.** Candidate Phase 2 method: train the student to repair the teacher's answer with k people deliberately
flipped, rewarded by agreement with the teacher's answer, raising k toward solving from scratch (proposal of
2026-11-21; closest work: hint annealing, GASP, reverse curriculum, see `results/lit_answer_only.md` §6). Such RL only
works if the untrained student reaches the teacher's answer often enough at small k. §27 showed it cannot verify, so it
may simply return the answer it is given.

**Test** (`scripts/kk_repair_probe.py`, GPU 0, seed 123, no training). 300 random training puzzles with a teacher
answer (target: the teacher's answer, as the method's reward). For k = 1, 2, 3, the teacher's answer with k random
people flipped, in two prompts: **open** ("this proposed answer may contain errors; check it and give the corrected
answer") and **count** ("exactly k people are labeled wrong; find them and give the corrected answer"). 8 replies per
item, temperature 1.0 (as the RL rollouts), 2,048 tokens, the student's chat template with the fixed date line; the
answer is read with the task's checker.

**Read-out per prompt and k.** Share of replies that reach the teacher's answer; puzzles reached at least once in 8
(the reward coverage); replies that return the given answer unchanged; replies changing exactly k people; unreadable
replies. Reference: a student that flips k random people reaches the target with probability 1 / C(N, k) per reply
(k = 1: 1/7 or 1/8; reached at least once in 8: about 66–71%; k = 2: about 33%; k = 3: about 21%).

**Addition before the full run (2026-11-24 13:15).** The smoke test (6 puzzles) gave 50–71% unreadable replies at
temperature 1.0 and at most 2% reaching the teacher's answer; some replies degenerate ("tennis on the other point").
The same test also runs at temperature 0.6 (the scoring temperature) on GPU 1, to separate sampling noise from
inability; the temperature-1.0 run stays the fixed one. Outputs: `repair_probe_s123` (1.0), `repair_probe_t06_s123`.

**Result (2026-11-24 13:58; `outputs/kk_phase1/repair_probe_s123/`, `repair_probe_t06_s123/`).** Per prompt and k:
share of replies reaching the teacher's answer / puzzles reached at least once in 8 / unreadable replies; random-flip
reference for puzzles reached in 8: 68.3% (k = 1), 28.8% (k = 2), 17.1% (k = 3).

| Temperature | Prompt | k = 1 | k = 2 | k = 3 |
| --- | --- | --- | --- | --- |
| 1.0 (fixed) | open | 2.1 / 13.7 / 50.5 | 1.0 / 7.7 / 52.7 | 0.8 / 6.7 / 54.2 |
| 1.0 (fixed) | count | 1.8 / 12.7 / 53.0 | 1.0 / 7.0 / 57.8 | 0.5 / 3.7 / 60.9 |
| 0.6 | open | 4.3 / 27.0 / 11.6 | 2.8 / 17.7 / 11.8 | 1.5 / 11.3 / 11.3 |
| 0.6 | count | 4.6 / 29.0 / 9.5 | 2.0 / 13.7 / 16.9 | 1.0 / 8.3 / 21.0 |

- The student rarely returns the given answer unchanged (2–8%), but it changes the wrong people: only 9–24% of replies
  change exactly k people, and it reaches the target less often than random flips of k people would.
- At temperature 1.0 half the replies are unreadable; at 0.6 a tenth to a fifth.
- Saying how many people are wrong (count) does not help.
- **Coverage against solving from scratch:** at k = 1 and temperature 0.6 the teacher's answer is reached on 27–29% of
  puzzles in 8 replies, against 6.8% when the student solves the puzzle itself (§26 D1, 8 tries at 0.8). The repair
  task gives an RL signal on about four times as many puzzles, far below the random-flip ceiling.
- Not in the paper (a development check).

## 29. Phase 2 exploration, wave 1: answer-space curricula (2026-11-24; the user asked for several ideas in parallel on up to 7 GPUs)

**Context.** The user asked to propose and try several method ideas, using GPUs 0, 1, 3–7 (GPU 2 stays free for the
user's other project), and to document ideas and checks in md files. The ideas and their rationale are in
`results/phase2_ideas.md` (I1–I5). Development on seed 123; results are interpreted after they are seen (§24 rule).

**Training sets** (`scripts/kk_task_sets.py`, `outputs/kk_phase2/sets/`; seed 123; only puzzles with a teacher
answer; target always the teacher's answer):
- `train_repair_k12`: the teacher's answer with k ∈ {1, 2} people flipped (882 / 830), open repair prompt (§28).
- `train_mask_m12`: the teacher's labels of all but m ∈ {1, 2} people given as known (882 / 830).
- `train_guessB1`: B1 s123's own answer on the training puzzle as the proposed answer (§26 D3 predictions).
- `train_mix`: half plain puzzles (889), half repair k ∈ {1, 2} (412 / 411).
- `train_repair_k1` (for I4): k = 1 for every puzzle.

**Runs** (launcher `scripts/64_phase2_wave1.sh`; students under `outputs/kk_phase2/students/`):
- **I1 repair RL, I2 infill RL, I3 guess-repair RL, I5 mixed RL**: GRPO from the untrained instruct student with B3's
  settings (§25.2: LoRA 128, Dr. GRPO, lr 5e-6, KL 1e-4, 16 × 8 replies per step, 4,096 tokens, one epoch, exact
  agreement with the teacher's answer) except **temperature 0.6** (§28: at 1.0 half the replies are unreadable).
- **I4 repair self-training**: `kk_star.py --no-hint` on `train_repair_k1` (8 tries at 0.6, 2,048 tokens, keep the
  first try reaching the teacher's answer), then B1's SFT recipe on the kept texts.
- **Baselines**: the untrained student on the three evaluation sets below.

**Evaluation** of every student (`kk_eval.py`, 1,000 test puzzles, seed 123): **solve** (the standard test);
**repair** (`test_repair_truth_k1`: the reference answer with one person flipped; diagnostic); **guess → repair**
(`test_guessB5`: B5 s123's test answers as the proposed answer; the attacker's two-step system); for I2 also
**infill** (`test_mask_truth_m1`). References: I0 0.8% (s123 1.3%), B5 s123 22.7%, U_inst s123 53.1%.

**Launched 2026-11-24 14:31** on GPUs 0, 1, 4, 5, 6, 7. GPU 3 had no job, so one sensitivity run was added before it
started: **I1 at learning rate 2e-5** (`I1_repair_lr2e5`, otherwise identical), because B3 and B4 barely moved at
5e-6 in 107 steps (§25.2); it guards against reading a too-slow learning rate as a failed idea.

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
