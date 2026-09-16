# Adding GSM8K — design

**Status: proposed 2026-09-15, not yet run.**

Prompted by **Allouah, Haghifam, Koyejo, Shokri — "The Distillation Game: Adaptive Attacks &
Efficient Defenses"** (arXiv:2605.22737), which evaluates distillation defenses on GSM8K and MATH.

---

## 1. Why this paper matters to us beyond the benchmark

Its core claim is **our Part 4/5 finding, reached independently**: *"adaptive students recover
substantially more capability than passive evaluation suggests."* Their numbers:

| defense | passive eval | adaptive eval |
| --- | --- | --- |
| ADS | 34% | **52%** |
| PoE | 39% | **49%** |

Three further overlaps worth noting before designing anything:

- **Their teacher is `DeepSeek-R1-Distill-Qwen-7B`** — the exact model we just used as the
  *attacker* in M6c. Same model, opposite side of the game.
- **Their adaptive attack is a `Score()` function.** It reweights training examples by gradient
  alignment, `v_grad(x,y) = -<grad L(theta_0), grad log pi_stu(y|x)>`, then exponentially
  normalises within each minibatch. That is a concrete instantiation of **proposal §4.1's
  `distributional_alignment` term**, which we have not built. We already vendored
  `_kim_score_samples.py` (per-token logprobs under a student) as its intended basis.
- **Their student is `Llama-3.2-3B`; ours is `Qwen2.5-7B`.** This is what drives the ceiling
  problem in §4.

---

## 2. What GSM8K is actually for here

**Evaluation only. Never training.** Allouah et al. use it as an eval benchmark, and our training
set must stay fixed at LIMO's 800 problems — "same problems, only the traces differ" is the entire
controlled-comparison design. Adding GSM8K to training would invalidate every existing number.

**`grade_school_math/` in the harness is NOT GSM8K.** Despite the name it is 210 rows of Chinese
elementary maths (`计算：$\frac{2}{37}+...$`). GSM8K genuinely has to be added.

---

## 3. Three reasons to add it, in decreasing strength

### 3.1 It is a falsifiable test of our difficulty-dependence claim — the best reason

We measured the corrected defense's effect as **monotonic in difficulty**:

| benchmark | hindsight v2 vs base |
| --- | --- |
| MATH500 (easiest we have) | **+1.8 pp** |
| AMC23 | −12.5 pp |
| AIME24 | −16.7 pp |

The explanation on record: epistemic verbalization is the channel for *recovering from a wrong
path*; easy problems the model solves directly still benefit from confident procedural form, hard
ones do not.

GSM8K sits well **below** MATH500 in difficulty. That yields a sharp prediction:

> **The defense should help *more* on GSM8K than on MATH500 — hindsight v2 should beat base by
> more than +1.8 pp.**

If it does not, the difficulty-dependence story is wrong. This makes GSM8K a *test*, not merely
more data, and that is what justifies the GPU time.

### 3.2 Statistical power

GSM8K's test split is **1,319 problems** against our entire current suite's 600. Our most-cited
methodological finding (`results/m2_limo.md` §2) is that a 30-problem benchmark cannot resolve the
effects this project studies — we measured +0.0 pp on AIME24 for an effect that is +13.0 pp over
600 problems. GSM8K materially improves resolution.

### 3.3 Comparability with Allouah et al.

They report GSM8K accuracy for defended and attacked students. Adding it lets our numbers sit
beside theirs — with the caveat that student models differ (§4).

---

## 4. The main risk: ceiling effects

**Their student is Llama-3.2-3B (34–52% on GSM8K). Ours is Qwen2.5-7B**, which is far stronger on
grade-school word problems — plausibly 80%+ zero-shot. Two consequences:

- **Compressed range.** A +13 pp distillation effect cannot appear if base is already at 85%.
- **Their numbers are not directly comparable to ours** despite the shared benchmark. Different
  student, different headroom. Any side-by-side must say so.

There is a second-order risk that is also an opportunity. Our LIMO-trained students generate
**~51k–90k characters** per response. On GSM8K's short word problems that is extreme overkill, and
our documented termination pathology may dominate: we could end up measuring *whether the model
stops* rather than *whether it reasons*. If LIMO-trained models score **below base** on GSM8K
through over-reasoning, that is a genuine finding about distilling long-form reasoning onto easy
tasks — but it must be reported as such, with `accuracy | finished` alongside pass@1 as everywhere
else.

**Contamination.** GSM8K is old and widely present in pretraining corpora. Base Qwen2.5-7B has
very likely seen it, which inflates the base row and further compresses measured effects. State
this; do not pretend the absolute number is clean.

---

## 5. Cost, and the resulting scope decision

Measured from `logs/eval_recon7b.log`: MATH500's 500 problems take **~3 h per condition** for an
epistemic-trained model (they generate to the token cap). Per-benchmark:

| benchmark | n | wall-clock per condition |
| --- | --- | --- |
| MATH500 | 500 | ~3 h |
| AMC23 | 40 | ~16 min |
| AIME24 / AIME25 | 30 each | ~13 min |
| **GSM8K, full** | **1,319** | **~8 h** |
| **GSM8K, 500-subsample** | **500** | **~3 h** |

Nine conditions exist (base, LIMO, defended v2, and six reconstruction runs). Full GSM8K is
**~72 h**; a 500-problem subsample is **~27 h**.

**Recommendation: a fixed 500-problem random subsample, seed pinned and committed.** It matches
MATH500's weight so the pooled score is not dominated by one benchmark, costs the same, and
retains far more resolution than AMC23 + AIME24 + AIME25 combined (100 problems). Note that
subsampling forfeits reason 3.2's full benefit — if the difficulty-dependence test (3.1) comes back
ambiguous, the full 1,319 is the follow-up.

---

## 6. Implementation

The harness is simple: a JSONL with `problem` and `answer` keys, plus a prompt module.

```
third_party/kim_eval/data/gsm8k/test.jsonl        {"problem": ..., "answer": ...}
third_party/kim_eval/prompts/qwen-instruct/gsm8k.py
```

`parse_ground_truth` returns `str(example["answer"])`, and `parse_question` accepts
`question`/`problem`. GSM8K's gold answers appear after `####` and carry thousands separators
(`1,000`), which must be stripped or the grader will compare `1,000` against `1000`.

**The prompt module must be byte-identical to `math.py`** — same system prompt, no few-shot, bare
question. Any deviation makes GSM8K non-comparable with the other four benchmarks *and* breaks the
train/eval format match that every fine-tuned condition depends on.

---

## 7. What this does not address

Adding GSM8K does **not** touch the two limitations that currently matter most:

- **The density/length collinearity** (r = 0.978, `results/m6c_weaker_attacker.md` §3). A new eval
  benchmark cannot separate those; only a new *training* condition can.
- **The `cutoff_len` control.**

GSM8K is worth doing for §3.1, but it should not displace the decoupling run at the top of the
queue.
