# GSM8K — results

**Status: complete (2026-09-17).** Nine conditions, 4,500 problem-evaluations.
Design and rationale: `results/gsm8k_proposal.md`.

**Read the setup notes before the numbers.**

- **Grading: last-number fallback, not boxed-only.** The vendored `extract_answer` never honours
  its own fallback (`results/deviations.md` §8). On GSM8K that is disqualifying: base answers
  277/500 problems in prose and scores 37.2% boxed-only against 72.2% with the fallback, while every
  fine-tuned condition boxes reliably. Every GSM8K number here is fallback-graded unless marked.
- **TP=1.** Only GPUs 6–7 were free, and greedy decoding is not bitwise-stable across
  tensor-parallel sizes, so all nine GSM8K rows use TP=1 for internal consistency. They are **not**
  token-for-token comparable with the MATH500/AMC/AIME rows, which used TP=4.
- **A fixed 500-problem subsample** of the 1,319 test set (seed 0).
- **Contamination.** GSM8K is old and widely present in pretraining corpora; the base row is very
  likely inflated by memorisation.

---

## 1. Results

| condition | attacker | epi /1kw | boxed-only | **fallback** | vs base | vs LIMO |
| --- | --- | --- | --- | --- | --- | --- |
| base (untrained) | — | — | 37.2% | **72.2%** | — | — |
| **defended v2** | — | 0.02 | 81.0% | **81.2%** | **+9.0 pp** | −4.0 pp |
| LIMO (undefended ceiling) | — | 35.57 | 80.4% | **85.2%** | +13.0 pp | — |
| A1 style | 32B | 13.71 | 82.8% | 84.2% | +12.0 pp | −1.0 pp |
| A2 search | 32B | 16.49 | 80.8% | 83.2% | +11.0 pp | −2.0 pp |
| A3 solo | 32B | 24.18 | 81.2% | 83.6% | +11.4 pp | −1.6 pp |
| A1 style | 7B | 4.66 | 84.4% | 85.6% | +13.4 pp | +0.4 pp |
| **A2 search** | **7B** | 18.82 | 87.2% | **88.4%** | **+16.2 pp** | **+3.2 pp** |
| **A3 solo** | **7B** | 25.99 | 87.0% | **88.6%** | **+16.4 pp** | **+3.4 pp** |

The ceiling risk flagged in the design did not materialise: base sits at 72.2%, leaving 28 pp of
headroom, and the effects resolve.

---

## 2. The defense HELPS on easy problems — the difficulty gradient crosses zero

**Defended v2 beats base by +9.0 pp on GSM8K.** Against the corrected (fallback-graded) numbers on
the other benchmarks:

| benchmark (easiest → hardest) | defense vs base | defense vs LIMO |
| --- | --- | --- |
| **GSM8K** | **+9.0 pp** | **−4.0 pp** |
| MATH500 | −5.4 pp | −15.6 pp |
| AMC23 | −15.0 pp | −27.5 pp |
| AIME24 | −16.7 pp | −16.7 pp |

This restores the difficulty-dependence claim that had to be retracted on 2026-09-15. The original
evidence — **+1.8 pp on MATH500** — was a grading artifact, and corrected it became −5.4 pp. But the
phenomenon was real: **confident procedural traces do help on problems the student can solve
directly; the sign change sits below MATH500, not at it.** This is the prediction
`gsm8k_proposal.md` §3.1 set out to test, and it came out the right way.

The defense's cost *relative to LIMO* shrinks the same way. **Hindsight distillation is nearly free
on easy problems and ruinous on hard ones.**

(AIME25, −3.3 pp, is out of order because base is only 6.7% there — a floor effect with little left
to lose.)

---

## 3. Epistemic density stops predicting the outcome on easy problems

| six reconstruction runs | r(epistemic density, accuracy) |
| --- | --- |
| hard benchmarks (pooled 600) | **+0.940** |
| **GSM8K** | **+0.255** |

On the hard suite, density is the strongest predictor in the project. **On GSM8K it largely stops
predicting.** That is what the error-recovery account predicts: epistemic verbalization is the
channel for *detecting and reversing a wrong path*, which matters when the problem is hard and
barely at all when it can be solved directly.

**What this does NOT do: it does not break the density/length confound.** Density and length remain
collinear across these same six pools, so *both* axes stop predicting on GSM8K together. The result
constrains *where* the axis matters, not *which* of the two is doing the work. Post-hoc stripping
(`results/decoupling_design.md`) is still required for that.

---

## 4. On easy problems, WHO generated the traces matters more than HOW

| attacker | GSM8K mean | spread across its three modes |
| --- | --- | --- |
| **7B** | **87.5%** | 3.0 pp |
| 32B | 83.7% | 1.0 pp |

The 7B attacker beats the 32B by **+3.9 pp**, while the three prompt modes within each attacker sit
within 1–3 pp of each other. On the hard suite the mode (and its epistemic density) was the dominant
factor; on GSM8K it is the attacker.

This has a notable consequence: **the 32B attacks fall *below* the undefended LIMO ceiling on GSM8K**
(83.2–84.2% vs 85.2%), while all three **7B attacks exceed it**. On the hard suite, every attack
except 7B `style` exceeded the ceiling. So "the reconstruction attack beats undefended distillation"
is **benchmark-dependent for the 32B attacker**.

It also reverses 7B `style`'s standing: worst attack on the hard suite (60.8%, below the ceiling),
but above both LIMO and all three 32B attacks on GSM8K (85.6%). Its traces are the shortest and least
doubtful of any attack (1,658 tokens, 4.66/1k words) — exactly the profile §2 says suits easy
problems.

---

## 5. The control replicates

| attacker | `solo` | `search` | gap |
| --- | --- | --- | --- |
| 32B | 83.6% | 83.2% | +0.4 pp |
| 7B | 88.6% | 88.4% | +0.2 pp |

`solo` ≥ `search` on GSM8K for both attackers, as on the hard suite. Both gaps are well inside noise,
so the statement is `solo` ≈ `search` — which still satisfies the pre-registered criterion: **the
attacker gains nothing from the defended traces**, now across five benchmarks and two attacker
capabilities.

---

## 6. Caveats

| | |
| --- | --- |
| **Fallback grading is a deviation from Kim et al.'s harness**, forced by the benchmark. It is the correct measure of what the models do, but GSM8K numbers are not comparable with anything scored boxed-only. |
| **TP=1 vs TP=4.** Internally consistent across the nine GSM8K rows; not token-for-token comparable with the other benchmarks. |
| **Contamination** likely inflates base and compresses measured effects. |
| **§3 does not resolve the confound** — it shows the density/length axis matters only on hard problems, not which of the two drives it. |
| **n = 6 for the correlations**, as in `m6c_weaker_attacker.md`. |

---

## 7. Reproduction

```bash
scripts/02_fetch_gsm8k.py                      # 500-problem subsample, seed 0
GPUS=6 bash scripts/10_eval.sh Qwen/Qwen2.5-7B base_gsm8k gsm8k 0.0 1
scripts/20_gsm8k_eval.sh                       # 8 conditions, two streams on GPUs 6/7
.venv-infer/bin/python scripts/33_regrade_fallback.py --bench gsm8k
```

~2 days wall-clock on two GPUs: individual trained conditions took up to ~11 h, because students
trained on long reasoning traces over-reason on short word problems. Base alone took ~55 min.
