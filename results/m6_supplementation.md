# M6a — Epistemic supplementation attack (proposal section 4.3)

**Status: complete (2026-09-06).** Three new training runs (mix50 / mix25 / mix10), 1,800 new
problem-evaluations, compared against the three cells from the G1 replication.

Threat model: the defender serves confident, epistemic-free traces (hindsight). The attacker
supplements them with epistemic traces obtained elsewhere — LIMO is a free public download.
Proposal section 4.3 predicts this should work, because procedural and epistemic information are
claimed to be orthogonal axes.

Every condition: **800 problems, identical hyperparameters, 1,500 steps.** The only variable is
what fraction of traces retain epistemic verbalization.

---

## 1. Headline — the attack works, but is NON-MONOTONIC

Greedy pass@1, pooled over 600 problems (MATH500 + AMC23 + AIME24 + AIME25).

| condition | LIMO traces | % problems | % **tokens** | pooled | vs base | gap recovered* |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0 | — | — | 49.8% | — | — |
| **LIMO** | 800 | 100% | 100% | **62.8%** | +13.0 pp | 100% |
| **mix50** | 400 | 50% | 90.4% | **63.8%** | +14.0 pp | **116%** |
| **mix25** | 200 | 25% | 74.4% | **60.0%** | +10.2 pp | 55% |
| **mix10** | 80 | 10% | 51.1% | **52.2%** | +2.3 pp | **-68%** |
| hindsight | 0 | 0% | 0% | 56.5% | +6.7 pp | 0% |

\* fraction of the LIMO-minus-hindsight gap recovered.

**Two findings, and the second is the interesting one:**

1. **Supplementation fully defeats the defense at 50%.** mix50 reaches 63.8% — statistically
   indistinguishable from pure LIMO (62.8%; a 6-problem difference in 600) despite half its traces
   being the defender's. For the proposal this is a clean positive: the cheapest conceivable
   attack — concatenating a public dataset, no curation, no algorithm change — neutralises
   hindsight distillation.

2. **At 10% it BACKFIRES. mix10 (52.2%) is worse than using no epistemic data at all
   (hindsight, 56.5%).** Adding 80 epistemic traces made the student *worse* than adding none.
   This is not noise: it is 26 problems out of 600, and the mechanism is identifiable (section 2).

---

## 2. Why it backfires: two effects moving in opposite directions

Decomposing MATH500 (500 of the 600 problems) separates reasoning quality from answer production:

| condition | pass@1 | finished | **accuracy \| finished** | epistemic/resp |
| --- | --- | --- | --- | --- |
| base | 55.0% | 415/500 | 66.0% | 0.1 |
| LIMO | 69.0% | 441/500 | **78.2%** | 228.7 |
| mix50 | 71.2% | 484/500 | **73.6%** | 95.0 |
| mix25 | 66.4% | 485/500 | **68.5%** | 54.0 |
| mix10 | 58.6% | **455/500** | **64.4%** | 29.6 |
| hindsight | 64.2% | **499/500** | 64.3% | 9.9 |

**Effect A — reasoning quality is perfectly monotonic in epistemic fraction.**
`accuracy | finished`: 78.2 -> 73.6 -> 68.5 -> 64.4 ~= 64.3 (hindsight). More epistemic content,
better reasoning, no exceptions. This is exactly Kim et al.'s thesis.

> **QUALIFIED 2026-09-15 by M6c.** The mechanism below holds *within this mixture series*, but it
> is **not general**. Across the six reconstruction runs of `results/m6c_weaker_attacker.md`, the
> correlation between no-stop token share and MATH500 finished count is **r = +0.637 — the wrong
> sign**: `7B solo` carries 36.2% no-stop tokens and finishes 477/500, while `32B solo` carries
> 22.1% and finishes 465/500. Across those runs no-stop share is largely a proxy for trace length,
> and length helps. Read what follows as a demonstrated mechanism for these mixtures, not as a law.

**Effect B — termination is damaged by the epistemic traces, and it is token-weighted.**
LIMO traces are long; at `cutoff_len: 16384` the longest lose their stop token entirely
(`results/m2_limo.md` section 4). Measured over each training set:

| dataset | examples truncated | **% of trained TOKENS with no stop signal** |
| --- | --- | --- |
| LIMO | 32.0% | **45.4%** |
| mix50 | 15.9% | **41.4%** |
| mix25 | 7.9% | **34.5%** |
| mix10 | 4.6% | **33.0%** |
| hindsight | 0.9% | **11.3%** |

**mix10 is the worst of both worlds.** Its 80 LIMO traces are the *longest* in the pool, so
although they are only 4.6% of examples they carry **33.0% of trained tokens with no stop
signal — nearly three times hindsight's rate** — while delivering barely more epistemic content than
hindsight (29.6 vs 9.9 per response). It inherits the non-termination pathology and earns **no**
reasoning benefit at all: `acc|finished` 64.4% against hindsight's 64.3% — a 0.1 pp difference,
well inside noise — but only 455/500 finished against hindsight's 499/500.

**The exact trade.** pass@1 factors as (fraction finished) x (accuracy given finished), so with
hindsight as reference the identity `dpass = F_x(A_x - A_h) + A_h(F_x - F_h)` splits each result
into a reasoning gain and a termination cost:

| MATH500 | finished | vs hind | `acc\|fin` | reasoning gain | termination cost | net |
| --- | --- | --- | --- | --- | --- | --- |
| hindsight | 499/500 | — | 64.3% | — | — | 64.2% |
| **mix10** | 455 | **-44** | 64.4% | **+0.1 pp** | **-5.7 pp** | **-5.6 pp** |
| mix25 | 485 | -14 | 68.5% | +4.0 pp | -1.8 pp | +2.2 pp |
| mix50 | 484 | -15 | 73.6% | +8.9 pp | -1.9 pp | +7.0 pp |
| LIMO | 441 | **-58** | 78.2% | +12.3 pp | **-7.5 pp** | +4.8 pp |

The decomposition is exact — the two components sum to the net in every row. Every mixture loses
finished-problems against hindsight; the reasoning gain must cover that loss. mix50 gives up 15
problems and earns 8.9 pp. **mix10 gives up 44 and earns 0.1 pp.**

It also explains why pure LIMO (69.0%) scores *below* mix50 (71.2%) on MATH500: LIMO pays the
largest termination tax of all (-7.5 pp). **A 50/50 attacker does not merely match the undefended
teacher's data — it beats it.**

At 25% and above, Effect A outgrows Effect B and the attack pays off. The crossover therefore lies
between 10% and 25% of problems — three points cannot pin it down further.

---

## 3. The token-vs-problem distinction matters enormously

Flagged as a risk *before* the first run, and it proved decisive:

| condition | % of PROBLEMS | % of TOKENS |
| --- | --- | --- |
| mix50 | 50% | **90.4%** |
| mix25 | 25% | **74.4%** |
| mix10 | 10% | **51.1%** |

LIMO traces average 11,450 trained tokens against hindsight's 1,260 — a 9x ratio. So **half the
problems is ninety percent of the tokens.** Reporting mix50 as "50% epistemic data fully recovers
performance" would be badly misleading; the honest statement is "90% of epistemic *tokens*
recovers performance, and 51% of tokens actively hurts."

Any future work on this axis should report both, and preferably control the token fraction
directly.

---

## 4. What this means for the proposal

**For the attack (section 4.3):** naive mixing works, but **has a threshold and is not safe to
apply blindly**. An attacker who obtains only a small quantity of epistemic traces is better off
not using them. The proposal's framing — "the student would learn domain knowledge from the
defended traces and *when to doubt itself* from the epistemic supplement" — holds only above the
threshold; below it, the supplement's side effects dominate.

**For the defense:** this is an exploitable asymmetry. A defender who can constrain how much
epistemic data an attacker obtains does not merely blunt the attack — at 10% of problems (51% of
tokens) the attacker ends up *worse than not attacking*. The threshold lies somewhere between 10%
and 25%; this sweep does not locate it more precisely. That is a stronger defensive position than
the proposal's taxonomy currently contemplates.

**For `Score(trace, student)` (section 4.1):** this is direct evidence that
`epistemic_density` must not be a simple monotone factor. A scoring function that just
up-weights epistemic traces would happily assemble a mix10-like pool and degrade the student.
The interaction with trace length (and hence truncation) has to be represented.

---

## 5. Limitation — confounded with the `cutoff_len` artifact

Effect B exists **because** `cutoff_len: 16384` strips stop tokens from long traces. That is
inherited from LIMO's default config and is present in Kim et al.'s setup too, but it means the
non-monotonicity measured here is partly an artifact of the training recipe rather than a pure
property of epistemic supplementation.

**The clean version of this experiment** raises `cutoff_len` above the longest trace and re-runs
the sweep. If the non-monotonicity disappears, Effect B is purely an artifact; if it survives,
epistemic traces carry an intrinsic cost. This is the single most valuable follow-up.

`scripts/32_trace_budget.py --cutoff-len 40960` shows the control is well-conditioned:

| | `cutoff_len 16384` (this study) | `cutoff_len 40960` (control) |
| --- | --- | --- |
| LIMO examples truncated | 256/800 (32.0%) | **0/800** |
| mix10 examples truncated | 37/800 (4.6%) | **0/800** |
| LIMO trained tokens | 9.16 M | 10.40 M (**+13%**) |
| mix50 / mix25 / mix10 epistemic token share | 90.4 / 74.4 / 51.1% | 90.8 / 74.6 / 52.9% |

Truncation goes to **exactly zero** for every dataset, the token budget grows only 13%, and the
epistemic token fractions barely move — so the control isolates Effect B while holding the
independent variable fixed. That is as clean as this comparison can be made.

**Cost caveat:** 40,960 is 2.5x the sequence length, and activation memory scales with it. The
16k runs already needed Liger fused cross-entropy to fit 46 GB (`results/deviations.md`), so the
control will likely need gradient checkpointing, more GPUs, or ZeRO-3 CPU offload. Budget for a
memory-tuning pass before assuming it is a drop-in config change.

---

## 6. Reproduction

```bash
for f in 0.50 0.25 0.10; do
  pct=$(python3 -c "print(int(float($f)*100))")
  .venv-infer/bin/python scripts/50_make_mixed_dataset.py --epistemic-fraction $f \
      --output data/curated/limo_mixed_${pct}.json
done
# two runs concurrently: separate NUMA nodes, no measured contention (16.1 vs 17.2 s/it alone)
GPUS=0,1,2,3 scripts/20_train.sh configs/train/qwen2.5-7b_mixed25.yaml &
GPUS=4,5,6,7 scripts/20_train.sh configs/train/qwen2.5-7b_mixed10.yaml &
scripts/15_mixsweep_eval.sh
.venv-infer/bin/python scripts/30_collect_results.py      # the pass@1 table
.venv-infer/bin/python scripts/32_trace_budget.py         # every token/truncation figure above
```

Training: mix50 7:29 (alone), mix25 7:14 and mix10 6:42 (concurrent).
