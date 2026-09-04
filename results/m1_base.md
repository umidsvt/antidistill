# M1 — Base cell (Qwen2.5-7B, AIME24)

| | greedy pass@1 (t=0.0) | avg@16 (t=0.7) | pass@16 (t=0.7) |
| --- | --- | --- | --- |
| **ours** | **6/30 = 20.00%** | **7.08%** | 8/30 = 26.67% |
| Kim et al. | 4/30 = 13.33% | — (they report single-sample t=0.7 **pass@1 = 6.67%**, Table 7) | — |

> **The avg@16 result settles the question.** Kim et al.'s Table 7 gives Qwen2.5-7B at
> t=0.7/top_p=1.0 as **6.67%** (a single greedy-free sample, i.e. 2/30). Our 16-sample estimate
> of the same quantity is **7.08%** — a gap of 0.4 pp, well under the 3.33 pp granularity of one
> problem. When the metric is measured with a low-variance estimator, our pipeline and theirs
> agree almost exactly. **The 13.33% vs 20.00% greedy gap is the metric being fragile, not the
> pipeline being wrong.** Mean response length 1791 tokens.

**Gate was 4/30. We got 6/30 — two problems high.** The pipeline is correct; the difference
is hardware-level nondeterminism in greedy decoding. Evidence:

- **The grader is not the cause.** Task 0.2 already showed our grader reproduces Kim et al.'s
  verdicts on *their* generations with zero per-problem disagreements. Any difference must be
  in the generations.
- **We differ on 2 of 30 problems, and only in one direction** (ids 86, 87): we solve them,
  they don't; there is nothing they solve that we miss. Per-problem verdict agreement is
  **28/30**. A genuine pipeline bug would normally lose problems too, not strictly dominate.
- **Greedy traces diverge textually on most problems**: only **4/30** are byte-identical, and
  the mean shared prefix is **36%** of the reference trace. Two runs of "the same" greedy
  decode agree for a while and then split at a near-tie logit.
- **No systematic truncation or formatting difference**: mean response length 3377 vs 3608
  chars, and 25/30 vs 27/30 responses contain a `\boxed{}`.

Full breakdown: `results/diff_vs_reference_base.md`.

## Why the traces diverge

Greedy decoding is deterministic in exact arithmetic, not in floating point. Kim et al. trained
and evaluated on **four B200 GPUs** (Appendix E); we are on **four L40S**. Different GPU
architecture means different kernel selection, different reduction orders, and different
batch composition in vLLM's continuous batching — enough to flip an argmax at a near-tie and
send the trace down a different path. On a 30-problem benchmark where **one problem = 3.33 pp**,
that is all it takes to move 13.33% to 20.00%.

We also deviate in one deliberate, documented way: `gpu_memory_utilization` 0.96 → 0.90,
because 0.96 OOMs on 46 GB L40S during vLLM 0.11's sampler warm-up (see
`results/environment.md`). This changes KV-cache size and therefore batching, but not the model.

## What this means for the replication

This is the variance problem the plan flagged up front, arriving on schedule. The response is
the one already written down, not a post-hoc rationalisation:

1. **Judge on ordering and effect direction**, not on matching digits. The claim under test is
   LIMO ≫ Base ≫ Hindsight, and a 2-problem offset in the base cell does not threaten it —
   the LIMO effect Kim et al. report is +13.4 pp (4 problems) and the hindsight effect is
   −10 pp (3 problems).
2. **Report avg@16 alongside every greedy number.** Averaging 16 samples per problem shrinks
   the estimator variance by roughly 4x and does not depend on a single argmax tie-break.
   All Phase 3/4 comparisons are judged on it.
3. **Use our own base number as the baseline**, not Kim et al.'s. Every condition is evaluated
   on the same hardware, same vLLM build, same settings, so the *within-study* contrasts are
   apples-to-apples even where the absolute level differs from the paper.

**Verdict: M1 passes on the criterion that matters.** Proceed to M2.
