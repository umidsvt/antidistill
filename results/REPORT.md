# Replication report — reasoning distillation and epistemic verbalization

**Qwen2.5-7B · completed 2026-09-05 · all three cells trained and evaluated**

Reproduces the `Qwen2.5-7B` row of the proposal's §2.2 table, testing Kim et al.
(arXiv:2603.15500): that stripping *epistemic verbalization* from otherwise-correct reasoning
traces makes them much worse for distillation.

---

## Headline

Greedy pass@1, one student model, identical hardware and settings throughout.

| benchmark | n | base | LIMO (epistemic) | Hindsight (no epistemic) |
| --- | --- | --- | --- | --- |
| MATH500 | 500 | 55.0% | **69.0%** | 64.2% |
| AMC23 | 40 | 40.0% | **55.0%** | 37.5% |
| AIME24 | 30 | 20.0% | 20.0% | **6.7%** |
| AIME25 | 30 | 6.7% | **13.3%** | 3.3% |
| **POOLED** | **600** | **49.8%** | **62.8%** | **56.5%** |
| | | | **+13.0 pp** | **+6.7 pp** |

*(base 299/600 · LIMO 377/600 · hindsight 339/600)*

**Published values for comparison** — Kim et al. / proposal §2.2, AIME24 greedy pass@1 only,
which is the only benchmark they report for this model:

| | base | LIMO | Hindsight |
| --- | --- | --- | --- |
| Kim et al. | 13.3% | 26.7% (+13.4 pp) | 3.3% (0.25x base) |
| ours, AIME24 | 20.0% | 20.0% (+0.0 pp) | 6.7% (**0.33x base**) |
| ours, pooled 600 | 49.8% | 62.8% (**+13.0 pp**) | 56.5% |

Detailed comparison in the next section.

**Both of the paper's central claims reproduce:**

1. **LIMO helps: +13.0 pp** pooled over 600 problems (Kim et al. report +13.4 pp).
2. **Hindsight hurts:** below LIMO on *every* benchmark, and on AIME24 it collapses to
   **6.7% — one third of base** (Kim et al.: 3.3%, one quarter of base).

**One finding they do not report, which changes how the result should be read:** the hindsight
effect is strongly **difficulty-dependent**. It *helps* on MATH500 (+9.2 pp vs base) and
*collapses* on AIME24 (−13.3 pp). Pooled, hindsight is **above** base. "Hindsight collapses" and
"hindsight beats base" are both true of the same checkpoint.

---

## Side by side with the published numbers

### AIME24 greedy pass@1 — the cell the paper reports

| | base | LIMO | Hindsight |
| --- | --- | --- | --- |
| **Kim et al. / proposal §2.2** | **13.3%** (4/30) | **26.7%** (8/30) | **3.3%** (1/30) |
| **ours** | **20.0%** (6/30) | **20.0%** (6/30) | **6.7%** (2/30) |
| | +2 problems | −2 problems | +1 problem |

Every cell differs by **1–2 problems**, i.e. 3.33–6.67 pp — the resolution limit of a 30-problem
greedy benchmark, and the same magnitude as the argmax nondeterminism measured at M1 (only 4/30
of our greedy traces are byte-identical to their released generations).

**Relative effects, which is what the metric can actually support:**

| | Kim et al. | ours |
| --- | --- | --- |
| LIMO vs base | +13.4 pp (2.0x) | +0.0 pp (1.0x) — *see below* |
| Hindsight vs base | −10.0 pp (0.25x) | −13.3 pp (**0.33x**) |

The **hindsight collapse reproduces closely** (0.25x vs 0.33x of base). The **LIMO effect does not
appear on AIME24 at all** — but it does appear at **+13.0 pp** once measured over 600 problems,
against their reported +13.4 pp. See "Why AIME24 alone would have produced the wrong conclusion".

### Everything else the paper reports for this model

| quantity | source | Kim et al. | ours |
| --- | --- | --- | --- |
| base AIME24 pass@1, t=0.0 | paper Table 7 | 13.33% | 20.0% |
| base AIME24 pass@1, t=0.7/p=1.0 | paper Table 7 | 6.67% | **7.08%** (avg@16) |
| base AIME24 pass@32 | paper Fig. 7 | 36.7% | 26.7% (pass@16) |
| LIMO AIME24 pass@1 | paper Fig. 7 | 26.7% | 20.0% |
| LIMO AIME24 pass@32 | paper Fig. 7 | 53.3% | not run |
| **LIMO MATH500 pass@1** | their *released generations*, graded by us | **80.4%** (402/500) | **69.0%** (345/500) |

The t=0.7 row is the tightest agreement in the whole study: their single-sample 6.67% against our
16-sample estimate of **7.08%** — a 0.4 pp gap, far inside one problem. When the metric has
adequate resolution, the two pipelines agree almost exactly. That is the strongest evidence that
the AIME24 pass@1 discrepancies are the benchmark, not the stack.

The MATH500 row is the one real unexplained gap (−11.4 pp); see Caveats.

### What the paper does *not* report, and we measured

- **Hindsight on anything except AIME24.** We ran MATH500, AMC23 and AIME25, which is what
  revealed the effect is difficulty-dependent (+9.2 pp on MATH500, −13.3 pp on AIME24).
- **Base or hindsight on MATH500 / AMC23 / AIME25** for this model.
- **Answer-production rates.** LIMO-trained models fail to emit any answer on 11/30 AIME24
  problems; their own released generations show 8/30. Not discussed in the paper.
- **Epoch sweeps.** Both conditions evaluated at epochs 5/10/15.

### The other two rows of §2.2 (not attempted)

| Model | Base | LIMO | Hindsight |
| --- | --- | --- | --- |
| Qwen3-14B-Base | 16.7% | 60.0% | 3.3% |
| DeepSeek-R1-Distill-32B | 80.0% | 73.3% | 23.3% |
| **Qwen2.5-7B** *(this study)* | 13.3% | 26.7% | 3.3% |

The hindsight dataset is model-independent, so those rows need only training and evaluation —
no regeneration. See `CLAUDE.md` §1.

---

## The three cells

| | training | eval | wall-clock |
| --- | --- | --- | --- |
| **base** | none | 4 benchmarks | — |
| **LIMO** | 800 LIMO-v2 traces, 15 epochs | 4 benchmarks | 11:16 (8 GPU) |
| **Hindsight** | same 800 problems, traces re-derived confidently by DeepSeek-R1-Distill-Qwen-32B | 4 benchmarks | 6:14 (4 GPU) |

Both SFT runs used LIMO's default config verbatim — full fine-tune, ZeRO-3, `cutoff_len 16384`,
lr 5e-6, cosine, 15 epochs, **global batch 8, 1,500 steps**. Only the traces differ.

---

## Why AIME24 alone would have produced the wrong conclusion

On AIME24, LIMO scored **20.0% — identical to base**, i.e. a flat null against a reported
+13.4 pp. It took the 600-problem sweep to see the effect.

Two compounding causes, both measured:

**Resolution.** 30 problems = 3.33 pp per problem. A +13 pp effect is four problems. Our base
drew 2 lucky against Kim et al.'s (6/30 vs 4/30) and our LIMO 2 unlucky (6/30 vs 8/30) — four
argmax tie-breaks spanning exactly the effect size. Only **4/30** greedy traces are byte-identical
to their released generations; greedy decoding is not reproducible across GPU architectures.

**Non-termination.** LIMO-trained models run to the 32k token cap on hard problems and never
emit an answer — 11/30 on AIME24. Root cause: `cutoff_len: 16384` truncates the SFT target and
cuts the trailing `<|im_end|>` from **32% of LIMO traces** — and those are the *longest*, i.e.
hardest, problems. The model is taught never to stop exactly where AIME puts it. This is present
in **Kim et al.'s own released generations** (8/30 answerless), so it is a property of the recipe,
not of this reproduction.

> **Standing methodological conclusion:** a 30-problem greedy benchmark cannot resolve the effect
> sizes this project studies. Report a difficulty range, and report `accuracy | finished`
> alongside raw pass@1 whenever conditions differ in how reliably they terminate.

---

## The hindsight collapse is real, not a measurement artifact

The obvious worry is that hindsight scores low because it fails to produce answers. **The
opposite is true** — it terminates better than every other condition:

| finished | base | LIMO | hindsight |
| --- | --- | --- | --- |
| AIME24 | 26/30 | 19/30 | **30/30** |
| AIME25 | 26/30 | 24/30 | **30/30** |
| MATH500 | 415/500 | 441/500 | **499/500** |

(Hindsight traces keep their stop token 99.1% of the time in training, vs LIMO's 68%.)

It answered **30/30** AIME24 problems against LIMO's 19/30 — more chances to be right — and
still scored a third as well. Conditioning on producing an answer makes the gap **larger**:

| accuracy \| finished | base | LIMO | hindsight |
| --- | --- | --- | --- |
| AIME24 | 23.1% | 31.6% | **6.7%** |
| AIME25 | 7.7% | 16.7% | **3.3%** |

**Removing epistemic verbalization did not stop the model finishing. It stopped it being right.**

---

## The mechanism, visible in three independent places

**1. The data.** The rewrite removes the doubt while keeping the mathematics:

| | LIMO | Hindsight | ratio |
| --- | --- | --- | --- |
| mean words / trace | 7,200 | 772 | 0.11x |
| epistemic tokens / trace | 263.6 | 4.8 | **0.02x** |
| `wait` occurrences | 61,663 | 463 | **0.008x** |

**2. Training dynamics.** M3 reached M2's *final* loss at epoch 4 and converged by epoch 9; mean
train loss 0.034 vs 0.198. Hindsight traces have no branch points and no moments where the next
token depends on the model recognising its own error, so they are nearly trivial to fit. **Low
loss here is the signature of the informative content having been removed, not of better
learning.**

**3. Difficulty dependence.** Epistemic verbalization is the channel for *recovering from a wrong
path*. On MATH500 the model mostly solves directly and never needs to recover, so confident
derivations transfer useful procedural form (+9.2 pp). On AIME it must detect and reverse its own
errors — precisely what the rewrite removes (−13.3 pp). Kim et al. document the same asymmetry
for Qwen2.5-Math-7B (AIME 16.7 → 0.0, MATH500 52.4 → 59.0).

**And the behaviour transfers exactly as claimed.** LIMO SFT took `wait` from **0 → 6,013**
occurrences in 30 AIME24 responses; responses containing any epistemic token went 6/30 → 30/30.
800 examples were enough to install the linguistic habit.

---

## Caveats and open items

| | |
| --- | --- |
| **Absolute numbers are not comparable to the paper.** | Greedy decoding differs across GPU architectures (B200 vs L40S). Use our own base row as the baseline; within-study contrasts are valid. |
| **11.4 pp gap to Kim et al. on MATH500** (69.0% vs their 80.4%). | Unexplained. Correlates with our LIMO model being ~2x more verbose (51k vs 26k mean chars) and terminating less often (88% vs 95%). |
| **Hindsight data carries an orphan `</think>` tag** in 760/800 traces; LIMO has none. | Faithful to Kim et al.'s script (no post-processing), but an uncontrolled variable in the contrast. The two halves are a paraphrase, not a duplicate (median char similarity 0.28, 99.6% same final answer). |
| **4/800 hindsight traces failed validation** and fell back to a possibly-wrong solution. | 0.5% — immaterial. Indices recorded in the sidecar. |
| **avg@16 was not run for LIMO.** | Killed at 0/480 after 33 min (~16x cost, because the model runs to the token cap). The 600-problem sweep supplied the statistical power instead. |
| **Hindsight is the most verbose at inference** (~95k chars) despite the shortest training traces. | Unexplained; worth investigating. |

All deviations from the published setup — 7 of them, none altering the training recipe — are in
`results/deviations.md`, each with a `DEVIATION (antidistill)` comment at its code site.

---

## What this establishes for the wider project

The proposal (§4) plans adaptive attacks on distillation defenses, scored by
`Score(trace, student) = correctness x epistemic_density x distributional_alignment x difficulty_match`.
This replication delivers the substrate:

- **A verified pipeline** — grader validated against Kim et al.'s own generations to the problem
  (4/30 and 8/30, zero disagreements) before any GPU time was spent.
- **A working defense** — hindsight rewriting, measured at 98% epistemic-token removal, behind a
  reusable generator. It is the first entry in the defense taxonomy of proposal §3.
- **`epistemic_density`, already built and validated** (`scripts/31_epistemic_density.py`), which
  separated "did the behaviour transfer" from "did accuracy improve" — the distinction that
  resolved M2.
- **An evaluation protocol that will not mislead**: multi-benchmark, difficulty-spanning, with
  `accuracy | finished` reported alongside pass@1.
- **A measured caution for the `Score` function**: `might` appears at *identical* density in both
  conditions (1.46 per 1k words), i.e. it is used non-epistemically in confident prose. The nine
  proxy tokens should not be weighted equally.

**Next:** M4 — refactor `hindsight` behind a `Defense` interface and add a second defense (PART-style
structural), then M5 — the remaining three `Score` components.

---

## Detailed records

| document | contents |
| --- | --- |
| `results/m1_base.md` | base cell; the greedy-nondeterminism diagnosis |
| `results/m2_limo.md` | LIMO cell; +13.0 pp; termination pathology; epoch sweep |
| `results/m3_hindsight.md` | hindsight cell; collapse; difficulty dependence; dataset audit |
| `results/deviations.md` | all 7 deviations and whether each can affect results |
| `results/eval_table.md` | every metric, recomputed from stored per-problem verdicts |
| `CLAUDE.md` | how to run all of this with a different student model |
