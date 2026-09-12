# Reasoning distillation, epistemic verbalization, and attacks on the defense

**Qwen2.5-7B · replication 2026-09-05 · first attack 2026-09-06 · defense pipeline corrected
2026-09-09 · reconstruction attack 2026-09-12 · ten training runs**

**Part 1** reproduces the `Qwen2.5-7B` row of the proposal's §2.2 table, testing Kim et al.
(arXiv:2603.15500): that stripping *epistemic verbalization* from otherwise-correct reasoning
traces makes them much worse for distillation. **Both claims reproduce.**

**Part 2** runs the proposal's §4.3 *epistemic supplementation* attack against that defense.
**It defeats the defense — but only above a threshold, and below it makes the attacker worse off
than not attacking at all.**

**Part 3** reports that the defended dataset everything above was built on **was generated
incorrectly**, what we changed, and what that does to the numbers. Short version: **the corrected
defense is much stronger** — it transfers nothing at all to the student — and it lands exactly on
Kim et al.'s published figure. Part 2's conclusions are consequently scoped to the weaker dataset
until its mixtures are rebuilt.

**Part 4** runs **epistemic reconstruction**: the attacker has only defended data and must
regenerate the doubt. **All three variants beat the defense and exceed the undefended ceiling** —
and the control, an attacker using *no defended data at all*, scores highest. On the criterion
fixed before running, that makes hindsight distillation **irrelevant rather than broken**: the
attacker never needed what it protects.

> **This document is the live record.** Headline numbers, the reasoning behind each design choice,
> and every caveat that changes how a number should be read live here; per-milestone detail lives
> in the `results/*.md` files referenced from each section.

---

## Headline

Greedy pass@1, one student model, identical hardware and settings throughout. `hindsight v2` is
the corrected defense and is the number to quote; `v1` is retained because Part 2 is built on it.

| benchmark | n | base | LIMO (epistemic) | hindsight **v2** | *hindsight v1* |
| --- | --- | --- | --- | --- | --- |
| MATH500 | 500 | 55.0% | **69.0%** | 56.8% | *64.2%* |
| AMC23 | 40 | 40.0% | **55.0%** | 27.5% | *37.5%* |
| AIME24 | 30 | 20.0% | 20.0% | **3.3%** | *6.7%* |
| AIME25 | 30 | 6.7% | **13.3%** | 3.3% | *3.3%* |
| **POOLED** | **600** | **49.8%** | **62.8%** | **49.5%** | *56.5%* |
| | | | **+13.0 pp** | **−0.3 pp** | *+6.7 pp* |

*(base 299/600 · LIMO 377/600 · hindsight v2 297/600 · hindsight v1 339/600)*

**Published values for comparison** — Kim et al. / proposal §2.2, AIME24 greedy pass@1, the only
benchmark they report for this model:

| | base | LIMO | Hindsight |
| --- | --- | --- | --- |
| Kim et al. | 13.3% | 26.7% (+13.4 pp) | **3.3%** (1/30) |
| ours, AIME24 | 20.0% | 20.0% (+0.0 pp) | **3.3%** (1/30) |
| ours, pooled 600 | 49.8% | 62.8% (**+13.0 pp**) | 49.5% |

**Both of the paper's central claims reproduce:**

1. **LIMO helps: +13.0 pp** pooled over 600 problems (Kim et al. report +13.4 pp).
2. **Hindsight hurts:** below LIMO on *every* benchmark, and on AIME24 it reaches **3.3% — one
   sixth of base**, which is *exactly* the cell Kim et al. report.

**The defense does not stop the student finishing. It stops it being right.** On MATH500 the
hindsight-v2 student produces an answer on **498/500** problems — far more reliably than the
untrained base model's 415/500 — and is still correct on only 57.0% of those, against base's
66.3%. Training on defended traces made the student *worse at reasoning than no training at all*,
while making it better at stopping.

**One thing the paper does not report:** the hindsight effect is strongly **difficulty-dependent**.
Against base it costs −16.7 pp on AIME24 but only −12.5 pp on AMC23 and **+1.8 pp on MATH500**.
Easy problems the model can solve directly still benefit from confident procedural form; hard ones,
which require detecting and reversing your own errors, do not.

> Under the **v1** dataset this was much stronger — hindsight scored *above* base overall
> (56.5% vs 49.8%), so "hindsight collapses" and "hindsight beats base" were both true of the same
> checkpoint. That caveat **does not survive the pipeline correction**; see Part 3.

---

## Side by side with the published numbers

### AIME24 greedy pass@1 — the cell the paper reports

| | base | LIMO | Hindsight |
| --- | --- | --- | --- |
| **Kim et al. / proposal §2.2** | **13.3%** (4/30) | **26.7%** (8/30) | **3.3%** (1/30) |
| **ours (v2 defense)** | **20.0%** (6/30) | **20.0%** (6/30) | **3.3%** (1/30) |
| | +2 problems | −2 problems | **exact match** |
| *ours (v1 defense)* | — | — | *6.7% (2/30)* |

The hindsight cell now matches **exactly**. The base and LIMO cells differ by **1–2 problems**, i.e. 3.33–6.67 pp — the resolution limit of a 30-problem
greedy benchmark, and the same magnitude as the argmax nondeterminism measured at M1 (only 4/30
of our greedy traces are byte-identical to their released generations).

**Relative effects, which is what the metric can actually support:**

| | Kim et al. | ours (v2) | *ours (v1)* |
| --- | --- | --- | --- |
| LIMO vs base | +13.4 pp (2.0x) | +0.0 pp (1.0x) — *see below* | — |
| Hindsight vs base | −10.0 pp (0.25x) | **−16.7 pp (0.17x)** | *−13.3 pp (0.33x)* |

The **hindsight collapse reproduces**, and the corrected pipeline lands on their exact figure
(1/30 vs 1/30). If anything v2 collapses slightly *harder* than they report (0.17x vs 0.25x of
base), though at 30 problems that is a one-problem difference. The **LIMO effect does not
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
  revealed the effect is difficulty-dependent (v2: +1.8 pp on MATH500, −12.5 pp on AMC23,
  −16.7 pp on AIME24).
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

## All seven conditions

| | training | wall-clock |
| --- | --- | --- |
| **base** | none | — |
| **LIMO** | 800 LIMO-v2 traces, 15 epochs | 11:16 (8 GPU) |
| **hindsight v1** | same 800 problems, defended by Kim et al.'s procedure | 6:14 (4 GPU) |
| **hindsight v2** | same 800 problems, defended through the **corrected** teacher call (Part 3) | 6:24 (4 GPU) |
| **mix50 / mix25 / mix10** | the same 800 problems, with 50 / 25 / 10% of traces taken from LIMO and the rest from hindsight **v1** | 7:29 / 7:14 / 6:42 (4 GPU) |

Every run uses LIMO's default config verbatim — full fine-tune, ZeRO-3, `cutoff_len 16384`,
lr 5e-6, cosine, 15 epochs, **global batch 8, 1,500 steps**. Every run sees exactly 800 problems.
**Only the traces differ**, so problem coverage, step count and compute are held fixed across all
six cells and every contrast in this report is attributable to trace content alone.

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

## Part 2 — the first attack: epistemic supplementation (proposal §4.3)

The defense above works. Does the cheapest possible attack beat it?

> **Caveat added 2026-09-09:** the three mixtures were built from the **v1** defended pool, so
> everything in this section measures supplementation against the weaker defense. With v2 the
> LIMO-minus-hindsight gap widens from 6.3 pp to 13.3 pp, so every "% of gap recovered" figure
> moves. **The non-monotonic mix10 finding is the one at risk** — it was defined by mix10 (52.2%)
> falling below hindsight (56.5%), and v2 hindsight is 49.5%, already below mix10. Rebuilding the
> mixtures on v2 is the correct follow-up; until then read this section as holding for the v1
> defense specifically.

**Threat model.** The defender serves hindsight traces. The attacker mixes in epistemic traces
obtained elsewhere — LIMO is a free public download. No curation, no algorithm change, just
concatenation. Three mixtures trained, everything else held fixed (800 problems, identical
hyperparameters, 1,500 steps).

| condition | % problems epistemic | % **tokens** epistemic | pooled (600) | gap recovered* |
| --- | --- | --- | --- | --- |
| base | — | — | 49.8% | — |
| LIMO | 100% | 100% | 62.8% | 100% |
| **mix50** | 50% | 90.4% | **63.8%** | **116%** |
| **mix25** | 25% | 74.4% | **60.0%** | 55% |
| **mix10** | 10% | 51.1% | **52.2%** | **−68%** |
| hindsight | 0% | 0% | 56.5% | 0% |

\* fraction of the LIMO-minus-hindsight gap recovered.

**Finding 1 — the attack works, and it is trivial.** mix50 reaches 63.8%, statistically
indistinguishable from pure LIMO (a 6-problem difference in 600) despite half its traces being the
defender's. Hindsight rewriting does not survive an attacker who has *any* independent source of
epistemic traces.

**Finding 2 — it is non-monotonic. At 10%, the attack backfires.** mix10 (52.2%) scores **below
hindsight alone** (56.5%). Adding 80 epistemic traces made the student worse than adding none.
That is 26 problems out of 600, with an identifiable mechanism.

### The mechanism — every mixture pays a termination tax

To score a point a model must do two independent things: **reason correctly**, and **stop and
write down an answer**. A model that reasons brilliantly but runs to the 32k token cap scores
zero. So pass@1 factors exactly:

```
pass@1  =  (fraction that finished)  ×  (accuracy among those that finished)
```

Measured separately on MATH500 (n=500), those two factors behave completely differently.

**Factor 1 — reasoning quality is perfectly monotonic in epistemic fraction.** Exactly Kim et
al.'s thesis, no exceptions:

| `accuracy \| finished` | LIMO | mix50 | mix25 | mix10 | hindsight |
| --- | --- | --- | --- | --- | --- |
| | **78.2%** | 73.6% | 68.5% | 64.4% | 64.3% |

Note where mix10 lands: **64.4% against hindsight's 64.3%.** Eighty LIMO traces bought
*no measurable reasoning improvement at all*.

**Factor 2 — termination is not monotonic, and every mixture is worse than hindsight at it.**
`cutoff_len: 16384` truncates each training example and cuts off its trailing `<|im_end|>`, so
the model sees 16k tokens of text that simply never stops. LIMO traces average 11,450 tokens and
32% exceed the cutoff; hindsight traces average 1,260 and 0.9% do.

The decisive point is that **training is token-level, so this must be counted in tokens, not
examples.** For mix10: 37 of 800 examples are truncated — 4.6%, apparently negligible — but each
contributes ~16,300 tokens against the hindsight examples' ~1,260, so they supply **33.0% of all
trained tokens**. Adding 80 long traces to a short-trace base nearly doubles the token budget
(1.01 M → 1.83 M) and most of what it adds is the harmful kind.

| | LIMO | mix50 | mix25 | mix10 | hindsight |
| --- | --- | --- | --- | --- | --- |
| examples truncated | 32.0% | 15.9% | 7.9% | **4.6%** | 0.9% |
| **% of trained tokens with no stop signal** | 45.4% | 41.4% | 34.5% | **33.0%** | 11.3% |
| **finished** /500 | 441 | 484 | 485 | **455** | 499 |

**Putting them together.** Taking hindsight as the reference, the identity
`Δpass@1 = F_x(A_x − A_h) + A_h(F_x − F_h)` splits each condition's result into a reasoning gain
and a termination cost (F = finished fraction, A = accuracy given finished):

| MATH500 | finished | vs hind | `acc\|fin` | vs hind | **reasoning gain** | **termination cost** | **net** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hindsight | 499/500 | — | 64.3% | — | — | — | 64.2% |
| **mix10** | 455 | **−44** | 64.4% | **+0.1 pp** | **+0.1 pp** | **−5.7 pp** | **−5.6 pp** |
| mix25 | 485 | −14 | 68.5% | +4.2 pp | +4.0 pp | −1.8 pp | +2.2 pp |
| mix50 | 484 | −15 | 73.6% | +9.3 pp | +8.9 pp | −1.9 pp | +7.0 pp |
| LIMO | 441 | **−58** | 78.2% | +13.9 pp | +12.3 pp | **−7.5 pp** | +4.8 pp |

*(The decomposition is exact — the two components sum to the net for every row.)*

**Every mixture loses finished-problems relative to hindsight; the reasoning gain has to cover
that loss.** mix50 gives up 15 problems and earns 8.9 pp — an excellent trade. **mix10 gives up
44 problems and earns 0.1 pp.** That is the entire result: it pays the full termination tax and
receives nothing for it.

The same arithmetic explains a second oddity in the headline table: **LIMO pays the largest tax
of all (−7.5 pp, 58 problems), which is why pure LIMO (69.0%) scores *below* mix50 (71.2%) on
MATH500.** mix50 is the optimum — near-LIMO reasoning with far better termination. An attacker
mixing 50/50 does not merely match the undefended teacher's data, it **beats** it.

**Why the token/problem distinction is not a technicality.** LIMO traces average 11,450 trained
tokens against hindsight's 1,260 — a 9x ratio, so **half the problems is ninety percent of the
tokens**. "50% epistemic data recovers full performance" would be a misleading way to report
mix50. Both weightings must be stated.

**Consequences for the proposal:**

- §4.3's orthogonality assumption holds *above a threshold* that lies between 10% and 25% of
  problems (three points do not locate it more precisely). Below it the supplement's side effects
  dominate and the attacker should not attack.
- This is an exploitable asymmetry **for the defender**: constraining how much epistemic data an
  attacker obtains does not merely blunt the attack, it inverts it. Not currently in the §3
  taxonomy.
- `epistemic_density` in `Score()` **must not be a simple monotone factor** — a scorer that just
  up-weights epistemic traces would assemble a mix10-like pool and degrade the student. Its
  interaction with trace length has to be represented.

**Limitation.** Effect B exists *because* of the `cutoff_len` truncation artifact, which is
inherited from LIMO's config and present in Kim et al.'s setup too. The clean follow-up is to
raise `cutoff_len` to 40,960 and re-run the sweep — that takes truncation to **exactly zero** for
every dataset at only 13% more tokens, leaving the epistemic token fractions within 2 pp
(`scripts/32_trace_budget.py --cutoff-len 40960`). If the non-monotonicity
survives, epistemic traces carry an intrinsic cost; if not, it is a recipe artifact. One training
run per point, though 2.5x the sequence length will need a memory-tuning pass first — the 16k runs
already needed Liger fused cross-entropy to fit 46 GB. Full write-up:
`results/m6_supplementation.md`.

---

## Part 3 — the defended dataset was generated incorrectly, and fixing it strengthens the defense

### Why we looked

The hindsight-v1 student emitted `</think>` in **499/500** MATH500 responses. LIMO: 0/500. Base:
0/500. A tag that appears nowhere in the undefended data and nowhere in the base model's behaviour,
reproduced by essentially every response, is not a curiosity — it means the defended traces were
teaching something other than "solve confidently".

### What was wrong

Kim et al.'s generator calls the teacher through the **raw completions endpoint**, bypassing the
chat template, and stores the whole reply verbatim (`new_item["output"] = teacher_text`).

DeepSeek-R1's template *opens* a reasoning block for the model — its generation prompt ends
`<|Assistant|><think>\n`. Bypassing the template means no opener is ever emitted, but the model
still closes one out of habit and then writes its answer. So every defended trace contained
**two complete solutions** — the model's scratchpad and its polished answer — separated by an
unmatched `</think>`. DeepSeek's own template defines the answer as
`content.split('</think>')[-1]`; the scratchpad is what the vendor discards.

Three defects, not one:

| | v1 (Kim et al.'s procedure) | v2 (corrected) |
| --- | --- | --- |
| endpoint | `completions` — no chat markers, no BOS, no `<think>` opener | `chat/completions` with the template |
| reasoning primer | literal prefix `"Okay, so I"` | the template's `<think>\n` |
| temperature | 0.4 | **0.6** (DeepSeek recommend 0.5–0.7; below that R1 is prone to endless repetition) |

The temperature is not cosmetic: 40/800 v1 traces ran to the 32k cap without ever closing their
block, which is the documented failure mode below DeepSeek's range.

### What changed in the data

| | v1 | v2 |
| --- | --- | --- |
| traces containing `</think>` | 760/800 | **0** |
| traces with >1 `\boxed` | 761 | 63 *(LIMO itself has 479)* |
| epistemic tokens / 1k words | 0.655 | **0.024** |
| mean trained tokens | 1,260 | 486 |
| max trained tokens | 32,788 | **1,369** |
| truncated at `cutoff_len 16384` | 7/800 | **0/800** |

### What changed in the result

| hindsight | MATH500 | AMC23 | AIME24 | AIME25 | pooled | vs base |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 64.2% | 37.5% | 6.7% | 3.3% | 56.5% | +6.7 pp |
| **v2** | 56.8% | 27.5% | **3.3%** | 3.3% | **49.5%** | **−0.3 pp** |

**The corrected defense transfers nothing.** 49.5% against an untrained baseline of 49.8%, and it
lands on Kim et al.'s published AIME24 cell exactly (1/30 vs 1/30). The student is also clean:
0.0 epistemic tokens per response, 0/30 responses containing any, and **0/30 containing `</think>`**
where v1 produced 29/30.

**The training dynamics did not change** — v2's mean train loss is 0.0389 against v1's 0.0338,
both collapsing to ~0.0001 by step 900 while LIMO is still at 0.076. That the loss signature
survives the correction is evidence it belongs to the defense (traces with no branch points are
near-trivial to fit) rather than to v1's defects.

### Limitation

**Two variables moved, not one.** The artifacts went away *and* the token budget fell 2.6×
(1.01 M → 388 k), because a v1 trace carried two solutions and a v2 trace carries one. There is no
token-neutral version of this fix. The defensible claim is that a correctly-generated defended
dataset transfers nothing — not that artifact removal alone caused the −7.0 pp.

Full analysis: `results/m3b_hindsight_v2.md`. Dataset provenance and file locations, with both
versions retained: `results/hindsight_versions.md`.

---

## Part 4 — epistemic reconstruction: the defense is bypassed, not broken

Full results **`results/m6b_reconstruction.md`** · design and predictions
**`results/m6b_reconstruction_design.md`** · prompts **`results/m6b_prompts.md`**

### The question

Part 2's attack assumed epistemic traces from elsewhere. This is the harder case: **only the
defended data, so the doubt must be regenerated.** The defense destroys it at *generation* time —
the rewrite prompt orders *"State everything with full confidence"*, so even the teacher's private
scratchpad carries 0.732 epistemic tokens/1k words against LIMO's 35.571. Nothing recoverable is
hidden inside the defended artifact.

### Results

| condition | epistemic /1k words | pooled (600) | vs base | vs defended |
| --- | --- | --- | --- | --- |
| base | — | 49.8% | — | — |
| **defended v2** | 0.02 | **49.5%** | −0.3 pp | — |
| LIMO (undefended ceiling) | 35.57 | 62.8% | +13.0 pp | +13.3 pp |
| **A1 style** — fabricated doubt | 13.71 | **64.2%** | +14.3 pp | **+14.7 pp** |
| **A2 search** — real doubt + defended hint | 16.49 | **64.3%** | +14.5 pp | **+14.8 pp** |
| **A3 solo** — attacker alone, THE CONTROL | 24.18 | **67.5%** | **+17.7 pp** | **+18.0 pp** |

**All three defeat the defense, and all three exceed the undefended ceiling the defense was
protecting.**

### The control decides it

The criterion was fixed before running: `search` > `solo` means the attack exploits the defended
data (defense *broken*); `search` ≈ `solo` means the attacker never needed it (defense
*irrelevant*).

**`solo` (67.5%) beats `search` (64.3%).** Including the defended traces was slightly *worse* than
ignoring them. Hindsight distillation does not protect the teacher's advantage — an attacker with
the problems and any competent reasoning model reproduces and exceeds undefended distillation
without touching the defended data.

### Trace correctness does not predict student quality

| pool | traces reaching LIMO's answer | resulting student |
| --- | --- | --- |
| A2 search | **710/800 = 89%** | 64.3% |
| A3 solo | **392/800 = 49%** | **67.5%** |

**The pool whose traces are wrong half the time produced the better student.** This bears directly
on §4.1's `Score = correctness x epistemic_density x ...`: a scorer multiplying by `correctness`
would have ranked `search` above `solo` and selected the worse pool. **Correctness cannot enter
`Score` as a positive multiplicative factor on this evidence.** What transfers is the *process* —
how to explore, doubt and backtrack — not the answer at the end.

### But they beat LIMO on termination, not reasoning

MATH500, decomposed against LIMO as the zero point:

| condition | finished | acc\|fin | **reasoning** | **termination** | net | truncated |
| --- | --- | --- | --- | --- | --- | --- |
| LIMO | 441/500 | 78.2% | — | — | — | 31.9% |
| A1 style | 467/500 | 75.8% | **−2.3 pp** | **+4.1 pp** | +1.8 pp | 2.7% |
| A2 search | 462/500 | 77.1% | **−1.1 pp** | **+3.3 pp** | +2.2 pp | 2.3% |
| A3 solo | 465/500 | **79.8%** | **+1.4 pp** | **+3.8 pp** | +5.2 pp | 13.2% |

**The attacks do not out-reason LIMO; they out-terminate it.** Reasoning is within ±2.3 pp for all
three, and the whole advantage is the termination term — LIMO truncates 31.9% of its traces at
`cutoff_len 16384` while the reconstructions truncate 2.3–13.2%. **This is a confound, not a
triumph:** the mechanism is trace length, not superior epistemic content, and a LIMO run at a
higher `cutoff_len` would likely close most of the gap. `solo` is the partial exception — the only
condition that genuinely out-reasons LIMO, and it does so despite the worst truncation of the three.

### H1 vs H2: density matters, provenance does not

Pooled accuracy is **monotonic in epistemic density** across the three attacks (64.2 → 64.3 → 67.5
for 13.71 → 16.49 → 24.18). But `style` (doubt **fabricated** by a rewriter that already knew the
answer) and `search` (the model's **real** search) differ by **0.1 pp**.

That is **H1 with a refinement**: *how much* epistemic content a trace carries matters; *whether it
was genuinely searched* does not. H2's specific claim — that doubt must sit where reasoning is
really fragile — is unsupported by this pair, though the two differ in density by only 2.8, so this
tests provenance only at matched density.

### Doubt placement — and one measure not yet run

The design predicted fabricated doubt would **clump** at openings or in a closing "let me verify".
It does not: across the final pools, doubt positions are early/mid/late **32/39/29** for `style`
against LIMO's **34/34/32**, median within 0.03. Fabricated doubt is not merely as *effective* as
real searched doubt, it is as *well-placed*.

**Outstanding:** the design's second fidelity measure — sampling the student to locate where its
solutions genuinely diverge, then testing whether a trace's doubt coincides with those steps — has
**not been run**. That is the measure that tests H2 directly; the H1 conclusion currently rests on
accuracy parity and positional parity, neither of which asks whether doubt sits where reasoning is
actually fragile.

### Caveats

- **The termination confound above** is the main one; the `cutoff_len` control settles it.
- **The attacker's model is the defender's teacher** (both DeepSeek-R1-32B). This fixes capability
  so the prompt is the only variable, but the realistic threat model gives the attacker a *weaker*
  model. **Whether `solo` still wins with a weaker attacker is the most important follow-up.**
- **AIME is noise-dominated**: `style` scores 1/30 on AIME25 vs 6/30 for the others, with normal
  termination — a reasoning gap that 30 problems cannot resolve. Quote the pooled 600.

---

## Caveats and open items

| | |
| --- | --- |
| **Absolute numbers are not comparable to the paper.** | Greedy decoding differs across GPU architectures (B200 vs L40S). Use our own base row as the baseline; within-study contrasts are valid. |
| **11.4 pp gap to Kim et al. on MATH500** (69.0% vs their 80.4%). | **Roughly half is non-termination, not reasoning.** Per-problem join over all 500: we solve 32 they miss, they solve 89 we miss. **Of those 89, 42 (47%) are ones where our model produced no answer at all**, running to the token cap at ~81k chars. Our LIMO model is ~3x more verbose (median 44k vs 14k chars) at comparable epistemic density (229 vs 190 per response). Both models trained on the same data with the same config, so this is run-to-run variance in how strongly the 32% stop-token-less targets are learned — not a pipeline difference. The residual (~47 problems) is genuine reasoning gap. |
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

- **A working attack and a quantified failure mode for it** (Part 2) — the cheapest attack in the
  proposal defeats the defense at high epistemic fractions and *inverts* at low ones. Any
  curation-based attack has to clear that threshold to be worth running.

**Next:** M4 — refactor `hindsight` behind a `Defense` interface and add a second defense (PART-style
structural), then M5 — the remaining three `Score` components. The `cutoff_len` control run for
Part 2 is the cheapest outstanding experiment and should go first.

---

## Detailed records

| document | contents |
| --- | --- |
| `results/m1_base.md` | base cell; the greedy-nondeterminism diagnosis |
| `results/m2_limo.md` | LIMO cell; +13.0 pp; termination pathology; epoch sweep |
| `results/m3_hindsight.md` | hindsight cell (v1); collapse; difficulty dependence; dataset audit |
| `results/m3b_hindsight_v2.md` | **the corrected defense: transfers nothing, matches the published AIME24 cell** |
| `results/m6_supplementation.md` | the supplementation attack; dose-response sweep; non-monotonicity |
| `results/m6b_reconstruction.md` | **the reconstruction attack: results, the control, the correctness inversion** |
| `results/m6b_reconstruction_design.md` | its design and pre-registered predictions; the defects the pilot caught |
| `results/m6b_prompts.md` | the three reconstruction prompts verbatim, and how generation runs |
| `results/hindsight_versions.md` | v1 vs v2 defended datasets: what differs, where everything lives |
| `results/deviations.md` | all 7 deviations and whether each can affect results |
| `results/eval_table.md` | every metric, recomputed from stored per-problem verdicts |
| `CLAUDE.md` | how to run all of this with a different student model |
