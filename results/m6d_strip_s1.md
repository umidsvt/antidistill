# M6d — Are Kim et al.'s epistemic markers causal? (S1 marker-stripping)

**Status: complete (2026-09-18).** One training run (10.83 h, GPUs 4–7), 600 problem-evaluations.
Design, pre-registered rule and frozen lexicon: `results/decoupling_design.md` §C.

**Why this run, and what it is NOT.** It began as an attempt to separate epistemic density from
trace length. Stripping turned out to remove the *vocabulary* of doubt, not the reconsideration
(held-out reconsideration language 4.01 -> 4.03), so it cannot do that. It was re-scoped **before
training** to a narrower question: **are the nine marker tokens causal, or a proxy for the
reasoning that produces them?** It is not the density/length decoupling.

**The intervention.** The 7B `solo` pool with only the nine markers deleted — 97.5% of length kept,
all 298 correct answers kept, held-out reconsideration unchanged. Training config byte-identical to
`recon7b_solo` apart from dataset and output directory, so 7B `solo` is the exact counterfactual.

---

## 1. Pre-registered verdict: PARTIAL

| | MATH500 | AMC23 | AIME24 | AIME25 | **pooled** | fallback |
| --- | --- | --- | --- | --- | --- | --- |
| defended v2 | 56.8% | 27.5% | 3.3% | 3.3% | **49.5%** | 49.7% |
| LIMO | 69.0% | 55.0% | 20.0% | 13.3% | **62.8%** | 66.0% |
| 7B `solo` (counterfactual) | 77.0% | 55.0% | 16.7% | 16.7% | **69.5%** | 69.5% |
| **S1 — markers deleted** | **69.6%** | 47.5% | 16.7% | 10.0% | **62.5%** | 63.5% |

Rule (fixed before training): within 2.0 pp of `solo` = proxy; within 2.0 pp of defended = causal;
otherwise partial. **S1 = 62.5%, 7.0 pp below `solo` and 13.0 pp above defended — 65% of the way
along the defended->`solo` span. PARTIAL.**

**Deleting nine words cost 42 problems of 600**, well outside the ±2 pp band. The loss is carried by
MATH500 (−7.4 pp on 500 problems), not the small benchmarks. The markers are **not merely a proxy**,
but they **do not carry the whole effect** — about two-thirds of `solo`'s advantage survives without
them.

Training loss is effectively unchanged (S1 0.1195 vs `solo` 0.1202): the stripped traces were no
easier to fit, consistent with their reconsideration structure being intact.

---

## 2. Validity: the stripped text is fluent (checked after training)

The design listed a fluency read as validity check 1. **It was not run before training**; it was run
afterwards. It passes. Representative deletions:

- *"Hmm, okay. Let me try to break this down"* -> *"Okay. Let me try to break this down"*
- *"**Wait**, but down to (0,0) would take him to a hamburger"* -> *"But down to (0,0) would take him to a hamburger"*
- *"Hmm, projections. I remember..."* -> *"Projections. I remember..."*

A crude disfluency count fell (27.7 -> 23.2 per trace). **The 7 pp is not a penalty for training on
broken prose.**

---

## 3. Exploratory: what the students DO at inference

*Not pre-registered; interpret as hypothesis-generating.* MATH500, 500 generations per student:

| student | markers emitted /1kw | held-out reconsideration /1kw | mean chars | finished | acc \| finished |
| --- | --- | --- | --- | --- | --- |
| defended v2 | 0.01 | 0.06 | 89,199 | 498/500 | 57.0% |
| LIMO | 28.25 | 5.14 | 51,042 | 441/500 | 78.2% |
| 7B `solo` | 13.46 | 2.99 | 53,876 | 477/500 | **80.7%** |
| **S1** | **0.06** | **2.43** | 67,482 | 467/500 | **74.5%** |

Three things:

1. **The S1 student never learned to say the markers** (0.06 vs 13.46) — expected, it never saw them.
2. **It kept most of the reconsideration behaviour.** Held-out reconsideration language at inference
   is 2.43 against `solo`'s 2.99 — 81% retained, and still 40x the defense.
3. **Its reasoning got worse anyway.** Accuracy-given-finished fell **80.7% -> 74.5%**, so the loss is
   in reasoning quality, not just in producing an answer. It also generates ~25% longer.

**The hypothesis this suggests:** the marker tokens function as explicit *pivots* — a learned
"wait" that initiates re-evaluation — rather than as decoration on reasoning that happens anyway. A
student that reconsiders without them does so less effectively, and circles longer. That is a
mechanistic claim this run does not establish; it would need, e.g., injecting "wait" at inference
into the S1 student (`third_party/kim_eval/eval_with_fixed_prefix.py` does exactly this kind of cue
injection) and checking whether accuracy recovers.

---

## 3b. GSM8K (added 2026-09-23) — the loss is LARGER on easy problems

Fallback-graded, TP=1, 500 problems. Every condition terminates on all 500, so these are pure
reasoning differences.

| student | GSM8K | vs `solo` | markers emitted /1kw | held-out /1kw | mean chars |
| --- | --- | --- | --- | --- | --- |
| base | 72.2% | — | 0.04 | 0.02 | 8,321 |
| defended v2 | 81.2% | — | 0.06 | 0.04 | 101,863 |
| LIMO | 85.2% | — | 22.19 | 3.84 | 57,432 |
| 7B `solo` | **88.6%** | — | 7.14 | 1.81 | 68,163 |
| **S1 stripped** | **78.4%** | **−10.2 pp** | 0.08 | 1.27 | 79,577 |

**This complicates §3's hypothesis.** Removing the markers costs **−10.2 pp on GSM8K** against
**−7.0 pp on the hard suite** — *more* on easy problems, not less. If the markers were pivots that
initiate error recovery, their removal should matter most where error recovery matters most, i.e.
on hard problems. The opposite is observed.

**S1 also drops below the defense on GSM8K** (78.4% vs 81.2%), the only benchmark where any attack
does. It retains just +6.2 pp over base where `solo` retains +16.4 pp.

Two readings, not distinguished by this data:

1. **The markers regulate something other than error recovery** — e.g. pacing or self-checking that
   pays off even on easy problems. Note S1 generates ~17% longer than `solo` here (79,577 vs 68,163
   chars) while scoring lower: it circles more without concluding better.
2. **A ceiling/compression artifact.** GSM8K spans base 72.2% to `solo` 88.6%, so a fixed loss in
   reasoning quality converts into more pp here than on the hard suite, where the spread is wider.

Reading 2 can be tested cheaply by expressing each result as a fraction of its benchmark's
base->`solo` span: GSM8K S1 = 38% of the span; hard suite S1 = 65%. On that normalisation the loss
is still **larger** on GSM8K, which argues against pure compression and for reading 1.

---

## 4. What this means

- **For Kim et al.'s framing:** their nine-token lexicon is not just a correlate. Removing the surface
  tokens alone, with everything else held fixed, costs about a third of the effect.
- **For our own r = 0.940:** partly vindicated as more than a proxy measure — but the
  density/length confound (r = 0.978) is still untouched, because S1 does not move length.
- **For attackers:** a surface-token filter is a weak defense. Stripping the words removes only ~35%
  of the advantage; a defender would have to remove the reconsideration itself, which is what
  hindsight rewriting does — and which the reconstruction attacks (Parts 4–5) regenerate.
- **For `Score()`:** `epistemic_density` computed on the nine markers captures real signal, but a
  held-out reconsideration measure would capture signal it misses.

---

## 5. Caveats

| | |
| --- | --- |
| **One seed, one run.** A 7 pp gap on 500 MATH500 problems is well clear of noise, but it is a single training run. |
| **§3 is exploratory**, and the pivot mechanism is a hypothesis, not a result. |
| **The fluency check was run after training, not before** as the design specified. It passed, but the ordering should be stated. |
| **GSM8K not yet run** for S1; planned as the next step. |

## 6. Reproduction

```bash
.venv-infer/bin/python src/antidistill/attacks/strip_epistemic.py --level s1 \
    --input data/curated/recon7b_solo.json --output data/curated/strip_s1_recon7b_solo.json
scripts/21_strip_s1_run.sh                        # GPUs 4-7; ~11 h under load
.venv-infer/bin/python scripts/35_strip_s1_analysis.py
```
