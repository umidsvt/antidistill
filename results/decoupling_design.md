# Separating epistemic density from trace length

**Status: design, 2026-09-15. Two approaches already ruled out empirically.**

## The problem

`results/m6c_weaker_attacker.md` §3 reports the project's strongest correlation: across the six
reconstruction runs, epistemic density predicts pooled accuracy at **r = +0.940**. But:

| | r |
| --- | --- |
| epistemic density vs pooled accuracy | +0.940 |
| trained tokens vs pooled accuracy | +0.938 |
| **epistemic density vs trained tokens** | **+0.978** |

**Nothing in the project separates "epistemic content helps" from "long traces help."** Every
headline claim that leans on epistemic density is exposed to this. It cannot be fixed with more
evaluation benchmarks; it needs a training condition where the two axes come apart.

---

## What has been ruled out

### A. Reselecting existing traces — FAILS

We own 4,800 traces (6 pools x 800 problems), so in principle one could pick, per problem, a
high-density and a low-density trace with matched length. Measured 2026-09-15:

| | |
| --- | --- |
| within-problem corr(density, length) | **r = +0.826** |
| problems where they separate (r < 0.5) | **12%** |
| best greedy length-matched construction | token ratio still **5.01x**, density ratio 7.7x |

A problem's densest trace is nearly always its longest. **Density and length are coupled inside
each problem, not merely across pools**, so no reselection works.

### B. Generating a "verbose but confident" pool — FAILS

Prompt the teacher for exhaustive step-by-step detail while banning every uncertainty marker,
aiming for long-like-`solo` and confident-like-the-defense. Both harvesting variants were piloted
on 20 problems:

| variant | tokens | epi/1kw | verdict |
| --- | --- | --- | --- |
| defended v2 (reference: short + confident) | 496 | 0.00 | — |
| **v1** — keep `reasoning_content` + answer | **8,512** | **18.83** | long, but the doubt returns |
| **v2** — suppress the scratchpad | **612** | **0.00** | confident, but the trace collapses |
| 7B solo (reference: long + doubtful) | 13,737 | 30.07 | — |

Each variant achieves exactly one property and loses the other. The confidence instruction
disciplines the *visible answer* while the private scratchpad doubts anyway; remove the scratchpad
and the model stops searching, so length collapses to below even `style`'s 1,292 tokens — despite
an explicit instruction to show every intermediate step in full.

**This is the same coupling as (A), appearing in generation rather than selection: for this model,
length comes from searching, and searching is what produces doubt.** The two are not
independently controllable by prompting.

---

## C. Post-hoc stripping — MEASURED 2026-09-17: removes the vocabulary, not the reconsideration

Implemented in `src/antidistill/attacks/strip_epistemic.py` (S1/S2/S3, LaTeX spans protected so
mathematics is never cut). Applied to the 7B `solo` pool, 800 traces:

| level | mean tokens | length retained | 9 markers /1kw | **held-out reconsideration /1kw** | answers kept |
| --- | --- | --- | --- | --- | --- |
| defended v2 (reference) | 484 | — | 0.02 | **0.07** | — |
| `solo` original | 11,612 | 100% | 26.06 | **4.01** | 298/298 |
| **S1 marker** | 11,319 | **97.5%** | 0.04 | **4.03** | **298/298** |
| S2 clause | 10,474 | 90.2% | 0.04 | **4.03** | 297/298 |
| S3 sentence | 8,640 | 74.4% | 0.04 | **2.35** | 293/298 |

**The nine-marker column is circular** — it counts exactly the words S1 deletes, so it cannot tell
whether epistemic *content* was removed. The held-out column uses reconsideration language the
stripper never targets (`let me double-check`, `that doesn't work`, `instead`, `is that right`,
`hold on`, `I made a mistake`, ...).

On that honest measure **S1 and S2 change nothing** (4.01 -> 4.03) and S3 cuts it only 41%, leaving
it 33x the defense. **Stripping removes the vocabulary of doubt, not the reconsideration.**
Reconsideration is expressed open-endedly: any lexicon removed, a held-out one catches what remains,
and chasing it costs length (S3 already loses 26%).

**This is the third approach to hit the same wall** (A: reselection; B: generation; C: stripping).
The density/length confound does not appear separable for this model's traces by any of them.

### What S1 CAN test instead

S1 is an unusually clean intervention: the nine markers gone, 97.5% of length kept, all 298 correct
answers preserved, every other form of reconsideration untouched. That isolates a different question,
central to Kim et al.:

> **Are the nine epistemic tokens causal, or a proxy for the reasoning that produces them?**

- **S1 ~ `solo` (69.5%)** -> the markers are a proxy. Kim et al.'s density metric measures a correlate,
  and our own r = 0.940 is not mechanistic evidence.
- **S1 ~ defended (49.5%)** -> the literal surface tokens carry the effect.

One training run plus evaluation. **It must be reported as a test of the marker lexicon, not as the
density/length decoupling** — it does not hold reconsideration content fixed at the defense's level.

---

### Runbook — S1 marker-stripping (prepared 2026-09-17, not yet run)

Everything is in the repo; nothing needs reconstructing from chat.

| artifact | path |
| --- | --- |
| stripper + **frozen held-out lexicon** | `src/antidistill/attacks/strip_epistemic.py` (`HELD_OUT`, `held_out_density`) |
| training pool (built) | `data/curated/strip_s1_recon7b_solo.json`, registered as `strip_s1_recon7b_solo` |
| training config | `configs/train/qwen2.5-7b_strip_s1.yaml` — **byte-identical to `qwen2.5-7b_recon7b_solo.yaml` except dataset and output_dir**, so `recon7b_solo` is the exact counterfactual |
| train + eval | `scripts/21_strip_s1_run.sh` (GPUs 4–7 only; `WITH_GSM8K=1` to add GSM8K) |
| **verdict** | `scripts/35_strip_s1_analysis.py` |

**Recorded pre-training numbers**, which `35_strip_s1_analysis.py` re-derives as a check:

| pool | 9 markers /1kw | held-out /1kw | mean tokens | correct answers kept |
| --- | --- | --- | --- | --- |
| 7B `solo` | 26.06 | 4.01 | 11,612 | 298/298 |
| **S1** | **0.04** | **4.03** | **11,319 (97.5%)** | **298/298** |
| defended v2 | 0.02 | 0.07 | 484 | — |

**Pre-registered decision rule** (pooled 600-problem hard suite, boxed-only, fixed before training):

| S1 result | verdict |
| --- | --- |
| within **2.0 pp** of `solo` (**69.5%**) | markers are a **proxy** — Kim et al.'s density metric measures a correlate, and our r = 0.940 is not mechanistic evidence |
| within **2.0 pp** of defended (**49.5%**) | markers are **causal** — the nine surface tokens alone reproduce the defense |
| otherwise | **partial** — report the position on the defended→`solo` span |

2.0 pp is 12 of 600 problems, about the `solo`-vs-`search` gap we already treat as noise.

**Do not edit `HELD_OUT` after seeing results.** Extending it to catch whatever remains is the
whack-a-mole that makes the measurement meaningless; add a versioned list and report both instead.

**Estimated wall-clock on GPUs 4–7** (from measured runs):

| step | basis | time |
| --- | --- | --- |
| training | 7B `solo` took 7.82 h; S1 has 97.5% of its tokens | ~7.5 h |
| hard-suite eval, TP=4 | 7B attacks took 2 h 20 m – 3 h 43 m | ~2.5–3.5 h |
| *optional* GSM8K, TP=1, GPU 7 | 7B `solo` took 7 h 12 m | ~7 h |
| **total** | | **~10–11 h**, or **~17–18 h** with GSM8K |

---

## C (original design, retained for the record)

**Idea.** Take the `solo` pool (longest, densest, and the best-performing attack) and **delete the
epistemic content in place**, leaving everything else — the equations, the intermediate algebra,
the structure, the order of exploration — untouched. Unlike regeneration, the length loss is
bounded by the fraction of tokens that are actually epistemic, not by the model's decision to stop
searching.

**Why it can work where A and B failed.** A and B both let the *model* choose the trade-off, and
it always couples them. Stripping is a text transform: we choose exactly what to remove.

### Procedure

Three strip levels, increasingly aggressive, so the length/density trade-off is measured rather
than assumed:

| level | what is removed | expected length retained |
| --- | --- | --- |
| **S1 marker-only** | the nine epistemic tokens as standalone words (`wait`, `alternatively`, `perhaps`, `maybe`, `hmm`, `actually`, `let me reconsider`, `on second thought`, `i think`), plus the immediately surrounding connective punctuation | ~95%+ |
| **S2 clause** | the *clause* containing each marker, up to sentence boundaries | ~85% |
| **S3 sentence** | the whole *sentence* containing each marker | ~70% |

Run all three, measure `(tokens retained, epistemic density)` for each, and pick the level that
lands nearest **`solo`'s length at the defense's density**. If none does, report the frontier —
that itself bounds how separable the axes are.

### Validity checks before training anything

1. **Grammaticality.** S2/S3 can leave dangling text. Sample ~30 stripped traces and read them;
   a trace that reads as broken teaches broken output, which would confound the result with a
   fluency effect.
2. **Answer preservation.** The final `\boxed{}` must survive and still match. Re-run
   `answers_agree()` against the undefended LIMO trace; any drop means we removed mathematics,
   not doubt.
3. **The search structure must survive.** Stripping should remove the *verbalisation* of
   reconsideration, not the reconsideration itself — a trace that tries an approach, abandons it,
   and tries another should still visibly do so, just without saying "wait, maybe". If S3 deletes
   the abandoned branches, it has become a different intervention (closer to the hindsight
   defense) and must be reported as such.
4. **Repetition guard.** Reuse `repetition_ratio >= 0.60`; stripping must not create loops.

### The experiment

| condition | length | density | source |
| --- | --- | --- | --- |
| `solo` (existing) | 10,452 tok | 25.99 | — |
| **`solo_stripped`** (new) | target >= 8,000 tok | target < 2 | S1/S2/S3 of `solo` |
| defended v2 (existing) | 484 tok | 0.02 | — |

One training run (~7 h) plus evaluation (~3 h). **No generation required.**

**Reading the result:**

- `solo_stripped` ≈ `solo` (69.5%) → **length is doing the work**; the r = 0.940 density finding
  collapses and every density-based claim in the report must be requalified.
- `solo_stripped` ≈ defended (49.5%) → **epistemic content is doing the work**; the finding holds
  and becomes considerably stronger, since length is now controlled.
- Intermediate → both contribute; report the split.

**The honest caveat to state either way:** stripping produces text no model would naturally emit.
A student trained on it may fail for distributional reasons rather than because the epistemic
content mattered. Validity check 1 partly guards this, but it cannot be eliminated — which is why
the *direction* of the result matters more than its magnitude.
