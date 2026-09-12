# M6b — Epistemic reconstruction attack: results

**Status: complete (2026-09-12).** Three training runs, 1,800 problem-evaluations.
Design and predictions, fixed before any result was seen: `results/m6b_reconstruction_design.md`.
Prompts: `results/m6b_prompts.md`.

---

## 1. Headline — the defense is not merely broken, it is bypassed

Greedy pass@1, pooled over 600 problems. All conditions: 800 problems, identical hyperparameters,
1,500 steps.

| condition | epistemic /1k words | pooled | vs base | vs defended |
| --- | --- | --- | --- | --- |
| base (untrained) | — | 49.8% | — | — |
| **defended v2** | 0.02 | **49.5%** | −0.3 pp | — |
| LIMO (undefended ceiling) | 35.57 | 62.8% | +13.0 pp | +13.3 pp |
| **A1 style** (fabricated doubt) | 13.71 | **64.2%** | +14.3 pp | **+14.7 pp** |
| **A2 search** (real doubt + defended hint) | 16.49 | **64.3%** | +14.5 pp | **+14.8 pp** |
| **A3 solo** (attacker alone, THE CONTROL) | 24.18 | **67.5%** | **+17.7 pp** | **+18.0 pp** |

**All three attacks defeat the defense completely** — and all three **exceed the undefended LIMO
ceiling** the defense was protecting.

---

## 2. The control decides it: the defense is IRRELEVANT, not broken

The criterion was fixed before running (design §3):

> **`search` > `solo`** → the defended traces contribute something the attacker cannot supply; the
> attack genuinely exploits the defense, which is therefore *broken*.
> **`search` ≈ `solo`** → the attacker never needed the defended data; the defense is *irrelevant*.

**`solo` (67.5%) beats `search` (64.3%) by 3.2 pp.** The attacker working with **no defended data
at all** produces the best student of the three. The defended traces did not merely fail to help —
including them was slightly *worse* than ignoring them.

So hindsight distillation does not protect the teacher's advantage. An attacker holding the
problems and any competent reasoning model reproduces — and exceeds — what undefended distillation
would have given them, without touching the defended traces.

---

## 3. The most consequential finding: trace correctness does not predict student quality

Measured at generation time, against the *undefended* LIMO trace:

| pool | traces reaching LIMO's answer | resulting student |
| --- | --- | --- |
| A2 search | **710/800 = 89%** | 64.3% |
| A3 solo | **392/800 = 49%** | **67.5%** |

**The pool whose traces are wrong 51% of the time produced the better student.** Training-trace
answer-correctness is not merely a weak predictor of distillation quality here — it is inverted.

This bears directly on proposal §4.1, whose curation function is
`Score = correctness x epistemic_density x distributional_alignment x difficulty_match`. A scorer
multiplying by `correctness` would have ranked `search` above `solo` and selected the worse pool.
**Correctness cannot enter `Score` as a positive multiplicative factor on this evidence.**

The natural reading is that what transfers is the *process* — how to explore, doubt, and backtrack
— not the answer at the end. A trace that searches honestly and arrives somewhere wrong still
demonstrates searching.

---

## 4. Why they beat LIMO: termination, not reasoning

Decomposing MATH500 against LIMO as the zero point
(`dpass = F_x(A_x - A_L) + A_L(F_x - F_L)`, F = finished fraction, A = accuracy given finished):

| condition | finished | vs LIMO | acc\|fin | **reasoning** | **termination** | net | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LIMO | 441/500 | — | 78.2% | — | — | — | 31.9% |
| A1 style | 467/500 | +26 | 75.8% | **−2.3 pp** | **+4.1 pp** | +1.8 pp | 2.7% |
| A2 search | 462/500 | +21 | 77.1% | **−1.1 pp** | **+3.3 pp** | +2.2 pp | 2.3% |
| A3 solo | 465/500 | +24 | **79.8%** | **+1.4 pp** | **+3.8 pp** | +5.2 pp | 13.2% |
| defended v2 | 498/500 | +57 | 57.0% | −21.1 pp | +8.9 pp | −12.2 pp | 0.0% |

**The attacks do not out-reason LIMO. They out-terminate it.** Reasoning quality is within
±2.3 pp of LIMO for all three; the entire advantage is the termination term.

The cause is the same `cutoff_len` pathology as M6a: LIMO truncates **31.9%** of its traces at
`cutoff_len 16384`, stripping the stop token from the longest ones, while the reconstructed pools
truncate 2.3–13.2%. The reconstruction is *incidentally* shorter, and shorter traces terminate.

**This is a confound, not a triumph, and it must be stated wherever the headline is.** "The attack
beats undefended distillation" is true, but the mechanism is trace length, not superior epistemic
content. A LIMO run at a `cutoff_len` above 32,901 would likely close most of the gap — the
control experiment already outstanding from M6a.

`solo` is the partial exception: it is the only condition that genuinely out-reasons LIMO
(+1.4 pp), and it does so *despite* the worst truncation of the three attacks (13.2%).

---

## 5. H1 vs H2: density matters, provenance does not

The two readings under test:

- **H1 — verbalization is BEHAVIOUR.** Any plausible doubt installs the habit.
- **H2 — verbalization is INFORMATION** about *which* steps are error-prone, found by searching.

| | epistemic /1k words | pooled | doubt is |
| --- | --- | --- | --- |
| A1 style | 13.71 | 64.2% | **fabricated** — the rewriter knew the answer and never searched |
| A2 search | 16.49 | 64.3% | **real** — the model's own search trace |
| A3 solo | 24.18 | 67.5% | **real**, unguided |

Two things fall out, and they pull in different directions:

**Density predicts outcome.** Across the three attacks, pooled accuracy is monotonic in epistemic
density (64.2 → 64.3 → 67.5 for 13.71 → 16.49 → 24.18).

**Provenance does not.** `style` (fabricated) and `search` (real search) are separated by
**0.1 pp** at similar density. Doubt the model invented after already knowing the answer teaches
the student as effectively as doubt it actually experienced.

That is **H1 with a refinement**: *how much* epistemic content a trace carries matters; *whether it
was genuinely searched* does not. H2's specific claim — that doubt must sit where the reasoning is
really fragile — is not supported by this pair.

The caveat: `style` and `search` differ in density by only 2.8, so this compares provenance at
roughly matched density and cannot rule out an effect that appears at larger separations.

---

### 5.1 Doubt placement — fabricated doubt is positionally indistinguishable

The design (§5) predicted fabricated doubt would **clump** at openings or in a closing
"let me verify". Measured on the final 800-trace pools:

| pool | occurrences | early | mid | late | median position |
| --- | --- | --- | --- | --- | --- |
| LIMO (real doubt) | 192,783 | 34% | 34% | 32% | 0.48 |
| **A1 style (fabricated)** | 25,646 | 32% | 39% | 29% | 0.49 |
| A2 search (real) | 48,816 | 30% | 40% | 30% | 0.50 |
| A3 solo (real) | 110,316 | 31% | 35% | 33% | 0.51 |

**The prediction was wrong.** Fabricated doubt is spread through the trace exactly as real searched
doubt is — no clumping, median position within 0.03 of LIMO's. It is not merely as *effective* as
real doubt (§5), it is as *well-placed*, which removes the most obvious mechanism by which H2 could
have been true and the accuracy comparison misleading.

### 5.2 NOT DONE — the placement-vs-difficulty correlation

The design's second fidelity measure has **not been run**: sample the student n times per problem,
locate where its solutions diverge (an empirical marker of genuine difficulty), and test whether a
trace's doubt coincides with those steps. Real traces should correlate; fabricated ones should not.

This is the measure that operationalises H2 **directly**. The H1 conclusion above rests on two
indirect lines of evidence — accuracy parity between `style` and `search` (0.1 pp) and positional
parity (§5.1) — neither of which asks whether doubt sits where the reasoning is *actually* fragile.
It requires n-sample generation over 800 problems (GPU) and remains outstanding.

---

## 6. Caveats

| | |
| --- | --- |
| **The termination confound (§4)** is the main one. The attacks' advantage over LIMO is length-driven, not reasoning-driven. Re-running LIMO at `cutoff_len 33792` is the control. |
| **AIME benchmarks are noise-dominated.** `style` scores 1/30 on AIME25 against `search`/`solo`'s 6/30 — a 5-problem gap at 3.33 pp per problem. Its termination there is normal (22/30), so it is a reasoning gap, but 30 problems cannot resolve it. The pooled 600 is the number to quote. |
| **The attacker's model is the defender's teacher.** Both are DeepSeek-R1-Distill-Qwen-32B. This holds capability fixed so the only variable is the prompt, but the realistic threat model gives the attacker a *weaker* model. Whether `solo` still beats `search` with a weaker attacker is untested and is the most important follow-up. |
| **`solo` failed to produce any trace on 50/800 problems** at first pass (repetition loops), recovered by salvage and regeneration. Those problems are ~11% harder than average. An unaided attacker's failures concentrate on hard problems. |
| **Four traces sit at 0.654–0.692 distinct-12-grams**, above our loop guard but below the 0.879 floor of any real LIMO trace. Accepted and documented rather than filtered by a third threshold. |

---

## 7. What this changes for the project

1. **Hindsight distillation is not a viable defense.** Not because it is breakable, but because it
   defends the wrong thing: the attacker never needed the defended traces.
2. **`Score`'s `correctness` factor is contraindicated** (§3). The 49%-correct pool beat the
   89%-correct pool.
3. **A defense must target the *process*, not the answer.** What transferred was how to search.
   Any defense that leaves the student able to observe or synthesise a search process leaves the
   channel open.
4. **The `cutoff_len` control is now the highest-value outstanding experiment.** It is load-bearing
   for M6a's non-monotonicity *and* for §4 here.

---

## 8. Reproduction

```bash
scripts/40_serve_teacher.sh                              # --reasoning-parser deepseek_r1
for m in style search solo; do
  .venv-infer/bin/python src/antidistill/attacks/epistemic_reconstruction.py --mode $m \
      --defended data/defended/limo_hindsight_chat.json \
      --output data/curated/limo_recon_$m.json --base-url http://127.0.0.1:8011/v1
done
GPUS=0,1,2,3 scripts/17_train_recon.sh                   # pruner attached per run
GPUS=0,1,2,3 scripts/18_recon_eval.sh search solo style
```

Generation ~14 h total; training 6.85 / 7.74 / 6.59 h; evaluation ~9 h.
**Do not run evaluation concurrently with training on this host** — GPUs 0–3 and 4–7 are on
separate NUMA nodes (`SYS` in `nvidia-smi topo -m`), and a concurrent eval ran 10x slower
(35 h projected vs 3 h) while training was unaffected.
