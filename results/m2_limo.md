# M2 — LIMO cell (Qwen2.5-7B)

**Status: complete and replicated (2026-09-04).**

Training: 1,500 steps / 15 epochs, 11:15:59 wall-clock, final loss 0.015 (from 0.858), mean train
loss 0.1984, cosine LR fully decayed, no crashes. Checkpoints kept at epochs 5/10/15.

---

## 1. Headline result — +13.0 pp pooled over 600 problems

Greedy pass@1, base vs LIMO epoch 15, identical hardware and settings throughout.

| benchmark | n | base | LIMO ep15 | delta | base finished | LIMO finished | LIMO mean chars |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MATH500 | 500 | 55.0% | **69.0%** | **+14.0 pp** | 415/500 | 441/500 | 51,042 |
| AMC23 | 40 | 40.0% | **55.0%** | **+15.0 pp** | 39/40 | 34/40 | 65,581 |
| AIME24 | 30 | 20.0% | 20.0% | +0.0 pp | 26/30 | 19/30 | 82,302 |
| AIME25 | 30 | 6.7% | **13.3%** | +6.7 pp | 26/30 | 24/30 | 77,621 |
| **POOLED** | **600** | **299/600 = 49.8%** | **377/600 = 62.8%** | **+13.0 pp** | | | |

**Kim et al. report +13.4 pp. We measure +13.0 pp** — on 20x the sample, with no benchmark
degrading. The LIMO distillation effect reproduces.

---

## 2. AIME24 alone would have called this a null

The only benchmark showing nothing is AIME24 — the one the paper headlines and the one this
milestone was originally scoped to. Two independent handicaps, both measured:

**Resolution.** 30 problems means 3.33 pp per problem. A +13 pp effect is four problems. Our base
drew 2 problems lucky against Kim et al.'s (6/30 vs 4/30) and our LIMO 2 unlucky (6/30 vs 8/30) —
four tie-breaks, exactly spanning the effect. At M1 we had already measured that greedy decoding
diverges across GPU architectures (only 4/30 traces byte-identical to their reference); this is
that caveat cashing out.

**Non-termination, concentrated on hard problems.** AIME24 has the worst answer-production of the
four (19/30). See section 4.

Judged on AIME24 alone this reads as a flat null. Judged on 600 problems it is a clean hit. The
methodological lesson is worth carrying into M3 and Phase 4: **a 30-problem greedy benchmark
cannot resolve the effect sizes this project studies.**

---

## 3. Epistemic verbalization transferred — emphatically

`scripts/31_epistemic_density.py`, AIME24 greedy traces:

| metric | base | LIMO ep15 |
| --- | --- | --- |
| mean words / response | 646 | **14,788** |
| epistemic tokens / response | 4.20 | **477.53** |
| responses with >=1 epistemic token | 6/30 | **30/30** |
| `wait` | **0** | **6,013** |
| `alternatively` | 0 | 2,329 |
| `perhaps` | 0 | 1,765 |
| `maybe` | 0 | 1,578 |
| bypass phrases (Appendix H.1) | 1 | 138 |

The base model essentially never verbalizes uncertainty; `wait` appears zero times in 30
responses. After 800 samples of SFT it does so constantly. **The linguistic habit transferred
exactly as Kim et al.'s thesis predicts** — and this was measurable before, and independently of,
any accuracy result.

---

## 4. The termination pathology (present in Kim et al.'s model too)

LIMO-trained models sometimes generate until the 32,768-token cap and never emit an answer. Those
are automatic zeros regardless of reasoning quality.

**Root cause — 32% of SFT targets have no stop token.** Measured on the training data as
LLaMA-Factory encodes it (`cutoff_len: 16384`, `template: qwen`):

```
examples within cutoff : 544  -> keep stop token: 544/544
examples over cutoff   : 256  -> keep stop token:   0/256
```

`target_ids = target_ids[:target_len]` (`data/processors/supervised.py`) cuts the trailing
`<|im_end|>` off every over-length example. Those 256 are the *longest* traces, i.e. the hardest
problems — so the never-stop behaviour is learned specifically for the regime that hard benchmarks
put the model in. Predicted in Task 0.3 (`results/task0.3_pool_stats.md`) before training ran.

**It is difficulty-dependent, and it is in their data too.** Kim et al.'s released generations:

| Kim et al. LIMO Qwen2.5-7B | finished | mean chars | pass@1 |
| --- | --- | --- | --- |
| MATH500 | 475/500 = **95%** | 26,480 | 402/500 = 80.4% |
| AIME24 | 22/30 = **73%** | 69,158 | 8/30 = 26.7% |

So this is a property of the LIMO recipe at this cutoff, not of our reproduction.

**Consequence for later milestones:** report `accuracy | finished` alongside raw pass@1 for every
condition. Hindsight traces are short and confident, so M3's model will terminate *more* reliably
than M2's — raw pass@1 would silently compare models with different answer-production rates and
attribute the difference to reasoning.

---

## 5. Epoch sweep — what the intermediate checkpoints bought

Greedy, AIME24. This is why epochs 5/10/15 were kept rather than only the final one.

| | pass@1 | finished | mean chars | accuracy \| finished |
| --- | --- | --- | --- | --- |
| base | 6/30 = 20.0% | 26/30 | 3,377 | 23% |
| LIMO ep5 | 4/30 = 13.3% | **5/30** | 90,316 | 80% *(n=5)* |
| LIMO ep10 | 6/30 = 20.0% | **19/30** | 80,833 | 32% |
| LIMO ep15 | 6/30 = 20.0% | **19/30** | 82,302 | 32% |
| kim LIMO | 8/30 = 26.7% | 22/30 | 69,158 | 36% |

**The extra epochs teach termination; they do not overfit it away.** The hypothesis under test was
that epoch 15 (train loss 0.015 on 800 examples) had memorised past its peak. The opposite holds:
answer-production climbs 5/30 → 19/30 from epoch 5 to 10. The 544 non-truncated examples teach the
model to stop and need many passes to outweigh the 256 that teach it not to.

**It saturates by epoch 10.** Epochs 10 and 15 are indistinguishable on every measure. **A
10-epoch budget would save ~3.7 h of every 11.3 h run at no measured cost** — worth taking for M3
and Phase 4, as a stated deviation.

*Caveat on ep5's 80%:* that is 4 correct out of 5 responses. A single flip moves it 20 pp, and the
responses terminating early are plausibly just the easy problems. `accuracy | finished` must
always be read jointly with the finished count, never alone.

---

## 6. Open: an 11.4 pp gap to Kim et al. on MATH500

Their generations for the same model and benchmark grade to **402/500 = 80.4%**; ours to
**345/500 = 69.0%**. Our data does not explain this away. The clearest correlate is verbosity:

| | ours | Kim et al. |
| --- | --- | --- |
| MATH500 pass@1 | 69.0% | 80.4% |
| MATH500 finished | 441/500 = 88.2% | 475/500 = 95.0% |
| MATH500 mean chars | **51,042** | **26,480** |
| AIME24 finished | 19/30 | 22/30 |

Our model generates roughly **twice as much text** and terminates less reliably on every
benchmark. That is consistent and directional, not noise. Plausible sources: run-to-run variance
in what the 256 stop-token-less examples teach, or a difference in their setup not captured by the
LIMO default config. **Unresolved and flagged rather than explained.**

---

## 7. What was not run

**avg@16 for LIMO.** Started and deliberately killed after 33 minutes at **0/480 completions** —
because the model runs to the token cap, 16 samples cost ~16x a greedy run (~12 h for three
checkpoints). Judged a poor use of a shared node once the dominant failure mode was identified.

This was a cost decision, not a finding. It mattered less than feared: the multi-benchmark sweep
delivered the statistical power avg@16 would have provided, for less GPU time (~2 h) and with the
bonus of testing difficulty-dependence. Should a future question need low-variance estimates on a
single benchmark, avg@16 on one checkpoint (~2 h) remains the tool.

---

## 8. Reproduction

```bash
# train (11:15:59 on 8x L40S)
scripts/20_train.sh configs/train/qwen2.5-7b_limo.yaml
scripts/21_prune_checkpoints.sh saves/Qwen2.5-7B_limo 500 1500   # required: ~100GB/checkpoint

# evaluate (~2 h)
scripts/12_multibench.sh
.venv-infer/bin/python scripts/30_collect_results.py
.venv-infer/bin/python scripts/31_epistemic_density.py <base.jsonl> <limo.jsonl>
```

Deviations from the published setup: `results/deviations.md` (none affect the recipe; the one
touching computation, Liger fused cross-entropy, is validated against a non-Liger run).
