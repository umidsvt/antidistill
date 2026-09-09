# M3b — the hindsight defense, regenerated correctly

**Status: complete (2026-09-09).** Training 6h24m on 4 GPUs, 1,500 steps, mean train loss 0.0389.

The v1 defended dataset was produced by Kim et al.'s procedure, which calls the teacher through
the raw completions endpoint and stores the whole reply — scratchpad *and* answer, two full
solutions separated by an unmatched `</think>`. v2 fixes the teacher call
(`results/hindsight_versions.md`). **Everything else is identical**: same 800 problems, same
hyperparameters, same eval, same prompts.

---

## 1. Headline — the defense is much stronger than v1 made it look

Greedy pass@1, pooled over 600 problems.

| condition | MATH500 | AMC23 | AIME24 | AIME25 | **pooled** | vs base |
| --- | --- | --- | --- | --- | --- | --- |
| base | 55.0% | 40.0% | 20.0% | 6.7% | **49.8%** | — |
| LIMO (epistemic) | 69.0% | 55.0% | 20.0% | 13.3% | **62.8%** | +13.0 pp |
| hindsight **v1** | 64.2% | 37.5% | 6.7% | 3.3% | **56.5%** | +6.7 pp |
| **hindsight v2** | **56.8%** | **27.5%** | **3.3%** | **3.3%** | **49.5%** | **−0.3 pp** |

**Distilling on correctly-generated defended traces buys the student nothing at all.** v2 lands at
49.5% against an untrained baseline of 49.8% — a 2-problem difference in 600. The v1 dataset's
+6.7 pp was substantially an artifact of its own defects.

**It reproduces the published cell exactly.** Kim et al. report hindsight AIME24 = **3.3% (1/30)**.
v1 gave 2/30; **v2 gives 1/30**. On the one cell the paper reports for this model, the corrected
pipeline lands on their number. (AIME24 is 30 problems, so ±1 is within noise — but v1 was
consistently *above* their figure and v2 is not.)

This retires a caveat that has been in `REPORT.md` since M3: *"'hindsight collapses' and 'hindsight
beats base' are both true of the same checkpoint."* Under v2 only the first is true.

---

## 2. It does not stop the model finishing — it stops it being right

MATH500 (n=500):

| condition | finished | **accuracy \| finished** | mean chars |
| --- | --- | --- | --- |
| base | 415/500 | 66.3% | 3,883 |
| LIMO | 441/500 | **78.2%** | 51,043 |
| hindsight v1 | 499/500 | 64.3% | 90,709 |
| **hindsight v2** | **498/500** | **57.0%** | 89,199 |

v2 terminates almost perfectly — **better than base** (498 vs 415) — and still reasons **worse than
base** (57.0% vs 66.3%). Training on correctly-defended traces made the student *worse at
reasoning than no training at all*, while making it better at stopping.

That is the sharpest available statement of Kim et al.'s thesis, and v1 could not make it: v1's
`acc|finished` of 64.3% sat just below base, whereas v2's 57.0% is clearly below.

---

## 3. The student is clean

AIME24 generations:

| condition | epistemic tokens / response | `wait` | responses with any | **containing `</think>`** |
| --- | --- | --- | --- | --- |
| base | 0.0 | 0 | 1/30 | 0/30 |
| LIMO | 455.9 | 6,013 | 30/30 | 0/30 |
| hindsight v1 | 3.5 | 0 | 3/30 | **29/30** |
| **hindsight v2** | **0.0** | **0** | **0/30** | **0/30** |

The `</think>` artifact is gone from the student, as intended — v1 taught it to 29/30 responses.
And epistemic verbalization is now *completely* absent (0/30 responses), against v1's 3/30. The
defense transfers exactly what it is supposed to transfer, and nothing else.

---

## 4. Training dynamics — unchanged, and that matters

| run | step 100 | 300 | 500 | 700 | 900 | 1500 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LIMO | 0.6448 | 0.4084 | 0.2645 | 0.1419 | 0.0762 | 0.0150 | 0.1984 |
| hindsight v1 | 0.2393 | 0.0330 | 0.0080 | 0.0015 | 0.0001 | 0.0001 | 0.0338 |
| hindsight v2 | 0.3220 | 0.0501 | 0.0077 | 0.0018 | ~0.0001 | ~0.0001 | 0.0389 |

Both defended runs collapse to near-zero by step 900 while LIMO is still at 0.076. **Fixing the
generation pipeline did not change this**, which is evidence the effect is a property of the
defense rather than of v1's defects: traces with no branch points are near-trivial to fit, and the
low loss is the signature of the informative content having been removed.

---

## 5. Limitation — two variables moved, not one

v2 removed the artifacts **and** cut the token budget 2.6x (1.01 M → 388 k trained tokens), because
a v1 trace carried two complete solutions and a v2 trace carries one. There is no token-neutral
version of this fix.

So the −7.0 pp drop from v1 to v2 cannot be attributed to artifact removal alone. Some of it is
plausibly "the student simply saw less text". The honest claim is: **a correctly-generated
defended dataset transfers nothing, and the v1 dataset's apparent +6.7 pp came from some
combination of its duplicated solutions and its `</think>` scaffolding.**

A token-matched control would need the v2 generator run at ~2.6x the trace count (2,080 problems),
which changes problem coverage instead. Worth stating as a limitation rather than engineering
around.

---

## 6. Consequence for M6a — the supplementation results rest on v1

**`limo_mixed_{50,25,10}.json` were all built from the v1 defended pool.** Every number in Part 2
of `REPORT.md` therefore measures supplementation against the *weaker* defense.

The direction of the correction is predictable but its size is not: with a v2 base of 49.5%
instead of 56.5%, the LIMO-minus-hindsight gap the attack has to close widens from 6.3 pp to
13.3 pp, so the "% of gap recovered" figures will all move. **The non-monotonic mix10 finding is
the one at risk** — it was defined by mix10 (52.2%) falling *below* hindsight (56.5%), and v2
hindsight is 49.5%, which is already below mix10.

Rebuilding the three mixtures on v2 and retraining is the correct follow-up. Until then, Part 2's
conclusions should be read as holding for the v1 defense specifically.

---

## 7. Reproduction

```bash
scripts/40_serve_teacher.sh                     # needs --reasoning-parser deepseek_r1
.venv-infer/bin/python src/antidistill/defenses/hindsight.py \
    --input data/raw/limo_v2.json --output data/defended/limo_hindsight_chat.json \
    --api chat --validator llm --temperature 0.6 --eval-max-tokens 8192
GPUS=4,5,6,7 scripts/20_train.sh configs/train/qwen2.5-7b_hindsight_v2.yaml
scripts/21_prune_checkpoints.sh saves/Qwen2.5-7B_hindsight_v2 500 1500 &   # REQUIRED
scripts/16_hindsight_v2_eval.sh
```

Generation 10h21m (800 traces, 453 retries); training 6h24m; evaluation ~2h.
