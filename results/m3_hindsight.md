# M3 — Hindsight cell (Qwen2.5-7B)

**Status: complete (2026-09-05).**

Training: 1,500 steps / 15 epochs on 4 GPUs, **6:14:12**, mean train loss **0.0338**, final loss
1e-4. Dataset: 800 LIMO problems with traces re-derived confidently by
DeepSeek-R1-Distill-Qwen-32B (`data/defended/limo_hindsight_ds32b.json`).

---

## 1. Result

Greedy pass@1, identical hardware and settings across all three conditions.

| benchmark | n | base | LIMO | **HINDSIGHT** | hindsight vs base |
| --- | --- | --- | --- | --- | --- |
| MATH500 | 500 | 55.0% | 69.0% | **64.2%** | **+9.2 pp** |
| AMC23 | 40 | 40.0% | 55.0% | **37.5%** | −2.5 pp |
| AIME24 | 30 | 20.0% | 20.0% | **6.7%** | **−13.3 pp** |
| AIME25 | 30 | 6.7% | 13.3% | **3.3%** | −3.4 pp |
| **POOLED** | **600** | **49.8%** | **62.8%** | **56.5%** | +6.7 pp |

`base 299/600 · LIMO 377/600 · hindsight 339/600`

**Hindsight is below LIMO on every single benchmark** (pooled −6.3 pp). Against *base* it is
sharply negative on the hard benchmarks and positive on the easy one.

### The AIME24 cell replicates

| | base | LIMO | hindsight | hindsight/base |
| --- | --- | --- | --- | --- |
| ours | 20.0% | 20.0% | **6.7%** | **0.33x** |
| Kim et al. | 13.3% | 26.7% | 3.3% | 0.25x |

Same direction, comparable magnitude. On the benchmark the paper reports, the hindsight collapse
reproduces.

---

## 2. The collapse is not a termination artifact — it survives a handicap

Hindsight traces keep their stop token 99.1% of the time in training versus LIMO's 68%
(§4), so the hindsight model terminates far more reliably at inference:

| finished (produced any parseable answer) | base | LIMO | hindsight |
| --- | --- | --- | --- |
| MATH500 | 415/500 | 441/500 | **499/500** |
| AMC23 | 39/40 | 34/40 | 37/40 |
| AIME24 | 26/30 | 19/30 | **30/30** |
| AIME25 | 26/30 | 24/30 | **30/30** |

It answered **30/30** AIME24 problems against LIMO's 19/30 — more chances to be right — and still
scored far lower. Conditioning on producing an answer makes the gap starker, not smaller:

| accuracy \| finished | base | LIMO | hindsight |
| --- | --- | --- | --- |
| MATH500 | 66.3% | 78.2% | 64.3% |
| AMC23 | 41.0% | 64.7% | 40.5% |
| AIME24 | 23.1% | 31.6% | **6.7%** |
| AIME25 | 7.7% | 16.7% | **3.3%** |

**Given that it produces an answer, the hindsight model is right 6.7% of the time on AIME24
versus LIMO's 31.6%.** Removing epistemic verbalization did not stop the model finishing — it
stopped it being right.

---

## 3. The effect is difficulty-dependent, and that is the mechanism

Hindsight *helps* on MATH500 (+9.2 pp) and *collapses* on AIME24 (−13.3 pp).

This is the same asymmetry Kim et al. document for Qwen2.5-Math-7B (AIME24 16.7% → 0.0% but
MATH500 52.4% → 59.0%), and it is what their framework predicts. Epistemic verbalization is the
channel through which a model **recovers from a wrong path**. On MATH500 the model mostly solves
directly and never needs to recover, so confident derivations transfer useful procedural form at
no cost. On AIME the model must detect and reverse its own errors — exactly the capability the
hindsight rewrite removes.

**Consequence for how this project reports results:** "hindsight collapses" and "hindsight beats
base" are both true of the same checkpoint. Which one you see depends entirely on benchmark
difficulty. Any future defense evaluation must report across a difficulty range or it will
support whichever conclusion the author already held.

---

## 4. Dataset verification (Task 1.6, run before training)

| metric | LIMO | HINDSIGHT | ratio |
| --- | --- | --- | --- |
| mean words / trace | 7,200 | 772 | 0.11x |
| epistemic tokens / trace | 263.6 | **4.8** | **0.02x** |
| traces with >=1 epistemic token | 800/800 | 206/800 | |
| `wait` | 61,663 | **463** | **0.008x** |
| truncated at `cutoff_len` (loses stop token) | **32.0%** | **0.9%** | |

Quality: **796 good / 4 exhausted** (0.5%; indices 158, 198, 522, 681). An "exhausted" trace
failed the GOOD/BAD validator 8 times and fell back to its last attempt, so it may contain a
wrong solution. At 0.5% this is immaterial; above ~2% the condition should be reported with and
without them.

### Known artifact: an orphan `</think>` tag

**760/800 (95%)** of hindsight traces contain a closing `</think>` with no opening tag; LIMO has
zero. Cause: DeepSeek-R1-Distill is a reasoning model, and Kim et al.'s forced `"Okay, so I"`
prefix drops it inside an implicit think block, so it closes the tag and writes its usual
"final answer" section. **This is faithful to their script** (no post-processing), so their
dataset almost certainly has it too.

Measured relationship between the two halves (I initially mischaracterised this as duplication):

| | |
| --- | --- |
| exactly identical halves | **0 / 760** |
| median char similarity (difflib) | **0.28** |
| B's vocabulary already in A | 88.2% |
| final boxed answers agree | 756/759 = **99.6%** |

So the structure is *informal confident derivation → `</think>` → formal restatement of the same
solution* — a paraphrase, not a copy. It is still an asymmetry versus LIMO beyond epistemic
content, and remains an uncontrolled variable in the LIMO-vs-hindsight contrast.

---

## 5. Training dynamics — the data is far easier to fit

| epoch | 1 | 2 | 3 | 4 | 5 | 7 | 9 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M2 (LIMO) | 0.645 | 0.634 | 0.408 | 0.351 | — | — | — | 0.015 |
| M3 (hindsight) | 0.239 | 0.088 | 0.033 | **0.015** | 0.008 | 0.0015 | 0.0001 | 0.0001 |

M3 reached M2's *final* loss at **epoch 4** and converged by epoch 9. Mean train loss over the
whole run: **0.0338 vs 0.1984** — 6x lower.

This is the thesis visible in the optimisation itself. Hindsight traces have no branch points, no
abandoned approaches, no moments where the next token depends on the model recognising its own
error, so they are nearly trivial to predict. **Low training loss here is not better learning —
it is the signature of the informative content having been removed.**

**Epoch sweep (AIME24 greedy):** ep5 1/30, ep10 1/30, ep15 2/30. Flat within noise (1–2
problems). Unlike M2 — where later epochs measurably improved termination — the extra epochs here
buy nothing, consistent with the loss being converged by epoch 9.

---

## 6. Open observation

The hindsight model is the **most verbose at inference** (~95k mean chars on AIME24) despite
training on the *shortest* traces (772 words mean). Base is ~3.4k. So verbosity at inference is
not inherited from trace length. Plausibly it generates confidently and at length without ever
reaching a conclusion it can commit to — but this is unexplained and worth investigating, since
it bears on how any future defense's output length should be interpreted.

---

## 7. Reproduction

```bash
# 1. generate the defended dataset (~2 h on 2x TP=4 replicas)
GPUS=0,1,2,3 PORT=8001 scripts/40_serve_teacher.sh
GPUS=4,5,6,7 PORT=8002 scripts/40_serve_teacher.sh
scripts/41_generate_hindsight.sh
.venv-infer/bin/python scripts/42_merge_hindsight.py \
    --shards data/defended/_shard0.json data/defended/_shard1.json data/defended/_shard3.json \
    --output data/defended/limo_hindsight_ds32b.json --expect 800

# 2. verify BEFORE training
.venv-infer/bin/python scripts/31_epistemic_density.py <limo> <hindsight>

# 3. train (6:14 on 4 GPUs) and evaluate
GPUS=0,1,2,3 scripts/20_train.sh configs/train/qwen2.5-7b_hindsight.yaml
scripts/21_prune_checkpoints.sh saves/Qwen2.5-7B_hindsight 500 1500 &
scripts/13_m3_eval.sh
```
