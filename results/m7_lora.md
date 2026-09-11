# M7 — LoRA arm: {LIMO, hindsight, mix50} × {16k, 32k}

**Status: complete (2026-09-11).** Six new training runs (53 h) and 4,200 new problem-evaluations
(21.7 h), plus a same-host `base` anchor. 28/28 eval jobs `ok`.

Two questions on top of the G1 replication: does the LIMO effect survive **parameter-efficient**
training, and how much of the measured gap was **truncation** rather than epistemics? The second
is why every condition is run at two context lengths — `cutoff_len 16384` truncates 32.0% of LIMO
traces but only 0.9% of hindsight ones, so a 16k comparison confounds epistemic content with
surviving truncation. At 32768 that is 0.2% vs 0.6%.

Every run: LoRA r=32, α=64, target `all`, lr 1e-4, 15 epochs, global batch 8, 1,500 steps —
**80.7M trainable params, 1.05% of the model.** Only the traces and `cutoff_len` differ.

---

## 1. Headline — the effect survives LoRA at 16k; the 32k half backfired

Greedy pass@1, pooled over 600 problems (MATH500 + AMC23 + AIME24 + AIME25).

| condition | epistemic | pooled | vs base | gap recovered* | **terminated** |
| --- | --- | --- | --- | --- | --- |
| base | — | 49.5% | — | — | **96.8%** |
| **LIMO 16k** | 100% | **61.8%** | +12.3 pp | 100% | 69.7% |
| mix50 16k | 50% | 55.7% | +6.2 pp | **5%** | 15.0% |
| hindsight 16k | 0% | 55.3% | +5.8 pp | 0% | 10.3% |
| **LIMO 32k** | 100% | **55.7%** | +6.2 pp | 100% | 50.3% |
| mix50 32k | 50% | 58.0% | +8.5 pp | 217% | 5.0% |
| hindsight 32k | 0% | 53.7% | +4.2 pp | 0% | 1.0% |

\* fraction of the LIMO-minus-hindsight gap recovered, within that context length.

### Per-benchmark breakdown

Greedy pass@1. Diagnostic only — the small benchmarks are individually unstable (section 6.2);
judge on the pooled column. Transcribed here because `outputs/` is gitignored.

| condition | AIME24 (30) | AMC23 (40) | AIME25 (30) | MATH500 (500) | pooled (600) |
| --- | --- | --- | --- | --- | --- |
| base | 16.7% | 42.5% | 3.3% | 54.8% | **49.5%** |
| LIMO 16k | 13.3% | 45.0% | 23.3% | 68.4% | **61.8%** |
| mix50 16k | 16.7% | 45.0% | 6.7% | 61.8% | **55.7%** |
| hindsight 16k | 3.3% | 40.0% | 6.7% | 62.6% | **55.3%** |
| LIMO 32k | 6.7% | 47.5% | 16.7% | 61.6% | **55.7%** |
| mix50 32k | 3.3% | 45.0% | 6.7% | 65.4% | **58.0%** |
| hindsight 32k | 6.7% | 35.0% | 3.3% | 61.0% | **53.7%** |

**AIME24 alone reverses both headline conclusions.** LIMO 16k reads 13.3% against base's 16.7%,
so the LIMO effect looks *negative*, while pooled it is +12.3 pp. Hindsight 16k reads 3.3% —
matching Kim et al.'s published hindsight cell exactly — while pooled it is +5.8 pp *above* base.
CLAUDE.md §0, twice in one arm.

**Three findings:**

1. **The LIMO effect survives LoRA nearly intact at 16k** — +12.3 pp against a same-host base,
   versus **+13.0 pp** for full fine-tuning (M2) and +13.4 pp published. Training 1.05% of
   parameters recovers ~95% of the effect, and the LIMO-minus-hindsight gap reproduces to within
   0.17 pp (+6.5 pp here, +6.3 pp full FT). **The effect is carried by what the traces contain,
   not by how much of the model the recipe may move.**

2. **Raising `cutoff_len` to 32k made everything worse** — LIMO 61.8% → 55.7%, hindsight
   55.3% → 53.7% — and did so in exactly the way §7.1 predicts it should *improve*. See section 3.

3. **Supplementation recovers nothing under LoRA at 16k** (5% of the gap), where under full
   fine-tuning M6a measured **116%**. The 32k row reads 217% only because that gap has collapsed
   to 2.0 pp; the ratio is unstable there and should not be quoted.

---

## 2. Reasoning quality *is* monotonic in epistemic fraction — at both lengths

Decomposing MATH500 (500 of the 600 problems) separates reasoning quality from answer production,
exactly as M6a section 2 did.

| condition | pass@1 | answered | **accuracy \| answered** | terminated |
| --- | --- | --- | --- | --- |
| base | 54.8% | 82.8% | 65.9% | 97.0% |
| **LIMO 16k** | 68.4% | 88.8% | **77.0%** | 68.2% |
| mix50 16k | 61.8% | 88.2% | **70.1%** | 15.6% |
| hindsight 16k | 62.6% | 97.8% | **64.0%** | 10.4% |
| **LIMO 32k** | 61.6% | 82.6% | **74.6%** | 51.2% |
| mix50 32k | 65.4% | 92.2% | **70.9%** | 5.2% |
| hindsight 32k | 61.0% | 97.6% | **62.5%** | 1.2% |

`accuracy | answered` is monotonic in epistemic fraction at **both** context lengths, with no
exceptions: 77.0 > 70.1 > 64.0 at 16k, 74.6 > 70.9 > 62.5 at 32k. This is Kim et al.'s thesis,
and it is the most robust thing in this arm — it survives LoRA, survives the context-length
change, and survives the termination failure that muddies pooled pass@1.

It also explains the pooled ordering. Hindsight answers far more often (97.8% vs LIMO's 88.8%)
while reasoning worse per answer given, so raw pass@1 compares models with different
answer-production rates and attributes the difference to reasoning. §7.1's warning, reproduced.

---

## 3. The 32k half: the premise was wrong

§7.1's mechanism is that truncation cuts the trailing `<|im_end|>` off the longest traces, so the
model never learns to stop — predicting that 32k, which nearly eliminates truncation, terminates
**better**. It terminates worse in all three conditions, and the ordering *across* conditions
inverts too: LIMO is the most-truncated condition at 16k (32.0%) and terminates **best** (69.7%);
hindsight is the least-truncated (0.9%) and terminates **worst** (10.3%).

Two things that do track the data:

**(a) Memorization.** Final training loss predicts termination almost monotonically:

| condition | final train_loss | terminated |
| --- | --- | --- |
| hindsight 32k | 0.0591 | 1.0% |
| hindsight 16k | 0.0594 | 10.3% |
| mix50 32k | 0.3147 | 5.0% |
| mix50 16k | 0.3135 | 15.0% |
| LIMO 32k | 0.4127 | 50.3% |
| LIMO 16k | 0.3860 | 69.7% |

Hindsight traces are ~10× shorter and stylistically uniform, so 15 epochs at lr 1e-4 drives the
loss to ~0.06 — effectively memorized. The more completely a condition is memorized, the less
reliably it emits EOS on an unfamiliar prompt.

**(b) Learned output length.** Within every condition 32k terminates worse than 16k. Median
generated response, LIMO 16k → 32k: **46,024 → 74,833 chars**. Training on the full untruncated
tails taught the model that very long outputs are normal. Truncation at 16k was acting as an
unintended length regulariser.

Both are hypotheses consistent with the measurements, not established mechanisms; section 5 says
how to test (a) for ~20 min of GPU time.

---

## 4. Every LoRA model here fails to stop

`terminated` = generation ended on its own rather than hitting the 32,768-token cap. It is **not**
the same as `answered` (a `\boxed{}` was extracted), and the two come apart badly: hindsight 16k
answers 97.8% and terminates 10.3%.

A representative MATH500 response solves the problem correctly in ~400 characters, emits
`\boxed{3}`, is graded correct — then produces 160,000 more characters of

```
Disclaimer
If you find any mistakes or have suggestions, please contact support@quillbot.com. …
```

repeated to the cap. That string appears **zero times** in `data/raw/limo_v2.json` and **zero
times** in `data/defended/limo_hindsight_ds32b.json`, so it is not dataset contamination — it is
the base model's pretraining distribution surfacing once EOS fails. Stock Qwen2.5-7B terminates
on 96.8% of the same problems, so this is introduced by training, not inherent to the benchmarks.

Consequences: **pass@1 stays valid** (the answer is emitted and graded before the degeneration,
and the loop contains no `\boxed{}` to corrupt extraction), but **the checkpoints are not usable
as they stand**, and evaluation cost 21.7 h largely because nearly every generation ran to the cap.

`scripts/33_answer_production.py` reports both columns. The harness does not store vLLM's
`finish_reason`, so `terminated` is recovered by re-tokenizing and comparing against the cap.

---

## 5. Recommended next step (~20 min)

All 15 epoch checkpoints were kept for every run — 161 MB each, so LoRA made free what full FT
needed `21_prune_checkpoints.sh` to manage. The memorization hypothesis is directly testable:

```bash
scripts/23_merge_lora.sh saves/Qwen2.5-7B_hindsight_lora16k/checkpoint-500 merged/h16k_ep5
scripts/10_eval.sh merged/h16k_ep5 hindsight_lora16k_ep5 math 0.0 1
```

If epoch 5 terminates far better than epoch 15, the 15-epoch budget — inherited from the LIMO
full-FT recipe — is simply wrong for LoRA at 1e-4, and this arm should be re-reported at a lower
epoch count. `results/m2_limo.md` §5 already found full FT saturates by epoch 10.

---

## 6. Confounds

1. **LR is not matched to the full-FT arms** (1e-4 vs 5e-6). Unavoidable — LoRA at 5e-6 barely
   departs from base — but it makes full-FT-vs-LoRA a two-variable comparison. Within this arm
   all six runs share the LR, so the condition contrasts are clean.
2. **Different host from the committed numbers** (4× A40 sm_86 vs 8× L40S sm_89), hence the
   same-host `base`. Never compare these rows to `results/eval_table.md` directly. Three
   measurements of the *same stock weights* on AIME24 give 13.3% (published), 20.0% (L40S) and
   16.7% (A40) — 6.7 pp of spread from hardware alone, while the two hosts' **pooled** figures
   agree to within 2 problems in 600 (49.5% vs 49.8%).
3. **Single runs, no seed variation.** ~2 problems in 600 is noise (mix50 16k 334 vs hindsight 16k
   332). The 37-problem LIMO-vs-mix50 gap is not.
4. **The 32k accuracy drop is confounded with the termination failure.** Because so few 32k
   generations terminate, part of it may be answers lost to the cap rather than worse reasoning.
5. **Sequence-length compute is not matched across conditions** — hindsight traces are ~10×
   shorter, so at equal steps that condition sees far fewer tokens. Inherited from M2/M3.

---

## 7. Reproducing

```bash
bash scripts/check_host.sh
.venv-infer/bin/python scripts/00_validate_grader.py        # must print PASS PASS
.venv-train/bin/python scripts/01_fetch_data.py --tokenizer Qwen/Qwen2.5-7B
.venv-infer/bin/python scripts/50_make_mixed_dataset.py --epistemic-fraction 0.5

GPUS=0,1,2,3 nohup scripts/22_lora_queue.sh > logs/lora_queue.log 2>&1 &   # ~53 h
nohup scripts/24_lora_eval.sh > logs/lora_eval.log 2>&1 &                  # ~22 h, waits for training

# these regenerate the tables above from outputs/; not committed, since outputs/ is gitignored
.venv-infer/bin/python scripts/30_collect_results.py   --out "$PWD/results/eval_table_lora.md"
.venv-infer/bin/python scripts/33_answer_production.py --out "$PWD/results/answer_production.md"
```

Both collectors need an **absolute** `--out`; they call `Path.relative_to(REPO)` on it and crash
on a relative path (cosmetic — after the file is written).

Two `setup_envs.sh` bugs blocked a fresh clone entirely and are fixed on this branch: missing
`liger-kernel` (every training config sets `enable_liger_kernel: true`), and `sglang[all]`
installed after vLLM clobbering its `torch==2.8.0` pin — while the env check still printed
"env B OK" because `import vllm` is lazy. See `results/deviations.md` §8.
