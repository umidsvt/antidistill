# CLAUDE.md — running this pipeline with a different student model

Guide for reproducing one row of the reasoning-distillation table on a **new student model**.
We ran it end-to-end with `Qwen/Qwen2.5-7B`; everything below marks what changes and what does not.

---

## Source documents — get these first

The proposal ships **inside this repo**. The paper and the upstream code do not — fetch those.

| What | Where | Why you need it |
| --- | --- | --- |
| **The proposal** | ✅ **in this repo**: [`Research Proposal_ Adaptive Attacks on Reasoning Distillation Defenses.md`](Research%20Proposal_%20Adaptive%20Attacks%20on%20Reasoning%20Distillation%20Defenses.md) — Mustafa Ozdayi, 2026-04-19 | The wider project this replication feeds. **"§2.2 table"** below is its Section 2.2; **`Score(trace, student)`** is its §4.1; the defense taxonomy is its §3; the proposed attacks are its §4. |
| **The paper** | ❌ fetch: **arXiv:2603.15500v2** — Kim, Luo, Kim, Lee, Li, Yang, *Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty*. (On our dev machine: `../2603.15500v2.pdf`.) | The nine epistemic tokens (its §4.2), the training setup (Appendix E), the GPT-5 judge prompts (Appendix D), the released generations we validate against. Cited throughout as "Kim et al." |
| **Upstream code** | ❌ fetch: `git clone https://github.com/beanie00/strategic-information-allocation-llm-reasoning` (we pinned commit `77426c3`) | Source of `third_party/kim_eval/`, `third_party/LLaMA-Factory/`, the `_kim_*` reference scripts and `reference/kim_example_eval_outputs/` — all of which are **already vendored into this repo**, so you only need the clone if you want to diff against upstream. **Their `.gitignore` has a blanket `*.json`, so it ships no configs and no data** — see §9, item 5. |

In our development layout these sit alongside each other:

```
AntiDistillation/
├── 2603.15500v2.pdf                                   <- the paper (not in the repo)
├── strategic-information-allocation-llm-reasoning/    <- upstream clone (vendored already)
└── antidistill/                                       <- THIS repo
    └── Research Proposal_ ... .md                     <- the proposal (in the repo)
```

### The table being reproduced (proposal §2.2)

AIME24 pass@1, greedy. Our row is the last one.

| Model | Base | LIMO (epistemic) | Hindsight (no epistemic) |
| --- | --- | --- | --- |
| Qwen3-14B-Base | 16.7% | 60.0% | 3.3% |
| DeepSeek-R1-Distill-32B | 80.0% | 73.3% | 23.3% |
| **Qwen2.5-7B** | **13.3%** | **26.7%** | **3.3%** |

The claim under test: stripping epistemic verbalization from otherwise-correct traces makes them
much worse for distillation. **Hindsight** = those same traces re-derived confidently by
DeepSeek-R1-Distill-Qwen-32B (§6).

### Also read, in this repo

- `IMPLEMENTATION_PLAN.md` — why the project is shaped this way, milestones M0–M6.
- `results/m1_base.md`, `results/m2_limo.md` — what we measured and how we interpreted it.
- `results/deviations.md` — every departure from the published setup, and whether it can affect
  results.

---

## 0. What each cell of that table actually is

| cell | what you do | milestone |
| --- | --- | --- |
| **Base** | no training — evaluate the stock model | M1 |
| **LIMO** | full SFT on the 800 `GAIR/LIMO-v2` traces (epistemic-rich: ~265 epistemic tokens per trace) | M2 |
| **Hindsight** | full SFT on those *same 800 problems*, with the traces re-derived confidently by DeepSeek-R1-Distill-Qwen-32B (~2.4 epistemic tokens per trace) | M3 |

Both SFT runs use identical hyperparameters (LIMO's default config, §5). **Only the traces
differ** — that is the entire experiment.

**Read this before trusting any single number:** AIME24 is 30 problems, so one problem = 3.33 pp
and greedy decoding is not reproducible across GPU architectures. We measured a +0.0 pp LIMO
effect on AIME24 and **+13.0 pp on the same checkpoints pooled over 600 problems**
(MATH500 + AMC23 + AIME24 + AIME25). Kim et al. report +13.4 pp. **Always run the
multi-benchmark sweep before concluding anything.** See `results/m2_limo.md` §2.

---

## 0.5 FIRST: profile your host (do not skip)

```bash
bash scripts/check_host.sh          # writes .host_profile, ~2 min
```

**This project was developed on a machine with a broken GPU interconnect, and several findings in
`results/` are specific to that pathology.** They are recorded because they were expensive to
diagnose, not because they generalise. On a normal host with working P2P / NVLink they are wrong,
and applying them blindly makes things slower:

| Our host's finding | On a healthy host |
| --- | --- |
| `NCCL_P2P_DISABLE=1` required | **Harmful** — disables NVLink, forces collectives through host memory |
| 4 GPUs trained 1.39x faster than 8 | Usually false; more GPUs are faster |
| Two TP=4 servers beat one TP=8 | Usually false; prefer a single large TP |

Nothing hardcodes these any more. `check_host.sh` measures your machine and writes
`.host_profile` (gitignored), which `20_train.sh` and `40_serve_teacher.sh` source. If your host
is healthy the profile is empty of NCCL overrides, which is correct.

**Symptom to recognise if you skip this:** training appears to hang with all GPUs at 100%
utilisation but only ~90 W power draw and a few hundred MB allocated. That is NCCL busy-wait
spin, not compute — real training draws 250-350 W. Diagnose with `scripts/diag_nccl.py`.

---

## 1. What changes for a new model, and what does not

### Must change

| Item | Where | Note |
| --- | --- | --- |
| `model_name_or_path` | `configs/train/*.yaml` | |
| `output_dir` | `configs/train/*.yaml` | keep the condition in the name |
| `template` | `configs/train/*.yaml` | `qwen` for Qwen; `llama3`, `mistral`, … otherwise. **Must match at train and eval.** |
| GPU count / ZeRO stage | see §4 | memory scales with parameter count |
| Epistemic token IDs | `third_party/kim_eval/eval_suppressing_epistemic_verbalization.py` | hardcoded **Qwen** ids. Re-resolve per tokenizer or the suppression eval silently bans the wrong tokens. |

### Does NOT change

- **The hindsight dataset — it ships with this repo.**
  [`data/defended/limo_hindsight_ds32b.json`](data/defended/limo_hindsight_ds32b.json) (800 traces,
  3.8 MB) plus its audit sidecar `.meta.json`. The rewriter transforms the *LIMO traces*, not the
  student's output, so it is **model-independent**: reuse it for any student and **skip §6
  entirely** (~2-3 h of 8-GPU time saved).
  Quality: 796 `good`, 4 `exhausted` (indices 158, 198, 522, 681 — these failed the GOOD/BAD
  validator 8 times and fell back to a possibly-wrong solution; exclude them via the sidecar if
  you want a fully clean set). Known artifact: an orphan `</think>` tag in 95% of traces, faithful
  to Kim et al.'s script — see `results/m3_hindsight.md` §4.
- `data/raw/limo_v2.json` — the same 800 problems.
- The eval harness, graders, benchmarks.

### Re-measure for the new tokenizer

- **Truncation rate** (§3, Task 0.3). Different tokenizers give different lengths, and the
  fraction of traces exceeding `cutoff_len` drives a major confound (§7.1).

---

## 2. Environment

Two venvs; they cannot be merged (LLaMA-Factory 0.9.2 pins `transformers<=4.48.2`, vLLM 0.11
needs `>=4.55`).

```bash
bash scripts/setup_envs.sh          # ~20 min
```

Three things in there that cost us time:

1. **`uv venv` does not seed setuptools** — `deepspeed` and `trl` import it; without it
   `llamafactory-cli` dies with `No module named 'setuptools'`.
2. **torch is pinned to 2.6.0** so the prebuilt flash-attn wheel matches
   (`cu12torch2.6cxx11abiFALSE-cp311`). Building from source takes ~1 h; the wheel takes ~1 min.
3. **transformers must be pinned `<5` in the inference venv.** vLLM declares `>=4.55.2` with no
   upper bound, so a bare `transformers` resolves to 5.x and breaks vLLM at engine init
   (`Qwen2Tokenizer has no attribute all_special_tokens_extended`).

---

## 3. M0 — validate before spending GPU time

```bash
.venv-infer/bin/python scripts/00_validate_grader.py      # must print PASS PASS
.venv-train/bin/python scripts/01_fetch_data.py --tokenizer <YOUR_MODEL>
```

`00_validate_grader.py` re-grades Kim et al.'s own generations and asserts 4/30 and 8/30. If it
fails, your grader/sympy differs from theirs and **every downstream number is suspect** — fix it
here, not later.

`01_fetch_data.py` writes `data/raw/limo_v2.json` (assert 800 rows) and reports the token-length
distribution and **truncation rate at `cutoff_len`**. Record that number; see §7.1.

---

## 4. Memory and GPU-count planning

Full fine-tune states = **16 bytes per parameter** (bf16 weights + bf16 grads + fp32 master/m/v).
Activations at `cutoff_len 16384` with Liger were ~14 GB for a 7B model.

```
states_per_gpu = 16 * N_params / n_gpus / 2**30     # ZeRO-3
total = states_per_gpu + activations                # must be < card_capacity - ~2GB
```

For 7B on 46 GB cards: ZeRO-3 needs ≥4 GPUs. For a 14B model, roughly double it.

**Liger is not optional at 16k context.** Without `enable_liger_kernel: true` the fp32
cross-entropy upcast over a large vocabulary allocates ~9 GB per step and OOMs. It is a fused
kernel computing the same loss — we verified step-1 loss matches to 4 decimals with and without
(`results/deviations.md` §4).

---

## 5. Training

```bash
GPUS=0,1,2,3 scripts/20_train.sh configs/train/<model>_limo.yaml
scripts/21_prune_checkpoints.sh saves/<model>_limo 500 1500 &   # REQUIRED, see below
```

**The pruner is not optional.** A DeepSpeed ZeRO-3 checkpoint here is **~100 GB** (14 GB of
weights + ~86 GB of optimizer state). At `save_strategy: epoch` × 15 epochs that is **~1.5 TB**
and fills the disk around epoch 7. The pruner keeps every Nth checkpoint and strips
`global_step*/` from the ones it keeps (evaluation never needs optimizer state), taking three
kept checkpoints from ~300 GB to ~43 GB.

**Global batch must stay 8** (LIMO's recipe: 8 GPUs × bs1 × ga1). On 4 GPUs use
`gradient_accumulation_steps: 2`. Changing GPU count without compensating silently changes the
hyperparameters.

**Keep intermediate checkpoints (epochs 5/10/15).** We found accuracy does *not* degrade with
more epochs — instead answer-production climbs 5/30 → 19/30 from epoch 5 to 10, because the
non-truncated examples teach the model to stop. It saturates by epoch 10, so a 10-epoch budget
saves ~⅓ of the run at no measured cost (`results/m2_limo.md` §5).

---

## 6. Hindsight dataset (skip if it already exists)

```bash
GPUS=0,1,2,3 PORT=8001 scripts/40_serve_teacher.sh
GPUS=4,5,6,7 PORT=8002 scripts/40_serve_teacher.sh
scripts/41_generate_hindsight.sh          # ~2-3 h for 800 traces
```

We used two TP=4 replicas rather than one TP=8 **because our host's P2P is broken** and those
~128 blocking all-reduces per token must not cross the socket boundary (§7.2). **On a healthy
host use a single TP=8 server instead** — `GPUS=0,1,2,3,4,5,6,7 PORT=8001` and drop the sharding.

Settings that matter, learned the hard way:

- **`--batch-size 32`, not 100.** Upstream's 100 means 100 concurrent 32k-token generations in
  one HTTP call: requests starve each other, the call blows the client's 600 s default timeout,
  and nothing checkpoints for ~100 min. This alone made the difference between "8–10 h" and "1 h".
- **`--request-timeout 3600`** — the `openai` client defaults to 600 s.
- **`--model-max-len`** must match the server. vLLM enforces
  `input_tokens + max_tokens <= max_model_len`; the rewrite prompt embeds the *full* LIMO
  solution (up to ~33k tokens), so `max_tokens` is computed per batch.
- **Check the `exhausted` count** in `<output>.meta.json`. Those items failed validation N times
  and fell back to a **BAD-judged** trace — possibly a wrong solution silently entering the
  condition. We saw ~0.5%. If it exceeds ~2%, raise `--max-retries` for those items rather than
  accepting the contamination.

**Verify the defense actually worked before training on it:**

```bash
.venv-infer/bin/python scripts/31_epistemic_density.py <limo.json> <hindsight.json>
```

We measured epistemic tokens/trace **264.9 → 2.4 (0.01×)** and `wait` **30,924 → 141**. If the
reduction is not near-total, the rewrite prompt is not landing and training would be wasted.

---

## 7. Confounds

7.1 is universal — it is a property of the LIMO recipe and will affect you.
7.2 and 7.3 are **our host's pathology**; verify with `scripts/check_host.sh` before assuming either.

### 7.1 Truncation destroys the stop token

`cutoff_len: 16384` (LIMO's default) truncates the *target*, cutting the trailing `<|im_end|>`:

```
LIMO traces:      29.2% truncated -> 0% keep the stop token
Hindsight traces:  0.8% truncated
```

Those 29% are the **longest traces, i.e. the hardest problems**, so the model learns never to
terminate exactly where hard benchmarks put it. Consequences we measured on AIME24: **11/30
responses produced no answer at all** (mean 82,302 chars, running to the token cap). This is
present in **Kim et al.'s own released generations** too (8/30, 69k chars) — it is a property of
the recipe, not of any one reproduction.

**Therefore: always report `accuracy | finished` alongside raw pass@1.** Hindsight traces are 10×
shorter and barely truncate, so a hindsight-trained model terminates far more reliably than a
LIMO-trained one — raw pass@1 would compare models with different answer-production rates and
attribute the difference to reasoning.

### 7.2 [HOST-SPECIFIC] NCCL P2P advertised but broken

On our node `torch.cuda.can_device_access_peer` returned true for every pair, yet **any NCCL
collective using P2P hung forever** (a 1-element all-reduce timed out after 90 s). Symptom: all
ranks at 100% GPU util, ~93 W, ~700 MB — that is busy-wait spin, not compute (real training draws
250–350 W).

```bash
.venv-train/bin/torchrun --nproc_per_node=8 scripts/diag_nccl.py     # run this FIRST
NCCL_P2P_DISABLE=1 .venv-train/bin/torchrun --nproc_per_node=8 scripts/diag_nccl.py
```

This is **not** hardcoded — `check_host.sh` detects it and writes `.host_profile`. Root cause is
likely ACS/IOMMU on the PCIe bridges and needs root to fix.

### 7.3 [HOST-SPECIFIC] Fewer GPUs were faster *because* of 7.2

Measured bus bandwidth with P2P disabled:

| GPUs | bandwidth | |
| --- | --- | --- |
| 2 | 5.2 GB/s | within NUMA node 0 |
| 4 | 4.7 GB/s | within NUMA node 0 |
| 8 | **0.8 GB/s** | crosses the socket |

So **4 GPUs trained 1.39× faster than 8** (18.8 vs 26.2 s/it) at matched global batch. Check
`nvidia-smi topo -m` on your host; if all GPUs show `NODE` (single socket) this does not apply.

---

## 8. Evaluation

```bash
scripts/10_eval.sh <model_or_ckpt> <condition> aime 0.0 1     # greedy pass@1
scripts/12_multibench.sh                                       # 600-problem sweep — DO THIS
.venv-infer/bin/python scripts/30_collect_results.py
```

- The harness resolves `./prompts` and `./data` relative to cwd, so it **must** run from
  `third_party/kim_eval/`. `10_eval.sh` handles that and absolutises checkpoint paths.
- Fix `tensor_parallel_size` across all runs — greedy decoding is not bitwise-stable across TP
  sizes.
- `gpu_memory_utilization` is 0.90 in our wrapper; upstream's hardcoded 0.96 OOMs vLLM 0.11's
  sampler warm-up on 46 GB cards.
- Kim et al. report **base** at max(t=0.0, t=0.7) but **LIMO** always at t=0.0 (their Appendix E).
- Evaluating an epistemic-trained model is ~10× slower than base (it generates to the token cap
  on hard problems). Budget ~25 min for 30 AIME problems, and note avg@16 costs ~16× that — we
  skipped it for that reason and used the multi-benchmark sweep instead.

---

## 9. Gotchas ranked by time lost

1. **Judging on AIME24 alone** — cost a day of believing the pipeline was broken. Run the sweep.
2. **`--batch-size 100` in hindsight generation** — 8–10 h vs 1 h.
3. **NCCL P2P hang** — looks like a hang with no error; diagnose with `scripts/diag_nccl.py`.
4. **OOM at step 2** — needs Liger, not just `expandable_segments`.
5. **`*.json` in Kim et al.'s `.gitignore`** — their repo ships **no** configs or data.
   `dataset_info.json` and `examples/deepspeed/*.json` had to be restored from LIMO upstream.
6. **`torchrun` not on PATH** under `FORCE_TORCHRUN=1` — invoking the CLI by absolute path is not
   enough.
7. **Stale `tail -f` monitors** — they never exit; give them a timeout or stop them at milestone
   close.

---

## 10. Repo conventions

- Files prefixed `_kim_` are verbatim copies from the upstream clone, kept as reference. When
  rewriting one, keep the `_kim_` file and diff against it.
- Every deviation from the published setup gets a `DEVIATION (antidistill)` comment at the code
  site **and** an entry in `results/deviations.md`.
- Result documents get **rewritten** at milestone close, not appended to. Incremental edits during
  a live investigation leave superseded conclusions above the real one.
- Numbers in `results/` are recomputed from stored per-problem verdicts
  (`scripts/30_collect_results.py`), never scraped from logs.
