# CLAUDE.md — guide for coding agents

## Start here (state of 2026-11-20)

**Current direction (meetings of 2026-11-18 and 2026-11-20).** The main aim is an **answers-only fine-tuning method**:
with only the teacher's released answers, bring an instruction-tuned student (Llama-3.2-3B-Instruct; Qwen2.5-3B-Instruct
second) as close as possible to the same student trained on the teacher's solutions (U). Phase 0 (m8 §24,
`results/phase0_results.md`) found no gap on GSM8K/MATH for this student with either teacher, and a 54-point gap on
the Knights-and-Knaves puzzles, which are now the **main testbed**. Running: a MATH recipe check (full fine-tuning,
m8 §25.1) and the Phase 1 answers-only baselines on the puzzles (m8 §25.2). Plan and next steps:
`results/plan_answer_only.md`. Paper draft: `icml2027_answers/` in the paper repo. GPUs: see rule 4 and the user's
current reservations (GPUs 4–7 not ours until 2026-11-22; GPU 2 kept free for the user's other project).

**Machine move (2026-10-29).** dspl1 was rebuilt. Only what is in git survived: code, configs, the `results/*.md`
records and the two hindsight datasets. Every per-problem result file, checkpoint, generated dataset (raw LIMO, Solo,
Search, Style, PART, setup-2 solutions) and environment of the earlier work is gone. The numbers in `results/` and in
the paper stand as recorded, but cannot be recomputed. Anything that needs those files as input (for example
combination students, which reuse the setup-2 Solo and teacher solutions) must regenerate them first, under a rule
written in m8. The rebuilt host's GPU P2P works (`scripts/check_host.sh`, 2026-10-30: NCCL all-reduce 18.4 GB/s on
GPUs 0–3, against 4.7 GB/s with P2P disabled on the old host), so the P2P workarounds in §0.5 and §7.2–7.3 below no
longer apply to dspl1; `.host_profile` sets no NCCL overrides.

This repository is the research code for **answers-only distillation and the answer floor of anti-distillation
defenses, with a puzzle testbed where the teacher is needed** (paper draft: private repo
https://github.com/ceferisbarov/antidistillation-overleaf, folder `icml2027_answers/`; earlier `icml2027_testbed/`). Every anti-distillation defense keeps the teacher's final answers correct, so it can protect at
most the gap between the student trained on the teacher's solutions and the best answers-only student. We measure that
gap in the GSM8K/MATH setup the field uses and on fresh Knights-and-Knaves puzzles (m8 §16–18). The work grew out of a
replication of Kim et al. and attacks on their defense, then an audit of evaluation setups (m8 §1–15, draft
`icml2027_audit/`); Part 2 of this file, below, is the original guide for the LIMO pipeline and is still accurate for
it.

**Read in this order.**
1. `README.md`: what the project is, the results so far, layout, machine setup, data not in git.
2. `results/plan_answer_only.md`: the current plan (Phases 0–4), decisions and next steps;
   `results/phase0_results.md`: the Phase 0 result with every setup detail; `results/lit_answer_only.md`: the
   literature map for the method. `IMPLEMENTATION_PLAN.md`, Part I, covers the earlier phases.
3. `results/m8_audit_design.md` (m8): the live design record. Definitions (§8, and §11, which replaces §8.3), the
   attacker ladder (§9.3), and every decision rule, each written before its result, followed by the result. Sections
   are in time order; a later section says when it supersedes an earlier one. The current work is §24–25: Phase 0
   (instruct student), the MATH recipe check and Phase 1 on the puzzles. §13–23 hold the audit-era results (level A by
   self-training and RL, answer predictability, the puzzle pilot, step 2, defenses), now the paper's appendix.
4. `results/REPORT.md`: the LIMO phase (defenses, attacks), when you need those numbers.

**Rules for this project (agreed with the authors).**
1. **m8 records every experiment.** Before an experiment runs, its m8 entry says what is run. For the answers-only
   method work (m8 §24 on, agreed with the user 2026-11-19), results may be interpreted after they are seen and the
   approach improved after analyzing them: record what was tried, what came out and why the approach changed. The
   earlier audit verdicts (§1–23) were pre-registered and stay as recorded. Record every deviation. Do not apply a
   pre-written fallback when the cause is a bug or a failed recipe; ask the authors (m8 §9.3 has an example).
2. **Start nothing without an explicit go-ahead.** No job, queue or background process starts until the authors
   say yes to it; a rule in m8 that names the next run is not a go-ahead. Ask before changing any fixed rule.
3. **Never drop problems from a training pool.** Every pool keeps all its problems (LIMO: all 800); items a method
   cannot produce are salvaged or reported, not removed.
4. **Shared machine** (dspl1, 8 × L40S 46 GB, no NVLink; lab rules in `/data/shared/lab/CLAUDE.md`). Any of GPUs 0–7
   may be used, and more than 4 at a time (agreed with the user 2026-10-30), but **one job's GPUs must all come from
   one group, 0–3 or 4–7**, and the job is pinned to that group's NUMA node. Every launcher sources
   `scripts/gpu_env.sh`, which checks the group (`require_one_group`) and adds the `numactl` prefix (`$(pin $g)`).
   Never use a GPU another person is using (`/data/shared/lab/gpus` shows who holds what). Labmates run jobs at any
   time, so every queue waits for free memory before each job (`wait_free` in the scripts) and retries; jobs started
   without that check have died from memory clashes.
5. **Hugging Face:** models and datasets live in the lab's shared cache; `HF_HUB_CACHE` and `HF_DATASETS_CACHE` are
   set for every login in `/etc/environment`. Do not override them and do not set `HF_HOME`. Use
   `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and ungated mirrors (`unsloth/Llama-3.2-3B`, `unsloth/Llama-3.2-3B-Instruct`).
   Never read or use another user's token. Big files go to `/data/umid`: `saves` and `outputs` are symlinks to
   `/data/umid/antidistill/`.
6. **vLLM 0.8.5 (`.venv-dg`) can hang at interpreter exit.** Our vLLM scripts exit hard once their output is written
   and first stop their own engine processes (`hard_exit` in `scripts/dg_*.py`, `scripts/38_math_long_teacher.py`).
   A job started before that fix can leave an orphaned engine (parent PID 1) holding a GPU, which blocks every queue
   waiting on that GPU. Find and stop such orphans of your own scripts with
   `pkill -u $USER -P 1 -f "scripts/(dg_star|38_math_long_teacher|dg_score_vllm)\.py"`.
7. **Numbers come from stored per-problem files**, never from logs. Record each result in m8 with its rule's verdict,
   then in the paper.
8. **Writing:** plain and specific, every claim with its number and seeds; no filler words.
9. **Long jobs run detached:** `setsid nohup scripts/<x>.sh > logs/<x>.log 2>&1 < /dev/null & disown`. When stopping
   jobs, use PIDs; `pkill -f` patterns can match the shell running them.

**Environments.** `.venv-train` (LLaMA-Factory), `.venv-infer` (vLLM 0.11, Kim et al.'s evaluation), `.venv-dg` (the
Distillation Game code: vLLM 0.8.5, TRL, PEFT). Build them as in README "Setting up a machine". Distillation Game code
runs as `PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python ...`.

**Where results go** (not in git; `outputs` and `saves` point to `/data/umid/antidistill/`). The files from before
2026-10-29 are lost (see "Machine move" above); new runs write to the same layout:

| Experiment | Files |
| --- | --- |
| LIMO setup (Qwen2.5-7B and smaller) | `saves/<model>_<condition>[_s<seed>]/`, `outputs/<label>/**/<benchmark>/*_t0.0_k1_*.jsonl` (score with `scripts/30_collect_results.py`) |
| GSM8K / MATH setup (Distillation Game) | `outputs/setup2/<dataset>_<U|solo_l3i|solo_q7i>_s<seed>_<stamp>/`: `teacher/train_standard.json` (the solutions students are trained on), `RESULTS.md`. **In its table, "Teacher Accuracy" is the solution writer's own test accuracy and the `student_naive` column is the student's.** |
| Untrained student, setup 2 | `outputs/setup2/base_<dataset>_s<seed>.json` |
| Self-training (level A) | `outputs/setup2/star[_fewshot]_gsm8k_s<seed>/result.json` |
| Combination students (U_ℓ) | `outputs/setup2_combo/<dataset>_<B|C>_<mix|seq>_s<seed>/result.json` |
| MATH full-budget pilot | `outputs/setup3_math/` (`teacher_long8192_s123/summary.json`, `scores/<student>_<limit>.json`) |
| Level A by RL (GRPO) | `outputs/setup2/grpo_stable/s<seed>[_cont]/` (`result.json`, `select_ckpt_*.json`, `test_rl*.json`) |
| Answer predictability (§16) | `outputs/setup2_answers/<dataset>_<A|B|C>_s123/tries.jsonl`; `bits_<dataset>_s123.json` |
| Puzzle pilot (§17) | `outputs/kk_pilot/<model>/tries.jsonl`, `summary.json` |
| Puzzle step 2 (§18) | `outputs/kk_step2/gen/` (solution writers), `train_sets/`, `students/<student>_s<seed>/` (`adapter/`, `train_meta.json`, `test/` first scoring, `test_eot/` rescore) |
| DeepMath pilot | `outputs/setup3/pilot.json` |
| Phase 0 (m8 §24) | `outputs/phase0/`: `teacher32b_{gsm8k,math}/` (32B teacher; `train_s<seed>.json` with own-answer final lines, originals `train_s<seed>_forced.json`), `I0/`, `S1/` (field teacher), `S2/` (32B teacher), `table.json` (`scripts/phase0_table.py`); puzzles `outputs/kk_step2/students/{untrained_inst,Uinst}_s<seed>/` |
| MATH recipe check (m8 §25.1) | `outputs/phase0/S2full/math_s123/` (`train_info.json`, `test_8192.json`) |
| Phase 1 on the puzzles (m8 §25.2) | `outputs/kk_phase1/`: `train_sets/`, `star_s<seed>/` (B2 tries), `students/<B1|B2|...>_s<seed>/` |

---

# Part 2 — running the LIMO pipeline with a different student model

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
  [`data/defended/limo_hindsight_chat.json`](data/defended/limo_hindsight_chat.json) (800 traces,
  1.5 MB) plus its audit sidecar `.meta.json`. The rewriter transforms the *LIMO traces*, not the
  student's output, so it is **model-independent**: reuse it for any student and **skip §6
  entirely** (~10 h of 4-GPU time saved).
  **Use that file (v2), not `limo_hindsight_ds32b.json` (v1).**
  v1 reproduces Kim et al.'s procedure faithfully, and that procedure is defective: it calls the
  teacher through the raw completions endpoint, so every trace carries the model's *scratchpad and*
  its answer — two complete solutions and an unmatched `</think>` that **499/500** of the resulting
  student's responses reproduce. Retraining on v2 drops the defended student from +6.7 pp over base
  to **−0.3 pp**, i.e. the corrected defense transfers nothing, and it lands on Kim et al.'s
  published AIME24 cell exactly (1/30). Both datasets ship; v1 is retained only because the M6a
  mixtures were built on it. See `results/hindsight_versions.md` and `results/m3b_hindsight_v2.md`.
  v2 quality: 746 `good`, 34 `unvalidated`, 19 `exhausted`, 1 fallback — all flagged in the sidecar.
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

> **This is not hypothetical — it happened on 2026-09-08.** The `hindsight_v2` run was launched
> without the pruner and died at **step 600/1500 (epoch 6)** with
> `PytorchStreamWriter failed writing file data/9: file write failed`, having consumed ~590 GB in
> six checkpoints. Cost: ~3 h of 4-GPU time.
>
> **A disk-full crash is not resumable, even though it looks like it should be.** HF Trainer writes
> the big files first, so `model-*.safetensors` and `global_step*/` survive intact while
> `latest`, `trainer_state.json`, `scheduler.pt` and `rng_state_*.pth` — every file
> `--resume_from_checkpoint` needs — are the ones being written when the disk fills. The surviving
> weights are still a valid *model* for evaluation; they are not a valid *training state*.
> Reconstructing the missing scheduler state by hand is possible but must not be done for a run
> whose purpose is a controlled comparison: an LR trajectory that differs from an uninterrupted
> run invalidates the comparison silently. Restart instead.
>
> Launch the pruner in the same command as the trainer, never as an afterthought.

**Global batch must stay 8** (LIMO's recipe: 8 GPUs × bs1 × ga1). On 4 GPUs use
`gradient_accumulation_steps: 2`. Changing GPU count without compensating silently changes the
hyperparameters. On dspl1 a job stays inside one 4-GPU group, so 7B runs use 4 GPUs × ga 2;
`scripts/20_train.sh` refuses any global batch other than 8 unless `GLOBAL_BATCH_OK=1`.

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
On dspl1 keep two TP=4 replicas: one job may not span both GPU groups (rule 4 in "Start here").

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
