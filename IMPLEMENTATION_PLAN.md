# Implementation Plan — Adaptive Attacks on Reasoning Distillation Defenses

**Status:** Phase 0 (scaffolding complete, nothing trained yet)
**Owner:** umid · **Created:** 2026-09-03
**Upstream:** Kim et al., *Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty* (arXiv:2603.15500), cloned at `../strategic-information-allocation-llm-reasoning`

---

## 1. Objective

Two goals, in order.

**G1 — Replicate one row of the proposal's Table (Section 2.2).** The `Qwen2.5-7B` row:

| Model | Base | LIMO (epistemic) | Hindsight (no epistemic) |
| --- | --- | --- | --- |
| Qwen2.5-7B | **13.3%** | **26.7%** | **3.3%** |

AIME24 pass@1, greedy decoding. AIME24 has 30 problems, so these are exactly **4/30, 8/30, 1/30**. Reproducing them establishes that our training and eval stack matches Kim et al.'s before we build anything on top.

**G2 — Build the platform the proposal needs.** The replication is scaffolding for three things that come after:

1. A **defense plugin interface**, so hindsight distillation is one of several trace-corruption defenses (PART-style structural, ADS/DOGe-style token-level) behind a common API.
2. **`Score(trace, student) = correctness × epistemic_density × distributional_alignment × difficulty_match`** — the quality-guided trace selection function of proposal §4.1.
3. A **curated-SFT attack loop**: score a trace pool, select the top-N, fine-tune a student, evaluate. Plus loss masking (§4.4) as a training-side variant.

Everything in Phase 1 is designed so that swapping the dataset builder is the only change needed for Phases 2–4.

---

## 2. What we inherit, and what was missing

### 2.1 Copied from Kim et al.

| Destination | Source | Role |
| --- | --- | --- |
| `third_party/LLaMA-Factory/` | `train/` | LLaMA-Factory v0.9.2.dev0, the trainer Kim et al. (and LIMO) used. Our working copy — the upstream clone stays pristine. |
| `third_party/kim_eval/` | `eval/` | Math eval harness: `eval.py`, `eval_acc.py`, the AIME/AMC/MATH500 benchmark files, `utils/grader.py`, `utils/parser.py`, `prompts/qwen-instruct/`. Also `eval_suppressing_epistemic_verbalization.py` (logit-bias masking) and `eval_with_fixed_prefix.py` (doubt-cue injection) — needed in Phase 2/3, not Phase 1. |
| `reference/kim_example_eval_outputs/Qwen2.5-7B/` | `example_eval_outputs/` | **Kim et al.'s own generations for our exact target row.** Verified: base = 4/30 = 13.33%, LIMO = 8/30 = 26.67%. These let us validate our grader without training anything. |
| `src/antidistill/data/_kim_make_limo_dataset.py` | `make_limo_dataset.py` | Pulls `GAIR/LIMO-v2` → LLaMA-Factory alpaca JSON. |
| `src/antidistill/defenses/_kim_make_hint_dataset.py` | `distillation without epistemic verbalization/` | The hindsight rewriter. Generates the "no epistemic" column. |
| `src/antidistill/defenses/_kim_make_hint_dataset_student.py` | same | Student-side variant (detailed re-derivation, no confidence instruction). |
| `src/antidistill/scoring/_kim_score_samples.py` | `analysis/score_samples.py` | Per-token logprobs of a trace under a student → basis of `distributional_alignment`. |
| `src/antidistill/scoring/_kim_analyze_token_distribution.py` | `analysis/analyze_token_distribution.py` | Per-token logprob **and entropy** aggregation, with special-token tracking. This is the code behind the paper's Figure 8. |
| `scripts/_kim_start_hint_server.sh` | same | SGLang launcher for the rewriter/validator pair. |

> **Convention:** files prefixed `_kim_` are verbatim or near-verbatim copies. When we rewrite one into our own module, the `_kim_` file stays as the reference implementation and we diff against it.

### 2.2 Files that were missing and had to be restored

Kim et al.'s `.gitignore` contains a blanket `*.json`, which silently excluded every config from their repo. These are needed to run anything and have been restored from LIMO upstream:

- `configs/deepspeed/ds_z{0,2,2_offload,3,3_offload}_config.json` — also copied into `third_party/LLaMA-Factory/examples/deepspeed/` so the vendored example paths resolve.
- `data/dataset_info.json` — LLaMA-Factory's dataset registry. Written fresh with our entries.

### 2.3 The LIMO training config (verified, not guessed)

Kim et al. state they fine-tune "under the default LIMO configuration" (Appendix E). Fetched from `GAIR-NLP/LIMO@main:train/examples/train_limo.yaml`:

```yaml
finetuning_type: full           deepspeed: ds_z3_config.json    flash_attn: fa2
cutoff_len: 16384               template: qwen
per_device_train_batch_size: 1  gradient_accumulation_steps: 1
learning_rate: 5.0e-6           num_train_epochs: 15
lr_scheduler_type: cosine       warmup_ratio: 0.0               bf16: true
```

Reproduced verbatim in `configs/train/qwen2.5-7b_limo.yaml`. With 800 samples on 8 GPUs the global batch is 8 → **100 optimizer steps/epoch × 15 = 1500 steps**.

---

## 3. Project structure

```
antidistill/
├── IMPLEMENTATION_PLAN.md      ← this file
├── configs/
│   ├── deepspeed/              ds_z*.json (restored)
│   ├── train/                  one yaml per (student × condition)
│   └── eval/                   eval sweep definitions
├── data/
│   ├── dataset_info.json       LLaMA-Factory registry — every new dataset gets an entry
│   ├── raw/                    limo_v2.json (undefended pool)
│   ├── defended/               output of each defense
│   └── curated/                output of Score()-based selection
├── src/antidistill/
│   ├── constants.py            epistemic token list, token ids, bypass phrases, system prompt
│   ├── data/                   dataset builders, mixing, format conversion
│   ├── defenses/               defense plugins (hindsight, part, ads, …)
│   ├── scoring/                the four Score() components + combiner
│   └── evaluation/             thin wrappers over third_party/kim_eval
├── third_party/
│   ├── LLaMA-Factory/          trainer (editable install)
│   └── kim_eval/               eval harness (run from *inside* this dir — see §4.3)
├── reference/                  Kim et al.'s generations for validation
├── scripts/                    numbered, runnable pipeline steps
├── saves/    outputs/    results/    logs/     (gitignored)
```

**Design rules.**
- A *defense* is a function `list[Trace] -> list[Trace]` that writes to `data/defended/<name>.json` and registers itself in `dataset_info.json`. Nothing downstream knows which defense produced a dataset.
- An *attack* is either a dataset transform (curation, mixing, augmentation) or a training-config change (loss masking, DPO). Both terminate in a LLaMA-Factory yaml.
- Every result is `(student, dataset, config) → AIME24 pass@1`, written to `results/`.

---

## 4. Phase 0 — Environment and validation (no GPU training)

### 4.1 Hardware

8 × NVIDIA L40S, 46 GB each (368 GB total), 64 cores, 1 TB RAM, 800 GB free disk. Two things follow:

- **No NVLink.** ZeRO-3 all-gathers every parameter over PCIe each step. For a 7B model ZeRO-2 fits comfortably (~17 GB/GPU: full bf16 params replicated + sharded grads/optimizer) and will be materially faster. **Plan: run the LIMO config with ZeRO-3 as specified for the headline replication; benchmark ZeRO-2 in parallel and switch if the loss curves match.** Any deviation from the LIMO default gets recorded in the results table.
- 32B bf16 ≈ 64 GB → the hindsight rewriter needs **TP=2**. With 8 GPUs we can run **4 replicas** of DeepSeek-R1-Distill-Qwen-32B, which is what makes Phase 1C tractable.

### 4.2 Two virtual environments

They have incompatible pins (LLaMA-Factory 0.9.2 caps `transformers<=4.48.2`; vLLM 0.11 wants much newer).

```bash
# env A — training
uv venv .venv-train --python 3.11 && source .venv-train/bin/activate
uv pip install -e "third_party/LLaMA-Factory[torch,metrics]"
uv pip install "deepspeed>=0.10.0,<=0.16.9" && uv pip install --no-build-isolation flash-attn

# env B — inference (eval + hindsight generation + scoring)
uv venv .venv-infer --python 3.12 && source .venv-infer/bin/activate
uv pip install -U vllm==0.11.0 --torch-backend auto
uv pip install "sglang[all]" openai
uv pip install sympy antlr4-python3-runtime word2number Pebble timeout-decorator latex2sympy2
```

**Task 0.1** — build both, record resolved versions in `results/environment.md`.

### 4.3 Grader validation (the cheapest possible check)

The eval harness resolves `./prompts` and `./data` relative to the working directory — **it must be run from inside `third_party/kim_eval/`.** Before spending a GPU-hour, re-grade Kim et al.'s own generations with our installed `utils/grader.py`:

**Task 0.2** — write `scripts/00_validate_grader.py`. Load `reference/kim_example_eval_outputs/{base,limo}/Qwen2.5-7B/aime/test_qwen-instruct_t0.0_k1_s0_e30.jsonl`, re-run `extract_answer` + `check_is_correct` on `generated_responses`, and assert we recover **4/30 and 8/30**. If our grader disagrees with their stored `is_correct` field on any item, the discrepancy is a grader/sympy-version problem, not a modelling problem — resolve it here.

**Task 0.3** — `scripts/01_fetch_data.sh`: download `Qwen/Qwen2.5-7B` and build `data/raw/limo_v2.json` via `_kim_make_limo_dataset.py`. Assert `len == 800`. Log the token-length distribution under the Qwen tokenizer and record **what fraction exceeds `cutoff_len: 16384`** — silently truncated traces are a real confound and we want the number on record.

---

## 5. Phase 1 — Replicate the Qwen2.5-7B row

### 5.1 Metric definition, stated precisely

The paper's Appendix E is explicit and slightly asymmetric, and getting this wrong would make us chase a number that does not exist:

> "we report the pass@1 performance of the **base** model using the larger value between temperatures 0.0 and 0.7, while the **LIMO** pass@1 performance is always reported with the temperature fixed at 0.0."

For Qwen2.5-7B, Table 7 gives base = {t0.0: **13.33**, t0.7/p1.0: 6.67, t0.7/p0.8: 3.33} → the reported 13.3% is the **t=0.0** cell, which also matches the reference generations exactly. So all three conditions in our row use **greedy decoding**. Good — no temperature sweep needed for the headline number.

> ⚠️ **Statistical caveat, to be stated in every write-up.** 30 problems, one greedy sample: the granularity is 3.33 pp and the 95% CI on 8/30 is roughly ±16 pp. The Hindsight cell (1/30) is one problem away from 0% or 6.7%. **We treat exact-number reproduction as a nice-to-have and the ordering + effect direction (LIMO ≫ Base ≫ Hindsight) as the real acceptance criterion.** Alongside every greedy number we also report **avg@16 at t=0.7** using `eval_acc.py`, which is a far lower-variance estimator of the same quantity and is what all Phase 3/4 comparisons will actually be judged on.

### 5.2 Condition A — Base (target 13.3% = 4/30)

**Task 1.1** — `scripts/10_eval.sh`, a parameterized wrapper. Run from `third_party/kim_eval/`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
python eval.py --model_name_or_path Qwen/Qwen2.5-7B \
  --data_name aime --prompt_type qwen-instruct \
  --temperature 0.0 --n_sampling 1 --k 1 --top_p 1 \
  --max_tokens 32768 --split test --start_idx 0 --end_idx -1 --seed 0 \
  --surround_with_messages --output_dir <repo>/outputs/base/Qwen2.5-7B
```

Fix `tensor_parallel_size` at 4 for every eval so results stay comparable (greedy decoding is not bitwise-stable across TP sizes). Cost: minutes.

**Gate:** 4/30. If we get 3/30 or 5/30, diff our per-problem `is_correct` against the reference file — it tells us exactly which problem flipped and whether it is a decoding or a grading difference.

### 5.3 Condition B — LIMO (target 26.7% = 8/30)

**Task 1.2** — train:

```bash
source .venv-train/bin/activate
FORCE_TORCHRUN=1 llamafactory-cli train configs/train/qwen2.5-7b_limo.yaml
```

`save_strategy: epoch` writes 15 checkpoints (~15 GB each ≈ 225 GB). Keep the final one plus epochs 5 and 10 for a sensitivity check, delete the rest. Cost estimate: 1500 steps, ~10–20 s/step under ZeRO-3 on PCIe → **4–8 h**.

**Task 1.3** — evaluate the final checkpoint exactly as in 1.1. **Gate: 8/30 (accept 6–10/30 given the variance above).**

### 5.4 Condition C — Hindsight (target 3.3% = 1/30)

This is the expensive one and it is worth being precise about what it is, because **it is not the paper's Table 5.** Table 5 ("SFT no EV", Qwen2.5-7B: 13.3 → 6.7) is *self*-distillation — the model's own traces regenerated under a suppression instruction. The proposal's Hindsight column is the *`make_hint_dataset.py`* experiment: **the LIMO traces themselves rewritten by DeepSeek-R1-Distill-Qwen-32B into confident, doubt-free derivations.** Same 800 problems, same answers, epistemic verbalization removed. That is why one dataset serves all three student rows.

The rewrite prompt (verbatim from `_kim_make_hint_dataset.py`) ends:

> *"…re-derive the result independently from scratch, step by step. Include all key equations and intermediate algebra. Do not express any uncertainty — never say 'I think,' 'probably,' or 'it seems.' State everything with full confidence."*

with the generation force-prefixed `"Okay, so I"`, and each output validated by the same 32B model against a GOOD/BAD rubric, retrying up to 20 times.

**Task 1.4** — `src/antidistill/defenses/hindsight.py`: rewrite `_kim_make_hint_dataset.py` into the defense-plugin API. Keep the prompt, the `"Okay, so I"` prefix, the validation loop, and the resume-on-restart behaviour byte-for-byte. Changes we *do* make:
- CLI args instead of module-level constants (teacher, input, output, ports).
- **A retry budget cap.** `MAX_RETRIES = 20` on a 32 k-token generation is an unbounded tail; cap total regenerations and log how many items exhausted the budget (those fall back to the last attempt, which may be a *wrong* trace — a confound worth quantifying).
- Emit a sidecar `data/defended/limo_hindsight_ds32b.meta.json` with per-item retry count, validator verdict, and output token length.

**Task 1.5** — serve 4 × TP=2 replicas of `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` (adapt `scripts/_kim_start_hint_server.sh`; note it currently hard-codes 4 GPUs per server with `--tp 2`, which leaves capacity idle). Generate all 800. **Cost estimate: 6–16 h.** This is the critical-path item — start it as soon as Phase 0 passes, in parallel with the LIMO training run.

**Task 1.6** — verify the defense actually did what it claims *before* training on it: epistemic token counts per sample in `limo_hindsight_ds32b.json` should be near zero versus LIMO's ~77 "wait" per sample (paper Figure 6). If not, the prompt is not landing and training is wasted. This check is the first consumer of `scoring/epistemic.py` and doubles as its smoke test.

**Task 1.7** — train (`configs/train/qwen2.5-7b_hindsight.yaml`, identical hyperparameters) and evaluate. **Gate: 1/30, i.e. a collapse well below base.**

### 5.5 Phase 1 deliverable

`results/phase1_replication.md`: the 3×2 table (greedy pass@1 + avg@16), our numbers beside Kim et al.'s, per-problem correctness diffs against the reference generations, and every deviation from the published setup.

---

## 6. Phase 2 — Defense plugin interface

**Task 2.1** — define the contract in `src/antidistill/defenses/base.py`:

```python
class Defense(Protocol):
    name: str
    def apply(self, traces: list[Trace]) -> list[Trace]: ...
    def metadata(self) -> dict: ...   # params, model versions, timing
```

Register `hindsight` (Phase 1) as the reference implementation. Then, in rough order of value to the proposal:

- **`part`** (§3.2) — structural: strip self-talk sentences and hoist sub-conclusions above their derivations. Cheap, no teacher model, directly targets the axis Li et al. found most damaging.
- **`token_swap`** (§3.1, ADS/DOGe surrogate) — we are not reimplementing ADS's proxy-student sampling. We implement a *controlled analogue*: perturb tokens under a budget, so the perturbation-budget hypothesis ("on hard problems the budget shrinks") becomes testable.
- **`summarize`** — connector-text-style compression, the real-world defense named in §3.3.

Each defense produces `data/defended/<name>.json`, gets a `dataset_info.json` entry and a `configs/train/qwen2.5-7b_<name>.yaml`. Phase 1's pipeline then runs unchanged.

---

## 7. Phase 3 — `Score(trace, student)`

`Score = correctness × epistemic_density × distributional_alignment × difficulty_match`, each factor in `[0, 1]`. Multiplicative means any factor can veto — which is the intent for correctness, and something to watch for the others (§7.6).

### 7.1 `correctness` — hard filter, ∈ {0, 1}

`utils/parser.extract_answer` + `utils/grader.check_is_correct` against the gold answer. LIMO ships gold answers; for pools without them, fall back to the LLM-judge rubric already in `_kim_make_hint_dataset.py`. Cheap, deterministic, reuses validated code.

### 7.2 `epistemic_density` — how much doubt-verify-proceed is in the trace

Two layers, because a lexical counter would score a *bypassed* defense at zero (Appendix H.1: models route around banned tokens with "hold on", "let me verify", a paragraph break):

- **Lexical:** word-boundary, case-insensitive counts of the nine tokens in `constants.EPISTEMIC_TOKENS`, plus `EPISTEMIC_BYPASS_PHRASES`, normalized per 1000 output tokens.
- **Structural:** count *episodes* — an epistemic marker followed within N tokens by a verification or re-derivation cue. This is the "behaviour, not tokens" claim of proposal §2.2 taken seriously.

Map to `[0,1]` with a **saturating** transform, `d = 1 − exp(−e/e₀)`, calibrated so LIMO's median sits around 0.7. Saturation matters: 77 "wait"s per sample is the top of the useful range, not a target to exceed.

**Calibration check:** LIMO ≫ hindsight-rewritten LIMO on the same 800 problems. We get this dataset pair for free from Phase 1.

### 7.3 `distributional_alignment` — can *this* student produce these tokens?

This is the component with real subtlety, and the paper tells us exactly what the statistic should be. Figure 8's finding is *not* "epistemic tokens should be high probability" — in models where LIMO **works**, epistemic tokens remain **low-probability and high-entropy** relative to other tokens. The failure mode is when they fall **outside the support entirely**.

So the statistic is a **gap**, not a level:

```
Δ = mean_logprob(all answer tokens) − mean_logprob(answer tokens ∈ epistemic set)
alignment = σ(−(Δ − Δ₀)/τ)          # decreasing in Δ; near 1 when the gap is ordinary
```

computed under the **student before any training**, reusing `_kim_score_samples.py` (per-token logprobs) and `_kim_analyze_token_distribution.py` (which already tracks entropy and per-token stats, and already resolves special tokens per-tokenizer — important, since `EPISTEMIC_TOKEN_IDS_QWEN` are Qwen-specific ids).

**Calibration check — and this one is decisive.** The metric must rank **Qwen2.5-7B** (LIMO works: 13.3 → 26.7) clearly **above Qwen2.5-Math-7B** (LIMO destroys it: 16.7 → 0.0) on the *same* LIMO traces. Both are 7B and both run on one GPU, so this is a cheap, falsifiable test of the whole component. If the metric does not separate them, it is wrong and no amount of downstream tuning will fix it.

### 7.4 `difficulty_match` — is this problem at the student's boundary?

Estimate the student's pass rate `p̂` per problem: k = 16 samples at t = 0.7, graded with the same rule-based grader. Then

```
m = 4 · p̂ · (1 − p̂)     # peaks at p̂ = 0.5, → 0 at trivial and impossible
```

with an asymmetric variant favouring `p̂` slightly below 0.5 as an ablation (LIMO's curation principle is "problems the *teacher* found hard"). One rollout pass per (student, pool) — the most expensive component, but cacheable and reusable across every defense.

### 7.5 Combiner and selection

**Task 3.5** — `scoring/score.py`: compute all four, cache each independently (they have wildly different costs), emit `results/scoring/<student>_<pool>.json` with per-trace factor breakdown. Select **top-800** to hold the training budget fixed against LIMO — every comparison must be at equal N, equal steps, equal hyperparameters.

### 7.6 Ablations that must be run

- Leave-one-factor-out (4 runs) — does each factor earn its place?
- **Random-800** from the same pool — the honest baseline. If curation doesn't beat random selection, the framework has no content.
- Full-pool (no selection) at matched step count.
- **Additive vs multiplicative** combiner. Multiplicative lets one noisy factor veto a good trace; worth one run to check that isn't happening.

---

## 8. Phase 4 — Attacks

Ordered by cost-to-signal ratio, cheapest first.

1. **Loss masking (§4.4)** — zero the SFT loss on epistemic token positions; the student still *sees* them in context. The proposal calls it "a one-line code change"; in practice it is a `label = IGNORE_INDEX` write in LLaMA-Factory's SFT preprocessing (`third_party/LLaMA-Factory/src/llamafactory/data/processor/supervised.py`) gated behind a new training arg. **This is why we vendored the trainer.** Best first experiment: cheapest change, and it directly tests the distributional-alignment mechanism — the sharpest test is Qwen2.5-**Math**-7B + LIMO, where the unmasked run is known to collapse to 0.0%.
2. **Epistemic supplementation (§4.3)** — naive mixing of hindsight-defended traces with LIMO epistemic traces. Needs only a dataset-mixing utility. Tests the proposal's central orthogonality claim.
3. **Quality-guided curation (§4.1)** — Phase 3's Score applied to a defended pool.
4. **Synthetic augmentation (§4.3)** — LLM-rewrite defended traces to inject fabricated self-doubt.
5. **Preference optimization (§4.3)** — ORPO/DPO on (epistemic, confident) pairs. LLaMA-Factory supports both natively; the Phase 1 pipeline gives us paired data for free.

---

## 9. Risks and open decisions

| Risk | Mitigation |
| --- | --- |
| **30-problem metric is extremely noisy.** 1 problem = 3.33 pp. | Report avg@16 alongside every greedy number; judge on effect direction. Consider adding AIME25 + AMC23 (already in `kim_eval/data/`) as secondary benchmarks. |
| **Hindsight generation is the critical path** (6–16 h, unbounded retry tail). | Start it in parallel with LIMO training. Cap the retry budget; log exhaustions. |
| **`cutoff_len: 16384` may truncate long LIMO traces.** | Measure the truncation rate in Task 0.3 and report it. Keep the LIMO default for the replication; treat any change as a documented deviation. |
| **ZeRO-3 over PCIe is slow.** | Benchmark ZeRO-2; switch only if loss curves match, and record it. |
| **Checkpoint disk: 15 epochs × ~15 GB.** | Keep epochs 5/10/15, prune the rest. 800 GB free is enough but not unlimited. |
| **Exact numbers may not reproduce.** | The reference generations let us diff *per problem*, separating decoding differences from grading differences. State the acceptance criterion (ordering) up front rather than post-hoc. |
| **`EPISTEMIC_TOKEN_IDS_QWEN` are tokenizer-specific.** | Always re-resolve ids per model; `_kim_analyze_token_distribution.py` already does this correctly and warns on multi-subtoken splits. |

**Open decisions to settle before Phase 3:** the exact student model set (Qwen2.5-7B only, or add Qwen2.5-Math-7B as the negative control? — recommend the latter, it is the sharpest test we have and it is only 7B); and whether the trace pool for curation is LIMO-only or LIMO + open-source rollouts.

---

## 10. Milestones

- [ ] **M0 — Environment.** Both venvs build; grader reproduces 4/30 and 8/30 from the reference generations; `limo_v2.json` has 800 rows with a logged truncation rate.
- [ ] **M1 — Base cell.** Qwen2.5-7B greedy AIME24 = 4/30.
- [ ] **M2 — LIMO cell.** SFT complete, eval ≈ 8/30. *Pipeline is now validated end to end.*
- [ ] **M3 — Hindsight cell.** Dataset generated and verified epistemic-poor; SFT complete; eval ≈ 1/30. **G1 done.**
- [ ] **M4 — Defense API.** `hindsight` refactored behind `Defense`; one additional defense (`part`) implemented and evaluated.
- [ ] **M5 — Score function.** All four components implemented and independently calibrated; alignment separates Qwen2.5-7B from Qwen2.5-Math-7B.
- [ ] **M6 — First attack.** Loss masking evaluated; curated-SFT beats random-800 at equal budget.

**Estimated wall-clock to M3:** 2–3 days, dominated by two SFT runs (4–8 h each) and hindsight generation (6–16 h), which overlap.
