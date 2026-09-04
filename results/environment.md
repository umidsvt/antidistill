# Task 0.1 — environment record

Built 2026-09-04T02:23Z on `dspl1.cs.vt.edu`. Reproduce with `bash scripts/setup_envs.sh`.

## Host

| item | value |
| --- | --- |
| GPUs | 8 x NVIDIA L40S, 46068 MiB each, compute capability 8.9 (no NVLink) |
| driver / CUDA | 580.95.05 / 13.0 |
| CUDA toolkit | /usr/local/cuda-13 /usr/local/cuda-13.0  |
| gcc | 11.5.0 |
| CPU / RAM | 64 cores / 1007 GB |
| disk (repo, `/`) | 777G free |
| HF cache | `/mnt/vault/huggingface_cache` (**shared, multi-user**) — 607G free |
| uv | 0.12.9 |

## env A — training (`.venv-train`, Python 3.11)

Used for: SFT via LLaMA-Factory, and any transformers-side scoring.

| package | version |
| --- | --- |
| torch | 2.6.0+cu124 (cuda 12.4, cxx11abi=False) |
| transformers | 4.48.2 |
| datasets | 3.2.0 |
| accelerate | 1.2.1 |
| peft | 0.12.0 |
| trl | 0.9.6 |
| deepspeed | 0.16.9 |
| flash-attn | 2.7.4.post1 |
| llamafactory | 0.9.2.dev0 |

## env B — inference (`.venv-infer`, Python 3.12)

Used for: vLLM evaluation, SGLang trace generation, grading.

| package | version |
| --- | --- |
| vllm | 0.11.0 |
| torch | 2.8.0+cu129 (cuda 12.9) |
| transformers | 5.16.1 |
| sympy | 1.14.0 |

## Notes and gotchas

- **`uv venv` does not seed setuptools.** Both `deepspeed` and `trl` import it at load time;
  without it `llamafactory-cli` dies with `No module named 'setuptools'`.
- **flash-attn from a prebuilt wheel.** torch is pinned to `2.6.0` precisely so the
  `cu12torch2.6cxx11abiFALSE-cp311` wheel matches. Building from source takes ~1h; the
  wheel takes ~1min. Verified with a real bf16 forward pass on-device, not just an import.
- **The eval harness must be run from `third_party/kim_eval/`** — it resolves `./prompts`
  and `./data` relative to the working directory. `scripts/10_eval.sh` handles this.
- **The HF cache is shared across users** (`umid`, `khizar`, `mokshitha`, …) and holds two
  layouts simultaneously: a legacy flat one (`$HF_HOME/models--*`, used by transformers 4.48
  because `TRANSFORMERS_CACHE` is exported in the shell profile) and the modern hub one
  (`$HF_HOME/hub/models--*`, used by huggingface_hub / vLLM). A model fetched by one env is
  therefore **not** visible to the other, and gets downloaded twice.
  `Qwen/Qwen2.5-7B` now exists in both (15 GB each); the flat copy predates this project and
  belongs to another user, so neither was removed. Set `HF_HUB_CACHE` explicitly if the
  duplication matters — `/mnt/vault` is 91% full.
- **`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` is not cached** (~65 GB). Needed for Phase 1C
  (hindsight generation); budget the download time into that step.
