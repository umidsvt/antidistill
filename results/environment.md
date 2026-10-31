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

## Rebuild 2026-10-30 (dspl1 after its reinstall)

Same scripts (`setup_third_party.sh`, `setup_envs.sh`, `setup_env_dg.sh`). Env A matches the versions recorded above; in
env B transformers is 4.57.1, where the record above lists 5.16.1 (written before the script pinned `<5`). Host: driver 615.71.09, CUDA 13.4 (`/usr/local/cuda-13.4`), gcc 15.2.0. GPU P2P now works:
`check_host.sh` measured an NCCL all-reduce at 18.4 GB/s on GPUs 0–3, so `.host_profile` sets no NCCL overrides.

| env | key versions | changes to the script |
| --- | --- | --- |
| A, `.venv-train` | torch 2.6.0+cu124, transformers 4.48.2, deepspeed 0.16.9, llamafactory 0.9.2.dev0, flash-attn 2.7.4.post1 | `pyarrow<22`: the unpinned 26.0.0 refuses numpy 1.26, which LLaMA-Factory keeps |
| B, `.venv-infer` | vllm 0.11.0, torch 2.8.0+cu129, transformers 4.57.1, sglang 0.5.5.post3 | `torch==2.8.0` in the sglang install: unpinned, sglang replaced torch and `vllm._C` failed to load; the check now imports `vllm._C` |
| C, `.venv-dg` | vllm 0.8.5.post1, torch 2.6.0+cu124, transformers 4.53.2, trl 0.19.1, peft 0.15.2 | none (installed from `scripts/env_dg.lock.txt`) |

Checks: `00_validate_grader.py` PASS PASS (4/30, 8/30); `01_fetch_data.py` gives 800 LIMO rows and pool statistics
identical to `task0.3_pool_stats.md`. Models are in the lab's shared cache (`HF_HUB_CACHE`): DeepSeek-R1-Distill-Qwen-7B,
Qwen2.5-7B, Qwen2.5-7B-Instruct, Qwen2.5-3B, unsloth/Llama-3.2-3B, unsloth/Llama-3.2-3B-Instruct. Not downloaded: the
32B models (R1-Distill-Qwen-32B, Qwen2.5-32B-Instruct), needed only to regenerate hindsight, PART or 32B attack data.
