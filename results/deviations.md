# Deviations from Kim et al. / LIMO

Every departure from the published setup, why it was forced, and whether it can affect results.
Nothing here is a silent patch — each has a `DEVIATION (antidistill)` comment at the code or
config site. Two are host workarounds, two are memory workarounds, none change the recipe.

## Summary

| # | Deviation | Forced by | Can it affect results? |
| --- | --- | --- | --- |
| 1 | `gpu_memory_utilization` 0.96 → 0.90 (eval) | vLLM 0.11 V1 sampler warm-up OOMs on 46 GB | No — eval-time memory sizing. Changes KV-cache size and hence batch composition, which is already nondeterministic across hardware. |
| 2 | `NCCL_P2P_DISABLE=1` (train) | GPU P2P advertised but non-functional on this host | No — transport only. Collectives are bit-identical, just slower. |
| 3 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (train) | Allocator fragmentation | No — allocator behaviour only. |
| 4 | `enable_liger_kernel: true` (train) | 9.28 GB cross-entropy tensor does not fit | Precision-level only. **Empirically verified — see §4.** |
| 5 | `eval_acc.py` honours `--output_dir` | Upstream hardcodes `avg_outputs/` | No — output path only. |

Unchanged and verbatim from LIMO's `train_limo.yaml`: `finetuning_type: full`, ZeRO-3,
`cutoff_len: 16384`, `template: qwen`, `per_device_train_batch_size: 1`,
`gradient_accumulation_steps: 1`, `learning_rate: 5.0e-6`, `num_train_epochs: 15`,
`lr_scheduler_type: cosine`, `warmup_ratio: 0.0`, `bf16: true` → **1,500 optimizer steps**.

---

## 1. Eval `gpu_memory_utilization`

Kim et al. hardcode `0.96` in both `eval.py` and `eval_acc.py`. Under vLLM 0.11's V1 engine on a
46 GB L40S this allocates a 2.8M-token KV cache and then OOMs warming up the sampler with 256
dummy requests, with ~295 MB left. Made a CLI arg in both scripts (default still `0.96`); our
wrapper passes `0.90`.

## 2. NCCL P2P disabled

The first M2 launch deadlocked in `dist.barrier()` inside HF's `main_process_first`, all ranks
spinning at 100% GPU util / 93 W / 700 MB — NCCL busy-wait, not compute. Isolated from the
training stack with `scripts/diag_nccl.py`:

| NCCL config | 8-GPU result |
| --- | --- |
| defaults | 1-element all-reduce **times out after 90 s** |
| `NCCL_P2P_LEVEL=NODE` | also hangs — P2P broken *within* a socket too |
| `NCCL_P2P_DISABLE=1` | passes, ~1.6 GB/s bus bw |
| `NCCL_P2P_DISABLE=1 NCCL_ALGO=Tree` | passes, ~1.4 GB/s |

`torch.cuda.can_device_access_peer` returns true for **every** pair, i.e. CUDA advertises P2P
while NCCL hangs the moment it uses it — the classic signature of ACS enabled on the PCIe
bridges (or IOMMU translation) silently black-holing peer DMA. `nvidia-smi topo -m` shows no
NVLink: GPUs 0–3 on NUMA node 0, 4–7 on node 1, all `SYS`.

**This is a host misconfiguration affecting every multi-GPU job on this machine, not just ours.**
Fixing it needs root (BIOS ACS / IOMMU). Until then all collectives stage through host shared
memory, which is what makes training ~27 s/step instead of a few seconds.

> Why the vLLM evals were unaffected: they used only GPUs 0–3 and vLLM probes P2P and falls back
> on its own. Training was the first job to actually depend on NCCL P2P.

## 3. `expandable_segments:True`

Attempt 2 OOMed at step 2 with 33.5 GB allocated, a 9.28 GB request failing, and **7–10 GB
reserved-but-unallocated**. The 9.28 GB is the fp32 cross-entropy upcast over Qwen's 152k vocab
at seq 16384 — allocated and freed every step, the pattern that fragments the caching allocator.

`expandable_segments` cut fragmentation to **170–310 MB**, confirming the mechanism — but attempt
3 still OOMed, which is what proved the shortfall was *real* rather than packing:
**38.18 GB steady + 9.28 GB = 47.5 GB vs 44.39 GB capacity.** Retained anyway; it is free.

## 4. Liger fused cross-entropy

The only deviation that touches computation, so it gets the most scrutiny.

Liger's fused linear cross-entropy never materialises the `[seq, vocab]` logits tensor, removing
the failing allocation instead of trying to fit around it. Measured effect: steady memory
**38.18 GB → ~16 GB**, peak on the long step-2 batch **47.5 GB (OOM) → 29.5 GB**.

Chosen over ZeRO-3 CPU optimizer offload (the alternative, which would free ~11 GB into the
host's 1 TB of RAM) because offload adds a PCIe round-trip per step on an already
transport-bound run, and DeepSpeedCPUAdam would have to JIT-compile against CUDA 13 `nvcc` with a
cu124 torch build.

**Numerical equivalence, measured.** Same data, same seed, same step, with and without Liger:

| | attempt 2 (no Liger) | attempt 4 (Liger) | rel. diff |
| --- | --- | --- | --- |
| step-1 loss | 0.8577 | 0.8576 | 1.2e-4 |
| step-1 grad_norm | 3.209986420470172 | 3.206699321171526 | 1.0e-3 |

That is fused-kernel float noise, far below the effect sizes under study (one AIME problem =
3.33 pp). Version pin: `liger-kernel==0.5.10` — 0.6+ requires transformers ≥4.52, which
LLaMA-Factory 0.9.2 forbids (`<=4.48.2`).

**It did not buy speed.** 34.76 s/it without → 36.78 s/it with (settling to ~27 s/it as warmup
cost amortised). The expectation that lower memory traffic would help was wrong: this run is
entirely NCCL-bound, so cutting compute and memory pressure moves nothing. Useful negative
result — it confirms deviation #2 is the binding constraint and that no further software change
on our side will materially speed this up.

## 5. `eval_acc.py` output path

Upstream hardcodes a relative `avg_outputs/` and silently ignores `--output_dir` (which
`eval.py` honours), scattering avg@k results into the harness directory. Now respects
`--output_dir`; default preserves upstream behaviour.
