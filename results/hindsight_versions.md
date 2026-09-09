# Two hindsight datasets — what differs, and where everything lives

The hindsight (defended) condition exists in two versions. **Both are kept.** v1 is what every
result in `REPORT.md` was produced from; v2 is the corrected regeneration. They must never be
conflated, and neither should be deleted: the v1-vs-v2 contrast is itself an experiment.

## What changed

| | **v1 (Kim et al.'s procedure)** | **v2 (corrected)** |
| --- | --- | --- |
| endpoint | `completions` — raw text, no chat markers, no BOS | `chat/completions` with the model's template |
| `<think>` opener | none (template bypassed) | emitted by the template's generation prompt |
| reasoning primer | literal prefix `"Okay, so I"` | none needed |
| temperature | 0.4 | **0.6** (DeepSeek recommend 0.5-0.7) |
| what was stored | the whole completion: scratchpad **and** answer | `.content` only — the answer |
| judge budget | `max_tokens=50` | `8192` |

v1's script does `new_item["output"] = teacher_text`, so each trace held **two complete solutions**
separated by a `</think>` whose opener never existed.

## Measured consequences

| | v1 | v2 |
| --- | --- | --- |
| traces containing `</think>` | 760/800 | **0** |
| traces with >1 `\boxed` | 761 | 63 *(LIMO itself has 479)* |
| epistemic tokens / 1k words | 0.655 | **0.024** |
| mean trained tokens | 1,260 | **486** |
| max trained tokens | 32,788 | **1,369** |
| truncated at `cutoff_len 16384` | 7/800 | **0/800** |
| tokens with no stop signal | 11.3% | **0.0%** |

**The artifact transferred to the student.** 499/500 MATH500 responses from the v1-trained model
contain `</think>`, against 0/500 for LIMO and 0/500 for base.

**Interpretation caveat:** two variables moved, not one. The artifacts were removed *and* the token
budget fell 2.6x (1.01M -> 388k), because a v1 trace carried two solutions and a v2 trace carries
one. There is no token-neutral version of this fix, so a v1-vs-v2 difference cannot be attributed
to the artifacts alone.

## Where the artifacts live

| | v1 | v2 |
| --- | --- | --- |
| dataset | `data/defended/limo_hindsight_ds32b.json` | `data/defended/limo_hindsight_chat.json` |
| sidecar | `...ds32b.meta.json` | `...chat.meta.json` |
| registry key | `limo_hindsight_ds32b` | `limo_hindsight_chat` |
| train config | `configs/train/qwen2.5-7b_hindsight.yaml` | `configs/train/qwen2.5-7b_hindsight_v2.yaml` |
| checkpoints | `saves/Qwen2.5-7B_hindsight/` | `saves/Qwen2.5-7B_hindsight_v2/` |
| eval outputs | `outputs/hindsight_ep{5,10,15}/` | `outputs/hindsight_v2_ep15/` |
| write-up | `results/m3_hindsight.md`, `REPORT.md` | pending |

Both datasets and both sidecars are un-gitignored and ship with the repo. Checkpoints and eval
outputs are gitignored (size) but stay on disk.

**Naming matters mechanically:** `scripts/30_collect_results.py` takes the condition label from the
first path component under `outputs/` (`condition = rel.parts[0]`). Distinct labels are what keep
v1 and v2 as separate rows in `results/eval_table.md`, which is regenerated wholesale from whatever
is on disk. Evaluate v2 as `hindsight_v2_ep15`, never as `hindsight_ep15`.

## Status

v2 training started 2026-09-08, 4 GPUs, ~6h20m. Nothing in v2 truncates at `cutoff_len 16384`, so
this run's result equals what a 33,792 run would give and does not depend on the pending
`cutoff_len` memory probe.
