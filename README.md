# antidistill

Adaptive attacks on reasoning-distillation defenses.

**New here / running on another machine? Start with [CLAUDE.md](CLAUDE.md)** — source documents, host profiling, and the full pipeline with a different student model.

Then: **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** for why the project is shaped this way, and [`Research Proposal_ Adaptive Attacks on Reasoning Distillation Defenses.md`](Research%20Proposal_%20Adaptive%20Attacks%20on%20Reasoning%20Distillation%20Defenses.md) for the wider research programme this feeds.

Immediate goal (G1): reproduce the `Qwen2.5-7B` row of the proposal's Section 2.2 table —
AIME24 pass@1 of **13.3% base / 26.7% LIMO / 3.3% hindsight** (= 4/30, 8/30, 1/30) —
then build the defense-plugin API, the `Score(trace, student)` curation function, and the
curated-SFT attack loop on top of that verified pipeline.

## Layout

| Path | Contents |
| --- | --- |
| `configs/` | DeepSpeed configs, LLaMA-Factory training yamls, eval sweeps |
| `data/` | `dataset_info.json` registry; `raw/` `defended/` `curated/` datasets (gitignored) |
| `src/antidistill/` | `constants.py`, dataset builders, `defenses/`, `scoring/`, `evaluation/` |
| `third_party/LLaMA-Factory/` | vendored trainer (v0.9.2.dev0) — we patch this for loss masking |
| `third_party/kim_eval/` | vendored math eval harness — **run from inside this directory** |
| `reference/` | Kim et al.'s own generations for the target row, used to validate our grader |
| `scripts/` | numbered pipeline steps |

Files prefixed `_kim_` are verbatim copies from
[`../strategic-information-allocation-llm-reasoning`](../strategic-information-allocation-llm-reasoning)
(Kim et al., arXiv:2603.15500), kept as reference implementations. That clone stays pristine;
this tree is the working copy.

## Attribution

Builds on Kim et al. (arXiv:2603.15500), LIMO (Ye et al., COLM 2025), and
LLaMA-Factory (Zheng et al., ACL 2024). See `third_party/*/LICENSE`.
