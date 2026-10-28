# antidistill

Research code for the paper *Is There Anything to Protect? Validity Checks for Evaluating Anti-Distillation Defenses
of Reasoning Models* (working title, aimed at ICML 2027).

The project started as a replication of Kim et al. (arXiv:2603.15500) and a set of adaptive attacks on reasoning
distillation defenses (results/REPORT.md, Parts 1–10). Since the advisor meeting of 2026-10-20 it is an **audit of
the setups used to evaluate those defenses**: a defense setup only makes sense if the attacker cannot get an equally
good student without the teacher's outputs, from an open model's solutions (Solo) or by improving the student on its
own (RL, self-training). The paper states the checks a setup must pass, applies them to the setups of published
defenses, and looks for a setup that passes.

## Start here

| If you are | Read |
| --- | --- |
| a coding agent (Claude Code, Codex, ...) | [CLAUDE.md](CLAUDE.md), section "Start here"; Codex and others: [AGENTS.md](AGENTS.md) |
| picking up the work | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), top section: status, running jobs, next steps, open decisions |
| looking for a number or a decision rule | [results/m8_audit_design.md](results/m8_audit_design.md): the live design record of the audit; every rule is written there before its result |
| looking for the LIMO-phase results (defenses, attacks) | [results/REPORT.md](results/REPORT.md) |
| looking for the paper | separate private repo https://github.com/ceferisbarov/antidistillation-overleaf, folder `icml2027_audit/` (ask the authors for access) |

## The checks, in one paragraph

U is the student fine-tuned on the teacher's undefended outputs, D on the defended outputs, B the untrained student.
At attacker level ℓ (A: owns only the student; B: also its instruction-tuned sibling; C: an open model of the
teacher's size; D: an open reasoning model), R_ℓ is the best student built without the teacher, and U_ℓ / A_ℓ the best
student built from the same resources plus the undefended / defended outputs. A setup must show headroom (U clearly
above B), **necessity** (U_ℓ clearly above R_ℓ), stability over seeds, consistency across test sets, and robustness to
the finishing confound. Protection is π_ℓ = (U_ℓ − A_ℓ) / (U_ℓ − R_ℓ). Definitions: m8 §8 and §11; the paper's §3–4.

## Results so far (2026-10-28)

| Setup | Student | U | Best student without the teacher | Verdict |
| --- | --- | --- | --- | --- |
| LIMO, 600 problems (Kim et al., PART) | Qwen2.5-7B, 3 seeds | 56.5% | Solo from R1-Distill-Qwen-7B: 66.2% (level D) | U below Solo on every seed; headroom over the untrained 49.8% not shown with 3 seeds (+13.0, +8.7, −1.6) |
| GSM8K-Platinum (ADS, PoE, trace rewriting, DOGe, TARS) | Llama-3.2-3B, 3 seeds | 59.0% | level A, few-shot self-training: 34.3%; level B, Solo from Llama-3.2-3B-Instruct: 59.4%; level C: 59.8% | passes at level A; fails at B and C; combining the attacker's data with the teacher's adds 1–3 points (seed 123), far below the margin |
| MATH, same five papers | Llama-3.2-3B, 3 seeds | 15.0% | level B: 14.9%; level C: 17.3% | fails at B and C (combination students: seed 123 done for B, running for C) |
| MATH, teacher allowed 8,192 tokens (pilot) | Llama-3.2-3B, seed 123 | 18.1% | Solo C: 18.5% | fails by the rule, because the student finishes only 41% of its answers; on finished answers 31.4% vs 19.0% |
| Hardest DeepMath problems (pilot) | – | – | an open 7B reasoning model fails only 10% of them | harder problems alone do not create the gap |

Details, seeds and margins: m8 §9.1–§11; the setup-2 token-limit finding (the teacher capped at 1,024 tokens is a
weaker solver than Qwen2.5-7B-Instruct): m8 §9.4.

## Setting up a machine

```bash
bash scripts/check_host.sh          # host profile (GPU interconnect); read CLAUDE.md §0.5 first
bash scripts/setup_envs.sh          # .venv-train (LLaMA-Factory) and .venv-infer (vLLM 0.11, eval)
bash scripts/setup_third_party.sh   # upstream clones under third_party/ext/, plus our patch
bash scripts/setup_env_dg.sh        # .venv-dg: the Distillation Game code and all scripts/dg_*.py
```

Model downloads: set `HF_HOME` (on dspl1: `/mnt/vault/huggingface_cache`) and `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`, and
use ungated mirrors (`unsloth/Llama-3.2-3B`, `unsloth/Llama-3.2-3B-Instruct`). Never use another user's token.

## Layout

| Path | Contents |
| --- | --- |
| `results/` | all write-ups; `m8_audit_design.md` is the live record of the audit, `REPORT.md` of the LIMO phase |
| `scripts/` | numbered steps: 00–27 the LIMO pipeline (training, eval, defenses, attacks), 28–43 the audit (index below) |
| `configs/train/` | LLaMA-Factory yamls, one per (student × condition × seed) |
| `configs/setup2/` | Distillation Game configs for the GSM8K/MATH setup: `{gsm8k,math}_{U,solo_l3i,solo_q7i}_s{123,124,125}.yaml` |
| `src/antidistill/` | defenses (hindsight, PART), attacks (style, re-voice, reconstruction), scoring, data builders |
| `data/` | `dataset_info.json` (LLaMA-Factory registry); `defended/` and `curated/` datasets (mostly not in git, see below) |
| `third_party/LLaMA-Factory/`, `third_party/kim_eval/` | vendored trainer and math eval harness (run the harness from inside its directory) |
| `third_party/ext/` | upstream clones recreated by `scripts/setup_third_party.sh` (see its README) |
| `reference/` | Kim et al.'s own generations, used to validate our grader |
| `tests/` | unit tests (`tests/test_part.py`) |
| `outputs/`, `saves/`, `logs/` | run outputs, checkpoints, logs (not in git; on dspl1 under `/home/umid/AntiDistillation/antidistill/`) |

**Audit scripts (m8 section in brackets).** `28` small students on LIMO (§9.2) · `29`, `31`, `33` setup-2 runs,
attacker ladder and job pool (§9.1, §9.3, §9.4) · `30`, `36_setup1_7b_seeds` LIMO seeds (§9.5) · `32`, `dg_rl_base.py`
RL from the student (§9.3) · `34` DeepMath pilot (§10) · `35`, `37_star_s123_train`, `dg_star.py` self-training,
level A (§9.3) · `38`–`41`, `dg_train_from_traces.py`, `dg_score_vllm.py` the MATH full-budget pilot (§10.1) · `42`
combination students for U_ℓ (§11) · `43` the pilot's combination students (§10.1, §11) · `dg_eval_base.py` the
untrained student.

## Data not in git

Datasets of tens of MB that are regenerated by a script stay out of git. They are on dspl1 under
`/home/umid/AntiDistillation/antidistill/` (ask umid for a copy):

| File(s) | Made by |
| --- | --- |
| `data/raw/limo_v2.json` | `scripts/01_fetch_data.py` |
| `data/defended/limo_part*.json` (PART) | `scripts/45_generate_part.sh` |
| `data/curated/recon7b_*.json`, `limo_recon_*.json` (Solo, Search, Style) | `src/antidistill/attacks/epistemic_reconstruction.py` (model served by `scripts/40_serve_teacher.sh`) |
| `data/curated/style_part*.json`, `revoice_part*.json` | `src/antidistill/attacks/style_part.py`, `revoice.py` |
| `data/curated/limo_mixed_*.json` | `scripts/50_make_mixed_dataset.py` |
| `outputs/setup2/` (teacher and Solo solutions of setup 2, per seed) | `scripts/29_setup2_run.sh`, `31_setup2_ladder.sh`, `33_setup2_pool.sh` |
| `outputs/setup3_math/` (long-budget teacher, pilot students and scores) | `scripts/39`–`41`, `40_math_pilot.sh` |

The hindsight datasets (`data/defended/limo_hindsight_*.json`, about 5 MB) are in git; use `limo_hindsight_chat.json`
(v2), not the v1 file (CLAUDE.md §1).

## Attribution

Builds on Kim et al. (arXiv:2603.15500), LIMO (Ye et al., COLM 2025), LLaMA-Factory (Zheng et al., ACL 2024), The
Distillation Game (Allouah et al., arXiv:2605.22737) and Antidistillation Sampling (Savani et al.). See
`third_party/*/LICENSE`. Files prefixed `_kim_` are verbatim copies of Kim et al.'s code, kept as references.
