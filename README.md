# antidistill

Research code for the paper *What Can Anti-Distillation Defenses Protect? An Answer Floor, and a Testbed Where the
Teacher Is Needed* (working title, aimed at ICML 2027).

The project started as a replication of Kim et al. (arXiv:2603.15500) and a set of adaptive attacks on reasoning
distillation defenses (results/REPORT.md). From 2026-10-20 it was an audit of the setups used to evaluate those
defenses (results/m8_audit_design.md §1–15). Since the meeting of 2026-11-08 the paper is built on one observation:
every anti-distillation defense keeps the teacher's **final answers** correct, so an attacker can always ignore the
defended solutions and train from the answers alone. The most any such defense can protect is therefore the gap
between a student trained on the teacher's solutions and the best answers-only student, the **answer floor**. We
measure that gap in the setups the field uses (it is small or zero) and build a **puzzle testbed** where it is large.

## Start here

| If you are | Read |
| --- | --- |
| a coding agent (Claude Code, Codex, ...) | [CLAUDE.md](CLAUDE.md), section "Start here"; Codex and others: [AGENTS.md](AGENTS.md) |
| picking up the work | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), Part I: status, running jobs, next phases, open decisions |
| working on a stronger level-A attacker (RL) | [results/briefing_levelA_rl.md](results/briefing_levelA_rl.md) |
| looking for a number or a decision rule | [results/m8_audit_design.md](results/m8_audit_design.md): the live design record; every rule is written there before its result (§16–18 for the floor and the testbed) |
| looking for the LIMO-phase results (defenses, attacks) | [results/REPORT.md](results/REPORT.md) |
| looking for the paper | private repo https://github.com/ceferisbarov/antidistillation-overleaf, folder `icml2027_testbed/` (the earlier audit draft is `icml2027_audit/`) |

## The argument, in one paragraph

U is the student trained on the teacher's (undefended) solutions, U₀ the untrained student. At attacker level ℓ (A:
owns only the student; B: also a small open instruct model; C: an open instruct model of the teacher's size; D: an
open reasoning model weaker than the teacher), R_ℓ is the best student built without the teacher and F_ℓ the best
student built with the help of the teacher's final answers only (self-training on own attempts that reach the
teacher's answer, or RL rewarded by it). Any answer-preserving defense leaves F_ℓ untouched, so it can protect at most
**P_ℓ = U − F_ℓ**. The teacher is *needed* at level ℓ if U − R_ℓ, and the protectable part *exists* if U − F_ℓ, exceeds a
quarter of U − U₀ with a 95% interval over 3 seeds that excludes 0 (m8 §11, §18).

## Results so far (2026-11-15)

| Setup | Result | m8 |
| --- | --- | --- |
| GSM8K setup (Distillation Game; teacher R1-Distill-Qwen-7B; Llama-3.2-3B student; 3 seeds) | U 59.8%. Level B and C students on open models' solutions: 59.8% and 58.9%, so **nothing protectable**. Level A: own student 0.6%, answers-only floor F_A (few-shot STaR) **35.2%**, so **at most 24.6 points protectable**; teacher needed, +24.5 (21.6 to 27.5) vs margin 14.8 | §12, §13 |
| Same, RL as the answers-only attack at level A | GRPO alone with Javaheri et al.'s stabilizers, 600 steps, best checkpoint: 32.1% (35.3 / 28.3 / 32.6); STaR then GRPO 30.2%; first GRPO settings collapsed (2.1%). **STaR stays the floor** | §13–15 |
| MATH version of the same setup | U 15.5%, level B 15.5%, level C 17.6%: nothing protectable at B, C. Level A not run | §12 |
| How predictable the teacher's answers are (16 tries per problem and model) | GSM8K bits per problem: base student 1.93, Llama-3.2-3B-Instruct 0.39, Qwen2.5-7B-Instruct 0.10. Pre-registered test held at C, failed at B (equal-weight pooling). MATH C still to run | §16 |
| Knights-and-Knaves pilot (Xie et al.'s generator; teacher R1-Distill-Qwen-32B) | At 7 and 8 people the teacher solves 76% and 71%, the best owned model (R1-Distill-Qwen-7B) 28% and 20%: go. Qwen3-8B solves 92–95%, more than the teacher (limitation) | §17 |
| Knights-and-Knaves step 2 (2,000 training puzzles, 1,000 test, 3 seeds) | **Provisional** (scoring bug, rescore running): U 41–49%, floor F_D 19–22%, R_D 11–12%, R_C 5%, R_B 0.4%. Teacher needed at B, C, D; **protectable part exists (+22 to +27 points)** | §18 |

Earlier results (LIMO setup, combination students, the MATH full-budget and DeepMath pilots): m8 §9–11 and
results/REPORT.md.

## Setting up a machine

```bash
bash scripts/setup_third_party.sh   # upstream clones under third_party/ext/, plus our patch
bash scripts/setup_envs.sh          # .venv-train (LLaMA-Factory) and .venv-infer (vLLM 0.11, eval)
bash scripts/setup_env_dg.sh        # .venv-dg: the Distillation Game code and all scripts/dg_*.py
bash scripts/check_host.sh          # host profile (GPU interconnect, needs .venv-train); read CLAUDE.md §0.5 first
```

Model downloads: on dspl1 models go to the lab's shared cache (`HF_HUB_CACHE`, set in `/etc/environment`; do not set
`HF_HOME`). Use `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and ungated mirrors (`unsloth/Llama-3.2-3B`,
`unsloth/Llama-3.2-3B-Instruct`). Never use another user's token. GPU placement (one group per job, NUMA pinning) is
handled by `scripts/gpu_env.sh`; see CLAUDE.md, "Start here", rule 4.

## Layout

| Path | Contents |
| --- | --- |
| `results/` | all write-ups; `m8_audit_design.md` is the live record of the audit, `REPORT.md` of the LIMO phase |
| `scripts/` | numbered steps: 00–27 the LIMO pipeline (training, eval, defenses, attacks), 28–45 the audit, 46–51 the answer floor and the puzzle testbed (index below) |
| `configs/train/` | LLaMA-Factory yamls, one per (student × condition × seed) |
| `configs/setup2/` | Distillation Game configs for the GSM8K/MATH setup: `{gsm8k,math}_{U,solo_l3i,solo_q7i}_s{123,124,125}.yaml` |
| `src/antidistill/` | defenses (hindsight, PART), attacks (style, re-voice, reconstruction), scoring, data builders |
| `data/` | `dataset_info.json` (LLaMA-Factory registry); `defended/` and `curated/` datasets (mostly not in git, see below) |
| `third_party/LLaMA-Factory/`, `third_party/kim_eval/` | vendored trainer and math eval harness (run the harness from inside its directory) |
| `third_party/ext/` | upstream clones recreated by `scripts/setup_third_party.sh` (see its README) |
| `reference/` | Kim et al.'s own generations, used to validate our grader |
| `tests/` | unit tests (`tests/test_part.py`) |
| `outputs/`, `saves/`, `logs/` | run outputs, checkpoints, logs (not in git; `outputs` and `saves` link to `/data/umid/antidistill/`) |

**Audit scripts (m8 section in brackets).** `28` small students on LIMO (§9.2) · `29`, `31`, `33` setup-2 runs,
attacker ladder and job pool (§9.1, §9.3, §9.4) · `30`, `36_setup1_7b_seeds` LIMO seeds (§9.5) · `32`, `dg_rl_base.py`
RL from the student (§9.3) · `34` DeepMath pilot (§10) · `35`, `37_star_s123_train`, `dg_star.py` self-training,
level A (§9.3) · `38`–`41`, `dg_train_from_traces.py`, `dg_score_vllm.py` the MATH full-budget pilot (§10.1) · `42`
combination students for U_ℓ (§11) · `43` the pilot's combination students (§10.1, §11) · `dg_eval_base.py` the
untrained student.

**Floor and testbed scripts (m8 section in brackets).** `46` STaR then GRPO, `47` GRPO alone, `48` GRPO with
stabilizers, all with `dg_rl_base.py` (§13–15) · `49_answer_dist.sh`, `dg_answer_dist.py`, `answer_bits.py` how
predictable the teacher's answers are (§16) · `kk_make.py` puzzles (Xie et al.'s generator), `50_kk_pilot.sh`,
`kk_pilot.py` the pilot and its answer checker (§17) · `51_kk_step2.sh`, `kk_build.py` (training sets, F_D's hint
round), `kk_sft.py` (student training), `kk_eval.py` (student scoring) step 2 (§18).

## Data not in git

Datasets of tens of MB that are regenerated by a script stay out of git. **The copies made before 2026-10-29 were lost
when dspl1 was rebuilt**; the numbers recorded in `results/` stand, but each file below must be regenerated before a
new experiment can use it (sampled data will differ from the lost copies):

| File(s) | Made by |
| --- | --- |
| `data/raw/limo_v2.json` | `scripts/01_fetch_data.py` |
| `data/defended/limo_part*.json` (PART) | `scripts/45_generate_part.sh` |
| `data/curated/recon7b_*.json`, `limo_recon_*.json` (Solo, Search, Style) | `src/antidistill/attacks/epistemic_reconstruction.py` (model served by `scripts/40_serve_teacher.sh`) |
| `data/curated/style_part*.json`, `revoice_part*.json` | `src/antidistill/attacks/style_part.py`, `revoice.py` |
| `data/curated/limo_mixed_*.json` | `scripts/50_make_mixed_dataset.py` |
| `outputs/setup2/` (teacher and Solo solutions of setup 2, per seed) | `scripts/29_setup2_run.sh`, `31_setup2_ladder.sh`, `33_setup2_pool.sh` |
| `outputs/setup3_math/` (long-budget teacher, pilot students and scores) | `scripts/39`–`41`, `40_math_pilot.sh` |
| `outputs/setup2_answers/` (16 tries per problem and owned model) | `scripts/49_answer_dist.sh` |
| `data/kk/*.jsonl` (puzzles; deterministic given the seeds in m8 §17–18) | `scripts/kk_make.py` |
| `outputs/kk_pilot/`, `outputs/kk_step2/` (solutions, training sets, students, scores) | `scripts/50_kk_pilot.sh`, `51_kk_step2.sh` |

The hindsight datasets (`data/defended/limo_hindsight_*.json`, about 5 MB) are in git; use `limo_hindsight_chat.json`
(v2), not the v1 file (CLAUDE.md §1).

## Attribution

Builds on Kim et al. (arXiv:2603.15500), LIMO (Ye et al., COLM 2025), LLaMA-Factory (Zheng et al., ACL 2024), The
Distillation Game (Allouah et al., arXiv:2605.22737), Antidistillation Sampling (Savani et al.) and the
Knights-and-Knaves generator of Xie et al. (arXiv:2410.23123, MIT). See
`third_party/*/LICENSE`. Files prefixed `_kim_` are verbatim copies of Kim et al.'s code, kept as references.
