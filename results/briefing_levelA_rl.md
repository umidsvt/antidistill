# Briefing: a stronger level-A attacker (answers-only RL)

For the co-author taking over the level-A work, 2026-11-15. It says where the project stands, what has been tried at
level A, how the code is organised, and the rules every new run follows. Numbers are from `results/m8_audit_design.md`
(m8), sections in brackets; read §11 and §13–15 there for the full details.

## 1. Why level A matters for the paper

The paper (draft: private repo `antidistillation-overleaf`, folder `icml2027_testbed/`) rests on one observation.
Every anti-distillation defense keeps the teacher's final answers correct. An attacker can ignore the defended
solutions and train only with the help of the answers, so the most any such defense can protect is

  **P = U − F**: the student trained on the teacher's solutions (U), minus the best answers-only student (F).

F is the *answer floor*. **Your task is to raise F at level A**, where the attacker owns nothing but the student
model and receives the teacher's final answers.

- Today F_A = **35.2%** (one round of few-shot self-training, STaR), against U = 59.8%. So at most 24.6 points are
  protectable on GSM8K.
- **Any improvement is a result.** A stronger answers-only attack raises the floor and shrinks what defenses can
  protect, so the paper's bounds stay valid and get tighter.
- **At about 45%, the verdict changes.** The pre-registered margin is a quarter of what training adds,
  (59.8 − 0.6) / 4 = 14.8, so above 59.8 − 14.8 = 45.0% the teacher's solutions are no longer clearly needed even at
  level A. On GSM8K the paper would then read "nothing protectable at any level".
- Javaheri et al. (2026) reach 40.7% with RL alone from the same base Llama-3.2-3B on GSM8K. But they use ground-truth
  rewards, full-parameter training, 1,024 prompts × 8 samples per step and greedy scoring on the original GSM8K test,
  so their number is not directly comparable. It does suggest 40%+ is reachable.

## 2. The setting (fixed; do not change without the authors)

| Item | Value |
| --- | --- |
| Code | the Distillation Game code (Allouah et al. 2026), `third_party/ext/distillation-game`, with our patch |
| Student | base Llama-3.2-3B (`unsloth/Llama-3.2-3B`), tokenizer `unsloth/Llama-3.2-3B-Instruct` (as in the setup) |
| Training pool | the 5,238 GSM8K training problems of each seed's split (the split is reshuffled per seed: 123, 124, 125) |
| Teacher answers | `extracted_answer` in `outputs/setup2/gsm8k_U_s<seed>_<stamp>/teacher/train_standard.json` (teacher R1-Distill-Qwen-7B at 1,024 tokens; wrong on about 12% of the pool) |
| Test | GSM8K-Platinum, 1,209 problems, the setup's evaluation (temperature 0.6, top-p 0.95, 1,024 tokens, answer forcing) |
| Report | mean of seeds 123–125 |

**What a level-A attacker may use:**
- the base student's weights;
- the training prompts;
- the teacher's final answers on them, including its wrong ones;
- a few human-written worked examples (STaR uses Wei et al.'s 8 GSM8K exemplars);
- any training method and any compute.

**What it may not use:**
- ground-truth labels;
- any test problem, for training or for choosing checkpoints;
- any other model (instruct versions, larger models);
- the teacher's solutions.

Checkpoints are chosen on held-out *training* problems scored against the teacher's answers (§15 uses the last 256
problems of the pool).

## 3. What has been tried

| Attempt | m8 | Result | What went wrong / what we learned |
| --- | --- | --- | --- |
| RL v1: GRPO, reward = matches the teacher's answer | §9.3 | reward stayed at 1–2%, nothing learned | the base model almost never ends with a boxed answer; 92% of groups had identical rewards |
| RL v2: + a format bonus for writing `\boxed{` | §9.3 | correctness 0, test 0/300 | reward hacking: the bonus paid for `\boxed{` anywhere, so samples wrote it and ran to the length limit |
| Self-training with the setup's chat prompt | §9.3 | 4.9% (one seed) | the base model does not follow the chat prompt |
| **Few-shot STaR** (8 exemplars, 8 tries at T=0.8, hints for unsolved problems, then SFT with U's recipe) | §9.3, §13 | **34.3% (first run), 35.2% (rerun: 35.8 / 34.4 / 35.5)** | the current floor |
| STaR, then GRPO (a labmate's settings) | §13 | 30.2%, lower than STaR on every seed | RL after SFT hurt with these settings |
| GRPO alone, plain prompt, format modes, LR 2e-5 cosine, no KL, 512 tokens | §14 | pilot passed; 600-step runs collapsed to 2.1% | learning rate stayed high (cosine over 600 steps); after steps 55–105, answers ran to the length limit and all groups had identical rewards |
| **GRPO alone with Javaheri et al.'s stabilizers** (reward +1 / −0.5 / −1 for no boxed answer; KL 1e-4; LR 5e-6 constant on LoRA; 32 prompts × 8; 2,048 tokens; checkpoint selection on 256 held-out problems) | §15 | 29.6% after 300 steps (31.7% greedy); **best of 600 steps 32.1%** (35.3 / 28.3 / 32.6) | no early collapse. Reward near 0.03 for about 150 steps, then rising. Seeds 124 and 125 collapsed late (held-out 4% around steps 450–500) |

Things not yet tried, each a plausible way up (each needs its m8 rule first, §5):
1. **The stabilized GRPO recipe (§15) started from the STaR student.** §13 used the labmate's unstabilized settings.
2. **More rounds of self-training:**
   - STaR from the improved student, 2–3 rounds;
   - several different correct attempts per problem instead of the first one (rejection-sampling fine-tuning,
     ReST-EM style).
3. **Javaheri et al.'s recipe as they ran it:**
   - full-parameter training, LR 5e-7, 1,024 prompts × 8 samples per step;
   - their code is cloned at `third_party/ext/treason` (commit 127de4c); "option A" in m8 §15 was deferred.
4. **Against the late collapse:**
   - learning-rate decay after the reward starts rising;
   - skipping groups whose samples all have the same reward;
   - a stronger KL penalty or a length penalty.
5. **Better exemplars or prompt,** so that the base model's early attempts are well-formatted more often (exploration
   is the bottleneck at the start).

Remember the teacher's answers are wrong on about 12% of the pool. The reward and the filters inherit that noise.

## 4. The code

Environment: `.venv-dg` (vLLM 0.8.5, TRL 0.19.1, PEFT 0.15.2, transformers 4.53.2). The Distillation Game code runs
with `PYTHONPATH=third_party/ext/distillation-game/src`.

| File | What it does |
| --- | --- |
| `scripts/dg_star.py` | STaR. `--phase gen` (vLLM; `--fewshot` for the 8-exemplar prompt) samples attempts and keeps the first that reaches the teacher's answer, with hint rounds; `--phase train` fine-tunes with U's recipe (LoRA r128, 3 epochs, LR 5e-4, batch 24), saves the adapter and scores it with the setup's evaluation |
| `scripts/dg_rl_base.py` | GRPO (TRL with vLLM colocate). Options: `--init-adapter` (start from a LoRA), `--prompt-style plain\|chat`, `--reward-mode bonus\|mix`, `--beta` (KL), `--lr`, `--lr-schedule cosine\|constant`, `--num-generations`, `--prompts-per-step`, `--max-steps`, `--max-completion`, `--micro-batch`, `--vllm-mem`, `--save-every`, `--holdout-select N` (checkpoint selection on the last N training problems), `--greedy-test`, `--eval-only`, `--test-limit`. `ChunkedGRPOTrainer` fixes an out-of-memory error in TRL 0.19.1's reference log-probabilities when `beta > 0` |
| `scripts/46_levelA_star_grpo.sh` | STaR then GRPO, 3 seeds (`GPUS`, `SEEDS`, `SMOKE=1`) |
| `scripts/47_levelA_grpo_only.sh` | §14's GRPO-only runs (`PHASE=check\|pilot\|full`) |
| `scripts/48_levelA_grpo_stable.sh` | §15's stabilized GRPO (`GPUS`, `SEEDS`, `SMOKE=1`, `CONT=1` to continue from step 300) |
| `configs/setup2/gsm8k_U_s<seed>.yaml` | the setup's config per seed (split, sampling, training recipe) |

Outputs:
- `outputs/setup2/star_fewshot_gsm8k_s<seed>/`: `traces.json`, `gen_summary.json`, `adapter/`, `test_star.json`
  (per problem), `result.json`.
- `outputs/setup2/grpo_stable/s<seed>[_cont]/`: `ckpt_*`, `select_ckpt_*.json` (held-out scores), `adapter_kept`,
  `test_rl*.json` (per problem), `result.json`.

Example commands, run from the repo root:

```bash
# STaR, one seed (generation, then training and scoring)
PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python scripts/dg_star.py --phase gen --fewshot \
    --config configs/setup2/gsm8k_U_s123.yaml \
    --teacher-train outputs/setup2/gsm8k_U_s123_20261031_150146/teacher/train_standard.json
# ... then the same with --phase train
# Stabilized GRPO: smoke test, then three seeds
SMOKE=1 GPUS=4 SEEDS=123 scripts/48_levelA_grpo_stable.sh
GPUS="4 5 6" SEEDS="123 124 125" setsid nohup scripts/48_levelA_grpo_stable.sh > logs/<name>.log 2>&1 < /dev/null & disown
```

## 5. Rules for every run (CLAUDE.md, "Start here", has the full text)

1. **The rule before the result.** Before a run, add a section to m8 (the next free number):
   - what is run and why;
   - the exact settings;
   - how the checkpoint is chosen;
   - what each outcome means: for example, "F_A = the best of STaR and this run's kept checkpoint, per seed; report
     the 3-seed mean and U − F_A with its 95% interval".

   Then get the authors' go-ahead for GPU runs. A recipe that fails because of a bug or a collapse is recorded as a
   failed recipe, not as a result.
2. **Numbers come from the per-problem files** (`test_*.json`), never from logs. Record the result in m8 with the
   rule's verdict.
3. **Never drop problems from the training pool;** report the ones a method cannot use.
4. **Shared machine (dspl1, 8 × L40S 46 GB):**
   - One job's GPUs come from one group, 0–3 or 4–7; the launchers source `scripts/gpu_env.sh`, which pins the NUMA
     node.
   - Never use a GPU someone else is using; `gpus` shows who holds what.
   - Every queue waits for free memory.
   - Models come from the lab cache (`HF_HUB_CACHE`); never set `HF_HOME` or use another user's token.
5. **Long jobs run detached** (`setsid nohup … & disown`); stop jobs by PID, never with `pkill -f` patterns.

## 6. Known pitfalls

- **vLLM can hang at exit** after writing its output. Our scripts exit hard (`hard_exit`), and the newer launchers
  have a watchdog. An orphaned engine holding a GPU is stopped with the command in CLAUDE.md, rule 6.
- **vLLM in a forked process** fails ("Cannot re-initialize CUDA"). Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
- **Stop tokens.** The base model's end token is `<|end_of_text|>`, but students trained in the Llama-3.2 chat template
  end their turn with `<|eot_id|>`. Generation must stop on the latter, or it runs past the answer. This bug affected
  the puzzle scoring (m8 §18); check it in any new evaluation code.
- **Format rewards get hacked** if they pay for a pattern anywhere in the text (§9.3). Reward only a well-formed final
  answer.
- **Long constant learning rates collapse GRPO** in this setup (§14). Watch the held-out score, not only the reward.

## 7. Later: the puzzle testbed

The same question arises on the Knights-and-Knaves testbed (m8 §17–18), where the floor at level D (the attacker owns
an open 7B reasoning model) is 19–22% against U 41–49%, provisionally. Level A there is not run yet. A recipe that
works on GSM8K is worth trying on the puzzles next (`scripts/kk_*.py`, `scripts/51_kk_step2.sh`).
