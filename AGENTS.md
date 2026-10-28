# AGENTS.md

The agent guide for this repository is [CLAUDE.md](CLAUDE.md); its instructions apply to every coding agent. Read its
"Start here" section first, then `README.md` and Part I of `IMPLEMENTATION_PLAN.md`.

The rules in brief (full text in CLAUDE.md):
- Write each experiment's decision rule in `results/m8_audit_design.md` before running it; record results there.
- Ask the authors before long GPU runs or changing a fixed rule.
- Never drop problems from a training pool.
- On dspl1, GPUs 0 and 1 are reserved for others; every job waits for free GPU memory.
- Never use another user's Hugging Face token; set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` and use ungated mirrors.
- Stop orphaned vLLM engines of your own jobs (command in CLAUDE.md).
