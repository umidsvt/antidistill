# third_party/ext

Upstream repositories used as they are, recreated by `bash scripts/setup_third_party.sh` (they are not vendored
because each is a git repository of its own):

| Directory | Upstream | Commit | Used for |
| --- | --- | --- | --- |
| `distillation-game/` | https://github.com/ysfalh/distillation-game (Allouah et al., *The Distillation Game*) | `56ecdf7` | the GSM8K/MATH setup ("setup 2") and every `scripts/dg_*.py` |
| `antidistillation-sampling/` | https://github.com/locuslab/antidistillation-sampling (Savani et al., ADS) | `8b79c72` | reference only |

`distillation-game.patch` holds our three fixes to the Distillation Game code, each marked `ANTIDISTILL PATCH` at the
code site (details in `results/m8_audit_design.md` §9 and §9.4):

1. `src/clean_sweep/data/__init__.py`: removed imports of dataset modules missing from the upstream repo.
2. `scripts/run_pipeline.py`: the proxy-gradient stage runs only when ADS is requested.
3. `src/clean_sweep/generation/core.py`: answer forcing takes the chat marker from the first problem that has one
   (upstream crashed when the first prompt was cut at 512 tokens).

Run its code with `PYTHONPATH=third_party/ext/distillation-game/src .venv-dg/bin/python ...` (env: `scripts/setup_env_dg.sh`).
